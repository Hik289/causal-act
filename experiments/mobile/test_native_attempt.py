from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from claude_m3a import InvalidModelCall
from native_attempt import (ANDROID_COMMIT, InvalidAttempt, _execute_attempt,
                            read_bound_json, run_native_base_development, sha,
                            validate_protocol)


class FixtureTask:
    def __init__(self, score=0):
        self.params = {"seed": 20261121, "value": "fixed"}
        self.app_names = ("clock",)
        self.goal = "fixed development goal"
        self.initialized = False
        self.is_successful = Mock(return_value=score)


class AttemptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = FixtureTask()
        self.spec = dict(scope="development", method="native_m3a_base",
            android_source_commit=ANDROID_COMMIT, run_id="offline-fixture",
            seed=20261121, task_family="FixtureTask", task_params=dict(self.task.params),
            app_names=["clock"], goal_sha256=sha(self.task.goal.encode()),
            max_steps=10, start_on_home_screen=True)
        self.events = []
        self.runner = Mock(return_value=SimpleNamespace(done=True, step_data={"step_number": [0]}))

    @contextmanager
    def lifecycle(self, path):
        self.events.append("initialize")
        try:
            yield
        finally:
            self.events.append("teardown")

    def run_attempt(self, lifecycle=None):
        return _execute_attempt(spec=self.spec, protocol_sha256="frozen",
            task=self.task, env="owned-env", agent="owned-agent", run_episode=self.runner,
            lifecycle=lifecycle or self.lifecycle, audit_dir=self.root/"attempt")

    def receipt(self):
        return json.loads((self.root/"attempt/result.json").read_text())

    def test_valid_zero_is_retained_with_native_evidence(self):
        result = self.run_attempt()
        self.assertTrue(result["valid"])
        self.assertEqual(result["score"], 0)
        self.assertEqual(self.events, ["initialize", "teardown"])
        raw = (self.root/"attempt/episode.pickle").read_bytes()
        self.assertEqual(result["episode_sha256"], sha(raw))
        self.task.is_successful.assert_called_once_with("owned-env")

    def test_development_success_is_not_a_formal_result(self):
        self.task.is_successful.return_value = 1
        result = self.run_attempt()
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["benchmark_scores"], 0)
        self.assertEqual(result["transfer_scores"], 0)

    def test_step_budget_exhaustion_still_uses_native_scorer(self):
        self.runner.return_value.done = False
        result = self.run_attempt()
        self.assertTrue(result["valid"])
        self.assertFalse(result["agent_done"])
        self.assertEqual(result["score"], 0)
        self.assertEqual(self.runner.call_args.kwargs["max_n_steps"], 10)

    def test_provider_failure_propagates_and_never_calls_scorer(self):
        error = InvalidModelCall("provider_refusal", "offline", 1)
        self.runner.side_effect = error
        with self.assertRaises(InvalidModelCall) as caught:
            self.run_attempt()
        self.assertIs(caught.exception, error)
        self.task.is_successful.assert_not_called()
        self.assertIsNone(self.receipt()["score"])
        self.assertEqual(self.events[-1], "teardown")

    def test_execution_failure_is_invalid_not_zero(self):
        self.runner.side_effect = OSError("device disconnected")
        with self.assertRaises(OSError):
            self.run_attempt()
        self.assertFalse(self.receipt()["valid"])
        self.assertIsNone(self.receipt()["score"])
        self.task.is_successful.assert_not_called()

    def test_native_scorer_error_is_invalid(self):
        self.task.is_successful.side_effect = OSError("scorer device error")
        with self.assertRaises(OSError):
            self.run_attempt()
        self.assertIsNone(self.receipt()["score"])
        self.assertEqual(self.receipt()["scorer_calls"], 1)

    def test_teardown_failure_preserves_candidate_without_committing_score(self):
        @contextmanager
        def failed_cleanup(path):
            yield
            raise OSError("teardown failed")
        self.task.is_successful.return_value = 1
        with self.assertRaises(OSError):
            self.run_attempt(failed_cleanup)
        result = self.receipt()
        self.assertEqual(result["scorer_value_before_cleanup"], 1)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["score"])

    def test_no_duplicate_attempt_can_overwrite_evidence_or_call_agent(self):
        self.run_attempt()
        before = (self.root/"attempt/result.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_attempt()
        self.assertEqual(self.runner.call_count, 1)
        self.assertEqual((self.root/"attempt/result.json").read_bytes(), before)

    def test_nan_scorer_output_is_invalid(self):
        self.task.is_successful.return_value = float("nan")
        with self.assertRaises(InvalidAttempt):
            self.run_attempt()
        self.assertIsNone(self.receipt()["score"])

    def test_private_evidence_has_restricted_permissions(self):
        self.run_attempt()
        for p in (self.root/"attempt").iterdir():
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)

    def test_frozen_protocol_hash_cannot_change_silently(self):
        p = self.root/"protocol.json"
        p.write_text(json.dumps(self.spec))
        digest = sha(p.read_bytes())
        self.assertEqual(read_bound_json(p, digest), self.spec)
        p.write_text(json.dumps(dict(self.spec, max_steps=9)))
        with self.assertRaises(InvalidAttempt):
            read_bound_json(p, digest)

    def test_task_seed_or_params_change_is_rejected(self):
        validate_protocol(self.spec, task=self.task)
        self.task.params["seed"] = 7
        with self.assertRaises(InvalidAttempt):
            validate_protocol(self.spec, task=self.task)

    def test_formal_scope_or_refiner_label_is_not_accepted_as_implemented(self):
        for change in [dict(scope="formal"), dict(method="refiner")]:
            with self.assertRaises(InvalidAttempt):
                validate_protocol(dict(self.spec, **change), task=self.task)

    def test_provider_not_ready_stops_before_native_imports_or_actions(self):
        provider = self.root/"provider.json"
        provider.write_text(json.dumps({"external_provider_transport_ready": False}))
        self.spec["provider_readiness_sha256"] = sha(provider.read_bytes())
        p = self.root/"protocol.json"
        p.write_text(json.dumps(self.spec))
        env = Mock()
        with self.assertRaisesRegex(InvalidAttempt, "provider readiness"):
            run_native_base_development(protocol_path=p, protocol_sha256=sha(p.read_bytes()),
                provider_readiness_path=provider, task=self.task, env=env, agent=Mock(),
                snapshot_device=Mock(), snapshot_archive=self.root/"not-opened",
                audit_dir=self.root/"not-created")
        self.assertFalse(env.mock_calls)
        self.assertFalse((self.root/"not-created").exists())


if __name__ == "__main__":
    unittest.main()
