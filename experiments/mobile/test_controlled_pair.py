from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from claude_m3a import InvalidModelCall
from controlled_pair import (_preflight, _run_pair, file_sha, implementation_identity,
                             run_native_pair_development)
from native_attempt import ANDROID_COMMIT, InvalidAttempt, _execute_attempt, sha


class Task:
    def __init__(self, score=0):
        self.params = {"seed": 20261123}
        self.app_names = ("clock",)
        self.goal = "development fixture"
        self.initialized = False
        self.is_successful = Mock(return_value=score)


class ControlledPairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = {name: Task() for name in ("base", "refiner")}
        self.spec = dict(scope="development", method="native_m3a_controlled_pair",
            run_id="offline-pair", android_source_commit=ANDROID_COMMIT,
            order=["base", "refiner"], attempts_per_arm=1,
            implementation_sha256=implementation_identity())
        self.native = dict(scope="development", method="native_m3a_base",
            run_id="offline-arm", android_source_commit=ANDROID_COMMIT,
            task_family="Task", task_params={"seed": 20261123}, seed=20261123,
            app_names=["clock"], goal_sha256=sha(b"development fixture"),
            max_steps=10, start_on_home_screen=True)
        provider = self.root/"provider.json"
        provider.write_text(json.dumps({"external_provider_transport_ready": True}))
        self.provider = provider
        self.native["provider_readiness_sha256"] = file_sha(provider)
        self.archive = self.root/"snapshot.tar"
        self.archive.write_bytes(b"frozen fixture")
        self.native["snapshot_archive_sha256"] = file_sha(self.archive)
        for name, value in [("native", self.native), ("bridge", {"scope": "development"})]:
            p = self.root/(name + ".json")
            p.write_text(json.dumps(value))
            self.spec["native_protocol" if name == "native" else "bridge_protocol"] = {
                "path": str(p), "sha256": file_sha(p)}
        self.calls = []
        self.events = []
        self.errors = {}
        self.outputs = {"base": 0, "refiner": 1}
        self.pair_dir = self.root/"pair"

    @contextmanager
    def lifecycle(self, path):
        path.mkdir()
        (path/"snapshot.json").write_text("same frozen snapshot")
        self.events.append("initialize")
        try:
            yield
        finally:
            self.events.append("teardown")

    def run_arm(self, arm, path):
        self.calls.append(arm)
        task = self.tasks[arm]
        task.is_successful.return_value = self.outputs[arm]

        def episode(**settings):
            if arm in self.errors:
                raise self.errors[arm]
            return SimpleNamespace(done=False)

        return _execute_attempt(spec=self.native,
            protocol_sha256=self.spec["native_protocol"]["sha256"], task=task,
            env="fixture", agent=arm, run_episode=episode, lifecycle=self.lifecycle,
            audit_dir=path, attempt_method=("native_m3a_base" if arm == "base"
                                          else "native_m3a_refiner_development"))

    def run_pair(self):
        return _run_pair(spec=self.spec, pair_sha="fixed-pair-sha",
                         audit_dir=self.pair_dir, run_arm=self.run_arm)

    def preflight(self, spec=None, agents=None, tasks=None):
        return _preflight(spec or self.spec, tasks=tasks or self.tasks,
            agents=agents or {"base": Mock(), "refiner": Mock()}, env="fixture",
            provider_readiness_path=self.provider, snapshot_archive=self.archive,
            refiner=Mock())

    def test_complete_pair_preserves_zero_and_uses_effective_denominator(self):
        out = self.run_pair()
        self.assertEqual(out["scores"], {"base": 0, "refiner": 1})
        self.assertEqual(out["effective_pair_denominator"], 1)
        self.assertEqual(out["refiner_minus_base"], 1)
        self.assertEqual(out["benchmark_scores"], 0)
        self.assertEqual(out["transfer_scores"], 0)
        self.assertEqual(self.events, ["initialize", "teardown"] * 2)

    def test_resume_complete_pair_never_calls_arms_again(self):
        self.run_pair()
        before = {str(p): p.read_bytes() for arm in ("base", "refiner")
                  for p in (self.pair_dir/arm).rglob("*") if p.is_file()}
        self.run_pair()
        self.assertEqual(self.calls, ["base", "refiner"])
        self.assertTrue(all(Path(p).read_bytes() == value for p, value in before.items()))

    def test_refiner_provider_failure_keeps_base_and_is_not_paired_zero(self):
        error = InvalidModelCall("provider_refusal", "offline", 1)
        self.errors["refiner"] = error
        with self.assertRaises(InvalidModelCall) as caught:
            self.run_pair()
        self.assertIs(caught.exception, error)
        self.tasks["refiner"].is_successful.assert_not_called()
        out = json.loads((self.pair_dir/"report000000.json").read_text())
        self.assertEqual(out["scores"], {"base": 0, "refiner": None})
        self.assertEqual(out["effective_pair_denominator"], 0)
        self.assertIsNone(out["refiner_minus_base"])
        self.errors.clear()
        self.run_pair()
        self.assertEqual(self.calls, ["base", "refiner"])

    def test_resume_runs_only_unstarted_arm_after_invalid_base(self):
        self.errors["base"] = OSError("fixture device failure")
        with self.assertRaises(OSError):
            self.run_pair()
        self.assertFalse((self.pair_dir/"refiner").exists())
        self.errors.clear()
        out = self.run_pair()
        self.assertEqual(self.calls, ["base", "refiner"])
        self.assertEqual(out["arm_state"], {"base": "invalid", "refiner": "valid"})
        self.assertEqual(out["effective_pair_denominator"], 0)

    def test_preselected_reverse_order_is_respected(self):
        self.spec["order"] = ["refiner", "base"]
        self.run_pair()
        self.assertEqual(self.calls, ["refiner", "base"])

    def test_missing_arm_receipt_blocks_without_rerun(self):
        self.run_pair()
        (self.pair_dir/"base.receipt.json").unlink()
        with self.assertRaisesRegex(InvalidAttempt, "Uncommitted"):
            self.run_pair()
        self.assertEqual(len(self.calls), 2)

    def test_changed_score_blocks_before_any_missing_arm(self):
        self.errors["refiner"] = OSError("offline")
        with self.assertRaises(OSError):
            self.run_pair()
        p = self.pair_dir/"base/result.json"
        value = json.loads(p.read_text()); value["score"] = 1
        p.write_text(json.dumps(value))
        with self.assertRaisesRegex(InvalidAttempt, "result changed"):
            self.run_pair()
        self.assertEqual(len(self.calls), 2)

    def test_changed_nested_snapshot_evidence_blocks_resume(self):
        self.run_pair()
        (self.pair_dir/"base/lifecycle/snapshot.json").write_text("changed")
        with self.assertRaisesRegex(InvalidAttempt, "evidence changed"):
            self.run_pair()

    def test_changed_pair_binding_or_order_cannot_reuse_evidence(self):
        self.run_pair()
        self.spec["order"].reverse()
        with self.assertRaisesRegex(InvalidAttempt, "protocol differs"):
            self.run_pair()

    def test_another_controller_cannot_start_same_pair(self):
        self.pair_dir.mkdir()
        with (self.pair_dir/"pair.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(InvalidAttempt, "already running"):
                self.run_pair()
        self.assertEqual(self.calls, [])

    def test_uncommitted_failure_retains_original_error_and_blocks_resume(self):
        error = OSError("failed result write fixture")
        def fail(arm, path):
            path.mkdir()
            raise error
        with self.assertRaises(OSError) as caught:
            _run_pair(spec=self.spec, pair_sha="fixed-pair-sha", audit_dir=self.pair_dir, run_arm=fail)
        self.assertIs(caught.exception, error)
        out = json.loads((self.pair_dir/"report000000.json").read_text())
        self.assertEqual(out["arm_state"]["base"], "invalid_uncommitted")
        with self.assertRaisesRegex(InvalidAttempt, "Uncommitted"):
            self.run_pair()

    def test_pending_provider_stops_before_imports_evidence_and_actions(self):
        self.provider.write_text(json.dumps({"external_provider_transport_ready": False}))
        self.native["provider_readiness_sha256"] = file_sha(self.provider)
        p = Path(self.spec["native_protocol"]["path"])
        p.write_text(json.dumps(self.native));self.spec["native_protocol"]["sha256"] = file_sha(p)
        protocol = self.root/"pair.json";protocol.write_text(json.dumps(self.spec))
        with patch("controlled_pair._bind_native_contract") as binder:
            with self.assertRaisesRegex(InvalidAttempt, "provider readiness"):
                run_native_pair_development(protocol_path=protocol, protocol_sha256=file_sha(protocol),
                    tasks=self.tasks, agents={"base": Mock(), "refiner": Mock()}, env=Mock(),
                    provider_readiness_path=self.provider, snapshot_device=Mock(),
                    snapshot_archive=self.archive, refiner=Mock(), audit_dir=self.pair_dir)
            binder.assert_not_called()
        self.assertFalse(self.pair_dir.exists())

    def test_task_param_change_or_state_reuse_fails_preflight(self):
        self.tasks["refiner"].params["seed"] = 7
        with self.assertRaisesRegex(InvalidAttempt, "seed"):
            self.preflight()
        shared = Task()
        with self.assertRaisesRegex(InvalidAttempt, "carry between"):
            self.preflight(tasks={"base": shared, "refiner": shared})

    def test_formal_scope_budget_or_code_change_fails_preflight(self):
        for change in [dict(scope="formal"), dict(attempts_per_arm=2),
                       dict(implementation_sha256={}), dict(order=["base", "base"])]:
            with self.subTest(change=change), self.assertRaises(InvalidAttempt):
                self.preflight(dict(self.spec, **change))

    def test_changed_snapshot_or_bridge_protocol_fails_preflight(self):
        self.archive.write_bytes(b"changed")
        with self.assertRaisesRegex(InvalidAttempt, "Snapshot"):
            self.preflight()
        Path(self.spec["bridge_protocol"]["path"]).write_text("{}")
        with self.assertRaisesRegex(InvalidAttempt, "Frozen input SHA"):
            self.preflight()

    def test_bad_refiner_binding_rejected_before_any_arm_initialization(self):
        with patch("controlled_pair._bind_native_contract", return_value=Mock()), \
             patch("controlled_pair.bind_native_refiner_contract", side_effect=InvalidAttempt("policy differs")):
            with self.assertRaisesRegex(InvalidAttempt, "policy differs"):
                self.preflight()
        self.assertFalse(self.pair_dir.exists())


if __name__ == "__main__":
    unittest.main()
