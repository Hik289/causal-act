from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from claude_m3a import InvalidModelCall
from refiner_bridge import (InvalidRefinerCall, RefinerOutput, _owned_bridge)


BASE = 'Reason: proposed\nAction: {"action_type":"click","index":0}'
REFINED = 'Reason: corrected\nAction: {"action_type":"click","index":1}'


class EpisodeAgent:
    def __init__(self, llm):
        self.llm = llm
        self.images = [bytearray(b"raw"), bytearray(b"som")]
        self.executed = []
        self.history = []

    def step(self, goal):
        text, safe, raw = self.llm.predict_mm("original action prompt", self.images)
        action = json.loads(text.split("Action:", 1)[1])
        self.executed.append(action)
        summary = self.llm.predict_mm("summarize " + json.dumps(action), self.images)
        self.history.append((text, raw, summary))
        return SimpleNamespace(done=False)


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base_tuple = (BASE, None, {"provider": "base"})
        self.summary_tuple = ("native summary", None, {"phase": "summary"})
        self.llm = Mock()
        self.llm.predict_mm.side_effect = [self.base_tuple, self.summary_tuple]
        self.agent = EpisodeAgent(self.llm)
        self.policy = Mock(return_value=RefinerOutput(REFINED, {"kind": "scripted"}))

    @staticmethod
    def parse(text):
        if not text.startswith("Reason:"):
            return None, None
        return "proposed", json.loads(text.split("Action:", 1)[1])

    def bridge(self):
        return _owned_bridge(self.agent, refiner=self.policy, parse_action=self.parse,
            audit_dir=self.root/"bridge", binding={"scope": "scripted-development"})

    def record(self):
        return json.loads((self.root/"bridge/step000000.json").read_text())

    def test_changed_action_reaches_execution_history_and_summary(self):
        with self.bridge():
            self.agent.step("goal")
        self.assertEqual(self.agent.executed, [{"action_type": "click", "index": 1}])
        self.assertEqual(self.agent.history[0][0], REFINED)
        self.assertIn('"index": 1', self.llm.predict_mm.call_args_list[1].args[0])
        self.assertIs(self.agent.history[0][2], self.summary_tuple)
        self.policy.assert_called_once()
        self.assertEqual(self.record()["summary_calls_forwarded"], 1)

    def test_original_model_inputs_are_unmodified_and_passed_once(self):
        images = deepcopy(self.agent.images)
        def policy(proposal):
            proposal.images[0][:] = b"mutated"
            proposal.base_action["index"] = 9
            return RefinerOutput(None, {})
        self.policy.side_effect = policy
        with self.bridge():
            self.agent.step("goal")
        self.assertEqual(self.agent.images, images)
        call = self.llm.predict_mm.call_args_list[0]
        self.assertEqual(call.args[0], "original action prompt")
        self.assertIs(call.args[1], self.agent.images)
        self.assertEqual(self.llm.predict_mm.call_count, 2)
        self.assertEqual(self.agent.executed[0]["index"], 0)

    def test_noop_keeps_exact_provider_response(self):
        self.policy.return_value = RefinerOutput(None, {})
        with self.bridge():
            self.agent.step("goal")
        self.assertIs(self.agent.history[0][1], self.base_tuple[2])
        self.assertFalse(self.record()["overridden"])

    def test_override_is_not_attributed_to_base_provider(self):
        with self.bridge():
            self.agent.step("goal")
        provenance = self.agent.history[0][1]
        self.assertIs(provenance["base_response"], self.base_tuple[2])
        self.assertEqual(provenance["refiner_evidence"], {"kind": "scripted"})

    def test_malformed_refined_action_reaches_native_parser_without_retry(self):
        self.policy.return_value = RefinerOutput("malformed nonempty model output", {})
        with self.bridge(), self.assertRaises(IndexError):
            self.agent.step("goal")
        self.assertEqual(self.llm.predict_mm.call_count, 1)
        self.assertEqual(self.record()["refined_output"], "malformed nonempty model output")

    def test_malformed_base_is_not_discarded_or_repaired_by_bridge(self):
        self.llm.predict_mm.side_effect = [("malformed base", None, {"raw": True})]
        with self.bridge(), self.assertRaises(IndexError):
            self.agent.step("goal")
        self.policy.assert_not_called()
        self.assertEqual(self.record()["passthrough"], "native malformed action handling")

    def test_provider_error_in_action_or_summary_propagates_unchanged(self):
        for phase in ["action", "summary"]:
            with self.subTest(phase=phase):
                error = InvalidModelCall("provider_refusal", "scripted", 1)
                self.llm.predict_mm.side_effect = ([error] if phase == "action"
                    else [self.base_tuple, error])
                self.root = self.root/phase
                with self.bridge(), self.assertRaises(InvalidModelCall) as caught:
                    self.agent.step("goal")
                self.assertIs(caught.exception, error)
                self.assertIs(self.agent.llm, self.llm)
                self.assertIsNone(self.record()["score"])

    def test_refiner_error_propagates_and_restores_owned_agent(self):
        original = self.agent.step.__func__
        self.policy.side_effect = OSError("local policy unavailable")
        with self.bridge(), self.assertRaises(OSError):
            self.agent.step("goal")
        self.assertIs(self.agent.llm, self.llm)
        self.assertIs(self.agent.step.__func__, original)
        self.assertNotIn("step", vars(self.agent))

    def test_empty_or_nonserializable_policy_result_is_invalid_before_action(self):
        for index, output in enumerate([RefinerOutput("", {}), RefinerOutput(None, {"x": float("nan")})]):
            self.root = self.root/str(index)
            self.llm.predict_mm.side_effect = [self.base_tuple]
            self.policy.return_value = output
            with self.bridge(), self.assertRaises(InvalidRefinerCall):
                self.agent.step("goal")
            self.assertFalse(self.agent.executed)
            self.assertIsNone(self.record()["score"])

    def test_duplicate_bridge_does_not_overwrite_evidence(self):
        with self.bridge():
            self.agent.step("goal")
        before = (self.root/"bridge/step000000.json").read_bytes()
        with self.assertRaises(FileExistsError):
            with self.bridge():
                self.fail("duplicate bridge entered")
        self.assertEqual((self.root/"bridge/step000000.json").read_bytes(), before)

    def test_bridge_evidence_is_private_and_never_a_transfer_score(self):
        with self.bridge():
            self.agent.step("goal")
        self.assertEqual(self.record()["benchmark_scores"], 0)
        self.assertEqual(self.record()["transfer_scores"], 0)
        for path in (self.root/"bridge").iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
