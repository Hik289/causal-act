from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

from claude_m3a import InvalidModelCall
from environment_guard import (InvalidEnvironmentCall, infrastructure_error,
                               native_environment_guard)


class FixtureEnvironment:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0
        self.interaction_cache = "before"

    def execute_action(self, action):
        self.calls += 1
        if self.error:
            raise self.error
        return "original result"


class CatchingAgent:
    def __init__(self, env):
        self.env = env
        self.calls = 0

    def step(self, goal):
        self.calls += 1
        try:
            result = self.env.execute_action(goal)
        except Exception:
            return SimpleNamespace(done=False, data={"native_action_error": True})
        return SimpleNamespace(done=False, data={"result": result})


class EnvironmentGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"guard"

    def test_swallowed_device_failure_stops_after_first_native_step(self):
        original = OSError("device disconnected")
        env = FixtureEnvironment(original)
        agent = CatchingAgent(env)
        with self.assertRaises(InvalidEnvironmentCall) as caught:
            with native_environment_guard(agent, env, audit_dir=self.path):
                agent.step("first")
                agent.step("must not execute")
        self.assertIs(caught.exception.__cause__, original)
        self.assertEqual(env.calls, 1)
        self.assertEqual(agent.calls, 1)
        self.assertIs(agent.env, env)
        self.assertNotIn("step", vars(agent))
        self.assertTrue((self.path/"failed_native_step.pickle").exists())

    def test_invalid_model_action_stays_with_native_behavior(self):
        env = FixtureEnvironment(ValueError("unsupported action argument"))
        agent = CatchingAgent(env)
        with native_environment_guard(agent, env, audit_dir=self.path):
            result = agent.step("invalid model action")
        self.assertFalse(result.done)
        value = json.loads((self.path/"receipt.json").read_text())
        self.assertTrue(value["valid"])
        self.assertEqual(value["native_action_errors_preserved"], 1)

    def test_arbitrary_runtime_error_is_not_reclassified_by_its_message(self):
        self.assertFalse(infrastructure_error(RuntimeError("device disconnected")))

    def test_pinned_adb_response_error_is_classified_by_origin(self):
        scope = {"__name__": "android_world.env.adb_utils"}
        exec("def check_ok():\n    raise RuntimeError('non-OK ADB response')", scope)
        try:
            scope["check_ok"]()
        except RuntimeError as error:
            self.assertTrue(infrastructure_error(error))

    def test_native_adb_controller_subclasses_and_grpc_errors_are_recognized(self):
        adb = type("AdbControllerError", (RuntimeError,), {"__module__": "android_env.components.errors"})
        grpc = type("RpcError", (Exception,), {"__module__": "grpc"})
        self.assertTrue(infrastructure_error(type("SpecificADBFailure", (adb,), {})()))
        self.assertTrue(infrastructure_error(type("SpecificRPCFailure", (grpc,), {})()))

    def test_healthy_execution_returns_exact_native_result(self):
        env = FixtureEnvironment()
        agent = CatchingAgent(env)
        with native_environment_guard(agent, env, audit_dir=self.path):
            result = agent.step("unchanged")
            agent.env.interaction_cache = "after"
        self.assertEqual(result.data, {"result": "original result"})
        self.assertEqual(env.interaction_cache, "after")
        self.assertEqual(env.calls, 1)

    def test_existing_instance_step_override_is_restored(self):
        env = FixtureEnvironment()
        agent = CatchingAgent(env)
        original = agent.step
        agent.step = original
        with native_environment_guard(agent, env, audit_dir=self.path):
            agent.step("unchanged")
        self.assertIs(vars(agent)["step"], original)

    def test_original_provider_exception_propagates_without_fallback(self):
        env = FixtureEnvironment()
        agent = CatchingAgent(env)
        original = InvalidModelCall("provider_refusal", "fixture", 1)
        def fail(goal):
            raise original
        agent.step = fail
        with self.assertRaises(InvalidModelCall) as caught:
            with native_environment_guard(agent, env, audit_dir=self.path):
                agent.step("goal")
        self.assertIs(caught.exception, original)
        self.assertEqual(env.calls, 0)
        self.assertIs(agent.step, fail)

    def test_duplicate_audit_cannot_reexecute_or_overwrite(self):
        env = FixtureEnvironment()
        agent = CatchingAgent(env)
        with native_environment_guard(agent, env, audit_dir=self.path):
            agent.step("first")
        before = (self.path/"receipt.json").read_bytes()
        with self.assertRaises(FileExistsError):
            with native_environment_guard(agent, env, audit_dir=self.path):
                agent.step("must not run")
        self.assertEqual(env.calls, 1)
        self.assertEqual((self.path/"receipt.json").read_bytes(), before)

    def test_receipt_and_failure_evidence_are_private(self):
        env = FixtureEnvironment(OSError("device disconnected"))
        agent = CatchingAgent(env)
        with self.assertRaises(InvalidEnvironmentCall):
            with native_environment_guard(agent, env, audit_dir=self.path):
                agent.step("goal")
        for p in self.path.iterdir():
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertIsNone(json.loads((self.path/"receipt.json").read_text()).get("score"))


if __name__ == "__main__":
    unittest.main()
