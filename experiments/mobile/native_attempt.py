"""A bounded, development-only native AndroidWorld Base attempt.

This starts no emulator, queue, model service or retry. A private protocol must
already bind the supplied task, native sources, snapshot and provider readiness.
The native episode runner owns actions and timing. Formal transfer and Refiner
evaluation are deliberately not implemented by this development entry point.
"""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import sys

from frozen_task import frozen_native_task
from environment_guard import native_environment_guard

ANDROID_COMMIT = "55ddaf5edc6f6c25673aed343907d5b9dca66f84"


class InvalidAttempt(RuntimeError):
    """Invalid preparation/execution, never a scored task failure."""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _private_write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o600)


def read_bound_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha256:
        raise InvalidAttempt("Frozen input SHA differs")
    return json.loads(raw)


def validate_protocol(spec, *, task):
    if spec.get("scope") != "development" or spec.get("method") != "native_m3a_base":
        raise InvalidAttempt("Only the registered development Base method is implemented")
    if spec.get("android_source_commit") != ANDROID_COMMIT:
        raise InvalidAttempt("Android source binding differs")
    if type(spec.get("seed")) is not int or task.params.get("seed") != spec["seed"]:
        raise InvalidAttempt("Freeze the native task seed in params before initialization")
    if (spec.get("task_family") != type(task).__name__ or spec.get("task_params") != task.params
            or spec.get("app_names") != list(task.app_names)
            or spec.get("goal_sha256") != sha(task.goal.encode("utf-8"))):
        raise InvalidAttempt("Task identity, params, apps or goal differs")
    if type(spec.get("max_steps")) is not int or not 1 <= spec["max_steps"] <= 1000:
        raise InvalidAttempt("A bounded positive native step budget is required")
    if type(spec.get("start_on_home_screen")) is not bool:
        raise InvalidAttempt("Bind the native start-screen setting")
    if task.initialized:
        raise InvalidAttempt("Task is already initialized")
    if not isinstance(spec.get("run_id"), str) or not spec["run_id"]:
        raise InvalidAttempt("Bind a private run identifier")


def _execute_attempt(*, spec, protocol_sha256, task, env, agent,
                     run_episode, lifecycle, audit_dir):
    """Shared control boundary; injectable functions support offline checks."""
    audit = Path(audit_dir)
    audit.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = dict(scope="development-only native Base; not benchmark or transfer",
        run_id=spec["run_id"], protocol_sha256=protocol_sha256,
        task_family=spec["task_family"], seed=spec["seed"],
        valid=False, score=None, benchmark_scores=0, transfer_scores=0,
        episode_returned=False, scorer_calls=0,
        started_utc=datetime.now(timezone.utc).isoformat())
    _private_write(audit/"started.json", json.dumps(record, indent=2).encode())
    try:
        with lifecycle(audit/"lifecycle"):
            episode = run_episode(goal=task.goal, agent=agent,
                max_n_steps=spec["max_steps"],
                start_on_home_screen=spec["start_on_home_screen"])
            record.update(episode_returned=True, agent_done=bool(episode.done))
            # Write-only private evidence. No pickle reader or arbitrary loader.
            raw = pickle.dumps(episode, protocol=pickle.HIGHEST_PROTOCOL)
            _private_write(audit/"episode.pickle", raw)
            record["episode_sha256"] = sha(raw)
            record["scorer_calls"] = 1
            candidate = float(task.is_successful(env))
            record["scorer_value_before_cleanup"] = candidate if math.isfinite(candidate) else repr(candidate)
            if not math.isfinite(candidate) or not 0 <= candidate <= 1:
                raise InvalidAttempt("Native scorer returned a value outside [0, 1]")
        # A valid zero is retained. A teardown failure prevents this commit.
        record.update(valid=True, score=candidate)
        return record
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _private_write(audit/"result.json", json.dumps(record, indent=2).encode())


def run_native_base_development(*, protocol_path, protocol_sha256,
                                provider_readiness_path, task, env, agent,
                                snapshot_device, snapshot_archive, audit_dir):
    """Bind actual native objects, then run one already registered task.

    Provider readiness is required before any task initialization or model call.
    The existing shared endpoint and native image/action semantics stay with the
    supplied PacedClaudeWrapper and M3A. There is no service fallback or retry.
    """
    spec = read_bound_json(protocol_path, protocol_sha256)
    validate_protocol(spec, task=task)
    provider = read_bound_json(provider_readiness_path, spec["provider_readiness_sha256"])
    if provider.get("external_provider_transport_ready") is not True:
        raise InvalidAttempt("Bound provider readiness does not permit model execution")
    if sha(Path(snapshot_archive).read_bytes()) != spec["snapshot_archive_sha256"]:
        raise InvalidAttempt("Snapshot archive differs")

    run_episode = _bind_native_contract(spec=spec, task=task, env=env, agent=agent)

    def lifecycle(path):
        task.set_device_time(env)
        return frozen_native_task(task, env, device=snapshot_device,
            archive=snapshot_archive, expected_sha256=spec["snapshot_archive_sha256"],
            audit_dir=path)

    def guarded_episode(**settings):
        with native_environment_guard(agent, env, audit_dir=Path(audit_dir)/"environment_guard"):
            return run_episode(**settings)

    return _execute_attempt(spec=spec, protocol_sha256=protocol_sha256,
        task=task, env=env, agent=agent, run_episode=guarded_episode,
        lifecycle=lifecycle, audit_dir=audit_dir)


def _bind_native_contract(*, spec, task, env, agent):
    from android_world import episode_runner, suite_utils
    from android_world.agents import base_agent, infer, m3a, m3a_utils
    from android_world.task_evals import task_eval
    from android_world.env import adb_utils, android_world_controller, interface, actuation
    from claude_m3a import PacedClaudeWrapper

    modules = [episode_runner, suite_utils, base_agent, infer, m3a, m3a_utils,
               task_eval, sys.modules[type(task).__module__]]
    observed = {module.__name__: sha(Path(inspect.getsourcefile(module)).read_bytes())
                for module in modules}
    if observed != spec["native_source_sha256"]:
        raise InvalidAttempt("Actual imported native sources differ")
    environment_sources = {module.__name__: sha(Path(inspect.getsourcefile(module)).read_bytes())
        for module in [adb_utils, android_world_controller, interface, actuation]}
    if environment_sources != spec["environment_source_sha256"]:
        raise InvalidAttempt("Actual imported native environment sources differ")
    if (not isinstance(task, task_eval.TaskEval)
            or type(agent) is not m3a.M3A or agent.env is not env
            or not isinstance(agent.llm, PacedClaudeWrapper)):
        raise InvalidAttempt("Bind unmodified native M3A and the explicit Claude wrapper")
    if (agent.llm.endpoint != spec["paced_endpoint"]
            or agent.llm.max_attempts != spec["model_max_attempts"]
            or list(agent.llm.retry_delays) != spec["model_retry_delays"]):
        raise InvalidAttempt("Provider route or declared attempt policy differs")
    if (agent.additional_guidelines is not None or agent.transition_pause != 1.0
            or agent.wait_after_action_seconds != 2.0):
        raise InvalidAttempt("Native M3A prompt or timing settings differ")
    if (spec["max_steps"] != suite_utils._allocate_step_budget(task.complexity)
            or spec["start_on_home_screen"] != task.start_on_home_screen
            or task.name.lower().startswith("miniwob")):
        raise InvalidAttempt("Only the original Android task episode settings are supported")

    return episode_runner.run_episode
