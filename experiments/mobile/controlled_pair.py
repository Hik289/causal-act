"""One frozen development Base/Refiner pair, with no launcher or retry budget.

Both arms use fresh native task/agent objects and the same owned environment,
snapshot and native settings. This supplies no learned policy or E7 result.
"""
from __future__ import annotations

from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
from pathlib import Path

from environment_guard import native_environment_guard
from frozen_task import frozen_native_task
from native_attempt import (ANDROID_COMMIT, InvalidAttempt, _bind_native_contract,
                            _execute_attempt, _private_write, read_bound_json,
                            validate_protocol)
from refiner_bridge import bind_native_refiner_contract, native_refiner_bridge

IMPLEMENTATION_FILES = ("controlled_pair.py", "native_attempt.py", "refiner_bridge.py",
                        "environment_guard.py", "frozen_task.py", "claude_m3a.py")
ARMS = ("base", "refiner")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_identity():
    return {name: file_sha(Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}


def _bound_protocol(spec, key):
    binding = spec.get(key)
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise InvalidAttempt("Missing frozen " + key)
    if not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute():
        raise InvalidAttempt("Protocol references must be absolute")
    return read_bound_json(binding["path"], binding["sha256"])


def _preflight(spec, *, tasks, agents, env, provider_readiness_path, snapshot_archive, refiner):
    if (spec.get("scope") != "development"
            or spec.get("method") != "native_m3a_controlled_pair"
            or spec.get("android_source_commit") != ANDROID_COMMIT
            or not isinstance(spec.get("run_id"), str) or not spec["run_id"]):
        raise InvalidAttempt("Only a registered development pair is implemented")
    if (spec.get("order") not in [["base", "refiner"], ["refiner", "base"]]
            or type(spec.get("attempts_per_arm")) is not int or spec["attempts_per_arm"] != 1):
        raise InvalidAttempt("Freeze both arm order and one attempt per arm")
    if spec.get("implementation_sha256") != implementation_identity():
        raise InvalidAttempt("Actual pair implementation differs")
    base = _bound_protocol(spec, "native_protocol")
    _bound_protocol(spec, "bridge_protocol")
    if set(tasks) != set(ARMS) or set(agents) != set(ARMS):
        raise InvalidAttempt("Both arms need separate native task and agent objects")
    if tasks["base"] is tasks["refiner"] or agents["base"] is agents["refiner"]:
        raise InvalidAttempt("Task and agent state must not carry between arms")
    for task in tasks.values():
        validate_protocol(base, task=task)
    provider = read_bound_json(provider_readiness_path, base["provider_readiness_sha256"])
    if provider.get("external_provider_transport_ready") is not True:
        raise InvalidAttempt("Bound provider readiness does not permit model execution")
    if file_sha(snapshot_archive) != base["snapshot_archive_sha256"]:
        raise InvalidAttempt("Snapshot archive differs")
    runners = {name: _bind_native_contract(spec=base, task=tasks[name], env=env,
                                         agent=agents[name]) for name in ARMS}
    # Reject a bad policy binding before either arm can initialize a task.
    bind_native_refiner_contract(agents["refiner"], refiner=refiner,
        protocol_path=spec["bridge_protocol"]["path"],
        protocol_sha256=spec["bridge_protocol"]["sha256"])
    return base, runners


def _validate_result(result, *, native_sha, method):
    if (result.get("protocol_sha256") != native_sha or result.get("method") != method
            or type(result.get("valid")) is not bool
            or result.get("benchmark_scores") != 0 or result.get("transfer_scores") != 0):
        raise InvalidAttempt("An arm receipt has the wrong identity or scope")
    score = result.get("score")
    if result["valid"]:
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise InvalidAttempt("An arm has an invalid native score")
        if result.get("scorer_calls") != 1 or not result.get("episode_returned"):
            raise InvalidAttempt("A valid arm needs the completed native scoring boundary")
    elif score is not None:
        raise InvalidAttempt("An invalid arm must retain a null score")


def _method(arm):
    return "native_m3a_base" if arm == "base" else "native_m3a_refiner_development"


def _arm_receipt(root, arm, *, pair_sha, native_sha):
    path = root/(arm + ".receipt.json")
    attempt = root/arm
    if not path.exists():
        if attempt.exists():
            raise InvalidAttempt("Uncommitted arm evidence requires inspection; never rerun it")
        return None
    receipt = json.loads(path.read_text())
    if (receipt.get("pair_protocol_sha256") != pair_sha or receipt.get("arm") != arm
            or receipt.get("attempts_consumed") != 1):
        raise InvalidAttempt("An arm receipt belongs to a different pair")
    result_path = attempt/"result.json"
    if file_sha(result_path) != receipt["result_sha256"]:
        raise InvalidAttempt("Existing arm result changed")
    result = json.loads(result_path.read_text())
    _validate_result(result, native_sha=native_sha, method=_method(arm))
    observed = {str(p.relative_to(attempt)) for p in attempt.rglob("*") if p.is_file()}
    if observed != set(receipt["evidence_sha256"]):
        raise InvalidAttempt("Existing arm evidence file set changed")
    for name, digest in receipt["evidence_sha256"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or (attempt/name).is_symlink()
                or any((attempt/parent).is_symlink() for parent in relative.parents)
                or file_sha(attempt/name) != digest):
            raise InvalidAttempt("Existing arm evidence changed")
    if result["valid"] and receipt["evidence_sha256"].get("episode.pickle") != result["episode_sha256"]:
        raise InvalidAttempt("Native episode evidence is missing")
    return result


def _commit_arm(root, arm, *, pair_sha, native_sha):
    attempt = root/arm
    result_path = attempt/"result.json"
    if not result_path.exists():
        # A crash or failed evidence write has consumed the arm. Preserve its
        # partial directory for inspection rather than granting another try.
        return
    result = json.loads(result_path.read_text())
    _validate_result(result, native_sha=native_sha, method=_method(arm))
    receipt = dict(pair_protocol_sha256=pair_sha, arm=arm, attempts_consumed=1,
        result_sha256=file_sha(result_path), evidence_sha256={
            str(p.relative_to(attempt)): file_sha(p) for p in attempt.rglob("*") if p.is_file()})
    _private_write(root/(arm + ".receipt.json"), json.dumps(receipt, indent=2).encode())


def _pair_report(spec, results, pair_sha):
    comparable = all(name in results and results[name]["valid"] for name in ARMS)
    return dict(scope="development only; not benchmark or transfer",
        run_id=spec["run_id"], pair_protocol_sha256=pair_sha,
        order=spec["order"], valid_pair=comparable, effective_pair_denominator=int(comparable),
        arm_state={name: ("pending" if name not in results else
                         "invalid_uncommitted" if results[name].get("uncommitted") else
                         "valid" if results[name]["valid"] else "invalid") for name in ARMS},
        scores={name: results[name]["score"] if name in results else None for name in ARMS},
        refiner_minus_base=(results["refiner"]["score"] - results["base"]["score"]
                            if comparable else None),
        benchmark_scores=0, transfer_scores=0, attempts_per_arm=1, automatic_retries=0)


def _run_pair(*, spec, pair_sha, audit_dir, run_arm):
    """Owned evidence/resume boundary, independently testable without Android."""
    root = Path(audit_dir)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = root/"pair.lock"
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InvalidAttempt("This pair is already running") from error
        binding = dict(pair_protocol_sha256=pair_sha, protocol=spec)
        binding_path = root/"pair_binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text()) != binding:
                raise InvalidAttempt("Existing pair protocol differs")
        else:
            if any(p.name != "pair.lock" for p in root.iterdir()):
                raise InvalidAttempt("Unbound existing pair evidence requires inspection")
            _private_write(binding_path, json.dumps(binding, indent=2).encode())
        native_sha = spec["native_protocol"]["sha256"]
        results = {}

        def report():
            value = _pair_report(spec, results, pair_sha)
            # Each resume writes a separate report, preserving earlier states.
            count = len(list(root.glob("report*.json")))
            _private_write(root/("report%06d.json" % count), json.dumps(value, indent=2).encode())
            return value

        # Check every existing arm before starting any missing one.
        for arm in ARMS:
            result = _arm_receipt(root, arm, pair_sha=pair_sha, native_sha=native_sha)
            if result is not None:
                results[arm] = result
        for arm in spec["order"]:
            if arm in results:
                continue
            try:
                run_arm(arm, root/arm)
            except BaseException:
                try:
                    _commit_arm(root, arm, pair_sha=pair_sha, native_sha=native_sha)
                    result = _arm_receipt(root, arm, pair_sha=pair_sha, native_sha=native_sha)
                    if result is not None:
                        results[arm] = result
                    else:
                        results[arm] = dict(valid=False, score=None, uncommitted=True)
                except Exception:
                    results[arm] = dict(valid=False, score=None, uncommitted=True)
                try:
                    report()
                except Exception:
                    pass  # Evidence failure must not hide the original failure.
                raise
            _commit_arm(root, arm, pair_sha=pair_sha, native_sha=native_sha)
            results[arm] = _arm_receipt(root, arm, pair_sha=pair_sha, native_sha=native_sha)
            if results[arm] is None:
                raise InvalidAttempt("Arm returned without committed native evidence")
        return report()


def run_native_pair_development(*, protocol_path, protocol_sha256, tasks, agents, env,
        provider_readiness_path, snapshot_device, snapshot_archive, refiner, audit_dir):
    """Resume only unstarted arms under the original one-attempt allowance.

    Provider readiness is checked before creating a pair or initializing a task.
    Fresh tasks/agents must be supplied even on resume. A valid zero and an
    invalid attempt are both immutable; an interrupted uncommitted arm is blocked.
    """
    spec = read_bound_json(protocol_path, protocol_sha256)
    base, runners = _preflight(spec, tasks=tasks, agents=agents, env=env,
        provider_readiness_path=provider_readiness_path,
        snapshot_archive=snapshot_archive, refiner=refiner)

    def run_arm(arm, path):
        task, agent = tasks[arm], agents[arm]

        def lifecycle(target):
            task.set_device_time(env)
            return frozen_native_task(task, env, device=snapshot_device,
                archive=snapshot_archive, expected_sha256=base["snapshot_archive_sha256"],
                audit_dir=target)

        def episode(**settings):
            bridge = (native_refiner_bridge(agent, refiner=refiner,
                protocol_path=spec["bridge_protocol"]["path"],
                protocol_sha256=spec["bridge_protocol"]["sha256"], audit_dir=path/"refiner_bridge")
                if arm == "refiner" else nullcontext())
            with bridge, native_environment_guard(agent, env, audit_dir=path/"environment_guard"):
                return runners[arm](**settings)

        return _execute_attempt(spec=base,
            protocol_sha256=spec["native_protocol"]["sha256"], task=task, env=env, agent=agent,
            run_episode=episode, lifecycle=lifecycle, audit_dir=path, attempt_method=_method(arm))

    return _run_pair(spec=spec, pair_sha=protocol_sha256, audit_dir=audit_dir, run_arm=run_arm)
