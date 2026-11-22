"""Development-only Refiner hook at the native M3A proposal boundary.

The original M3A parses, executes and summarizes the returned proposal. This
module supplies no learned policy, additional model request, device action or
scoring loop.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import threading

from environment_guard import _write
from native_attempt import ANDROID_COMMIT, InvalidAttempt, read_bound_json

# This exact native step calls predict_mm first for an action, then for its
# summary. The phase boundary must be re-audited before accepting other source.
M3A_SHA256 = "672f96be7073c7fc35e0e7810d1ba7d0e277e639c4abc16e572480f95daa564d"


@dataclass(frozen=True)
class RefinerInput:
    goal: str
    prompt: str
    images: tuple
    base_output: str
    base_reason: str
    base_action: dict


@dataclass(frozen=True)
class RefinerOutput:
    # None means keep the exact original tuple. A nonempty replacement is
    # handed to the native parser, including a malformed model action.
    text: str | None
    evidence: dict


class InvalidRefinerCall(RuntimeError):
    """Broken Refiner interface/transport; never a scored task zero."""


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StepInference:
    def __init__(self, original, *, refiner, parse_action, goal, record):
        self.original = original
        self.refiner, self.parse_action = refiner, parse_action
        self.goal, self.record = goal, record
        self.calls = 0

    def predict_mm(self, text_prompt, images):
        self.calls += 1
        response = self.original.predict_mm(text_prompt, images)
        if self.calls != 1:
            self.record["summary_calls_forwarded"] += 1
            return response
        text, is_safe, raw = response
        self.record["base_output"] = text
        if is_safe is False or not raw:
            self.record["passthrough"] = "native safety or response handling"
            return response
        try:
            reason, action = self.parse_action(text)
        except Exception:
            reason, action = None, None
        if not reason or action is None:
            self.record["passthrough"] = "native malformed action handling"
            return response
        # A policy cannot change the native screenshots, prompt or action by
        # mutating its inputs. It sees no task object, scorer or success label.
        proposal = RefinerInput(goal=self.goal, prompt=text_prompt,
            images=tuple(deepcopy(image) for image in images), base_output=text,
            base_reason=reason, base_action=deepcopy(action))
        self.record["refiner_calls"] += 1
        refined = self.refiner(proposal)
        if not isinstance(refined, RefinerOutput) or not isinstance(refined.evidence, dict):
            raise InvalidRefinerCall("Refiner must return text/evidence using RefinerOutput")
        # Evidence must be serializable before anything can be executed.
        try:
            evidence = json.loads(json.dumps(refined.evidence, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise InvalidRefinerCall("Refiner evidence is not finite JSON") from error
        self.record["refiner_evidence"] = evidence
        if refined.text is None:
            self.record["passthrough"] = "Refiner kept native proposal"
            return response
        if not isinstance(refined.text, str) or not refined.text.strip():
            raise InvalidRefinerCall("Refiner produced an empty response")
        self.record.update(refined_output=refined.text, overridden=refined.text != text)
        if refined.text == text:
            return response
        # Do not attribute a changed proposal to the original provider output.
        provenance = dict(base_response=raw, refiner_evidence=evidence,
                          scope="development Refiner proposal")
        return refined.text, is_safe, provenance


@contextmanager
def _owned_bridge(agent, *, refiner, parse_action, audit_dir, binding):
    audit = Path(audit_dir)
    audit.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write(audit/"binding.json", json.dumps(binding, indent=2).encode())
    original_step, original_llm = agent.step, agent.llm
    missing = object()
    instance_step = vars(agent).get("step", missing)
    lock = threading.Lock()
    sequence = 0

    def step(goal):
        nonlocal sequence
        if not lock.acquire(blocking=False):
            raise InvalidRefinerCall("Concurrent steps on the same native agent are not permitted")
        record = dict(scope="scripted/development bridge; not benchmark or transfer",
            step=sequence, valid=False, score=None, refiner_calls=0,
            summary_calls_forwarded=0, overridden=False, benchmark_scores=0,
            transfer_scores=0)
        path = audit/f"step{sequence:06d}.json"
        sequence += 1
        failure = None
        try:
            if agent.llm is not original_llm:
                raise InvalidRefinerCall("The owned agent inference binding changed")
            agent.llm = _StepInference(original_llm, refiner=refiner,
                parse_action=parse_action, goal=goal, record=record)
            result = original_step(goal)
            record["valid"] = True
            return result
        except BaseException as error:
            failure = error
            record.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            agent.llm = original_llm
            lock.release()
            try:
                _write(path, json.dumps(record, indent=2).encode())
            except OSError:
                if failure is None:
                    raise

    try:
        agent.step = step
        yield
    finally:
        agent.llm = original_llm
        if instance_step is missing:
            del agent.step
        else:
            agent.step = instance_step


@contextmanager
def native_refiner_bridge(agent, *, refiner, protocol_path, protocol_sha256, audit_dir):
    """Bind an already supplied development policy to the exact native M3A.

    This is not a formal runner or trained checkpoint verification. The caller
    must separately gate provider readiness and native task initialization. The
    identity manifest binds the supplied policy's code/assets/settings; it must
    not label a scripted development callback as a learned Refiner result.
    """
    from android_world.agents import agent_utils, m3a, m3a_utils
    from android_world.env import json_action
    from claude_m3a import PacedClaudeWrapper

    spec = read_bound_json(protocol_path, protocol_sha256)
    if (spec.get("scope") != "development" or spec.get("method") != "native_m3a_refiner_bridge"
            or spec.get("android_source_commit") != ANDROID_COMMIT):
        raise InvalidAttempt("Only the registered development bridge is implemented")
    sources = {module.__name__: _sha_file(inspect.getsourcefile(module))
               for module in [m3a, m3a_utils, agent_utils, json_action]}
    if sources != spec.get("native_source_sha256") or sources[m3a.__name__] != M3A_SHA256:
        raise InvalidAttempt("Native proposal/summary boundary differs")
    if (type(agent) is not m3a.M3A or not isinstance(agent.llm, PacedClaudeWrapper)
            or not callable(refiner)):
        raise InvalidAttempt("Bind the original M3A, explicit Claude interface and policy")
    if (agent.additional_guidelines is not None or agent.transition_pause != 1.0
            or agent.wait_after_action_seconds != 2.0):
        raise InvalidAttempt("Native prompt and timing defaults must remain fixed")
    assets = spec.get("refiner_files_sha256")
    if not isinstance(assets, dict) or not assets or any(_sha_file(p) != h for p, h in assets.items()):
        raise InvalidAttempt("Bind the development policy code, assets and settings")
    policy_source = inspect.getsourcefile(refiner if inspect.isfunction(refiner) else type(refiner))
    if policy_source is None or str(Path(policy_source).resolve()) not in assets:
        raise InvalidAttempt("The actual supplied policy source must be in its identity manifest")

    def parse_action(text):
        reason, action = m3a_utils.parse_reason_action_output(text)
        if not reason or not action:
            return None, None
        return reason, json_action.JSONAction(**agent_utils.extract_json(action)).as_dict()

    with _owned_bridge(agent, refiner=refiner, parse_action=parse_action,
                       audit_dir=audit_dir, binding=spec):
        yield
