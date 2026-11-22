"""Preserve typed device failures even when native M3A catches an action error."""
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import pickle
import subprocess


class InvalidEnvironmentCall(RuntimeError):
    """A device/transport failure, not a model action failure or task zero."""


def infrastructure_error(error):
    if isinstance(error, (OSError, http.client.HTTPException, subprocess.TimeoutExpired)):
        return True
    for kind in type(error).__mro__:
        if kind.__module__ == "grpc" and kind.__name__ == "RpcError":
            return True
        if (kind.__module__ == "android_env.components.errors"
                and kind.__name__ == "AdbControllerError"):
            return True
    # The pinned ADB helper uses plain RuntimeError for a non-OK response.
    # Do not classify arbitrary RuntimeError or invalid action ValueError as
    # infrastructure failures based on their text.
    trace = error.__traceback__
    while trace:
        frame = trace.tb_frame
        if (isinstance(error, RuntimeError)
                and frame.f_globals.get("__name__") == "android_world.env.adb_utils"
                and frame.f_code.co_name == "check_ok"):
            return True
        trace = trace.tb_next
    return False


def _write(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


class _ObservedEnvironment:
    def __init__(self, original, record):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_record", record)
        object.__setattr__(self, "_failure", None)

    def __getattr__(self, name):
        return getattr(self._original, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._original, name, value)

    def execute_action(self, *args, **kwargs):
        try:
            return self._original.execute_action(*args, **kwargs)
        except Exception as error:
            if infrastructure_error(error):
                if self._failure is None:
                    object.__setattr__(self, "_failure", error)
                    self._record.update(environment_failure_type=type(error).__name__,
                                        environment_failure=str(error))
            else:
                self._record["native_action_errors_preserved"] += 1
            raise

    def raise_if_failed(self):
        if self._failure is not None:
            raise InvalidEnvironmentCall("Typed device failure during native action execution") from self._failure


@contextmanager
def native_environment_guard(agent, env, *, audit_dir):
    """Forward native calls unchanged, and stop before scoring a swallowed fault.

    Only the owned agent is wrapped temporarily. The environment object, native
    source, prompts, actions and pauses are never patched. A model action error
    keeps the original native behavior. No retries or model calls occur here.
    """
    if agent.env is not env:
        raise ValueError("The supplied agent must own this environment")
    audit = Path(audit_dir)
    audit.mkdir(mode=0o700, parents=True, exist_ok=False)
    record = dict(scope="native environment-error boundary only", valid=False, score=None,
                  native_action_errors_preserved=0, benchmark_scores=0, model_requests=0)
    proxy = _ObservedEnvironment(env, record)
    original_step = agent.step
    missing = object()
    instance_step = vars(agent).get("step", missing)
    failure = None

    def guarded_step(*args, **kwargs):
        try:
            result = original_step(*args, **kwargs)
        except BaseException:
            proxy.raise_if_failed()
            raise
        if proxy._failure is not None:
            # Preserve the native failed-step evidence without exposing it to
            # the task scorer. This file may contain private screenshots/text.
            try:
                _write(audit/"failed_native_step.pickle", pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            except (OSError, TypeError, pickle.PickleError) as error:
                record["failed_step_evidence_error"] = type(error).__name__
        proxy.raise_if_failed()
        return result

    try:
        agent.env = proxy
        agent.step = guarded_step
        yield
        proxy.raise_if_failed()
        record["valid"] = True
    except BaseException as error:
        failure = error
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        agent.env = env
        if instance_step is missing:
            del agent.step
        else:
            agent.step = instance_step
        record["original_agent_restored"] = True
        try:
            _write(audit/"receipt.json", json.dumps(record, indent=2).encode())
        except OSError:
            if failure is None:
                raise
