"""Bounded recovery for malformed S2 DAG responses, without changing plans.

Install only for an explicitly recorded repair run. Invalid replies are retained
and never converted into guessed plans or scored actions.
"""

from functools import wraps
import base64
import hashlib
import json
import logging
from pathlib import Path
import shlex
import time
import uuid


def screenshot_image_bytes(screenshot):
    """Keep native image bytes for OCR and the unchanged model image encoder."""
    if isinstance(screenshot, (bytes, bytearray, memoryview)):
        return bytes(screenshot)
    if isinstance(screenshot, str):
        if screenshot.startswith("data:image/"):
            metadata, screenshot = screenshot.split(",", 1)
            if not metadata.endswith(";base64"):
                raise ValueError("Screenshot data URL must use base64")
        return base64.b64decode(screenshot, validate=True)
    import numpy as np
    if isinstance(screenshot, np.ndarray):
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.fromarray(screenshot).save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError("Screenshot must be image bytes, a base64 image, or an array")


def install_s2_image_bytes():
    """Support older wrappers that encoded screenshots before Agent S2 OCR."""
    from gui_agents.s2.agents.agent_s import AgentS2
    if getattr(AgentS2, "_causalact_image_bytes_installed", False):
        return
    predict = AgentS2.predict

    @wraps(predict)
    def predict_with_image_bytes(self, instruction, obs, *args, **kwargs):
        observation = obs.copy()
        observation["screenshot"] = screenshot_image_bytes(obs["screenshot"])
        return predict(self, instruction, observation, *args, **kwargs)

    AgentS2.predict = predict_with_image_bytes
    AgentS2._causalact_image_bytes_installed = True


class InvalidDagResponse(RuntimeError):
    """The planner returned text that its existing DAG parser rejected."""


class UnrenderedDesktop(RuntimeError):
    """A run requiring a visible desktop received a known blank framebuffer."""


def cursor_crop_fingerprint(screenshot):
    from PIL import Image
    from io import BytesIO
    image = Image.open(BytesIO(screenshot_image_bytes(screenshot))).convert("RGB")
    box = image.getbbox()
    if box is None:
        return None
    crop = image.crop(box)
    return f"{crop.width}x{crop.height}:" + hashlib.sha256(crop.tobytes()).hexdigest()


def initial_frame_problem(screenshot, known_cursor_crops=()):
    fingerprint = cursor_crop_fingerprint(screenshot)
    if fingerprint is None:
        return "all-black initial framebuffer"
    if fingerprint in known_cursor_crops:
        return "initial framebuffer contains only the known cursor on black"
    return None


def guarded_initial_reset(reset, audit_directory, known_cursor_crops=()):
    fingerprints = frozenset(known_cursor_crops)

    @wraps(reset)
    def checked(self, *args, **kwargs):
        obs = reset(self, *args, **kwargs)
        problem = initial_frame_problem(obs["screenshot"], fingerprints)
        if problem:
            directory = Path(audit_directory)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (uuid.uuid4().hex + ".png")
            path.write_bytes(screenshot_image_bytes(obs["screenshot"]))
            path.with_suffix(".json").write_text(json.dumps({
                "task_id": getattr(self, "task_id", None),
                "reason": problem, "classification": "invalid_environment_unscored",
            }))
            raise UnrenderedDesktop(problem + "; no model action has run")
        return obs

    return checked


def install_initial_frame_guard(audit_directory, known_cursor_crops=()):
    """Check the first model input after the runner's existing readiness wait."""
    import lib_run_single
    if getattr(lib_run_single, "_causalact_initial_frame_guard", False):
        return
    lib_run_single.run_single_example = guarded_initial_example(
        lib_run_single.run_single_example, audit_directory, known_cursor_crops
    )
    lib_run_single._causalact_initial_frame_guard = True


def guarded_initial_example(run_example, audit_directory, known_cursor_crops=()):
    """Leave reset, readiness waits, and later observations unchanged."""
    fingerprints = frozenset(known_cursor_crops)

    @wraps(run_example)
    def checked(agent, env, example, *args, **kwargs):
        predict = agent.predict
        first = True

        @wraps(predict)
        def first_prediction(instruction, obs, *predict_args, **predict_kwargs):
            nonlocal first
            if first:
                problem = initial_frame_problem(obs["screenshot"], fingerprints)
                if problem:
                    directory = Path(audit_directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / (uuid.uuid4().hex + ".png")
                    path.write_bytes(screenshot_image_bytes(obs["screenshot"]))
                    path.with_suffix(".json").write_text(json.dumps({
                        "task_id": example.get("id"),
                        "reason": problem,
                        "classification": "invalid_environment_unscored",
                        "observation_stage": "first_model_input_after_runner_wait",
                    }))
                    raise UnrenderedDesktop(problem + "; no model action has run")
                first = False
            return predict(instruction, obs, *predict_args, **predict_kwargs)

        agent.predict = first_prediction
        try:
            return run_example(agent, env, example, *args, **kwargs)
        finally:
            agent.predict = predict

    return checked


def checked_dag_parser(parser, retain):
    @wraps(parser)
    def checked(text):
        dag = parser(text)
        if dag is None:
            retain(text)
            raise InvalidDagResponse("S2 returned an incomplete or invalid DAG")
        return dag

    return checked


def bounded_dag_generator(generator, attempts=3):
    if attempts < 1:
        raise ValueError("attempts must be positive")

    @wraps(generator)
    def generate(*args, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                return generator(*args, **kwargs)
            except InvalidDagResponse:
                logging.getLogger(__name__).warning(
                    "Rejected invalid DAG reply, attempt %d/%d", attempt, attempts
                )
                if attempt == attempts:
                    raise

    return generate


def install_s2_plan_retry(audit_directory):
    from gui_agents.s2.agents import manager

    if getattr(manager, "_causalact_dag_repair_installed", False):
        return
    directory = Path(audit_directory)
    directory.mkdir(parents=True, exist_ok=True)

    def retain(text):
        path = directory / ("invalid_dag_" + uuid.uuid4().hex + ".json")
        path.write_text(
            json.dumps({"raw_response": text, "classification": "invalid_unscored"}),
            encoding="utf-8",
        )

    manager.parse_dag = checked_dag_parser(manager.parse_dag, retain)
    manager.Manager._generate_dag = bounded_dag_generator(
        manager.Manager._generate_dag, attempts=3
    )
    manager._causalact_dag_repair_installed = True


def guest_execute(controller, command, timeout=30):
    import requests

    response = requests.post(
        controller.http_server + "/execute",
        json={"command": command, "shell": False},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") != "success":
        raise RuntimeError("Guest command did not complete: " + str(result))
    return result


def wait_for_desktop(controller, timeout=360):
    """The guest HTTP service can start minutes before its window manager."""
    deadline = time.monotonic() + timeout
    last_error = "window manager not ready"
    while time.monotonic() < deadline:
        try:
            result = guest_execute(controller, ["wmctrl", "-m"], timeout=20)
            if result.get("returncode") == 0 and "Name: GNOME Shell" in result.get("output", ""):
                return
            last_error = result.get("error") or last_error
        except Exception as error:
            last_error = str(error)
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    raise RuntimeError("Guest desktop readiness timed out: " + last_error)


def prepare_expect_dependency(controller, packages_directory):
    """Install the official evaluator's dependency from cached Ubuntu packages."""
    result = guest_execute(controller, ["dpkg-query", "-W", "-f=${Status}", "expect"])
    if result.get("returncode") == 0 and result.get("output") == "install ok installed":
        return
    directory = Path(packages_directory)
    packages = [
        *directory.glob("tcl8.6_*.deb"),
        *directory.glob("tcl-expect_*.deb"),
        *directory.glob("expect_*.deb"),
    ]
    if len(packages) != 3:
        raise RuntimeError("Expected exactly the three cached evaluator dependency packages")
    target = "/tmp/causalact_evaluator_dependencies"
    controller._upload_file_setup([
        {"local_path": str(p), "path": target + "/" + p.name} for p in packages
    ])
    command = (
        "printf '%s\\n' " + shlex.quote(controller.client_password)
        + " | sudo -S -- dpkg -i "
        + " ".join(shlex.quote(target + "/" + p.name) for p in packages)
    )
    result = guest_execute(controller, ["bash", "-lc", command], timeout=150)
    if result.get("returncode") != 0:
        raise RuntimeError("Evaluator dependency installation failed; task remains unscored")
    result = guest_execute(controller, ["dpkg-query", "-W", "-f=${Status}", "expect"])
    if result.get("returncode") != 0 or result.get("output") != "install ok installed":
        raise RuntimeError("Evaluator dependency was not installed")


def prepare_cssselect_dependency(controller, command):
    """Install the evaluator's existing wheel offline, then verify its version."""
    expected = ["pip", "install", "/home/user/cssselect-1.3.0-py3-none-any.whl"]
    if command != expected:
        raise ValueError("Expected the recorded local cssselect evaluator wheel")
    result = guest_execute(
        controller,
        ["env", "PIP_NO_INDEX=1", "PIP_DISABLE_PIP_VERSION_CHECK=1", *command],
        timeout=150,
    )
    if result.get("returncode") != 0:
        raise RuntimeError("Evaluator wheel installation failed; task remains unscored")
    result = guest_execute(
        controller,
        ["python", "-c", "from importlib.metadata import version; print(version('cssselect'))"],
        timeout=150,
    )
    if result.get("returncode") != 0 or result.get("output", "").strip() != "1.3.0":
        raise RuntimeError("Evaluator wheel version check failed; task remains unscored")


def install_environment_repairs(packages_directory, offline_evaluator_wheel=False):
    import inspect
    from desktop_env.controllers.setup import SetupController
    from desktop_env.desktop_env import DesktopEnv

    if getattr(DesktopEnv, "_causalact_environment_repair_installed", False):
        return
    start = DesktopEnv._start_emulator
    execute = SetupController._execute_setup
    signature = inspect.signature(execute)

    @wraps(start)
    def start_when_ready(self, *args, **kwargs):
        result = start(self, *args, **kwargs)
        try:
            wait_for_desktop(self.setup_controller)
        except Exception:
            self.provider.stop_emulator(self.path_to_vm)
            raise
        return result

    @wraps(execute)
    def execute_with_dependencies(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        command = bound.arguments.get("command")
        if isinstance(command, str) and "apt-get install -y expect" in command:
            prepare_expect_dependency(self, packages_directory)
        if (offline_evaluator_wheel and not bound.arguments.get("shell", False)
                and command == ["pip", "install", "/home/user/cssselect-1.3.0-py3-none-any.whl"]):
            return prepare_cssselect_dependency(self, command)
        return execute(self, *args, **kwargs)

    DesktopEnv._start_emulator = start_when_ready
    SetupController._execute_setup = execute_with_dependencies
    DesktopEnv._causalact_environment_repair_installed = True
