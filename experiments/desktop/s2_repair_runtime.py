"""Bounded recovery for malformed S2 DAG responses, without changing plans.

Install only for an explicitly recorded repair run. Invalid replies are retained
and never converted into guessed plans or scored actions.
"""

from functools import wraps
import json
import logging
from pathlib import Path
import shlex
import time
import uuid


class InvalidDagResponse(RuntimeError):
    """The planner returned text that its existing DAG parser rejected."""


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


def install_environment_repairs(packages_directory):
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
        return execute(self, *args, **kwargs)

    DesktopEnv._start_emulator = start_when_ready
    SetupController._execute_setup = execute_with_dependencies
    DesktopEnv._causalact_environment_repair_installed = True
