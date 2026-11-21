"""Restore a hash-bound app snapshot archive before native task initialization.

The caller supplies an already owned Android environment. No emulator, model,
task selection, task scoring or retry loop is started here.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import logging
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import tarfile
import threading
import uuid


class SnapshotError(RuntimeError):
    """Invalid environment preparation, not a benchmark task zero."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def archive_manifest(raw):
    result = {}
    canonical_names = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or str(name) in canonical_names:
                raise SnapshotError("Invalid or duplicate archive path")
            canonical_names.add(str(name))
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise SnapshotError("Unsupported archive member type")
            if member.issym() or member.islnk():
                target = member.linkname if member.islnk() else posixpath.join(str(name.parent), member.linkname)
                target = posixpath.normpath(target)
                if target.startswith("/") or target == ".." or target.startswith("../"):
                    raise SnapshotError("Archive link leaves snapshot directory")
            content = archive.extractfile(member).read() if member.isfile() else b""
            result[member.name] = dict(type=member.type.decode(), mode=member.mode,
                uid=member.uid, gid=member.gid, linkname=member.linkname, sha256=sha(content))
    if not result:
        raise SnapshotError("Empty snapshot archive")
    return result


class NativeSnapshotDevice:
    """Native ADB transport for the original AndroidWorld snapshot directory."""

    def __init__(self, controller, *, adb_path, serial, server_port=5037):
        from android_world.env import adb_utils, device_constants
        from android_world.utils import file_utils
        self.controller, self.adb_utils, self.file_utils = controller, adb_utils, file_utils
        self.directory = device_constants.SNAPSHOT_DATA
        self.adb = [str(adb_path), "-P", str(server_port), "-s", serial]

    def shell(self, args):
        self.adb_utils.check_ok(self.adb_utils.issue_generic_request(["shell"] + args, self.controller))

    def exists(self):
        return self.file_utils.check_directory_exists(self.directory, self.controller)

    def capture(self):
        return subprocess.check_output(self.adb + ["exec-out", "tar", "-C", self.directory, "-cf", "-", "."], timeout=90)

    def preserve(self, suffix):
        destination = self.directory + ".preserved_" + suffix
        self.shell(["mv", self.directory, destination])
        return destination

    def install(self, archive):
        remote = "/data/local/tmp/causalact_snapshot_" + uuid.uuid4().hex + ".tar"
        subprocess.run(self.adb + ["push", str(archive), remote], check=True,
                       capture_output=True, timeout=90)
        self.shell(["mkdir", "-p", self.directory])
        self.shell(["tar", "-xpf", remote, "-C", self.directory])


def restore_frozen_snapshots(device, *, archive, expected_sha256, audit_dir):
    """Restore exactly the frozen files, preserving any different prior snapshot.

    Real app data is left to the native task initializer. Only snapshot storage
    is restored here. A matching existing archive is verified without writing.
    All failures propagate; there are no automatic retries or score outputs.
    """
    audit = Path(audit_dir)
    audit.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = dict(scope="snapshot preparation only", valid=False, score=None,
                   benchmark_scores=0, model_requests=0, archive_sha256=expected_sha256)
    try:
        archive = Path(archive)
        raw = archive.read_bytes()
        if sha(raw) != expected_sha256:
            raise SnapshotError("Frozen archive SHA differs")
        expected = archive_manifest(raw)
        receipt.update(members=len(expected), existing=device.exists(), restored=False)
        if receipt["existing"]:
            before = device.capture()
            (audit / "before.tar").write_bytes(before)
            if archive_manifest(before) != expected:
                receipt["preserved_device_directory"] = device.preserve(uuid.uuid4().hex)
                device.install(archive)
                receipt["restored"] = True
        else:
            device.install(archive)
            receipt["restored"] = True
        observed = archive_manifest(device.capture())
        (audit / "observed_manifest.json").write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")
        if observed != expected:
            raise SnapshotError("Restored contents or UID/GID/mode/link metadata differ")
        receipt.update(valid=True, manifest_equal=True)
        return receipt
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (audit / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


@contextmanager
def frozen_native_task(task, env, *, device, archive, expected_sha256, audit_dir):
    """Restore, initialize and tear down the supplied native task without scoring.

    The caller freezes task identity/params/splits separately. Body and model
    exceptions propagate even if native teardown also fails. Partial native
    initialization still receives a cleanup attempt, whose error is retained.
    """
    audit = Path(audit_dir)
    snapshot = restore_frozen_snapshots(device, archive=archive,
        expected_sha256=expected_sha256, audit_dir=audit)
    receipt = dict(scope="native task lifecycle only; no success evaluation",
                   task_family=type(task).__name__, snapshot=snapshot,
                   valid=False, score=None, benchmark_scores=0, model_requests=0)
    failure = None
    class SnapshotWarnings(logging.Handler):
        def __init__(self):
            super().__init__();self.thread=threading.get_ident();self.seen=set();self.messages=[]
        def emit(self, record):
            if record.thread==self.thread and record.getMessage().startswith("Skipping app snapshot loading") and id(record) not in self.seen:
                self.seen.add(id(record));self.messages.append(record.getMessage())
    warnings = SnapshotWarnings()
    loggers = [logging.getLogger(), logging.getLogger("absl")]
    for logger in loggers:logger.addHandler(warnings)
    try:
        task.initialize_task(env)
        if warnings.messages:
            raise SnapshotError("Native initializer skipped app snapshot restoration")
        receipt["initialized"] = True
        yield receipt
        receipt["body_returned"] = True
    except BaseException as error:
        failure = error
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            previous_warnings = len(warnings.messages)
            task.tear_down(env)
            if len(warnings.messages)>previous_warnings:
                raise SnapshotError("Native teardown skipped app snapshot restoration")
            receipt["torn_down"] = True
            receipt["valid"] = failure is None
        except Exception as error:
            receipt.update(teardown_error_type=type(error).__name__, teardown_error=str(error))
            if failure is None:
                raise
        finally:
            for logger in loggers:logger.removeHandler(warnings)
            receipt["native_snapshot_warnings"] = warnings.messages
            (audit / "lifecycle.json").write_text(json.dumps(receipt, indent=2) + "\n")
