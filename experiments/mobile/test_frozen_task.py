"""Offline integrity/lifecycle checks, without Android or model requests."""
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile
import tempfile
import unittest

from frozen_task import archive_manifest, frozen_native_task, restore_frozen_snapshots, SnapshotError


def tar(data=b"original", *, mode=0o600, uid=10123, name="./app/file"):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo(name);member.size=len(data);member.mode=mode;member.uid=uid;member.gid=uid
        archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


class Device:
    def __init__(self, current=None, corrupt=False):
        self.current, self.corrupt = current, corrupt
        self.backups, self.installs = {}, 0
    def exists(self):return self.current is not None
    def capture(self):return self.current
    def preserve(self, suffix):
        name="snapshots.preserved_"+suffix;self.backups[name]=self.current;self.current=None;return name
    def install(self, source):
        self.installs+=1;self.current=tar(b"corrupt") if self.corrupt else Path(source).read_bytes()


class SnapshotTest(unittest.TestCase):
    def inputs(self, root):
        source=Path(root)/'frozen.tar';source.write_bytes(tar())
        return dict(archive=source,expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audit_dir=Path(root)/'audit')

    def test_missing_snapshots_restore_with_exact_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            device=Device();receipt=restore_frozen_snapshots(device,**self.inputs(root))
            self.assertTrue(receipt['manifest_equal']);self.assertEqual(device.installs,1)

    def test_identical_snapshot_is_verified_without_writing(self):
        with tempfile.TemporaryDirectory() as root:
            device=Device(tar());receipt=restore_frozen_snapshots(device,**self.inputs(root))
            self.assertFalse(receipt['restored']);self.assertEqual(device.installs,0)

    def test_different_snapshot_and_extra_files_are_preserved_not_overlaid(self):
        with tempfile.TemporaryDirectory() as root:
            before=tar(b"extra",name='./app/extra');device=Device(before)
            restore_frozen_snapshots(device,**self.inputs(root))
            self.assertEqual(list(device.backups.values()),[before]);self.assertNotIn('./app/extra',archive_manifest(device.current))
            self.assertEqual((Path(root)/'audit/before.tar').read_bytes(),before)

    def test_permission_or_owner_changes_require_restoration(self):
        for before in [tar(mode=0o777),tar(uid=0)]:
            with self.subTest(),tempfile.TemporaryDirectory() as root:
                device=Device(before);restore_frozen_snapshots(device,**self.inputs(root));self.assertEqual(device.installs,1)

    def test_wrong_source_digest_never_touches_device(self):
        with tempfile.TemporaryDirectory() as root:
            args=self.inputs(root);args['expected_sha256']='0'*64;device=Device()
            with self.assertRaises(SnapshotError):restore_frozen_snapshots(device,**args)
            self.assertEqual(device.installs,0)

    def test_corrupt_device_restore_fails_without_task_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            class Task:
                def initialize_task(self,env):calls.append('init')
                def tear_down(self,env):calls.append('down')
            with self.assertRaises(SnapshotError):
                with frozen_native_task(Task(),None,device=Device(corrupt=True),**self.inputs(root)):pass
            self.assertEqual(calls,[])
            self.assertIsNone(json.loads((Path(root)/'audit/receipt.json').read_text())['score'])

    def test_restore_precedes_native_initialization_and_teardown_follows_body(self):
        with tempfile.TemporaryDirectory() as root:
            args=self.inputs(root);device=Device();calls=[]
            class Task:
                def initialize_task(self,env):
                    self.device_ready=device.current==Path(args['archive']).read_bytes();calls.append('init')
                def tear_down(self,env):calls.append('down')
            task=Task()
            with frozen_native_task(task,None,device=device,**args):calls.append('body')
            self.assertTrue(task.device_ready);self.assertEqual(calls,['init','body','down'])

    def test_provider_failure_survives_a_simultaneous_teardown_failure(self):
        with tempfile.TemporaryDirectory() as root:
            class ProviderFailure(RuntimeError):pass
            class Task:
                def initialize_task(self,env):pass
                def tear_down(self,env):raise ValueError('cleanup issue')
            with self.assertRaisesRegex(ProviderFailure,'provider unavailable'):
                with frozen_native_task(Task(),None,device=Device(),**self.inputs(root)):raise ProviderFailure('provider unavailable')
            receipt=json.loads((Path(root)/'audit/lifecycle.json').read_text())
            self.assertFalse(receipt['valid']);self.assertEqual(receipt['teardown_error_type'],'ValueError')

    def test_partial_native_initialization_receives_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            class Task:
                def initialize_task(self,env):raise ValueError('partial init')
                def tear_down(self,env):calls.append('down')
            with self.assertRaisesRegex(ValueError,'partial init'):
                with frozen_native_task(Task(),None,device=Device(),**self.inputs(root)):pass
            self.assertEqual(calls,['down'])

    def test_escaping_archive_paths_are_rejected(self):
        with self.assertRaises(SnapshotError):archive_manifest(tar(name='../outside'))

    def test_native_swallowed_snapshot_warning_stops_before_task_body(self):
        with tempfile.TemporaryDirectory() as root:
            calls=[]
            class Task:
                def initialize_task(self,env):logging.warning('Skipping app snapshot loading : missing snapshot')
                def tear_down(self,env):calls.append('down')
            with self.assertRaisesRegex(SnapshotError,'initializer skipped'):
                with frozen_native_task(Task(),None,device=Device(),**self.inputs(root)):calls.append('body')
            self.assertEqual(calls,['down'])
            receipt=json.loads((Path(root)/'audit/lifecycle.json').read_text())
            self.assertFalse(receipt['valid']);self.assertEqual(len(receipt['native_snapshot_warnings']),1)


if __name__=='__main__':unittest.main()
