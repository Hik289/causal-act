import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import uitars_deferred_queue as deferred


class DeferredRepairTests(unittest.TestCase):
    def setup_protocol(self, root, allowance=1):
        protocol=root/'protocol.json'
        protocol.write_text(json.dumps({'start_after':{'protocol_file':'predecessor','protocol_sha256':'digest','run_id':'previous'},
            'frozen_files':{},'attempts_per_case':3,'technical_preflight_images':['original'],
            'invalid_backoff_seconds':120}))
        repair=root/'repair.json'
        repair.write_text(json.dumps({'base_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'source_dir':str(Path(deferred.__file__).parent),'environment_repair_only':True,
            'technical_preflight_attempts':allowance,'technical_preflight_images':['corrected'],
            'repair_id':'synthetic_fixture_geometry','source_files':{}}))
        return protocol,repair

    def test_repair_uses_corrected_fixture_once_and_preserves_original_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);protocol,repair=self.setup_protocol(root);before=protocol.read_bytes()
            original=[]
            for index in range(1,4):
                p=root/('technical_attempt'+str(index));p.mkdir();raw=p/'invalid.json';raw.write_text('{"valid":false,"score":null}')
                original.append((raw,raw.read_bytes()))
            with patch.object(deferred,'completed_method',return_value=True), patch.object(deferred,'active_evaluators',return_value=[]), \
                 patch.object(deferred.Queue,'resource_ready',return_value=True), patch.object(deferred.Queue,'__init__',return_value=None), \
                 patch.object(deferred.Queue,'run'), patch.object(deferred.signal,'signal'), \
                 patch.object(deferred,'technical_preflight',return_value={'valid_transport':True,'fixture':True,'benchmark_scores':0}) as check:
                deferred.run(protocol,repair)
                deferred.run(protocol,repair)
                self.assertEqual(check.call_count,1)
                self.assertEqual(check.call_args[0][0]['technical_preflight_images'],['corrected'])
            self.assertEqual(protocol.read_bytes(),before)
            self.assertTrue(all(p.read_bytes()==raw for p,raw in original))
            self.assertTrue((root/'synthetic_fixture_geometry/technical_attempt1/receipt.json').exists())

    def test_repair_cannot_reset_a_three_attempt_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            protocol,repair=self.setup_protocol(Path(folder),allowance=3)
            with patch.object(deferred,'technical_preflight') as check, self.assertRaisesRegex(RuntimeError,'allowance'):
                deferred.run(protocol,repair)
            check.assert_not_called()


if __name__=='__main__':
    unittest.main()
