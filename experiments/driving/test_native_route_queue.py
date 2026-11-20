import json
from pathlib import Path
import tempfile
import unittest

from native_route_queue import mapped_vulkan_index, valid_record


class NativeRouteQueueTests(unittest.TestCase):
    def test_vulkan_order_is_matched_by_uuid_not_nvidia_index(self):
        devices = [dict(vulkan_index=0, vendor_id=0x10de, device_uuid='00112233445566778899aabbccddeeff'),
                   dict(vulkan_index=3, vendor_id=0x10de, device_uuid='ffeeddccbbaa99887766554433221100')]
        self.assertEqual(mapped_vulkan_index(devices, 'GPU-ffeeddcc-bbaa-9988-7766-554433221100'), 3)
        with self.assertRaises(RuntimeError):
            mapped_vulkan_index(devices, 'GPU-unknown')
        with self.assertRaises(RuntimeError):
            mapped_vulkan_index(devices + [devices[1]], 'GPU-ffeeddcc-bbaa-9988-7766-554433221100')

    def test_software_vendor_cannot_match_a_physical_gpu(self):
        with self.assertRaises(RuntimeError):
            mapped_vulkan_index([dict(vulkan_index=0, vendor_id=0x10005, device_uuid='1234')], 'GPU-1234')

    def fixture(self, root, status='Failed - TickRuntime', score=0.):
        raw, log = root/'results.json', root/'evaluation.log'
        record = dict(route_id='RouteScenario_1_rep0', status=status,
            scores={'score_composed':score}, infractions={'red_light':[],'collisions_vehicle':[]})
        raw.write_text(json.dumps({'entry_status':'Finished','_checkpoint':{'records':[record]}}))
        log.write_text('Game time = 200.000\nStopping the route\nRegistering the route statistics\n')
        return raw, log

    def test_original_scored_negative_is_valid_and_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, log = self.fixture(Path(directory))
            before = raw.read_bytes(), log.read_bytes()
            result = valid_record(raw, log, 'RouteScenario_1_rep0')
            self.assertEqual(result['scores']['score_composed'], 0.)
            self.assertEqual((raw.read_bytes(),log.read_bytes()),before)

    def test_environment_zero_and_wrong_route_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for status in ['Failed - Agent couldn\'t be set up', 'Failed - Simulation crashed']:
                raw, log=self.fixture(root,status=status)
                self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))
            raw, log=self.fixture(root)
            self.assertIsNone(valid_record(raw,log,'RouteScenario_2_rep0'))
            log.write_text('Traceback (most recent call last):\nCUDA kernel failed\n')
            self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))

    def test_rejected_and_nonfinite_and_multiple_records_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw,log=self.fixture(root,score=float('nan'))
            self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))
            raw,log=self.fixture(root)
            data=json.loads(raw.read_text());data['entry_status']='Rejected';raw.write_text(json.dumps(data))
            self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))
            data['entry_status']='Crashed';raw.write_text(json.dumps(data))
            self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))
            data['entry_status']='Finished';data['_checkpoint']['records']*=2;raw.write_text(json.dumps(data))
            self.assertIsNone(valid_record(raw,log,'RouteScenario_1_rep0'))


if __name__=='__main__':
    unittest.main()
