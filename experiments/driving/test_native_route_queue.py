import json
from pathlib import Path
import tempfile
import unittest

import native_route_queue
from native_route_queue import (mapped_vulkan_index, valid_record, mesa_pci_selector,
    isolated_simulator_environment, isolated_vulkan_index, device_namespace_command, runtime_repair, sha)


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

    def test_pci_identity_normalizes_nvidia_domain_and_rejects_bad_input(self):
        self.assertEqual(mesa_pci_selector('00000000:4B:00.0'), 'pci-0000_4b_00_0')
        self.assertEqual(mesa_pci_selector('0001:4B:00.7'), 'pci-0001_4b_00_7')
        for value in ['00010000:4B:00.0', '0000:4B:00.8', '0000:4B:00.0!', 'pci-0000_4b_00_0']:
            with self.assertRaises(RuntimeError):
                mesa_pci_selector(value)

    def test_simulator_filter_matches_uuid_without_changing_policy_environment(self):
        env = dict(CUDA_VISIBLE_DEVICES='GPU-0011', NODEVICE_SELECT='1', OMP_NUM_THREADS='2')
        rows = 'GPU-9988, 00000000:31:00.0\nGPU-0011, 00000000:4B:00.0\n'
        result = isolated_simulator_environment('GPU-0011', env, rows)
        self.assertEqual(result['DRI_PRIME'], 'pci-0000_4b_00_0')
        self.assertEqual(result['MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'], '1')
        self.assertNotIn('NODEVICE_SELECT', result)
        self.assertEqual(env, dict(CUDA_VISIBLE_DEVICES='GPU-0011', NODEVICE_SELECT='1', OMP_NUM_THREADS='2'))
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], env['CUDA_VISIBLE_DEVICES'])
        for invalid in [rows + rows, 'GPU-other, 00000000:4B:00.0\n']:
            with self.assertRaises(RuntimeError):
                isolated_simulator_environment('GPU-0011', env, invalid)

    def test_filtered_enumeration_must_expose_only_the_selected_gpu(self):
        selected = dict(vulkan_index=0, vendor_id=0x10de, device_uuid='0011')
        self.assertEqual(isolated_vulkan_index([selected], 'GPU-0011'), 0)
        for devices in [[selected, dict(vulkan_index=1, vendor_id=0x10de, device_uuid='9988')],
                        [dict(selected, vulkan_index=1)],
                        [dict(selected, device_uuid='9988')]]:
            with self.assertRaises(RuntimeError):
                isolated_vulkan_index(devices, 'GPU-0011')

    def test_namespace_hides_other_device_nodes_and_preserves_simulator_arguments(self):
        nodes = ['/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia1']
        drm = ['/dev/dri/card2', '/dev/dri/renderD129']
        command = ['CarlaUE4.sh', '-RenderOffScreen', '-graphicsadapter=0']
        result = device_namespace_command(command, nodes, drm)
        self.assertEqual(result[:9], ['bwrap', '--bind', '/', '/', '--proc', '/proc', '--dev', '/dev', '--dev-bind'])
        self.assertEqual(result[-4:], ['--'] + command)
        self.assertNotIn('/dev/nvidia0', result)
        self.assertNotIn('/dev/dri/card1', result)
        self.assertEqual(nodes, ['/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia1'])
        for bad_nodes, bad_drm in [(nodes + ['/dev/nvidia0'], drm),
                                   (nodes + ['/dev/nvidia-caps'], drm),
                                   (nodes, []), (nodes, ['/dev/dri/../nvidia0'])]:
            with self.assertRaises(RuntimeError):
                device_namespace_command(command, bad_nodes, bad_drm)

    def test_bound_repair_preserves_budget_and_invalid_attempt_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, repair, invalid = root/'protocol.json', root/'manifest.json', root/'invalid.json'
            protocol.write_text('{"attempts_per_route":3}')
            invalid.write_text('{"valid":false,"score":null,"error":"DeviceIdentityError"}')
            source = str(Path(native_route_queue.__file__).resolve())
            value = dict(repair_id='physical_gpu_isolation', original_protocol_sha256=sha(protocol),
                environment_repair_only=True, GPU_isolation='nvidia_device_namespace_and_mesa_pci',
                additional_route_attempts=[], source_files={source:sha(source)},
                protected_files={str(invalid):sha(invalid)})
            repair.write_text(json.dumps(value))
            before = protocol.read_bytes(), invalid.read_bytes()
            accepted = runtime_repair(protocol, repair)
            self.assertEqual(accepted['manifest_sha256'], sha(repair))
            self.assertEqual((protocol.read_bytes(), invalid.read_bytes()), before)
            for patch in [dict(additional_route_attempts=['vad/22535']),
                          dict(original_protocol_sha256='wrong'), dict(environment_repair_only=False),
                          dict(source_files={}), dict(protected_files={str(invalid):'wrong'})]:
                repair.write_text(json.dumps(dict(value, **patch)))
                with self.assertRaises(RuntimeError):
                    runtime_repair(protocol, repair)

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
