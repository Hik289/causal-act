import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from native_debug_bev import without_debug_bev


SOURCE = '''
SAVE_PATH = None
def get_entry_point(): return 'Native'
class Native:
    def setup(self): self.save_path = None
    def sensors(self):
        return [dict(id='front', type='sensor.camera.rgb'),
            dict(type='sensor.camera.rgb', x=0., y=0., z=50., roll=0., pitch=-90.,
                 yaw=0., width=512, height=512, fov=50., id='bev')]
    def tick(self, input_data):
        bev = input_data['bev']
        result = {'front': input_data['front'], 'speed': input_data['speed'], 'bev': bev}
        return result
    def run_step(self, input_data):
        tick_data = self.tick(input_data)
        if SAVE_PATH is not None: self.save(tick_data)
        return tick_data['front'], tick_data['speed']
    def save(self, tick_data): return tick_data['bev']
'''


class DebugBEVTests(unittest.TestCase):
    def load(self, folder, source=SOURCE):
        path = Path(folder) / 'native.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('native_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_only_debug_input_removed_and_native_control_inherited(self):
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder)
            original = Path(module.__file__).read_bytes()
            cls = without_debug_bev(module, digest)
            agent = cls()
            agent.setup()
            frame = bytearray(b'unchanged camera')
            self.assertEqual(agent.run_step({'front': frame, 'speed': 7}), (frame, 7))
            self.assertIs(agent.tick({'front': frame, 'speed': 7})['front'], frame)
            self.assertIs(cls.run_step, module.Native.run_step)
            self.assertEqual(agent.sensors(), [dict(id='front', type='sensor.camera.rgb')])
            self.assertEqual(Path(module.__file__).read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                agent.save({})

    def test_rejects_model_consumption_of_camera(self):
        for expression in ["tick_data['bev']", "tick_data.get('bev')"]:
            source = SOURCE.replace("return tick_data['front'], tick_data['speed']", 'return ' + expression)
            with tempfile.TemporaryDirectory() as folder:
                module, digest = self.load(folder, source)
                with self.assertRaisesRegex(ValueError, 'consumed'):
                    without_debug_bev(module, digest)

    def test_rejects_unfrozen_source_and_enabled_logger(self):
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder)
            with self.assertRaisesRegex(ValueError, 'changed'):
                without_debug_bev(module, 'different')
            module.SAVE_PATH = '/enabled'
            with self.assertRaisesRegex(ValueError, 'disabled'):
                without_debug_bev(module, digest)

    def test_rejects_unguarded_logging_and_changed_sensor(self):
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder, SOURCE.replace('if SAVE_PATH is not None: self.save(tick_data)', 'self.save(tick_data)'))
            with self.assertRaisesRegex(ValueError, 'not disabled'):
                without_debug_bev(module, digest)
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder, SOURCE.replace('z=50.', 'z=1.'))
            agent = without_debug_bev(module, digest)()
            with self.assertRaisesRegex(ValueError, 'Unexpected logging camera'):
                agent.sensors()


if __name__ == '__main__':
    unittest.main()
