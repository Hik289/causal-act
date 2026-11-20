import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from thinktwice_no_debug import without_recording_camera


SOURCE = '''
SAVE_PATH = None
def get_entry_point(): return 'Native'
class Native:
    def setup(self):
        self.save_path = None
        self.model = object()
        self.history = []
        if SAVE_PATH is not None: string = 'recording'
        self.folder_name = string
    def sensors(self):
        return [dict(id='front', type='sensor.camera.rgb'),
            dict(id='lidar', type='sensor.lidar.ray_cast'),
            dict(type='sensor.camera.rgb', x=0., y=0., z=50., roll=0., pitch=-90.,
                 yaw=0., width=512, height=512, fov=50., id='topdown')]
    def tick(self, input_data):
        topdown = input_data['topdown']
        self.history.append(input_data['lidar'])
        return {'front': input_data['front'], 'lidar': input_data['lidar'],
            'history': tuple(self.history), 'topdown': topdown}
    def run_step(self, input_data):
        tick_data = self.tick(input_data)
        if SAVE_PATH is not None: self.save(tick_data)
        return tick_data['front'], tick_data['lidar'], tick_data['history']
    def save(self, tick_data): return tick_data['topdown']
'''


class RecordingTests(unittest.TestCase):
    def load(self, folder, source=SOURCE):
        path = Path(folder)/'native.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('thinktwice_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_no_recording_setup_and_temporal_policy_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder)
            with self.assertRaises(UnboundLocalError):
                module.Native().setup()
            cls = without_recording_camera(module, digest)
            agent = cls()
            agent.setup()
            self.assertIsNone(agent.folder_name)
            self.assertIs(cls.run_step, module.Native.run_step)
            frame, points = bytearray(b'RGB'), bytearray(b'LiDAR')
            for index in range(3):
                result = agent.run_step({'front': frame, 'lidar': points})
                self.assertIs(result[0], frame)
                self.assertIs(result[1], points)
                self.assertEqual(len(result[2]), index+1)
            self.assertEqual(agent.sensors(), module.Native.sensors(agent)[:-1])
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                agent.save({})

    def test_refuses_policy_topdown_consumption_and_unguarded_logger(self):
        changes = [SOURCE.replace("return tick_data['front'], tick_data['lidar'], tick_data['history']", "return tick_data.get('topdown')"),
                   SOURCE.replace('if SAVE_PATH is not None: self.save(tick_data)', 'self.save(tick_data)')]
        for source in changes:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                module, digest = self.load(folder, source)
                with self.assertRaises(ValueError):
                    without_recording_camera(module, digest)

    def test_refuses_folder_consumption_and_extra_debug_dependencies(self):
        changes = [SOURCE.replace("return tick_data['front'], tick_data['lidar'], tick_data['history']", 'return self.folder_name'),
                   SOURCE.replace("self.history.append(input_data['lidar'])", "self.history.append(topdown)"),
                   SOURCE.replace("self.folder_name = string", "self.folder_name = 'unexpected'")]
        for source in changes:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                module, digest = self.load(folder, source)
                with self.assertRaises(ValueError):
                    without_recording_camera(module, digest)

    def test_refuses_changed_source_camera_and_enabled_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder)
            with self.assertRaisesRegex(ValueError, 'changed'):
                without_recording_camera(module, 'wrong')
            module.SAVE_PATH = '/enabled'
            with self.assertRaisesRegex(ValueError, 'disabled'):
                without_recording_camera(module, digest)
            module.SAVE_PATH = None
            cls = without_recording_camera(module, digest)
            module.SAVE_PATH = '/enabled'
            with self.assertRaisesRegex(RuntimeError, 'enabled'):
                cls().setup()
        with tempfile.TemporaryDirectory() as folder:
            module, digest = self.load(folder, SOURCE.replace('z=50.', 'z=2.'))
            with self.assertRaisesRegex(ValueError, 'camera'):
                without_recording_camera(module, digest)().sensors()


if __name__ == '__main__':
    unittest.main()
