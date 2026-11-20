from pathlib import Path
import tempfile
import unittest

from uitars_contract import screen_box, thumbnail_cap


class UITarsContractTests(unittest.TestCase):
    def test_out_of_frame_predictions_are_preserved(self):
        self.assertEqual(screen_box('[1.2, .5, 1.2, .5]', [1280, 720]), [1536., 360.])

    def test_normalized_box_midpoint_maps_without_code_execution(self):
        self.assertEqual(screen_box('[.1, .2, .3, .4]', [1280, 720]), [256., 216.])
        with self.assertRaises(ValueError):
            screen_box("__import__('os').getcwd()", [1280, 720])

    def test_malformed_geometry_is_rejected(self):
        for value, size in [('[0,0,0]', [1280, 720]), ([0, 0, float('nan'), 1], [1280, 720]),
                            ([0, 0, 1, 1], [0, 720])]:
            with self.assertRaises(ValueError):
                screen_box(value, size)

    def test_actual_service_cap_is_checked_instead_of_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'service.py'
            path.write_text('image.thumbnail((640,640))')
            self.assertEqual(thumbnail_cap(path), 640)
            path.write_text('image.thumbnail((640,640))\nother.thumbnail((1280,1280))')
            with self.assertRaises(RuntimeError):
                thumbnail_cap(path)


if __name__ == '__main__':
    unittest.main()
