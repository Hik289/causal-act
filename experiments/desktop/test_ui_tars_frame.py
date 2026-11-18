import unittest
from ui_tars_frame import checked_size, model_to_screen


class FrameTests(unittest.TestCase):
    def test_processed_pixels_map_through_both_axes(self):
        self.assertEqual(model_to_screen([257, 92], [644, 364], [1920, 1080]), [766, 273])
        # The old 1000x1000 interpretation put this same point in the toolbar.
        self.assertEqual(model_to_screen([257, 92], [1000, 1000], [1920, 1080]), [493, 99])

    def test_origin_and_frame_edge_are_preserved(self):
        self.assertEqual(model_to_screen([0, 0], [644, 364], [1920, 1080]), [0, 0])
        self.assertEqual(model_to_screen([644, 364], [644, 364], [1920, 1080]), [1920, 1080])

    def test_portrait_frame_does_not_swap_axes(self):
        self.assertEqual(model_to_screen([182, 322], [364, 644], [1080, 1920]), [540, 960])

    def test_outside_prediction_is_not_clamped(self):
        self.assertEqual(model_to_screen([-10, 700], [644, 364], [1920, 1080]), [-30, 2077])

    def test_unknown_or_invalid_frame_fails_closed(self):
        for size in ([0, 364], [644], [644, -1], [644.5, 364]):
            with self.assertRaises(ValueError):
                checked_size(size)
        with self.assertRaises(ValueError):
            model_to_screen([float("nan"), 10], [644, 364], [1920, 1080])


if __name__ == "__main__":
    unittest.main()
