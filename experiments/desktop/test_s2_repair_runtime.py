import json
import base64
from io import BytesIO
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from s2_repair_runtime import (
    InvalidDagResponse,
    bounded_dag_generator,
    checked_dag_parser,
    prepare_expect_dependency,
    prepare_cssselect_dependency,
    screenshot_image_bytes,
    wait_for_desktop,
    cursor_crop_fingerprint,
    initial_frame_problem,
    guarded_initial_reset,
    UnrenderedDesktop,
)


class ScreenshotInputTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        buffer = BytesIO()
        Image.new("RGB", (17, 11), (12, 34, 56)).save(buffer, format="PNG")
        self.png = buffer.getvalue()

    def test_native_bytes_remain_readable_by_ocr_without_reencoding(self):
        from PIL import Image
        result = screenshot_image_bytes(self.png)
        self.assertEqual(result, self.png)
        self.assertEqual(Image.open(BytesIO(result)).getpixel((0, 0)), (12, 34, 56))

    def test_old_base64_input_keeps_identical_model_image_payload(self):
        encoded = base64.b64encode(self.png).decode("ascii")
        result = screenshot_image_bytes(encoded)
        self.assertEqual(result, self.png)
        self.assertEqual(base64.b64encode(result).decode("ascii"), encoded)

    def test_array_preserves_dimensions_and_pixel_values(self):
        import numpy as np
        from PIL import Image
        array = np.array(Image.open(BytesIO(self.png)))
        result = Image.open(BytesIO(screenshot_image_bytes(array)))
        self.assertEqual(result.size, (17, 11))
        self.assertTrue(np.array_equal(np.array(result), array))

    def test_data_url_and_buffer_inputs_preserve_original_bytes(self):
        encoded = base64.b64encode(self.png).decode("ascii")
        for value in ("data:image/png;base64," + encoded, bytearray(self.png), memoryview(self.png)):
            self.assertEqual(screenshot_image_bytes(value), self.png)

    def test_malformed_base64_is_not_replaced_with_a_placeholder(self):
        with self.assertRaises(ValueError):
            screenshot_image_bytes("not valid image data!")


class DagRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.retained = []

        def parser(text):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

        self.parse = checked_dag_parser(parser, self.retained.append)

    def test_truncated_reply_is_retained_then_valid_reply_is_used_unchanged(self):
        replies = iter(['{"dag": {"nodes": [', '{"dag": {"nodes": [], "edges": []}}'])
        generate = bounded_dag_generator(lambda: self.parse(next(replies)))
        self.assertEqual(generate(), {"dag": {"nodes": [], "edges": []}})
        self.assertEqual(self.retained, ['{"dag": {"nodes": ['])

    def test_exhausted_format_retries_raise_without_synthesizing_a_plan(self):
        calls = []

        def generate():
            calls.append(1)
            return self.parse('{"dag":')

        with self.assertRaises(InvalidDagResponse):
            bounded_dag_generator(generate)()
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.retained, ['{"dag":'] * 3)

    def test_unrelated_failure_is_not_retried(self):
        calls = []

        def generate():
            calls.append(1)
            raise OSError("VM disconnected")

        with self.assertRaises(OSError):
            bounded_dag_generator(generate)()
        self.assertEqual(len(calls), 1)

    def test_http_service_without_window_manager_does_not_pass_readiness(self):
        replies = [
            {"returncode": 1, "output": "", "error": "Cannot get window manager info"},
            {"returncode": 0, "output": "Name: GNOME Shell\n"},
        ]
        with patch("s2_repair_runtime.guest_execute", side_effect=replies) as execute:
            with patch("s2_repair_runtime.time.sleep"):
                wait_for_desktop(object())
        self.assertEqual(execute.call_count, 2)

    def test_missing_offline_packages_fail_before_installing(self):
        with TemporaryDirectory() as directory:
            with patch("s2_repair_runtime.guest_execute", return_value={"returncode": 1}):
                with self.assertRaises(RuntimeError):
                    prepare_expect_dependency(object(), directory)


class EvaluatorWheelTests(unittest.TestCase):
    def test_failed_install_is_unscored_even_when_http_request_succeeded(self):
        with patch("s2_repair_runtime.guest_execute", return_value={"returncode": 1}):
            with self.assertRaisesRegex(RuntimeError, "task remains unscored"):
                prepare_cssselect_dependency(
                    object(), ["pip", "install", "/home/user/cssselect-1.3.0-py3-none-any.whl"]
                )

    def test_successful_install_with_wrong_evaluator_version_is_rejected(self):
        results = [{"returncode": 0}, {"returncode": 0, "output": "1.2.0\n"}]
        with patch("s2_repair_runtime.guest_execute", side_effect=results):
            with self.assertRaisesRegex(RuntimeError, "version check failed"):
                prepare_cssselect_dependency(
                    object(), ["pip", "install", "/home/user/cssselect-1.3.0-py3-none-any.whl"]
                )


class InitialFrameTests(unittest.TestCase):
    def png(self, image):
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def test_all_black_is_rejected_before_a_visible_desktop_run(self):
        from PIL import Image
        self.assertIsNotNone(initial_frame_problem(self.png(Image.new("RGB", (80, 60)))))

    def test_known_cursor_is_translation_invariant(self):
        from PIL import Image
        first = Image.new("RGB", (80, 60))
        first.paste((255, 255, 255), (2, 3, 5, 9))
        known = cursor_crop_fingerprint(self.png(first))
        second = Image.new("RGB", (80, 60))
        second.paste((255, 255, 255), (60, 40, 63, 46))
        self.assertIsNotNone(initial_frame_problem(self.png(second), [known]))

    def test_dark_ui_is_preserved_and_is_not_a_cursor_only_frame(self):
        from PIL import Image
        cursor = Image.new("RGB", (80, 60))
        cursor.paste((255, 255, 255), (2, 3, 5, 9))
        known = cursor_crop_fingerprint(self.png(cursor))
        cursor.paste((30, 30, 30), (20, 20, 50, 40))
        self.assertIsNone(initial_frame_problem(self.png(cursor), [known]))

    def test_initial_guard_retains_exact_frame_and_raises_without_replacing_it(self):
        from PIL import Image
        png = self.png(Image.new("RGB", (80, 60)))
        class Environment:
            task_id = "diagnostic"
        with TemporaryDirectory() as folder:
            reset = guarded_initial_reset(lambda env: {"screenshot": png}, folder)
            with self.assertRaises(UnrenderedDesktop):
                reset(Environment())
            from pathlib import Path
            self.assertEqual(next(Path(folder).glob("*.png")).read_bytes(), png)


if __name__ == "__main__":
    unittest.main()
