import json
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from s2_repair_runtime import (
    InvalidDagResponse,
    bounded_dag_generator,
    checked_dag_parser,
    prepare_expect_dependency,
    wait_for_desktop,
)


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


if __name__ == "__main__":
    unittest.main()
