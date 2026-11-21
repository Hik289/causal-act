import unittest

from bedrock_response import completion_text


class ResponseTest(unittest.TestCase):
    def test_empty_provider_refusal_remains_distinct(self):
        self.assertEqual(completion_text({"content": [], "stop_reason": "refusal"}),
                         ("", "provider_refusal"))

    def test_text_from_refusal_cannot_be_executed(self):
        self.assertEqual(completion_text({"content": [{"type": "text", "text": "{}"}],
                                          "stop_reason": "refusal"}), ("", "provider_refusal"))

    def test_ordinary_empty_completion_retains_its_classification(self):
        self.assertEqual(completion_text({"content": [], "stop_reason": "end_turn"}),
                         ("", "empty_completion"))

    def test_normal_text_blocks_are_unchanged(self):
        self.assertEqual(completion_text({"content": [{"type": "text", "text": "one"},
            {"type": "thinking", "thinking": "private"}, {"type": "text", "text": "two"}],
            "stop_reason": "end_turn"}), ("onetwo", None))

    def test_partial_output_at_token_limit_remains_native_output(self):
        self.assertEqual(completion_text({"content": [{"type": "text", "text": "{"}],
                                          "stop_reason": "max_tokens"}), ("{", None))


if __name__ == "__main__":
    unittest.main()
