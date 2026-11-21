import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class RelayTest(unittest.TestCase):
    def invoke(self, provider_response):
        with tempfile.TemporaryDirectory() as root:
            with patch.dict(os.environ, {"CAUSALACT_BEDROCK_MODEL": "test-model",
                     "AWS_BEARER_TOKEN_BEDROCK": "offline-fixture-only",
                     "OPENAI_API_KEY": "offline-client-only"}), \
                 patch.object(sys, "argv", ["bedrock_relay.py", root]), \
                 patch("http.server.ThreadingHTTPServer"):
                module = runpy.run_path(str(Path(__file__).with_name("bedrock_relay.py")))
            handler = object.__new__(module["Handler"])
            payload = json.dumps({"messages": [{"role": "user", "content": "fixture"}]}).encode()
            handler.path = "/worker/offline/v1/chat/completions"
            handler.headers = {"Authorization": "Bearer offline-client-only",
                               "Content-Length": str(len(payload))}
            handler.rfile = io.BytesIO(payload)
            handler.send_json = Mock()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(provider_response).encode())) as upstream:
                handler.do_POST()
            self.assertEqual(upstream.call_count, 1)
            records = [json.loads(line) for line in (Path(root)/"requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            return handler.send_json.call_args.args, records[0]

    def test_refusal_is_nonretryable_and_has_no_action_response(self):
        (status, payload), audit = self.invoke({"content": [{"type": "text", "text": "{}"}],
            "stop_reason": "refusal", "usage": {"input_tokens": 275, "output_tokens": 0}})
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "provider_refusal")
        self.assertFalse(payload["error"]["retryable"])
        self.assertNotIn("choices", payload)
        self.assertEqual(audit["stop_reason"], "refusal")
        self.assertEqual(audit["error_type"], "provider_refusal")

    def test_normal_response_and_usage_reach_native_parser_unchanged(self):
        text = '{"action_type":"status","goal_status":"complete"}'
        (status, payload), audit = self.invoke({"content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 7}})
        self.assertEqual(status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], text)
        self.assertEqual(payload["usage"]["total_tokens"], 19)
        self.assertEqual(audit["stop_reason"], "end_turn")


if __name__ == "__main__":
    unittest.main()
