"""Offline wire/error tests; no provider, Android device or task is contacted."""
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from claude_m3a import InvalidModelCall, MODEL, PacedClaudeWrapper


def completion(text="Reason: fixture\nAction: {\"action_type\":\"wait\"}", **extra):
    return dict(model=MODEL, choices=[dict(message=dict(content=text), finish_reason="stop")], **extra)


@contextmanager
def fixture(plan):
    records = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            records.append(dict(path=self.path, body=body, authorization=self.headers.get("Authorization")))
            status, data, headers = plan.pop(0)
            raw = json.dumps(data).encode()
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/worker/android_fixture/v1/chat/completions", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Image:
    shape = (2, 3, 3)

    def __init__(self, pixels):
        self.pixels = pixels

    def tobytes(self):
        return self.pixels


def wire_encoder(image):
    # Stand-in encoder for wire identity tests, not a native JPEG/vision fixture.
    return b"\xff\xd8" + image.pixels + b"\xff\xd9"


class ClaudeTransportTest(unittest.TestCase):
    def wrapper(self, endpoint, directory, **options):
        return PacedClaudeWrapper(endpoint=endpoint, credential="fixture-private-token", audit_dir=directory,
                                  encode_image=wire_encoder, timeout=2, **options)

    def test_preserves_prompt_order_and_exact_encoder_bytes_without_resizing(self):
        with tempfile.TemporaryDirectory() as directory, fixture([(200, completion(), {})]) as (endpoint, records):
            images = [Image(b"first"), Image(b"second")]
            text, safe, raw = self.wrapper(endpoint, directory).predict_mm("native prompt 汉字", images)
            payload = json.loads(records[0]["body"])
            self.assertEqual(payload["max_tokens"], 1000)
            self.assertEqual(payload["model"], MODEL)
            self.assertNotIn("temperature", payload)
            blocks = payload["messages"][0]["content"]
            self.assertEqual(blocks[0]["text"], "native prompt 汉字")
            for block, image in zip(blocks[1:], images):
                self.assertEqual(base64.b64decode(block["image_url"]["url"].split(",")[1]), wire_encoder(image))
            self.assertIsNone(safe)
            self.assertEqual(raw["model"], MODEL)
            self.assertTrue(text)
            self.assertTrue(all("fixture-private-token" not in p.read_text() for p in Path(directory).rglob("*.json*")))
            receipt = json.loads(next(Path(directory).glob("*/input_receipt.json")).read_text())
            self.assertEqual([v["shape"] for v in receipt["images"]], [[2, 3, 3], [2, 3, 3]])

    def test_provider_failure_budget_and_declared_backoff_are_exact(self):
        with tempfile.TemporaryDirectory() as directory, fixture([(503, {}, {})] * 3) as (endpoint, records):
            sleeps = []
            wrapper = self.wrapper(endpoint, directory, max_attempts=3, retry_delays=(120, 240), sleep=sleeps.append)
            with self.assertRaises(InvalidModelCall) as caught:
                wrapper.predict("summary phase")
            self.assertEqual(caught.exception.code, "http_503")
            self.assertEqual(caught.exception.attempts, 3)
            self.assertEqual(sleeps, [120, 240])
            self.assertEqual(len(records), 3)
            self.assertEqual(len({v["body"] for v in records}), 1)
            events = [json.loads(s) for s in (Path(directory) / "calls.jsonl").read_text().splitlines()]
            self.assertTrue(all(v["score"] is None and not v["valid"] for v in events))

    def test_http_auth_failure_is_terminal_without_extra_attempts(self):
        with tempfile.TemporaryDirectory() as directory, fixture([(401, {}, {})]) as (endpoint, records):
            with self.assertRaises(InvalidModelCall) as caught:
                self.wrapper(endpoint, directory, max_attempts=3, retry_delays=(1, 2)).predict("prompt")
            self.assertEqual(caught.exception.code, "http_401")
            self.assertEqual(len(records), 1)

    def test_service_retry_after_is_not_shortened(self):
        with tempfile.TemporaryDirectory() as directory, fixture([(529, {}, {"Retry-After": "180"}), (200, completion(), {})]) as (endpoint, records):
            sleeps = []
            self.wrapper(endpoint, directory, max_attempts=2, retry_delays=(120,), sleep=sleeps.append).predict("prompt")
            self.assertEqual(sleeps, [180])
            self.assertEqual(len(records), 2)

    def test_successful_response_must_identify_claude(self):
        response = completion()
        response["model"] = "gpt-placeholder"
        with tempfile.TemporaryDirectory() as directory, fixture([(200, response, {})]) as (endpoint, records):
            with self.assertRaises(InvalidModelCall) as caught:
                self.wrapper(endpoint, directory).predict("prompt")
            self.assertEqual(caught.exception.code, "model_mismatch")
            self.assertEqual(len(records), 1)

    def test_empty_or_malformed_provider_response_never_becomes_native_error_text(self):
        for response, code in [(completion(" "), "empty_completion"),
                               (dict(model=MODEL, choices=[]), "invalid_response_schema"),
                               (dict(model=MODEL, choices=[dict(message=dict(content=None))]), "empty_completion")]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory, fixture([(200, response, {})]) as (endpoint, records):
                with self.assertRaises(InvalidModelCall) as caught:
                    self.wrapper(endpoint, directory).predict("summary")
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(len(records), 1)

    def test_nonempty_model_syntax_error_and_length_output_remain_native_responsibility(self):
        response = completion("not a parseable action")
        response["choices"][0]["finish_reason"] = "length"
        with tempfile.TemporaryDirectory() as directory, fixture([(200, response, {})]) as (endpoint, records):
            self.assertEqual(self.wrapper(endpoint, directory).predict("prompt")[0], "not a parseable action")

    def test_redirect_does_not_forward_auth_or_bypass_the_paced_route(self):
        with tempfile.TemporaryDirectory() as directory, fixture([(303, {}, {"Location": "http://127.0.0.1:1/leak"})]) as (endpoint, records):
            with self.assertRaises(InvalidModelCall) as caught:
                self.wrapper(endpoint, directory).predict("prompt")
            self.assertEqual(caught.exception.code, "http_303")
            self.assertEqual(len(records), 1)

    def test_explicit_local_worker_endpoint_and_retry_schedule_are_required(self):
        for endpoint in ["https://api.openai.com/v1/chat/completions", "http://localhost:18822/v1/chat/completions",
                         "http://127.0.0.1:18822/v1/chat/completions", "http://token@127.0.0.1:18822/worker/android_a/v1/chat/completions"]:
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.wrapper(endpoint, directory)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.wrapper("http://127.0.0.1:18822/worker/android_a/v1/chat/completions", directory,
                             max_attempts=3, retry_delays=())

    def test_invalid_native_image_is_rejected_before_any_request(self):
        with tempfile.TemporaryDirectory() as directory, fixture([]) as (endpoint, records):
            wrapper = self.wrapper(endpoint, directory)
            wrapper.encode_image = lambda image: b"not jpeg"
            with self.assertRaises(InvalidModelCall) as caught:
                wrapper.predict_mm("prompt", [Image(b"pixels")])
            self.assertEqual(caught.exception.attempts, 0)
            self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
