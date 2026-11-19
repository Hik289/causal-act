import base64
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
import urllib.error
from unittest.mock import patch

from seeact_adapter import MODEL, ModelUnavailable, PacedClaude, normalize_image_mime, validate_task


class Response:
    status = 200

    def __init__(self, model=MODEL, content='ELEMENT: A\nACTION: CLICK\nVALUE: None'):
        self.value = {'model': model, 'choices': [{'message': {'content': content}}]}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return json.dumps(self.value).encode()


class SeeActTransportTests(unittest.TestCase):
    def test_reference_images_cannot_be_silently_dropped(self):
        for image in ['https://example.com/reference.png', ['local-reference.png']]:
            with self.assertRaisesRegex(ValueError, 'keep this case pending'):
                validate_task({'image': image})
        validate_task({'image': None})

    def test_next_case_rebinds_the_cached_native_engine_to_its_own_audit(self):
        cached = types.ModuleType('seeact.demo_utils.inference_engine')
        with patch.dict(sys.modules, {'seeact.demo_utils.inference_engine': cached}):
            first = PacedClaude('http://fixture', '/unused', '/unused')
            second = PacedClaude('http://fixture', '/unused', '/unused')
            first.install_native_transport()
            self.assertIs(cached.litellm.completion.__self__, first)
            second.install_native_transport()
            self.assertIs(cached.litellm.completion.__self__, second)

    def test_recovered_http_error_does_not_invalidate_a_later_scored_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'credential').write_text('fixture-only')
            bridge = PacedClaude('http://fixture', p / 'credential', p / 'audit.jsonl')
            failure = urllib.error.HTTPError('http://fixture', 503, 'unavailable', {}, None)
            with patch('urllib.request.urlopen', side_effect=[failure, Response()]):
                with self.assertRaises(ModelUnavailable):
                    bridge.completion(model=MODEL, messages=[{'role': 'user', 'content': 'same operation'}])
                with self.assertRaises(ModelUnavailable):
                    bridge.validate_completed_calls()
                bridge.completion(model=MODEL, messages=[{'role': 'user', 'content': 'same operation'}])
            bridge.validate_completed_calls()

    def test_wrong_mime_is_corrected_without_changing_any_image_byte(self):
        raw = b'\x89PNG\r\n\x1a\n' + b'fixture'
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}}]}]
        normalized, audit = normalize_image_mime(messages)
        url = normalized[0]['content'][0]['image_url']['url']
        self.assertTrue(url.startswith('data:image/png;'))
        self.assertEqual(base64.b64decode(url.split(',', 1)[1]), raw)
        self.assertTrue(messages[0]['content'][0]['image_url']['url'].startswith('data:image/jpeg;'))
        self.assertEqual(audit[0]['bytes'], len(raw))

    def test_non_inline_image_is_not_silently_sent_or_scored(self):
        with self.assertRaises(ModelUnavailable):
            normalize_image_mime([{'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.com/a'}}]}])

    def test_unexpected_or_empty_served_model_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'credential').write_text('fixture-only')
            for response in [Response(model='wrong-model'), Response(content='')]:
                bridge = PacedClaude('http://fixture', p / 'credential', p / 'audit.jsonl')
                with patch('urllib.request.urlopen', return_value=response), self.assertRaises(ModelUnavailable):
                    bridge.completion(model='gpt-4o', messages=[{'role': 'user', 'content': 'fixture'}])
                self.assertIn('error', bridge.calls[-1])

    def test_valid_reply_preserves_native_completion_shape_and_requested_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / 'credential').write_text('fixture-only')
            bridge = PacedClaude('http://fixture', p / 'credential', p / 'audit.jsonl')
            with patch('urllib.request.urlopen', return_value=Response()):
                result = bridge.completion(model='gpt-4o', messages=[{'role': 'user', 'content': 'fixture'}])
            self.assertIn('ACTION: CLICK', result.choices[0]['message']['content'])
            self.assertEqual(bridge.calls[0]['transport_alias'], 'gpt-4o')
            self.assertEqual(bridge.calls[0]['served_model'], MODEL)
            self.assertNotIn('fixture-only', (p / 'audit.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
