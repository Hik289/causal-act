import base64
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
import urllib.error
from unittest.mock import patch

from seeact_adapter import (MODEL, ModelUnavailable, PacedClaude, normalize_image_mime,
                            load_task_references, attach_reference_images)
from seeact_adapter import verify_reference_delivery


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
    def test_every_planning_request_keeps_reference_inputs_but_evaluator_is_separate(self):
        calls = [{'stage': 'planning', 'images': [{'sha256': 'reference'}, {'sha256': 'screenshot'}]},
                 {'stage': 'evaluation', 'images': []}]
        verify_reference_delivery(calls, ['reference'])
        with self.assertRaises(ModelUnavailable):
            verify_reference_delivery(calls + [{'stage': 'planning', 'images': []}], ['reference'])
        with self.assertRaises(ModelUnavailable):
            verify_reference_delivery([calls[1]], [])

    def test_all_task_reference_images_are_loaded_and_hashed(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            Image.new('RGB', (12, 8), 'blue').save(p / 'reference.png')
            raw = (p / 'reference.png').read_bytes()
            loaded, audit = load_task_references({'image': ['reference.png', 'reference.png']}, p / 'task.json', p)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit[0]['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(audit[0]['encoded_sha256'], audit[0]['source_sha256'])
            self.assertEqual((audit[0]['width'], audit[0]['height']), (12, 8))
            with self.assertRaises(OSError):
                load_task_references({'image': 'missing.png'}, p / 'task.json', p)

    def test_reference_inputs_keep_native_prompt_and_screenshot_and_skip_evaluator_calls(self):
        raw = b'\x89PNG\r\n\x1a\nfixture'
        block = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}}
        messages = [{'role': 'system', 'content': 'native system'}, {'role': 'user',
                    'content': [{'type': 'text', 'text': 'native task'}, block]}]
        attached = attach_reference_images(messages, [block])
        self.assertEqual(attached[0], messages[0])
        self.assertEqual(attached[1]['content'][:2], messages[1]['content'])
        self.assertEqual(len(messages[1]['content']), 2)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); (p / 'credential').write_text('fixture-only')
            bridge = PacedClaude('http://fixture', p / 'credential', p / 'audit.jsonl')
            bridge.reference_inputs = [block]
            with patch('urllib.request.urlopen', return_value=Response()):
                bridge.native_completion(model=MODEL, messages=messages)
                bridge.completion(model=MODEL, messages=messages)
            self.assertEqual([c['native_reference_count'] for c in bridge.calls], [1, 0])
            self.assertEqual([len(c['images']) for c in bridge.calls], [2, 1])

    def test_input_error_before_http_is_audited_and_invalidates_the_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory); (p / 'credential').write_text('fixture-only')
            bridge = PacedClaude('http://fixture', p / 'credential', p / 'audit.jsonl')
            with patch('urllib.request.urlopen') as request, self.assertRaises(ModelUnavailable):
                bridge.completion(messages=[{'content': [{'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,@@'}}]}])
            request.assert_not_called()
            with self.assertRaises(ModelUnavailable):
                bridge.validate_completed_calls()

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
