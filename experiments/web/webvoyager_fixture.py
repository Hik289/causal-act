"""Unscored WebVoyager integration fixture; never writes benchmark result.json."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import types

from seeact_adapter import PacedClaude, MODEL, attach_reference_images, normalize_image_mime, verify_reference_delivery
from webvoyager_adapter import run_webvoyager_case


class ScriptedBridge(PacedClaude):
    """Exercise native code without representing scripted responses as API calls."""
    def completion(self, **kwargs):
        messages = attach_reference_images(kwargs['messages'], self.reference_inputs)
        _, images = normalize_image_mime(messages)
        if not self.calls:
            text = '\n'.join(x['text'] for message in messages for x in message.get('content', [])
                             if isinstance(x, dict) and x.get('type') == 'text')
            label = re.search(r'\[(\d+)\]: <button> "Option B"', text).group(1)
            content = 'Thought: fixture click.\nAction: Click [' + label + ']'
        else:
            content = 'Thought: fixture done.\nAction: ANSWER; ready-local'
        record = {'stage': 'planning', 'images': images, 'fixture_scripted': True,
                  'served_model': MODEL, 'status': None, 'not_an_actual_model_request': True}
        self.calls.append(record)
        self.audit_path.write_text(''.join(json.dumps(row) + '\n' for row in self.calls))
        response = {'id': 'scripted-fixture', 'object': 'chat.completion', 'created': 0, 'model': MODEL,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        return types.SimpleNamespace(raw_response=response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--browser', required=True)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--endpoint')
    parser.add_argument('--credential-file')
    parser.add_argument('--scripted', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key, value in {'DATASET': 'visualwebarena', 'REDDIT': 'http://127.0.0.1:9999',
        'SHOPPING': 'http://127.0.0.1:7770', 'CLASSIFIEDS': 'http://127.0.0.1:9980',
        'CLASSIFIEDS_RESET_TOKEN': 'unused-fixture', 'WIKIPEDIA': 'http://127.0.0.1:8888',
        'HOMEPAGE': 'http://127.0.0.1:4399', 'OPENAI_API_KEY': 'local-fixture'}.items():
        os.environ.setdefault(key, value)
    hits = {'legacy': 0, 'owned_proof': 0}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.server is legacy:
                hits['legacy'] += 1
            if self.path == '/proof':
                hits['owned_proof'] += self.server is owned
                body, mime = b'proof', 'text/plain'
            else:
                body = ('''<html><body style="font:24px sans-serif;padding:60px"><h1>Reference check</h1>
                <input id="proof" readonly value="not ready"><p>
                <button style="padding:30px;background:#ff8800">Option A</button>
                <button style="padding:30px;background:#1565c0;color:white" onclick="
                  fetch('ORIGIN/proof').then(r=>r.text()).then(()=>document.querySelector('#proof').value='ready-local')">
                Option B</button></p></body></html>'''.replace('ORIGIN', old_origin)).encode()
                mime = 'text/html'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass
    legacy = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    owned = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    old_origin, origin = ['http://127.0.0.1:' + str(s.server_port) for s in [legacy, owned]]
    for server in [legacy, owned]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    from PIL import Image
    reference = out / 'reference.png'
    Image.new('RGB', (80, 80), '#1565c0').save(reference)
    task = {'task_id': -900003, 'intent': 'Click the button whose background matches the task reference image, then terminate.',
        'image': str(reference), 'start_url': origin, 'sites': ['reddit'], 'storage_state': None,
        'geolocation': None, 'require_reset': False,
        'eval': {'eval_types': ['program_html'], 'reference_answers': None, 'reference_url': '',
            'program_html': [{'url': 'last', 'locator': 'document.querySelector("#proof").value',
                'required_contents': {'exact_match': 'ready-local'}}]}}
    config = out / 'fixture.json'
    config.write_text(json.dumps(task, indent=2) + '\n')
    bridge = (ScriptedBridge if args.scripted else PacedClaude)(args.endpoint, args.credential_file or '/unused', out / 'requests.jsonl')
    try:
        result = run_webvoyager_case(args.source, args.checkout, config, out / 'execution', bridge,
            browser_binary=args.browser, chromedriver=args.driver,
            original_origin=old_origin, environment_origin=origin, max_steps=3)
        verify_reference_delivery(bridge.calls, [hashlib.sha256(reference.read_bytes()).hexdigest()])
        passed = result['score'] == 1 and hits['legacy'] == 0 and hits['owned_proof'] > 0
        metadata = {'kind': 'synthetic fixture, excluded from every benchmark denominator',
            'scripted': args.scripted, 'actual_model_requests': 0 if args.scripted else len(bridge.calls),
            'fixture_passed': passed, 'benchmark_score_created': False, 'routing_hits': hits, 'evaluation': result}
        (out / 'preflight.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({k: v for k, v in metadata.items() if k != 'evaluation'}))
        if not passed:
            raise RuntimeError('Native action, reference, routing or retained DOM check failed')
    finally:
        for server in [legacy, owned]:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
