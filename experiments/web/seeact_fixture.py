"""Unscored integration check: native click, retained DOM state, VWA evaluator."""
import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from seeact_adapter import PacedClaude, run_case


HTML = b'''<!doctype html><html><head><title>Local adapter fixture</title></head>
<body style="font:24px sans-serif;padding:60px"><h1>Adapter readiness check</h1>
<label>State <input id="proof" readonly value="not ready"></label>
<p><button style="padding:25px;background:#1565c0;color:white;font-size:24px"
onclick="document.querySelector('#proof').value='ready-local';this.innerText='Done'">
Continue</button></p><p>Continue changes the local state to ready-local.</p></body></html>'''

REFERENCE_HTML = b'''<!doctype html><html><head><title>Local reference input fixture</title></head>
<body style="font:24px sans-serif;padding:60px"><h1>Reference input check</h1>
<label>State <input id="proof" readonly value="not ready"></label><p>
<button style="padding:30px;background:#ff8800;color:black;font-size:24px"
onclick="document.querySelector('#proof').value='other'">Option A</button>
<button style="padding:30px;background:#1565c0;color:white;font-size:24px"
onclick="document.querySelector('#proof').value='ready-local'">Option B</button>
</p></body></html>'''


class FixtureHandler(BaseHTTPRequestHandler):
    html = HTML

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', str(len(self.html)))
        self.end_headers()
        self.wfile.write(self.html)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeact-source', required=True)
    parser.add_argument('--vwa-checkout', required=True)
    parser.add_argument('--relay-endpoint', required=True)
    parser.add_argument('--credential-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', choices=['seeact', 'base'], default='seeact')
    parser.add_argument('--reference-image-check', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    defaults = {'DATASET': 'visualwebarena', 'REDDIT': 'http://127.0.0.1:9999',
                'SHOPPING': 'http://127.0.0.1:7770', 'CLASSIFIEDS': 'http://127.0.0.1:9980',
                'CLASSIFIEDS_RESET_TOKEN': 'unused-by-this-synthetic-fixture',
                'WIKIPEDIA': 'http://127.0.0.1:8888', 'HOMEPAGE': 'http://127.0.0.1:4399'}
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if args.reference_image_check:
        from PIL import Image
        Image.new('RGB', (80, 80), '#1565c0').save(out / 'task_reference.png')
        FixtureHandler.html = REFERENCE_HTML
    server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    task = {'task_id': -900001, 'intent': 'Click Continue once to make the State ready-local, then terminate.',
            'start_url': 'http://127.0.0.1:' + str(server.server_port), 'sites': ['reddit'],
            'storage_state': None, 'geolocation': None, 'require_reset': False,
            'eval': {'eval_types': ['program_html'], 'reference_answers': None, 'reference_url': '',
                     'program_html': [{'url': 'last', 'locator': 'document.querySelector("#proof").value',
                                       'required_contents': {'exact_match': 'ready-local'}}]}}
    if args.reference_image_check:
        task['task_id'] = -900002
        task['image'] = str(out / 'task_reference.png')
        task['intent'] = 'Click the button whose background color matches the task reference image, then terminate.'
    config = out / 'fixture.json'
    config.write_text(json.dumps(task, indent=2) + '\n')
    bridge = PacedClaude(args.relay_endpoint, args.credential_file, out / 'requests.jsonl')
    try:
        if args.method == 'seeact':
            result = asyncio.run(run_case(args.seeact_source, args.vwa_checkout, config,
                                          out / 'attempt', bridge, max_steps=3))
        else:
            from base_forum_adapter import run_base_case
            result = run_base_case(args.vwa_checkout, config, out / 'attempt', bridge, max_steps=3)
        metadata = {'kind': 'synthetic adapter fixture; excluded from all benchmark denominators',
                    'fixture_passed': result['score'] == 1., 'benchmark_score_created': False,
                    'evaluation': result}
        (out / 'preflight.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata))
        if not metadata['fixture_passed']:
            raise RuntimeError('Synthetic click/state/evaluator check did not pass')
    except Exception as error:
        (out / 'error.json').write_text(json.dumps({'kind': 'unscored fixture failure',
            'error_type': type(error).__name__, 'message': str(error)}, indent=2) + '\n')
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
