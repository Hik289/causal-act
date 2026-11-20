"""Exercise the complete browser/evaluator adapter with zero model requests."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

from seeact_adapter import normalize_image_mime
from uitars_adapter import run_uitars_case


class FixturePage(BaseHTTPRequestHandler):
    def do_GET(self):
        body = ('<!doctype html><html><body><form onsubmit="event.preventDefault();'
                "document.getElementById('proof').textContent=document.getElementById('entry').value;\">"
                '<input id="entry" style="position:absolute;left:50px;top:50px;width:300px;height:50px">'
                '<button type="submit">Submit</button></form><div id="proof">empty</div></body></html>').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class NoModelBridge:
    calls = []

    def sdk_create(self, *args, **kwargs):
        raise AssertionError('Scripted DOM fixture must not call an evaluation model')


def run(checkout, contract, output):
    from PIL import Image
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(('127.0.0.1', 0), FixturePage)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = 'http://127.0.0.1:' + str(server.server_port)
    os.environ.update(DATASET='visualwebarena', REDDIT=origin, SHOPPING='http://127.0.0.1:7770',
        CLASSIFIEDS='http://127.0.0.1:9980', CLASSIFIEDS_RESET_TOKEN='unused-by-forum-only-cases',
        WIKIPEDIA='http://127.0.0.1:8888', HOMEPAGE='http://127.0.0.1:4399',
        OPENAI_API_KEY='fixture-does-not-call-a-model', CUDA_VISIBLE_DEVICES='')
    geometry = json.loads(Path(contract).read_text())
    width, height = geometry['screenshot_size']
    mw, mh = geometry['model_frame_size']
    images = []
    for index, color in enumerate(['red', 'blue']):
        path = out / ('reference_' + str(index) + '.png')
        Image.new('RGB', (17 + index, 19 + index), color).save(path)
        images.append(str(path.resolve()))
    responses = ["Thought: address navigation fixture.\nAction: hotkey(key='ctrl l')",
        "Thought: address navigation fixture.\nAction: type(content='" + origin + "/target\\n')",
        "Thought: input fixture.\nAction: click(point='<point>" + str(round(100 * mw / width)) + ' ' +
            str(round(75 * mh / height)) + "</point>')",
        "Thought: unicode fixture.\nAction: type(content='测试成功\\n')",
        "Thought: fixture done.\nAction: finished(content='done')"]

    class ScriptedClient:
        def __init__(self, endpoint, audit_file, max_tokens):
            self.calls, self.audit_file = [], Path(audit_file)

        def complete(self, messages):
            messages, inputs = normalize_image_mime(messages)
            response = responses[len(self.calls)]
            record = dict(stage='planning', model='scripted fixture; no inference', status=200,
                images=inputs, response=response, request_sha256=hashlib.sha256(json.dumps(messages).encode()).hexdigest())
            self.calls.append(record)
            with self.audit_file.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            return response

    results = []
    try:
        for index, expected in enumerate(['测试成功', 'different expected content']):
            task = dict(task_id=index, intent='Synthetic adapter fixture; type the supplied fixture text and submit.',
                image=images, start_url=origin, eval={'eval_types': ['program_html'], 'program_html': [
                    {'url': 'last', 'locator': "document.getElementById('proof').textContent",
                     'required_contents': {'exact_match': expected}}]})
            task_file = out / ('task_' + str(index) + '.json')
            task_file.write_text(json.dumps(task))
            result = run_uitars_case(checkout, task_file, out / ('case_' + str(index)), NoModelBridge(),
                endpoint='fixture://no-model', frame_contract=contract, original_origin='http://127.0.0.1:9999',
                environment_origin=origin, max_steps=10, client_factory=ScriptedClient, fixture=True)
            assert result['score'] == (1.0 if index == 0 else 0.0), result
            assert result['fixture'] and result['evaluation_model_calls'] == 0 and result['steps'] == 5
            calls = [json.loads(line) for line in (out / ('case_' + str(index)) / 'uitars_requests.jsonl').read_text().splitlines()]
            assert all(len(call['images']) == 2 + min(step + 1, 3) for step, call in enumerate(calls))
            result['raw_file'] = str(out / ('case_' + str(index)) / 'fixture_result.json')
            Path(result['raw_file']).write_text(json.dumps(result, indent=2) + '\n')
            results.append(result)
    finally:
        server.shutdown()
        server.server_close()
    receipt = dict(fixture_passed=True, actual_model_requests=0, benchmark_scores=0,
        description='Scripted full browser/native-parser/reference-history/official-DOM-evaluator fixture; all outputs excluded',
        positive_fixture_score=results[0]['score'], negative_fixture_score=results[1]['score'],
        case_results=[r['raw_file'] for r in results])
    (out / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.checkout, args.contract, args.output)
