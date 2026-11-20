"""One isolated Forum case; only a completed official evaluation writes a score."""
import argparse
import asyncio
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from seeact_adapter import PacedClaude, run_case, verify_reference_delivery
from base_forum_adapter import run_base_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['base', 'seeact', 'webvoyager', 'uitars'], required=True)
    parser.add_argument('--task-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkout', required=True)
    parser.add_argument('--seeact-source', required=True)
    parser.add_argument('--relay-endpoint', required=True)
    parser.add_argument('--credential-file', required=True)
    parser.add_argument('--origin', required=True)
    parser.add_argument('--original-origin', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--original-task-sha256', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--reference-manifest', required=True)
    parser.add_argument('--max-steps', type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=True)
    if (output / 'result.json').exists() or (output / 'invalid.json').exists():
        raise RuntimeError('Attempt already has a terminal record; do not overwrite it')
    os.environ.update(DATASET='visualwebarena', REDDIT=args.origin,
                      SHOPPING='http://127.0.0.1:7770', CLASSIFIEDS='http://127.0.0.1:9980',
                      WIKIPEDIA='http://127.0.0.1:8888', HOMEPAGE='http://127.0.0.1:4399',
                      OPENAI_API_KEY='local-audited-transport', CUDA_VISIBLE_DEVICES='')
    # The official package asserts this unrelated site's setting on import.
    # Forum-only tasks never call its reset endpoint.
    os.environ.setdefault('CLASSIFIEDS_RESET_TOKEN', 'unused-by-forum-only-cases')
    bridge = PacedClaude(args.relay_endpoint, args.credential_file, output / 'requests.jsonl')
    task_file = Path(args.task_file).resolve()
    try:
        repair_file = os.environ.get('CAUSALACT_RUNTIME_REPAIR_MANIFEST')
        repair = json.loads(Path(repair_file).read_text()) if repair_file else None
        navigation_timeout = repair.get('evaluation_navigation_timeout_ms') if repair else None
        if navigation_timeout is not None and (isinstance(navigation_timeout, bool) or navigation_timeout != 120000):
            raise RuntimeError('Unexpected evaluation navigation repair budget')
        if repair:
            if repair['base_protocol_sha256'] != args.protocol_sha256 or Path(repair['source_dir']).resolve() != Path(__file__).resolve().parent:
                raise RuntimeError('Runtime source repair/protocol mismatch')
            for name, digest in repair['source_files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Runtime repaired source hash mismatch')
        task = json.loads(task_file.read_text())
        if task.get('sites') != ['reddit']:
            raise RuntimeError('This isolated factory accepts Forum-only tasks')
        if args.method in ('seeact', 'webvoyager', 'uitars') and task.get('storage_state'):
            auth = output / 'auth'
            auth.mkdir(mode=0o700)
            with (output / 'login.log').open('w') as log:
                subprocess.run([sys.executable, 'browser_env/auto_login.py',
                                '--auth_folder', str(auth), '--site_list', 'reddit'],
                               cwd=args.checkout, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=120)
            state = auth / 'reddit_state.json'
            if not state.exists():
                raise RuntimeError('Official login did not create authentication state')
            state.chmod(0o600)
            task['storage_state'] = str(state)
            task_file = output / 'authenticated_task.json'
            task_file.write_text(json.dumps(task, indent=2) + '\n')
        if args.method == 'seeact':
            result = asyncio.run(run_case(args.seeact_source, args.checkout, task_file,
                                          output / 'execution', bridge, max_steps=args.max_steps,
                                          original_origin=args.original_origin, environment_origin=args.origin,
                                          evaluation_navigation_timeout_ms=navigation_timeout))
        elif args.method == 'webvoyager':
            from webvoyager_adapter import run_webvoyager_case
            result = run_webvoyager_case(os.environ['CAUSALACT_WEBVOYAGER_SOURCE'], args.checkout,
                task_file, output / 'execution', bridge, max_steps=args.max_steps,
                browser_binary=os.environ['CAUSALACT_WEBVOYAGER_BROWSER'],
                chromedriver=os.environ['CAUSALACT_WEBVOYAGER_DRIVER'],
                original_origin=args.original_origin, environment_origin=args.origin,
                evaluation_navigation_timeout_ms=navigation_timeout)
        elif args.method == 'uitars':
            from uitars_adapter import run_uitars_case
            result = run_uitars_case(args.checkout, task_file, output / 'execution', bridge,
                endpoint=os.environ['CAUSALACT_UITARS_ENDPOINT'],
                frame_contract=os.environ['CAUSALACT_UITARS_FRAME_CONTRACT'],
                original_origin=args.original_origin, environment_origin=args.origin,
                max_steps=args.max_steps, evaluation_navigation_timeout_ms=120000)
        else:
            result = run_base_case(args.checkout, task_file, output / 'execution', bridge,
                                   max_steps=args.max_steps, original_origin=args.original_origin,
                                   environment_origin=args.origin,
                                   evaluation_navigation_timeout_ms=navigation_timeout)
        if not math.isfinite(result['score']) or not 0 <= result['score'] <= 1:
            raise RuntimeError('Official evaluator returned an invalid score')
        if result.get('origin_guard_contexts', 0) < 1:
            raise RuntimeError('Independent environment origin guard was not installed')
        if result.get('origin_guard_errors'):
            raise RuntimeError('Independent environment origin routing failed')
        references = json.loads(Path(args.reference_manifest).read_text())
        if args.method != 'uitars':
            verify_reference_delivery(bridge.calls, [x['transport_sha256'] for x in references])
        result.update(run_id=args.run_id, method_key=args.method,
                      original_task_sha256=args.original_task_sha256,
                      protocol_sha256=args.protocol_sha256,
                      fixture=False, environment_origin=args.origin)
        if repair:
            result.update(runtime_repair_id=repair['repair_id'], runtime_source_commit=repair['source_commit'],
                          runtime_repair_sha256=hashlib.sha256(Path(repair_file).read_bytes()).hexdigest(),
                          runtime_source_files=repair['source_files'],
                          evaluation_navigation_timeout_ms=navigation_timeout)
        temporary = output / 'result.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output / 'result.json')
    except Exception as error:
        (output / 'invalid.json').write_text(json.dumps({
            'valid': False, 'score': None, 'kind': 'inference or environment failure; not a task zero',
            'error': type(error).__name__, 'message': str(error),
            'run_id': args.run_id, 'method_key': args.method}, indent=2) + '\n')
        (output / 'error.log').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
