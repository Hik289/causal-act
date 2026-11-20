"""Start one frozen UI-TARS queue after the existing WebVoyager lane finishes."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import threading

from forum_pair_queue import Queue, atomic, valid_score
from seeact_adapter import normalize_image_mime, verify_reference_delivery
from uitars_adapter import UITarsClient, image_part, native_prompt
from uitars_contract import parsed_action


def completed_method(protocol_file, method, expected_sha):
    path = Path(protocol_file)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError('Predecessor protocol changed')
    protocol = json.loads(path.read_text())
    registry = json.loads((path.parent / 'score_registry.json').read_text())
    for task in protocol['tasks']:
        record = registry.get(str(task['task_id']) + '/' + method)
        if not record:
            return False
        raw = Path(record['raw_file'])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != record['raw_sha256']:
            raise RuntimeError('Predecessor raw score changed')
        valid_score(raw, task['task_id'], method, protocol['run_id'], task['sha256'], expected_sha)
    return True


def active_evaluators():
    rows = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            if process.stat().st_uid != os.getuid() or (process / 'stat').read_text().split()[2] == 'Z':
                continue
            args = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = str((process / 'cwd').resolve())
            if ('forum_case.py' in args or 'run_multienv' in args or
                ('python' in args and 'run.py' in args and ('OSWorld' in cwd or '/forum/checkout' in cwd))):
                rows.append({'pid': int(process.name), 'args': args})
        except (OSError, ValueError):
            continue
    return rows


def technical_preflight(protocol, root):
    """One real request per attempt. Semantic predictions are retained, never selected."""
    contract_file = protocol['environment']['CAUSALACT_UITARS_FRAME_CONTRACT']
    contract = json.loads(Path(contract_file).read_text())
    inputs = protocol['technical_preflight_images']
    content = [image_part(Path(name).read_bytes()) for name in inputs]
    content.append({'type': 'text', 'text': native_prompt(contract,
        'Synthetic transport fixture. The first two images are references. The last image is the current browser. '
        'Click the text input if visible, or finish. This is not a benchmark question.')})
    client = UITarsClient(protocol['environment']['CAUSALACT_UITARS_ENDPOINT'], root / 'technical_requests.jsonl')
    response = client.complete([{'role': 'user', 'content': content}])
    expected = [item['sha256'] for item in normalize_image_mime([{'role':'user','content':content}])[1]]
    verify_reference_delivery(client.calls, expected)
    receipt = dict(valid_transport=True, benchmark_scores=0, fixture=True,
        served_model=client.calls[-1]['served_model'], image_hashes=expected,
        current_screen_last=True, semantic_response=response,
        semantic_prediction_is_not_a_start_condition=True)
    try:
        receipt['parsed_actions'] = parsed_action(response, contract)
    except Exception as error:
        receipt['native_parser_error'] = type(error).__name__
    return receipt


def run(protocol_file):
    path = Path(protocol_file).resolve()
    root = path.parent
    protocol = json.loads(path.read_text())
    lock = (root / 'deferred_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    predecessor = protocol['start_after']
    probe = object.__new__(Queue)
    probe.protocol = protocol
    while not stop.is_set():
        complete = completed_method(predecessor['protocol_file'], 'webvoyager', predecessor['protocol_sha256'])
        actors = active_evaluators()
        ready = (complete and len(actors) <= 2 and not any(predecessor['run_id'] in x['args'] for x in actors)
                 and probe.resource_ready())
        atomic(root / 'deferred_status.json', dict(controller_pid=os.getpid(),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), predecessor_method_complete=complete,
            active_evaluator_pids=[x['pid'] for x in actors], ready=ready,
            status='technical preflight' if ready else 'waiting for predecessor completion and released evaluation slot'))
        if ready:
            break
        stop.wait(30)
    if stop.is_set():
        return
    # Check every frozen source/input before making any real request.
    for name, digest in protocol['frozen_files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen UI-TARS input/source changed')
    receipt_file = root / 'technical_preflight.json'
    if not receipt_file.exists():
        for index in range(protocol['attempts_per_case']):
            attempt = root / ('technical_attempt' + str(index + 1))
            if attempt.exists():
                continue
            attempt.mkdir()
            try:
                receipt = technical_preflight(protocol, attempt)
                atomic(attempt / 'receipt.json', receipt)
                atomic(receipt_file, dict(receipt, attempt=str(attempt)))
                break
            except Exception as error:
                atomic(attempt / 'invalid.json', dict(valid=False, score=None, error=type(error).__name__, message=str(error)))
                if stop.wait(protocol['invalid_backoff_seconds']):
                    return
    if not receipt_file.exists():
        atomic(root / 'deferred_status.json', dict(controller_pid=os.getpid(),
            status='technical transport attempts exhausted; invalid, no benchmark score'))
        return
    atomic(root / 'deferred_status.json', dict(controller_pid=os.getpid(), status='formal frozen queue started',
        technical_preflight=str(receipt_file)))
    Queue(path).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    run(args.protocol)
