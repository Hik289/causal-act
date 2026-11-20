"""Freeze one WebVoyager queue against an existing same-initial-state Base run."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

from forum_pair_queue import Queue, atomic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    root = Path(args.output).resolve()
    if root.exists():
        raise RuntimeError('Run already exists; inspect and resume, never reprepare it')
    control_file = Path(args.control_protocol).resolve()
    original = json.loads(control_file.read_text())
    dependencies = Path(args.dependencies).resolve()
    dep = json.loads(dependencies.read_text())
    fixture = Path(args.claude_fixture).resolve()
    check = json.loads(fixture.read_text())
    if not check['fixture_passed'] or check['scripted'] or check['benchmark_score_created'] or check['actual_model_requests'] < 2:
        raise RuntimeError('A real Claude reference/routing/state preflight is required')
    source = Path(dep['native_source'])
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != dep['commit']:
        raise RuntimeError('Native source commit differs from dependency manifest')
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    evaluation = check['evaluation']
    if (evaluation['model'] != original['model'] or evaluation['viewport'] != [1280, 720, 1]
        or evaluation['official_evaluator_sha256'] != sha(Path(original['checkout']) / 'evaluation_harness/evaluators.py')):
        raise RuntimeError('Fixture used a different model, viewport or official evaluator')
    for name, digest in evaluation['native_files_sha256'].items():
        if sha(source / name) != digest:
            raise RuntimeError('Fixture used different native source')
    calls = [json.loads(line) for line in (fixture.parent / 'requests.jsonl').read_text().splitlines()]
    if len(calls) != check['actual_model_requests'] or any(c.get('error') or c.get('fixture_scripted')
            or c.get('status') != 200 or c.get('served_model') != original['model'] for c in calls):
        raise RuntimeError('Real fixture request audit does not match its reported preflight')
    if sha(dep['chromedriver']) != dep['driver_sha256'] or sha(dep['browser']) != dep['browser_sha256']:
        raise RuntimeError('Browser or driver differs from tested binaries')
    protocol = copy.deepcopy(original)
    protocol.update(run_id='webvoyager_forum170_20261120', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        public_source_commit=args.public_source_commit, methods=['webvoyager'], planned_method_scores=len(original['tasks']),
        workers=1, python=dep['python'], first_port=9964, max_tokens_per_planning_request=1000,
        reference_control={'protocol_file': str(control_file), 'protocol_sha256': sha(control_file),
            'note': 'Reference only already-valid raw Base scores from this separately owned queue; no Base submissions here'},
        method_order='original manifest order; one WebVoyager worker; paired intersection with existing Base reported separately',
        webvoyager_source=str(source), webvoyager_commit=commit,
        native_configuration={'max_attached_imgs': 3, 'fix_box_color': True, 'text_only': False,
            'native_seed_argument': 42, 'actual_sampling': 'unchanged shared relay default; native seed not transmitted',
            'window_adaptation': 'force actual inner viewport1280x720 at scale1 using same Chromium binary as controls',
            'unsupported_native_PDF_Assistant_API': 'invalid attempt, never substitute a task zero'},
        dependencies=dep, preparation_fixture_file=str(fixture), preparation_fixture_sha256=sha(fixture))
    protocol['environment'].update(PATH=str(Path(dep['python']).parent) + os.pathsep + original['environment']['PATH'],
        TMPDIR=str(root / 'tmp'), CIVISUAL_USAGE_LOG=str(root / 'native_usage.jsonl'),
        CAUSALACT_WEBVOYAGER_SOURCE=str(source), CAUSALACT_WEBVOYAGER_BROWSER=dep['browser'],
        CAUSALACT_WEBVOYAGER_DRIVER=dep['chromedriver'])
    files = protocol['frozen_files']
    for name in subprocess.check_output(['git', '-C', str(source), 'ls-files'], text=True).splitlines():
        path = source / name
        if path.is_file():
            files[str(path)] = sha(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        files[str(path)] = sha(path)
    for path in [control_file, dependencies, fixture, fixture.parent / 'requests.jsonl', Path(dep['chromedriver']), Path(dep['browser'])]:
        files[str(path)] = sha(path)
    repair = control_file.parent / 'navigation_observation_repair_20261119/manifest.json'
    if repair.exists():
        files[str(repair)] = sha(repair)
        files.update(json.loads(repair.read_text())['source_files'])
    for name, digest in files.items():
        if sha(name) != digest:
            raise RuntimeError('Frozen dependency/input/source changed: ' + name)
    root.mkdir()
    (root / 'tmp').mkdir()
    atomic(root / 'protocol.json', protocol)
    queue = Queue(root / 'protocol.json')
    try:
        queue.set_state(0, {'status': 'prepared; no formal worker started'})
        queue.save()
    finally:
        queue.lock.close()
    atomic(root / 'readiness.json', {'prepared': True, 'benchmark_scores': 0,
        'protocol_sha256': sha(root / 'protocol.json'), 'frozen_files': len(files),
        'new_workers': 1, 'same_initialization_control': str(control_file),
        'fixture_is_not_a_benchmark_score': True})
    print(json.dumps({'run': str(root), 'protocol_sha256': sha(root / 'protocol.json'), 'frozen_files': len(files)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control-protocol', required=True)
    parser.add_argument('--dependencies', required=True)
    parser.add_argument('--claude-fixture', required=True)
    parser.add_argument('--public-source-commit', required=True)
    prepare(parser.parse_args())
