"""Resume a frozen matched Forum queue without replacing valid score artifacts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

from prepare_forum_seed import command, owned_container


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def remap(value, before, after):
    if isinstance(value, str):
        return value.replace(before, after)
    if isinstance(value, list):
        return [remap(x, before, after) for x in value]
    if isinstance(value, dict):
        return {k: remap(v, before, after) for k, v in value.items()}
    return value


def valid_score(path, task_id, method, run_id, original_sha, protocol_sha):
    if not path.exists():
        return None
    score = json.loads(path.read_text())
    if (score.get('valid') is not True or score.get('fixture') is not False
        or score.get('model') != ('ui-tars-1.5-7b' if method == 'uitars' else 'anthropic.claude-opus-5-5')
        or score.get('task_id') != task_id or score.get('method_key') != method
        or score.get('run_id') != run_id or score.get('original_task_sha256') != original_sha
        or score.get('protocol_sha256') != protocol_sha
        or not isinstance(score.get('score'), (int, float))
        or not math.isfinite(score['score']) or not 0 <= score['score'] <= 1):
        raise RuntimeError('Unexpected score identity; manual reconciliation required')
    return score


class Queue:
    def __init__(self, protocol_file, runtime_repair_file=None):
        self.protocol_file = Path(protocol_file).resolve()
        self.root = self.protocol_file.parent
        self.protocol = json.loads(self.protocol_file.read_text())
        self.methods = self.protocol.get('methods', ['base', 'seeact'])
        if self.methods not in (['base', 'seeact'], ['webvoyager'], ['uitars']):
            raise RuntimeError('Undeclared native method set')
        self.protocol_hash = hashlib.sha256(self.protocol_file.read_bytes()).hexdigest()
        self.lock = (self.root / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.stop = threading.Event()
        self.states, self.scores = {}, {}
        self.reservations = set()
        self.finished = False
        self.runtime_repair_file = Path(runtime_repair_file).resolve() if runtime_repair_file else None
        self.runtime_repair = None
        self.runtime_repair_hash = None
        self.repair_limits = {}
        self.repair_limits_file = None
        self.repair_limits_hash = None
        if self.runtime_repair_file:
            self.runtime_repair_hash = hashlib.sha256(self.runtime_repair_file.read_bytes()).hexdigest()
            self.runtime_repair = json.loads(self.runtime_repair_file.read_text())
            if (self.runtime_repair['base_protocol_sha256'] != self.protocol_hash
                or Path(self.runtime_repair['source_dir']).resolve() != Path(__file__).resolve().parent
                or not self.runtime_repair['environment_repair_only']):
                raise RuntimeError('Runtime repair does not match this frozen protocol/source')
        self.protected = json.loads((self.root / 'protected_scores.json').read_text()) if (self.root / 'protected_scores.json').exists() else {}
        if self.protocol['workers'] not in (1, 2):
            raise RuntimeError('This queue reserves at most two Forum workers')
        self.verify_frozen()
        for task in self.protocol['tasks']:
            for method in self.methods:
                self.find_score(task, method)
        if self.runtime_repair:
            repair_id = self.runtime_repair['repair_id']
            if not repair_id.replace('_', '').isalnum():
                raise RuntimeError('Invalid repair identity')
            self.repair_limits_file = self.root / repair_id / 'attempt_limits.json'
            self.repair_limits_file.parent.mkdir(exist_ok=True)
            if self.repair_limits_file.exists():
                limits = json.loads(self.repair_limits_file.read_text())
                if limits['runtime_repair_sha256'] != self.runtime_repair_hash:
                    raise RuntimeError('Existing repaired-attempt budget belongs to another manifest')
            else:
                allowed = self.runtime_repair.get('one_post_repair_attempt', [])
                valid_keys = {str(t['task_id']) + '/' + m for t in self.protocol['tasks'] for m in self.methods}
                if not set(allowed).issubset(valid_keys):
                    raise RuntimeError('Repair attempt identity is outside the task manifest')
                limits = {'runtime_repair_sha256': self.runtime_repair_hash, 'limits': {
                    key: len(list((self.root / 'cases' / key).glob('attempt*'))) + 1 for key in allowed}}
                atomic(self.repair_limits_file, limits)
            self.repair_limits = limits['limits']
            self.repair_limits_hash = hashlib.sha256(self.repair_limits_file.read_bytes()).hexdigest()
        self.save()

    def verify_frozen(self):
        with self.mutex:
            if hashlib.sha256(self.protocol_file.read_bytes()).hexdigest() != self.protocol_hash:
                raise RuntimeError('Frozen protocol changed')
            for name, digest in self.protocol['frozen_files'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Frozen source, task, or reference input changed: ' + name)
            for name, digest in self.protected.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('A valid score artifact changed: ' + name)
            if self.runtime_repair:
                if hashlib.sha256(self.runtime_repair_file.read_bytes()).hexdigest() != self.runtime_repair_hash:
                    raise RuntimeError('Runtime repair manifest changed')
                for name, digest in self.runtime_repair['source_files'].items():
                    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                        raise RuntimeError('Runtime repair source changed: ' + name)
            if self.repair_limits_hash and hashlib.sha256(self.repair_limits_file.read_bytes()).hexdigest() != self.repair_limits_hash:
                raise RuntimeError('Repaired-attempt budget changed')

    def find_score(self, task, method):
        key = str(task['task_id']) + '/' + method
        for path in sorted((self.root / 'cases' / str(task['task_id']) / method).glob('attempt*/result.json')):
            value = valid_score(path, task['task_id'], method, self.protocol['run_id'],
                                task['sha256'], self.protocol_hash)
            if value:
                if ([x['source_sha256'] for x in value['reference_inputs']]
                    != [x['sha256'] for x in task['reference_inputs']]):
                    raise RuntimeError('Scored reference inputs differ from frozen paired inputs')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if str(path) in self.protected and self.protected[str(path)] != digest:
                    raise RuntimeError('Protected valid score changed')
                self.protected[str(path)] = digest
                self.scores[key] = dict(value, raw_file=str(path), raw_sha256=digest)
                return value
        return None

    def save(self):
        with self.mutex:
            self.verify_frozen()
            methods = {m: [x for x in self.scores.values() if x['method_key'] == m] for m in self.methods}
            controls = self.reference_controls() if self.protocol.get('reference_control') else None
            paired = [t['task_id'] for t in self.protocol['tasks'] if all(str(t['task_id']) + '/' + m in self.scores for m in methods)]
            if controls is not None:
                paired = [t for t in paired if t in controls]
            report = dict(run_id=self.protocol['run_id'], planned_pairs=len(self.protocol['tasks']),
                          valid_pairs=len(paired), complete=len(paired) == len(self.protocol['tasks']),
                          valid_method_scores=len(self.scores), planned_method_scores=len(self.methods) * len(self.protocol['tasks']),
                          methods={m: {'valid': len(values), 'fully_passed': sum(x['score'] == 1 for x in values),
                                       'score_sum': sum(x['score'] for x in values)} for m, values in methods.items()},
                          paired={m: {'score_sum': sum(self.scores[str(t) + '/' + m]['score'] for t in paired),
                                      'denominator': len(paired)} for m in methods},
                          comparison=self.protocol.get('comparison', 'new matched initialization; historical Base/CausalAct scores remain separate'))
            if controls is not None:
                report['paired']['base'] = {'score_sum': sum(controls[t]['score'] for t in paired), 'denominator': len(paired)}
                report['referenced_control_scores'] = len(controls)
                report['control_protocol_sha256'] = self.protocol['reference_control']['protocol_sha256']
                atomic(self.root / 'referenced_control_scores.json', controls)
            report['runtime_repair_id'] = self.runtime_repair['repair_id'] if self.runtime_repair else None
            atomic(self.root / 'protected_scores.json', self.protected)
            atomic(self.root / 'score_registry.json', self.scores)
            atomic(self.root / ('pair_report.json' if report['complete'] else 'partial_pair_report.json'), report)
            atomic(self.root / 'status.json', dict(report, controller_pid=os.getpid(),
                workers=self.states, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status='complete' if report['complete'] else ('needs repair; retries exhausted' if self.finished else 'running or waiting; see actual worker states')))

    def reference_controls(self):
        """Read already-scored controls, never run or replace them in this queue."""
        info = self.protocol['reference_control']
        path = Path(info['protocol_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['protocol_sha256']:
            raise RuntimeError('Referenced control protocol changed')
        protocol = json.loads(path.read_text())
        for key in ['seed_image', 'max_steps', 'viewport', 'original_origin']:
            if self.protocol[key] != protocol[key]:
                raise RuntimeError('Referenced control setting mismatch: ' + key)
        if self.methods != ['uitars'] and self.protocol['model'] != protocol['model']:
            raise RuntimeError('Referenced control model mismatch')
        if self.methods == ['uitars'] and self.protocol.get('control_model_comparison') != 'different native planning models; no causal-effect claim':
            raise RuntimeError('UI-TARS comparison must identify the different planning model')
        original_tasks = {t['task_id']: t for t in protocol['tasks']}
        controls = {}
        for task in self.protocol['tasks']:
            if original_tasks.get(task['task_id']) != task:
                raise RuntimeError('Referenced control task or inputs differ')
            for raw in sorted((path.parent / 'cases' / str(task['task_id']) / 'base').glob('attempt*/result.json')):
                value = valid_score(raw, task['task_id'], 'base', protocol['run_id'], task['sha256'], info['protocol_sha256'])
                if value:
                    if [x['source_sha256'] for x in value['reference_inputs']] != [x['sha256'] for x in task['reference_inputs']]:
                        raise RuntimeError('Referenced control reference images differ')
                    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
                    self.protected[str(raw)] = self.protected.get(str(raw), digest)
                    if self.protected[str(raw)] != digest:
                        raise RuntimeError('Referenced valid control changed')
                    controls[task['task_id']] = dict(score=value['score'], raw_file=str(raw), raw_sha256=digest)
                    break
        return controls

    def set_state(self, lane, value):
        with self.mutex:
            self.states[str(lane)] = value

    def resource_ready(self):
        return (all(shutil.disk_usage(p).free / 1024**3 >= self.protocol['disk_floor_gib'] for p in self.protocol['work_disks'])
                and int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1]) / 1024**2 >= 48)

    def other_evaluator_count(self):
        own = {s.get('pid') for s in self.states.values()}
        count = 0
        for process in Path('/proc').glob('[0-9]*'):
            try:
                if int(process.name) in own or process.stat().st_uid != os.getuid():
                    continue
                args = (process / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
                cwd = str((process / 'cwd').resolve())
                if ('forum_case.py' in args or 'run_multienv' in args
                    or ('python' in args and 'run.py' in args
                        and ('OSWorld' in cwd or '/forum/checkout' in cwd))):
                    count += 1
            except (OSError, ValueError):
                continue
        return count

    def reserve(self, lane):
        with self.mutex:
            if self.resource_ready() and self.other_evaluator_count() + len(self.reservations) < 3:
                self.reservations.add(lane)
                return True
            return False

    def case(self, task, method, lane):
        with self.mutex:
            if self.find_score(task, method):
                return
        folder = self.root / 'cases' / str(task['task_id']) / method
        folder.mkdir(parents=True, exist_ok=True)
        attempts = sorted(folder.glob('attempt*'))
        attempt_limit = self.repair_limits.get(str(task['task_id']) + '/' + method, self.protocol['attempts_per_case'])
        while len(attempts) < attempt_limit and not self.stop.is_set():
            while not self.reserve(lane) and not self.stop.wait(30):
                self.set_state(lane, {'status': 'waiting for disk, host memory or API worker slot', 'task_id': task['task_id'], 'method': method})
                self.save()
            if self.stop.is_set():
                with self.mutex:
                    self.reservations.discard(lane)
                return
            self.verify_frozen()
            attempt = folder / ('attempt' + str(len(attempts) + 1))
            attempt.mkdir()
            port = self.protocol['first_port'] + lane
            origin = 'http://127.0.0.1:' + str(port)
            name = self.protocol['run_id'] + '_lane' + str(lane)
            owner = self.protocol['run_id'] + '_lane' + str(lane)
            if command('docker', 'inspect', name, check=False).returncode == 0:
                raise RuntimeError('Existing factory container needs reconciliation; no duplicate case')
            created, actor = False, None
            try:
                container = command('docker', 'run', '-d', '--name', name, '--label', 'causalact.owner=' + owner,
                    '--cpus', '2', '--memory', '8g', '-p', '127.0.0.1:' + str(port) + ':80', self.protocol['seed_image']).stdout.strip()
                created = True
                atomic(attempt / 'environment.json', {'container_id': container, 'seed_image': self.protocol['seed_image'],
                       'owner': owner, 'origin': origin, 'factory_reset_replaces_require_reset': True})
                ready = False
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and not self.stop.is_set():
                    try:
                        with urllib.request.urlopen(origin + '/f/food', timeout=5) as response:
                            ready = response.status == 200
                        if ready:
                            break
                    except (OSError, urllib.error.HTTPError):
                        pass
                    time.sleep(2)
                if not ready:
                    raise RuntimeError('Owned clean-seed environment did not become ready')
                config = remap(json.loads(Path(task['config_file']).read_text()), self.protocol['original_origin'], origin)
                config['image'] = [x['cache_file'] for x in task['reference_inputs']] or None
                config['require_reset'] = False
                if config.get('storage_state'):
                    config['storage_state'] = str(attempt / 'reddit_state.json')
                atomic(attempt / 'task.json', config)
                atomic(attempt / 'reference_manifest.json', task['reference_inputs'])
                endpoint = self.protocol['relay_root'] + '/worker/seeact_matched_' + str(lane) + '_' + method + '/v1/chat/completions'
                env = os.environ.copy()
                env.update(self.protocol['environment'])
                env['CUDA_VISIBLE_DEVICES'] = ''
                if self.runtime_repair_file:
                    env['CAUSALACT_RUNTIME_REPAIR_MANIFEST'] = str(self.runtime_repair_file)
                args = [self.protocol['python'], '-u', str(Path(__file__).with_name('forum_case.py')),
                    '--method', method, '--task-file', str(attempt / 'task.json'), '--output', str(attempt),
                    '--checkout', self.protocol['checkout'], '--seeact-source', self.protocol['seeact_source'],
                    '--relay-endpoint', endpoint, '--credential-file', self.protocol['credential_file'],
                    '--origin', origin, '--original-origin', self.protocol['original_origin'],
                    '--run-id', self.protocol['run_id'],
                    '--original-task-sha256', task['sha256'], '--protocol-sha256', self.protocol_hash,
                    '--reference-manifest', str(attempt / 'reference_manifest.json'),
                    '--max-steps', str(self.protocol['max_steps'])]
                with (attempt / 'worker.log').open('w') as log:
                    actor = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    self.set_state(lane, {'status': 'executing', 'method': method, 'task_id': task['task_id'],
                                         'pid': actor.pid, 'attempt': str(attempt), 'container': name})
                    self.save()
                    exit_code = actor.wait(timeout=self.protocol['case_timeout_seconds'])
                atomic(attempt / 'worker_exit.json', {'exit_code': exit_code})
                if not (attempt / 'result.json').exists():
                    raise RuntimeError('Case exited without a valid official score')
                with self.mutex:
                    self.find_score(task, method)
                    self.save()
                return
            except Exception as error:
                if (attempt / 'result.json').exists():
                    raise
                atomic(attempt / 'controller_invalid.json', {'valid': False, 'score': None,
                    'error': type(error).__name__, 'message': str(error)})
            finally:
                if actor and actor.poll() is None:
                    os.killpg(actor.pid, signal.SIGTERM)
                    try:
                        actor.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(actor.pid, signal.SIGKILL)
                        actor.wait(timeout=10)
                if created and owned_container(name, owner):
                    logs = command('docker', 'logs', name, check=False)
                    (attempt / 'world.log').write_text(logs.stdout + logs.stderr)
                    command('docker', 'rm', '-f', name)
                with self.mutex:
                    self.reservations.discard(lane)
                    self.states[str(lane)] = {'status': 'owned case cleaned', 'task_id': task['task_id'], 'method': method}
                    self.save()
            attempts = sorted(folder.glob('attempt*'))
            self.set_state(lane, {'status': 'invalid attempt retained; backoff', 'task_id': task['task_id'], 'method': method})
            self.save()
            self.stop.wait(self.protocol['invalid_backoff_seconds'])

    def lane(self, lane):
        for index, task in enumerate(self.protocol['tasks']):
            if index % self.protocol['workers'] != lane or self.stop.is_set():
                continue
            # Fixed order counterbalance, chosen before any new scores.
            for method in (self.methods if index % 2 == 0 else list(reversed(self.methods))):
                self.case(task, method, lane)
        self.set_state(lane, {'status': 'finished shard; inspect missing/invalid denominators'})
        self.save()

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        with ThreadPoolExecutor(max_workers=self.protocol['workers']) as executor:
            jobs = [executor.submit(self.lane, lane) for lane in range(self.protocol['workers'])]
            for job in as_completed(jobs):
                try:
                    job.result()
                except Exception:
                    self.stop.set()
                    raise
        # An external native method can finish before its separately owned
        # control queue. Refresh that paired intersection without re-running it.
        if self.protocol.get('reference_control') and len(self.scores) == len(self.protocol['tasks']):
            while len(self.reference_controls()) < len(self.protocol['tasks']) and not self.stop.wait(30):
                self.set_state('control', {'status': 'waiting for already-owned Base scores; no duplicate submission'})
                self.save()
        self.finished = True
        self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--runtime-repair')
    args = parser.parse_args()
    Queue(args.protocol, args.runtime_repair).run()
