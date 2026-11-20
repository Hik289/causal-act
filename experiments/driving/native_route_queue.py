"""Resume pinned native driving methods on the original frozen 100 routes."""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import xml.etree.ElementTree as ET


class DeviceIdentityError(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def active(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


def group_pids(group):
    found = set()
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                found.add(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def clean_group(child):
    # Only groups explicitly created by this controller are ever signalled.
    if group_pids(child.pid):
        os.killpg(child.pid, signal.SIGTERM)
        for _ in range(40):
            if not group_pids(child.pid):
                break
            time.sleep(.5)
        if group_pids(child.pid):
            os.killpg(child.pid, signal.SIGKILL)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Owned child did not exit after cleanup')


def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    gpus = [dict(index=int(x.split(',')[0]), uuid=x.split(',')[1].strip(), free=int(x.split(',')[2])) for x in rows]
    busy = collections.defaultdict(set)
    for row in processes:
        if row.strip():
            uuid, pid = row.split(',')[:2]
            busy[uuid.strip()].add(int(pid))
    return gpus, busy


def rendered_gpu_uuids(pids):
    tree = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
    return {g.findtext('uuid') for g in tree.findall('gpu')
            if any(int(p.findtext('pid')) in pids for p in g.findall('processes/process_info'))}


def normalized_uuid(value):
    value = value.lower()
    return (value[4:] if value.startswith('gpu-') else value).replace('-', '')


def mapped_vulkan_index(devices, gpu_uuid):
    matches = [d['vulkan_index'] for d in devices if d['vendor_id'] == 0x10de
               and normalized_uuid(d['device_uuid']) == normalized_uuid(gpu_uuid)]
    if len(matches) != 1:
        raise DeviceIdentityError('Vulkan/NVIDIA device identity is missing or ambiguous')
    return matches[0]


def valid_record(raw_file, log_file, expected_route):
    """Retain scored task failures; exclude initialization, kernel and environment errors."""
    if not raw_file.exists() or not log_file.exists():
        return None
    raw = json.loads(raw_file.read_text())
    if raw.get('entry_status') in ('Rejected', 'Crashed'):
        return None
    log = log_file.read_text(errors='replace')
    if re.search(r'Traceback \(most recent call last\)|VK_ERROR_OUT_OF_DEVICE_MEMORY', log):
        return None
    rows = raw.get('_checkpoint', {}).get('records', [])
    if len(rows) != 1 or rows[0].get('route_id') != expected_route:
        return None
    record = rows[0]
    status = record.get('status', '')
    score = record.get('scores', {}).get('score_composed')
    if (not (status in ('Perfect', 'Completed') or status.startswith('Failed - '))
        or re.search(r'crash|initialization|set[ _-]*up|invalid|rejected', status, re.I)
        or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0):
        return None
    return dict(record, raw_file=str(raw_file), raw_sha256=sha(raw_file),
                log_file=str(log_file), log_sha256=sha(log_file))


class Queue:
    def __init__(self, protocol_file):
        self.path = Path(protocol_file).resolve()
        self.root = self.path.parent
        self.protocol = json.loads(self.path.read_text())
        self.protocol_sha = sha(self.path)
        self.lock = (self.root / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = threading.Event()
        self.records = {}
        self.state = dict(controller_pid=os.getpid(), run_id=self.protocol['run_id'],
            benchmark_scores=0, workers_max=1, phase='verifying frozen sources and inputs')
        self.routes = ET.parse(self.protocol['routes_file']).getroot().findall('route')
        assert len(self.routes) == 100 and len({r.get('id') for r in self.routes}) == 100
        for name, digest in self.protocol['frozen_files'].items():
            if sha(name) != digest:
                raise RuntimeError('Frozen native input/source changed: ' + name)
        self.fingerprints = {name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                             for name in self.protocol['frozen_files']}
        self.load_scores()

    def verify(self):
        if sha(self.path) != self.protocol_sha:
            raise RuntimeError('Frozen route protocol changed')
        for name, stat in self.fingerprints.items():
            current = Path(name).stat()
            if (current.st_size, current.st_mtime_ns) != stat:
                raise RuntimeError('Frozen file changed during this controller: ' + name)
        for value in self.records.values():
            if sha(value['raw_file']) != value['raw_sha256'] or sha(value['log_file']) != value['log_sha256']:
                raise RuntimeError('A protected native result changed')

    def load_scores(self):
        for method in self.protocol['methods']:
            for route in self.routes:
                route_id = 'RouteScenario_' + route.get('id') + '_rep0'
                for folder in sorted((self.root / 'cases' / method / route.get('id')).glob('attempt*')):
                    receipt = folder / 'verified_result.json'
                    if not receipt.exists():
                        continue
                    value = json.loads(receipt.read_text())
                    if (value['protocol_sha256'] != self.protocol_sha or value['method'] != method
                        or value['fixture'] or value['route_id'] != route_id):
                        raise RuntimeError('Existing native result identity mismatch')
                    record = valid_record(folder / 'results.json', folder / 'evaluation.log', route_id)
                    if not record or record != value['record']:
                        raise RuntimeError('Protected native result mismatch')
                    self.records[method + '/' + route.get('id')] = record
                    break
        self.save()

    def save(self):
        self.verify()
        methods = {}
        for method in self.protocol['methods']:
            rows = [v for key, v in self.records.items() if key.startswith(method + '/')]
            methods[method] = dict(valid=len(rows), planned=100, remaining=100-len(rows),
                mean_driving_score=sum(x['scores']['score_composed'] for x in rows)/len(rows) if rows else None,
                successes=sum(x['status'] in ('Perfect','Completed') for x in rows),
                red_light=sum(len(x['infractions']['red_light']) for x in rows),
                collisions=sum(len(v) for x in rows for k,v in x['infractions'].items() if k.startswith('collisions_')))
        self.state.update(methods=methods, benchmark_scores=len(self.records), protocol_sha256=self.protocol_sha,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic(self.root / 'status.json', self.state)
        atomic(self.root / 'verified_records.json', self.records)
        if len(self.records) == 100*len(self.protocol['methods']):
            atomic(self.root / 'report.json', self.state)

    def wait_predecessor(self):
        path = Path(self.protocol['GPU_setup_protocol'])
        if sha(path) != self.protocol['GPU_setup_protocol_sha256']:
            raise RuntimeError('GPU setup protocol changed')
        status_file = path.parent / 'status.json'
        while not self.stop.is_set():
            state = json.loads(status_file.read_text())
            if not active(state['controller_pid']) and not state.get('active_pid'):
                return state
            self.state.update(phase='waiting for the existing native GPU setup controller to finish')
            self.save()
            self.stop.wait(30)
        return {}

    def free_gpu(self):
        stable_since, candidate = None, None
        while not self.stop.is_set():
            gpus, busy = inventory()
            ready = [g for g in gpus if not busy[g['uuid']] and g['free'] >= 36000]
            disk = shutil.disk_usage(self.root).free / 1024**3 >= 10
            ports_free = True
            for port in self.protocol['ports']:
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', port)) == 0:
                        ports_free = False
            if candidate not in [g['uuid'] for g in ready] or not disk or not ports_free:
                stable_since, candidate = None, None
            if candidate is None and ready and disk and ports_free:
                candidate = sorted(ready, key=lambda g: [1,2,0,3].index(g['index']))[0]['uuid']
                stable_since = time.monotonic()
            if candidate and time.monotonic() - stable_since >= 60:
                return next(g for g in ready if g['uuid'] == candidate)
            self.state.update(phase='waiting for continuously unoccupied GPU, memory, ports and disk', inventory=gpus)
            self.save()
            self.stop.wait(5)
        return None

    def case(self, method, spec, route):
        key = method + '/' + route.get('id')
        if key in self.records:
            return
        folder = self.root / 'cases' / method / route.get('id')
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(1, self.protocol['attempts_per_route'] + 1):
            out = folder / ('attempt' + str(index))
            if out.exists():
                continue
            gpu = self.free_gpu()
            if gpu is None:
                return
            self.verify()
            devices = json.loads(subprocess.check_output([self.protocol['vulkan_identity_tool']], text=True))
            adapter = mapped_vulkan_index(devices, gpu['uuid'])
            out.mkdir()
            xml = ET.Element('routes');xml.append(ET.fromstring(ET.tostring(route)))
            ET.ElementTree(xml).write(out / 'route.xml', encoding='utf-8', xml_declaration=True)
            env = dict(os.environ, **spec['environment'], CUDA_VISIBLE_DEVICES=gpu['uuid'])
            env.pop('SAVE_PATH', None)
            atomic(out / 'runtime.json', dict(GPU=gpu, vulkan_adapter=adapter, vulkan_devices=devices,
                native_config=spec['config'], checkpoint=spec['checkpoint'], protocol_sha256=self.protocol_sha))
            simulator = actor = None
            try:
                with (out / 'simulator.log').open('x') as log:
                    simulator = subprocess.Popen([self.protocol['carla_launcher'], '-RenderOffScreen',
                        '-graphicsadapter=' + str(adapter), '-carla-rpc-port=' + str(self.protocol['ports'][0]), '-nosound'],
                        cwd=self.protocol['carla_root'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                self.state.update(phase='starting owned simulator', method=method, route=key,
                    simulator_pid=simulator.pid, GPU=gpu, attempt=str(out))
                self.save()
                if self.stop.wait(55) or simulator.poll() is not None:
                    raise RuntimeError('Simulator did not remain alive through boot')
                actual = rendered_gpu_uuids(group_pids(simulator.pid))
                if actual != {gpu['uuid']}:
                    raise DeviceIdentityError('Simulator rendered on an unverified or different physical GPU: ' + str(actual))
                _, busy = inventory()
                if busy[gpu['uuid']] - group_pids(simulator.pid):
                    raise RuntimeError('GPU acquired by another compute job before native evaluation')
                command = [spec['python'], '-u', self.protocol['evaluator'],
                    '--routes=' + str(out / 'route.xml'), '--repetitions=1', '--track=SENSORS',
                    '--traffic-manager-seed=0', '--checkpoint=' + str(out / 'results.json'),
                    '--debug-checkpoint=' + str(out / 'live_results.txt'), '--agent=' + spec['entrypoint'],
                    '--agent-config=' + spec['config'] + '+' + spec['checkpoint'] + '+' + self.protocol['run_id'],
                    '--debug=0', '--host=127.0.0.1', '--port=' + str(self.protocol['ports'][0]),
                    '--traffic-manager-port=' + str(self.protocol['ports'][1])]
                with (out / 'evaluation.log').open('x') as log:
                    actor = subprocess.Popen(command, cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.state.update(phase='native simulator lifecycle and route evaluation', evaluation_pid=actor.pid)
                    self.save()
                    while actor.poll() is None and not self.stop.wait(5):
                        _, busy = inventory()
                        if busy[gpu['uuid']] - group_pids(simulator.pid) - group_pids(actor.pid):
                            raise RuntimeError('Foreign compute entered allocated GPU; own attempt stopped')
                        if time.time() - (out / 'evaluation.log').stat().st_mtime > self.protocol['environment_stall_seconds']:
                            raise RuntimeError('Owned native evaluation produced no progress for the frozen environment stall window')
                    if self.stop.is_set():
                        raise RuntimeError('Controller stop requested')
                # Let the owned Unreal process flush its log before preserving hashes.
                clean_group(simulator)
                simulator = None
                record = valid_record(out / 'results.json', out / 'evaluation.log',
                    'RouteScenario_' + route.get('id') + '_rep0')
                if record is None:
                    raise RuntimeError('No valid native official result; inspect retained logs')
                atomic(out / 'verified_result.json', dict(valid=True, fixture=False, method=method,
                    route_id=record['route_id'], record=record, protocol_sha256=self.protocol_sha,
                    native_planning_model=method, same_model_causal_effect_claim=False))
                self.records[key] = record
                self.save()
                return
            except Exception as error:
                atomic(out / 'invalid.json', dict(valid=False, score=None, error=type(error).__name__, message=str(error)))
                if isinstance(error, DeviceIdentityError):
                    self.state['needs_repair'] = str(error)
                    self.stop.set()
            finally:
                if actor is not None:
                    clean_group(actor)
                if simulator is not None:
                    clean_group(simulator)
                self.state.pop('simulator_pid', None);self.state.pop('evaluation_pid', None)
                self.state['phase'] = 'device identity needs repair; own attempt stopped' if self.state.get('needs_repair') else 'owned attempt cleaned; valid scores retained'
                self.save()
            if self.stop.wait(self.protocol['invalid_backoff_seconds']):
                return

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        previous = self.wait_predecessor()
        if self.stop.is_set():
            return
        receipt_file = self.root / 'GPU_setup_receipt.json'
        receipt = {}
        for method, value in previous.get('methods', {}).items():
            raw = Path(self.protocol['GPU_setup_protocol']).parent / method / 'result.json'
            if raw.exists():
                if json.loads(raw.read_text()) != value:
                    raise RuntimeError('GPU setup state and original method result disagree')
                receipt[method] = dict(raw_file=str(raw), raw_sha256=sha(raw), result=value)
        if receipt_file.exists() and json.loads(receipt_file.read_text()) != receipt:
            raise RuntimeError('Original GPU setup receipt changed')
        atomic(receipt_file, receipt)
        for method, spec in self.protocol['methods'].items():
            result = previous.get('methods', {}).get(method, {})
            if not result.get('native_GPU_setup_passed'):
                self.state.setdefault('blocked_methods', {})[method] = 'GPU setup has no successful result; no route submitted'
                self.save()
                continue
            for route in self.routes:
                if self.stop.is_set():
                    return
                self.case(method, spec, route)
                if not any(key.startswith(method + '/') for key in self.records):
                    self.state.setdefault('blocked_methods', {})[method] = 'first native route has no valid official result after bounded attempts; remaining99 await environment repair'
                    self.save()
                    break
        self.state['phase'] = 'complete' if len(self.records) == 100*len(self.protocol['methods']) else 'missing or invalid routes remain; inspect method gates and original logs'
        self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    queue = Queue(args.protocol)
    try:
        queue.run()
    except Exception as error:
        queue.state.update(phase='needs repair; original results retained', error=type(error).__name__, message=str(error))
        queue.save()
        raise
