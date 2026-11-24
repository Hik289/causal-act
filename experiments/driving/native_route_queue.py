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


def wait_simulator_device(simulator, gpu_uuid, stop, timeout_seconds=545):
    """After the original 55 s warmup, confirm the device within 600 s total.

    A live launcher with no registered GPU process is still unverified. Wait
    without starting the evaluator; a wrong device or foreign compute remains
    an immediate failure. The native task and its tick budget have not begun.
    """
    deadline = time.monotonic() + timeout_seconds
    while not stop.is_set():
        if simulator.poll() is not None:
            raise RuntimeError('Simulator exited before physical GPU confirmation')
        pids = group_pids(simulator.pid)
        actual = rendered_gpu_uuids(pids)
        if actual and actual != {gpu_uuid}:
            raise DeviceIdentityError('Simulator rendered on an unverified or different physical GPU: ' + str(actual))
        _, busy = inventory()
        if busy[gpu_uuid] - pids:
            raise RuntimeError('GPU acquired by another compute job before native evaluation')
        if actual == {gpu_uuid}:
            return actual
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DeviceIdentityError('No verified simulator GPU appeared within the native boot deadline')
        stop.wait(min(5, remaining))
    raise RuntimeError('Controller stop requested')


def normalized_uuid(value):
    value = value.lower()
    return (value[4:] if value.startswith('gpu-') else value).replace('-', '')


def mapped_vulkan_index(devices, gpu_uuid):
    matches = [d['vulkan_index'] for d in devices if d['vendor_id'] == 0x10de
               and normalized_uuid(d['device_uuid']) == normalized_uuid(gpu_uuid)]
    if len(matches) != 1:
        raise DeviceIdentityError('Vulkan/NVIDIA device identity is missing or ambiguous')
    return matches[0]


def mesa_pci_selector(bus_id):
    match = re.fullmatch(r'([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', bus_id)
    if not match or int(match[1], 16) > 0xffff:
        raise DeviceIdentityError('Invalid NVIDIA PCI bus identity')
    return 'pci-{:04x}_{}_{}_{}'.format(int(match[1], 16), match[2].lower(), match[3].lower(), match[4])


def isolated_simulator_environment(gpu_uuid, environment, pci_rows):
    matches = [row.split(',')[1].strip() for row in pci_rows.splitlines()
               if len(row.split(',')) == 2 and normalized_uuid(row.split(',')[0].strip()) == normalized_uuid(gpu_uuid)]
    if len(matches) != 1:
        raise DeviceIdentityError('NVIDIA PCI identity is missing or ambiguous')
    isolated = dict(environment)
    isolated.pop('NODEVICE_SELECT', None)
    isolated.update(DRI_PRIME=mesa_pci_selector(matches[0]), MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE='1')
    return isolated


def isolated_vulkan_index(devices, gpu_uuid):
    index = mapped_vulkan_index(devices, gpu_uuid)
    if len(devices) != 1 or index != 0:
        raise DeviceIdentityError('Vulkan did not expose exactly the selected physical GPU')
    return index


def device_namespace_command(command, nvidia_nodes, drm_nodes):
    controls = {'/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools', '/dev/nvidia-modeset'}
    numbered = [path for path in nvidia_nodes if re.fullmatch(r'/dev/nvidia[0-9]+', path)]
    if (len(numbered) != 1 or '/dev/nvidiactl' not in nvidia_nodes
        or any(path not in controls and path not in numbered for path in nvidia_nodes)
        or not drm_nodes or any(not re.fullmatch(r'/dev/dri/(card[0-9]+|renderD[0-9]+)', path) for path in drm_nodes)):
        raise DeviceIdentityError('Invalid single-GPU namespace device list')
    prefix = ['bwrap', '--bind', '/', '/', '--proc', '/proc', '--dev', '/dev']
    for path in sorted(set(nvidia_nodes + drm_nodes)):
        prefix.extend(['--dev-bind', path, path])
    return prefix + ['--'] + command


def physical_gpu_namespace(gpu_uuid, simulator_env):
    nvidia_nodes = subprocess.check_output(['nvidia-container-cli', 'list', '--device=' + gpu_uuid], text=True).splitlines()
    pci = simulator_env['DRI_PRIME'][4:].replace('_', ':', 2).rsplit('_', 1)
    pci = pci[0] + '.' + pci[1]
    drm_nodes = ['/dev/dri/' + path.name for path in Path('/sys/class/drm').iterdir()
                 if re.fullmatch(r'card[0-9]+|renderD[0-9]+', path.name)
                 and (path / 'device').resolve().name.lower() == pci]
    return device_namespace_command([], nvidia_nodes, drm_nodes), nvidia_nodes + drm_nodes


def runtime_repair(protocol_file, repair_file):
    if repair_file is None:
        return None
    repair_path = Path(repair_file).resolve()
    value = json.loads(repair_path.read_text())
    if (value.get('original_protocol_sha256') != sha(protocol_file)
        or value.get('environment_repair_only') is not True
        or value.get('GPU_isolation') != 'nvidia_device_namespace_and_mesa_pci'
        or value.get('additional_route_attempts') != []):
        raise RuntimeError('Native environment repair does not match the frozen protocol/budget')
    if value['source_files'].get(str(Path(__file__).resolve())) != sha(__file__):
        raise RuntimeError('Running native controller is not bound by the repair')
    if any(path in value['source_files'] and value['source_files'][path] != digest
           for path, digest in value['protected_files'].items()):
        raise RuntimeError('Conflicting native repair artifact identities')
    for path, digest in dict(value['source_files'], **value['protected_files']).items():
        if sha(path) != digest:
            raise RuntimeError('Native environment repair source/protected artifact changed: ' + path)
    return dict(value, manifest_file=str(repair_path), manifest_sha256=sha(repair_path))


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


def finished_native_record(raw_file, log_file, expected_route):
    """Freeze a fully committed native outcome, including a scored zero.

    A native actor can remain alive during teardown after writing its final
    score. Resource checks after this boundary must not invalidate that score.
    Incomplete writes and environment failures still require the original guard.
    """
    try:
        raw = json.loads(raw_file.read_text())
        if (raw.get('entry_status') != 'Finished' or raw.get('eligible') is not True
                or raw.get('_checkpoint', {}).get('progress') != [1, 1]):
            return None
        return valid_record(raw_file, log_file, expected_route)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def wait_native_actor(actor, simulator_pid, gpu_uuid, raw_file, log_file,
                      expected_route, stop, stall_seconds):
    """Watch the owned episode; final scoring precedes post-score cleanup."""
    while actor.poll() is None and not stop.wait(5):
        completed = finished_native_record(raw_file, log_file, expected_route)
        if completed is not None:
            return completed
        _, busy = inventory()
        if busy[gpu_uuid] - group_pids(simulator_pid) - group_pids(actor.pid):
            # Scoring may finish while nvidia-smi is collecting its snapshot.
            completed = finished_native_record(raw_file, log_file, expected_route)
            if completed is not None:
                return completed
            raise RuntimeError('Foreign compute entered allocated GPU; own attempt stopped')
        if time.time() - log_file.stat().st_mtime > stall_seconds:
            raise RuntimeError('Owned native evaluation produced no progress for the frozen environment stall window')
    if stop.is_set():
        raise RuntimeError('Controller stop requested')
    return finished_native_record(raw_file, log_file, expected_route)


def same_native_outcome(before, after):
    """Cleanup may append logs, but cannot replace the committed score data."""
    metadata = {'raw_file', 'raw_sha256', 'log_file', 'log_sha256'}
    return ({key: value for key, value in before.items() if key not in metadata}
            == {key: value for key, value in after.items() if key not in metadata})


def recoverable_finished_record(raw_file, log_file, invalid_file, expected_route):
    """Read-only qualification of a score committed before a late abort.

    Callers must separately bind the original protocol/runtime and preserve all
    earlier aggregation and receipts. This rule never selects by score value.
    """
    try:
        invalid = json.loads(invalid_file.read_text())
        if (invalid.get('valid') is not False or invalid.get('score') is not None
                or invalid.get('error') != 'RuntimeError'
                or invalid.get('message') != 'Foreign compute entered allocated GPU; own attempt stopped'
                or max(raw_file.stat().st_mtime_ns, log_file.stat().st_mtime_ns)
                   >= invalid_file.stat().st_mtime_ns):
            return None
        return finished_native_record(raw_file, log_file, expected_route)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class Queue:
    def __init__(self, protocol_file, runtime_repair_file=None):
        self.path = Path(protocol_file).resolve()
        self.root = self.path.parent
        self.protocol = json.loads(self.path.read_text())
        self.protocol_sha = sha(self.path)
        self.repair = runtime_repair(self.path, runtime_repair_file)
        self.lock = (self.root / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.stop = threading.Event()
        self.records = {}
        self.state = dict(controller_pid=os.getpid(), run_id=self.protocol['run_id'],
            benchmark_scores=0, workers_max=1, phase='verifying frozen sources and inputs')
        if self.repair:
            self.state.update(runtime_repair_id=self.repair['repair_id'],
                runtime_repair_sha256=self.repair['manifest_sha256'])
        self.routes = ET.parse(self.protocol['routes_file']).getroot().findall('route')
        assert len(self.routes) == 100 and len({r.get('id') for r in self.routes}) == 100
        for name, digest in self.protocol['frozen_files'].items():
            if sha(name) != digest:
                raise RuntimeError('Frozen native input/source changed: ' + name)
        self.fingerprints = {name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                             for name in self.protocol['frozen_files']}
        if self.repair:
            self.fingerprints.update({name: (Path(name).stat().st_size, Path(name).stat().st_mtime_ns)
                for name in dict(self.repair['source_files'], **self.repair['protected_files'])})
        self.load_scores()

    def verify(self):
        if sha(self.path) != self.protocol_sha:
            raise RuntimeError('Frozen route protocol changed')
        if self.repair and sha(self.repair['manifest_file']) != self.repair['manifest_sha256']:
            raise RuntimeError('Bound native environment repair changed')
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
            env = dict(os.environ, **spec['environment'], CUDA_VISIBLE_DEVICES=gpu['uuid'])
            env.pop('SAVE_PATH', None)
            simulator_env = env
            simulator_prefix, simulator_nodes = [], []
            if self.repair:
                pci_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                    '--format=csv,noheader'], text=True)
                simulator_env = isolated_simulator_environment(gpu['uuid'], env, pci_rows)
                simulator_prefix, simulator_nodes = physical_gpu_namespace(gpu['uuid'], simulator_env)
            devices = json.loads(subprocess.check_output(simulator_prefix + [self.protocol['vulkan_identity_tool']],
                text=True, env=simulator_env))
            adapter = isolated_vulkan_index(devices, gpu['uuid']) if self.repair else mapped_vulkan_index(devices, gpu['uuid'])
            out.mkdir()
            xml = ET.Element('routes');xml.append(ET.fromstring(ET.tostring(route)))
            ET.ElementTree(xml).write(out / 'route.xml', encoding='utf-8', xml_declaration=True)
            atomic(out / 'runtime.json', dict(GPU=gpu, vulkan_adapter=adapter, vulkan_devices=devices,
                native_config=spec['config'], checkpoint=spec['checkpoint'], protocol_sha256=self.protocol_sha,
                simulator_environment={key: simulator_env[key] for key in
                    ['DRI_PRIME', 'MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'] if key in simulator_env},
                simulator_device_nodes=simulator_nodes,
                runtime_repair_sha256=self.repair['manifest_sha256'] if self.repair else None))
            simulator = actor = None
            try:
                with (out / 'simulator.log').open('x') as log:
                    simulator = subprocess.Popen(simulator_prefix + [self.protocol['carla_launcher'], '-RenderOffScreen',
                        '-graphicsadapter=' + str(adapter), '-carla-rpc-port=' + str(self.protocol['ports'][0]), '-nosound'],
                        cwd=self.protocol['carla_root'], env=simulator_env, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                self.state.update(phase='starting owned simulator', method=method, route=key,
                    simulator_pid=simulator.pid, GPU=gpu, attempt=str(out))
                self.save()
                if self.stop.wait(55) or simulator.poll() is not None:
                    raise RuntimeError('Simulator did not remain alive through boot')
                wait_simulator_device(simulator, gpu['uuid'], self.stop)
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
                    completed = wait_native_actor(actor, simulator.pid, gpu['uuid'],
                        out / 'results.json', out / 'evaluation.log',
                        'RouteScenario_' + route.get('id') + '_rep0', self.stop,
                        self.protocol['environment_stall_seconds'])
                # The native task is finished. Only the owned teardown remains;
                # flush both groups before hashing the immutable score/log.
                clean_group(actor)
                actor = None
                # Let the owned Unreal process flush its log before preserving hashes.
                clean_group(simulator)
                simulator = None
                record = valid_record(out / 'results.json', out / 'evaluation.log',
                    'RouteScenario_' + route.get('id') + '_rep0')
                if record is None:
                    raise RuntimeError('No valid native official result; inspect retained logs')
                if completed is None or not same_native_outcome(completed, record):
                    raise RuntimeError('Native final outcome changed during owned cleanup; inspect retained records')
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
        if receipt_file.exists():
            if json.loads(receipt_file.read_text()) != receipt:
                raise RuntimeError('Original GPU setup receipt changed')
        else:
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
    parser.add_argument('--runtime-repair')
    args = parser.parse_args()
    queue = Queue(args.protocol, args.runtime_repair)
    try:
        queue.run()
    except Exception as error:
        queue.state.update(phase='needs repair; original results retained', error=type(error).__name__, message=str(error))
        queue.save()
        raise
