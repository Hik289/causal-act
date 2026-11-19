"""Use explicitly idle GPUs without preempting the immutable original study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def admit(original_jobs, side_jobs, active_followups, gpu, free_mib, disk_bytes, active_gpu_pids=()):
    if active_gpu_pids:
        return False
    if any(job.get('status') == 'queued' for job in original_jobs.values()):
        return False
    if any(job.get('status') == 'running' and job.get('gpu') == gpu
           for job in list(original_jobs.values()) + list(side_jobs.values())):
        return False
    running = sum(job.get('status') == 'running'
                  for job in list(original_jobs.values()) + list(side_jobs.values()))
    return (running + active_followups < 4 and free_mib >= 36000
            and disk_bytes >= 10 * 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = Path(args.run)
    p = json.loads((run / 'protocol.json').read_text())
    amendment = json.loads((run / 'compute_scheduling_amendment_20261119.json').read_text())
    assert amendment['model_training_and_gate_changed'] is False
    lock = (run / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'phase': 'using explicitly idle GPUs; original study has priority',
             'pid': os.getpid(), 'jobs': {}, 'scheduling_amendment': amendment}

    def save():
        path = run / 'status.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(path)

    def allowed(gpu, active_count):
        old = json.loads((Path(p['original_run']) / 'status.json').read_text())
        side_path = Path(amendment['side_status'])
        side = json.loads(side_path.read_text())['jobs'] if side_path.exists() else {}
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        uuid = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                    '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
        owners = [line.split(',')[1].strip() for line in processes.splitlines()
                  if line.split(',')[0].strip() == uuid]
        state.setdefault('waiting_on_compute_pids', {})[str(gpu)] = owners
        return admit(old['jobs'], side, active_count, gpu, free, shutil.disk_usage(run).free, owners)

    def launch(method, gpu):
        stream = (run / (method + '.log')).open('a')
        child = subprocess.Popen([p['python'], '-u', str(run / 'code/balanced_followup_worker.py'),
                '--run', str(run), '--method', method], stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                         TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4'))
        state['jobs'][method] = {'status': 'running', 'pid': child.pid, 'gpu': gpu}
        save()
        return method, child, stream

    assert (run / 'data/dataset_complete.json').exists()
    # This is a scheduling amendment before any training or score output.
    pending = list(p['methods'])
    assert all(not (run / method).exists() for method in pending)
    active = {}
    save()
    while active or pending:
        for gpu in [0, 1]:
            if gpu not in active and pending and allowed(gpu, len(active)):
                active[gpu] = launch(pending.pop(0), gpu)
        for gpu, (method, child, stream) in list(active.items()):
            if child.poll() is None:
                continue
            stream.close()
            state['jobs'][method].update(status='finished' if child.returncode == 0 else 'invalid; needs repair',
                                         exit_code=child.returncode, pid=None)
            del active[gpu]
            save()
        if active or pending:
            time.sleep(15)
    passed = [method for method in p['methods'] if (run / method / 'frozen_config.json').exists()
              and json.loads((run / method / 'frozen_config.json').read_text())['passed']]
    if passed:
        while not allowed(0, 0):
            time.sleep(20)
        method, child, stream = launch('base', 0)
        code = child.wait()
        stream.close()
        state['jobs']['base'].update(status='finished' if code == 0 else 'invalid; needs repair',
                                     exit_code=code, pid=None)
    state.update(phase='finished' if all(j['status'] == 'finished' for j in state['jobs'].values())
                 else 'needs repair', validation_passed=passed)
    save()


if __name__ == '__main__':
    main()
