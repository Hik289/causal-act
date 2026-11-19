"""Wait behind the original immutable study, then execute one fixed followup."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


run = Path(__file__).resolve().parents[1]
p = json.loads((run / 'protocol.json').read_text())
lock = (run / 'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = {'phase': 'waiting for original study and verified new data', 'pid': os.getpid(), 'jobs': {}}


def save():
    temporary = run / 'status.json.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(run / 'status.json')


save()
while True:
    old = json.loads((Path(p['original_run']) / 'status.json').read_text())
    if old['phase'] == 'needs repair':
        state['phase'] = 'waiting; original study needs repair'
    if old['phase'] == 'finished' and (run / 'data/dataset_complete.json').exists():
        break
    save()
    time.sleep(20)
active = {}
pending = ['static_cf', 'action_refiner']
state['phase'] = 'training one pre-registered class-prior adjustment'
while active or pending:
    for gpu in [0, 1]:
        if gpu in active or not pending:
            continue
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 36000 or shutil.disk_usage(run).free < 10 * 2**30:
            continue
        method = pending.pop(0)
        stream = (run / (method + '.log')).open('x')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
        child = subprocess.Popen([p['python'], '-u', str(run / 'code/balanced_followup_worker.py'),
                '--run', str(run), '--method', method], env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
        active[gpu] = (method, child, stream)
        state['jobs'][method] = dict(status='running', pid=child.pid, gpu=gpu)
        save()
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
passed = [method for method in ['static_cf', 'action_refiner'] if
          (run / method / 'frozen_config.json').exists() and
          json.loads((run / method / 'frozen_config.json').read_text())['passed']]
if passed:
    while int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip()) < 36000:
        time.sleep(20)
    with (run / 'base.log').open('x') as stream:
        child = subprocess.Popen([p['python'], '-u', str(run / 'code/balanced_followup_worker.py'),
                '--run', str(run), '--method', 'base'], stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4'))
        state['jobs']['base'] = dict(status='running', pid=child.pid)
        save()
        code = child.wait()
        state['jobs']['base'].update(status='finished' if code == 0 else 'invalid; needs repair', pid=None)
state.update(phase='finished' if all(x['status'] == 'finished' for x in state['jobs'].values())
             else 'needs repair', validation_passed=passed)
save()
