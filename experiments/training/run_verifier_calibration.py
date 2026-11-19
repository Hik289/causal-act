"""Run the registered inference-only followup on one explicitly free GPU."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


run = Path(__file__).resolve().parents[1]
protocol = json.loads((run / 'protocol.json').read_text())
lock = (run / 'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = {'phase': 'waiting for explicitly free GPU', 'pid': os.getpid(), 'jobs': {}}


def save():
    p = run / 'status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(p)


save()
for method in protocol['methods']:
    finished = run / method / 'status.json'
    if finished.exists() and json.loads(finished.read_text())['phase'].startswith('finished'):
        state['jobs'][method] = {'status': 'finished; existing outputs preserved'}
        save()
        continue
    gpu = protocol['gpu']
    while True:
        old = json.loads((Path(protocol['original_run']) / 'status.json').read_text())
        occupied = any(job.get('status') == 'running' and job.get('gpu') == gpu
                       for job in old['jobs'].values())
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
                   '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        active = sum(job.get('status') == 'running' for job in old['jobs'].values())
        if not occupied and active < 4 and free >= 36000 and shutil.disk_usage(run).free >= 10 * 2**30:
            break
        save()
        time.sleep(20)
    with (run / (method + '.log')).open('a') as stream:
        child = subprocess.Popen([protocol['python'], '-u', str(run / 'code/verifier_calibration_followup.py'),
                                  '--run', str(run), '--method', method],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                         TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4'))
        state.update(phase='validation calibration and gated paired holdout')
        state['jobs'][method] = {'status': 'running', 'pid': child.pid, 'gpu': gpu}
        save()
        code = child.wait()
    state['jobs'][method].update(status='finished' if code == 0 else 'invalid; needs repair',
                                 exit_code=code, pid=None)
    save()
state['phase'] = ('finished' if all(j['status'].startswith('finished') for j in state['jobs'].values())
                  else 'needs repair')
save()
