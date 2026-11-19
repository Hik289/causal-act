"""Wait for an old queue's active cases, then resume missing scores once."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from forum_pair_queue import Queue, atomic


def live_command(pid):
    try:
        return (Path('/proc') / str(pid) / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
    except OSError:
        return ''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--runtime-repair', required=True)
    parser.add_argument('--old-controller', type=int, required=True)
    args = parser.parse_args()
    protocol_file = Path(args.protocol).resolve()
    root = protocol_file.parent
    protocol = json.loads(protocol_file.read_text())
    repair_file = Path(args.runtime_repair).resolve()
    repair = json.loads(repair_file.read_text())
    folder = repair_file.parent
    lock = (folder / 'handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = folder / 'handoff_status.json'
    if status_file.exists() and json.loads(status_file.read_text()).get('status') == 'started':
        print('A handoff was already started; inspect its actual controller instead of relaunching')
        return
    while True:
        protected = json.loads((root / 'protected_scores.json').read_text())
        for name, digest in protected.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('A valid score changed during handoff')
        actors = []
        for process in Path('/proc').glob('[0-9]*'):
            command = live_command(process.name)
            if 'forum_case.py' in command and protocol['run_id'] in command:
                actors.append(int(process.name))
        old = live_command(args.old_controller)
        if old and ('forum_pair_queue.py' not in old or protocol['run_id'] not in old):
            raise RuntimeError('Old PID belongs to another process; do not act on it')
        if old or actors:
            atomic(status_file, {'status': 'waiting for original active cases; no duplicate worker',
                'old_controller': args.old_controller, 'active_case_pids': actors,
                'protected_valid_scores': len(protected), 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            time.sleep(15)
            continue
        # Queue's original lock is the final admission check. Its preflight
        # also persists the one-time repaired-attempt allowance before launch.
        preflight = Queue(protocol_file, repair_file)
        preflight.lock.close()
        with (folder / 'controller.log').open('x') as log:
            process = subprocess.Popen([protocol['python'], '-u',
                str(Path(__file__).with_name('forum_pair_queue.py')),
                '--protocol', str(protocol_file), '--runtime-repair', str(repair_file)],
                cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError('Repaired controller exited; inspect its log before another launch')
        atomic(status_file, {'status': 'started', 'controller_pid': process.pid,
            'runtime_repair_id': repair['repair_id'], 'old_controller': args.old_controller,
            'protected_valid_scores': len(protected),
            'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        (root / 'controller.pid').write_text(str(process.pid) + '\n')
        print(json.dumps(json.loads(status_file.read_text())))
        return


if __name__ == '__main__':
    main()
