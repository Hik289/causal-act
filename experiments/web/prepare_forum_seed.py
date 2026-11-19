"""Prepare an isolated, cleanly stopped forum seed; never touch a live service."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request


def command(*args, check=True, timeout=60):
    return subprocess.run(list(args), capture_output=True, text=True,
                          check=check, timeout=timeout)


def verify_source(info, storage):
    if info.get('Mounts') or info['Config'].get('Volumes'):
        raise RuntimeError('Source needs a separately reviewed volume snapshot')
    if storage['Driver'] != 'overlay2':
        raise RuntimeError('Preparation requires copy-on-write overlay2 storage')


def owned_container(name, owner):
    value = command('docker', 'inspect', name, check=False)
    if value.returncode:
        return None
    info = json.loads(value.stdout)[0]
    if info['Config'].get('Labels', {}).get('causalact.owner') != owner:
        raise RuntimeError('Refusing to operate on a container owned by another run')
    return info


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    storage = json.loads(command('docker', 'info', '--format', '{{json .}}').stdout)
    source = json.loads(command('docker', 'image', 'inspect', args.source_image).stdout)[0]
    verify_source(source, storage)
    if command('docker', 'inspect', args.name, check=False).returncode == 0:
        raise RuntimeError('Preparation container already exists; inspect it before continuing')
    if command('docker', 'image', 'inspect', args.seed_tag, check=False).returncode == 0:
        raise RuntimeError('Seed tag already exists; do not replace it')
    root = storage['DockerRootDir']
    def disk():
        available = shutil.disk_usage(root).free / 1024**3
        if available < args.disk_floor_gib:
            raise RuntimeError('Available Docker disk is below the preparation floor')
        return available
    state = dict(kind='environment preparation, no benchmark scores',
                 benchmark_scores=0, source_image=source['Id'], owner=args.owner,
                 name=args.name, seed_tag=args.seed_tag, http_history=[], ready=False,
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 disk_free_before_gib=disk(), http_probe_mutates_visit_state=True,
                 protocol_note='Use this same seed for every new paired control; historical controls remain separate')
    def save():
        temporary = out / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(out / 'status.json')
    created = False
    try:
        state['container_id'] = command('docker', 'run', '-d', '--name', args.name,
            '--label', 'causalact.owner=' + args.owner, '--cpus', '4', '--memory', '16g',
            '-p', '127.0.0.1:' + str(args.port) + ':80', source['Id']).stdout.strip()
        created = True
        save()
        start = time.monotonic()
        while time.monotonic() - start < args.startup_timeout:
            disk()
            info = owned_container(args.name, args.owner)
            if not info or not info['State']['Running'] or info['State']['OOMKilled']:
                raise RuntimeError('Owned preparation container stopped or exhausted memory')
            observation = dict(elapsed_s=round(time.monotonic() - start))
            try:
                with urllib.request.urlopen('http://127.0.0.1:' + str(args.port) + '/f/food', timeout=8) as response:
                    html = response.read()
                    observation.update(status=response.status, bytes=len(html))
                    (out / 'last_response.html').write_bytes(html)
                    if response.status == 200 and b'food' in html.lower():
                        state['ready'] = True
            except urllib.error.HTTPError as exc:
                observation['status'] = exc.code
            except (OSError, TimeoutError) as exc:
                observation['error'] = type(exc).__name__
            state['http_history'].append(observation)
            save()
            if state['ready']:
                break
            time.sleep(3)
        if not state['ready']:
            raise RuntimeError('Forum did not become ready within the recorded startup budget')
        # Wait for PostgreSQL's own readiness before requesting its clean stop.
        check = command('docker', 'exec', args.name, 'pg_isready')
        (out / 'pg_isready.log').write_text(check.stdout + check.stderr)
        stopped = command('docker', 'exec', args.name, 'supervisorctl', 'stop', 'postgres', timeout=120)
        (out / 'postgres_stop.log').write_text(stopped.stdout + stopped.stderr)
        status = command('docker', 'exec', args.name, 'supervisorctl', 'status', 'postgres', check=False)
        if 'STOPPED' not in status.stdout:
            raise RuntimeError('PostgreSQL has not stopped cleanly; no seed will be published')
        state['postgres_supervisor_state'] = status.stdout.strip()
        command('docker', 'stop', '--time', '30', args.name, timeout=60)
        final = owned_container(args.name, args.owner)
        if final['State']['Running'] or final['State']['OOMKilled']:
            raise RuntimeError('Cannot commit an uncleanly stopped preparation container')
        logs = command('docker', 'logs', args.name, check=False)
        (out / 'docker_final.log').write_text(logs.stdout + logs.stderr)
        if 'database system is shut down' not in logs.stdout + logs.stderr:
            raise RuntimeError('Missing clean PostgreSQL shutdown evidence; no seed will be published')
        state['clean_shutdown_verified'] = True
        state['seed_image'] = command('docker', 'commit',
            '--change', 'LABEL causalact.owner=' + args.owner,
            '--change', 'LABEL causalact.parent_image=' + source['Id'],
            args.name, args.seed_tag, timeout=180).stdout.strip()
        state.update(status='clean_seed_created', disk_free_after_gib=disk())
        save()
    except Exception as exc:
        state.update(status='preparation_failed', error=type(exc).__name__, message=str(exc))
        save()
        raise
    finally:
        if created and owned_container(args.name, args.owner):
            logs = command('docker', 'logs', args.name, check=False)
            (out / 'docker_final.log').write_text(logs.stdout + logs.stderr)
            info = owned_container(args.name, args.owner)
            (out / 'container_final.json').write_text(json.dumps(info, indent=2) + '\n')
            command('docker', 'rm', '-f', args.name)
            state['owned_container_removed'] = True
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-image', required=True)
    parser.add_argument('--seed-tag', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=9961)
    parser.add_argument('--startup-timeout', type=int, default=600)
    parser.add_argument('--disk-floor-gib', type=float, default=10)
    prepare(parser.parse_args())
