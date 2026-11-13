"""Run the fixed OSWorld Base and replay arms and report matched outcomes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="Run directory for logs and scores")
parser.add_argument("--protocol", choices=("calibration", "timing"), default="timing")
parser.add_argument("--tasks", type=Path, help="Override the bundled task manifest")
parser.add_argument("--verifier-url", default="http://127.0.0.1:18833/pfail")
parser.add_argument("--checkpoint", type=Path, required=True, help="Model weights served by the verifier")
args = parser.parse_args()

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
lock = (out / "run.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
task_file = (args.tasks or HERE / f"{args.protocol}_tasks.json").resolve()
tasks = json.loads(task_file.read_text())
model = os.environ["CAUSALACT_BASE_MODEL"]
with args.checkpoint.open("rb") as weights:
    digest = hashlib.sha256()
    for chunk in iter(lambda: weights.read(1024 * 1024), b""):
        digest.update(chunk)

config = {
    "protocol": args.protocol, "model": model, "max_steps": 12, "gate_threshold": 0.97,
    "verifier_streak": 2, "verifier_url": args.verifier_url,
    "tasks_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
    "checkpoint_sha256": digest.hexdigest(),
    "code_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                    for name in ("run_protocol.py", "arm.py", "calibrated_gate.py")},
}

def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)

saved_config = out / "protocol.json"
if saved_config.exists() and json.loads(saved_config.read_text()) != config:
    parser.error("Use a new output directory when changing the experiment configuration.")
write_json(saved_config, config)

def result_root(arm):
    return out / arm / "pyautogui" / "screenshot" / model

def scores(arm):
    results = {}
    for domain, ids in tasks.items():
        for task in ids:
            folder = result_root(arm) / domain / task
            try:
                value = float((folder / "result.txt").read_text())
            except (OSError, ValueError):
                continue
            if math.isfinite(value) and (folder / "traj.jsonl").is_file():
                results[domain, task] = value == 1
    return results

def run_arm(arm, eligible):
    manifest = out / f"{arm}_tasks.json"
    env = dict(os.environ, CAUSALACT_REPLAY_OUTPUT=str(out), CIV_DOCKER_DYNAMIC_PORTS="1",
               CIV_RETIMING_ARM=arm, CIV_BASE_ROOT=str(result_root("base")),
               CIV_VURL=args.verifier_url, CIV_VGATE="1", CIV_VTAU="0.9", CIV_VSTREAK="2",
               CIV_GATE_POLICY="incumbent" if arm == "incumbent" else "filtered")
    cmd = [sys.executable, "-u", str(HERE / "arm.py"),
           "--provider_name", "docker", "--headless", "--observation_type", "screenshot",
           "--action_space", "pyautogui", "--model", model, "--domain", "all",
           "--max_steps", "12", "--test_all_meta_path", str(manifest), "--result_dir", str(out / arm)]
    for attempt in range(2):
        scored = scores(arm)
        missing = {domain: [task for task in ids if (domain, task) not in scored]
                   for domain, ids in eligible.items()}
        if not any(missing.values()):
            return
        if attempt:
            time.sleep(120)
        write_json(manifest, missing)
        with (out / f"{arm}.log").open("a") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

run_arm("base", tasks)
base = scores("base")
eligible = {domain: [task for task in ids if (domain, task) in base]
            for domain, ids in tasks.items()}
arms = ("incumbent", "filtered") if args.protocol == "calibration" else ("paced",)
results = {"base": base}
for arm in arms:
    run_arm(arm, eligible)
    results[arm] = scores(arm)
matched = set.intersection(*(set(s) for s in results.values()))
report = {
    "planned": sum(map(len, tasks.values())), "matched": len(matched),
    "scored_by_arm": {arm: len(s) for arm, s in results.items()},
    "success_by_arm": {arm: sum(s[k] for k in matched) for arm, s in results.items()},
    "rescues": {arm: sum(not base[k] and results[arm][k] for k in matched) for arm in arms},
    "regressions": {arm: sum(base[k] and not results[arm][k] for k in matched) for arm in arms},
}
write_json(out / "pair_report.json", report)
print(json.dumps(report, indent=2))
