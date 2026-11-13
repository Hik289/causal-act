"""Run disjoint 20-route validation and 100-route Bench2Drive 0.0.4 evaluation."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
import argparse
import fcntl
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("split", type=Path, help="Directory written by split.py")
parser.add_argument("output", type=Path, help="Run directory for scores and logs")
args = parser.parse_args()
protocol = Path(__file__).resolve().parent
routes_dir = args.split.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
split = json.loads((routes_dir / "split_manifest.json").read_text())["splits"]
assert len(set(split["validation20"]["route_ids"])) == 20
assert len(set(split["holdout100"]["route_ids"])) == 100
assert not set(split["validation20"]["route_ids"]) & set(split["holdout100"]["route_ids"])
for name, manifest in split.items():
    assert sha256((routes_dir / (name + ".xml")).read_bytes()).hexdigest() == manifest["xml_sha256"]

lock = (output / "queue.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status_path = output / "status.json"
candidate = {
    "model": os.environ["CIV_DRIVE_MODEL"],
    "traffic_manager_seed": 0,
    "decision_every_ticks": 20,
    "max_output_tokens": 2048,
    "files": {name: sha256((protocol / name).read_bytes()).hexdigest()
              for name in ("agent.py", "run_pair.sh", "run.py")},
    "evaluator_sha256": sha256((Path(os.environ["CAUSALACT_BENCH2DRIVE_DIR"]) /
                                "leaderboard/leaderboard/leaderboard_evaluator.py").read_bytes()).hexdigest(),
    "route_hashes": {name: manifest["xml_sha256"] for name, manifest in split.items()},
}

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def write(path, record):
    temp = path.with_suffix(".new")
    temp.write_text(json.dumps(record,indent=2)+"\n")
    temp.replace(path)

def save(status, phase, **details):
    write(status_path, {"status": status, "phase": phase, "worker_pid": os.getpid(),
                        "updated_utc": timestamp(), **details})

def simulator_port_is_free():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", int(os.environ.get("CARLA_PORT", "2000")))) != 0

def read_records(folder, mode):
    path = folder / ("results_" + mode + ".json")
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())["_checkpoint"]["records"]
    log = (folder / ("eval_" + mode + ".log")).read_text(errors="replace")
    bad = set()
    route = None
    for line in log.splitlines():
        found = re.search(r"Preparing (RouteScenario_\S+) \(repetition (\d+)\)",line)
        if found:
            route = found[1] + "_rep" + found[2]
        if re.search(r"LLM err:|VK_ERROR_OUT_OF_DEVICE_MEMORY|Traceback \(most recent call last\)",line):
            bad.add(route)
    if None in bad:
        return {}
    return {r["route_id"]: r for r in rows if r["route_id"] not in bad
            and (r["status"] in ("Perfect","Completed") or r["status"].startswith("Failed - "))
            and not re.search(r"crash|initialization|setup",r["status"],re.I)
            and math.isfinite(r["scores"]["score_composed"]) and r["scores"]["score_composed"] >= 0}

def execute(phase, routes, out, modes):
    out.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        ["bash", str(protocol/"run_pair.sh"), str(routes), str(out), *modes],
        env=env, cwd=str(protocol), stdin=subprocess.DEVNULL)
    save("running", phase, driver_pid=proc.pid, output=out.name)
    return proc.wait()

def metrics(rows):
    return {"driving_score": mean(r["scores"]["score_composed"] for r in rows),
            "successes": sum(r["status"] in ("Perfect","Completed") for r in rows),
            "red_lights": sum(len(r["infractions"].get("red_light",[])) for r in rows),
            "collisions": sum(len(v) for r in rows for k,v in r["infractions"].items()
                              if k.startswith("collisions_"))}

def evaluate(phase):
    routes = routes_dir / (phase + ".xml")
    route_ids = split[phase]["route_ids"]
    expected = {"RouteScenario_" + id_ + "_rep0" for id_ in route_ids}
    folder = output / phase
    execute(phase, routes, folder, ("base","causalact"))
    scores = {}
    for mode in ("base","causalact"):
        valid = {k:v for k,v in read_records(folder,mode).items() if k in expected}
        retry = folder / ("retry_" + mode)
        missing = expected - valid.keys()
        if missing:
            retry.mkdir(exist_ok=True)
            tree = ET.ElementTree(ET.Element("routes"))
            for route in ET.parse(routes).getroot().findall("route"):
                if "RouteScenario_" + route.get("id") + "_rep0" in missing:
                    tree.getroot().append(route)
            remaining = retry / "routes.xml"
            if not remaining.exists():
                tree.write(remaining, encoding="utf-8", xml_declaration=True)
            else:
                previous_ids = {route.get("id") for route in ET.parse(remaining).getroot()}
                assert {route.get("id") for route in tree.getroot()} <= previous_ids
            time.sleep(120)
            if simulator_port_is_free() and shutil.disk_usage(output).free >= 10 * 1024**3:
                execute(phase, remaining, retry, (mode,))
                valid.update({k:v for k,v in read_records(retry,mode).items() if k in missing})
        scores[mode] = valid
    matched = sorted(scores["base"].keys() & scores["causalact"].keys())
    report = {"phase":phase, "planned":len(expected), "matched":len(matched),
              "valid_by_arm":{m:len(r) for m,r in scores.items()},
              "invalid_or_missing_by_arm":{m:sorted(expected - r.keys()) for m,r in scores.items()}}
    if matched:
        report.update({m:metrics([r[k] for k in matched]) for m,r in scores.items()})
        report["driving_score_delta"] = mean(scores["causalact"][k]["scores"]["score_composed"] -
                                                scores["base"][k]["scores"]["score_composed"] for k in matched)
    write(folder / "pair_report.json",report)
    return report

config_path = output / "candidate_config.json"
if config_path.exists() and json.loads(config_path.read_text()) != candidate:
    parser.error("Use a new output directory when the model, code, or route split changes.")
write(config_path, candidate)
if not simulator_port_is_free():
    parser.error("Set CARLA_PORT to an available simulator port.")
env = dict(os.environ, CIV_DECIDE_EVERY=str(candidate["decision_every_ticks"]), CIV_LLM_TIMEOUT="100")
report = evaluate("validation20")
if report["matched"] != 20:
    save("needs attention", "validation20", reason="validation pairs incomplete", report=report)
elif report["driving_score_delta"] <= 0:
    save("validation did not improve", "validation20", report=report)
else:
    write(output/"frozen_config.json", {"config": candidate, "validation": report,
                                          "holdout": split["holdout100"],"frozen_utc":timestamp()})
    report = evaluate("holdout100")
    save("finished" if report["matched"] == 100 else "needs attention", "holdout100",report=report)
