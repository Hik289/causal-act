"""Select a one-shot override filter on the existing OSWorld validation split."""
from collections import defaultdict
from pathlib import Path
import json
import math
import re
import sys
import os

root, out = map(Path, sys.argv[1:])
model = os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol")
base = root / "osw_base_full/pyautogui/screenshot" / model
refiner = root / "osw_v5full/pyautogui/screenshot" / model
out.mkdir(parents=True, exist_ok=True)
shards = Path(os.environ.get("CAUSALACT_SPLIT_DIR", Path(__file__).resolve().parents[1] / "shards"))
validation = json.loads((shards / "v5_val.json").read_text())
holdout = json.loads((shards / "v5_test.json").read_text())
domains = "chrome os vs_code libreoffice_writer thunderbird libreoffice_calc libreoffice_impress multi_apps".split()

def task_keys(mapping):
    return {(app, task) for app, ids in mapping.items() for task in ids}

assert not task_keys(validation) & task_keys(holdout)
assert set(validation) == set(holdout)

# Restrict filtering to the first logged intervention after a pure replay.
audits = defaultdict(set)
pattern = re.compile(r"^task=([^ ]+) step=(\d+) pf=([^ ]+) mode=(\S+)")
for file in sorted(root.glob("v5full_audit_*.log")):
    history = []
    last_task = None
    last_step = None
    for line in file.read_text(errors="replace").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        task, step, probability, mode = m.groups()
        global_step = int(step)
        # The audit logs use a worker-wide counter; reset at task boundaries.
        if task != last_task or (last_step is not None and global_step != last_step + 1):
            history = []
        last_task, last_step = task, global_step
        step = len(history) + 1
        if mode == "TRIGGER" and step >= 2:
            if all(record[0] == "replay" for record in history):
                try:
                    score = min(float(history[-1][1]), float(probability))
                except ValueError:
                    score = -1
                if score >= .90:
                    audits[task].add((step, round(score, 7)))
        history.append((mode, probability))

def read_score(folder):
    try:
        value = float((folder/"result.txt").read_text())
        return int(value == 1) if math.isfinite(value) else None
    except (OSError, ValueError):
        return None

def repeated_action(folder, step):
    if step < 3:
        return False
    try:
        rows = [json.loads(line) for line in (folder/"traj.jsonl").read_text().splitlines()]
        actions = {int(r["step_num"]): r["action"] for r in rows if "action" in r}
        a,b = actions[step-2],actions[step-1]
        return bool(a) and a == b
    except (OSError, ValueError, TypeError, KeyError):
        return False

def evaluate_rows(keys):
    rows = []
    for app, task in sorted(keys):
        b, v = read_score(base/app/task), read_score(refiner/app/task)
        if b is None or v is None:
            continue
        options = audits[task]
        trigger = next(iter(options)) if len(options)==1 else None
        rows.append({"domain": app, "task": task, "base": b, "refiner": v, "trigger": trigger,
                     "repeated": repeated_action(base/app/task, trigger[0]) if trigger else False,
                     "ambiguous_audit": len(options)>1})
    return rows

def allowed(rule, row):
    if not row["trigger"]:
        return False
    step, confidence = row["trigger"]
    return {"v5_all": True, "confidence_095": confidence >= .95,
            "confidence_097": confidence >= .97, "step_ge_4": step >= 4,
            "step_ge_6": step >= 6, "repeat_last_two": row["repeated"],
            "never": False}[rule]

def metrics(rows, rule):
    accepted = [r for r in rows if allowed(rule,r)]
    rescued = sum(r["base"]==0 and r["refiner"]==1 for r in accepted)
    harmed = sum(r["base"]==1 and r["refiner"]==0 for r in accepted)
    return dict(tasks=len(rows), base=sum(r["base"] for r in rows), triggered=len(accepted),
                rescued=rescued, harmed=harmed, net=rescued-harmed,
                reconstructed=sum(r["base"] for r in rows)+rescued-harmed)

def observed(rows):
    return {"n": len(rows), "base": sum(r["base"] for r in rows), "refiner": sum(r["refiner"] for r in rows),
            "audited": sum(bool(r["trigger"]) for r in rows),
            "ambiguous": sum(r["ambiguous_audit"] for r in rows)}

candidates = ("v5_all","confidence_095","confidence_097","step_ge_4","step_ge_6","repeat_last_two","never")
val_rows = evaluate_rows(task_keys(validation))
val = {rule:metrics(val_rows,rule) for rule in candidates}
# Penalize a broken previously successful task twice as much as a rescue.
# If scores tie, keep the incumbent so a filter must demonstrate a validation advantage.
def objective(rule):
    m=val[rule]
    return (m["rescued"]-2*m["harmed"], rule=="v5_all", -m["harmed"], -m["triggered"])
chosen = max(candidates,key=objective)
selection = {"rule":chosen, "objective":"rescued - 2* harmed", "validation":val,
             "validation_observed":observed(val_rows)}
(out/"frozen_policy.json").write_text(json.dumps(selection,indent=2)+"\n")
print("VALIDATION:",json.dumps(selection),flush=True)

# Load scored holdout outcomes only after the filter is written.
test_rows=evaluate_rows(task_keys(holdout))
other={(app,p.name) for app in domains if app not in validation for p in (base/app).iterdir() if p.is_dir()}
other_rows=evaluate_rows(other)
report={"selection":selection}
for name,rows in (("holdout",test_rows),("other_domains",other_rows)):
    report[name]={"observed":observed(rows),
                  "incumbent":metrics(rows,"v5_all"), "selected":metrics(rows,chosen)}
print("HELD-OUT:",json.dumps({name:report[name] for name in ("holdout","other_domains")}),flush=True)
(out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
