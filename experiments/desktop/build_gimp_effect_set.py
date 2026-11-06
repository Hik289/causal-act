"""Build gimp action-effect causal dataset from base trajectories.
Targets diagnosed failure mode: base misreads whether a click took effect
(menu opened? dialog appeared? image changed?) -> loops on same action.

V task: given (prev screenshot, action taken, next screenshot) -> did the action
take EFFECT (screen changed meaningfully) ? yes/no.
Positive-effect pairs (screen changed) + no-effect pairs (screen identical/near-identical,
= the misread/looping case). forum=task-disjoint handled by caller.
"""
from causalact_runtime import location
import os, sys, json, glob, hashlib, re
from PIL import Image, ImageChops
import numpy as np

BASE = location('CAUSALACT_RUN_DIR', 'desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol/gimp')
OUT = location('CAUSALACT_RUN_DIR', 'gimp_effect_set.jsonl')
IMGDIR = location('CAUSALACT_RUN_DIR', 'gimp_effect_imgs')
os.makedirs(IMGDIR, exist_ok=True)

def img_diff_frac(a, b):
    """fraction of pixels changed between two screenshots"""
    try:
        ia = Image.open(a).convert("RGB"); ib = Image.open(b).convert("RGB")
        if ia.size != ib.size: ib = ib.resize(ia.size)
        diff = np.asarray(ImageChops.difference(ia, ib))
        changed = (diff.sum(axis=2) > 20).mean()  # frac pixels with notable change
        return float(changed)
    except Exception as e:
        return -1.0

rows = []
task_dirs = sorted([d for d in glob.glob(BASE + "/*") if os.path.isdir(d)])
for td in task_dirs:
    tid = os.path.basename(td)
    steps = sorted(glob.glob(td + "/step_*.png"),
                   key=lambda p: int(re.search(r"step_(\d+)_", p).group(1)))
    # parse actions from runtime.log if present
    actions = []
    rl = os.path.join(td, "runtime.log")
    if os.path.exists(rl):
        for line in open(rl, errors="ignore"):
            m = re.search(r"(pyautogui\.\w+\([^\)]*\))", line)
            if m: actions.append(m.group(1))
    for i in range(len(steps) - 1):
        prev, nxt = steps[i], steps[i+1]
        frac = img_diff_frac(prev, nxt)
        if frac < 0: continue
        took_effect = frac > 0.005  # >0.5% pixels changed = effect
        act = actions[i] if i < len(actions) else "unknown_action"
        rows.append({
            "task_id": tid,
            "step": i+1,
            "prev_screenshot": prev,
            "next_screenshot": nxt,
            "action": act,
            "diff_frac": round(frac, 5),
            "gt_effect": took_effect,  # V learns: did action take effect?
        })

with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")

eff = sum(1 for r in rows if r["gt_effect"]); noeff = len(rows) - eff
print(f"built {len(rows)} action-effect pairs from {len(task_dirs)} gimp tasks")
print(f"took_effect(screen changed)={eff}  no_effect(misread/loop case)={noeff}")
print(f"-> {OUT}")
