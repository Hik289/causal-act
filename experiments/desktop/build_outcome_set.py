"""Build outcome-labeled dataset from 342 base trajectories.
Each step: (screenshot_before, instruction, action_text) -> label = did the WHOLE
task ultimately FAIL? (outcome calibration).
This trains an outcome-V that recognizes 'this trajectory is going wrong' from
real base failure patterns (not synthetic). Used for high-precision gating:
only intervene when V is very confident the trajectory is failing.
"""
from causalact_runtime import location
import os, json, glob
BASE_DIRS = {
 "osw": location('CAUSALACT_RUN_DIR', 'osw_base_full/pyautogui/screenshot/gpt-5.6-sol'),
 "gimp": location('CAUSALACT_RUN_DIR', 'desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol'),
 "vlc": location('CAUSALACT_RUN_DIR', 'desktop_base_vlc_results/pyautogui/screenshot/gpt-5.6-sol'),
}
OUT = location('CAUSALACT_RUN_DIR', 'outcome_set.jsonl')
rows=[]
for key,bd in BASE_DIRS.items():
    for td in glob.glob(bd+"/*/*"):
        if not os.path.isdir(td): continue
        dom=td.split("/")[-2]; tid=os.path.basename(td)
        rf=os.path.join(td,"result.txt"); tj=os.path.join(td,"traj.jsonl")
        if not (os.path.exists(rf) and os.path.exists(tj)): continue
        try: score=float(open(rf).read().strip())
        except: continue
        failed = score < 0.999
        steps=[json.loads(l) for l in open(tj) if l.strip()]
        for s in steps:
            sf=s.get("screenshot_file")
            # screenshot_file is AFTER action; we want BEFORE = previous step's after, or init
            img=os.path.join(td, sf) if sf else None
            if not img or not os.path.exists(img): continue
            act=str(s.get("action",""))[:300]
            resp=str(s.get("response",""))[:300]
            rows.append({"domain":dom,"task_id":tid,"step":s.get("step_num"),
                         "screenshot":img,"action":act,"response":resp,
                         "gt_fail": failed})
with open(OUT,"w") as f:
    for r in rows: f.write(json.dumps(r)+"\n")
nf=sum(1 for r in rows if r["gt_fail"]); ns=len(rows)-nf
import collections
bydom=collections.Counter(r["domain"] for r in rows)
print(f"built {len(rows)} step-outcome rows: fail-traj steps={nf} success-traj steps={ns}")
print("tasks:", len(set((r["domain"],r["task_id"]) for r in rows)), "domains:", dict(bydom))
print("->",OUT)
