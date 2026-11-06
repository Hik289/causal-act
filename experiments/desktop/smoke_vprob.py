from causalact_runtime import location
import v_prob_scorer as V, glob, json, os, random
rows=[json.loads(l) for l in open(location('CAUSALACT_RUN_DIR', 'outcome_set.jsonl'))]
random.seed(1); random.shuffle(rows)
tested=0
for r in rows:
    sp=r["screenshot"]
    if not sp.endswith(".png"):
        tid=r.get("task_id",""); step=str(r.get("step",""))
        cand=glob.glob(os.path.join(sp,"*",tid,"*.png")) or glob.glob(os.path.join(sp,tid+"*","*.png"))
        # match step
        cand=[c for c in cand if ("step_"+step+"_") in os.path.basename(c)] or cand
        sp=cand[0] if cand else None
    if sp and os.path.exists(sp):
        p=V.p_fail(sp, r["action"])
        gt=r["gt_fail"]
        print("gt_fail="+str(gt)+" P_fail="+("%.3f"%p))
        tested+=1
        if tested>=8: break
print("tested",tested)
