"""Run gimp with loop-breaker refiner installed. CIV_LOOPBREAK=1 -> B+A, =0 -> B.
Usage: python run_gimp_lb.py --domain gimp --model gpt-5.6-sol ... (same as run.py)
"""
from causalact_runtime import location
import os, sys
import mm_agents.agent as agent_module
import loopbreak_v3 as loopbreak_patch

AUDIT = os.environ.get("CIV_AUDIT", location('CAUSALACT_RUN_DIR', 'gimp_lb_audit.log'))
state = loopbreak_patch.install(agent_module, AUDIT)
print(f"[loopbreak] installed, CIV_LOOPBREAK={os.environ.get('CIV_LOOPBREAK','0')}, audit={AUDIT}", flush=True)

# now run run.py's flow
import run as runmod
args = runmod.config()
import json
with open(args.test_all_meta_path, "r") as f:
    test_all_meta = json.load(f)
if args.domain != "all":
    test_all_meta = {args.domain: test_all_meta[args.domain]} if args.domain in test_all_meta else test_all_meta
test_file_list = runmod.get_unfinished(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)
runmod.test(args, test_file_list)
print(f"[loopbreak] DONE overrides={state.get("ov",0)} steps={state.get("steps",0)}", flush=True)
