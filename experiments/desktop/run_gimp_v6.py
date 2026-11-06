"""v4 trajectory-shared passthrough refiner runner.
Replays base responses verbatim until the frozen-screen gate triggers, then diverges.
Sets agent._civ_task so replay loads the correct base trajectory & gate state resets per task.
"""
from causalact_runtime import location
import os, json
import mm_agents.agent as agent_module
import conj_refiner as patch

AUDIT = os.environ.get("CIV_AUDIT", location('CAUSALACT_RUN_DIR', 'gimp_v6_audit.log'))
state = patch.install(agent_module, AUDIT)
print(f"[v4] installed CIV_REPLAY={os.environ.get('CIV_REPLAY','1')} STREAK={os.environ.get('CIV_STREAK','4')} audit={AUDIT}", flush=True)

import lib_run_single
_orig_rse = lib_run_single.run_single_example
def _rse(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    try:
        agent._civ_task = example.get("id")
    except Exception:
        pass
    return _orig_rse(agent, env, example, max_steps, instruction, args, example_result_dir, scores)
lib_run_single.run_single_example = _rse

import run as runmod
args = runmod.config()
with open(args.test_all_meta_path, "r") as f:
    test_all_meta = json.load(f)
if args.domain != "all":
    test_all_meta = {args.domain: test_all_meta[args.domain]} if args.domain in test_all_meta else test_all_meta
test_file_list = runmod.get_unfinished(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)
runmod.test(args, test_file_list)
print(f"[v4] DONE overrides={state.get('ov',0)} steps={state.get('steps',0)}", flush=True)
