"""Run one OSWorld arm, pacing only the actions shared with the Base trajectory."""
import functools
from datetime import datetime
import json
import os
from pathlib import Path
import time

import lib_run_single
import mm_agents.agent as agent_module

run = Path(os.environ["CAUSALACT_REPLAY_OUTPUT"])
arm = os.environ["CIV_RETIMING_ARM"]
agent_predict = agent_module.PromptAgent.predict

@functools.wraps(agent_predict)
def checked_predict(self, *args, **kwargs):
    response, actions = agent_predict(self, *args, **kwargs)
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("Empty planning response after transport retries")
    return response, actions

agent_module.PromptAgent.predict = checked_predict
state = None
if arm != "base":
    import calibrated_gate
    state = calibrated_gate.install(agent_module, str(run/f"{arm}.audit.log"))

original = lib_run_single.run_single_example

def run_example(agent, env, example, max_steps, instruction, args, result_dir, scores):
    task = example["id"]
    agent._civ_task = task
    if arm != "paced":
        return original(agent, env, example, max_steps, instruction, args, result_dir, scores)

    path = Path(os.environ["CIV_BASE_ROOT"])/Path(result_dir).parent.name/task/"traj.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    actions = [(r["action"],datetime.strptime(r["action_timestamp"],"%Y%m%d@%H%M%S%f"))
               for r in rows if "action" in r and "action_timestamp" in r]
    if not actions:
        raise RuntimeError("Base trajectory has no action timestamps")
    step = env.step
    replay_start = None
    idx = 0
    matching = True

    def timed_step(action, pause):
        nonlocal replay_start, idx, matching
        if matching and not state["triggered"] and idx < len(actions) and action == actions[idx][0]:
            now = time.monotonic()
            if replay_start is None:
                replay_start = now
            offset = (actions[idx][1] - actions[0][1]).total_seconds()
            delay = max(0, replay_start + offset - now)
            if delay:
                time.sleep(delay)
            with (run/"timing.jsonl").open("a") as log:
                log.write(json.dumps({"task":task,"action_index":idx+1,
                                      "target_seconds":offset,"wait_seconds":delay})+"\n")
        else:
            matching = False
        idx += 1
        return step(action,pause)

    env.step = timed_step
    try:
        return original(agent, env, example, max_steps, instruction, args, result_dir, scores)
    finally:
        env.step = step

lib_run_single.run_single_example = run_example

import run as runner

args = runner.config()
manifest = json.loads(Path(args.test_all_meta_path).read_text())
tasks = runner.get_unfinished(args.action_space,args.model,args.observation_type,args.result_dir,manifest)
if any(tasks.values()):
    runner.test(args,tasks)
