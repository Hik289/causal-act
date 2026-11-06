import os, json, glob
os.environ["CIV_REPLAY"]="1"; os.environ["CIV_STREAK"]="4"
import mm_agents.agent as agent_module
import replay_refiner as patch
# verify base trajectory loads
tid="e2dd0213-26db-4349-abe5-d5667bfd725c"
rows=patch._load_base(tid)
print("base_resp loaded:", None if rows is None else len(rows), "responses")
if rows: print("first resp head:", repr(rows[0])[:100])
# verify parse_actions works on a base response (need a PromptAgent instance)
try:
    a=agent_module.PromptAgent(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"), observation_type="screenshot", action_space="pyautogui")
    acts=a.parse_actions(rows[0], None)
    print("parsed actions from base resp:", repr(acts)[:120])
except Exception as e:
    print("parse err:", repr(e)[:200])
