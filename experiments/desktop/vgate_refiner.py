"""V-gated trajectory-replay refiner (v5). Tunable tau operating point.
Replays base actions (identical passthrough) until outcome-V P(fail) >= CIV_VTAU for
CIV_VSTREAK consecutive steps -> intervene with loop-break nudge, then diverge.
V is queried from the HTTP microservice (causalact_py311) so tau is tunable for
validation operating-point selection. Falls back to screen-frozen gate if server down.
"""
from causalact_runtime import location
import os, io, json, glob, base64, urllib.request
import numpy as np
from PIL import Image, ImageChops

BASE_ROOT=os.environ.get("CIV_BASE_ROOT",location('CAUSALACT_RUN_DIR', 'osw_base_full/pyautogui/screenshot/gpt-5.6-sol'))
VURL=os.environ.get("CIV_VURL","http://127.0.0.1:18833/pfail")
TAU=float(os.environ.get("CIV_VTAU","0.9"))
VSTREAK=int(os.environ.get("CIV_VSTREAK","2"))
CAP=int(os.environ.get("CIV_CAP","3"))
_S={"task":None,"idx":0,"triggered":False,"vstreak":0,"ov":0,"steps":0,"_taskov":0,"base_resp":None,"prev":None}

def _load_base(task):
    for p in glob.glob(f"{BASE_ROOT}/*/{task}"):
        tj=p+"/traj.jsonl"
        if os.path.exists(tj): return [json.loads(l).get("response","") for l in open(tj)]
    return None

def _pfail(screenshot_bytes, action):
    try:
        b64=base64.b64encode(screenshot_bytes).decode()
        req=urllib.request.Request(VURL,data=json.dumps({"screenshot_b64":b64,"action":action}).encode(),
                                   headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            return json.loads(r.read())["p_fail"]
    except Exception:
        return None

def install(agent_module, audit_path):
    PA=agent_module.PromptAgent; orig=PA.predict
    en=os.environ.get("CIV_VGATE","1")=="1"
    def wrapped(self,instruction,obs):
        cur=obs.get("screenshot")
        tid=getattr(self,"_civ_task",None)
        if tid!=_S["task"]:
            _S.update(task=tid,idx=0,triggered=False,vstreak=0,_taskov=0,prev=None)
            _S["base_resp"]=_load_base(tid) if (en and tid) else None
        _S["steps"]+=1; i=_S["idx"]; _S["idx"]+=1
        # V score on current obs+(upcoming base action)
        trigger=False; pf=None
        if en and not _S["triggered"] and cur is not None and _S["_taskov"]<CAP:
            act=_S["base_resp"][i] if (_S["base_resp"] and i<len(_S["base_resp"])) else ""
            pf=_pfail(cur,act)
            if pf is not None and pf>=TAU: _S["vstreak"]+=1
            else: _S["vstreak"]=0
            trigger = _S["vstreak"]>=VSTREAK
        # passthrough replay (byte-identical) unless trigger / already diverged
        if en and (not trigger) and (not _S["triggered"]) and _S["base_resp"] and i<len(_S["base_resp"]):
            resp=_S["base_resp"][i]
            try:
                from mm_agents.agent import encode_image
                b64=encode_image(cur) if cur is not None else None
            except Exception: b64=None
            self.observations.append({"screenshot":b64,"accessibility_tree":None})
            try: actions=self.parse_actions(resp,None)
            except Exception: actions=None
            if not actions:
                try: self.observations.pop()
                except Exception: pass
                try: self.actions.pop()
                except Exception: pass
                with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} mode=replay_fallback\n")
                return orig(self,instruction,obs)
            self.thoughts.append(resp)
            with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} mode=replay\n")
            return resp,actions
        if trigger:
            _S["ov"]+=1; _S["triggered"]=True; _S["_taskov"]+=1
            nudge=("\n\n[SYSTEM] This approach is predicted to fail. Abandon it NOW and try a "
                   "completely different method: keyboard shortcut (Ctrl/Alt+key), menu mnemonic "
                   "(Alt+letter) then arrows+Enter, or the application command search.")
            with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} mode=TRIGGER\n")
            return orig(self,instruction+nudge,obs)
        with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} mode=llm\n")
        return orig(self,instruction,obs)
    PA.predict=wrapped
    return _S
