"""v6 CONJUNCTION-gate trajectory-replay refiner.
Intervene only when BOTH: (1) outcome-V P(fail) >= CIV_VTAU AND (2) screen frozen
(diff < CIV_FROZEN) for >= CIV_VSTREAK consecutive steps. Requiring both signals kills
transient V false-positives on a still-moving screen (the 4 triggered hurts in v5),
pushing trigger precision up. Replays base actions verbatim (byte-identical passthrough)
until the conjunction fires, then diverges.
"""
from causalact_runtime import location
import os, io, json, glob, base64, urllib.request
import numpy as np
from PIL import Image, ImageChops

BASE_ROOT=os.environ.get("CIV_BASE_ROOT",location('CAUSALACT_RUN_DIR', 'osw_base_full/pyautogui/screenshot/gpt-5.6-sol'))
VURL=os.environ.get("CIV_VURL","http://127.0.0.1:18833/pfail")
TAU=float(os.environ.get("CIV_VTAU","0.9"))
VSTREAK=int(os.environ.get("CIV_VSTREAK","2"))
FROZEN=float(os.environ.get("CIV_FROZEN","0.004"))
CAP=int(os.environ.get("CIV_CAP","3"))
_S={"task":None,"idx":0,"triggered":False,"vstreak":0,"ov":0,"steps":0,"_taskov":0,"base_resp":None,"prev":None}

def _load_base(task):
    for p in glob.glob(f"{BASE_ROOT}/*/{task}"):
        tj=p+"/traj.jsonl"
        if os.path.exists(tj): return [json.loads(l).get("response","") for l in open(tj)]
    return None

def _frz(a,b):
    try:
        ia=Image.open(io.BytesIO(a)).convert("RGB"); ib=Image.open(io.BytesIO(b)).convert("RGB")
        if ia.size!=ib.size: ib=ib.resize(ia.size)
        return float((np.asarray(ImageChops.difference(ia,ib)).sum(2)>20).mean())
    except Exception: return 1.0

def _pfail(sb, action):
    try:
        b64=base64.b64encode(sb).decode()
        req=urllib.request.Request(VURL,data=json.dumps({"screenshot_b64":b64,"action":action}).encode(),
                                   headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read())["p_fail"]
    except Exception: return None

def install(agent_module, audit_path):
    PA=agent_module.PromptAgent; orig=PA.predict
    en=os.environ.get("CIV_CONJ","1")=="1"
    def wrapped(self,instruction,obs):
        cur=obs.get("screenshot"); tid=getattr(self,"_civ_task",None)
        if tid!=_S["task"]:
            _S.update(task=tid,idx=0,triggered=False,vstreak=0,_taskov=0,prev=None)
            _S["base_resp"]=_load_base(tid) if (en and tid) else None
        _S["steps"]+=1; i=_S["idx"]; _S["idx"]+=1
        trigger=False; pf=None; frac=None
        if en and not _S["triggered"] and cur is not None and _S["_taskov"]<CAP:
            act=_S["base_resp"][i] if (_S["base_resp"] and i<len(_S["base_resp"])) else ""
            # screen-frozen signal
            frac=_frz(_S["prev"],cur) if _S["prev"] is not None else 1.0
            frozen = frac < FROZEN
            # V signal only evaluated when frozen (save V calls + conjunction)
            if frozen:
                pf=_pfail(cur,act)
                if pf is not None and pf>=TAU: _S["vstreak"]+=1
                else: _S["vstreak"]=0
            else:
                _S["vstreak"]=0
            trigger = frozen and _S["vstreak"]>=VSTREAK
        _S["prev"]=cur
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
                with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} frac={frac} mode=replay_fallback\n")
                return orig(self,instruction,obs)
            self.thoughts.append(resp)
            with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} frac={frac} mode=replay\n")
            return resp,actions
        if trigger:
            _S["ov"]+=1; _S["triggered"]=True; _S["_taskov"]+=1
            nudge=("\n\n[SYSTEM] This approach is stuck and predicted to fail (no screen change). "
                   "Abandon it NOW and try a completely different method: keyboard shortcut (Ctrl/Alt+key), "
                   "menu mnemonic (Alt+letter) then arrows+Enter, or the application command search.")
            with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} frac={frac} mode=TRIGGER\n")
            return orig(self,instruction+nudge,obs)
        with open(audit_path,"a") as f: f.write(f"task={tid} step={_S['steps']} pf={pf} frac={frac} mode=llm\n")
        return orig(self,instruction,obs)
    PA.predict=wrapped
    return _S
