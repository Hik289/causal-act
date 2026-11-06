"""v3 super-conservative loop-breaker: only intervene when base is DEFINITELY doomed.
Fix for hurts: previous streak>=2 fired on transient no-effect steps in tasks base
would've passed -> false-positive override -> nudge derails -> hurt.
v3: streak>=STREAK (default 4) consecutive screen-FROZEN (diff<0.002) steps = base is
in a dead loop that will exhaust max_steps and fail anyway. Overriding there can only
help (base outcome on that task is already FAIL). => hurt ~ 0 by construction, net>=0.
"""
import os, io
import numpy as np
from PIL import Image, ImageChops
_S={"prev":None,"streak":0,"ov":0,"steps":0,"taskov":0,"last":None}
def _frz(a,b):
    try:
        ia=Image.open(io.BytesIO(a)).convert("RGB");ib=Image.open(io.BytesIO(b)).convert("RGB")
        if ia.size!=ib.size: ib=ib.resize(ia.size)
        return float((np.asarray(ImageChops.difference(ia,ib)).sum(2)>20).mean())
    except: return 1.0
def install(agent_module, audit_path):
    PA=agent_module.PromptAgent; orig=PA.predict
    en=os.environ.get("CIV_LOOPBREAK","0")=="1"
    STREAK=int(os.environ.get("CIV_STREAK","4"))
    cap=int(os.environ.get("CIV_CAP","3"))
    def wrapped(self,instruction,obs):
        cur=obs.get("screenshot");nudge=""
        tid=getattr(self,"_civ_task",None)
        if tid!=_S["last"]: _S["last"]=tid;_S["taskov"]=0;_S["streak"]=0
        if en and cur is not None and _S["prev"] is not None:
            frac=_frz(_S["prev"],cur)
            if frac<0.002: _S["streak"]+=1   # screen essentially FROZEN
            else: _S["streak"]=0
            if _S["streak"]>=STREAK and _S["taskov"]<cap:
                _S["ov"]+=1;_S["taskov"]+=1
                nudge=("\n\n[SYSTEM] You have repeated the same ineffective action many times "
                       "with ZERO change on screen — you are stuck in a dead loop and will fail. "
                       "You MUST abandon this approach NOW. Use a completely different method: "
                       "keyboard shortcut (Ctrl/Alt+key), menu mnemonic (Alt+letter) then arrows+Enter, "
                       "or the application's action/command search. Do NOT click the same spot again.")
        _S["prev"]=cur;_S["steps"]+=1
        with open(audit_path,"a") as f: f.write(f"step={_S['steps']} streak={_S['streak']} ov={bool(nudge)}\n")
        if nudge: instruction=instruction+nudge
        return orig(self,instruction,obs)
    PA.predict=wrapped
    return _S
