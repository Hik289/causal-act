"""Outcome-V gated refiner: use trained outcome-V (fail_prec 0.916) to gate.
Intervene ONLY when outcome-V says current step is 'failing' with high conf.
High precision => few false interventions => net positive (hopefully).
Combines: outcome-V gate (is traj failing?) + loop-breaker nudge (concrete keyboard alt).
CIV_OUTCOME_GATE=1 enables. CIV_VTAU = min confidence.
"""
from causalact_runtime import location
import os, io
import numpy as np
from PIL import Image, ImageChops
import torch

_S={"prev_png":None,"noeff":0,"overrides":0,"steps":0,"taskov":0,"last_task":None,"V":None,"proc":None}

def _diff(a,b):
    try:
        ia=Image.open(io.BytesIO(a)).convert("RGB");ib=Image.open(io.BytesIO(b)).convert("RGB")
        if ia.size!=ib.size: ib=ib.resize(ia.size)
        return float((np.asarray(ImageChops.difference(ia,ib)).sum(2)>20).mean())
    except: return 1.0

def _load_v():
    from transformers import AutoProcessor, AutoModelForVision2Seq
    ck=location('CAUSALACT_RUN_DIR', 'v_outcome_ckpt/ckpt')
    _S["proc"]=AutoProcessor.from_pretrained(ck)
    _S["V"]=AutoModelForVision2Seq.from_pretrained(ck,torch_dtype=torch.float16).to("cuda").eval()

@torch.no_grad()
def _v_fail(png, action):
    im=Image.open(io.BytesIO(png)).convert("RGB"); im.thumbnail((512,512))
    pr=("You are a task-outcome verifier for a GUI agent. Screenshot = current state. "
        f"Action the agent just took: {action[:200]}\n"
        "Based on the state + action, is this trajectory FAILING (agent is stuck/doing "
        "wrong things/will not complete the task)? Answer ONLY 'fail' or 'ok'.")
    m=[{"role":"user","content":[{"type":"image"},{"type":"text","text":pr}]}]
    t=_S["proc"].apply_chat_template(m,add_generation_prompt=True)
    inp=_S["proc"](text=t,images=[im],return_tensors="pt").to("cuda")
    g=_S["V"].generate(**inp,max_new_tokens=3,do_sample=False,output_scores=True,return_dict_in_generate=True)
    out=_S["proc"].batch_decode(g.sequences[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip().lower()
    return out.startswith("f")

def install(agent_module, audit_path):
    PA=agent_module.PromptAgent; orig=PA.predict
    enabled=os.environ.get("CIV_OUTCOME_GATE","0")=="1"
    cap=int(os.environ.get("CIV_CAP","2"))
    if enabled and _S["V"] is None: _load_v()
    def wrapped(self,instruction,obs):
        cur=obs.get("screenshot"); nudge=""
        tid=getattr(self,"_civ_task",None)
        if tid!=_S["last_task"]: _S["last_task"]=tid;_S["taskov"]=0;_S["noeff"]=0
        if enabled and cur is not None and _S["prev_png"] is not None:
            frac=_diff(_S["prev_png"],cur)
            if frac<0.005: _S["noeff"]+=1
            else: _S["noeff"]=0
            # gate: no-effect streak AND outcome-V says failing AND under cap
            if _S["noeff"]>=2 and _S["taskov"]<cap:
                last_act=str(self.actions[-1]) if getattr(self,"actions",None) else ""
                if _v_fail(cur, last_act):
                    _S["overrides"]+=1;_S["taskov"]+=1
                    nudge=("\n\n[SYSTEM] Your recent actions changed NOTHING and this approach is "
                           "failing. STOP repeating. Use KEYBOARD: menu Alt-mnemonic (Alt+C Colors, "
                           "Alt+F File, Alt+L seLect) then arrows+Enter; or the app's action search.")
        _S["prev_png"]=cur;_S["steps"]+=1
        with open(audit_path,"a") as f: f.write(f"step={_S['steps']} noeff={_S['noeff']} ov={bool(nudge)}\n")
        if nudge: instruction=instruction+nudge
        return orig(self,instruction,obs)
    PA.predict=wrapped
    return _S
