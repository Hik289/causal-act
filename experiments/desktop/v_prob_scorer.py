"""Compute outcome-V P(fail) for a screenshot+action (tunable tau gate).
Loads ep2 checkpoint, returns prob of 'fail' token vs 'ok' token at first answer position.
"""
from causalact_runtime import location
import os, torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

_M={"model":None,"proc":None,"dev":None,"fid":None,"oid":None}
CKPT=os.environ.get("CIV_VCKPT",location('CAUSALACT_RUN_DIR', 'v_outcome_ckpt/ckpt'))

def _load():
    if _M["model"] is not None: return
    dev="cuda" if torch.cuda.is_available() else "cpu"
    proc=AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
    model=AutoModelForVision2Seq.from_pretrained(CKPT,torch_dtype=torch.bfloat16).to(dev).eval()
    _M.update(model=model,proc=proc,dev=dev,
              fid=proc.tokenizer("fail",add_special_tokens=False).input_ids[0],
              oid=proc.tokenizer("ok",add_special_tokens=False).input_ids[0])

def bp(proc,action):
    m=[{"role":"user","content":[{"type":"image"},{"type":"text",
        "text":f"The agent is attempting a desktop task. Given this screenshot and the action it just took:\n{action}\nWill this trajectory ultimately FAIL the task? Answer 'fail' or 'ok'."}]}]
    return proc.apply_chat_template(m,add_generation_prompt=True)

@torch.no_grad()
def p_fail(screenshot_path, action):
    _load()
    im=Image.open(screenshot_path).convert("RGB"); im.thumbnail((512,512))
    inp=_M["proc"](text=bp(_M["proc"],action),images=[im],return_tensors="pt").to(_M["dev"])
    out=_M["model"](**inp)
    last=out.logits[0,-1,:]
    pf=last[_M["fid"]]; po=last[_M["oid"]]
    p=torch.softmax(torch.stack([po,pf]),0)[1].item()
    return p
