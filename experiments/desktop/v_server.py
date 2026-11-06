"""Outcome-V HTTP microservice (runs in causalact_py311, transformers 4.49).
POST /pfail {screenshot_b64, action} -> {p_fail}. Lets the OSWorld runner (osworld_py311)
query V per step so the gate threshold tau is tunable for validation operating-point selection.
"""
from causalact_runtime import location
import os, io, base64, torch
from PIL import Image
from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForVision2Seq

CKPT=os.environ.get("CIV_VCKPT",location('CAUSALACT_RUN_DIR', 'v_outcome_ckpt/ckpt'))
dev="cuda" if torch.cuda.is_available() else "cpu"
proc=AutoProcessor.from_pretrained(CKPT)
model=AutoModelForVision2Seq.from_pretrained(CKPT,torch_dtype=torch.float32,attn_implementation="eager").to(dev).eval()
fid=proc.tokenizer("fail",add_special_tokens=False).input_ids[0]
oid=proc.tokenizer("ok",add_special_tokens=False).input_ids[0]

def bp(action):
    m=[{"role":"user","content":[{"type":"image"},{"type":"text",
        "text":f"The agent is attempting a desktop task. Given this screenshot and the action it just took:\n{action}\nWill this trajectory ultimately FAIL the task? Answer 'fail' or 'ok'."}]}]
    return proc.apply_chat_template(m,add_generation_prompt=True)

app=Flask(__name__)
@app.route("/pfail",methods=["POST"])
@torch.no_grad()
def pfail():
    d=request.get_json()
    im=Image.open(io.BytesIO(base64.b64decode(d["screenshot_b64"]))).convert("RGB"); im.thumbnail((512,512))
    inp=proc(text=bp(d.get("action","")),images=[im],return_tensors="pt").to(dev)
    out=model(**inp); last=out.logits[0,-1,:]
    p=torch.softmax(torch.stack([last[oid],last[fid]]),0)[1].item()
    return jsonify({"p_fail":p})
@app.route("/health")
def h(): return jsonify({"ok":True})
if __name__=="__main__":
    app.run(host="127.0.0.1",port=int(os.environ.get("CIV_VPORT","18833")),threaded=False)
