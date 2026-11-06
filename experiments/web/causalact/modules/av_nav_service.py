"""A+V nav refiner 服务 (官方 SoM, V-gated corrector, tau val定)。

端到端每导航步: base 选了 click 目标(SoM id)。
- V 判 base 目标对不对 → P(no)
- 若 P(no)>=tau 且 A 给出不同目标 → 用 A; 否则透传 base
tau 默认 0.7 (val 定, 0.5-0.95 稳定 net+1)。

接口 POST /nav_vgate:
  {image_b64, intent, candidate_ids:[...], base_target_id}
返回: {refined_id, overridden, a_pred, v_conf_no, reason}
⚠️ ASSERT: 图非空白(SoM框可见)——复用 sanity 的彩色像素校验。
"""
from causalact_runtime import location
import os, io, base64, json
import torch
from PIL import Image
from flask import Flask, request, jsonify

A_DIR = os.environ.get("CIV_A_NAV_CKPT", location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt/ckpt'))
V_DIR = os.environ.get("CIV_V_NAV_CKPT", location('CAUSALACT_RUN_DIR', 'v_official_ckpt/ckpt'))
TAU = float(os.environ.get("CIV_NAV_TAU", "0.7"))
device = "cuda"
from transformers import AutoProcessor, AutoModelForVision2Seq
print("loading A from", A_DIR, flush=True)
aproc = AutoProcessor.from_pretrained(A_DIR); A = AutoModelForVision2Seq.from_pretrained(A_DIR, torch_dtype=torch.float16).to(device); A.eval()
print("loading V from", V_DIR, flush=True)
vproc = AutoProcessor.from_pretrained(V_DIR); V = AutoModelForVision2Seq.from_pretrained(V_DIR, torch_dtype=torch.float16).to(device); V.eval()
app = Flask(__name__)


def nav_prompt(intent, cids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\nGoal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")

def v_prompt(intent, qid):
    return (f"You are a web navigation verifier. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes on clickable elements.\nAn agent wants to: {intent}\n"
            f"The agent is about to click the comments link marked {qid}.\n"
            f"Look at the numbered boxes to read the post title next to box {qid}. "
            f"Is clicking {qid} the correct action for the goal? Answer ONLY 'yes' or 'no'.")

def parse(t, cids):
    for c in cids:
        if c in t or c.split("_")[1] in t.split(): return c
    import re; m = re.search(r"\d+", t)
    if m and ("elem_" + m.group(0)) in cids: return "elem_" + m.group(0)
    return None


@torch.no_grad()
def a_ground(img, intent, cids):
    pr = aproc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": nav_prompt(intent, cids)}]}], add_generation_prompt=True)
    inp = aproc(text=pr, images=[img], return_tensors="pt").to(device)
    g = A.generate(**inp, max_new_tokens=8, do_sample=False)
    return parse(aproc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), cids)


@torch.no_grad()
def v_pno(img, intent, qid):
    vp = vproc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": v_prompt(intent, qid)}]}], add_generation_prompt=True)
    vi = vproc(text=vp, images=[img], return_tensors="pt").to(device)
    vg = V.generate(**vi, max_new_tokens=4, do_sample=False, output_scores=True, return_dict_in_generate=True)
    probs = torch.softmax(vg.scores[0][0].float(), dim=-1)
    no_ids = [tid for tid in range(probs.shape[0]) if vproc.tokenizer.decode([tid]).strip().lower().startswith("n")]
    return float(probs[no_ids].sum()) if no_ids else 0.0


def som_visible(img):
    ex = img.resize((64, 64)); px = list(ex.getdata())
    return sum(1 for (r, g, b) in px if max(r, g, b) - min(r, g, b) > 40) >= 15


@app.route("/nav_vgate", methods=["POST"])
def nav_vgate():
    d = request.get_json()
    img = Image.open(io.BytesIO(base64.b64decode(d["image_b64"]))).convert("RGB")
    intent = d.get("intent", ""); cids = d.get("candidate_ids", []); base_id = str(d.get("base_target_id", ""))
    # ASSERT 输入对称: SoM 框可见
    vis = som_visible(img)
    if not cids or not base_id or not vis:
        return jsonify({"refined_id": base_id, "overridden": False, "reason": "no_cands_or_invisible", "som_visible": vis})
    vno = v_pno(img, intent, base_id)
    overridden = False; refined = base_id; a_pred = None; reason = "v_ok_passthrough"
    if vno >= TAU:
        a_pred = a_ground(img, intent, cids)
        if a_pred is not None and a_pred != base_id:
            refined = a_pred; overridden = True; reason = "v_flag_a_correct"
        else:
            reason = "v_flag_but_a_same"
    return jsonify({"refined_id": refined, "overridden": overridden, "a_pred": a_pred,
                    "v_conf_no": vno, "base_id": base_id, "reason": reason, "som_visible": vis})


@app.route("/health")
def health():
    return jsonify({"ok": True, "tau": TAU, "A": A_DIR, "V": V_DIR})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("CIV_NAV_PORT", "18902")), threaded=False)
