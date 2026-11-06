"""A 导航 grounding refiner 服务 (官方 SoM, trained nav ckpt 88.5%)。

端到端每步: base 选了一个 click 目标(SoM id)。A 独立 grounding 选目标 + 给置信度。
gated corrector (保 B+A≥base):
  - A 与 base 同意 → 透传 base
  - A 与 base 分歧 且 A 置信度 >= tau → 用 A 的目标 (仅在 A 高置信时纠正)
  - 否则 → 透传 base
tau 在 val 上调, 使 override 净正 (不在 test 上调)。

接口 POST /nav_refine:
  {image_b64, intent, candidate_ids:[...], base_target_id}
返回: {refined_id, overridden, a_pred, a_conf}
"""
from causalact_runtime import location
import os, io, base64, json
import torch
from PIL import Image
from flask import Flask, request, jsonify

MODEL_DIR = os.environ.get("CIV_A_NAV_CKPT", location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt/ckpt'))
TAU = float(os.environ.get("CIV_NAV_TAU", "0.7"))
device = "cuda"
from transformers import AutoProcessor, AutoModelForVision2Seq
print("loading A nav refiner from", MODEL_DIR, flush=True)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
model.eval()
app = Flask(__name__)


def prompt_text(intent, cand_ids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\n"
            f"Goal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cand_ids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")


@torch.no_grad()
def a_ground(image, intent, cand_ids):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(intent, cand_ids)}]}]
    pr = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = processor(text=pr, images=[image], return_tensors="pt").to(device)
    gen = model.generate(**inp, max_new_tokens=8, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    seq = gen.sequences[:, inp["input_ids"].shape[1]:]
    text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
    # confidence: softmax prob of first generated token
    conf = float(torch.softmax(gen.scores[0][0].float(), dim=-1).max())
    # parse elem_N
    pred = None
    for cid in cand_ids:
        if cid in text or cid.split("_")[1] in text.split():
            pred = cid; break
    if pred is None:
        import re
        m = re.search(r"\d+", text)
        if m:
            cand = "elem_" + m.group(0)
            if cand in cand_ids:
                pred = cand
    return pred, conf, text


@app.route("/nav_refine", methods=["POST"])
def nav_refine():
    d = request.get_json()
    img = Image.open(io.BytesIO(base64.b64decode(d["image_b64"]))).convert("RGB")
    intent = d.get("intent", "")
    cand_ids = d.get("candidate_ids", [])
    base_id = str(d.get("base_target_id", ""))
    a_pred, a_conf, raw = a_ground(img, intent, cand_ids)
    overridden = False; refined = base_id
    if a_pred is not None and a_pred != base_id and a_conf >= TAU:
        refined = a_pred; overridden = True
    return jsonify({"refined_id": refined, "overridden": overridden,
                    "a_pred": a_pred, "a_conf": a_conf, "base_id": base_id})


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_dir": MODEL_DIR, "tau": TAU})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("CIV_NAV_PORT", "18901")), threaded=False)
