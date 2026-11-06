"""Stop-sensitivity refiner 服务 (V stop-verifier, 官方 SoM)。
端到端 looping-gate 触发时调: 判"目标是否已达成→该 stop 吗" → P(stop)。
P(stop)>=tau → 指示 agent 发 stop。

接口 POST /stop_check:
  {image_b64, intent, action_history:[...]}
返回: {should_stop:bool, p_stop:float, tau}
⚠️ ASSERT: SoM 框可见(彩色像素)。
"""
from causalact_runtime import location
import os, io, base64, json
import torch
from PIL import Image
from flask import Flask, request, jsonify

V_DIR = os.environ.get("CIV_V_STOP_CKPT", location('CAUSALACT_RUN_DIR', 'v_stop_ckpt/ckpt'))
TAU = float(os.environ.get("CIV_STOP_TAU", "0.7"))
device = "cuda"
from transformers import AutoProcessor, AutoModelForVision2Seq
print("loading V stop from", V_DIR, flush=True)
proc = AutoProcessor.from_pretrained(V_DIR)
V = AutoModelForVision2Seq.from_pretrained(V_DIR, torch_dtype=torch.float16).to(device); V.eval()
app = Flask(__name__)


def v_prompt(intent, hist):
    h = " -> ".join(hist[-3:]) if hist else "None"
    return (f"You are a web task verifier. The screenshot shows the current page with "
            f"numbered Set-of-Mark boxes.\nTask goal: {intent}\n"
            f"Recent actions: {h}\n"
            f"Look at the screenshot: has the goal ALREADY been achieved (e.g. you are now on "
            f"the target post's comment section)? If achieved, the agent should STOP.\n"
            f"Answer ONLY 'yes' (goal achieved, stop) or 'no' (not yet, continue).")


def som_visible(img):
    ex = img.resize((64, 64)); px = list(ex.getdata())
    return sum(1 for (r, g, b) in px if max(r, g, b) - min(r, g, b) > 40) >= 15


@torch.no_grad()
def p_stop(img, intent, hist):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": v_prompt(intent, hist)}]}]
    pr = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=pr, images=[img], return_tensors="pt").to(device)
    g = V.generate(**inp, max_new_tokens=4, do_sample=False, output_scores=True, return_dict_in_generate=True)
    probs = torch.softmax(g.scores[0][0].float(), dim=-1)
    yes_ids = [tid for tid in range(probs.shape[0]) if proc.tokenizer.decode([tid]).strip().lower().startswith("y")]
    return float(probs[yes_ids].sum()) if yes_ids else 0.0


@app.route("/stop_check", methods=["POST"])
def stop_check():
    d = request.get_json()
    img = Image.open(io.BytesIO(base64.b64decode(d["image_b64"]))).convert("RGB")
    intent = d.get("intent", ""); hist = d.get("action_history", [])
    vis = som_visible(img)
    if not vis:
        return jsonify({"should_stop": False, "p_stop": None, "reason": "som_invisible", "tau": TAU})
    ps = p_stop(img, intent, hist)
    return jsonify({"should_stop": ps >= TAU, "p_stop": ps, "tau": TAU})


@app.route("/health")
def health():
    return jsonify({"ok": True, "tau": TAU, "V": V_DIR})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("CIV_STOP_PORT", "18903")), threaded=False)
