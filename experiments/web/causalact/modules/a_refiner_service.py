"""A-Refiner HTTP 服务 (causalact_py311 env)。

VWA 跑在 the benchmark environment(transformers 4.34), A/SmolVLM 需 causalact_py311(4.49) → 版本冲突。
故把 A 做成独立 HTTP 服务, VWA 侧 HTTP 调用 refine。

接口 POST /refine:
  {"image_b64":..., "intent":..., "base_action":{"action_type":"click|stop","element_id":"N"},
   "candidates":[{"id":"N","text":...}], "page_has_subscribe": bool}
返回:
  {"refined":{"action_type":..,"element_id":..}, "overridden":bool, "a_pred":..., "confidence":..}

保守策略 (降低窄 A 破坏 base 的风险):
  - A 只对**它训过的决策类型**(stop vs click, 即"目标是否已达成")做 refine。
  - 仅当 A 高置信(answer token prob >= tau)**且**与 base 分歧时才覆盖; 否则透传 base。
  - 对明显超出 A 能力的多步导航 click 目标选择, **不改 element_id**(A 没训 grounding), 只可能把
    base 的 click 改成 stop(当 A 确信目标已达成), 或反之。
"""
from causalact_runtime import location
import os, io, base64, json
import torch
from PIL import Image
from flask import Flask, request, jsonify

MODEL_DIR = os.environ.get("CIV_A_CKPT", location('CAUSALACT_RUN_DIR', 'a_refiner_ckpt/ckpt'))
TAU = float(os.environ.get("CIV_A_TAU", "0.6"))
device = "cuda"

from transformers import AutoProcessor, AutoModelForVision2Seq
print("loading A refiner from", MODEL_DIR, flush=True)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
model.eval()

app = Flask(__name__)


def build_prompt(intent):
    instr = (f"You are a web action agent. A red box labeled [1] marks the subscribe/unsubscribe "
             f"button in the screenshot. Goal: {intent}\n"
             f"Decide the next action:\n"
             f"- answer 'elem_1' to click the marked button\n"
             f"- answer 'stop' if the goal is ALREADY satisfied (no action needed)\n"
             f"Look at the button text in the image to decide. Answer with ONLY 'elem_1' or 'stop'.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def a_decide(image, intent):
    """A 输出 stop vs elem_1 + 置信度 (第一个判别 token 的 softmax prob)。"""
    prompt = build_prompt(intent)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=6, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    seq = gen.sequences[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(seq, skip_special_tokens=True)[0].strip().lower()
    # 置信度: 第一步 logits 的 top prob
    first = torch.softmax(gen.scores[0][0].float(), dim=-1)
    conf = float(first.max())
    if "stop" in text:
        return "stop", conf
    if "elem" in text:
        return "elem_1", conf
    return text.split()[0] if text.split() else "", conf


@app.route("/refine", methods=["POST"])
def refine():
    d = request.get_json()
    img = Image.open(io.BytesIO(base64.b64decode(d["image_b64"]))).convert("RGB")
    intent = d.get("intent", "")
    base = d.get("base_action", {})
    page_has_sub = d.get("page_has_subscribe", False)

    # A 只在"页面有 subscribe 控件"(= A 训过的 task 类型) 时介入; 否则透传
    if not page_has_sub:
        return jsonify({"refined": base, "overridden": False, "a_pred": None,
                        "confidence": None, "reason": "no_subscribe_control_passthrough"})

    a_pred, conf = a_decide(img, intent)
    base_type = base.get("action_type", "")
    # 保守覆盖: 仅 A 高置信 且 与 base 分歧
    override = False; refined = base; reason = "agree_or_lowconf_passthrough"
    if conf >= TAU:
        # A 说 stop 但 base 不是 stop → 覆盖为 stop (目标已达成)
        if a_pred == "stop" and base_type != "stop":
            refined = {"action_type": "stop", "answer": ""}; override = True; reason = "A_says_stop"
        # A 说 click elem_1 但 base 是 stop → 覆盖为 click subscribe 控件
        elif a_pred == "elem_1" and base_type == "stop":
            refined = {"action_type": "click", "element_id": d.get("subscribe_elem_id", base.get("element_id"))}
            override = True; reason = "A_says_click"
    return jsonify({"refined": refined, "overridden": override, "a_pred": a_pred,
                    "confidence": conf, "reason": reason})


@app.route("/health")
def health():
    return jsonify({"ok": True, "model_dir": MODEL_DIR, "tau": TAU})


if __name__ == "__main__":
    port = int(os.environ.get("CIV_A_PORT", "18900"))
    app.run(host="127.0.0.1", port=port, threaded=False)
