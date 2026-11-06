"""I-1 最小因果 loop: A (SmolVLM-500M) 训练 L_act + L_spec。

A = Action Refiner, 生成式: 输入 [screenshot, goal, 候选元素列表] → 生成目标元素编号。
目标 = 让 A 预测正确 click 目标 (L_act), 且对 unrelated 视觉扰动保持稳定 (L_spec)。

Loss (README §8.1 L_act + §8.3 L_spec):
  L_act  = LM CE, 在 ORIGINAL 图上生成正确 target id 的负对数似然
  L_spec = 一致性损失, unrelated 图上答案 token 分布 应 == original 图上分布
           实现: KL( softmax(logits_unrel[ans_pos]) || softmax(logits_orig[ans_pos]).detach() )
           (让 unrelated 的预测对齐 original 的预测, original 作 teacher)
  L = L_act + lambda_spec * L_spec

评估: Unrelated Action Accuracy = 在 unrelated 图上 A 预测的 target == GT target 的比例。
对比 A 训练前(zero-shot SmolVLM) vs 训练后, 验证 L_spec 是否让 Unrel Acc 不降/提升。

单卡 2080Ti 11GB, fp16/bf16(Turing 无bf16用fp16), batch=1 + grad accum。
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def build_prompt(processor, intent, elems_desc, image):
    """构造多模态 prompt: 截图 + 目标 + 候选元素, 要求输出目标编号。"""
    if VISUAL_GROUNDING:
        instr = (f"You are a web action agent. The screenshot has numbered boxes (Set-of-Mark) "
                 f"overlaid on clickable elements. Goal: {intent}\n"
                 f"Candidate elements (id: bounding box):\n{elems_desc}\n"
                 f"Look at the image to identify which numbered box matches the goal. "
                 f"Answer with ONLY the element id (e.g. elem_3).")
    else:
        instr = (f"You are a web action agent. Goal: {intent}\n"
                 f"Interactable elements (id: text):\n{elems_desc}\n"
                 f"Which element should you click to achieve the goal? "
                 f"Answer with ONLY the element id (e.g. elem_3).")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    return prompt


VISUAL_GROUNDING = os.environ.get("CIV_VISUAL_GROUNDING", "0") == "1"


def load_elems_desc(sample):
    """候选描述。两种模式:
    - 默认 (I-1): id + text (文字匹配)
    - CIV_VISUAL_GROUNDING=1 (I-1b 方案a): id + bbox, **NO text** → A 必须看图认框。
    GT = original_action.target_id。
    """
    gt = sample.original_action.target_id  # elem_k
    cands = sample.oracle_metadata.get("candidates", [])
    if not cands:
        cands = [{"id": gt, "text": "(target)"}]
    ids = [c["id"] for c in cands]
    random.shuffle(ids)
    if os.environ.get("CIV_PURE_VISUAL", "0") == "1":
        # 纯视觉: 只列编号, 不给 bbox 也不给文字 → A 必须从图上 SoM 框读位置
        desc = "Candidate box ids: " + ", ".join(sorted(ids, key=lambda x: int(x.split("_")[1])))
    elif VISUAL_GROUNDING:
        # 只给编号 + bbox 坐标 (图上有对应 SoM 框), 不给文字
        id2bbox = {c["id"]: c.get("bbox") for c in cands}
        desc = "\n".join(f"{cid}: box at {id2bbox.get(cid)}" for cid in ids)
    else:
        id2text = {c["id"]: (c.get("text") or "(no text)") for c in cands}
        desc = "\n".join(f"{cid}: {id2text.get(cid,'(interactable)')}" for cid in ids)
    return desc, gt, ids


def get_answer_token_logits(model, processor, prompt, image, device, gt_text):
    """前向一次, 返回生成第一个 answer token 位置的 logits + 该位置 CE loss(对 gt first token)。"""
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    # teacher forcing: 拼上答案
    ans_ids = processor.tokenizer(gt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([inputs["input_ids"], ans_ids], dim=1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(ans_ids)], dim=1)
    # pixel_values 保持
    out = model(input_ids=full_ids, attention_mask=attn,
                pixel_values=inputs.get("pixel_values"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"))
    logits = out.logits  # [1, seq, vocab]
    # answer tokens 位置: 最后 len(ans_ids) 个
    n_ans = ans_ids.shape[1]
    ans_logits = logits[:, -n_ans - 1:-1, :]  # 预测 answer 的 logits
    # CE loss over answer tokens
    ce = F.cross_entropy(ans_logits.reshape(-1, ans_logits.size(-1)),
                         ans_ids.reshape(-1))
    # L_spec 用**判别性 token** 的分布: elem_N 的前两 token ('elem','_') 所有候选共享,
    # 只有最后一个 token (数字) 判别 → 用最后 answer token 位置的 logits, 否则 L_spec 对齐噪声。
    disc_logits = ans_logits[:, -1, :]  # [1, vocab] 判别 token (数字) 位置
    return ce, disc_logits


@torch.no_grad()
def _predict(model, processor, s, img_key, device):
    desc, gt, ids = load_elems_desc(s)
    img = Image.open(getattr(s, img_key)).convert("RGB")
    prompt = build_prompt(processor, s.intent, desc, img)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    pred = text.strip().split()[0] if text.strip() else ""
    return pred, gt


@torch.no_grad()
def eval_unrelated_acc(model, processor, samples, device, img_key="unrelated_screenshot_path"):
    """生成式评估: 在指定图上 A 预测的 target == GT 的比例。"""
    model.eval()
    correct = 0; total = 0
    for s in samples:
        pred, gt = _predict(model, processor, s, img_key, device)
        total += 1
        if gt in pred:
            correct += 1
    return correct / total if total else 0.0, correct, total


@torch.no_grad()
def eval_consistency(model, processor, samples, device):
    """一致性: pred(original) == pred(unrelated) 的比例 (L_spec 直接优化的量, 比 acc 更敏感)。"""
    model.eval()
    consist = 0; total = 0
    for s in samples:
        po, _ = _predict(model, processor, s, "screenshot_path", device)
        pu, _ = _predict(model, processor, s, "unrelated_screenshot_path", device)
        total += 1
        if po == pu:
            consist += 1
    return consist / total if total else 0.0, consist, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=location('CAUSALACT_RUN_DIR', 'web/data/unrelated_set.jsonl'))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=location('CAUSALACT_RUN_DIR', 'web/checkpoints/a_lspec_out'))
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda"
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # 混合精度 (AMP): fp32 master 权重 (防 NaN) + fp16 compute (防 OOM) + GradScaler (loss scaling)。
    # 纯 fp16 full-FT 在 Turing 上 NaN; 纯 fp32 11GB OOM → AMP 是正解。
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    # 冻结视觉塔 (86M) 省显存+激活; 训练 text_model(362M)+connector(12M)+lm_head。
    # 模型全参数 507M<1B 不变 (README 约束是模型规模, 非可训练量; 冻结≠LoRA 规避)。
    for p in model.model.vision_model.parameters():
        p.requires_grad = False
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot_p = sum(p.numel() for p in model.parameters())
    print(f"params: total={n_tot_p} (<1B:{n_tot_p < 1e9}), trainable={n_train_p} (vision frozen)")
    scaler = torch.cuda.amp.GradScaler()

    samples = load_jsonl(args.data)
    random.shuffle(samples)
    n_val = max(10, len(samples) // 5)
    val = samples[:n_val]; train = samples[n_val:]
    print(f"train={len(train)} val={len(val)}")

    # ===== 训练前基线: Unrelated Acc + Consistency (zero-shot) =====
    pre_unrel_acc, pc, pt = eval_unrelated_acc(model, processor, val, device, "unrelated_screenshot_path")
    pre_orig_acc, poc, pot = eval_unrelated_acc(model, processor, val, device, "screenshot_path")
    pre_consist, pcc, pct = eval_consistency(model, processor, val, device)
    print(f"[PRE] Unrelated Acc = {pre_unrel_acc:.4f} ({pc}/{pt}) | Original Acc = {pre_orig_acc:.4f} ({poc}/{pot}) | Consistency = {pre_consist:.4f} ({pcc}/{pct})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_log = []
    model.train()
    step = 0
    for ep in range(args.epochs):
        random.shuffle(train)
        opt.zero_grad()
        for i, s in enumerate(train):
            desc, gt, ids = load_elems_desc(s)
            img_o = Image.open(s.screenshot_path).convert("RGB")
            img_u = Image.open(s.unrelated_screenshot_path).convert("RGB")
            prompt = build_prompt(processor, s.intent, desc, img_o)
            prompt_u = build_prompt(processor, s.intent, desc, img_u)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # L_act on original
                ce_o, logits_o = get_answer_token_logits(model, processor, prompt, img_o, device, gt)
                # L_spec: unrelated 分布对齐 original (original 作 teacher, detach)
                _, logits_u = get_answer_token_logits(model, processor, prompt_u, img_u, device, gt)
                p_o = F.log_softmax(logits_o.detach().float(), dim=-1)
                p_u = F.log_softmax(logits_u.float(), dim=-1)
                l_spec = F.kl_div(p_u, p_o.exp(), reduction="batchmean")
                loss = (ce_o + args.lambda_spec * l_spec) / args.grad_accum
            # NaN/Inf 守卫: 跳过坏 step, 不污染权重
            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss at ep{ep} step{step} (L_act={ce_o.item()}, L_spec={l_spec.item()})")
                opt.zero_grad()
                step += 1
                continue
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
            loss_log.append({"step": step, "epoch": ep, "l_act": float(ce_o.detach()),
                             "l_spec": float(l_spec.detach()), "loss": float(loss.detach() * args.grad_accum)})
            if step % 5 == 0:
                print(f"  ep{ep} step{step} L_act={ce_o.item():.3f} L_spec={l_spec.item():.4f}")
        # epoch end eval
        ua, c, tt = eval_unrelated_acc(model, processor, val, device, "unrelated_screenshot_path")
        print(f"[EP{ep}] Unrelated Acc = {ua:.4f} ({c}/{tt})")
        model.train()

    # ===== 训练后: Unrelated Acc + Original Acc + Consistency =====
    post_unrel_acc, cc, ct = eval_unrelated_acc(model, processor, val, device, "unrelated_screenshot_path")
    post_orig_acc, oc, ot = eval_unrelated_acc(model, processor, val, device, "screenshot_path")
    post_consist, pcc2, pct2 = eval_consistency(model, processor, val, device)
    print(f"[POST] Unrelated Acc = {post_unrel_acc:.4f} ({cc}/{ct}) | Original Acc = {post_orig_acc:.4f} ({oc}/{ot}) | Consistency = {post_consist:.4f} ({pcc2}/{pct2})")

    result = {
        "model": MODEL_ID, "params_lt_1B": True,
        "n_train": len(train), "n_val": len(val),
        "epochs": args.epochs, "lr": args.lr, "lambda_spec": args.lambda_spec,
        "seed": args.seed,
        "pre": {"unrelated_acc": pre_unrel_acc, "original_acc": pre_orig_acc, "consistency": pre_consist},
        "post": {"unrelated_acc": post_unrel_acc, "original_acc": post_orig_acc, "consistency": post_consist},
        "delta_unrelated_acc": post_unrel_acc - pre_unrel_acc,
        "delta_consistency": post_consist - pre_consist,
        "loss_log": loss_log,
    }
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT:", json.dumps({k: result[k] for k in ("pre", "post", "delta_unrelated_acc")}, indent=2))


if __name__ == "__main__":
    main()
