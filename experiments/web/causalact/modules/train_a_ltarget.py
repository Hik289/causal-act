"""I-2 (H0.method_2.3): A (SmolVLM-500M) 训练 L_act + L_target, 测 sensitivity。

数据: target_set.jsonl, 每个 pair (s, s_c) 共享 intent, A*(s)=click elem_1, A*(s_c)=stop。
候选动作空间: {click elem_1, stop} (A 看图判断 subscribe 控件状态 → 决定 click 还是 stop)。

Loss:
  L_act    = 对 **original(s)** 样本生成正确动作的 CE (标准行为克隆)
  L_target = 对 **edited(s_c)** 样本生成正确(已改变)动作的 CE (README §8.2, 只对
             环境核验 A*(s)!=A*(s_c) 的 pair 施加 = 翻转监督)
  L = L_act + lambda_target * L_target

三条件对照 (由 lambda 控制):
  - base       : 不训练 (zero-shot), 评估
  - L_act only : 只用 original(s) 样本训练 (lambda_target=0, 且训练集只含 s 侧)
  - L_act+L_target: s 和 s_c 都训 (lambda_target>0)

评估 (val, pair 级, n>=30):
  - TargetedAcc  = A 在 s_c (改变后) 上预测正确(新)动作的比例
  - PairCausalAcc = pair 中 A(s) 对 且 A(s_c) 对 且 A(s)!=A(s_c) 的比例
  - OriginalAcc  = A 在 s 上预测正确动作的比例 (保证没为 sensitivity 牺牲原动作)
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def build_prompt(processor, intent, image):
    instr = (f"You are a web action agent. A red box labeled [1] marks the subscribe/unsubscribe "
             f"button in the screenshot. Goal: {intent}\n"
             f"Decide the next action:\n"
             f"- answer 'elem_1' to click the marked button\n"
             f"- answer 'stop' if the goal is ALREADY satisfied (no action needed)\n"
             f"Look at the button text in the image to decide. Answer with ONLY 'elem_1' or 'stop'.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def gt_text(sample):
    a = sample.original_action
    return "elem_1" if a.action_type == "click" else "stop"


def ce_loss(model, processor, sample, device):
    img = Image.open(sample.screenshot_path).convert("RGB")
    prompt = build_prompt(processor, sample.intent, img)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    gt = gt_text(sample)
    ans_ids = processor.tokenizer(gt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([inputs["input_ids"], ans_ids], dim=1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(ans_ids)], dim=1)
    out = model(input_ids=full, attention_mask=attn,
                pixel_values=inputs.get("pixel_values"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"))
    n_ans = ans_ids.shape[1]
    ans_logits = out.logits[:, -n_ans - 1:-1, :]
    return F.cross_entropy(ans_logits.reshape(-1, ans_logits.size(-1)), ans_ids.reshape(-1))


@torch.no_grad()
def predict(model, processor, sample, device):
    img = Image.open(sample.screenshot_path).convert("RGB")
    prompt = build_prompt(processor, sample.intent, img)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    gen = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
    if "elem_1" in text or "elem" in text:
        return "elem_1"
    if "stop" in text:
        return "stop"
    return text.split()[0] if text.split() else ""


def group_pairs(samples):
    """按 pair_id 分组: {pair_id: {'s':sample, 'sc':sample}}"""
    pairs = {}
    for s in samples:
        pid = s.oracle_metadata["pair_id"]
        tid = s.task_id
        slot = "s" if tid.endswith("_s") else "sc"
        pairs.setdefault(pid, {})[slot] = s
    return {k: v for k, v in pairs.items() if "s" in v and "sc" in v}


@torch.no_grad()
def eval_pairs(model, processor, pairs, device):
    model.eval()
    tgt_ok = tgt_n = 0
    orig_ok = orig_n = 0
    pair_ok = pair_n = 0
    for pid, pr in pairs.items():
        s, sc = pr["s"], pr["sc"]
        ps = predict(model, processor, s, device)
        psc = predict(model, processor, sc, device)
        gs, gsc = gt_text(s), gt_text(sc)
        orig_n += 1; orig_ok += (ps == gs)
        tgt_n += 1; tgt_ok += (psc == gsc)
        pair_n += 1
        if ps == gs and psc == gsc and ps != psc:
            pair_ok += 1
    return {
        "TargetedAcc": tgt_ok / tgt_n if tgt_n else 0, "_tgt": [tgt_ok, tgt_n],
        "OriginalAcc": orig_ok / orig_n if orig_n else 0, "_orig": [orig_ok, orig_n],
        "PairCausalAcc": pair_ok / pair_n if pair_n else 0, "_pair": [pair_ok, pair_n],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=location('CAUSALACT_RUN_DIR', 'web/data/target_set.jsonl'))
    ap.add_argument("--mode", choices=["base", "lact_only", "lact_both", "lact_ltarget"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda_target", type=float, default=1.0)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    for p in model.model.vision_model.parameters():
        p.requires_grad = False
    scaler = torch.cuda.amp.GradScaler()

    samples = load_jsonl(args.data)
    pairs = group_pairs(samples)
    # ⚠️ 防泄漏: 按 **forum** 切分 (同 forum 的 A/B×paraphrase 共享同一截图)。
    # 否则 val 会见到 train 的同图 → 图像级泄漏 → 虚高分数。
    def forum_of(pid):
        return pid.rsplit("_", 1)[0]
    forums = sorted(set(forum_of(pid) for pid in pairs))
    random.shuffle(forums)
    n_val_forum = max(8, len(forums) // 5)   # ~20% forum 作 val
    val_forums = set(forums[:n_val_forum]); train_forums = set(forums[n_val_forum:])
    val_pairs = {k: v for k, v in pairs.items() if forum_of(k) in val_forums}
    train_pairs = {k: v for k, v in pairs.items() if forum_of(k) in train_forums}
    assert not (val_forums & train_forums), "forum leak!"
    print(f"pairs: total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)} "
          f"| forums train={len(train_forums)} val={len(val_forums)} (disjoint)")

    # PRE (base) eval
    pre = eval_pairs(model, processor, val_pairs, device)
    print(f"[PRE/base] Targeted={pre['TargetedAcc']:.3f}{pre['_tgt']} Original={pre['OriginalAcc']:.3f}{pre['_orig']} PairCausal={pre['PairCausalAcc']:.3f}{pre['_pair']}")

    if args.mode == "base":
        result = {"mode": "base", "n_val": len(val_pairs), "pre": pre, "post": pre}
        json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2)
        print("RESULT(base):", json.dumps(pre, indent=2)); return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_log = []
    step = 0
    for ep in range(args.epochs):
        model.train()
        order = list(train_pairs.keys()); random.shuffle(order)
        opt.zero_grad()
        for i, pid in enumerate(order):
            pr = train_pairs[pid]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                l_act = ce_loss(model, processor, pr["s"], device)      # original(s): click
                if args.mode == "lact_ltarget":
                    # L_target: 对 edited(s_c) 加权翻转监督 (lambda_target)
                    l_tgt = ce_loss(model, processor, pr["sc"], device)
                    loss = (l_act + args.lambda_target * l_tgt) / args.grad_accum
                elif args.mode == "lact_both":
                    # 公平基线: s 和 s_c 都训但等权 (见过 counterfactual 数据, 无特殊 target 加权)
                    l_tgt = ce_loss(model, processor, pr["sc"], device)
                    loss = (l_act + l_tgt) / args.grad_accum
                else:  # lact_only: 只训 s 侧 (无 counterfactual 数据)
                    l_tgt = torch.tensor(0.0)
                    loss = l_act / args.grad_accum
            if not torch.isfinite(loss):
                print(f"  [skip] non-finite ep{ep} step{step}"); opt.zero_grad(); step += 1; continue
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            step += 1
            loss_log.append({"step": step, "l_act": float(l_act.detach()),
                             "l_tgt": float(l_tgt.detach()) if torch.is_tensor(l_tgt) else 0.0})
            if step % 10 == 0:
                print(f"  ep{ep} step{step} L_act={float(l_act):.3f} L_tgt={float(l_tgt):.3f}")
        ev = eval_pairs(model, processor, val_pairs, device)
        print(f"[EP{ep}] Targeted={ev['TargetedAcc']:.3f} Original={ev['OriginalAcc']:.3f} PairCausal={ev['PairCausalAcc']:.3f}")

    post = eval_pairs(model, processor, val_pairs, device)
    print(f"[POST/{args.mode}] Targeted={post['TargetedAcc']:.3f}{post['_tgt']} Original={post['OriginalAcc']:.3f}{post['_orig']} PairCausal={post['PairCausalAcc']:.3f}{post['_pair']}")
    result = {"mode": args.mode, "lambda_target": args.lambda_target,
              "n_train": len(train_pairs), "n_val": len(val_pairs),
              "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
              "pre": pre, "post": post, "loss_log": loss_log}
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2)
    if os.environ.get("CIV_SAVE_CKPT", "0") == "1":
        ck = os.path.join(args.out, "ckpt")
        model.save_pretrained(ck); processor.save_pretrained(ck)
        print(f"saved checkpoint -> {ck}")
    print("RESULT:", json.dumps({"pre": pre, "post": post}, indent=2))


if __name__ == "__main__":
    main()
