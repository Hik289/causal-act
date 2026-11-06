"""E2 核心: 训练 A 作 Verifier-gated corrector 的 V 部分 —— 判断 base 选的导航目标对不对。

设计依据 (决定性诊断): base 导航 grounding 93%, A grounding 仅 30%。
→ A 不能当"重新 grounding 的 refiner"(会把 93%换成30%)。
→ A 当 **Verifier**: 看 (截图, intent, base 选的目标框) → 判断"这个目标对不对"(二元)。
   这是聚焦的 yes/no (单目标验证), 可能比 5 选 1 grounding 容易。
   corrector 只在 V 高置信说"base 错"时才介入。

训练数据 (从 nav_set 构造, Oracle 核验标签):
  正例 (base_correct=True): query 目标 = GT 目标 → label "yes"(对)
  负例 (base_correct=False): query 目标 = 同页另一个帖子 → label "no"(错)
  标签由 DOM GT 确定性给出 (环境核验), 不自造。

测 val: V 的 precision/recall/acc 在判断"目标对不对"上。
这是 go/no-go: V 若不能可靠检测 base 错误, corrector 无法净正增益。
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def build_prompt(processor, intent, query_cand):
    instr = (f"You are a web navigation verifier. The screenshot is a forum post list. "
             f"An agent wants to achieve: {intent}\n"
             f"The agent is about to click the comments link at bounding box {query_cand['bbox']} "
             f"(candidate id {query_cand['id']}).\n"
             f"Look at the screenshot to read the post title near that box. "
             f"Is clicking THIS comments link the correct action for the goal? "
             f"Answer ONLY 'yes' or 'no'.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def make_examples(samples, neg_per_pos=1, seed=42):
    """每个 nav 样本 → 1 正(query=GT) + neg_per_pos 负(query=同页其它帖)。"""
    rng = random.Random(seed)
    ex = []
    for s in samples:
        cands = s.oracle_metadata.get("candidates", [])
        gt = s.original_action.target_id
        gt_c = next((c for c in cands if c["id"] == gt), None)
        if gt_c is None:
            continue
        ex.append({"sample": s, "query": gt_c, "label": "yes"})
        negs = [c for c in cands if c["id"] != gt]
        rng.shuffle(negs)
        for c in negs[:neg_per_pos]:
            ex.append({"sample": s, "query": c, "label": "no"})
    return ex


def ce_loss(model, processor, e, device):
    s = e["sample"]
    img = Image.open(s.screenshot_path).convert("RGB")
    prompt = build_prompt(processor, s.intent, e["query"])
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    ans_ids = processor.tokenizer(e["label"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([inputs["input_ids"], ans_ids], dim=1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(ans_ids)], dim=1)
    out = model(input_ids=full, attention_mask=attn,
                pixel_values=inputs.get("pixel_values"), pixel_attention_mask=inputs.get("pixel_attention_mask"))
    n = ans_ids.shape[1]
    lg = out.logits[:, -n - 1:-1, :]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), ans_ids.reshape(-1))


@torch.no_grad()
def eval_v(model, processor, exs, device):
    model.eval()
    tp = tn = fp = fn = 0
    for e in exs:
        s = e["sample"]
        img = Image.open(s.screenshot_path).convert("RGB")
        prompt = build_prompt(processor, s.intent, e["query"])
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
        gen = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        t = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        pred_yes = t.startswith("y")
        gold_yes = (e["label"] == "yes")
        if pred_yes and gold_yes: tp += 1
        elif pred_yes and not gold_yes: fp += 1
        elif not pred_yes and gold_yes: fn += 1
        else: tn += 1
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0
    # "detect base wrong" = predict 'no' on a wrong target. 负类(label=no) 的 recall/precision
    neg_recall = tn / (tn + fp) if (tn + fp) else 0   # of all wrong targets, how many flagged 'no'
    neg_prec = tn / (tn + fn) if (tn + fn) else 0     # of all 'no' predictions, how many truly wrong
    return {"acc": acc, "neg_recall": neg_recall, "neg_prec": neg_prec,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}


def forum_split(samples, val_frac=0.25, seed=42):
    forums = sorted(set(s.oracle_metadata["forum"] for s in samples))
    random.Random(seed).shuffle(forums)
    nval = max(4, int(len(forums) * val_frac))
    vf = set(forums[:nval])
    return [s for s in samples if s.oracle_metadata["forum"] not in vf], \
           [s for s in samples if s.oracle_metadata["forum"] in vf], vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=location('CAUSALACT_RUN_DIR', 'web/data/nav_set.jsonl'))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=location('CAUSALACT_RUN_DIR', 'v_verifier_ckpt'))
    ap.add_argument("--save_best", action="store_true")
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
    tr_s, va_s, vf = forum_split(samples, seed=args.seed)
    train = make_examples(tr_s, neg_per_pos=1, seed=args.seed)
    val = make_examples(va_s, neg_per_pos=1, seed=args.seed)
    print(f"train ex={len(train)} val ex={len(val)} val_forums={sorted(vf)}")

    pre = eval_v(model, processor, val, device)
    print(f"[PRE] {json.dumps(pre)}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = pre["acc"]; best_ep = -1; hist = [{"epoch": -1, **pre}]
    for ep in range(args.epochs):
        model.train(); random.shuffle(train); opt.zero_grad()
        for i, e in enumerate(train):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = ce_loss(model, processor, e, device) / args.grad_accum
            if not torch.isfinite(loss): opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        ev = eval_v(model, processor, val, device)
        hist.append({"epoch": ep, **ev})
        print(f"[EP{ep}] {json.dumps(ev)}")
        if args.save_best and ev["acc"] > best:
            best = ev["acc"]; best_ep = ep
            ck = os.path.join(args.out, "ckpt"); model.save_pretrained(ck); processor.save_pretrained(ck)
            print(f"  saved best acc={ev['acc']:.3f}")
    json.dump({"pre": pre, "best_acc": best, "best_ep": best_ep, "history": hist,
               "lr": args.lr, "epochs": args.epochs}, open(os.path.join(args.out, "result.json"), "w"), indent=2)
    print("DONE best_acc", best, "ep", best_ep)


if __name__ == "__main__":
    main()
