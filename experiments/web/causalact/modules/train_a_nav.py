"""E2: 训练 A (SmolVLM-500M) 做导航 grounding — 从 SoM 候选里选正确帖子的 comments 链接。

数据: nav_set.jsonl (intent=导航到标题X的帖子, GT=该帖 comments 链接 id, 候选=同页所有帖子 bbox)。
A 输入: 截图 + intent + 候选(id:bbox) → 输出 正确 elem_id。
grounding: A 必须读截图在 bbox 处的标题文字, 匹配 intent 的标题 → 选对 id。

forum-disjoint 切 train/val (防图像级泄漏: 同 forum 多页共享布局)。
AMP(fp16 compute + fp32 master) + 冻结视觉塔 (I-1 验证的稳定配置)。
val 选 best checkpoint (grounding accuracy)。
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def build_prompt(processor, intent, cands):
    lines = "\n".join(f"{c['id']}: comments link at box {c['bbox']}" for c in cands)
    instr = (f"You are a web navigation agent. The screenshot is a forum post list. "
             f"Each candidate is a 'comments' link for one post, given by its id and bounding box.\n"
             f"Goal: {intent}\n"
             f"Candidates:\n{lines}\n"
             f"Look at the screenshot to read each post's title near its comments link, "
             f"and pick the comments link of the post whose title matches the goal. "
             f"Answer with ONLY the element id (e.g. elem_3).")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def cands_of(s):
    return s.oracle_metadata.get("candidates", [])


def ce_loss(model, processor, s, device):
    img = Image.open(s.screenshot_path).convert("RGB")
    prompt = build_prompt(processor, s.intent, cands_of(s))
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
    gt = s.original_action.target_id
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
def eval_acc(model, processor, samples, device):
    model.eval()
    correct = 0; total = 0
    for s in samples:
        img = Image.open(s.screenshot_path).convert("RGB")
        prompt = build_prompt(processor, s.intent, cands_of(s))
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(device)
        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        pred = text.split()[0] if text.split() else ""
        gt = s.original_action.target_id
        total += 1
        if gt == pred or gt in pred:
            correct += 1
    return correct / total if total else 0.0, correct, total


def forum_split(samples, val_frac=0.25, seed=42):
    forums = sorted(set(s.oracle_metadata["forum"] for s in samples))
    random.Random(seed).shuffle(forums)
    n_val = max(4, int(len(forums) * val_frac))
    val_forums = set(forums[:n_val])
    train = [s for s in samples if s.oracle_metadata["forum"] not in val_forums]
    val = [s for s in samples if s.oracle_metadata["forum"] in val_forums]
    return train, val, val_forums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=location('CAUSALACT_RUN_DIR', 'web/data/nav_set.jsonl'))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=location('CAUSALACT_RUN_DIR', 'a_nav_ckpt'))
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
    train, val, val_forums = forum_split(samples, seed=args.seed)
    print(f"total={len(samples)} train={len(train)} val={len(val)} val_forums={sorted(val_forums)}")

    pre_acc, pc, pt = eval_acc(model, processor, val, device)
    print(f"[PRE] val grounding acc = {pre_acc:.4f} ({pc}/{pt})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best_acc = pre_acc; best_ep = -1; hist = [{"epoch": -1, "val_acc": pre_acc}]
    for ep in range(args.epochs):
        model.train(); random.shuffle(train); opt.zero_grad()
        for i, s in enumerate(train):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = ce_loss(model, processor, s, device) / args.grad_accum
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        acc, c, t = eval_acc(model, processor, val, device)
        hist.append({"epoch": ep, "val_acc": acc})
        print(f"[EP{ep}] val grounding acc = {acc:.4f} ({c}/{t})")
        if args.save_best and acc > best_acc:
            best_acc = acc; best_ep = ep
            ck = os.path.join(args.out, "ckpt")
            model.save_pretrained(ck); processor.save_pretrained(ck)
            print(f"  saved best ckpt (acc={acc:.4f}) -> {ck}")
    result = {"pre_acc": pre_acc, "best_acc": best_acc, "best_ep": best_ep,
              "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
              "n_train": len(train), "n_val": len(val), "val_forums": sorted(val_forums),
              "history": hist}
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=2)
    print("RESULT:", json.dumps({"pre": pre_acc, "best": best_acc, "best_ep": best_ep}, indent=2))


if __name__ == "__main__":
    main()
