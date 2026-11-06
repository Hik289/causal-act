"""E2 修复版: 训练 A 导航 grounding, 用**官方 SoM 图**(与 base 输入对称)。

与 train_a_nav 区别: 图是 VWA 官方 SoM 标注图(框画在像素上), 候选用 candidate_ids(官方 SoM id),
prompt 让 A 看图上编号框匹配标题。forum-disjoint, AMP+冻结视觉塔, val 选 best。
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def prompt_text(intent, cand_ids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\n"
            f"Goal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cand_ids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")




def _cids(r):
    om = r["oracle_metadata"]
    if "candidate_ids" in om:
        return [str(x) for x in om["candidate_ids"]]
    return [str(c["id"]) for c in om.get("candidates", [])]

def build_prompt(proc, r):
    msgs = [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": prompt_text(r["intent"], _cids(r))}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)


def ce_loss(model, proc, r, device):
    img = Image.open(r["screenshot_path"]).convert("RGB")
    prompt = build_prompt(proc, r)
    inputs = proc(text=prompt, images=[img], return_tensors="pt").to(device)
    gt = str(r["original_action"]["target_id"])
    ans = proc.tokenizer(gt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([inputs["input_ids"], ans], dim=1)
    attn = torch.cat([inputs["attention_mask"], torch.ones_like(ans)], dim=1)
    out = model(input_ids=full, attention_mask=attn, pixel_values=inputs.get("pixel_values"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"))
    n = ans.shape[1]; lg = out.logits[:, -n - 1:-1, :]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), ans.reshape(-1))


@torch.no_grad()
def eval_acc(model, proc, rows, device):
    model.eval(); corr = tot = 0
    for r in rows:
        img = Image.open(r["screenshot_path"]).convert("RGB")
        inputs = proc(text=build_prompt(proc, r), images=[img], return_tensors="pt").to(device)
        g = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        t = proc.batch_decode(g[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        import re as _re
        gt = str(r["original_action"]["target_id"]); tot += 1
        _m = _re.search(r"\d+", t)
        pred = _m.group(0) if _m else t.strip()
        _gtnum = gt.split("_")[1] if "_" in gt else gt
        if pred == _gtnum:
            corr += 1
    return corr / tot if tot else 0, corr, tot


def forum_split(rows, val_frac=0.25, seed=42):
    forums = sorted(set(r["oracle_metadata"]["forum"] for r in rows))
    random.Random(seed).shuffle(forums)
    nval = max(4, int(len(forums) * val_frac)); vf = set(forums[:nval])
    return [r for r in rows if r["oracle_metadata"]["forum"] not in vf], \
           [r for r in rows if r["oracle_metadata"]["forum"] in vf], vf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt'))
    ap.add_argument("--save_best", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    for p in model.model.vision_model.parameters():
        p.requires_grad = False
    scaler = torch.cuda.amp.GradScaler()

    rows = load(NAV)
    tr, va, vf = forum_split(rows, seed=args.seed)
    print(f"total={len(rows)} train={len(tr)} val={len(va)} val_forums={sorted(vf)}")
    pre, pc, pt = eval_acc(model, proc, va, device)
    print(f"[PRE] val grounding acc = {pre:.4f} ({pc}/{pt})")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = pre; best_ep = -1; hist = [{"epoch": -1, "val_acc": pre}]
    for ep in range(args.epochs):
        model.train(); random.shuffle(tr); opt.zero_grad()
        for i, r in enumerate(tr):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = ce_loss(model, proc, r, device) / args.grad_accum
            if not torch.isfinite(loss): opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        acc, c, t = eval_acc(model, proc, va, device); hist.append({"epoch": ep, "val_acc": acc})
        print(f"[EP{ep}] val grounding acc = {acc:.4f} ({c}/{t})")
        if args.save_best and acc > best:
            best = acc; best_ep = ep
            ck = os.path.join(args.out, "ckpt"); model.save_pretrained(ck); proc.save_pretrained(ck)
            print(f"  saved best acc={acc:.3f}")
    json.dump({"pre": pre, "best_acc": best, "best_ep": best_ep, "history": hist,
               "lr": args.lr, "epochs": args.epochs, "n_train": len(tr), "n_val": len(va)},
              open(os.path.join(args.out, "result.json"), "w"), indent=2)
    print("DONE pre", pre, "best", best, "ep", best_ep)


if __name__ == "__main__":
    main()
