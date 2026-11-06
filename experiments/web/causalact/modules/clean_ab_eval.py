"""E2 清洁 apples-to-apples: 同一 val 集上测 A(ckpt) vs base(gpt-5.6-sol) 导航 grounding。
统一: 同数据文件 + 同 forum_split(seed42) + 同 prompt + 同 GT。
用法: python clean_ab_eval.py <A_ckpt_dir> <data.jsonl>
"""
import os, sys, json, random, re, io, base64
import torch
from PIL import Image

A_CKPT = sys.argv[1]
DATA = sys.argv[2]
rows = [json.loads(l) for l in open(DATA) if l.strip()]


def cand_ids(r):
    om = r["oracle_metadata"]
    if "candidate_ids" in om:
        return [str(x) for x in om["candidate_ids"]]
    return [str(c["id"]) for c in om.get("candidates", [])]


def gt(r):
    v=str(r["original_action"]["target_id"]); import re as _r; m=_r.search(r"\d+",v); return m.group(0) if m else v


def prompt_text(r):
    ids = cand_ids(r)
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\n"
            f"Goal: {r['intent']}\n"
            f"Candidate comments-link element ids: {', '.join(ids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id.")


def forum_split(rs, val_frac=0.25, seed=42):
    fs = sorted(set(r["oracle_metadata"]["forum"] for r in rs))
    random.Random(seed).shuffle(fs)
    nval = max(4, int(len(fs) * val_frac))
    vf = set(fs[:nval])
    return [r for r in rs if r["oracle_metadata"]["forum"] not in vf], \
           [r for r in rs if r["oracle_metadata"]["forum"] in vf], vf


tr, va, vf = forum_split(rows)
print(f"total={len(rows)} val={len(va)} val_forums={sorted(vf)}", flush=True)


def norm(t, ids):
    m = re.search(r"\d+", t)
    return m.group(0) if m else (t.strip() if t.strip() in ids else "")


# ---- A ----
from transformers import AutoProcessor, AutoModelForVision2Seq
proc = AutoProcessor.from_pretrained(A_CKPT)
m = AutoModelForVision2Seq.from_pretrained(A_CKPT, torch_dtype=torch.float16).to("cuda")
m.eval()
a_corr = 0
with torch.no_grad():
    for r in va:
        img = Image.open(r["screenshot_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(r)}]}]
        pr = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=pr, images=[img], return_tensors="pt").to("cuda")
        g = m.generate(**inp, max_new_tokens=6, do_sample=False)
        t = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if norm(t, cand_ids(r)) == gt(r):
            a_corr += 1
a_acc = a_corr / len(va)
print(f"A_ckpt val grounding acc = {a_acc:.4f} ({a_corr}/{len(va)})", flush=True)
del m
torch.cuda.empty_cache()

# ---- base ----
from openai import OpenAI
c = OpenAI(max_retries=4)
b_corr = 0
for r in va:
    img = Image.open(r["screenshot_path"]).convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        rr = c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"), messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_text(r)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}])
        t = (rr.choices[0].message.content or "").strip()
    except Exception:
        t = ""
    if norm(t, cand_ids(r)) == gt(r):
        b_corr += 1
b_acc = b_corr / len(va)
print(f"BASE val grounding acc = {b_acc:.4f} ({b_corr}/{len(va)})", flush=True)
print(json.dumps({"A_acc": a_acc, "base_acc": b_acc, "n_val": len(va), "delta_pp": (a_acc - b_acc) * 100}))
