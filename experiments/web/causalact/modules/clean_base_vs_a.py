"""Director #1: 单一可比数对 — 同一次运行, 同 val, 同官方SoM图, 同 prompt, 同 GT, 同判对,
同时测 base(gpt-5.6-sol) 和 trained-A(official ckpt) nav grounding。消除 58.7 vs 96.2 歧义。

在 full 94 样本上跑(大 n), 也分 forum-disjoint val 子集单独报, 两个口径都给。
"""
from causalact_runtime import location
import os, sys, json, io, base64, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch

NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')
A_CKPT = location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt/ckpt')


def load(p): return [json.loads(l) for l in open(p) if l.strip()]


def val_forums(rows, seed=42, vf=0.25):
    fs = sorted(set(r["oracle_metadata"]["forum"] for r in rows)); random.Random(seed).shuffle(fs)
    nv = max(4, int(len(fs) * vf)); return set(fs[:nv])


def nav_prompt(intent, cids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\nGoal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")


def parse(t, cids):
    for c in cids:
        if c in t or c.split("_")[1] in t.split(): return c
    import re; m = re.search(r"\d+", t)
    if m and ("elem_" + m.group(0)) in cids: return "elem_" + m.group(0)
    return None


def main():
    rows = load(NAV)
    vf = val_forums(rows)
    print(f"total={len(rows)} val_forums={sorted(vf)}", flush=True)
    from openai import OpenAI
    c = OpenAI(max_retries=4)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(A_CKPT)
    A = AutoModelForVision2Seq.from_pretrained(A_CKPT, torch_dtype=torch.float16).to("cuda"); A.eval()
    recs = []
    for i, r in enumerate(rows):
        img = Image.open(r["screenshot_path"]).convert("RGB")
        cids = r["oracle_metadata"]["candidate_ids"]; gt = r["original_action"]["target_id"]
        buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            resp = c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"), messages=[{"role": "user", "content": [
                {"type": "text", "text": nav_prompt(r["intent"], cids)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}])
            b_pred = parse((resp.choices[0].message.content or "").strip(), cids)
        except Exception:
            b_pred = None
        pr = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": nav_prompt(r["intent"], cids)}]}], add_generation_prompt=True)
        inp = proc(text=pr, images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = A.generate(**inp, max_new_tokens=8, do_sample=False)
        a_pred = parse(proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), cids)
        recs.append({"forum": r["oracle_metadata"]["forum"], "in_val": r["oracle_metadata"]["forum"] in vf,
                     "n_cand": len(cids), "gt": gt, "b": b_pred, "a": a_pred})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rows)} done", flush=True)

    def acc(sub, who):
        if not sub: return None
        return sum(1 for x in sub if x[who] == x["gt"]) / len(sub)
    val = [x for x in recs if x["in_val"]]
    out = {
        "schema": "same run, same official SoM image, same prompt, same GT, same parse",
        "n_all": len(recs), "n_val": len(val),
        "ALL": {"base_acc": acc(recs, "b"), "a_acc": acc(recs, "a")},
        "VAL_forumdisjoint": {"base_acc": acc(val, "b"), "a_acc": acc(val, "a")},
        "avg_n_cand": sum(x["n_cand"] for x in recs) / len(recs),
    }
    print(json.dumps(out, indent=2))
    json.dump({**out, "recs": recs}, open(location('CAUSALACT_RUN_DIR', 'web/clean_base_vs_a.json'), "w"), indent=2)


if __name__ == "__main__":
    main()
