"""Val 级离线模拟: base vs base+A(gated corrector) 在 nav grounding 上, 扫 tau。
用已有 official_nav_set val, 不跑慢端到端。直接答: A 纠正能否净正 + 最优 tau。
每样本: base grounding(经 proxy) + A grounding+conf(trained ckpt)。
B+A(tau) = A 分歧且 conf>=tau 用 A, 否则用 base。比 success(命中 GT)。
"""
from causalact_runtime import location
import os, sys, json, io, base64, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch

NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')
A_CKPT = location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt/ckpt')
MODEL_ID = A_CKPT


def load(p): return [json.loads(l) for l in open(p) if l.strip()]


def forum_val(rows, seed=42, val_frac=0.25):
    forums = sorted(set(r["oracle_metadata"]["forum"] for r in rows))
    random.Random(seed).shuffle(forums)
    nval = max(4, int(len(forums) * val_frac)); vf = set(forums[:nval])
    return [r for r in rows if r["oracle_metadata"]["forum"] in vf]


def prompt_text(intent, cand_ids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\n"
            f"Goal: {intent}\nCandidate comments-link element ids: {', '.join(cand_ids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")


def parse(text, cand_ids):
    for cid in cand_ids:
        if cid in text or cid.split("_")[1] in text.split():
            return cid
    import re
    m = re.search(r"\d+", text)
    if m and ("elem_" + m.group(0)) in cand_ids:
        return "elem_" + m.group(0)
    return None


def main():
    rows = forum_val(load(NAV))
    print("val n:", len(rows))
    # base via proxy
    from openai import OpenAI
    c = OpenAI(max_retries=4)
    # A trained
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    m = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda"); m.eval()

    recs = []
    for r in rows:
        img = Image.open(r["screenshot_path"]).convert("RGB")
        cand = r["oracle_metadata"]["candidate_ids"]; gt = r["original_action"]["target_id"]
        buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            resp = c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"), messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_text(r["intent"], cand)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}])
            bt = (resp.choices[0].message.content or "").strip()
        except Exception:
            bt = ""
        b_pred = parse(bt, cand)
        # A
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(r["intent"], cand)}]}]
        pr = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=pr, images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=8, do_sample=False, output_scores=True, return_dict_in_generate=True)
        at = proc.batch_decode(g.sequences[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        a_conf = float(torch.softmax(g.scores[0][0].float(), dim=-1).max())
        a_pred = parse(at, cand)
        recs.append({"gt": gt, "b": b_pred, "a": a_pred, "a_conf": a_conf})

    base_acc = sum(1 for x in recs if x["b"] == x["gt"]) / len(recs)
    a_acc = sum(1 for x in recs if x["a"] == x["gt"]) / len(recs)
    print(f"base nav acc={base_acc:.3f}  A nav acc={a_acc:.3f}")
    # scan tau for B+A gated corrector
    out = {"n": len(recs), "base_acc": base_acc, "a_acc": a_acc, "tau_scan": []}
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        ba = 0
        for x in recs:
            use_a = (x["a"] is not None and x["a"] != x["b"] and x["a_conf"] >= tau)
            pred = x["a"] if use_a else x["b"]
            if pred == x["gt"]:
                ba += 1
        ba_acc = ba / len(recs)
        out["tau_scan"].append({"tau": tau, "ba_acc": ba_acc, "delta_vs_base": ba_acc - base_acc})
        print(f"  tau={tau} B+A acc={ba_acc:.3f} delta={ba_acc-base_acc:+.3f}")
    json.dump({**out, "recs": recs}, open(location('CAUSALACT_RUN_DIR', 'web/offline_refine_sim.json'), "w"), indent=2)
    print("saved offline_refine_sim.json")


if __name__ == "__main__":
    main()
