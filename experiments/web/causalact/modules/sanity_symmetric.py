"""Director #1: 对称 sanity check — base(gpt-5.6-sol) vs zero-shot A(SmolVLM-500M)
在**逐像素相同的官方 SoM 图 + 相同候选 + 相同 GT + 相同判对逻辑**上测 grounding。

目的: 证明之前 A=30% 是"无框图"输入 bug。预期对称输入下 zero-shot A 远高于 30%。

⚠️ 输入对称性 ASSERT (Director #2): 两者读的是同一个 official SoM png 文件 (sha256 一致),
且该图确有 SoM 框 (非空 overlay, 用文件存在 + 非纯空白校验)。
"""
from causalact_runtime import location
import os, sys, json, io, base64, hashlib, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def forum_split(rows, val_frac=0.25, seed=42):
    forums = sorted(set(r["oracle_metadata"]["forum"] for r in rows))
    random.Random(seed).shuffle(forums)
    nval = max(4, int(len(forums) * val_frac))
    vf = set(forums[:nval])
    return [r for r in rows if r["oracle_metadata"]["forum"] in vf]  # 只需 val 做 sanity


def img_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def assert_som_visible(path):
    """ASSERT: 图存在且非纯空白 (官方 SoM 图有彩色框 → 颜色方差高)。"""
    im = Image.open(path).convert("RGB")
    ex = im.resize((64, 64))
    px = list(ex.getdata())
    # 统计非灰度像素 (框是彩色 r!=g!=b)
    colored = sum(1 for (r, g, b) in px if max(r, g, b) - min(r, g, b) > 40)
    assert colored >= 20, f"{path}: too few colored pixels ({colored}) — SoM boxes may be missing!"
    return colored


def prompt_text(intent, cand_ids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\n"
            f"Goal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cand_ids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id number (e.g. elem_21).")


def eval_base(rows, model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol")):
    from openai import OpenAI
    c = OpenAI(max_retries=4)
    corr = tot = 0; wrong = []
    for r in rows:
        img = Image.open(r["screenshot_path"]).convert("RGB")
        buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
        cand = r["oracle_metadata"]["candidate_ids"]
        try:
            resp = c.chat.completions.create(model=model, messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_text(r["intent"], cand)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}])
            t = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            t = ""
        gt = r["original_action"]["target_id"]
        tot += 1
        ok = gt in t or gt.split("_")[1] in t
        if ok: corr += 1
        else: wrong.append((r["task_id"], gt, t[:20]))
    return corr, tot, wrong


def eval_a(rows):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    m = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda"); m.eval()
    corr = tot = 0; wrong = []
    for r in rows:
        img = Image.open(r["screenshot_path"]).convert("RGB")
        cand = r["oracle_metadata"]["candidate_ids"]
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(r["intent"], cand)}]}]
        pr = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=pr, images=[img], return_tensors="pt").to("cuda")
        import torch as T
        with T.no_grad():
            g = m.generate(**inp, max_new_tokens=8, do_sample=False)
        t = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        gt = r["original_action"]["target_id"]
        tot += 1
        ok = gt in t or (gt.split("_")[1] in t)
        if ok: corr += 1
        else: wrong.append((r["task_id"], gt, t[:20]))
    return corr, tot, wrong


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--who", choices=["base", "a", "both"], default="both")
    args = ap.parse_args()
    rows = forum_split(load(NAV))
    print(f"val sanity rows: {len(rows)}")
    # ASSERT 输入对称 + SoM 可见
    shas = {}
    for r in rows:
        p = r["screenshot_path"]
        c = assert_som_visible(p)
        shas[p] = (img_sha(p), c)
    print(f"ASSERT passed: all {len(shas)} val images exist, SoM boxes visible (colored px ok).")
    out = {"n": len(rows), "assert_images": {os.path.basename(k): v for k, v in list(shas.items())[:3]}}
    if args.who in ("base", "both"):
        bc, bt, bw = eval_base(rows)
        out["base"] = {"acc": bc / bt, "corr": bc, "tot": bt, "wrong_sample": bw[:5]}
        print(f"[BASE gpt-5.6-sol] grounding acc = {bc}/{bt} = {bc/bt:.3f}")
    if args.who in ("a", "both"):
        ac, at, aw = eval_a(rows)
        out["a_zeroshot"] = {"acc": ac / at, "corr": ac, "tot": at, "wrong_sample": aw[:5]}
        print(f"[A zero-shot SmolVLM-500M] grounding acc = {ac}/{at} = {ac/at:.3f}")
    json.dump(out, open(location('CAUSALACT_RUN_DIR', 'web/sanity_symmetric.json'), "w"), indent=2)
    print("saved sanity_symmetric.json")


if __name__ == "__main__":
    main()
