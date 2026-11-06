"""Val 离线模拟 V-gated corrector (cheap, 决定端到端前先验证能否净正)。
每 val 样本: base grounding(proxy) + A grounding(ckpt) + V 判 base_pred 对不对(V ckpt)。
B+A 策略: 若 V 判 base_pred='no'(错) 且 A_pred != base_pred → 用 A_pred; 否则透传 base。
比 success(命中 GT)。报 base vs B+A(V-gated) delta。
"""
from causalact_runtime import location
import os, sys, json, io, base64, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import torch

NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')
A_CKPT = location('CAUSALACT_RUN_DIR', 'a_nav_official_ckpt/ckpt')
V_CKPT = location('CAUSALACT_RUN_DIR', 'v_official_ckpt/ckpt')


def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def fval(rows, seed=42, vf=0.25):
    fs=sorted(set(r["oracle_metadata"]["forum"] for r in rows)); random.Random(seed).shuffle(fs)
    nv=max(4,int(len(fs)*vf)); v=set(fs[:nv]); return [r for r in rows if r["oracle_metadata"]["forum"] in v]

def nav_prompt(intent, cids):
    return (f"You are a web navigation agent. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes drawn on clickable elements.\nGoal: {intent}\n"
            f"Candidate comments-link element ids: {', '.join(cids)}\n"
            f"Look at the numbered boxes in the image to find the post whose title matches the goal, "
            f"and pick that post's comments link. Answer ONLY the element id (e.g. elem_21).")

def v_prompt(intent, qid):
    return (f"You are a web navigation verifier. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes on clickable elements.\nAn agent wants to: {intent}\n"
            f"The agent is about to click the comments link marked {qid}.\n"
            f"Look at the numbered boxes to read the post title next to box {qid}. "
            f"Is clicking {qid} the correct action for the goal? Answer ONLY 'yes' or 'no'.")

def parse(t, cids):
    for c in cids:
        if c in t or c.split("_")[1] in t.split(): return c
    import re; m=re.search(r"\d+", t)
    if m and ("elem_"+m.group(0)) in cids: return "elem_"+m.group(0)
    return None

def main():
    rows = fval(load(NAV)); print("val n:", len(rows))
    from openai import OpenAI
    c = OpenAI(max_retries=4)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    aproc = AutoProcessor.from_pretrained(A_CKPT); A = AutoModelForVision2Seq.from_pretrained(A_CKPT, torch_dtype=torch.float16).to("cuda"); A.eval()
    vproc = AutoProcessor.from_pretrained(V_CKPT); V = AutoModelForVision2Seq.from_pretrained(V_CKPT, torch_dtype=torch.float16).to("cuda"); V.eval()
    recs=[]
    for r in rows:
        img=Image.open(r["screenshot_path"]).convert("RGB"); cids=r["oracle_metadata"]["candidate_ids"]; gt=r["original_action"]["target_id"]
        buf=io.BytesIO(); img.save(buf,format="PNG"); b64=base64.b64encode(buf.getvalue()).decode()
        try:
            resp=c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"),messages=[{"role":"user","content":[{"type":"text","text":nav_prompt(r["intent"],cids)},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}])
            b_pred=parse((resp.choices[0].message.content or "").strip(),cids)
        except Exception: b_pred=None
        # A
        pr=aproc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":nav_prompt(r["intent"],cids)}]},],add_generation_prompt=True)
        inp=aproc(text=pr,images=[img],return_tensors="pt").to("cuda")
        with torch.no_grad(): g=A.generate(**inp,max_new_tokens=8,do_sample=False)
        a_pred=parse(aproc.batch_decode(g[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip(),cids)
        # V judges base_pred
        v_no=False
        if b_pred is not None:
            vp=vproc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":v_prompt(r["intent"],b_pred)}]},],add_generation_prompt=True)
            vi=vproc(text=vp,images=[img],return_tensors="pt").to("cuda")
            with torch.no_grad(): vg=V.generate(**vi,max_new_tokens=4,do_sample=False)
            vt=vproc.batch_decode(vg[:,vi["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip().lower()
            v_no = vt.startswith("n")
        recs.append({"gt":gt,"b":b_pred,"a":a_pred,"v_no":v_no})
    base_acc=sum(1 for x in recs if x["b"]==x["gt"])/len(recs)
    a_acc=sum(1 for x in recs if x["a"]==x["gt"])/len(recs)
    # V-gated: if V says base wrong and A differs -> use A
    bg=0; n_override=0; override_helped=0; override_hurt=0
    for x in recs:
        use_a = (x["v_no"] and x["a"] is not None and x["a"]!=x["b"])
        pred = x["a"] if use_a else x["b"]
        if use_a:
            n_override+=1
            if x["b"]!=x["gt"] and pred==x["gt"]: override_helped+=1
            if x["b"]==x["gt"] and pred!=x["gt"]: override_hurt+=1
        if pred==x["gt"]: bg+=1
    bg_acc=bg/len(recs)
    out={"n":len(recs),"base_acc":base_acc,"a_acc":a_acc,"vgated_ba_acc":bg_acc,
         "delta_vs_base":bg_acc-base_acc,"n_override":n_override,"override_helped":override_helped,"override_hurt":override_hurt}
    print(json.dumps(out,indent=2))
    json.dump({**out,"recs":recs},open(location('CAUSALACT_RUN_DIR', 'web/offline_vgated_sim.json'),"w"),indent=2)


if __name__=="__main__": main()
