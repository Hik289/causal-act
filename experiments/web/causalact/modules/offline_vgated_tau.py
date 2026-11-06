"""Val 调 V-gate tau 到净增益最大 (net = helps - hurts)。
全 94 样本: base_pred + a_pred + V对base_pred的'no'置信度。
V-gate: 若 V说base错(conf_no>=tau) 且 a!=base → 用 a。扫 tau, 报 helps/hurts/net/B+A acc。
val=forum-disjoint 子集选 tau (paper-grade); ALL 仅参考。
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
def val_forums(rows, seed=42, vf=0.25):
    fs=sorted(set(r["oracle_metadata"]["forum"] for r in rows)); random.Random(seed).shuffle(fs)
    return set(fs[:max(4,int(len(fs)*vf))])

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
    rows=load(NAV); vf=val_forums(rows)
    print(f"total={len(rows)} val_forums={sorted(vf)}", flush=True)
    from openai import OpenAI
    c=OpenAI(max_retries=4)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    aproc=AutoProcessor.from_pretrained(A_CKPT); A=AutoModelForVision2Seq.from_pretrained(A_CKPT,torch_dtype=torch.float16).to("cuda"); A.eval()
    vproc=AutoProcessor.from_pretrained(V_CKPT); V=AutoModelForVision2Seq.from_pretrained(V_CKPT,torch_dtype=torch.float16).to("cuda"); V.eval()
    recs=[]
    for i,r in enumerate(rows):
        img=Image.open(r["screenshot_path"]).convert("RGB"); cids=r["oracle_metadata"]["candidate_ids"]; gt=r["original_action"]["target_id"]
        buf=io.BytesIO(); img.save(buf,format="PNG"); b64=base64.b64encode(buf.getvalue()).decode()
        try:
            resp=c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"),messages=[{"role":"user","content":[{"type":"text","text":nav_prompt(r["intent"],cids)},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}])
            b_pred=parse((resp.choices[0].message.content or "").strip(),cids)
        except Exception: b_pred=None
        pr=aproc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":nav_prompt(r["intent"],cids)}]}],add_generation_prompt=True)
        inp=aproc(text=pr,images=[img],return_tensors="pt").to("cuda")
        with torch.no_grad(): g=A.generate(**inp,max_new_tokens=8,do_sample=False)
        a_pred=parse(aproc.batch_decode(g[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip(),cids)
        # V judges base_pred, get P(no)
        conf_no=0.0
        if b_pred is not None:
            vp=vproc.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":v_prompt(r["intent"],b_pred)}]}],add_generation_prompt=True)
            vi=vproc(text=vp,images=[img],return_tensors="pt").to("cuda")
            with torch.no_grad(): vg=V.generate(**vi,max_new_tokens=4,do_sample=False,output_scores=True,return_dict_in_generate=True)
            probs=torch.softmax(vg.scores[0][0].float(),dim=-1)
            # P(no): sum over tokens starting 'n'/'N'
            no_ids=[tid for tid in range(probs.shape[0]) if vproc.tokenizer.decode([tid]).strip().lower().startswith("n")]
            conf_no=float(probs[no_ids].sum()) if no_ids else 0.0
        recs.append({"forum":r["oracle_metadata"]["forum"],"in_val":r["oracle_metadata"]["forum"] in vf,
                     "gt":gt,"b":b_pred,"a":a_pred,"conf_no":conf_no})
        if (i+1)%20==0: print(f"  {i+1}/{len(rows)}",flush=True)

    def scan(sub,label):
        base_acc=sum(1 for x in sub if x["b"]==x["gt"])/len(sub)
        res=[]
        for tau in [0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
            helps=hurts=ba=0
            for x in sub:
                use_a=(x["conf_no"]>=tau and x["a"] is not None and x["a"]!=x["b"])
                pred=x["a"] if use_a else x["b"]
                if use_a and x["b"]!=x["gt"] and pred==x["gt"]: helps+=1
                if use_a and x["b"]==x["gt"] and pred!=x["gt"]: hurts+=1
                if pred==x["gt"]: ba+=1
            res.append({"tau":tau,"ba_acc":ba/len(sub),"delta":ba/len(sub)-base_acc,"helps":helps,"hurts":hurts,"net":helps-hurts})
        print(f"=== {label} (n={len(sub)}, base={base_acc:.3f}) ===")
        for x in res: print(f"  tau={x['tau']} B+A={x['ba_acc']:.3f} d={x['delta']:+.3f} helps={x['helps']} hurts={x['hurts']} net={x['net']}")
        return {"base_acc":base_acc,"scan":res}
    out={"n_all":len(recs),"ALL":scan(recs,"ALL"),"VAL":scan([x for x in recs if x["in_val"]],"VAL")}
    # best tau by val net
    best=max(out["VAL"]["scan"],key=lambda x:(x["net"],x["ba_acc"]))
    out["best_tau_val"]=best["tau"]; print("BEST tau (val net):",best)
    json.dump({**out,"recs":recs},open(location('CAUSALACT_RUN_DIR', 'web/offline_vgated_tau.json'),"w"),indent=2)


if __name__=="__main__": main()
