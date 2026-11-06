"""Deliverable 2: Verifier 重训, 对称官方 SoM 输入 (之前 V=50% 随机疑似同一输入 bug)。

数据: official_nav_set (官方 SoM 图)。每样本 → 1 正(query=GT, label yes) + 1 负(query=别帖, label no)。
A/V 看官方 SoM 图(框可见) + query 候选 id, 判"点这个对不对"。forum-disjoint, val 选 best。
看 V acc 是否从 50% 翻转。
"""
from causalact_runtime import location
import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
NAV = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')


def load(p): return [json.loads(l) for l in open(p) if l.strip()]


def prompt_text(intent, qid):
    return (f"You are a web navigation verifier. The screenshot is a forum post list with "
            f"numbered Set-of-Mark boxes on clickable elements.\n"
            f"An agent wants to: {intent}\n"
            f"The agent is about to click the comments link marked {qid}.\n"
            f"Look at the numbered boxes to read the post title next to box {qid}. "
            f"Is clicking {qid} the correct action for the goal? Answer ONLY 'yes' or 'no'.")


def build_prompt(proc, intent, qid):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text(intent, qid)}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)


def make_ex(rows, seed=42):
    rng = random.Random(seed); ex = []
    for r in rows:
        cids = r["oracle_metadata"]["candidate_ids"]; gt = r["original_action"]["target_id"]
        if gt not in cids: continue
        ex.append({"r": r, "qid": gt, "label": "yes"})
        negs = [c for c in cids if c != gt]; rng.shuffle(negs)
        if negs: ex.append({"r": r, "qid": negs[0], "label": "no"})
    return ex


def ce(model, proc, e, device):
    r = e["r"]; img = Image.open(r["screenshot_path"]).convert("RGB")
    inp = proc(text=build_prompt(proc, r["intent"], e["qid"]), images=[img], return_tensors="pt").to(device)
    ans = proc.tokenizer(e["label"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full = torch.cat([inp["input_ids"], ans], 1); attn = torch.cat([inp["attention_mask"], torch.ones_like(ans)], 1)
    out = model(input_ids=full, attention_mask=attn, pixel_values=inp.get("pixel_values"), pixel_attention_mask=inp.get("pixel_attention_mask"))
    n = ans.shape[1]; lg = out.logits[:, -n-1:-1, :]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), ans.reshape(-1))


@torch.no_grad()
def ev(model, proc, exs, device):
    model.eval(); tp=tn=fp=fn=0
    for e in exs:
        r = e["r"]; img = Image.open(r["screenshot_path"]).convert("RGB")
        inp = proc(text=build_prompt(proc, r["intent"], e["qid"]), images=[img], return_tensors="pt").to(device)
        g = model.generate(**inp, max_new_tokens=4, do_sample=False)
        t = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        py = t.startswith("y"); gy = (e["label"]=="yes")
        if py and gy: tp+=1
        elif py and not gy: fp+=1
        elif not py and gy: fn+=1
        else: tn+=1
    n=tp+tn+fp+fn
    return {"acc":(tp+tn)/n if n else 0, "neg_recall": tn/(tn+fp) if (tn+fp) else 0,
            "neg_prec": tn/(tn+fn) if (tn+fn) else 0, "tp":tp,"tn":tn,"fp":fp,"fn":fn,"n":n}


def fsplit(rows, seed=42, vf=0.25):
    fs=sorted(set(r["oracle_metadata"]["forum"] for r in rows)); random.Random(seed).shuffle(fs)
    nv=max(4,int(len(fs)*vf)); v=set(fs[:nv])
    return [r for r in rows if r["oracle_metadata"]["forum"] not in v],[r for r in rows if r["oracle_metadata"]["forum"] in v],v


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--grad_accum",type=int,default=8); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default=location('CAUSALACT_RUN_DIR', 'v_official_ckpt')); ap.add_argument("--save_best",action="store_true")
    a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out,exist_ok=True); device="cuda"
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc=AutoProcessor.from_pretrained(MODEL_ID)
    model=AutoModelForVision2Seq.from_pretrained(MODEL_ID,torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    for p in model.model.vision_model.parameters(): p.requires_grad=False
    scaler=torch.cuda.amp.GradScaler()
    rows=load(NAV); tr,va,vf=fsplit(rows,a.seed)
    tre=make_ex(tr,a.seed); vae=make_ex(va,a.seed)
    print(f"train ex={len(tre)} val ex={len(vae)} val_forums={sorted(vf)}")
    pre=ev(model,proc,vae,device); print(f"[PRE] {json.dumps(pre)}")
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    best=pre["acc"]; best_ep=-1; hist=[{"epoch":-1,**pre}]
    for ep in range(a.epochs):
        model.train(); random.shuffle(tre); opt.zero_grad()
        for i,e in enumerate(tre):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss=ce(model,proc,e,device)/a.grad_accum
            if not torch.isfinite(loss): opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (i+1)%a.grad_accum==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        e2=ev(model,proc,vae,device); hist.append({"epoch":ep,**e2})
        print(f"[EP{ep}] {json.dumps(e2)}")
        if a.save_best and e2["acc"]>best:
            best=e2["acc"]; best_ep=ep; ck=os.path.join(a.out,"ckpt"); model.save_pretrained(ck); proc.save_pretrained(ck); print(f"  saved best {e2['acc']:.3f}")
    json.dump({"pre":pre,"best_acc":best,"best_ep":best_ep,"history":hist},open(os.path.join(a.out,"result.json"),"w"),indent=2)
    print("DONE pre_acc",pre["acc"],"best",best,"ep",best_ep)


if __name__=="__main__": main()
