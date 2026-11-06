"""Train outcome-V (SmolVLM-500M): predict if a base step belongs to a FAILING trajectory.
Multi-epoch, task-disjoint split, per-epoch val eval with precision@highconf.
Goal: find an epoch+tau where gating (intervene when V says 'failing' with conf>=tau)
could net-positive. We report val precision of 'fail' detection at high conf —
high precision = can gate without hurting success trajectories.
"""
from causalact_runtime import location
import os, json, argparse, random
import torch, torch.nn.functional as F
from PIL import Image
MODEL_ID="HuggingFaceTB/SmolVLM-500M-Instruct"
DATA=location('CAUSALACT_RUN_DIR', 'outcome_set.jsonl')
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def vprompt(instr, action):
    return ("You are a task-outcome verifier for a GUI agent. Screenshot = current state. "
            f"Action the agent just took: {action[:200]}\n"
            "Based on the state + action, is this trajectory FAILING (agent is stuck/doing "
            "wrong things/will not complete the task)? Answer ONLY 'fail' or 'ok'.")
def bp(proc,r):
    m=[{"role":"user","content":[{"type":"image"},{"type":"text","text":vprompt(r.get("domain",""),r["action"])}]}]
    return proc.apply_chat_template(m, add_generation_prompt=True)
def lab(r): return "fail" if r["gt_fail"] else "ok"
def img(r):
    im=Image.open(r["screenshot"]).convert("RGB"); im.thumbnail((512,512)); return [im]
def ce(model,proc,r,dev):
    inp=proc(text=bp(proc,r),images=img(r),return_tensors="pt").to(dev)
    ans=proc.tokenizer(lab(r),add_special_tokens=False,return_tensors="pt").input_ids.to(dev)
    full=torch.cat([inp["input_ids"],ans],1);attn=torch.cat([inp["attention_mask"],torch.ones_like(ans)],1)
    out=model(input_ids=full,attention_mask=attn,pixel_values=inp.get("pixel_values"),pixel_attention_mask=inp.get("pixel_attention_mask"))
    n=ans.shape[1];lg=out.logits[:,-n-1:-1,:]
    return F.cross_entropy(lg.reshape(-1,lg.size(-1)),ans.reshape(-1))
@torch.no_grad()
def ev(model,proc,rows,dev):
    model.eval();tp=tn=fp=fn=0
    for r in rows:
        inp=proc(text=bp(proc,r),images=img(r),return_tensors="pt").to(dev)
        g=model.generate(**inp,max_new_tokens=3,do_sample=False)
        t=proc.batch_decode(g[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip().lower()
        pf=t.startswith("f");gf=r["gt_fail"]
        if pf and gf:tp+=1
        elif pf and not gf:fp+=1
        elif not pf and gf:fn+=1
        else:tn+=1
    n=tp+tn+fp+fn
    return {"acc":(tp+tn)/n if n else 0,"fail_prec":tp/(tp+fp) if (tp+fp) else 0,
            "fail_recall":tp/(tp+fn) if (tp+fn) else 0,"tp":tp,"tn":tn,"fp":fp,"fn":fn}
def tsplit(rows,seed=42,vf=0.25):
    ts=sorted(set((r["domain"],r["task_id"]) for r in rows));random.Random(seed).shuffle(ts)
    nv=max(10,int(len(ts)*vf));v=set(ts[:nv])
    return [r for r in rows if (r["domain"],r["task_id"]) not in v],[r for r in rows if (r["domain"],r["task_id"]) in v]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--epochs",type=int,default=6);ap.add_argument("--lr",type=float,default=1e-5)
    ap.add_argument("--grad_accum",type=int,default=8);ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default=location('CAUSALACT_RUN_DIR', 'v_outcome_ckpt'));ap.add_argument("--cap",type=int,default=2500)
    a=ap.parse_args();random.seed(a.seed);torch.manual_seed(a.seed);os.makedirs(a.out,exist_ok=True);dev="cuda"
    from transformers import AutoProcessor,AutoModelForVision2Seq
    proc=AutoProcessor.from_pretrained(MODEL_ID)
    model=AutoModelForVision2Seq.from_pretrained(MODEL_ID,torch_dtype=torch.float32).to(dev)
    model.gradient_checkpointing_enable()
    for p in model.model.vision_model.parameters(): p.requires_grad=False
    scaler=torch.cuda.amp.GradScaler()
    rows=load(DATA);tr,va=tsplit(rows,a.seed)
    # balance train (downsample majority fail)
    random.shuffle(tr)
    fails=[r for r in tr if r["gt_fail"]];oks=[r for r in tr if not r["gt_fail"]]
    k=min(len(fails),len(oks)*2)  # keep some imbalance but not 4:1
    tr=oks+fails[:k];random.shuffle(tr)
    if len(tr)>a.cap: tr=tr[:a.cap]
    print(f"total={len(rows)} train={len(tr)}(ok={len(oks)},fail={min(len(fails),k)}) val={len(va)}",flush=True)
    pre=ev(model,proc,va,dev);print(f"[PRE] {json.dumps(pre)}",flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    best=0;best_ep=-1;hist=[{"epoch":-1,**pre}]
    for ep in range(a.epochs):
        model.train();random.shuffle(tr);opt.zero_grad()
        for i,r in enumerate(tr):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss=ce(model,proc,r,dev)/a.grad_accum
            if not torch.isfinite(loss):opt.zero_grad();continue
            scaler.scale(loss).backward()
            if (i+1)%a.grad_accum==0:
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                scaler.step(opt);scaler.update();opt.zero_grad()
        e2=ev(model,proc,va,dev);hist.append({"epoch":ep,**e2});print(f"[EP{ep}] {json.dumps(e2)}",flush=True)
        # select best by fail_prec (high precision = safe gating) with recall>0.1
        score=e2["fail_prec"] if e2["fail_recall"]>=0.1 else 0
        if score>best:
            best=score;best_ep=ep;ck=os.path.join(a.out,"ckpt");model.save_pretrained(ck);proc.save_pretrained(ck);print(f"  saved best fail_prec={e2['fail_prec']:.3f} ep{ep}",flush=True)
    json.dump({"pre":pre,"best_fail_prec":best,"best_ep":best_ep,"history":hist},open(os.path.join(a.out,"result.json"),"w"),indent=2)
    print(f"DONE best_fail_prec={best} ep{best_ep}")
if __name__=="__main__": main()
