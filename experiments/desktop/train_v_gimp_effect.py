"""Train gimp effect-verifier V: given (prev screenshot, action, next screenshot),
did the action take EFFECT? Targets gimp base failure: 73% actions no-effect + loops.
task-disjoint split, AMP + frozen vision tower, val-select best.
"""
from causalact_runtime import location
import os, json, argparse, random
import torch, torch.nn.functional as F
from PIL import Image
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA = location('CAUSALACT_RUN_DIR', 'gimp_effect_set.jsonl')

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def v_prompt(action):
    return ("You are a GUI action-effect verifier. You see TWO screenshots: BEFORE (image 1) "
            "and AFTER (image 2) an action.\n"
            f"Action taken: {action}\n"
            "Did this action take a MEANINGFUL effect (screen changed: menu opened, dialog "
            "appeared, image/selection changed)? If the screen is essentially unchanged, the "
            "action had NO effect (it missed / misfired).\n"
            "Answer ONLY 'yes' (took effect) or 'no' (no effect).")

def build_prompt(proc, r):
    msgs=[{"role":"user","content":[{"type":"image"},{"type":"image"},
          {"type":"text","text":v_prompt(r["action"])}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True)

def label(r): return "yes" if r["gt_effect"] else "no"

def imgs(r):
    a=Image.open(r["prev_screenshot"]).convert("RGB")
    b=Image.open(r["next_screenshot"]).convert("RGB")
    # downscale to limit tokens/mem
    for im in (a,b):
        im.thumbnail((448,448))
    return [a,b]

def ce(model, proc, r, device):
    inp=proc(text=build_prompt(proc,r), images=imgs(r), return_tensors="pt").to(device)
    ans=proc.tokenizer(label(r), add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full=torch.cat([inp["input_ids"],ans],1); attn=torch.cat([inp["attention_mask"],torch.ones_like(ans)],1)
    out=model(input_ids=full, attention_mask=attn, pixel_values=inp.get("pixel_values"), pixel_attention_mask=inp.get("pixel_attention_mask"))
    n=ans.shape[1]; lg=out.logits[:,-n-1:-1,:]
    return F.cross_entropy(lg.reshape(-1,lg.size(-1)), ans.reshape(-1))

@torch.no_grad()
def ev(model, proc, rows, device):
    model.eval(); tp=tn=fp=fn=0
    for r in rows:
        inp=proc(text=build_prompt(proc,r), images=imgs(r), return_tensors="pt").to(device)
        g=model.generate(**inp, max_new_tokens=4, do_sample=False)
        t=proc.batch_decode(g[:,inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
        py=t.startswith("y"); gy=r["gt_effect"]
        if py and gy: tp+=1
        elif py and not gy: fp+=1
        elif not py and gy: fn+=1
        else: tn+=1
    n=tp+tn+fp+fn
    # "no-effect" is the key class to detect (for loop-breaking) -> report neg(no) precision/recall
    return {"acc":(tp+tn)/n if n else 0,
            "noeff_prec": tn/(tn+fn) if (tn+fn) else 0,
            "noeff_recall": tn/(tn+fp) if (tn+fp) else 0,
            "tp":tp,"tn":tn,"fp":fp,"fn":fn}

def tsplit(rows, seed=42, vf=0.3):
    ts=sorted(set(r["task_id"] for r in rows)); random.Random(seed).shuffle(ts)
    nv=max(3,int(len(ts)*vf)); v=set(ts[:nv])
    return [r for r in rows if r["task_id"] not in v],[r for r in rows if r["task_id"] in v],v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--grad_accum",type=int,default=4); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default=location('CAUSALACT_RUN_DIR', 'v_gimp_effect_ckpt')); ap.add_argument("--save_best",action="store_true")
    a=ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out,exist_ok=True); device="cuda"
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc=AutoProcessor.from_pretrained(MODEL_ID)
    model=AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    for p in model.model.vision_model.parameters(): p.requires_grad=False
    scaler=torch.cuda.amp.GradScaler()
    rows=load(DATA); tr,va,vf=tsplit(rows,a.seed)
    print(f"total={len(rows)} train={len(tr)} val={len(va)} val_tasks={len(vf)}", flush=True)
    pre=ev(model,proc,va,device); print(f"[PRE] {json.dumps(pre)}", flush=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    best=pre["acc"]; best_ep=-1; hist=[{"epoch":-1,**pre}]
    for ep in range(a.epochs):
        model.train(); random.shuffle(tr); opt.zero_grad()
        for i,r in enumerate(tr):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss=ce(model,proc,r,device)/a.grad_accum
            if not torch.isfinite(loss): opt.zero_grad(); continue
            scaler.scale(loss).backward()
            if (i+1)%a.grad_accum==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
        e2=ev(model,proc,va,device); hist.append({"epoch":ep,**e2}); print(f"[EP{ep}] {json.dumps(e2)}", flush=True)
        if a.save_best and e2["acc"]>best:
            best=e2["acc"]; best_ep=ep; ck=os.path.join(a.out,"ckpt"); model.save_pretrained(ck); proc.save_pretrained(ck); print(f"  saved best {e2['acc']:.3f}", flush=True)
    json.dump({"pre":pre,"best_acc":best,"best_ep":best_ep,"history":hist}, open(os.path.join(a.out,"result.json"),"w"), indent=2)
    print("DONE pre", pre["acc"], "best", best, "ep", best_ep)

if __name__=="__main__": main()
