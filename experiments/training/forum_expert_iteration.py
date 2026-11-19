"""Two-round V -> A -> I study on independently split, verified forum quads.

The Interventioner selects an executable program from a validated edit pool.
This constrained component experiment is not an end-to-end benchmark result.
All three modules include the entire frozen SmolVLM backbone in their size
audit; separate low-rank parameters and prediction heads are trained per role.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import traceback

import torch
from torch import nn
import torch.nn.functional as F


VARIANTS = {
    "base": {"categories": []},
    "action_only": {"categories": []},
    "static_cf": {"categories": ["target", "unrelated", "transform"]},
    "action_refiner": {"categories": ["target", "unrelated", "transform"]},
    "action_interventioner": {"categories": ["target", "unrelated", "transform"], "I": True},
    "action_verifier": {"categories": ["target", "unrelated", "transform"], "V": True},
    "without_targeted": {"categories": ["unrelated", "transform"], "V": True, "I": True},
    "without_unrelated": {"categories": ["target", "transform"], "V": True, "I": True},
    "without_equivariance": {"categories": ["target", "unrelated"], "V": True, "I": True},
    "full": {"categories": ["target", "unrelated", "transform"], "V": True, "I": True},
}
CATEGORIES = ["target", "unrelated", "transform"]


def atomic(path, data):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n"); tmp.replace(path)


class RoleLoRA(nn.Module):
    def __init__(self, linear, rank=8):
        super().__init__(); self.base = linear; self.active = "A"; self.scale = 2.0
        self.down = nn.ParameterDict(); self.up = nn.ParameterDict()
        for role in "AVI":
            self.down[role] = nn.Parameter(torch.empty(rank, linear.in_features))
            self.up[role] = nn.Parameter(torch.zeros(linear.out_features, rank))
            nn.init.kaiming_uniform_(self.down[role], a=math.sqrt(5))
        for p in linear.parameters(): p.requires_grad = False

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.down[self.active]), self.up[self.active]) * self.scale


def role_parameters(module, role):
    return [p for n, p in module.named_parameters()
            if n.endswith("." + role) or n.startswith("heads." + role + ".")]


def iou(a, b):
    inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union > 0 else 0.0


def norm_box(frame, size):
    return [v/size[i % 2] for i,v in enumerate(frame["bbox"])]


def causal_report(rows, predictions):
    counts = {k: [0, 0] for k in ["TargetedAcc", "UnrelatedAcc", "TargetPairAcc", "PairCausalAcc", "EvidenceRecall", "UnsafeAction"]}
    buckets = {k: {m:[0,0] for m in ["PairCausalAcc","EvidenceRecall","UnsafeAction"]}
               for k in ["<0.1%","0.1-1%","1-5%",">5%"]}
    for row in rows:
        p = predictions[row["id"]]; v = row["variants"]
        pair = p["original"]["action"] == v["original"]["action"] and p["target"]["action"] == v["target"]["action"]
        pair = pair and p["original"]["action"] != p["target"]["action"]
        counts["TargetPairAcc"][0] += int(pair); counts["TargetPairAcc"][1] += 1
        pair = pair and p["unrelated"]["action"] == v["unrelated"]["action"]
        counts["PairCausalAcc"][0] += int(pair); counts["PairCausalAcc"][1] += 1
        for metric, category in [("TargetedAcc","target"),("UnrelatedAcc","unrelated")]:
            counts[metric][0] += p[category]["action"] == v[category]["action"]; counts[metric][1] += 1
        box = norm_box(v["original"], row["size"]); area=(box[2]-box[0])*(box[3]-box[1])
        bucket = "<0.1%" if area < .001 else "0.1-1%" if area < .01 else "1-5%" if area < .05 else ">5%"
        evidence = p["original"].get("bbox") is not None and iou(p["original"]["bbox"], box) >= .5
        unsafe = p["target"]["action"] == 0 and v["target"]["action"] == 1
        for metric, ok in [("EvidenceRecall",evidence),("UnsafeAction",unsafe)]:
            counts[metric][0] += int(ok); counts[metric][1] += 1
        for metric, ok in [("PairCausalAcc",pair),("EvidenceRecall",evidence),("UnsafeAction",unsafe)]:
            buckets[bucket][metric][0] += int(ok); buckets[bucket][metric][1] += 1
    return {"metrics": {k:{"numerator":v[0],"denominator":v[1],"value":v[0]/v[1] if v[1] else None} for k,v in counts.items()},
            "area_buckets": {b:{k:{"numerator":v[0],"denominator":v[1],"value":v[0]/v[1] if v[1] else None} for k,v in ms.items()} for b,ms in buckets.items()}}


class Modules(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.backbone = AutoModelForVision2Seq.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
        for p in self.backbone.parameters(): p.requires_grad = False
        patched=[]
        for name, module in list(self.backbone.named_modules()):
            if "text_model" in name and name.endswith(("q_proj", "v_proj")) and isinstance(module, nn.Linear):
                parent, attr=name.rsplit(".",1); setattr(self.backbone.get_submodule(parent), attr, RoleLoRA(module)); patched.append(name)
        if not patched: raise RuntimeError("No text attention projections matched; do not silently train heads only")
        hidden=self.backbone.config.text_config.hidden_size
        self.heads=nn.ModuleDict({"A":nn.Linear(hidden,6),"V":nn.Linear(hidden,2),"I":nn.Linear(hidden,3)})
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.backbone.enable_input_require_grads()
        total_backbone=sum(p.numel() for n,p in self.backbone.named_parameters() if ".down." not in n and ".up." not in n)
        self.audit={role:{"full_module_parameters":total_backbone+sum(p.numel() for p in role_parameters(self,role)),
                          "trainable_parameters":sum(p.numel() for p in role_parameters(self,role))} for role in "AVI"}
        assert all(v["full_module_parameters"] < 1_000_000_000 for v in self.audit.values())
        self.active="A"

    @lru_cache(maxsize=128)
    def inputs(self, image, prompt):
        from PIL import Image
        text=self.processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}],add_generation_prompt=True)
        return self.processor(text=text, images=[Image.open(image).convert("RGB")],return_tensors="pt")

    def forward(self, role, row, category="original", candidate=None):
        for module in self.backbone.modules():
            if isinstance(module, RoleLoRA): module.active=role
        frame=row["variants"][category]
        prompt="Goal: "+row["goal"]+"\n"
        if role=="A": prompt+="Decide whether to click the forum subscription button or stop because the goal is already satisfied. Locate the button that provides evidence."
        elif role=="V": prompt+="Candidate action: "+("click the subscription button" if candidate==0 else "stop")+". Is this action correct for the goal and visible state?"
        else: prompt+="Select a backend-validated edit program to expose the current action refiner's errors: subscription toggle, unrelated background hue, or control translation."
        inp={k:v.to(next(self.parameters()).device) for k,v in self.inputs(frame["image"],prompt).items()}
        with torch.autocast("cuda",dtype=torch.float16):
            out=self.backbone.model(**inp,use_cache=False,return_dict=True)
            hidden=out.last_hidden_state[:,-1,:]
            return self.heads[role](hidden).float()

    def snapshot(self, role):
        ids={id(p) for p in role_parameters(self,role)}
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if id(p) in ids}

    def restore(self, state):
        with torch.no_grad():
            for n,p in self.named_parameters():
                if n in state: p.copy_(state[n].to(p.device))

    def zero_shot_action(self, row, category):
        """Use the pretrained language head, never a random new classifier."""
        for module in self.backbone.modules():
            if isinstance(module, RoleLoRA): module.active="A"
        frame=row["variants"][category]
        prompt="Goal: "+row["goal"]+"\nChoose click to click the forum subscription button, or stop if the goal is already satisfied. Answer only click or stop."
        device=next(self.parameters()).device
        inp={k:v.to(device) for k,v in self.inputs(frame["image"],prompt).items()}
        scores=[]
        for word in ["click","stop"]:
            ids=self.processor.tokenizer(word,add_special_tokens=False,return_tensors="pt").input_ids.to(device)
            full=dict(inp,input_ids=torch.cat([inp["input_ids"],ids],dim=1),
                      attention_mask=torch.cat([inp["attention_mask"],torch.ones_like(ids)],dim=1))
            with torch.autocast("cuda",dtype=torch.float16):
                hidden=self.backbone.model(**full,use_cache=False,return_dict=True).last_hidden_state
                logits=self.backbone.lm_head(hidden[:,-ids.shape[1]-1:-1,:]).float()
            scores.append(float(F.log_softmax(logits,dim=-1).gather(-1,ids.unsqueeze(-1)).mean()))
        return {"action":max(range(2),key=lambda j:scores[j]),"bbox":None,"scores":scores,
                "source":"pretrained language-head conditional likelihood; no trained classifier"}


def a_loss(net, row, categories, variant, rng):
    orig=net("A",row); gt=torch.tensor([0],device=orig.device)
    box=torch.tensor([norm_box(row["variants"]["original"],row["size"])],device=orig.device)
    loss=F.cross_entropy(orig[:,:2],gt)+F.mse_loss(orig[:,2:].sigmoid(),box)
    if not categories: return loss
    if variant.get("I"):
        with torch.no_grad():
            scores=net("I",row)[0]; permitted=[CATEGORIES.index(c) for c in categories]
            category=CATEGORIES[max(permitted,key=lambda j:float(scores[j]))]
    else: category=rng.choice(categories)
    pred=net("A",row,category);frame=row["variants"][category]
    box2=torch.tensor([norm_box(frame,row["size"])],device=pred.device)
    loss=loss+F.cross_entropy(pred[:,:2],torch.tensor([frame["action"]],device=pred.device))+F.mse_loss(pred[:,2:].sigmoid(),box2)
    if variant.get("relations",True) and category=="unrelated":
        loss=loss+F.kl_div(F.log_softmax(pred[:,:2],dim=-1),F.softmax(orig[:,:2].detach(),dim=-1),reduction="batchmean")
    if variant.get("relations",True) and category=="transform":
        shift=torch.tensor([[row["tau"]["dx"],row["tau"]["dy"],row["tau"]["dx"],row["tau"]["dy"]]],device=pred.device)
        loss=loss+F.mse_loss(pred[:,2:].sigmoid(),orig[:,2:].sigmoid().detach()+shift)
    return loss/2


def hard_edit_labels(net, rows, categories):
    labels={};net.eval()
    with torch.no_grad():
        for row in rows:
            losses=[]
            for c in categories:
                p=net("A",row,c); losses.append(float(F.cross_entropy(p[:,:2],torch.tensor([row["variants"][c]["action"]],device=p.device))))
            labels[row["id"]]=CATEGORIES.index(categories[max(range(len(losses)),key=lambda j:losses[j])])
    return labels


def train_role(net, role, train, val, variant, args, round_index, out):
    rng=random.Random(args.seed+round_index*17+ord(role));categories=variant["categories"]
    labels=hard_edit_labels(net,train+val,categories) if role=="I" else None
    if labels is not None:
        atomic(out/f"round{round_index+1}_I_training_labels.json",
               {"selection":"largest current A error among permitted verified edits",
                "training":{r["id"]:labels[r["id"]] for r in train},
                "validation":{r["id"]:labels[r["id"]] for r in val},
                "category_ids":CATEGORIES,"test_used":False})
    # Validation chooses checkpoints only. Neither held-out labels nor images enter here.
    optimizer=torch.optim.AdamW(role_parameters(net,role),lr=args.lr)
    scaler=torch.amp.GradScaler("cuda");best=float("inf");chosen=None;history=[]
    def loss(row, evaluate=False):
        if role=="A":return a_loss(net,row,categories,variant,rng)
        if role=="V":
            c=rng.choice(["original"]+categories);candidate=rng.randrange(2);p=net(role,row,c,candidate)
            return F.cross_entropy(p,torch.tensor([int(candidate==row["variants"][c]["action"])],device=p.device))
        p=net(role,row);return F.cross_entropy(p,torch.tensor([labels[row["id"]]],device=p.device))
    for epoch in range(args.epochs):
        net.train();order=list(train);rng.shuffle(order);optimizer.zero_grad(set_to_none=True);total=0.0
        for i,row in enumerate(order):
            l=loss(row)
            if not torch.isfinite(l):raise RuntimeError("Nonfinite training loss; no score")
            scaler.scale(l/args.accumulation).backward();total+=float(l.detach())
            if (i+1)%args.accumulation==0 or i+1==len(order):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(role_parameters(net,role),1.0)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        net.eval();saved=rng.getstate();rng.seed(args.seed+ord(role));values=[]
        with torch.no_grad():
            for row in val:values.append(float(loss(row,True)))
        rng.setstate(saved);vl=sum(values)/len(values);history.append({"epoch":epoch+1,"training_loss":total/len(order),"validation_loss":vl})
        if vl<best:best=vl;chosen=net.snapshot(role)
        atomic(out/"status.json",{"phase":"training","round":round_index+1,"role":role,"history":history})
        print(json.dumps({"round":round_index+1,"role":role,**history[-1]}),flush=True)
    net.restore(chosen);ck=out/f"round{round_index+1}_{role}.pt";torch.save(chosen,ck)
    atomic(out/f"round{round_index+1}_{role}.json",{"selection":"minimum validation loss","history":history,"checkpoint":ck.name,
           "train_cases":len(train),"validation_cases":len(val),"test_used":False,"sha256":hashlib.sha256(ck.read_bytes()).hexdigest()})


def evaluate(net, rows, variant, out, base=False):
    net.eval();predictions={};vok=vn=0;edit_records=[]
    with (out/"heldout_predictions.jsonl").open("x") as stream,torch.no_grad():
        for row in rows:
            preds={}
            for c in ["original"]+CATEGORIES:
                if base:
                    preds[c]=net.zero_shot_action(row,c)
                    continue
                p=net("A",row,c);scores=F.log_softmax(p[:,:2],dim=-1)[0]
                if variant.get("V"):
                    for candidate in range(2):
                        v=net("V",row,c,candidate);scores[candidate]+=F.log_softmax(v,dim=-1)[0,1]
                        vok+=int(int(v.argmax())==int(candidate==row["variants"][c]["action"]));vn+=1
                preds[c]={"action":int(scores.argmax()),"bbox":p[0,2:].sigmoid().cpu().tolist(),"scores":scores.cpu().tolist()}
            if variant.get("I"):
                scores=net("I",row)[0];allowed=[CATEGORIES.index(c) for c in variant["categories"]]
                chosen=max(allowed,key=lambda j:float(scores[j]));category=CATEGORIES[chosen]
                edit_records.append({"id":row["id"],"category":category,"executable_program":row["edit_pool"][chosen],
                                     "oracle_valid":all(row["variants"][category]["oracle"][k] for k in ["render","state_preserved","control_executable"]),
                                     "exposes_action_error":preds[category]["action"]!=row["variants"][category]["action"]})
            predictions[row["id"]]=preds;stream.write(json.dumps({"id":row["id"],"forum":row["forum"],"predictions":preds})+"\n");stream.flush()
    report=causal_report(rows,predictions)
    report.update(test_cases=len(rows),independent_forums=len({r["forum"] for r in rows}),
                  case_structure="two goals and two paraphrases per forum; cases are clustered, not IID tasks",
                  scope="controlled forum component test; not Web/Desktop/Driving end-to-end",
                  action_metric_scope="semantic click/stop correctness; pointer execution is not measured here",
                  safety_metric_scope="oracle disagreement on an already-satisfied subscription goal; not real-world safety",
                  verifier_agreement={"numerator":vok,"denominator":vn,"value":vok/vn if vn else None},
                  base_evidence_output="not provided by the conditional-likelihood baseline" if base else None,
                  interventioner={"setting":"learned selection from backend-validated programs; not unconstrained generation","records":edit_records})
    atomic(out/"report.json",report);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",required=True);ap.add_argument("--model",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--variant",choices=list(VARIANTS),required=True);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--rounds",type=int,default=2)
    ap.add_argument("--epochs",type=int,default=3);ap.add_argument("--lr",type=float,default=1e-5);ap.add_argument("--accumulation",type=int,default=4)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():raise RuntimeError("Result exists; do not rerun a held-out evaluation")
    random.seed(args.seed);torch.manual_seed(args.seed)
    rows=[json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
    for row in rows:
        for frame in row["variants"].values():
            image=Path(args.data).parent/frame["image"]
            if hashlib.sha256(image.read_bytes()).hexdigest()!=frame["sha256"]:
                raise RuntimeError("Captured image changed; do not score different pixels")
            frame["image"]=str(image)
    splits={s:[r for r in rows if r["split"]==s] for s in ["train","val","test"]}
    assert all(splits.values())
    groups={s:set(r["forum"] for r in v) for s,v in splits.items()};assert not any(groups[a]&groups[b] for a,b in [("train","val"),("train","test"),("val","test")])
    variant=dict(VARIANTS[args.variant]);variant["relations"]=args.variant not in ["action_only","static_cf"]
    net=Modules(args.model).to("cuda")
    atomic(out/"protocol.json",{**vars(args),"variant_settings":variant,"parameter_audit":net.audit,"train_val_test_forums":{k:sorted(v) for k,v in groups.items()},
           "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
           "data_sha256":hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),"candidate_pool_constraint":True,"full_benchmark":False,
           "selection":"minimum validation loss per role; fixed two rounds; held-out evaluated once"})
    try:
        for round_index in range(0 if args.variant=="base" else args.rounds):
            if variant.get("V"):train_role(net,"V",splits["train"],splits["val"],variant,args,round_index,out)
            train_role(net,"A",splits["train"],splits["val"],variant,args,round_index,out)
            if variant.get("I"):train_role(net,"I",splits["train"],splits["val"],variant,args,round_index,out)
        atomic(out/"frozen_config.json",{"selection_complete":True,"test_not_used_for_selection":True,"rounds":args.rounds,"checkpoints":[p.name for p in out.glob("round*.pt")]})
        report=evaluate(net,splits["test"],variant,out,base=args.variant=="base")
        atomic(out/"status.json",{"phase":"finished","test_cases":report["test_cases"]})
    except Exception:
        atomic(out/"status.json",{"phase":"invalid; needs repair","error":traceback.format_exc()});raise


if __name__=="__main__":main()
