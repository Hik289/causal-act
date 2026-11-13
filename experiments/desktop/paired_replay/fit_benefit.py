"""Train a small intervention-benefit head and choose its epoch on validation."""
from pathlib import Path
import json
import math
import os
import random
import sys

from PIL import Image
import torch
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor

root, work = map(Path,sys.argv[1:])
work.mkdir(parents=True, exist_ok=True)
source=Path(__file__).with_name("calibrate.py")
prefix=source.read_text().split("\ncandidates =",1)[0]
audit = {"__name__":"archived_audit","__file__":str(source)}
exec(compile(prefix,str(source),"exec"),audit)
osw=Path(os.environ["CAUSALACT_OSWORLD_DIR"])
base,refiner=audit["base"],audit["refiner"]

def task_instruction(domain,task):
    p=osw/"evaluation_examples/examples"/domain/f"{task}.json"
    try:return json.loads(p.read_text())["instruction"]
    except (OSError,KeyError):return None

def trajectory(folder):
    try:
        return {int(r["step_num"]):r for r in map(json.loads,(folder/"traj.jsonl").read_text().splitlines())}
    except (OSError,KeyError,ValueError):
        return {}

def branch_examples(rows):
    examples=[]
    for row in rows:
        if row["trigger"] is None:
            continue
        domain,task=row["domain"],row["task"]
        step,_=row["trigger"]
        before=refiner/domain/task
        prev=trajectory(before).get(step-1,{})
        proposed=trajectory(base/domain/task).get(step,{})
        instruction=task_instruction(domain,task)
        filename=prev.get("screenshot_file")
        action=proposed.get("response") or proposed.get("action")
        if not (filename and action and instruction):
            continue
        frame=before/Path(filename).name
        if not frame.is_file():
            continue
        examples.append({**row,"frame":str(frame),"action":str(action),"instruction":instruction,
                         "benefit":row["base"]==0 and row["refiner"]==1,
                         "harm":row["base"]==1 and row["refiner"]==0})
    return examples

# Historical application domains are used for training. Validation and held-out
# tasks come from the pre-existing disjoint OSWorld splits for the other four domains.
train_keys={(app,p.name) for app in audit["domains"] if app not in audit["validation"]
            for p in (base/app).iterdir() if p.is_dir()}
val_keys=audit["task_keys"](audit["validation"])
test_keys=audit["task_keys"](audit["holdout"])
assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
train=branch_examples(audit["evaluate_rows"](train_keys))
val_rows=audit["evaluate_rows"](val_keys)
val=branch_examples(val_rows)
print("Usable branch examples:",
      {"train":len(train),"train_rescues":sum(e["benefit"] for e in train),
       "validation":len(val),"validation_rescues":sum(e["benefit"] for e in val)},flush=True)
if not train or not val or not any(e["benefit"] for e in train):
    raise SystemExit("Archived branch frames or outcomes are missing")

random.seed(42)
torch.manual_seed(42)
device="cuda"
model_id="HuggingFaceTB/SmolVLM-500M-Instruct"
proc=AutoProcessor.from_pretrained(model_id,local_files_only=True)
model=AutoModelForVision2Seq.from_pretrained(model_id,torch_dtype=torch.float16,
                                            attn_implementation="eager",local_files_only=True).to(device).eval()
model.requires_grad_(False)

def make_prompt(instruction,action):
    text=("Assess whether replacing the proposed action is likely to turn a GUI task failure into a success. "
          f"Task: {instruction[:300]}\nProposed base action: {action[:320]}\n"
          "Would asking the agent to try a different approach help?")
    return proc.apply_chat_template([{"role":"user","content":[
        {"type":"image"},{"type":"text","text":text}]}],add_generation_prompt=True)

@torch.inference_mode()
def features(rows,label):
    vectors=[]
    for i,e in enumerate(rows,1):
        with Image.open(e["frame"]) as source_image:
            image=source_image.convert("RGB")
        image.thumbnail((512,512))
        inputs=proc(text=make_prompt(e["instruction"],e["action"]),images=[image],return_tensors="pt").to(device)
        output=model(**inputs,output_hidden_states=True,use_cache=False,return_dict=True)
        vectors.append(output.hidden_states[-1][0,-1].float().cpu())
        if i%20==0 or i==len(rows):print(f"{label} features {i}/{len(rows)}",flush=True)
    return torch.stack(vectors)

train_f,val_f=features(train,"train"),features(val,"validation")
center=train_f.mean(0)
scale=train_f.std(0,unbiased=False).clamp(min=1e-3)
def normalize(x):
    return ((x-center)/scale).clamp(-6,6)

x, val_x=normalize(train_f),normalize(val_f)
y=torch.tensor([e["benefit"] for e in train],dtype=torch.float32)
head=nn.Linear(x.shape[1],1)
nn.init.zeros_(head.weight)
nn.init.zeros_(head.bias)
optimizer=torch.optim.AdamW(head.parameters(),lr=0.002,weight_decay=0.04)
positive=float(y.sum())
loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(8.0,(len(y)-positive)/positive)))
thresholds=(.35,.50,.65)
epochs=6
history=[]
best=None
best_key=None

def measure(rows,accept):
    selected=[r for r,keep in zip(rows,accept) if keep]
    rescued=sum(r["benefit"] for r in selected)
    harmed=sum(r["harm"] for r in selected)
    return {"triggered":len(selected),"rescued":rescued,"harmed":harmed,
            "net":rescued-harmed,"utility":rescued-2*harmed}

for epoch in range(1,epochs+1):
    head.train()
    for index in torch.randperm(len(y)).split(8):
        optimizer.zero_grad()
        logits=head(x[index]).squeeze(-1)
        loss=loss_fn(logits,y[index])
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        probs=head(val_x).squeeze(-1).sigmoid()
    results=[]
    for threshold in thresholds:
        m={"epoch":epoch,"threshold":threshold,**measure(val,(probs>=threshold).tolist())}
        results.append(m)
        # Rank only on validation. Break ties by fewer harms, then fewer interventions.
        key=(m["utility"],-m["harmed"],m["rescued"],-m["triggered"],-epoch,threshold)
        if best_key is None or key>best_key:
            best,best_key=m,key
            torch.save({"head":head.state_dict(),"center":center,"scale":scale},work/"selected_head.pt")
    history.extend(results)
    (work/"validation_history.json").write_text(json.dumps(history,indent=2)+"\n")
    print(f"Epoch {epoch} validation: {json.dumps(results)}",flush=True)
selection={"model":model_id,"seed":42,"epochs_tried":epochs,"selected":best,
           "validation_tasks":len(val_rows),"validation_branch_examples":len(val),
           "train_branch_examples":len(train),
           "validation_original_gate":measure(val,[True]*len(val)),
           "validation_fixed_097":measure(val,[r["trigger"][1]>=.97 for r in val])}
(work/"frozen_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
print("Frozen selection:",json.dumps(selection),flush=True)

# Evaluate the held-out outcomes once, after both the epoch and threshold are frozen.
test_rows=audit["evaluate_rows"](test_keys)
test=branch_examples(test_rows)
if not test:
    raise SystemExit("No usable holdout branch examples")
checkpoint=torch.load(work/"selected_head.pt",map_location="cpu",weights_only=True)
head.load_state_dict(checkpoint["head"])
head.eval()
with torch.no_grad():
    test_prob=head(normalize(features(test,"holdout"))).squeeze(-1).sigmoid()
result={"selection":selection,"holdout_tasks":len(test_rows),"holdout_base_success":sum(r["base"] for r in test_rows),
        "holdout_branch_examples":len(test),
        "holdout_original_gate":measure(test,[True]*len(test)),
        "holdout_fixed_097":measure(test,[r["trigger"][1]>=.97 for r in test]),
        "holdout_selected":measure(test,(test_prob>=best["threshold"]).tolist())}
temp=work/"report.tmp"
temp.write_text(json.dumps(result,indent=2)+"\n")
temp.replace(work/"report.json")
print("Final held-out reconstruction:",json.dumps(result),flush=True)
