# 评估已训练模型的 POST consistency (pred_orig==pred_unrel) + acc, 比 acc 更敏感
import os,sys,json,random,argparse
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_a_lspec import build_prompt, load_elems_desc, MODEL_ID
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt',required=False,default=None); ap.add_argument('--data',required=True); ap.add_argument('--n',type=int,default=24)
a=ap.parse_args()
random.seed(42); torch.manual_seed(42)
from transformers import AutoProcessor, AutoModelForVision2Seq
proc=AutoProcessor.from_pretrained(MODEL_ID)
src = a.ckpt if a.ckpt else MODEL_ID
m=AutoModelForVision2Seq.from_pretrained(src, torch_dtype=torch.float16).to('cuda'); m.eval()
S=load_jsonl(a.data); random.shuffle(S)
# use same val split as training (first n_val after shuffle seed42) — replicate: n_val=max(10,len//5)
n_val=max(10,len(S)//5); val=S[:n_val]
def predict(s,imgpath):
    desc,gt,ids=load_elems_desc(s); img=Image.open(imgpath).convert('RGB')
    pr=build_prompt(proc,s.intent,desc,img); inp=proc(text=pr,images=[img],return_tensors='pt').to('cuda')
    g=m.generate(**inp,max_new_tokens=8,do_sample=False)
    t=proc.batch_decode(g[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
    return (t.split()[0] if t else ''),gt
consist=orig_r=unrel_r=n=0
with torch.no_grad():
    for s in val:
        po,gt=predict(s,s.screenshot_path); pu,_=predict(s,s.unrelated_screenshot_path); n+=1
        if po==pu: consist+=1
        if gt in po: orig_r+=1
        if gt in pu: unrel_r+=1
print(json.dumps({'ckpt':src,'n':n,'consistency':consist/n,'orig_acc':orig_r/n,'unrel_acc':unrel_r/n},indent=2))
