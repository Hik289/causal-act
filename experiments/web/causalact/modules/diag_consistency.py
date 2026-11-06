from causalact_runtime import location
# 诊断: base(未训练) SmolVLM 在 orig vs unrel 图上预测一致性 + 正确率
# 若 base 已高度一致 → unrelated 扰动太弱 → L_spec 无用武之地 (这是关键诊断)
import os,sys,json,random
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_a_lspec import build_prompt, load_elems_desc, MODEL_ID
random.seed(42); torch.manual_seed(42)
from transformers import AutoProcessor, AutoModelForVision2Seq
proc=AutoProcessor.from_pretrained(MODEL_ID)
m=AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to('cuda'); m.eval()
S=load_jsonl(location('CAUSALACT_RUN_DIR', 'web/data/unrelated_set.jsonl'))
random.shuffle(S); S=S[:20]
def predict(s, imgpath):
    desc,gt,ids=load_elems_desc(s)
    img=Image.open(imgpath).convert('RGB')
    pr=build_prompt(proc,s.intent,desc,img)
    inp=proc(text=pr,images=[img],return_tensors='pt').to('cuda')
    g=m.generate(**inp,max_new_tokens=8,do_sample=False)
    t=proc.batch_decode(g[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
    return (t.split()[0] if t else ''), gt
consist=0; both_right=0; orig_right=0; unrel_right=0; n=0
with torch.no_grad():
    for s in S:
        po,gt=predict(s,s.screenshot_path)
        pu,_=predict(s,s.unrelated_screenshot_path)
        n+=1
        if po==pu: consist+=1
        if gt in po: orig_right+=1
        if gt in pu: unrel_right+=1
        if gt in po and gt in pu: both_right+=1
print(json.dumps({'n':n,'base_consistency(pred_orig==pred_unrel)':consist/n,
  'base_orig_acc':orig_right/n,'base_unrel_acc':unrel_right/n},indent=2))
