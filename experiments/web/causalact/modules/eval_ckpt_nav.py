from causalact_runtime import location
import os,sys,json,random,re
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.intervention_schema import load_jsonl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_a_nav_official import build_prompt_text, forum_split, cands_of
CK=sys.argv[1]
from transformers import AutoProcessor, AutoModelForVision2Seq
proc=AutoProcessor.from_pretrained(CK)
m=AutoModelForVision2Seq.from_pretrained(CK,torch_dtype=torch.float16).to('cuda'); m.eval()
S=load_jsonl(location('CAUSALACT_RUN_DIR', 'web/data/nav_official_set.jsonl'))
tr,va,vf=forum_split(S,seed=42)
corr=tot=0
with torch.no_grad():
  for s in va:
    img=Image.open(s.screenshot_path).convert('RGB')
    msgs=[{'role':'user','content':[{'type':'image'},{'type':'text','text':build_prompt_text(s.intent,cands_of(s))}]}]
    pr=proc.apply_chat_template(msgs,add_generation_prompt=True)
    inp=proc(text=pr,images=[img],return_tensors='pt').to('cuda')
    g=m.generate(**inp,max_new_tokens=6,do_sample=False)
    t=proc.batch_decode(g[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
    mm=re.search(r'\d+',t); pred=mm.group(0) if mm else ''
    tot+=1; corr+=(pred==str(s.original_action.target_id))
print(json.dumps({'ckpt':CK,'val_acc':corr/tot,'corr':corr,'tot':tot,'n_val_forums':len(vf)}))
