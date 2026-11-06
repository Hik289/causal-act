from causalact_runtime import location
# 测 gpt-5.6-sol base 在 nav grounding val 集上的 accuracy, 对比 A(0.30)
import os,sys,json,random,base64,io
sys.path.insert(0,'.')
from PIL import Image
from data.intervention_schema import load_jsonl
REDDIT='http://127.0.0.1:9999'
from openai import OpenAI
c=OpenAI(max_retries=4)
S=load_jsonl(location('CAUSALACT_RUN_DIR', 'web/data/nav_set.jsonl'))
# same forum_split seed42
import random as R
forums=sorted(set(s.oracle_metadata['forum'] for s in S)); R.Random(42).shuffle(forums)
nval=max(4,int(len(forums)*0.25)); valf=set(forums[:nval])
val=[s for s in S if s.oracle_metadata['forum'] in valf]
def prompt(s):
    cands=s.oracle_metadata['candidates']
    lines='\n'.join(f"{x['id']}: comments link at box {x['bbox']}" for x in cands)
    return (f"You are a web navigation agent. The screenshot is a forum post list. Each candidate is a 'comments' link for one post.\n"
            f"Goal: {s.intent}\nCandidates:\n{lines}\n"
            f"Look at the screenshot to read each post title near its comments link and pick the matching post's comments link. Answer ONLY the element id.")
corr=0;tot=0
for s in val:
    img=Image.open(s.screenshot_path).convert('RGB'); buf=io.BytesIO(); img.save(buf,format='PNG'); b64=base64.b64encode(buf.getvalue()).decode()
    try:
        r=c.chat.completions.create(model=os.environ.get("CAUSALACT_BASE_MODEL", "gpt-5.6-sol"), messages=[{'role':'user','content':[{'type':'text','text':prompt(s)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+b64}}]}])
        t=(r.choices[0].message.content or '').strip()
    except Exception as e:
        t=''
    gt=s.original_action.target_id; tot+=1
    if gt in t: corr+=1
print(json.dumps({'base_nav_grounding_acc':corr/tot if tot else None,'corr':corr,'tot':tot}))
