"""I-1b (H0.method_2.2): 构造 visual-grounding 干预集 — 候选只给坐标/bbox, 不给文字。

与 I-1 (build_unrelated_set.py) 的关键区别:
  - 在 original / unrelated 截图上**绘制 SoM 编号框** (候选元素位置画 [N] 彩框)
  - 候选只给 {id: elem_N, bbox:[...]} (NO text label) → A 必须**看图认框** 才知哪个对应 intent
  - GT target = intent 相关的内容元素 (同 I-1 选法)
  → 任务从"文字匹配"变成"视觉 grounding", 无关视觉扰动会扰动图像特征。

unrelated 扰动: 同 I-1 (背景/装饰图/侧栏变色), 但**SoM 框保持可见** (框是 overlay, 不随扰动变)。
oracle 核验: render-validity + unrelated-preservation (GT 元素仍在)。
"""
from causalact_runtime import location
import os, sys, json, re
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle.forum_oracles import (
    JS_RENDER_FACTS, js_target_exists,
    oracle_render_validity, oracle_unrelated_preservation,
)
from data.intervention_schema import InterventionSample, EditSpec, ActionRec, dump_jsonl

REDDIT = os.environ.get("REDDIT", "http://127.0.0.1:9999")
AUTH = location('CAUSALACT_VWA_AUTH', '')
VWA_CONFIG = location('CAUSALACT_VWA_DIR', 'config_files/vwa/test_reddit')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/visual_grounding')
OUT_JSONL = location('CAUSALACT_RUN_DIR', 'web/data/visual_grounding_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)
VIEWPORT = {"width": 1280, "height": 720}

JS_UNRELATED_EDIT = """
(() => {
  document.body.style.backgroundColor = '#f2c744';
  let imgs = 0;
  document.querySelectorAll('img').forEach(im => { im.style.filter = 'hue-rotate(160deg) saturate(1.4)'; imgs++; });
  document.querySelectorAll('.sidebar, footer, .footer, aside').forEach(e => { e.style.backgroundColor = '#cde'; });
  return { imgs_changed: imgs };
})()
"""

JS_GET_INTERACTABLES = """
(() => {
  const sel = 'a[href], button, input:not([type=hidden]), [role=button], [role=link]';
  const els = Array.from(document.querySelectorAll(sel));
  const out = []; let i = 1;
  for (const e of els) {
    const r = e.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.top < 0 || r.top > 720) continue;
    if (r.left < 0 || r.left > 1280) continue;
    const text = (e.textContent || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 40);
    const href = e.getAttribute('href') || '';
    out.push({tag:e.tagName, href, text,
      center:[Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)],
      bbox:[Math.round(r.x), Math.round(r.y), Math.round(r.x+r.width), Math.round(r.y+r.height)]});
    i++; if (i > 60) break;
  }
  return out;
})()
"""

# 在页面上绘制 SoM 编号框 (overlay, 不影响底层 DOM 交互元素)
def js_draw_som(boxes):
    # boxes: [{id_num, bbox:[x0,y0,x1,y1]}]
    data = json.dumps(boxes)
    return f"""
(() => {{
  // 清除旧 overlay
  document.querySelectorAll('.civ-som-box').forEach(e=>e.remove());
  const boxes = {data};
  const colors = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6','#bcf60c'];
  boxes.forEach((b,idx) => {{
    const [x0,y0,x1,y1] = b.bbox;
    const d = document.createElement('div');
    d.className = 'civ-som-box';
    d.style.cssText = `position:absolute; left:${{x0}}px; top:${{y0 + window.scrollY}}px; width:${{x1-x0}}px; height:${{y1-y0}}px; border:2px solid ${{colors[idx%8]}}; z-index:99999; pointer-events:none; box-sizing:border-box;`;
    const lab = document.createElement('span');
    lab.textContent = b.id_num;
    lab.style.cssText = `position:absolute; left:0; top:-16px; background:${{colors[idx%8]}}; color:#fff; font-size:12px; font-weight:bold; padding:0 3px; line-height:14px;`;
    d.appendChild(lab);
    document.body.appendChild(d);
  }});
  return boxes.length;
}})()
"""

SITE_CHROME = {"postmill","forums","wiki","submit","log in","login","sign up","search query","",
               "messages","notifications","hot","new","top","controversial","active",
               "subscribe","unsubscribe","hide this forum"}

def _is_content(e):
    t = (e["text"] or "").strip().lower()
    if t in SITE_CHROME or len(t) < 4: return False
    href = e.get("href","")
    if "comment" in t: return True
    if re.search(r"/f/[^/]+/\d+/", href): return True
    if e["tag"] == "A" and len(t) >= 10: return True
    return False

def pick_gt(intent, elems):
    content = [e for e in elems if _is_content(e)]
    if not content: return None, "no_content"
    itl = intent.lower()
    kws = [w for w in re.findall(r"[a-z]{4,}", itl)
           if w not in ("this","that","with","find","navigate","please","trying","section","which",
                        "there","their","about","post","comment","research","doing","some","want","would","like","tell","many")]
    best=None; bs=0
    for e in content:
        sc = sum(1 for k in kws if k in e["text"].lower())
        if sc>bs: bs=sc; best=e
    if best and bs>0: return best, "intent_match"
    for e in content:
        if "comment" in e["text"].lower(): return e,"comment_link"
    return content[0],"first_content"


def load_tasks(max_n=120):
    tasks=[]
    files=sorted(os.listdir(VWA_CONFIG), key=lambda x:int(x.split(".")[0]) if x.split(".")[0].isdigit() else 1e9)
    for fn in files:
        if not fn.endswith(".json"): continue
        d=json.load(open(os.path.join(VWA_CONFIG,fn)))
        su=d.get("start_url","")
        if "9999" in su and "8888" not in su and "7770" not in su and " |AND| " not in su:
            tasks.append({"task_id":str(d.get("task_id",fn.split(".")[0])),"intent":d.get("intent",""),
                          "start_url":su.split(" |AND| ")[0] if su else f"{REDDIT}/"})
        if len(tasks)>=max_n: break
    return tasks


def build(max_samples=60, n_cand=8):
    tasks=load_tasks(120)
    print(f"loaded {len(tasks)} tasks")
    samples=[]; stats={"attempted":0,"render_fail":0,"no_content":0,"unrel_fail":0,"ok":0}
    with sync_playwright() as pw:
        b=pw.chromium.launch(headless=True)
        ctx=b.new_context(storage_state=AUTH, viewport=VIEWPORT)
        page=ctx.new_page()
        for t in tasks:
            if len(samples)>=max_samples: break
            stats["attempted"]+=1; tid=t["task_id"]
            try:
                page.goto(t["start_url"], wait_until="domcontentloaded", timeout=15000); page.wait_for_timeout(500)
            except Exception: continue
            if not oracle_render_validity(page.evaluate(JS_RENDER_FACTS)).valid:
                stats["render_fail"]+=1; continue
            elems=page.evaluate(JS_GET_INTERACTABLES)
            gt,how=pick_gt(t["intent"], elems)
            if gt is None: stats["no_content"]+=1; continue
            # 候选 = GT + 干扰 (content + 少量其他), 取 n_cand 个, 分配编号
            content=[e for e in elems if _is_content(e)]
            others=[e for e in elems if e not in content]
            pool=[gt]+[e for e in content if e is not gt][:n_cand-1]
            if len(pool)<n_cand:
                pool += others[:n_cand-len(pool)]
            pool=pool[:n_cand]
            # 编号 (打乱, 记录 GT 的编号)
            import random as _r; _r.seed(int(tid) if tid.isdigit() else 0)
            _r.shuffle(pool)
            boxes=[{"id_num":i+1, "bbox":e["bbox"], "center":e["center"]} for i,e in enumerate(pool)]
            gt_num=next(i+1 for i,e in enumerate(pool) if e is gt)

            tgt_text=gt["text"][:15] if gt["text"] else None
            before_exists=page.evaluate(js_target_exists(tgt_text)) if tgt_text else {"found":True,"disabled":False}

            # 画 SoM 框 → 截 original
            page.evaluate(js_draw_som(boxes)); page.wait_for_timeout(150)
            orig_png=f"{IMG_DIR}/{tid}_orig.png"; page.screenshot(path=orig_png)

            # unrelated 扰动 (SoM 框 overlay 保持) → 重画框 (扰动可能触发 reflow) → 截 unrel
            page.evaluate(JS_UNRELATED_EDIT); page.wait_for_timeout(150)
            page.evaluate(js_draw_som(boxes)); page.wait_for_timeout(150)
            unrel_png=f"{IMG_DIR}/{tid}_unrel.png"; page.screenshot(path=unrel_png)

            after_exists=page.evaluate(js_target_exists(tgt_text)) if tgt_text else {"found":True,"disabled":False}
            if tgt_text and not oracle_unrelated_preservation(before_exists, after_exists).valid:
                stats["unrel_fail"]+=1; continue
            if not oracle_render_validity(page.evaluate(JS_RENDER_FACTS)).valid:
                stats["render_fail"]+=1; continue

            # GT action: click 目标框 (target_id = elem_<gt_num>), 候选只存 id+bbox (无 text)
            act=ActionRec(action_type="click", target_id=f"elem_{gt_num}",
                          coordinates=gt["center"], evidence_bbox=gt["bbox"], confidence=1.0)
            s=InterventionSample(
                task_id=tid, site="reddit", intent=t["intent"],
                screenshot_path=orig_png, image_size=[VIEWPORT["width"],VIEWPORT["height"]],
                base_action=act, original_action=act,
                unrelated_edit=EditSpec(edit_type="recolor_bg_img", category="unrelated",
                                        attribute="backgroundColor+img.filter", value="#f2c744/hue160",
                                        backend_mutated=False),
                unrelated_action=act, unrelated_screenshot_path=unrel_png,
                oracle_metadata={"gt_pick":how, "gt_num":gt_num, "n_cand":len(boxes),
                                 "render_valid":True, "unrelated_preserved":True,
                                 # 候选只给 id + bbox, NO text (visual grounding)
                                 "candidates":[{"id":f"elem_{bx['id_num']}","bbox":bx["bbox"],"center":bx["center"]} for bx in boxes]},
                data_split="train")
            samples.append(s); stats["ok"]+=1
        b.close()
    dump_jsonl(samples, OUT_JSONL)
    print(json.dumps(stats, indent=2)); print(f"saved {len(samples)} -> {OUT_JSONL}")
    return samples, stats


if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 60
    build(max_samples=n)
