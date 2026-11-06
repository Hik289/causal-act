"""E2: 构造导航 grounding 反事实集 (navigate/find 类, forum task 大头 65%)。

每个 forum 列表页 → 多个帖子, 每帖有 (title, comments链接, SoM id)。
对每帖生成一条样本:
  intent = "Navigate to the comment section of the post titled '<title>'."
  GT action = click(该帖 Comments 链接的 SoM id)
同一页的其它帖子 = 天然的反事实对照(同截图, 不同 intent → 不同 GT) → A 必须读视觉证据(标题)定位。

环境可核验: GT 来自 DOM (title → comments link 映射), 确定性, 无需后端。
截图带 SoM 框 (所有帖子 Comments 链接编号)。候选 = 该页所有帖子的 Comments 链接 id。
forum-disjoint 切分由下游训练脚本按 forum 做 (这里记录每样本的 forum)。
"""
from causalact_runtime import location
import os, sys, json, re
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle.forum_oracles import JS_RENDER_FACTS, oracle_render_validity
from data.intervention_schema import InterventionSample, EditSpec, ActionRec, dump_jsonl

REDDIT = os.environ.get("REDDIT", "http://127.0.0.1:9999")
AUTH = location('CAUSALACT_VWA_AUTH', '')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/nav')
OUT_JSONL = location('CAUSALACT_RUN_DIR', 'web/data/nav_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)
VIEWPORT = {"width": 1280, "height": 720}

# 提取帖子: 按 post id 聚合 title 链接 + comments 链接, 在 comments 链接上画 SoM 框。
JS_GET_POSTS_AND_DRAW = """
(() => {
  document.querySelectorAll('.civ-som-box').forEach(e=>e.remove());
  const links = Array.from(document.querySelectorAll('a[href]'));
  const byId = {};
  for (const a of links) {
    const h = a.getAttribute('href')||'';
    const m = h.match(/\\/f\\/[^/]+\\/(\\d+)/);
    if (!m) continue;
    const pid = m[1];
    const t = (a.textContent||'').trim();
    const r = a.getBoundingClientRect();
    if (!byId[pid]) byId[pid] = {title:null, comments:null};
    if (/comment/i.test(t)) {
      if (r.width>2 && r.height>2 && r.top>=0 && r.top<=720 && r.left>=0 && r.left<=1280)
        byId[pid].comments = {x:r.x, y:r.y, w:r.width, h:r.height};
    } else if (t.length>=6 && !byId[pid].title) {
      byId[pid].title = t;
    }
  }
  const colors=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6','#bcf60c','#008080','#9a6324'];
  const posts=[]; let idnum=1;
  for (const pid in byId) {
    const o = byId[pid];
    if (!o.title || !o.comments) continue;
    const r = o.comments;
    const box=document.createElement('div'); box.className='civ-som-box';
    box.setAttribute('style', 'position:absolute; left:'+r.x+'px; top:'+(r.y+window.scrollY)+'px; width:'+r.w+'px; height:'+r.h+'px; border:2px solid '+colors[(idnum-1)%10]+'; z-index:99999; pointer-events:none; box-sizing:border-box;');
    const lab=document.createElement('span'); lab.textContent=idnum;
    lab.setAttribute('style', 'position:absolute; left:0; top:-15px; background:'+colors[(idnum-1)%10]+'; color:#fff; font-size:12px; font-weight:bold; padding:0 3px;');
    box.appendChild(lab); document.body.appendChild(box);
    posts.push({id:'elem_'+idnum, title:o.title.slice(0,80),
      center:[Math.round(r.x+r.w/2), Math.round(r.y+r.h/2)],
      bbox:[Math.round(r.x),Math.round(r.y),Math.round(r.x+r.w),Math.round(r.y+r.h)]});
    idnum++;
    if (idnum>10) break;
  }
  return posts;
})()
"""


def get_forums(page, limit=25):
    page.goto(f"{REDDIT}/forums", wait_until="domcontentloaded"); page.wait_for_timeout(500)
    fs = page.evaluate("""() => Array.from(document.querySelectorAll('a[href^="/f/"]')).map(a=>a.getAttribute('href')).filter(h=>h.split('/').length===3)""")
    return sorted(set(fs))[:limit]


def build(per_forum_pages=2, max_per_page=6):
    samples = []
    stats = {"forums": 0, "pages": 0, "samples": 0, "render_fail": 0, "no_posts": 0}
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=AUTH, viewport=VIEWPORT)
        page = ctx.new_page()
        forums = get_forums(page, 25)
        print(f"forums: {len(forums)}")
        for furl in forums:
            fname = furl.split("/")[-1]
            stats["forums"] += 1
            # 多个列表页变体 (hot/new/top) 增加页面多样性
            for variant in ["", "/new", "/top"][:per_forum_pages]:
                url = f"{REDDIT}{furl}{variant}"
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000); page.wait_for_timeout(500)
                except Exception:
                    continue
                if not oracle_render_validity(page.evaluate(JS_RENDER_FACTS)).valid:
                    stats["render_fail"] += 1; continue
                posts = page.evaluate(JS_GET_POSTS_AND_DRAW)
                page.wait_for_timeout(120)
                if len(posts) < 2:
                    stats["no_posts"] += 1; continue
                stats["pages"] += 1
                # 截图一次 (带所有 SoM 框), 多条样本共享该图, GT=不同帖子
                img_path = f"{IMG_DIR}/{fname}{variant.replace('/','_')}.png"
                page.screenshot(path=img_path)
                cands = [{"id": p["id"], "bbox": p["bbox"], "center": p["center"]} for p in posts]
                for p in posts[:max_per_page]:
                    intent = f"Navigate to the comment section of the post titled \"{p['title']}\"."
                    act = ActionRec(action_type="click", target_id=p["id"],
                                    coordinates=p["center"], evidence_bbox=p["bbox"], confidence=1.0)
                    s = InterventionSample(
                        task_id=f"{fname}{variant.replace('/','_')}_{p['id']}",
                        site="reddit", intent=intent,
                        screenshot_path=img_path, image_size=[VIEWPORT["width"], VIEWPORT["height"]],
                        base_action=act, original_action=act,
                        oracle_metadata={"forum": fname, "task_type": "navigate",
                                         "gt_title": p["title"], "candidates": cands,
                                         "n_cand": len(cands)},
                        data_split="train")
                    samples.append(s); stats["samples"] += 1
        b.close()
    dump_jsonl(samples, OUT_JSONL)
    print(json.dumps(stats, indent=2)); print(f"saved {len(samples)} -> {OUT_JSONL}")
    return samples, stats


if __name__ == "__main__":
    build()
