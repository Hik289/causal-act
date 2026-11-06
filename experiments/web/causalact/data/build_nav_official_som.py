"""E2-fix: 用 VWA 官方 SoM pipeline 重造导航 grounding 数据 (与 base 输入逐像素对称)。

修复 已确认的 bug: 之前手绘框不渲染 + 候选无文本 → A 输入损坏。
现在: ScriptBrowserEnv image_som obs → 官方标注图(框画在像素上) + 官方 `[id][TAG][text]` 文本列表。
A 看到和 base 完全相同的输入。

导航样本构造:
- 对每个 forum 列表页, 从官方 SoM 文本列表解析出所有"comments"链接元素 (id + 它所属帖子标题)。
- 每条样本: intent="导航到标题X的帖子评论区", GT = 该帖 comments 链接的官方 SoM id。
- 候选 = 该页所有 comments 链接 id (带官方文本)。
- 存官方 SoM 图路径 + 候选(id + text)。
GT 由 DOM 结构确定性导出(标题↔comments 同属一个 submission 容器)。
"""
from causalact_runtime import location
import os, sys, json, re, tempfile

os.environ.setdefault("DATASET", "visualwebarena")
os.environ.setdefault("REDDIT", "http://127.0.0.1:9999")
os.environ.setdefault("SHOPPING", "http://127.0.0.1:7770")
os.environ.setdefault("CLASSIFIEDS", "http://127.0.0.1:9980")

os.environ.setdefault("WIKIPEDIA", "http://127.0.0.1:8888")
os.environ.setdefault("HOMEPAGE", "http://127.0.0.1:4399")



sys.path.insert(0, location('CAUSALACT_WEB_SCRIPTS'))
from causalact.data.intervention_schema import InterventionSample, ActionRec, dump_jsonl
from browser_env import ScriptBrowserEnv
from PIL import Image

REDDIT = os.environ["REDDIT"]
AUTH = location('CAUSALACT_VWA_AUTH', '')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/nav_official')
OUT = location('CAUSALACT_RUN_DIR', 'web/data/nav_official_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)


def mkcfg(url):
    cfg = {"task_id": 99000, "require_login": True, "storage_state": AUTH,
           "start_url": url, "geolocation": None, "intent_template": "", "instantiation_dict": {},
           "intent": "x", "require_reset": False,
           "eval": {"eval_types": ["string_match"], "reference_answers": {"must_include": ["x"]}},
           "sites": ["reddit"]}
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cfg, tf); tf.close()
    return tf.name


def parse_som(text_obs):
    """解析官方 SoM 文本 `[id] [TAG] [text]` → list of (id, tag, text)。"""
    out = []
    for line in text_obs.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s*\[([A-Z]+)\]\s*\[(.*)\]\s*$", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3).strip()))
    return out


def get_forums(env):
    fn = mkcfg(f"{REDDIT}/forums")
    obs, info = env.reset(options={"config_file": fn})
    els = parse_som(obs.get("text", ""))
    # forum links: text like a forum name, href via /f/ — but text obs 没 href;
    # 退回: 用 playwright page 读 forum 列表
    page = env.page
    fs = page.evaluate("""() => Array.from(document.querySelectorAll('a[href^="/f/"]')).map(a=>a.getAttribute('href')).filter(h=>h.split('/').length===3)""")
    return sorted(set(fs))


def build(max_forums=25, variants=("", "/new", "/top", "/hot?page=2", "/new?page=2", "/?page=3")):
    env = ScriptBrowserEnv(headless=True, observation_type="image_som",
                           current_viewport_only=True, viewport_size={"width": 1280, "height": 720})
    samples = []
    stats = {"pages": 0, "samples": 0, "no_posts": 0}
    forums = get_forums(env)[:max_forums]
    print(f"forums: {len(forums)}", flush=True)
    for furl in forums:
        fname = furl.split("/")[-1]
        for v in variants:
            url = f"{REDDIT}{furl}{v}"
            try:
                obs, info = env.reset(options={"config_file": mkcfg(url)})
            except Exception as e:
                continue
            text_obs = obs.get("text", "")
            som_list = parse_som(text_obs)
            # 用 playwright 把 post id ↔ (title, comments 元素文本) 对应; 官方 SoM id 需与 comments 链接对齐
            page = env.page
            # 抓每个 submission: title 文本 + 它的 comments 链接文本(用于在 som_list 里定位 id)
            posts = page.evaluate("""() => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                const byId = {};
                for (const a of links) {
                    const h=a.getAttribute('href')||''; const m=h.match(/\\/f\\/[^/]+\\/(\\d+)/); if(!m) continue;
                    const pid=m[1]; const t=(a.textContent||'').trim();
                    if(!byId[pid]) byId[pid]={title:null, comments:null};
                    if(/comment/i.test(t)) byId[pid].comments=t; else if(t.length>=6 && !byId[pid].title) byId[pid].title=t;
                }
                return Object.values(byId).filter(o=>o.title&&o.comments);
            }""")
            if len(posts) < 2:
                stats["no_posts"] += 1; continue
            # 官方 SoM 里找 comments 链接元素 (TAG=A, text 含 'comment')
            som_comments = [(sid, txt) for (sid, tag, txt) in som_list if tag == "A" and "comment" in txt.lower()]
            if len(som_comments) < 2:
                stats["no_posts"] += 1; continue
            # 把 playwright posts 的 comments 文本 匹配到 som id (按 comments 文本精确匹配, 如 "16 comments")
            cand = []
            title_by_somid = {}
            for (sid, txt) in som_comments:
                # 找 title: playwright posts 里 comments==txt 的那个
                match = next((p for p in posts if p["comments"].strip().lower() == txt.strip().lower()), None)
                if match:
                    title_by_somid[sid] = match["title"][:80]
                    cand.append({"id": sid, "text": txt})
            if len(cand) < 2:
                stats["no_posts"] += 1; continue
            stats["pages"] += 1
            img = Image.fromarray(obs["image"])
            img_path = f"{IMG_DIR}/{fname}{v.replace('/','_')}.png"
            img.save(img_path)
            for sid, title in title_by_somid.items():
                intent = f"Navigate to the comment section of the post titled \"{title}\"."
                act = ActionRec(action_type="click", target_id=sid, coordinates=None, evidence_bbox=None, confidence=1.0)
                s = InterventionSample(
                    task_id=f"{fname}{v.replace('/','_')}_s{sid}", site="reddit", intent=intent,
                    screenshot_path=img_path, image_size=[1280, 720],
                    base_action=act, original_action=act,
                    oracle_metadata={"forum": fname, "task_type": "navigate", "gt_title": title,
                                     "candidates": cand, "n_cand": len(cand), "som_official": True},
                    data_split="train")
                samples.append(s); stats["samples"] += 1
    env.close()
    dump_jsonl(samples, OUT)
    print(json.dumps(stats, indent=2)); print(f"saved {len(samples)} -> {OUT}", flush=True)


if __name__ == "__main__":
    build()
