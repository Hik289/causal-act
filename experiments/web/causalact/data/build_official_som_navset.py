"""修复版: 用 VWA 官方 SoM pipeline 生成导航 grounding 数据 (A 与 base 看逐像素相同图)。

对每个 forum 列表页, 用 ScriptBrowserEnv(image_som) 取:
  - obs['image']: 官方 SoM 标注图 (框画在像素上, base 也看这张)
  - obs['text']:  官方 SoM 元素列表 [id] [TAG] [text]
从元素列表找帖子的 Comments 链接 (TAG=A, text 含 'comment') + 对应标题。
GT = 该帖 Comments 链接的官方 SoM id。候选 = 本页所有 Comments 链接 id。
intent = "Navigate to the comment section of the post titled '<title>'."

输出: official_nav_set.jsonl, 图存 /data2 (官方 SoM 图)。
forum 字段供 forum-disjoint 切分。
⚠️ 关键: 图是官方 SoM 图, A 看得见编号框 → 与 base 输入对称。
"""
from causalact_runtime import location
import os, sys, json, re, tempfile
os.environ.setdefault("DATASET", "visualwebarena")
for k, v in [("REDDIT", "http://127.0.0.1:9999"), ("SHOPPING", "http://127.0.0.1:7770"),
             ("CLASSIFIEDS", "http://127.0.0.1:9980"),
             ("WIKIPEDIA", "http://127.0.0.1:8888"), ("HOMEPAGE", "http://127.0.0.1:4399")]:
    os.environ.setdefault(k, v)

from browser_env import ScriptBrowserEnv
from PIL import Image

REDDIT = os.environ["REDDIT"]
AUTH = location('CAUSALACT_VWA_AUTH', '')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/official_nav')
OUT = location('CAUSALACT_RUN_DIR', 'web/data/official_nav_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)


def parse_som(text_obs):
    """解析 [id] [TAG] [text] 行 → list of (id, tag, text)。"""
    els = []
    for line in text_obs.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s*\[([A-Z]+)\]\s*\[(.*)\]\s*$", line)
        if m:
            els.append((m.group(1), m.group(2), m.group(3).strip()))
    return els


def extract_posts(els):
    """从 SoM 元素找帖子: Comments 链接(A, text 含 comment) + 标题链接(A, 长文本, 非comment)。
    Postmill: 标题链接在 comments 链接前面附近。用顺序启发式配对。"""
    posts = []
    # 收集所有标题候选(A, 文本>=10, 不含 comment/submitted/ago) 和 comments 链接
    titles = []
    comments = []
    for (eid, tag, txt) in els:
        low = txt.lower()
        if tag != "A":
            continue
        if "comment" in low and re.match(r"^\d+\s+comment", low):
            comments.append((eid, txt))
        elif len(txt) >= 10 and "comment" not in low and "submitted" not in low and "ago" not in low:
            titles.append((eid, txt))
    # 配对: 每个 comments 链接, 在其前面窗口内选**最长文本**的标题链接 (排除短用户名/版块名)。
    # 帖子标题通常是窗口里最长的 A 文本; 提交者用户名较短。
    order = {eid: i for i, (eid, tag, txt) in enumerate(els)}
    for (cid, ctxt) in comments:
        ci = order.get(cid, 1e9)
        cands_t = [(tid, ttxt) for (tid, ttxt) in titles
                   if order.get(tid, -1) < ci and (ci - order.get(tid, -1)) <= 8]
        if not cands_t:
            continue
        # 选窗口内最长文本 = 帖子标题 (用户名/版块名短)
        best = max(cands_t, key=lambda x: len(x[1]))
        if len(best[1]) >= 12:  # 真标题通常 >=12 字符
            posts.append({"comments_id": "elem_" + cid, "title": best[1][:90]})
    # 去重 (同标题)
    seen = set(); uniq = []
    for p in posts:
        if p["title"] not in seen:
            seen.add(p["title"]); uniq.append(p)
    return uniq


def get_forums(env):
    cfg = _mk_cfg("http://127.0.0.1:9999/forums")
    obs, info = env.reset(options={"config_file": cfg})
    els = parse_som(obs.get("text", ""))
    forums = []
    for (eid, tag, txt) in els:
        pass
    # 直接从 page 拿 forum 列表更稳
    fs = env.page.evaluate("""() => Array.from(document.querySelectorAll('a[href^="/f/"]')).map(a=>a.getAttribute('href')).filter(h=>h.split('/').length===3)""")
    return sorted(set(fs))


def _mk_cfg(url):
    cfg = {"task_id": 99999, "require_login": True, "storage_state": AUTH,
           "start_url": url, "geolocation": None, "intent_template": "", "instantiation_dict": {},
           "intent": "t", "require_reset": False,
           "eval": {"eval_types": ["string_match"], "reference_answers": {"must_include": ["x"]}},
           "sites": ["reddit"]}
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(cfg, tf); tf.close()
    return tf.name


def _new_env():
    return ScriptBrowserEnv(headless=True, observation_type="image_som",
                            current_viewport_only=True, viewport_size={"width": 1280, "height": 720})


FALLBACK_FORUMS = ["/f/AskReddit", "/f/Futurology", "/f/Jokes", "/f/LifeProTips", "/f/Music",
    "/f/Showerthoughts", "/f/books", "/f/explainlikeimfive", "/f/funny", "/f/gaming",
    "/f/memes", "/f/mildlyinteresting", "/f/movies", "/f/news", "/f/personalfinance",
    "/f/pics", "/f/science", "/f/space", "/f/sports", "/f/technology",
    "/f/television", "/f/tifu", "/f/todayilearned", "/f/videos", "/f/worldnews"]


def main(limit_forums=25):
    env = _new_env()
    try:
        forums = get_forums(env)[:limit_forums]
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        env = _new_env()
        forums = FALLBACK_FORUMS[:limit_forums]
    if len(forums) < 5:
        forums = FALLBACK_FORUMS[:limit_forums]
    print("forums:", len(forums), flush=True)
    samples = []
    stats = {"pages": 0, "samples": 0, "no_posts": 0, "env_recreate": 0}
    OUT_P = OUT  # incremental save
    for furl in forums:
        fname = furl.split("/")[-1]
        for variant in ["", "/new", "/top"]:
            url = f"{REDDIT}{furl}{variant}"
            try:
                obs, info = env.reset(options={"config_file": _mk_cfg(url)})
            except Exception as e:
                # env 可能已损坏(timeout后browser context死) → 重建 env
                stats["env_recreate"] += 1
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    env = _new_env()
                except Exception:
                    pass
                continue
            els = parse_som(obs.get("text", ""))
            posts = extract_posts(els)
            if len(posts) < 2:
                stats["no_posts"] += 1; continue
            stats["pages"] += 1
            img_path = f"{IMG_DIR}/{fname}{variant.replace('/','_')}.png"
            Image.fromarray(obs["image"]).save(img_path)
            cand_ids = [p["comments_id"] for p in posts]
            # 完整 SoM 可点击元素 id (A/BUTTON/INPUT) = 真实 VWA 难度候选集 (不只 comments 链接)
            full_clickable = ["elem_" + eid for (eid, tag, txt) in els if tag in ("A", "BUTTON", "INPUT")]
            for p in posts:
                intent = f"Navigate to the comment section of the post titled \"{p['title']}\"."
                s = {"task_id": f"{fname}{variant.replace('/','_')}_{p['comments_id']}",
                     "site": "reddit", "intent": intent,
                     "screenshot_path": img_path, "image_size": [1280, 720],
                     "base_action": {"action_type": "click", "target_id": p["comments_id"]},
                     "original_action": {"action_type": "click", "target_id": p["comments_id"]},
                     "oracle_metadata": {"forum": fname, "task_type": "navigate",
                                         "gt_title": p["title"], "candidate_ids": cand_ids,
                                         "n_cand": len(cand_ids), "som_official": True,
                                         "full_clickable_ids": full_clickable,
                                         "n_full": len(full_clickable)},
                     "data_split": "train"}
                samples.append(s); stats["samples"] += 1
            # 增量保存 (防中途崩溃丢全部)
            with open(OUT, "w") as f:
                for ss in samples:
                    f.write(json.dumps(ss) + "\n")
    try:
        env.close()
    except Exception:
        pass
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(json.dumps(stats, indent=2)); print("saved", len(samples), "->", OUT)


if __name__ == "__main__":
    main()
