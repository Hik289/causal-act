"""E3: 构造 stop-sensitivity 数据集 (官方 SoM, Oracle 核验)。

每条 = (截图, intent, 最近动作历史, GT: stop / continue)。
三类样本:
  1. s_done (目标达成 → GT stop): 导航类 intent, 当前已在**目标帖子的评论区页**
     (URL 含 /f/xxx/<id>/ 且页面是单帖详情 = 评论区). Oracle: URL 匹配目标帖 id.
  2. s_continue (未达成 → GT continue): 同 intent, 当前还在**列表页** (未进入帖子).
  3. looping-negative (looping 但未达成 → GT continue): 列表页 + 动作历史显示重复动作
     (base 卡住), 但目标未达成 → **仍应 continue** (别因 loop 就 stop). 防 V 退化成"一loop就stop".

intent 模板: "Navigate to the comment section of the post titled '<title>'."
官方 SoM 图 (ScriptBrowserEnv image_som). forum 字段供 forum-disjoint 切分.
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
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/stop')
OUT = location('CAUSALACT_RUN_DIR', 'web/data/stop_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)

FALLBACK_FORUMS = ["/f/AskReddit", "/f/Futurology", "/f/Jokes", "/f/LifeProTips", "/f/Music",
    "/f/Showerthoughts", "/f/books", "/f/explainlikeimfive", "/f/funny", "/f/gaming",
    "/f/memes", "/f/mildlyinteresting", "/f/movies", "/f/news", "/f/personalfinance",
    "/f/pics", "/f/science", "/f/space", "/f/sports", "/f/technology",
    "/f/television", "/f/tifu", "/f/todayilearned", "/f/videos", "/f/worldnews"]


def parse_som(text_obs):
    els = []
    for line in text_obs.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s*\[([A-Z]+)\]\s*\[(.*)\]\s*$", line)
        if m:
            els.append((m.group(1), m.group(2), m.group(3).strip()))
    return els


def _cfg(url):
    cfg = {"task_id": 99999, "require_login": True, "storage_state": AUTH, "start_url": url,
           "geolocation": None, "intent_template": "", "instantiation_dict": {}, "intent": "t",
           "require_reset": False, "eval": {"eval_types": ["string_match"], "reference_answers": {"must_include": ["x"]}},
           "sites": ["reddit"]}
    tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(cfg, tf); tf.close()
    return tf.name


def list_posts(els):
    """从列表页 SoM 找帖子: (post_id, title, comments_href)。"""
    byid = {}
    for (eid, tag, txt) in els:
        pass
    return els


def extract_posts_from_list(env):
    """返回 [(post_id, title)] from current list page via JS."""
    return env.page.evaluate("""() => {
        const links=Array.from(document.querySelectorAll('a[href]'));
        const byId={};
        for(const a of links){const h=a.getAttribute('href')||''; const m=h.match(/\\/f\\/[^/]+\\/(\\d+)/); if(!m) continue;
          const pid=m[1]; const t=(a.textContent||'').trim();
          if(!byId[pid]) byId[pid]={title:null, href:null};
          if(/comment/i.test(t)){byId[pid].href=h;}
          else if(t.length>=12 && !byId[pid].title){byId[pid].title=t; byId[pid].thref=h;}}
        return Object.entries(byId).filter(([k,v])=>v.title&&v.href).slice(0,6).map(([k,v])=>({pid:k,title:v.title,href:v.thref}));
    }""")


def main(limit_forums=25, max_samples=160):
    env = ScriptBrowserEnv(headless=True, observation_type="image_som",
                           current_viewport_only=True, viewport_size={"width": 1280, "height": 720})
    samples = []
    stats = {"s_done": 0, "s_continue": 0, "loop_neg": 0, "env_recreate": 0}
    forums = FALLBACK_FORUMS[:limit_forums]
    for furl in forums:
        fname = furl.split("/")[-1]
        if len(samples) >= max_samples:
            break
        # 列表页
        try:
            obs, info = env.reset(options={"config_file": _cfg(f"{REDDIT}{furl}")})
        except Exception:
            stats["env_recreate"] += 1
            try: env.close()
            except Exception: pass
            env = ScriptBrowserEnv(headless=True, observation_type="image_som", current_viewport_only=True, viewport_size={"width":1280,"height":720})
            continue
        try:
            posts = extract_posts_from_list(env)
        except Exception:
            posts = []
        if len(posts) < 2:
            continue
        list_img = f"{IMG_DIR}/{fname}_list.png"
        Image.fromarray(obs["image"]).save(list_img)
        # 对前 3 个帖子各生成一组 (s_continue + loop_neg + s_done)
        for pi, target in enumerate(posts[:3]):
            intent = f"Navigate to the comment section of the post titled \"{target['title'][:80]}\"."
            # s_continue: 列表页, 目标未达成 → continue
            samples.append({"task_id": f"{fname}_{pi}_continue", "site": "reddit", "intent": intent,
                "screenshot_path": list_img, "image_size": [1280, 720], "action_history": ["None"],
                "original_action": {"action_type": "continue"}, "gt_stop": False,
                "oracle_metadata": {"forum": fname, "kind": "s_continue", "target_pid": target["pid"],
                                    "goal_reached": False, "som_official": True}})
            stats["s_continue"] += 1
            # loop_neg: 列表页 + 重复动作历史, 目标未达成 → continue
            samples.append({"task_id": f"{fname}_{pi}_loopneg", "site": "reddit", "intent": intent,
                "screenshot_path": list_img, "image_size": [1280, 720],
                "action_history": ["click [6]", "click [6]", "click [6]"],
                "original_action": {"action_type": "continue"}, "gt_stop": False,
                "oracle_metadata": {"forum": fname, "kind": "loop_neg", "target_pid": target["pid"],
                                    "goal_reached": False, "looping": True, "som_official": True}})
            stats["loop_neg"] += 1
            # s_done: 进入 target 帖子评论区 → 达成 → stop (Oracle: URL 含 pid)
            post_url = f"{REDDIT}{target['href']}" if target['href'].startswith('/') else target['href']
            try:
                obs2, info2 = env.reset(options={"config_file": _cfg(post_url)})
                cur_url = env.page.url
                reached = ("/" + target["pid"] + "/") in cur_url or ("/" + target["pid"]) in cur_url
                if reached:
                    done_img = f"{IMG_DIR}/{fname}_{target['pid']}_done.png"
                    Image.fromarray(obs2["image"]).save(done_img)
                    samples.append({"task_id": f"{fname}_{pi}_done", "site": "reddit", "intent": intent,
                        "screenshot_path": done_img, "image_size": [1280, 720],
                        "action_history": ["click [some]"],
                        "original_action": {"action_type": "stop"}, "gt_stop": True,
                        "oracle_metadata": {"forum": fname, "kind": "s_done", "target_pid": target["pid"],
                                            "goal_reached": True, "cur_url": cur_url, "som_official": True}})
                    stats["s_done"] += 1
            except Exception:
                pass
        with open(OUT, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    try: env.close()
    except Exception: pass
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(json.dumps(stats, indent=2)); print("saved", len(samples), "->", OUT)


if __name__ == "__main__":
    main()
