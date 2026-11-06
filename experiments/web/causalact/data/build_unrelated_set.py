"""I-1: 构造 unrelated 干预集 (original / unrelated 配对) on forum。

流程 (每条样本):
  1. 打开 forum 页 (task start_url), 截图 original
  2. 取 SoM 可交互元素 bbox, 选一个 GT 目标 (按 intent 关键词匹配或取显著可交互元素)
  3. 应用 unrelated 编辑 (背景色 + 装饰图 hue-rotate/grayscale + 无关图标), 截图 unrelated
  4. oracle 核验: render-validity(两张图都有效) + unrelated-preservation(GT 目标编辑后仍在且未禁用)
  5. 过核验 → 入库 (InterventionSample JSONL, 截图存 /data2)

GT 动作: click(target_id), target_id = SoM element id (稳定锚点), coordinates=元素中心,
evidence_bbox = 元素 bbox。unrelated_action = original_action (specificity: 正确动作不变)。

口径: forum 可完成子集 (排除跨站点 task, 对齐 anchor_1)。
"""
from causalact_runtime import location
import os, json, sys, re, hashlib
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle.forum_oracles import (
    JS_RENDER_FACTS, js_target_exists,
    oracle_render_validity, oracle_unrelated_preservation,
)
from data.intervention_schema import (
    InterventionSample, EditSpec, ActionRec, dump_jsonl,
)

REDDIT = os.environ.get("REDDIT", "http://127.0.0.1:9999")
AUTH = location('CAUSALACT_VWA_AUTH', '')
VWA_CONFIG = location('CAUSALACT_VWA_DIR', 'config_files/vwa/test_reddit')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/unrelated')
OUT_JSONL = location('CAUSALACT_RUN_DIR', 'web/data/unrelated_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)

VIEWPORT = {"width": 1280, "height": 720}


# unrelated 编辑: 改背景 + 装饰图滤镜 + 无关图标色 (不碰可交互元素的几何/文字)
JS_UNRELATED_EDIT = """
(() => {
  document.body.style.backgroundColor = '#f2c744';
  let imgs = 0;
  document.querySelectorAll('img').forEach(im => { im.style.filter = 'hue-rotate(160deg) saturate(1.4)'; imgs++; });
  // 无关装饰: 页脚/侧栏非交互区域轻微变色 (不碰 button/a/input)
  document.querySelectorAll('.sidebar, footer, .footer, aside').forEach(e => { e.style.backgroundColor = '#cde'; });
  return { imgs_changed: imgs };
})()
"""

# 取可交互元素 (SoM 风格), 返回 [{id, tag, text, center:[x,y], bbox:[x0,y0,x1,y1]}]
JS_GET_INTERACTABLES = """
(() => {
  const sel = 'a[href], button, input:not([type=hidden]), [role=button], [role=link]';
  const els = Array.from(document.querySelectorAll(sel));
  const out = [];
  let i = 1;
  for (const e of els) {
    const r = e.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    // 仅视口内可见元素 (排除 off-screen skip-link 等, x/y 都要在界内)
    if (r.top < 0 || r.top > 720) continue;
    if (r.left < 0 || r.left > 1280) continue;
    if (r.right < 0 || r.bottom < 0) continue;
    const text = (e.textContent || e.value || e.getAttribute('aria-label') || '').trim().slice(0, 40);
    out.push({
      id: 'elem_' + i,
      tag: e.tagName,
      href: e.getAttribute('href') || '',
      text: text,
      center: [Math.round(r.x + r.width/2), Math.round(r.y + r.height/2)],
      bbox: [Math.round(r.x), Math.round(r.y), Math.round(r.x + r.width), Math.round(r.y + r.height)],
    });
    i++;
    if (i > 60) break;
  }
  return out;
})()
"""


def load_completable_tasks(max_n=80):
    """加载 forum 可完成的 reddit task (单站点, start_url 指向 forum)。"""
    tasks = []
    files = sorted(os.listdir(VWA_CONFIG), key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 1e9)
    for fn in files:
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(VWA_CONFIG, fn)))
        su = d.get("start_url", "")
        # 单站点 forum: start_url 只含 reddit:9999, 不含 wikipedia/shopping
        if "9999" in su and "8888" not in su and "7770" not in su and " |AND| " not in su:
            tasks.append({
                "task_id": str(d.get("task_id", fn.split(".")[0])),
                "intent": d.get("intent", ""),
                "start_url": su.split(" |AND| ")[0] if su else f"{REDDIT}/",
            })
        if len(tasks) >= max_n:
            break
    return tasks


SITE_CHROME = {"postmill", "forums", "wiki", "submit", "log in", "login", "sign up",
               "search query", "", "messages", "notifications", "hot", "new", "top",
               "controversial", "active", "subscribe", "unsubscribe", "hide this forum"}


def _is_content(e):
    """内容型目标 (帖子标题/评论链接), 排除站点 chrome 与导航。"""
    t = (e["text"] or "").strip().lower()
    if t in SITE_CHROME or len(t) < 4:
        return False
    href = e.get("href", "")
    # 帖子/评论链接: /f/xxx/<id>/ 或含 comment
    if "comment" in t:
        return True
    if re.search(r"/f/[^/]+/\d+/", href):
        return True
    # 带实质文字的链接 (帖子标题通常较长)
    if e["tag"] == "A" and len(t) >= 10:
        return True
    return False


def pick_gt_target(intent, elems):
    """选 GT 目标: 只在**内容型元素**里选 (排除站点 chrome), 优先 intent 关键词命中。"""
    content = [e for e in elems if _is_content(e)]
    if not content:
        return None, "no_content"
    itl = intent.lower()
    kws = [w for w in re.findall(r"[a-z]{4,}", itl)
           if w not in ("this", "that", "with", "find", "navigate", "please", "trying",
                        "section", "which", "there", "their", "about", "post", "comment",
                        "research", "doing", "some", "want", "would", "like", "tell", "many")]
    best = None; best_score = 0
    for e in content:
        t = e["text"].lower()
        score = sum(1 for k in kws if k in t)
        if score > best_score:
            best_score = score; best = e
    if best is not None and best_score > 0:
        return best, "intent_match"
    # fallback: 第一个评论链接, 否则第一个内容链接
    for e in content:
        if "comment" in e["text"].lower():
            return e, "comment_link"
    return content[0], "first_content"


def build(max_samples=60):
    tasks = load_completable_tasks(max_n=120)
    print(f"loaded {len(tasks)} forum-completable tasks")
    samples = []
    stats = {"attempted": 0, "render_fail": 0, "no_elem": 0, "unrel_preserve_fail": 0, "ok": 0}

    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=AUTH, viewport=VIEWPORT)
        page = ctx.new_page()

        for t in tasks:
            if len(samples) >= max_samples:
                break
            stats["attempted"] += 1
            tid = t["task_id"]
            try:
                page.goto(t["start_url"], wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(500)
            except Exception:
                continue

            # original 渲染核验
            facts = page.evaluate(JS_RENDER_FACTS)
            if not oracle_render_validity(facts).valid:
                stats["render_fail"] += 1
                continue

            elems = page.evaluate(JS_GET_INTERACTABLES)
            if not elems:
                stats["no_elem"] += 1
                continue
            gt, how = pick_gt_target(t["intent"], elems)
            if gt is None:
                stats["no_elem"] += 1
                continue

            # 截图 original
            orig_png = f"{IMG_DIR}/{tid}_orig.png"
            page.screenshot(path=orig_png)
            img_w, img_h = VIEWPORT["width"], VIEWPORT["height"]

            # GT target 编辑前存在性 (用文字锚点)
            tgt_text = gt["text"][:15] if gt["text"] else None
            before_exists = page.evaluate(js_target_exists(tgt_text)) if tgt_text else {"found": True, "disabled": False}

            # 应用 unrelated 编辑 + 截图
            edit_info = page.evaluate(JS_UNRELATED_EDIT)
            page.wait_for_timeout(200)
            unrel_png = f"{IMG_DIR}/{tid}_unrel.png"
            page.screenshot(path=unrel_png)

            # unrelated-preservation 核验: GT 目标编辑后仍在且未禁用
            after_exists = page.evaluate(js_target_exists(tgt_text)) if tgt_text else {"found": True, "disabled": False}
            if tgt_text and not oracle_unrelated_preservation(before_exists, after_exists).valid:
                stats["unrel_preserve_fail"] += 1
                continue

            # unrelated 后渲染也要有效
            facts2 = page.evaluate(JS_RENDER_FACTS)
            if not oracle_render_validity(facts2).valid:
                stats["render_fail"] += 1
                continue

            # 入库: original_action = unrelated_action = click(gt)
            act = ActionRec(action_type="click", target_id=gt["id"],
                            coordinates=gt["center"], evidence_bbox=gt["bbox"], confidence=1.0)
            s = InterventionSample(
                task_id=tid, site="reddit", intent=t["intent"],
                screenshot_path=orig_png, image_size=[img_w, img_h],
                base_action=act, original_action=act,
                unrelated_edit=EditSpec(edit_type="recolor_bg_img", category="unrelated",
                                        attribute="backgroundColor+img.filter", value="#f2c744/hue160",
                                        region_bbox=None, backend_mutated=False),
                unrelated_action=act,
                unrelated_screenshot_path=unrel_png,
                oracle_metadata={"gt_pick": how, "imgs_changed": edit_info.get("imgs_changed", 0),
                                 "render_valid": True, "unrelated_preserved": True,
                                 "n_interactables": len(elems),
                                 # 持久化候选元素 (id+text+center) 供 A 训练: 取 GT + 至多 9 个干扰
                                 "candidates": [
                                     {"id": e["id"], "text": e["text"][:40], "center": e["center"]}
                                     for e in ([gt] + [x for x in elems if x["id"] != gt["id"]][:9])
                                 ]},
                data_split="train",
            )
            samples.append(s)
            stats["ok"] += 1

        b.close()

    dump_jsonl(samples, OUT_JSONL)
    print(json.dumps(stats, indent=2))
    print(f"saved {len(samples)} samples -> {OUT_JSONL}")
    return samples, stats


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    build(max_samples=n)
