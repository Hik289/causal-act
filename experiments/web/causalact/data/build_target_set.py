"""I-2 (H0.method_2.3): 构造 target 干预集 — 真实后端状态改变 (非 DOM 伪装)。

机制: Postmill forum **subscribe toggle** (真实 POST /f/xxx/subscribe, 已 anchor_4 证可核验可逆)。

每个 forum 产出**两个方向**的因果对 (s, s_c), 共享同一 intent, 正确动作集合确实改变:
  方向 A (intent='subscribe to this forum'):
    s   = 未订阅 → A*(s)   = click(Subscribe 按钮)
    s_c = 已订阅 → A*(s_c) = stop (目标已达成, 无需动作)
  方向 B (intent='unsubscribe from this forum'):
    s   = 已订阅 → A*(s)   = click(Unsubscribe 按钮)
    s_c = 未订阅 → A*(s_c) = stop

semantic-target oracle 核验: A*(s) != A*(s_c) (动作集合确实改变) 才入库 (README §2.3)。
每对保存 original(s) 截图 + edited(s_c) 截图 + GT 动作 + 候选(含 Subscribe 控件 + stop 选项)。

截图带 SoM 框 (订阅控件框号 + 一个显式 "stop/done" 语义), A 要么 click 订阅框, 要么输出 stop。

intent 释义 ×3 增广 → 25 forum × 2 方向 × 3 paraphrase = 150 样本。
还原: 每 forum 操作后复位到原订阅态; 批量后可 docker reset 兜底。
"""
from causalact_runtime import location
import os, sys, json, re, random
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oracle.forum_oracles import JS_RENDER_FACTS, oracle_render_validity, oracle_semantic_target
from data.intervention_schema import InterventionSample, EditSpec, ActionRec, dump_jsonl

REDDIT = os.environ.get("REDDIT", "http://127.0.0.1:9999")
AUTH = location('CAUSALACT_VWA_AUTH', '')
IMG_DIR = location('CAUSALACT_RUN_DIR', 'intervention_imgs/target')
OUT_JSONL = location('CAUSALACT_RUN_DIR', 'web/data/target_set.jsonl')
os.makedirs(IMG_DIR, exist_ok=True)
VIEWPORT = {"width": 1280, "height": 720}

INTENT_SUB = [
    "Subscribe to this forum.",
    "Make sure you are subscribed to this forum.",
    "Join this community by subscribing to it.",
]
INTENT_UNSUB = [
    "Unsubscribe from this forum.",
    "Make sure you are not subscribed to this forum.",
    "Leave this community by unsubscribing from it.",
]

# 在订阅控件上画一个 SoM 框 (elem_1 = subscribe/unsubscribe 按钮)
def js_draw_sub_box():
    return """
(() => {
  document.querySelectorAll('.civ-som-box').forEach(e=>e.remove());
  const f = Array.from(document.querySelectorAll('form')).find(f=>/subscribe/i.test(f.action||''));
  if(!f) return {found:false};
  const btn = f.querySelector('button');
  const r = btn.getBoundingClientRect();
  const d=document.createElement('div'); d.className='civ-som-box';
  d.style.cssText=`position:absolute; left:${r.x}px; top:${r.y+window.scrollY}px; width:${r.width}px; height:${r.height}px; border:3px solid #e6194b; z-index:99999; pointer-events:none; box-sizing:border-box;`;
  const lab=document.createElement('span'); lab.textContent='1';
  lab.style.cssText='position:absolute; left:0; top:-16px; background:#e6194b; color:#fff; font-size:13px; font-weight:bold; padding:0 4px;';
  d.appendChild(lab); document.body.appendChild(d);
  return {found:true, center:[Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)],
          bbox:[Math.round(r.x),Math.round(r.y),Math.round(r.x+r.width),Math.round(r.y+r.height)]};
})()
"""


def read_sub_state(page):
    return page.evaluate("""() => {
        const f=Array.from(document.querySelectorAll('form')).find(f=>/subscribe/i.test(f.action||''));
        if(!f) return {state:'unknown', has_form:false};
        return {state: /\\/unsubscribe/i.test(f.action)? 'subscribed':'unsubscribed', has_form:true};
    }""")


def click_sub(page):
    try:
        page.click('form.subscribe-form button', timeout=5000)
        page.wait_for_load_state('domcontentloaded'); page.wait_for_timeout(700)
        return True
    except Exception:
        return False


def get_forums(page, limit=25):
    page.goto(f"{REDDIT}/forums", wait_until="domcontentloaded"); page.wait_for_timeout(600)
    fs = page.evaluate("""() => Array.from(document.querySelectorAll('a[href^="/f/"]')).map(a=>a.getAttribute('href')).filter(h=>h.split('/').length===3)""")
    return sorted(set(fs))[:limit]


# 候选集: elem_1 = subscribe 控件(图上框); 语义动作空间 = {click elem_1, stop}
# A 的输出: "elem_1" (点订阅控件) 或 "stop" (已达成)
def make_sample(tid, intent, img_path, gt_action, state_label, pair_id, box):
    cands = [{"id": "elem_1", "bbox": box["bbox"], "center": box["center"]},
             {"id": "stop", "bbox": None, "center": None}]
    return InterventionSample(
        task_id=tid, site="reddit", intent=intent,
        screenshot_path=img_path, image_size=[VIEWPORT["width"], VIEWPORT["height"]],
        base_action=ActionRec(**gt_action), original_action=ActionRec(**gt_action),
        oracle_metadata={"pair_id": pair_id, "state": state_label, "candidates": cands,
                         "target_control": "subscribe_toggle"},
        data_split="train")


def build(forums_limit=25):
    samples = []
    stats = {"forums": 0, "pairs": 0, "flip_ok": 0, "flip_fail": 0, "render_fail": 0}
    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=AUTH, viewport=VIEWPORT)
        page = ctx.new_page()
        forums = get_forums(page, forums_limit)
        print(f"forums: {len(forums)}")

        for furl in forums:
            fname = furl.split("/")[-1]
            try:
                page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded", timeout=15000); page.wait_for_timeout(500)
            except Exception:
                continue
            st0 = read_sub_state(page)
            if not st0.get("has_form"):
                continue
            stats["forums"] += 1
            orig_state = st0["state"]  # 记录原态用于最终还原

            # 规范到"未订阅"起点 (方向 A 的 s)
            if st0["state"] == "subscribed":
                click_sub(page); page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded"); page.wait_for_timeout(400)
            # 现在应为 unsubscribed
            st_unsub = read_sub_state(page)
            if st_unsub["state"] != "unsubscribed":
                continue
            if not oracle_render_validity(page.evaluate(JS_RENDER_FACTS)).valid:
                stats["render_fail"] += 1; continue
            box_u = page.evaluate(js_draw_sub_box()); page.wait_for_timeout(120)
            if not box_u.get("found"): continue
            img_unsub = f"{IMG_DIR}/{fname}_unsub.png"; page.screenshot(path=img_unsub)

            # toggle → subscribed (方向 A 的 s_c)
            click_sub(page); page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded"); page.wait_for_timeout(400)
            st_sub = read_sub_state(page)
            # semantic-target oracle: 状态确实改变?
            v = oracle_semantic_target({"state": "unsubscribed"}, {"state": st_sub["state"]}, expect_change=True)
            if not v.valid or st_sub["state"] != "subscribed":
                stats["flip_fail"] += 1
                # 还原尽力
                continue
            stats["flip_ok"] += 1
            if not oracle_render_validity(page.evaluate(JS_RENDER_FACTS)).valid:
                stats["render_fail"] += 1; continue
            box_s = page.evaluate(js_draw_sub_box()); page.wait_for_timeout(120)
            if not box_s.get("found"): continue
            img_sub = f"{IMG_DIR}/{fname}_sub.png"; page.screenshot(path=img_sub)

            # 构造两方向 × 3 paraphrase
            for k in range(3):
                pid_a = f"{fname}_A{k}"
                # 方向 A (subscribe): s=unsub→click elem_1 ; s_c=sub→stop
                samples.append(make_sample(f"{pid_a}_s", INTENT_SUB[k], img_unsub,
                    {"action_type": "click", "target_id": "elem_1", "coordinates": box_u["center"], "evidence_bbox": box_u["bbox"]},
                    "unsubscribed", pid_a, box_u))
                samples.append(make_sample(f"{pid_a}_sc", INTENT_SUB[k], img_sub,
                    {"action_type": "stop", "target_id": "stop", "coordinates": None, "evidence_bbox": box_s["bbox"]},
                    "subscribed", pid_a, box_s))
                pid_b = f"{fname}_B{k}"
                # 方向 B (unsubscribe): s=sub→click elem_1 ; s_c=unsub→stop
                samples.append(make_sample(f"{pid_b}_s", INTENT_UNSUB[k], img_sub,
                    {"action_type": "click", "target_id": "elem_1", "coordinates": box_s["center"], "evidence_bbox": box_s["bbox"]},
                    "subscribed", pid_b, box_s))
                samples.append(make_sample(f"{pid_b}_sc", INTENT_UNSUB[k], img_unsub,
                    {"action_type": "stop", "target_id": "stop", "coordinates": None, "evidence_bbox": box_u["bbox"]},
                    "unsubscribed", pid_b, box_u))
                stats["pairs"] += 2

            # 还原到原态
            cur = read_sub_state(page)
            if cur["state"] != orig_state:
                click_sub(page); page.wait_for_timeout(300)

        b.close()
    # 标注 pair 的 targeted_changes=True (动作集合确实改变: click vs stop)
    dump_jsonl(samples, OUT_JSONL)
    print(json.dumps(stats, indent=2)); print(f"saved {len(samples)} samples -> {OUT_JSONL}")
    return samples, stats


if __name__ == "__main__":
    build(25)
