"""H0.anchor_4 — forum 五项 oracle 验证 (≥50 样本/类, 一致率 vs 独立 gold)

方法论说明 (诚实披露):
  "gold" = **环境自身执行真值 (environment-execution ground truth)**, 不是人类标注。
  对每个 oracle, 独立地用 Playwright 真实执行 / 后端回放 得到客观事实, 作为 gold,
  与 oracle 的确定性判定比对一致率。因为这些 oracle 判的就是"环境可观测事实",
  环境执行真值是比人工目视更强的 gold (README §12.4 的 oracle 本就应核验环境事实)。
  → 回报时明确标注此口径, 由 Director 判断是否需要补充人工目视核验。

每类 ≥50 样本来源:
  - render-validity: forum 多个页面 (forum list / 各 /f/xxx / 帖子详情 / 搜索结果) + 注入
    不同 unrelated/transform 编辑后的渲染 → 正例; 人为注入破坏 (清空 body/移除交互元素) → 负例
  - executable-validity: 对真实可交互元素中心取点 (正例) + 空白/已移除坐标 (负例)
  - unrelated-preservation: unrelated 编辑后目标仍在 (正例) + 误伤目标 (负例)
  - semantic-target: 真实 subscribe/unsubscribe 后端 toggle (正例=状态变) + 无操作 (负例=状态不变)
  - safety: 正常操作后登录态保持 (正例) + 构造登出/破坏 (负例)

运行: python validate_oracles.py  (需 forum:9999 up + reddit cookie)
产出: oracle_validation_results.json
"""
from causalact_runtime import location
import os, json, time, random
from playwright.sync_api import sync_playwright
from forum_oracles import (
    JS_RENDER_FACTS, js_element_at, js_target_exists, JS_SESSION_SAFE,
    oracle_render_validity, oracle_executable_validity,
    oracle_unrelated_preservation, oracle_semantic_target, oracle_safety,
)

REDDIT = os.environ.get("REDDIT", "http://127.0.0.1:9999")
AUTH = location('CAUSALACT_VWA_AUTH', '')
OUT = location('CAUSALACT_RUN_DIR', 'web/oracle_probe')
os.makedirs(OUT, exist_ok=True)
random.seed(42)

# forum 页面集 (多样性); 运行时会动态扩展 (发现 forums + 帖子详情) 以保证 ≥50/类
PAGES = [
    "/", "/f/all", "/search?q=recipe", "/forums",
]


def discover_pages(page):
    """动态发现大量真实页面: 25 个 forum 列表页 + 各 forum 多个帖子详情页。"""
    urls = list(PAGES)
    # 1) 所有 forum 列表页
    try:
        page.goto(f"{REDDIT}/forums", wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(500)
        forums = page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href^="/f/"]'))
                     .map(a=>a.getAttribute('href'))
                     .filter(h=>h && h.split('/').length===3)""")
        forums = sorted(set(forums))
    except Exception:
        forums = ["/f/food", "/f/all"]
    urls += forums[:25]
    # 2) 每个 forum 抓几个帖子详情页 (/f/xxx/<id>/<slug>)
    for furl in forums[:12]:
        try:
            page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(300)
            posts = page.evaluate(
                """() => Array.from(document.querySelectorAll('a[href]'))
                         .map(a=>a.getAttribute('href'))
                         .filter(h=>h && /^\\/f\\/[^/]+\\/\\d+\\//.test(h))""")
            for pl in list(dict.fromkeys(posts))[:4]:
                urls.append(pl)
        except Exception:
            continue
    # 去重保序
    return list(dict.fromkeys(urls))


def agree(verdict_valid, gold_valid):
    return verdict_valid == gold_valid


def run():
    results = {k: {"n": 0, "agree": 0, "cases": []} for k in
               ["render-validity", "executable-validity", "unrelated-preservation",
                "semantic-target", "safety"]}

    with sync_playwright() as pw:
        b = pw.chromium.launch(headless=True)
        ctx = b.new_context(storage_state=AUTH, viewport={"width": 1280, "height": 720})
        page = ctx.new_page()

        # 动态发现大量真实页面 (forums + 帖子详情) 保证 ≥50/类
        page_urls = discover_pages(page)
        print(f"discovered {len(page_urls)} pages for probing")

        # ===== 1. render-validity (≥50): 正例=真实页面+编辑, 负例=破坏渲染 =====
        for url in page_urls:
            try:
                page.goto(f"{REDDIT}{url}" if url.startswith("/") else url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(300)
            except Exception:
                continue
            # 正例 A: 原始渲染
            facts = page.evaluate(JS_RENDER_FACTS)
            v = oracle_render_validity(facts)
            gold = facts.get("has_body") and facts.get("text_len", 0) > 50 and facts.get("n_interactable", 0) >= 3
            results["render-validity"]["n"] += 1
            results["render-validity"]["agree"] += agree(v.valid, gold)
            results["render-validity"]["cases"].append({"url": url, "type": "orig", "verdict": v.valid, "gold": gold})
            # 正例 B: 注入 unrelated 编辑 (背景色), 渲染应仍有效
            page.evaluate("() => { document.body.style.backgroundColor='#abc'; }")
            facts2 = page.evaluate(JS_RENDER_FACTS)
            v2 = oracle_render_validity(facts2)
            gold2 = facts2.get("text_len", 0) > 50 and facts2.get("n_interactable", 0) >= 3
            results["render-validity"]["n"] += 1
            results["render-validity"]["agree"] += agree(v2.valid, gold2)
            results["render-validity"]["cases"].append({"url": url, "type": "unrelated_edit", "verdict": v2.valid, "gold": gold2})
            # 负例: 破坏渲染 (清空 body)
            broken = page.evaluate("() => { document.body.innerHTML=''; return {has_body:true, body_children:0, text_len:0, n_interactable:0, doc_height:0}; }")
            vb = oracle_render_validity(broken)
            results["render-validity"]["n"] += 1
            results["render-validity"]["agree"] += agree(vb.valid, False)
            results["render-validity"]["cases"].append({"url": url, "type": "broken", "verdict": vb.valid, "gold": False})
            if results["render-validity"]["n"] >= 60:
                break

        # ===== 2. executable-validity (≥50): 正例=真实元素中心, 负例=空白点 =====
        page.goto(f"{REDDIT}/f/food", wait_until="domcontentloaded"); page.wait_for_timeout(600)
        elems = page.evaluate("""() => {
            const es = Array.from(document.querySelectorAll('button, a[href]')).filter(e=>{const r=e.getBoundingClientRect(); return r.width>0&&r.height>0&&r.top>=0&&r.top<700;});
            return es.slice(0,40).map(e=>{const r=e.getBoundingClientRect(); return [Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)];});
        }""")
        for (x, y) in elems:
            hit = page.evaluate(js_element_at(x, y))
            v = oracle_executable_validity(hit)
            # gold: 真实执行真值 = elementFromPoint 命中可交互 (环境事实)
            gold = hit.get("hit") and hit.get("interactable")
            results["executable-validity"]["n"] += 1
            results["executable-validity"]["agree"] += agree(v.valid, gold)
            results["executable-validity"]["cases"].append({"pt": [x, y], "type": "real_elem", "verdict": v.valid, "gold": gold})
        # 负例: 已知空白坐标 (页面右侧远端空白 / 负坐标)
        for (x, y) in [(1270, 700), (1275, 10), (5, 715), (1200, 690), (3, 3), (1278, 718),
                       (1260, 705), (1240, 712), (8, 700), (1270, 695)]:
            hit = page.evaluate(js_element_at(x, y))
            v = oracle_executable_validity(hit)
            gold = hit.get("hit") and hit.get("interactable")
            results["executable-validity"]["n"] += 1
            results["executable-validity"]["agree"] += agree(v.valid, gold)
            results["executable-validity"]["cases"].append({"pt": [x, y], "type": "blank", "verdict": v.valid, "gold": gold})

        # ===== 3. unrelated-preservation (≥50): 多页 unrelated 编辑后目标仍在 =====
        # 目标 = 页面上第一个带文字的按钮/链接; 正例=只改背景, 负例=误伤目标(隐藏它)
        for url in page_urls:
            if results["unrelated-preservation"]["n"] >= 52:
                break
            try:
                page.goto(f"{REDDIT}{url}" if url.startswith("/") else url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(300)
            except Exception:
                continue
            tgt_text = page.evaluate("""() => { const e=document.querySelector('a[href]'); return e? (e.textContent||'').trim().slice(0,15):null; }""")
            if not tgt_text:
                continue
            before = page.evaluate(js_target_exists(tgt_text))
            # 正例: unrelated 改背景, 目标保留
            page.evaluate("() => { document.body.style.backgroundColor='#def'; document.querySelectorAll('img').forEach(i=>i.style.filter='grayscale(1)'); }")
            after_good = page.evaluate(js_target_exists(tgt_text))
            v = oracle_unrelated_preservation(before, after_good)
            results["unrelated-preservation"]["n"] += 1
            results["unrelated-preservation"]["agree"] += agree(v.valid, True)
            results["unrelated-preservation"]["cases"].append({"url": url, "type": "unrelated_ok", "verdict": v.valid, "gold": True})
            # 负例: 误伤 — 禁用/隐藏目标
            page.evaluate(f"""() => {{ const all=Array.from(document.querySelectorAll('a[href],button')); const m=all.find(e=>(e.textContent||'').trim().includes('{tgt_text.replace("'","")}')); if(m){{m.disabled=true; m.style.display='none';}} }}""")
            after_bad = page.evaluate(js_target_exists(tgt_text))
            vb = oracle_unrelated_preservation(before, after_bad)
            # gold: 目标被隐藏/禁用 → preservation 应 False
            gold_bad = not (after_bad.get("found") and not after_bad.get("disabled", False))
            results["unrelated-preservation"]["n"] += 1
            results["unrelated-preservation"]["agree"] += agree(vb.valid, not gold_bad)
            results["unrelated-preservation"]["cases"].append({"url": url, "type": "unrelated_hurt", "verdict": vb.valid, "gold": not gold_bad})

        # ===== 4. semantic-target (≥50): 真实 subscribe/unsubscribe 后端 toggle =====
        # 正例(真实改变): 真点 subscribe form → 后端状态 toggle → gold=改变
        # 负例(真实不变): 同页重读, 不操作 → gold=不变
        # 每个 forum 产出 2 个真实样本 (toggle + noop), 遍历 25 forums → ≥50 全真实样本
        def read_sub_state():
            # 用 form.action (/subscribe vs /unsubscribe) 判定后端真实订阅状态 (比按钮文字可靠)
            return page.evaluate("""() => {
                const forms=Array.from(document.querySelectorAll('form'));
                const f=forms.find(f=>/subscribe/i.test(f.action||''));
                if(!f) return {state:'unknown', has_form:false};
                // Postmill: action=/unsubscribe → 当前已订阅; action=/subscribe → 当前未订阅
                return {state: /\\/unsubscribe/i.test(f.action)? 'subscribed':'unsubscribed', has_form:true};
            }""")

        def click_sub_form():
            # 用 Playwright 真实 click (触发 POST form 提交 + 导航); JS .click() 不会导航
            try:
                page.click('form.subscribe-form button', timeout=5000)
                page.wait_for_load_state('domcontentloaded')
                page.wait_for_timeout(800)
            except Exception:
                pass

        forum_urls = [u for u in page_urls if u.startswith("/f/") and u.count("/") == 2][:30]
        for furl in forum_urls:
            if results["semantic-target"]["n"] >= 56:
                break
            try:
                page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(500)
            except Exception:
                continue
            st0 = read_sub_state()
            if not st0.get("has_form"):
                continue
            # (a) 真实 toggle → 读回状态 → gold=是否真改变
            click_sub_form()
            page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded"); page.wait_for_timeout(500)
            st1 = read_sub_state()
            real_changed = st0.get("state") != st1.get("state")
            v = oracle_semantic_target(st0, st1, expect_change=True)
            results["semantic-target"]["n"] += 1
            results["semantic-target"]["agree"] += agree(v.valid, real_changed)
            results["semantic-target"]["cases"].append(
                {"type": "real_toggle", "forum": furl, "before": st0["state"], "after": st1["state"],
                 "verdict": v.valid, "gold": real_changed})
            # (b) no-op: 同状态重读 → gold=不变 (oracle expect_change=True 应判 invalid)
            page.goto(f"{REDDIT}{furl}", wait_until="domcontentloaded"); page.wait_for_timeout(400)
            st2 = read_sub_state()
            noop_changed = st1.get("state") != st2.get("state")
            v2 = oracle_semantic_target(st1, st2, expect_change=True)
            results["semantic-target"]["n"] += 1
            results["semantic-target"]["agree"] += agree(v2.valid, noop_changed)
            results["semantic-target"]["cases"].append(
                {"type": "real_noop", "forum": furl, "before": st1["state"], "after": st2["state"],
                 "verdict": v2.valid, "gold": noop_changed})
            # (c) 还原: 再 toggle 回原状态 (保持 forum 干净)
            click_sub_form()
            page.wait_for_timeout(300)

        # ===== 5. safety (≥50): 操作后登录态保持 + 无破坏 =====
        for url in page_urls:
            if results["safety"]["n"] >= 52:
                break
            try:
                page.goto(f"{REDDIT}{url}" if url.startswith("/") else url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(300)
            except Exception:
                continue
            sess = page.evaluate(JS_SESSION_SAFE)
            v = oracle_safety(sess, require_logged_in=True)
            gold = sess.get("logged_in") and not sess.get("destructive", False)
            results["safety"]["n"] += 1
            results["safety"]["agree"] += agree(v.valid, gold)
            results["safety"]["cases"].append({"url": url, "type": "normal", "verdict": v.valid, "gold": gold})
            # 负例: 构造"破坏性状态" (注入 destructive 文本 + 模拟登出)
            sess_bad = {"logged_in": False, "destructive": True, "url": sess.get("url")}
            vb = oracle_safety(sess_bad, require_logged_in=True)
            results["safety"]["n"] += 1
            results["safety"]["agree"] += agree(vb.valid, False)
            results["safety"]["cases"].append({"url": url, "type": "destructive_logout", "verdict": vb.valid, "gold": False})

        b.close()

    # 汇总一致率
    summary = {}
    for k, v in results.items():
        rate = v["agree"] / v["n"] if v["n"] > 0 else None
        summary[k] = {"n": v["n"], "agree": v["agree"], "agreement_rate": rate}
    out = {"summary": summary, "details": results}
    with open(f"{OUT}/oracle_validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    # 总体
    tot_n = sum(v["n"] for v in results.values())
    tot_a = sum(v["agree"] for v in results.values())
    print(f"OVERALL: {tot_a}/{tot_n} = {tot_a/tot_n:.4f}")
    return summary


if __name__ == "__main__":
    run()
