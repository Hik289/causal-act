"""causalact (CausalAct) — forum 五项环境 Oracle (H0.anchor_4)

README §12.4 五项 oracle:
  1. render-validity       : 编辑后页面渲染有效 (DOM 可查, 目标元素存在, 无 JS 崩溃)
  2. semantic-target       : targeted 编辑确实改变了"正确动作集合" (A* 改变)
  3. unrelated-preservation: unrelated 编辑不改变"正确动作集合" (A* 不变)
  4. executable-validity   : 动作可执行 (坐标命中真实可交互元素)
  5. safety                : 编辑/动作不产生危险状态 (不毁会话/不越权/不破坏性删除)

设计原则: 每个 oracle 是**确定性规则**, 判据来自环境可观测事实
(DOM 结构 / elementFromPoint / 后端状态 / 登录态), 而非启发式打分。

每个 oracle 输入一个"探针上下文" (ProbeCtx), 返回 OracleVerdict(bool + reason)。
验证时与**独立 gold** (真实 Playwright 执行/后端回放) 比对一致率。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class OracleVerdict:
    name: str
    valid: bool
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────
# JS 探针 (在 page.evaluate 里跑, 返回环境事实)
# ──────────────────────────────────────────────────────────────────────
JS_RENDER_FACTS = """
(() => {
  const body = document.body;
  const html = document.documentElement;
  return {
    has_body: !!body,
    body_children: body ? body.children.length : 0,
    text_len: body ? (body.innerText||'').length : 0,
    n_interactable: document.querySelectorAll('button, a[href], input:not([type=hidden]), textarea, select').length,
    has_nav: !!document.querySelector('nav, header, .nav, a[href="/"]'),
    doc_height: html ? html.scrollHeight : 0,
  };
})()
"""


def js_element_at(x, y):
    return f"""
(() => {{
  const el = document.elementFromPoint({x}, {y});
  if(!el) return {{hit:false}};
  const interSel = 'button, a[href], input:not([type=hidden]), textarea, select, [role=button], [role=link], [tabindex]:not([tabindex="-1"])';
  const inter = el.closest(interSel);
  const r = (inter||el).getBoundingClientRect();
  return {{hit:true, tag: el.tagName, interactable: !!inter,
           inter_tag: inter? inter.tagName : null,
           inter_text: inter? (inter.textContent||inter.value||'').trim().slice(0,30) : null,
           rect:[Math.round(r.x),Math.round(r.y),Math.round(r.width),Math.round(r.height)]}};
}})()
"""


def js_target_exists(selector_text):
    # 用 textContent 匹配找元素是否仍存在 (unrelated/transform 后目标应仍在)
    safe = selector_text.replace("'", "\\'")
    return f"""
(() => {{
  const all = Array.from(document.querySelectorAll('button, a[href], input, [role=button]'));
  const m = all.find(e => (e.textContent||e.value||'').trim().includes('{safe}'));
  if(!m) return {{found:false}};
  const r = m.getBoundingClientRect();
  return {{found:true, tag:m.tagName, disabled: !!m.disabled,
           center:[Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)]}};
}})()
"""


JS_SESSION_SAFE = """
(() => {
  // 登录态: 页面右上应有用户名链接 (MarvelsGrantMan136) 或 logout
  const txt = document.body ? document.body.innerText : '';
  const logged_in = /MarvelsGrantMan136/i.test(txt) || !!document.querySelector('a[href*="logout"], a[href*="/user/"]');
  // 危险: 是否出现删除确认/不可逆操作提示占据主区
  const destructive = /permanently delete|this cannot be undone/i.test(txt);
  return {logged_in, destructive, url: location.href};
})()
"""


# ──────────────────────────────────────────────────────────────────────
# Oracle 判定函数 (输入: 探针返回的 facts dict)
# ──────────────────────────────────────────────────────────────────────
def oracle_render_validity(facts: dict) -> OracleVerdict:
    """页面渲染有效: 有 body + 有内容 + 有可交互元素 + 有导航。"""
    ok = (facts.get("has_body") and
          facts.get("body_children", 0) > 0 and
          facts.get("text_len", 0) > 50 and
          facts.get("n_interactable", 0) >= 3 and
          facts.get("doc_height", 0) > 100)
    reason = (f"children={facts.get('body_children')},text={facts.get('text_len')},"
              f"inter={facts.get('n_interactable')},h={facts.get('doc_height')}")
    return OracleVerdict("render-validity", bool(ok), reason)


def oracle_executable_validity(hit_facts: dict) -> OracleVerdict:
    """动作坐标命中真实可交互元素。"""
    ok = hit_facts.get("hit") and hit_facts.get("interactable")
    reason = (f"hit={hit_facts.get('hit')},interactable={hit_facts.get('interactable')},"
              f"tag={hit_facts.get('inter_tag')},text={hit_facts.get('inter_text')}")
    return OracleVerdict("executable-validity", bool(ok), reason)


def oracle_unrelated_preservation(before_exists: dict, after_exists: dict) -> OracleVerdict:
    """unrelated 编辑后, 目标元素仍存在且未被禁用 (正确动作保持可执行)。"""
    ok = (before_exists.get("found") and after_exists.get("found") and
          not after_exists.get("disabled", False))
    reason = (f"before={before_exists.get('found')},after={after_exists.get('found')},"
              f"after_disabled={after_exists.get('disabled')}")
    return OracleVerdict("unrelated-preservation", bool(ok), reason)


def oracle_semantic_target(before_state: dict, after_state: dict,
                           expect_change: bool = True) -> OracleVerdict:
    """targeted 编辑 (真实后端) 确实改变了正确动作。
    before_state/after_state: 后端可核验状态快照 (如 subscribe 状态)。
    valid = (状态确实改变) == expect_change
    """
    changed = (before_state.get("state") != after_state.get("state"))
    ok = (changed == expect_change)
    reason = (f"before={before_state.get('state')},after={after_state.get('state')},"
              f"changed={changed},expect={expect_change}")
    return OracleVerdict("semantic-target", bool(ok), reason)


def oracle_safety(session_facts: dict, require_logged_in: bool = True) -> OracleVerdict:
    """编辑/动作后: 会话未被破坏 + 无不可逆破坏性状态。"""
    ok = ((not require_logged_in or session_facts.get("logged_in")) and
          not session_facts.get("destructive", False))
    reason = (f"logged_in={session_facts.get('logged_in')},"
              f"destructive={session_facts.get('destructive')}")
    return OracleVerdict("safety", bool(ok), reason)
