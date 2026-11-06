from causalact_runtime import location
# 不训练, 只验证: 能否可控改 DOM + 截图证明 + 刷新能否还原 (替代 reset)
import os, base64, json
from playwright.sync_api import sync_playwright

REDDIT = os.environ.get('REDDIT','http://127.0.0.1:9999')
OUT = location('CAUSALACT_RUN_DIR', 'web/intervention_probe')
os.makedirs(OUT, exist_ok=True)
results = {}

def shot(page, name):
    p = f'{OUT}/{name}.png'
    page.screenshot(path=p, full_page=False)
    return p

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    ctx = b.new_context(storage_state=location('CAUSALACT_VWA_AUTH', ''), viewport={'width':1280,'height':720})
    page = ctx.new_page()
    page.goto(f'{REDDIT}/f/food', wait_until='domcontentloaded')
    page.wait_for_timeout(1500)

    # baseline: count interactable buttons/links
    base_counts = page.evaluate('''() => {
        const btns = document.querySelectorAll('button, a[href], input[type="submit"]');
        return {buttons: document.querySelectorAll('button').length, links: document.querySelectorAll('a[href]').length, total_interactable: btns.length};
    }''')
    results['baseline'] = base_counts
    shot(page, 'baseline')

    # === TARGET: disable the first submit/vote button (改任务相关语义) ===
    target = page.evaluate('''() => {
        // find a vote/submit/action button
        const candidates = Array.from(document.querySelectorAll('button, input[type="submit"], a.btn'));
        if(candidates.length===0) return {ok:false, reason:'no button'};
        const el = candidates[0];
        const before = {disabled: el.disabled, text: (el.textContent||el.value||'').slice(0,40), tag: el.tagName};
        el.disabled = true;
        el.setAttribute('aria-disabled','true');
        el.style.opacity='0.4';
        const after = {disabled: el.disabled, aria: el.getAttribute('aria-disabled')};
        return {ok:true, before, after};
    }''')
    results['target_disable'] = target
    shot(page, 'target_disabled')

    # === UNRELATED: change page background + decorative images (无关视觉) ===
    unrelated = page.evaluate('''() => {
        document.body.style.backgroundColor = '#ffcc00';
        const imgs = document.querySelectorAll('img');
        let changed=0;
        imgs.forEach(im => { im.style.filter='hue-rotate(180deg)'; changed++; });
        return {ok:true, bg:'#ffcc00', imgs_recolored: changed};
    }''')
    results['unrelated_bg_img'] = unrelated
    shot(page, 'unrelated_bg')

    # === TRANSFORM: move/translate a target element (位置变, 语义不变) ===
    transform = page.evaluate('''() => {
        const el = document.querySelector('button, a.btn, a[href]');
        if(!el) return {ok:false};
        const r1 = el.getBoundingClientRect();
        el.style.position='relative'; el.style.left='120px'; el.style.top='40px';
        const r2 = el.getBoundingClientRect();
        return {ok:true, before_xy:[Math.round(r1.x),Math.round(r1.y)], after_xy:[Math.round(r2.x),Math.round(r2.y)], dx:Math.round(r2.x-r1.x), dy:Math.round(r2.y-r1.y)};
    }''')
    results['transform_move'] = transform
    shot(page, 'transform_moved')

    # === RESET test: reload page, confirm DOM patches gone (前端干预可逆) ===
    page.reload(wait_until='domcontentloaded')
    page.wait_for_timeout(1500)
    after_reload = page.evaluate('''() => {
        const el = document.querySelector('button, input[type="submit"], a.btn');
        return {bg: document.body.style.backgroundColor||'(none)', first_btn_disabled: el? el.disabled : null};
    }''')
    results['after_reload'] = after_reload
    shot(page, 'after_reload')

    b.close()

open(f'{OUT}/results.json','w').write(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
