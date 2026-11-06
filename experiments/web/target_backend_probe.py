from causalact_runtime import location
# 验证 TARGET 干预的核心难点: 能否让'环境可核验的正确动作集合'真的改变?
# 以 Postmill 订阅状态为例: 订阅前正确动作='Subscribe', 订阅后='Unsubscribe'
# 测两条路: (A)纯DOM伪装(改按钮文字) (B)真实后端状态改变(点击订阅->后端记录)
import os, json
from playwright.sync_api import sync_playwright
REDDIT=os.environ.get('REDDIT','http://127.0.0.1:9999')
OUT=location('CAUSALACT_RUN_DIR', 'web/intervention_probe')
res={}
with sync_playwright() as pw:
    b=pw.chromium.launch(headless=True)
    ctx=b.new_context(storage_state=location('CAUSALACT_VWA_AUTH', ''), viewport={'width':1280,'height':720})
    page=ctx.new_page()
    page.goto(f'{REDDIT}/f/food', wait_until='domcontentloaded'); page.wait_for_timeout(1200)
    # baseline subscribe button state (backend truth)
    st0=page.evaluate('''() => {
        const btns=Array.from(document.querySelectorAll('button'));
        const sub=btns.find(x=>/subscribe/i.test(x.textContent));
        return sub? {found:true, text:sub.textContent.trim()} : {found:false};
    }''')
    res['subscribe_baseline']=st0
    # (A) DOM-only: flip the button text -> looks subscribed but backend unchanged
    domA=page.evaluate('''() => {
        const btns=Array.from(document.querySelectorAll('button'));
        const sub=btns.find(x=>/subscribe/i.test(x.textContent));
        if(!sub) return {ok:false};
        const before=sub.textContent.trim();
        sub.textContent='Unsubscribe';
        return {ok:true, before, after:sub.textContent.trim(), note:'DOM-only, backend NOT changed'};
    }''')
    res['target_dom_only']=domA
    # (B) real backend change: actually click subscribe, then reload to read true state
    page.goto(f'{REDDIT}/f/food', wait_until='domcontentloaded'); page.wait_for_timeout(1000)
    clicked=page.evaluate('''() => {
        const btns=Array.from(document.querySelectorAll('button'));
        const sub=btns.find(x=>/^subscribe$/i.test(x.textContent.trim()));
        if(!sub) return {ok:false, reason:'no exact Subscribe btn'};
        return {ok:true, willClick:sub.textContent.trim()};
    }''')
    res['target_backend_precheck']=clicked
    try:
        # click the real subscribe button via Playwright (triggers backend POST)
        page.get_by_role('button', name='Subscribe', exact=True).first.click(timeout=4000)
        page.wait_for_timeout(1500)
        page.reload(wait_until='domcontentloaded'); page.wait_for_timeout(1200)
        st1=page.evaluate('''() => {
            const btns=Array.from(document.querySelectorAll('button'));
            const un=btns.find(x=>/unsubscribe/i.test(x.textContent));
            return {now_unsubscribe_visible: !!un, text: un? un.textContent.trim():'(subscribe still)'};
        }''')
        res['target_backend_after_click']=st1
        # revert: unsubscribe to restore clean state
        try:
            page.get_by_role('button', name='Unsubscribe', exact=True).first.click(timeout=4000)
            page.wait_for_timeout(1000)
            res['revert']='unsubscribed_back'
        except Exception as e:
            res['revert']='revert_failed: '+str(e)[:80]
    except Exception as e:
        res['target_backend_after_click']={'error':str(e)[:120]}
    b.close()
open(f'{OUT}/target_backend_results.json','w').write(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
