from causalact_runtime import location
import sys
from playwright.sync_api import sync_playwright
import os
R = os.environ.get('REDDIT', 'http://127.0.0.1:9999')
JS = r'''(() => {
  const links=Array.from(document.querySelectorAll("a[href]"));
  let n=0;
  for(const a of links){
    const t=(a.textContent||"").trim();
    if(!/comment/i.test(t)) continue;
    const h=a.getAttribute("href")||"";
    if(!/\/f\/[^/]+\/\d+/.test(h)) continue;
    const r=a.getBoundingClientRect();
    if(r.width<2||r.top<0||r.top>720) continue;
    const d=document.createElement("div");
    d.setAttribute("style","position:fixed;left:"+r.x+"px;top:"+r.y+"px;width:"+r.width+"px;height:"+r.height+"px;border:3px solid red;z-index:2147483647;pointer-events:none;");
    document.body.appendChild(d); n++;
  }
  return n;
})()'''
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    ctx = b.new_context(storage_state=location('CAUSALACT_VWA_AUTH', ''), viewport={'width': 1280, 'height': 720})
    p = ctx.new_page()
    p.goto(f'{R}/f/AskReddit', wait_until='domcontentloaded'); p.wait_for_timeout(800)
    n = p.evaluate(JS); p.wait_for_timeout(300)
    print('boxes drawn:', n)
    p.screenshot(path=location('CAUSALACT_RUN_DIR', 'scratch/nav_check2.png'))
    b.close()
