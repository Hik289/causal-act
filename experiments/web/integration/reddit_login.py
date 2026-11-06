import os
from playwright.sync_api import sync_playwright
from browser_env.env_config import REDDIT, ACCOUNTS

auth_folder='./.auth'
os.makedirs(auth_folder, exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    ctx=browser.new_context()
    page=ctx.new_page()
    page.goto(f'{REDDIT}/login')
    page.get_by_label('Username').fill(ACCOUNTS['reddit']['username'])
    page.get_by_label('Password').fill(ACCOUNTS['reddit']['password'])
    page.get_by_role('button', name='Log in').click()
    page.wait_for_timeout(3000)
    # verify logged in
    page.goto(f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account")
    page.wait_for_timeout(2000)
    body=page.content()
    ok='Delete' in body or 'Preferences' in body or ACCOUNTS['reddit']['username'] in body
    ctx.storage_state(path=f'{auth_folder}/reddit_state.json')
    print('LOGIN_OK' if ok else 'LOGIN_UNCERTAIN', 'cookie_saved=reddit_state.json')
    browser.close()
