"""Run the pinned upstream Selenium WebVoyager loop on an owned VWA case.

Prompts, set-of-mark extraction, parser, context clipping and actions execute
from upstream run.py. Only startup/authentication, model transport, reference
inputs and the official VWA evaluator are adapted. No upstream source is edited.
"""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import types

from seeact_adapter import MODEL, ModelUnavailable, free_loopback_port, load_task_references, official_score


class BrowserGuard:
    """Service CDP routing on another thread while Selenium blocks its caller."""
    def __init__(self, port, original, current, storage_state, audit):
        self.port, self.original, self.current = port, original, current
        self.storage_state, self.audit = storage_state, audit
        self.ready, self.stop_event = threading.Event(), threading.Event()
        self.failure = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(30):
            raise RuntimeError('CDP environment guard did not become ready')
        self.check()

    def check(self):
        if self.failure:
            raise RuntimeError('CDP environment guard failed') from self.failure
        if any(row.get('event') == 'guard_error' for row in self.audit):
            raise RuntimeError('Owned environment routing failed')

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('CDP environment guard did not stop')
        self.check()

    def run(self):
        async def serve():
            from playwright.async_api import async_playwright
            from forum_origin_guard import guard_async_context
            async with async_playwright() as playwright:
                browser = await playwright.chromium.connect_over_cdp('http://127.0.0.1:' + str(self.port))
                context = browser.contexts[0]
                await guard_async_context(context, self.original, self.current, self.audit)
                if self.storage_state:
                    state = json.loads(Path(self.storage_state).read_text())
                    await context.add_cookies(state.get('cookies', []))
                    for origin in state.get('origins', []):
                        script = '''if (location.origin === ORIGIN) {
                          for (const item of ITEMS) localStorage.setItem(item.name, item.value);
                        }'''.replace('ORIGIN', json.dumps(origin['origin'])).replace('ITEMS', json.dumps(origin.get('localStorage', [])))
                        await context.add_init_script(script)
                self.ready.set()
                while not self.stop_event.is_set():
                    await asyncio.sleep(.05)
                # Closing this CDP connection would close Selenium's browser.
                # Exiting the Playwright client disconnects; Selenium owns quit.
        try:
            asyncio.run(serve())
        except BaseException as error:
            self.failure = error
            self.ready.set()


def native_actions(messages, parser):
    actions, answer = [], ''
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        text = message.get('content') or ''
        row = {'step': len(actions) + 1, 'response': text}
        try:
            chosen = re.split(r'Thought:|Action:|Observation:', text)[2].strip()
            key, info = parser(chosen)
            row.update(action=key, arguments=info)
            if key == 'answer':
                answer = str(info['content'])
        except (IndexError, KeyError, TypeError, ValueError, AssertionError):
            row['action'] = 'native format error'
        actions.append(row)
    return actions, answer


def run_webvoyager_case(source, checkout, task_file, output, bridge, *,
                       browser_binary, chromedriver, original_origin,
                       environment_origin, max_steps=10,
                       evaluation_navigation_timeout_ms=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task_file = Path(task_file).resolve()
    task = json.loads(task_file.read_text())
    bridge.reference_inputs, reference_audit = load_task_references(task, task_file, output, base_dir=checkout)
    source = Path(source).resolve()
    for name in ['run.py', 'utils.py', 'utils_webarena.py', 'prompts.py']:
        if not (source / name).is_file():
            raise RuntimeError('Missing native WebVoyager source: ' + name)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('pinned_webvoyager_run', source / 'run.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    from openai.types.chat import ChatCompletion
    def create(**kwargs):
        return ChatCompletion(**bridge.native_completion(**kwargs).raw_response)
    native.OpenAI = lambda **_: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    unsupported_pdf = []
    def reject_pdf(*_, **__):
        unsupported_pdf.append(True)
        raise ModelUnavailable('Native Assistant PDF API has no declared Claude equivalent')
    native.get_pdf_retrieval_ans_from_assistant = reject_pdf
    original_driver = native.webdriver.Chrome
    original_print = native.print_message
    captured = {'messages': None, 'result': None}
    guards, drivers, redirect_audit = [], [], []
    port = free_loopback_port()

    def print_messages(messages, save_dir=None):
        captured['messages'] = copy.deepcopy(messages)
        original_print(messages, save_dir)
    native.print_message = print_messages

    def factory(*args, **kwargs):
        from selenium.webdriver.chrome.service import Service
        options = kwargs['options']
        options.binary_location = str(browser_binary)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--remote-debugging-port=' + str(port))
        driver = original_driver(*args, service=Service(str(chromedriver), log_output=str(output / 'chromedriver.log')), **kwargs)
        drivers.append(driver)
        guard = BrowserGuard(port, original_origin, environment_origin, task.get('storage_state'), redirect_audit)
        guards.append(guard)
        guard.start()
        original_size, original_quit = driver.set_window_size, driver.quit
        def set_size(width, height, *a, **kw):
            result = original_size(width, height, *a, **kw)
            driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False})
            return result
        driver.set_window_size = set_size
        if task.get('geolocation'):
            driver.execute_cdp_cmd('Emulation.setGeolocationOverride', task['geolocation'])

        def evaluate_and_quit():
            try:
                guard.check()
                bridge.validate_completed_calls()
                if unsupported_pdf:
                    raise ModelUnavailable('Unsupported native PDF inference; not a task zero')
                logs = list((output / 'native').glob('*/task*/agent.log'))
                if len(logs) != 1:
                    raise RuntimeError('Missing or ambiguous native task log')
                log = logs[0].read_text()
                if 'Driver error when adding set-of-mark.' in log or 'Driver error when obtaining accessibility tree.' in log:
                    raise RuntimeError('Native observation extraction failed; not a task zero')
                if captured['messages'] is None:
                    raise RuntimeError('Native loop did not save its trajectory')
                actions, answer = native_actions(captured['messages'], native.extract_information)
                driver.save_screenshot(str(output / 'final.png'))
                (output / 'final.html').write_text(driver.page_source)
                observation = {'text': driver.find_element('tag name', 'body').text}
                target = driver.execute_cdp_cmd('Target.getTargetInfo', {})['targetInfo']['targetId']
                final_url = driver.current_url
                dimensions = driver.execute_script('return [innerWidth, innerHeight, devicePixelRatio]')
                if dimensions != [1280, 720, 1]:
                    raise RuntimeError('Browser viewport differs from frozen paired protocol')
                bridge.stage = 'evaluation'
                try:
                    score = official_score(checkout, task_file, port, target, answer, observation, bridge,
                                           evaluation_navigation_timeout_ms, output / 'evaluation_navigation.jsonl')
                finally:
                    bridge.stage = 'planning'
                bridge.validate_completed_calls()
                guard.check()
                captured['result'] = dict(task_id=task['task_id'], score=score, valid=True,
                    steps=len(actions), final_url_before_evaluation=final_url, answer=answer,
                    actions=actions, model=MODEL, model_calls=len(bridge.calls), max_steps=max_steps,
                    method='native Selenium WebVoyager with Claude transport',
                    reference_inputs=reference_audit, viewport=dimensions,
                    origin_redirects=[x for x in redirect_audit if 'mode' in x],
                    origin_guard_errors=[x for x in redirect_audit if x.get('event') == 'guard_error'],
                    origin_guard_contexts=sum(x.get('event') == 'context_guard_installed' for x in redirect_audit),
                    task_sha256=hashlib.sha256(task_file.read_bytes()).hexdigest(),
                    official_evaluator_sha256=hashlib.sha256((Path(checkout) / 'evaluation_harness/evaluators.py').read_bytes()).hexdigest(),
                    native_files_sha256={n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in ['run.py', 'utils.py', 'utils_webarena.py', 'prompts.py']})
            finally:
                try:
                    guard.close()
                finally:
                    original_quit()
                    drivers.remove(driver)
        driver.quit = evaluate_and_quit
        return driver

    native.webdriver.Chrome = factory
    downloads = output / 'downloads'
    downloads.mkdir()
    task_list = output / 'native_tasks.jsonl'
    task_list.write_text(json.dumps({'id': task['task_id'], 'ques': task['intent'], 'web': task['start_url']}) + '\n')
    old_argv = sys.argv
    try:
        sys.argv = [str(source / 'run.py'), '--test_file', str(task_list), '--output_dir', str(output / 'native'),
            '--download_dir', str(downloads), '--api_key', 'local-audited-transport', '--api_model', MODEL,
            '--headless', '--force_device_scale', '--fix_box_color', '--seed', '42',
            '--max_iter', str(max_steps), '--max_attached_imgs', '3', '--window_width', '1280', '--window_height', '720']
        native.main()
        if captured['result'] is None:
            raise RuntimeError('No completed official evaluation')
        return captured['result']
    finally:
        sys.argv = old_argv
        native.webdriver.Chrome = original_driver
        try:
            for guard in guards:
                if guard.thread.is_alive():
                    guard.close()
        finally:
            for driver in drivers:
                original_driver.quit(driver)
