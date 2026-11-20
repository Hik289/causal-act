"""Adapt the pinned UI-TARS prompt/parser to an owned Forum browser."""
import base64
import copy
import hashlib
import importlib.util
import json
import time
import urllib.request
from pathlib import Path

from seeact_adapter import ModelUnavailable, load_task_references, normalize_image_mime, verify_reference_delivery
from uitars_contract import file_sha256, parsed_action, screen_box

MODEL = 'ui-tars-1.5-7b'


def image_part(raw):
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(raw).decode(), 'detail': 'high'}}


def planning_messages(prompt, references, screenshots, responses):
    """Keep every reference and the last three screens; current screen is last."""
    if len(screenshots) != len(responses) + 1:
        raise ValueError('Each completed response requires its preceding screen')
    first = max(0, len(screenshots) - 3)
    messages = []
    for index in range(first, len(screenshots)):
        content = []
        if index == first:
            content += copy.deepcopy(references)
            content.append({'type': 'text', 'text': prompt + (
                '\nThe first ' + str(len(references)) + ' image(s) are task reference inputs. '
                'The final image in this message is the current webpage at this history step.' if references else '')})
        content.append(image_part(screenshots[index]))
        messages.append({'role': 'user', 'content': content})
        if index < len(responses):
            messages.append({'role': 'assistant', 'content': responses[index]})
    return messages


class UITarsClient:
    def __init__(self, endpoint, audit_file, max_tokens=512):
        self.endpoint, self.audit_file, self.max_tokens = endpoint, Path(audit_file), max_tokens
        self.calls = []

    def complete(self, messages):
        messages, images = normalize_image_mime(messages)
        body = json.dumps(dict(model=MODEL, messages=messages, max_tokens=self.max_tokens)).encode()
        record = dict(stage='planning', model=MODEL, started_unix=time.time(),
                      images=images, request_sha256=hashlib.sha256(body).hexdigest(), request_bytes=len(body),
                      decoding='deployed service do_sample=False', max_tokens=self.max_tokens)
        try:
            request = urllib.request.Request(self.endpoint, data=body, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                record['status'] = response.status
                result = json.load(response)
            record['served_model'] = result.get('model')
            if result.get('model') != MODEL:
                raise ModelUnavailable('Unexpected UI-TARS served model')
            text = result['choices'][0]['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ModelUnavailable('Empty UI-TARS response')
            record.update(response=text, usage=result.get('usage'), finish_reason=result['choices'][0].get('finish_reason'))
            return text
        except Exception as error:
            record['error'] = type(error).__name__
            raise ModelUnavailable('UI-TARS transport or response failure') from error
        finally:
            record['ended_unix'] = time.time()
            self.calls.append(record)
            with self.audit_file.open('a') as stream:
                stream.write(json.dumps(record) + '\n')


KEYS = {'ctrl': 'Control', 'control': 'Control', 'shift': 'Shift', 'alt': 'Alt',
        'win': 'Meta', 'command': 'Meta', 'super': 'Meta', 'enter': 'Enter', 'return': 'Enter',
        'esc': 'Escape', 'escape': 'Escape', 'space': 'Space', ' ': 'Space', 'tab': 'Tab',
        'backspace': 'Backspace', 'delete': 'Delete', 'left': 'ArrowLeft', 'right': 'ArrowRight',
        'up': 'ArrowUp', 'down': 'ArrowDown', 'arrowleft': 'ArrowLeft', 'arrowright': 'ArrowRight',
        'arrowup': 'ArrowUp', 'arrowdown': 'ArrowDown', 'home': 'Home', 'end': 'End',
        'pageup': 'PageUp', 'pagedown': 'PageDown'}


class BrowserActions:
    """Explicit page equivalents; never execute generated Python or clamp points."""
    def __init__(self, page, screen_size):
        self.page, self.screen_size = page, screen_size
        self.address = None

    def apply(self, action):
        kind, data = action['action_type'], action['action_inputs']
        detail = dict(action_type=kind, action_inputs=data)
        point = screen_box(data['start_box'], self.screen_size) if data.get('start_box') else None
        if point is not None:
            detail['screen_point'] = point
        if kind in ('click', 'left_single', 'left_double', 'right_single', 'hover'):
            if point is None:
                raise ValueError('Native pointer action has no point')
            self.address = None
            if kind == 'hover':
                self.page.mouse.move(*point)
            else:
                self.page.mouse.click(*point, button='right' if kind == 'right_single' else 'left',
                                      click_count=2 if kind == 'left_double' else 1)
        elif kind in ('drag', 'select'):
            if point is None or not data.get('end_box'):
                raise ValueError('Native drag has no endpoint')
            end = screen_box(data['end_box'], self.screen_size)
            detail['screen_end'] = end
            self.page.mouse.move(*point)
            self.page.mouse.down()
            try:
                for index in range(1, 21):
                    self.page.mouse.move(point[0] + (end[0] - point[0]) * index / 20,
                                         point[1] + (end[1] - point[1]) * index / 20)
                    self.page.wait_for_timeout(50)
            finally:
                self.page.mouse.up()
        elif kind == 'type':
            content = data.get('content', '')
            submit = content.endswith('\n') or content.endswith('\\n')
            text = content.rstrip('\\n').rstrip('\n') if submit else content
            if self.address is not None:
                self.address = text
                if submit:
                    self.navigate_address()
            else:
                if text:
                    self.page.evaluate('(text) => navigator.clipboard.writeText(text)', text)
                    self.page.keyboard.press('Control+v')
                    self.page.wait_for_timeout(500)
                if submit:
                    self.page.keyboard.press('Enter')
        elif kind in ('hotkey', 'press', 'keydown', 'release', 'keyup'):
            raw = data.get('key', data.get('hotkey', data.get('press', '')))
            keys = [KEYS.get(x.lower(), x) for x in raw.split()]
            combo = '+'.join(keys)
            if not keys:
                raise ValueError('Empty native key action')
            if kind == 'hotkey' and combo in ('Control+l', 'Meta+l'):
                self.address = ''
            elif self.address is not None and combo == 'Enter':
                self.navigate_address()
            elif self.address is not None and combo in ('Control+a', 'Meta+a'):
                self.address = ''
            elif self.address is not None and combo == 'Escape':
                self.address = None
            elif kind in ('press', 'keydown'):
                self.page.keyboard.down(combo)
            elif kind in ('release', 'keyup'):
                self.page.keyboard.up(combo)
            else:
                self.page.keyboard.press(combo)
        elif kind == 'scroll':
            if point is not None:
                self.page.mouse.move(*point)
            direction = data.get('direction', '').lower()
            if 'up' in direction or 'down' in direction:
                # Native converter emits +/-5 wheel ticks. Freeze 120px/tick
                # as the browser adaptation, before any benchmark scores.
                delta = -600 if 'up' in direction else 600
                self.page.mouse.wheel(0, delta)
                detail['wheel_delta_pixels'] = delta
            else:
                detail['native_noop'] = 'Pinned converter emits no horizontal wheel event'
        elif kind == 'wait':
            self.page.wait_for_timeout(5000)
        elif kind == 'finished':
            return dict(detail, terminal=True, answer=data.get('content', ''))
        else:
            detail['native_noop'] = 'Unrecognized by the pinned converter'
        return dict(detail, terminal=False)

    def navigate_address(self):
        target, self.address = self.address, None
        if not target.startswith(('http://', 'https://')):
            raise ValueError('Browser address is not an absolute HTTP URL')
        self.page.goto(target)


def native_prompt(contract, instruction):
    path = Path(contract['native_source']) / 'codes/ui_tars/prompt.py'
    if file_sha256(path) != contract['native_files_sha256']['codes/ui_tars/prompt.py']:
        raise RuntimeError('Pinned UI-TARS prompt changed')
    spec = importlib.util.spec_from_file_location('pinned_uitars_prompt', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COMPUTER_USE_DOUBAO.format(language='English', instruction=instruction)


def run_uitars_case(checkout, task_file, output, bridge, *, endpoint, frame_contract,
                    original_origin, environment_origin, max_steps=10,
                    evaluation_navigation_timeout_ms=120000, client_factory=UITarsClient,
                    evaluate=True, fixture=False):
    from playwright.sync_api import sync_playwright
    from forum_origin_guard import guard_sync_contexts
    from seeact_adapter import evaluate_live_page
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads(Path(frame_contract).read_text())
    if contract['screenshot_size'] != [1280, 720]:
        raise RuntimeError('Unexpected frozen UI-TARS screenshot geometry')
    task_file = Path(task_file)
    task = json.loads(task_file.read_text())
    references, reference_audit = load_task_references(task, task_file, output, base_dir=checkout)
    client = client_factory(endpoint, output / 'uitars_requests.jsonl', max_tokens=512)
    prompt = native_prompt(contract, task['intent'])
    screenshots, responses, actions, redirects = [], [], [], []
    answer, stopped, score = '', 'decision budget exhausted', None
    with guard_sync_contexts(original_origin, environment_origin, redirects), sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=['--disable-gpu'])
        context = browser.new_context(viewport={'width': 1280, 'height': 720},
            storage_state=task.get('storage_state'), geolocation=task.get('geolocation'),
            permissions=['clipboard-read', 'clipboard-write'])
        page = context.new_page()
        try:
            page.goto(task['start_url'])
            executor = BrowserActions(page, contract['screenshot_size'])
            for step in range(max_steps):
                raw = page.screenshot()
                (output / ('screen_' + str(step) + '.png')).write_bytes(raw)
                screenshots.append(raw)
                response = client.complete(planning_messages(prompt, references, screenshots, responses))
                responses.append(response)
                try:
                    chosen = parsed_action(response, contract)
                except (AssertionError, ValueError, TypeError, KeyError, SyntaxError) as error:
                    stopped = 'native parser rejected model action: ' + type(error).__name__
                    break
                for action in chosen:
                    try:
                        effect = executor.apply(action)
                    except ValueError as error:
                        stopped = 'native action arguments rejected: ' + str(error)
                        effect = dict(terminal=True, answer='', argument_error=str(error))
                    actions.append(dict(step=step + 1, **effect))
                    if effect.get('terminal'):
                        answer = effect.get('answer', '')
                        stopped = 'native finished' if action['action_type'] == 'finished' else stopped
                        break
                if actions and actions[-1].get('terminal'):
                    break
                page.wait_for_timeout(2500)
                live_pages = [candidate for candidate in context.pages if not candidate.is_closed()]
                if not live_pages:
                    raise RuntimeError('Native agent closed every browser page')
                page = live_pages[-1]
                executor.page = page
            verify_reference_delivery(client.calls, [x['sha256'] for x in normalize_image_mime([{'role':'user','content':references}])[1]])
            final_url = page.url
            page.screenshot(path=str(output / 'final.png'))
            (output / 'final.html').write_text(page.content())
            if evaluate:
                bridge.stage = 'evaluation'
                score = evaluate_live_page(checkout, task_file, page, answer,
                    {'text': page.inner_text('body')}, bridge,
                    evaluation_navigation_timeout_ms, output / 'evaluation_navigation.jsonl')
                if bridge.calls:
                    bridge.validate_completed_calls()
            return dict(task_id=task['task_id'], valid=evaluate, score=score,
                fixture=fixture or not evaluate, model=MODEL, evaluation_model='anthropic.claude-opus-5-5',
                steps=len(responses), model_calls=len(client.calls), evaluation_model_calls=len(bridge.calls),
                max_steps=max_steps, final_url_before_evaluation=final_url, answer=answer,
                actions=actions, stop_reason=stopped, reference_inputs=reference_audit,
                origin_redirects=redirects, origin_guard_errors=[x for x in redirects if x.get('event')=='guard_error'],
                origin_guard_contexts=sum(x.get('event')=='context_guard_installed' for x in redirects),
                frame_contract_sha256=file_sha256(frame_contract),
                evaluation_navigation_timeout_ms=evaluation_navigation_timeout_ms,
                official_evaluator_sha256=file_sha256(Path(checkout)/'evaluation_harness/evaluators.py') if evaluate else None)
        finally:
            (output / 'actions.json').write_text(json.dumps(actions, indent=2)+'\n')
            browser.close()
