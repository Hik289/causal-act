"""Native SeeAct loop with a paced Claude transport and official VWA scoring.

The upstream planner, referring prompt, grounding parser, and actions stay
unchanged. This is a Claude adaptation, not the upstream GPT-4 configuration.
"""
import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import types
import urllib.error
import urllib.request


MODEL = 'anthropic.claude-opus-5-5'


class ModelUnavailable(RuntimeError):
    """Invalid inference, never a scored task failure."""


def validate_task(task):
    if task.get('image'):
        raise ValueError('Reference-image task input is not implemented; keep this case pending, not scored')


def normalize_image_mime(messages):
    messages = copy.deepcopy(messages)
    images = []
    for message in messages:
        for item in message.get('content', []) if isinstance(message.get('content'), list) else []:
            if item.get('type') != 'image_url':
                continue
            url = item['image_url']['url']
            if not url.startswith('data:') or ';base64,' not in url:
                raise ModelUnavailable('Only inline screenshot bytes are supported')
            raw = base64.b64decode(url.split(';base64,', 1)[1], validate=True)
            if raw.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif raw.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            else:
                raise ModelUnavailable('Unsupported screenshot encoding')
            item['image_url']['url'] = 'data:' + mime + ';base64,' + base64.b64encode(raw).decode()
            images.append({'mime': mime, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return messages, images


class PacedClaude:
    def __init__(self, endpoint, credential_path, audit_path):
        self.endpoint = endpoint
        self.credential_path = Path(credential_path)
        self.audit_path = Path(audit_path)
        self.calls = []

    def completion(self, **kwargs):
        messages, images = normalize_image_mime(kwargs['messages'])
        body = {'model': MODEL, 'messages': messages, 'max_tokens': kwargs.get('max_tokens', 4096)}
        raw = json.dumps(body).encode()
        if len(raw) > 28 * 1024**2:
            raise ModelUnavailable('Request exceeds the existing relay ingress limit')
        record = {'started_unix': time.time(), 'transport_alias': kwargs.get('model'),
                  'actual_model_expected': MODEL, 'images': images, 'request_bytes': len(raw),
                  'request_sha256': hashlib.sha256(raw).hexdigest(),
                  'requested_temperature': kwargs.get('temperature'),
                  'temperature_handling': 'unchanged shared Bedrock relay default'}
        req = urllib.request.Request(self.endpoint, data=raw,
                headers={'Content-Type': 'application/json',
                         'Authorization': 'Bearer ' + self.credential_path.read_text().strip()})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                value = json.load(response)
                record['status'] = response.status
            record['served_model'] = value.get('model')
            if value.get('model') != MODEL:
                raise ModelUnavailable('Unexpected served model')
            content = value['choices'][0]['message'].get('content')
            if not isinstance(content, str) or not content.strip():
                raise ModelUnavailable('Empty model response')
            record.update(served_model=value['model'], usage=value.get('usage'))
            return types.SimpleNamespace(choices=[{'message': {'content': content}}], raw_response=value)
        except urllib.error.HTTPError as exc:
            record.update(status=exc.code, error='HTTPError')
            raise ModelUnavailable('Shared paced endpoint HTTP ' + str(exc.code)) from exc
        except (OSError, ValueError, KeyError, IndexError) as exc:
            record.update(error=type(exc).__name__)
            raise ModelUnavailable('Transport or response error') from exc
        except ModelUnavailable:
            record['error'] = 'invalid_model_response'
            raise
        finally:
            record['ended_unix'] = time.time()
            self.calls.append(record)
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    def install_native_transport(self):
        # Native OpenAIEngine uses only litellm.completion. Keep its prompt
        # construction and parser; do not inherit its unbounded API retries.
        shim = types.ModuleType('litellm')
        shim.completion = self.completion
        sys.modules['litellm'] = shim
        existing = sys.modules.get('seeact.demo_utils.inference_engine')
        if existing is not None:
            existing.litellm = shim

    def sdk_create(self, **kwargs):
        from openai.types.chat import ChatCompletion
        return ChatCompletion(**self.completion(**kwargs).raw_response)

    def validate_completed_calls(self):
        if not self.calls:
            raise ModelUnavailable('No audited native model calls')
        for index, call in enumerate(self.calls):
            if call.get('served_model') not in (None, MODEL):
                raise ModelUnavailable('A different model was served')
            if call.get('error'):
                recovered = any(later.get('request_sha256') == call['request_sha256']
                                and later.get('served_model') == MODEL and not later.get('error')
                                for later in self.calls[index + 1:])
                if not recovered:
                    raise ModelUnavailable('An inference request did not recover')


def free_loopback_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def official_score(checkout, task_file, cdp_port, target_id, answer, observation, bridge):
    sys.path.insert(0, str(checkout))
    from playwright.sync_api import sync_playwright
    from browser_env.actions import create_stop_action
    from evaluation_harness.evaluators import evaluator_router
    from llms.providers import openai_utils
    openai_utils.client.chat.completions.create = bridge.sdk_create
    task = json.loads(Path(task_file).read_text())
    captioning_fn = None
    if 'page_image_query' in task['eval']['eval_types']:
        from evaluation_harness.image_utils import get_captioning_fn
        import torch
        torch.set_num_threads(2)
        captioning_fn = get_captioning_fn('cpu', torch.float32, 'Salesforce/blip2-flan-t5-xl')
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp('http://127.0.0.1:' + str(cdp_port))
        page = None
        for context in browser.contexts:
            for candidate in context.pages:
                session = context.new_cdp_session(candidate)
                info = session.send('Target.getTargetInfo')['targetInfo']
                session.detach()
                if info['targetId'] == target_id:
                    page = candidate
        if page is None:
            raise RuntimeError('Cannot locate the native page for official evaluation')
        trajectory = [{'observation': observation, 'info': {}}, create_stop_action(answer)]
        return float(evaluator_router(task_file, captioning_fn=captioning_fn)(
            trajectory=trajectory, config_file=str(task_file), page=page))


async def run_case(source, checkout, task_file, output, bridge, max_steps=10):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    task_file = Path(task_file)
    task = json.loads(task_file.read_text())
    validate_task(task)
    sys.path.insert(0, str(Path(source) / 'seeact_package'))
    bridge.install_native_transport()
    os.environ['OPENAI_API_KEY'] = 'local-transport-shim-not-an-upstream-key'
    os.environ['OPENAI_BASE_URL'] = bridge.endpoint.rsplit('/chat/completions', 1)[0]
    from seeact import agent as native
    from seeact.demo_utils.inference_engine import OpenAIEngine
    original_launch = native.normal_launch_async
    original_context = native.normal_new_context_async
    original_factory = native.engine_factory
    port = free_loopback_port()

    async def launch(playwright, headless=False, args=None):
        # The pinned upstream helper hardcodes headless=False. This wrapper
        # changes browser startup only, retaining the same Chromium executable.
        return await playwright.chromium.launch(headless=True,
            args=list(args or []) + ['--disable-gpu', '--remote-debugging-port=' + str(port)])

    async def context(browser, **kwargs):
        if task.get('storage_state'):
            kwargs['storage_state'] = task['storage_state']
        kwargs['geolocation'] = task.get('geolocation')
        return await original_context(browser, **kwargs)

    native.normal_launch_async = launch
    native.normal_new_context_async = context
    # Select the native OpenAI-shaped engine as a transport implementation;
    # every actual request is still sent to and checked against Claude.
    native.engine_factory = lambda model=None, **kwargs: OpenAIEngine(model=MODEL, **kwargs)
    agent = None
    summaries, answer, result = [], '', None
    try:
        agent = native.SeeActAgent(default_task=task['intent'], default_website=task['start_url'],
            save_file_dir=str(output / 'native'), headless=True, model=MODEL,
            max_auto_op=max_steps, grounding_strategy='text_choice_som',
            viewport={'width': 1280, 'height': 720})
        await agent.start()
        for step in range(max_steps):
            if agent.complete_flag:
                break
            prediction = await agent.predict()
            if prediction is None and not agent.complete_flag:
                raise RuntimeError('Native planner returned no prediction')
            if prediction is None:
                break
            if prediction['action'] == 'SAY':
                answer = str(prediction.get('value') or '')
            summaries.append({'step': step + 1, 'action': prediction['action'],
                              'value': prediction.get('value'),
                              'description': (prediction.get('element') or {}).get('description')})
            await agent.execute(prediction)
        await agent.page.screenshot(path=str(output / 'final.png'))
        html = await agent.page.content()
        (output / 'final.html').write_text(html)
        session = await agent.session_control['context'].new_cdp_session(agent.page)
        target = (await session.send('Target.getTargetInfo'))['targetInfo']['targetId']
        await session.detach()
        final_url = agent.page.url
        observation = {'text': await agent.page.inner_text('body')}
        score = await asyncio.to_thread(official_score, checkout, task_file, port, target, answer, observation, bridge)
        bridge.validate_completed_calls()
        result = {'task_id': task['task_id'], 'score': score, 'valid': True,
                  'steps': len(summaries), 'final_url_before_evaluation': final_url,
                  'answer': answer, 'actions': summaries, 'model': MODEL,
                  'method': 'native SeeAct with Claude transport, text_choice_som',
                  'model_calls': len(bridge.calls), 'max_steps': max_steps,
                  'task_sha256': hashlib.sha256(task_file.read_bytes()).hexdigest(),
                  'official_evaluator_sha256': hashlib.sha256(
                      (Path(checkout) / 'evaluation_harness/evaluators.py').read_bytes()).hexdigest()}
        return result
    finally:
        (output / 'action_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        try:
            try:
                if agent is not None:
                    await agent.stop()
            finally:
                browser = agent.session_control.get('browser') if agent is not None else None
                if browser:
                    await browser.close()
                if hasattr(agent, 'playwright'):
                    await agent.playwright.stop()
        finally:
            native.normal_launch_async = original_launch
            native.normal_new_context_async = original_context
            native.engine_factory = original_factory
