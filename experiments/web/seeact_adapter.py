"""Native SeeAct loop with a paced Claude transport and official VWA scoring.

The upstream planner, referring prompt, grounding parser, and actions stay
unchanged. This is a Claude adaptation, not the upstream GPT-4 configuration.
"""
import asyncio
import base64
import copy
from collections import Counter
import hashlib
import io
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


def load_task_references(task, config_file, output, base_dir=None):
    """Load every declared task image; missing input is an invalid attempt."""
    from PIL import Image
    references = task.get('image') or []
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, list) or not all(isinstance(p, str) for p in references):
        raise ModelUnavailable('Invalid reference-image declaration')
    loaded, audit = [], []
    for index, name in enumerate(references):
        if name.startswith(('http://', 'https://')):
            request = urllib.request.Request(name, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(32 * 1024**2 + 1)
        else:
            path = Path(name)
            if not path.is_absolute():
                path = (Path(base_dir) if base_dir else Path(config_file).parent) / path
            raw = path.read_bytes()
        if len(raw) > 32 * 1024**2:
            raise ModelUnavailable('Reference image exceeds the recorded input budget')
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            dimensions, original_format = image.size, image.format
            if original_format in ('PNG', 'JPEG'):
                sent = raw
                mime = 'image/png' if original_format == 'PNG' else 'image/jpeg'
            else:
                stream = io.BytesIO()
                image.save(stream, format='PNG')
                sent, mime = stream.getvalue(), 'image/png'
        destination = Path(output) / ('reference_' + str(index) + '.source')
        destination.write_bytes(raw)
        audit.append({'index': index, 'source': name, 'bytes': len(raw),
                      'source_sha256': hashlib.sha256(raw).hexdigest(),
                      'encoded_sha256': hashlib.sha256(sent).hexdigest(),
                      'width': dimensions[0], 'height': dimensions[1],
                      'original_format': original_format, 'encoded_mime': mime})
        loaded.append({'type': 'image_url', 'image_url': {
            'url': 'data:' + mime + ';base64,' + base64.b64encode(sent).decode(), 'detail': 'high'}})
    (Path(output) / 'reference_inputs.json').write_text(json.dumps(audit, indent=2) + '\n')
    return loaded, audit


def attach_reference_images(messages, references):
    messages = copy.deepcopy(messages)
    if not references:
        return messages
    for message in messages:
        if message.get('role') == 'user':
            content = message.get('content', [])
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            message['content'] = content + [{'type': 'text', 'text':
                'The following image(s) are task reference inputs, separate from the current webpage screenshot.'}] + copy.deepcopy(references)
            return messages
    raise ModelUnavailable('Native prompt has no user message for task reference images')


def verify_reference_delivery(calls, expected_hashes):
    planning = [call for call in calls if call.get('stage') == 'planning' and not call.get('error')]
    if not planning:
        raise ModelUnavailable('No completed planning requests')
    required = Counter(expected_hashes)
    for call in planning:
        observed = Counter(image['sha256'] for image in call.get('images', []))
        if required - observed:
            raise ModelUnavailable('A frozen task reference image was missing from a planning request')


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
        self.reference_inputs = []
        self.stage = 'planning'

    def completion(self, **kwargs):
        record = {'started_unix': time.time(), 'transport_alias': kwargs.get('model'),
                  'actual_model_expected': MODEL, 'request_sha256': None,
                  'stage': self.stage,
                  'requested_temperature': kwargs.get('temperature'),
                  'temperature_handling': 'unchanged shared Bedrock relay default'}
        try:
            messages = kwargs['messages']
            if kwargs.get('_native'):
                messages = attach_reference_images(messages, self.reference_inputs)
            messages, images = normalize_image_mime(messages)
            # Match the existing shared relay's large-image encoding, without
            # resizing pixels. This also keeps the complete body below ingress.
            if any(image['bytes'] > 3_700_000 for image in images):
                from bedrock_image_transport import normalize_messages
                messages, conversions = normalize_messages(messages)
            else:
                conversions = []
            messages, sent_images = normalize_image_mime(messages)
            body = {'model': MODEL, 'messages': messages, 'max_tokens': kwargs.get('max_tokens', 4096)}
            raw = json.dumps(body).encode()
            record.update(images=sent_images, image_conversions=conversions,
                          native_reference_count=len(self.reference_inputs) if kwargs.get('_native') else 0,
                          request_bytes=len(raw), request_sha256=hashlib.sha256(raw).hexdigest())
            if len(raw) > 28 * 1024**2:
                raise ModelUnavailable('Request exceeds the existing relay ingress limit')
            req = urllib.request.Request(self.endpoint, data=raw,
                    headers={'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + self.credential_path.read_text().strip()})
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

    def native_completion(self, **kwargs):
        return self.completion(_native=True, **kwargs)

    def install_native_transport(self):
        # Native OpenAIEngine uses only litellm.completion. Keep its prompt
        # construction and parser; do not inherit its unbounded API retries.
        shim = types.ModuleType('litellm')
        shim.completion = self.native_completion
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
                recovered = bool(call.get('request_sha256')) and any(later.get('request_sha256') == call['request_sha256']
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
    bridge.reference_inputs, reference_audit = load_task_references(task, task_file, output, base_dir=checkout)
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
        bridge.stage = 'evaluation'
        try:
            score = await asyncio.to_thread(official_score, checkout, task_file, port, target, answer, observation, bridge)
        finally:
            bridge.stage = 'planning'
        bridge.validate_completed_calls()
        result = {'task_id': task['task_id'], 'score': score, 'valid': True,
                  'steps': len(summaries), 'final_url_before_evaluation': final_url,
                  'answer': answer, 'actions': summaries, 'model': MODEL,
                  'method': 'native SeeAct with Claude transport, text_choice_som',
                  'model_calls': len(bridge.calls), 'max_steps': max_steps,
                  'reference_inputs': reference_audit,
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
