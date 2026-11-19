"""Retry a navigation-invalidated observation only before any model request."""
import asyncio
import json
from pathlib import Path
import shutil
import time


async def stable_prediction(agent, bridge, audit_file, attempts=3):
    audit_file = Path(audit_file)
    for index in range(attempts):
        step_before = agent.time_step
        calls_before = len(bridge.calls)
        try:
            await agent.page.wait_for_load_state('domcontentloaded', timeout=30000)
            return await agent.predict()
        except Exception as error:
            navigation_error = 'Execution context was destroyed' in str(error) or 'Cannot find context with specified id' in str(error)
            if not navigation_error or len(bridge.calls) != calls_before:
                raise
            # The pinned native predict increments only this counter before
            # collecting elements. No model request or action has occurred.
            agent.time_step = step_before
            screenshot = Path(getattr(agent, 'main_path', audit_file.parent)) / 'screenshots' / ('screen_' + str(step_before + 1) + '.png')
            if screenshot.exists():
                archive = audit_file.parent / 'navigation_observations'
                archive.mkdir(exist_ok=True)
                shutil.copy2(screenshot, archive / (str(time.time_ns()) + '.png'))
            record = dict(step=step_before + 1, observation_attempt=index + 1,
                          error=type(error).__name__, message=str(error),
                          model_calls_before=calls_before, model_calls_after=len(bridge.calls),
                          model_request_repeated=False, native_step_counter_restored=True)
            with audit_file.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if index + 1 == attempts:
                raise
            await asyncio.sleep(0.25)
