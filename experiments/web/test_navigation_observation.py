from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from navigation_observation import stable_prediction


class Agent:
    def __init__(self, bridge, permanent=False, after_model=False):
        self.bridge, self.permanent, self.after_model = bridge, permanent, after_model
        self.time_step, self.predictions = 0, 0
        self.page = types.SimpleNamespace(wait_for_load_state=AsyncMock())

    async def predict(self):
        self.time_step += 1
        self.predictions += 1
        if self.after_model:
            self.bridge.calls.append('completed request')
        if self.permanent or self.predictions == 1:
            raise RuntimeError('Execution context was destroyed, most likely because of a navigation')
        self.bridge.calls.append('completed request')
        return {'action': 'TERMINATE'}


class ObservationRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovered_dom_read_does_not_spend_model_or_action_budget_twice(self):
        bridge = types.SimpleNamespace(calls=[])
        agent = Agent(bridge)
        with tempfile.TemporaryDirectory() as directory, patch('navigation_observation.asyncio.sleep', new=AsyncMock()):
            result = await stable_prediction(agent, bridge, Path(directory) / 'audit.jsonl')
        self.assertEqual(result['action'], 'TERMINATE')
        self.assertEqual(agent.predictions, 2)
        self.assertEqual(agent.time_step, 1)
        self.assertEqual(len(bridge.calls), 1)

    async def test_error_after_a_model_request_is_not_replayed(self):
        bridge = types.SimpleNamespace(calls=[])
        agent = Agent(bridge, after_model=True)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RuntimeError):
            await stable_prediction(agent, bridge, Path(directory) / 'audit.jsonl')
        self.assertEqual(agent.predictions, 1)
        self.assertEqual(len(bridge.calls), 1)

    async def test_persistent_navigation_errors_are_bounded_and_remain_unscored(self):
        bridge = types.SimpleNamespace(calls=[])
        agent = Agent(bridge, permanent=True)
        with tempfile.TemporaryDirectory() as directory, patch('navigation_observation.asyncio.sleep', new=AsyncMock()), self.assertRaises(RuntimeError):
            await stable_prediction(agent, bridge, Path(directory) / 'audit.jsonl')
        self.assertEqual(agent.predictions, 3)
        self.assertEqual(agent.time_step, 0)
        self.assertEqual(bridge.calls, [])
