import base64
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
import urllib.error
from unittest.mock import patch

from uitars_adapter import BrowserActions, UITarsClient, planning_messages, MODEL
from seeact_adapter import ModelUnavailable
from forum_pair_queue import valid_score


class Recorder:
    def __init__(self):
        self.calls = []
        self.mouse = self.keyboard = self

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return call


class UITarsAdapterTests(unittest.TestCase):
    def test_http_failure_retains_bounded_diagnostics_and_has_no_response(self):
        with tempfile.TemporaryDirectory() as folder:
            client=UITarsClient('http://fixture',Path(folder)/'audit.jsonl')
            failure=urllib.error.HTTPError('http://fixture',500,'error',{},io.BytesIO(b'processor failed '*1000))
            with patch('urllib.request.urlopen',side_effect=failure), self.assertRaisesRegex(ModelUnavailable,'500'):
                client.complete([{'role':'user','content':'task'}])
            record=client.calls[0]
            self.assertEqual(record['status'],500)
            self.assertEqual(len(record['error_body']),4096)
            self.assertNotIn('response',record)

    def test_history_retains_every_reference_and_current_image_last(self):
        references = [{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(b'ref').decode()}}]
        messages = planning_messages('native prompt', references, [b'0',b'1',b'2',b'3'], ['a0','a1','a2'])
        self.assertEqual(len(messages), 5)
        first = messages[0]['content']
        self.assertEqual(first[0], references[0])
        self.assertEqual(base64.b64decode(first[-1]['image_url']['url'].split(',')[1]), b'1')
        self.assertEqual(base64.b64decode(messages[-1]['content'][-1]['image_url']['url'].split(',')[1]), b'3')
        self.assertEqual(messages[1]['content'],'a1')
        self.assertEqual(len(references),1)

    def test_pointer_coordinates_remain_out_of_frame(self):
        page = Recorder()
        result = BrowserActions(page,[1280,720]).apply({'action_type':'click','action_inputs':{'start_box':'[1.1,0.5,1.1,0.5]'}})
        self.assertEqual(result['screen_point'],[1408.,360.])
        self.assertEqual(page.calls[0][1],(1408.,360.))

    def test_native_type_clipboard_submit_and_address_hotkey(self):
        page = Recorder();actions=BrowserActions(page,[1280,720])
        actions.apply({'action_type':'type','action_inputs':{'content':"quote' 中文\\n"}})
        self.assertEqual(page.calls[0][1][1],"quote' 中文")
        self.assertEqual(page.calls[-1],('press',('Enter',),{}))
        actions.apply({'action_type':'hotkey','action_inputs':{'key':'ctrl l'}})
        actions.apply({'action_type':'type','action_inputs':{'content':'http://fixture/f/a\\n'}})
        self.assertEqual(page.calls[-1],('goto',('http://fixture/f/a',),{}))

    def test_vertical_scroll_matches_native_five_ticks(self):
        page=Recorder();actions=BrowserActions(page,[1280,720])
        a=actions.apply({'action_type':'scroll','action_inputs':{'direction':'down'}})
        self.assertEqual(a['wheel_delta_pixels'],600)
        b=actions.apply({'action_type':'scroll','action_inputs':{'direction':'left'}})
        self.assertIn('native_noop',b)
        self.assertEqual(len(page.calls),1)

    def test_wrong_model_or_empty_response_is_invalid(self):
        for value in [{'model':'other','choices':[{'message':{'content':'x'}}]},
                      {'model':MODEL,'choices':[{'message':{'content':''}}]}]:
            with tempfile.TemporaryDirectory() as folder:
                client=UITarsClient('http://fixture',Path(folder)/'audit.jsonl')
                response=io.BytesIO(json.dumps(value).encode());response.status=200
                with patch('urllib.request.urlopen',return_value=response), self.assertRaises(ModelUnavailable):
                    client.complete([{'role':'user','content':'task'}])
                self.assertIn('error',client.calls[0])

    def test_ui_fixture_and_other_planner_cannot_enter_formal_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'result.json'
            raw=dict(task_id=3,method_key='uitars',run_id='native',fixture=False,valid=True,
                score=0.,model=MODEL,original_task_sha256='task',protocol_sha256='protocol')
            p.write_text(json.dumps(raw));self.assertEqual(valid_score(p,3,'uitars','native','task','protocol')['score'],0.)
            for change in [{'fixture':True},{'model':'anthropic.claude-opus-5-5'}]:
                p.write_text(json.dumps(dict(raw,**change)))
                with self.assertRaises(RuntimeError):valid_score(p,3,'uitars','native','task','protocol')


if __name__=='__main__':unittest.main()
