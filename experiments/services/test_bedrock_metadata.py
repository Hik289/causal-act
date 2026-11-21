import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from bedrock_metadata import record_upstream


class MetadataTest(unittest.TestCase):
    def test_disabled_diagnostics_do_not_write(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'missing'
            self.assertIsNone(record_upstream(worker='other',model='model',request_bytes=b'x',response={},output_dir=target,workers={'selected'}))
            self.assertFalse(target.exists())

    def test_exact_response_and_stop_reason_preserved_without_request_body(self):
        with tempfile.TemporaryDirectory() as root:
            response={'content':[],'stop_reason':'end_turn','usage':{'input_tokens':275,'output_tokens':0},'id':'provider-id'}
            before=copy.deepcopy(response);request=b'private original prompt and image'
            path=record_upstream(worker='selected',model='model',request_bytes=request,response=response,output_dir=root,workers={'selected'})
            value=json.loads(Path(path).read_text())
            self.assertEqual(value['provider_response'],before)
            self.assertEqual(response,before)
            self.assertEqual(value['request_sha256'],hashlib.sha256(request).hexdigest())
            self.assertNotIn(request.decode(),Path(path).read_text())
            self.assertEqual(Path(path).stat().st_mode & 0o777,0o600)

    def test_recording_does_not_mutate_nonempty_text_or_thinking_blocks(self):
        response={'content':[{'type':'text','text':'native output'},{'type':'thinking','thinking':'private provider block'}],'stop_reason':'max_tokens'}
        before=copy.deepcopy(response)
        with tempfile.TemporaryDirectory() as root:
            record_upstream(worker='selected',model='model',request_bytes=b'x',response=response,output_dir=root,workers={'selected'})
        self.assertEqual(response,before)

    def test_io_failure_does_not_change_inference_control_flow(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'file';target.write_text('keep')
            self.assertIsNone(record_upstream(worker='selected',model='model',request_bytes=b'x',response={},output_dir=target,workers={'selected'}))
            self.assertEqual(target.read_text(),'keep')

    def test_each_record_has_a_unique_file(self):
        with tempfile.TemporaryDirectory() as root:
            names=[record_upstream(worker='selected',model='model',request_bytes=b'x',response={},output_dir=root,workers={'selected'}) for _ in range(2)]
            self.assertNotEqual(*names)
            self.assertEqual(len(list(Path(root).glob('*.json'))),2)


if __name__=='__main__':unittest.main()
