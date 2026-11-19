import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from forum_pair_queue import Queue, remap, valid_score


class MatchedQueueTests(unittest.TestCase):
    def make_protocol(self, root):
        task = root / 'original.json'
        task.write_text('{"task_id": 3}')
        digest = hashlib.sha256(task.read_bytes()).hexdigest()
        protocol = root / 'protocol.json'
        protocol.write_text(json.dumps({'workers': 2, 'run_id': 'test_run',
            'frozen_files': {str(task): digest},
            'tasks': [{'task_id': 3, 'config_file': str(task), 'sha256': digest, 'reference_inputs': []}]}))
        return protocol, digest

    def make_score(self, root, task_digest, protocol_digest, **changes):
        path = root / 'cases/3/base/attempt1/result.json'
        path.parent.mkdir(parents=True)
        score = dict(task_id=3, method_key='base', run_id='test_run', fixture=False, valid=True,
                     score=0., model='anthropic.claude-opus-5-5', original_task_sha256=task_digest,
                     protocol_sha256=protocol_digest, reference_inputs=[])
        score.update(changes)
        path.write_text(json.dumps(score))
        return path

    def test_resume_keeps_valid_negative_and_protects_its_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, digest = self.make_protocol(root)
            raw = self.make_score(root, digest, hashlib.sha256(protocol.read_bytes()).hexdigest())
            before = raw.read_bytes()
            queue = Queue(protocol)
            try:
                self.assertEqual(queue.scores['3/base']['score'], 0.)
                queue.save()
                self.assertEqual(raw.read_bytes(), before)
                raw.write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'valid score artifact changed'):
                    queue.verify_frozen()
            finally:
                queue.lock.close()

    def test_synthetic_or_other_model_or_other_protocol_cannot_enter_denominator(self):
        for change in [{'fixture': True}, {'model': 'other'}, {'protocol_sha256': 'other'}, {'score': float('nan')}]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol, digest = self.make_protocol(root)
                sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
                raw = self.make_score(root, digest, sha, **change)
                with self.assertRaises(RuntimeError):
                    valid_score(raw, 3, 'base', 'test_run', digest, sha)

    def test_two_existing_evaluators_leave_only_one_new_slot(self):
        queue = object.__new__(Queue)
        queue.mutex = threading.RLock()
        queue.reservations = set()
        with patch.object(queue, 'resource_ready', return_value=True), patch.object(queue, 'other_evaluator_count', return_value=2):
            self.assertTrue(queue.reserve(0))
            self.assertFalse(queue.reserve(1))

    def test_origin_mapping_preserves_task_values_and_reference_order(self):
        task = {'task_id': 3, 'start_url': 'http://old/f/a',
                'eval': {'reference_url': 'http://old/f/b'}, 'image': ['first', 'second']}
        changed = remap(task, 'http://old', 'http://new')
        self.assertEqual(task['start_url'], 'http://old/f/a')
        self.assertEqual(changed['eval']['reference_url'], 'http://new/f/b')
        self.assertEqual(changed['image'], task['image'])


if __name__ == '__main__':
    unittest.main()
