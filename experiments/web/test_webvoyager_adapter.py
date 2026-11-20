import json
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest

from webvoyager_adapter import BrowserGuard, native_actions
from forum_pair_queue import Queue


class WebVoyagerTests(unittest.TestCase):
    def test_native_answer_and_invalid_output_are_distinct(self):
        def parser(action):
            if action == 'answer; ready':
                return 'answer', {'content': 'ready'}
            return 'click', ['0']
        actions, answer = native_actions([
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': 'malformed'},
            {'role': 'assistant', 'content': 'Thought: done\nAction: answer; ready'}], parser)
        self.assertEqual(answer, 'ready')
        self.assertEqual([a['action'] for a in actions], ['native format error', 'answer'])

    def test_guard_failure_does_not_become_task_zero(self):
        guard = BrowserGuard(1, 'http://old', 'http://owned', None, [{'event': 'guard_error'}])
        with self.assertRaisesRegex(RuntimeError, 'routing failed'):
            guard.check()

    def test_single_method_control_score_must_have_same_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'protocol.json'
            protocol = {'model': 'claude', 'seed_image': 'seed-one', 'max_steps': 10,
                'viewport': [1280, 720], 'original_origin': 'http://old', 'tasks': []}
            path.write_text(json.dumps(protocol))
            import hashlib
            queue = object.__new__(Queue)
            queue.protocol = dict(protocol, reference_control={'protocol_file': str(path),
                'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            queue.protocol['seed_image'] = 'seed-two'
            with self.assertRaisesRegex(RuntimeError, 'setting mismatch: seed_image'):
                queue.reference_controls()

    def test_only_the_valid_intersection_is_paired_and_control_bytes_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / 'control'
            run = root / 'external'
            control.mkdir()
            run.mkdir()
            tasks = [{'task_id': i, 'sha256': 'task-' + str(i), 'reference_inputs': []} for i in range(2)]
            settings = {'model': 'anthropic.claude-opus-5-5', 'seed_image': 'same-seed', 'max_steps': 10,
                        'viewport': [1280, 720], 'original_origin': 'http://old', 'tasks': tasks}
            cp = control / 'protocol.json'
            cp.write_text(json.dumps(dict(settings, run_id='control')))
            cs = hashlib.sha256(cp.read_bytes()).hexdigest()
            ep = run / 'protocol.json'
            ep.write_text(json.dumps(dict(settings, run_id='external', methods=['webvoyager'], workers=1,
                frozen_files={str(cp): cs}, reference_control={'protocol_file': str(cp), 'protocol_sha256': cs})))
            es = hashlib.sha256(ep.read_bytes()).hexdigest()
            def score(folder, task, method, run_id, protocol_sha, value):
                path = folder / 'cases' / str(task['task_id']) / method / 'attempt1/result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(task_id=task['task_id'], method_key=method, run_id=run_id,
                    valid=True, fixture=False, model=settings['model'], original_task_sha256=task['sha256'],
                    protocol_sha256=protocol_sha, reference_inputs=[], score=value)))
                return path
            raw = score(control, tasks[0], 'base', 'control', cs, 0.)
            score(run, tasks[0], 'webvoyager', 'external', es, 1.)
            score(run, tasks[1], 'webvoyager', 'external', es, 0.)
            before = raw.read_bytes()
            queue = Queue(ep)
            try:
                report = json.loads((run / 'partial_pair_report.json').read_text())
                self.assertEqual(report['valid_method_scores'], 2)
                self.assertEqual(report['valid_pairs'], 1)
                self.assertEqual(report['paired']['base'], {'score_sum': 0., 'denominator': 1})
                self.assertFalse((run / 'pair_report.json').exists())
                self.assertEqual(raw.read_bytes(), before)
                raw.write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'valid score artifact changed'):
                    queue.verify_frozen()
            finally:
                queue.lock.close()


if __name__ == '__main__':
    unittest.main()
