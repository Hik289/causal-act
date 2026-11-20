"""Evaluation patience preserves native arguments, errors and action scope."""
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_navigation import evaluation_navigation


class Page:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def goto(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return 'original response'


class EvaluationNavigationTest(unittest.TestCase):
    def test_scoped_budget_keeps_predicate_and_restores(self):
        page = Page()
        before = page.goto
        with evaluation_navigation(page, 120000):
            self.assertEqual(page.goto('http://fixture/item', referer='http://fixture/',
                                      wait_until='load', timeout=30000), 'original response')
        self.assertEqual(page.goto, before)
        self.assertEqual(page.calls, [('http://fixture/item', dict(
            referer='http://fixture/', wait_until='load', timeout=120000))])
        page.goto('action')
        self.assertEqual(page.calls[-1], ('action', {}))

    def test_unbounded_longer_and_unamended_waits_remain(self):
        page = Page()
        with evaluation_navigation(page, 120000):
            page.goto('a', timeout=0)
            page.goto('b', timeout=180000, wait_until='domcontentloaded')
        with evaluation_navigation(page):
            page.goto('c')
        self.assertEqual(page.calls, [('a', {'timeout': 0}),
            ('b', {'timeout': 180000, 'wait_until': 'domcontentloaded'}), ('c', {})])

    def test_error_propagates_and_audit_preserves_predicate(self):
        page = Page(TimeoutError('environment timeout'))
        before = page.goto
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / 'audit.jsonl'
            with self.assertRaisesRegex(TimeoutError, 'environment timeout'):
                with evaluation_navigation(page, 120000, audit):
                    page.goto('private-url')
            record = json.loads(audit.read_text())
            self.assertEqual(record['wait_until'], 'load')
            self.assertEqual(record['error'], 'TimeoutError')
            self.assertNotIn('private-url', audit.read_text())
        self.assertEqual(page.goto, before)

    def test_unrecorded_budget_is_rejected(self):
        for budget in [True, 60000, -1]:
            with self.assertRaises(ValueError):
                with evaluation_navigation(Page(), budget):
                    self.fail('must fail before evaluator')


if __name__ == '__main__':
    unittest.main()
