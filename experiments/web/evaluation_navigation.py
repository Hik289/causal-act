"""Bounded navigation patience scoped to the official evaluator's live page."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time


@contextmanager
def evaluation_navigation(page, timeout_ms=None, audit_file=None):
    """Extend a frozen evaluation budget without changing its load predicate.

    Actions, observation collection and model calls are outside this scope.
    Explicit unbounded or longer waits keep the evaluator's original setting.
    Exceptions propagate, and the page's original method is always restored.
    """
    if timeout_ms is None:
        yield
        return
    if isinstance(timeout_ms, bool) or timeout_ms != 120000:
        raise ValueError('The recorded evaluation repair requires 120000 ms')
    original = page.goto

    def goto(url, **kwargs):
        options = dict(kwargs)
        requested = options.get('timeout')
        if requested is None or 0 < requested < timeout_ms:
            options['timeout'] = timeout_ms
        record = dict(url_sha256=hashlib.sha256(str(url).encode()).hexdigest(),
                      wait_until=options.get('wait_until', 'load'),
                      requested_timeout_ms=requested,
                      effective_timeout_ms=options.get('timeout'))
        start = time.monotonic()
        try:
            return original(url, **options)
        except Exception as exc:
            record['error'] = type(exc).__name__
            raise
        finally:
            record['elapsed_s'] = time.monotonic() - start
            if audit_file is not None:
                with Path(audit_file).open('a') as stream:
                    stream.write(json.dumps(record) + '\n')

    page.goto = goto
    try:
        yield
    finally:
        page.goto = original
