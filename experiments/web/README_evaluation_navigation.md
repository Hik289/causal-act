# Official evaluation navigation patience

The official HTML evaluator navigates the method's live page after the native
agent finishes. A slow response can exhaust Playwright's default 30-second
navigation wait even though the environment later completes the same load.
This is an invalid environmental attempt, not a failed task score.

`evaluation_navigation.py` extends only `page.goto` calls inside official
evaluation to a frozen 120,000 ms budget. It preserves the URL, referer and
load predicate, including the default `load`. Explicit unbounded or longer
timeouts remain intact. The method is restored after evaluation, including on
errors. Navigation failures propagate; no fallback score is created. Audit
records contain URL hashes, elapsed time, predicate and both timeout values.

Base, SeeAct and WebVoyager use the same optional setting. It is activated by
the hash-checked runtime repair manifest, with immutable original task,
reference, model, sampling, decision-budget and evaluator files. An unamended
run keeps its original navigation behavior. Existing valid positive and
negative scores are protected and skipped. A repair can grant at most one
persisted extra attempt to explicitly listed exhausted, still-unscored tasks.

The handoff stops new admissions and waits for the previous queue and active
cases before starting the repaired source. It preserves both the original
protocol and all failed attempt logs. A synthetic delayed-resource fixture
checks the same `load` predicate, and an independent clean-seed diagnostic
must establish that the affected environment responds within the new budget.
Neither diagnostic produces model requests or benchmark scores.

Run unit checks with:

```sh
PYTHONPATH=experiments/web python -m unittest experiments/web/test_evaluation_navigation.py
```
