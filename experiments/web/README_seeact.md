# Native SeeAct with a paced Claude transport

`seeact_adapter.py` keeps the upstream SeeAct action-generation prompt,
referring prompt, grounding parser, and browser actions. The pinned source is
[`2434627`](https://github.com/OSU-NLP-Group/SeeAct/tree/2434627b196b33d1b0668a418188dd2f348883bb).
The model is AWS Bedrock `anthropic.claude-opus-5-5`; this is a Claude adaptation,
not a reproduction of the upstream GPT-4 scores.

The adapter replaces the transport behind the native engine with the existing
shared paced endpoint. Every accepted response must name the expected Claude
model and contain nonempty text. HTTP failures and empty responses remain
invalid inference, with an audit record. A recovered retry of the same request
does not discard a subsequently valid result. The endpoint retains the shared
eight-request concurrency limit and eight starts per six seconds. Credentials
are read from a caller-supplied local file and are never placed in the audit.

Upstream labels its PNG screenshot bytes as JPEG. The adapter corrects the MIME
header without changing screenshot bytes. For images above the relay's existing
3,700,000-byte limit, both executors use its recorded JPEG conversion without
resizing dimensions. Original files and conversion metadata remain available. It also
makes the upstream browser helper honor headless execution and accepts the
benchmark's existing authentication state. These are recorded environment and
transport adaptations. Native prompt strings, parser rules, and actions are unchanged;
SeeAct's user message gains a labeled block containing every task reference image.
The deployed browser package is recorded separately from upstream requirements.

The official VisualWebArena evaluator attaches through CDP to the exact native
page target. It does not reopen the final URL, which would lose unsaved form
values and other browser state. Evaluator model requests use the same audited
Claude endpoint. Evidence, traces, request audit, final page, source hashes,
step budget, and the official evaluator's result remain separate artifacts.

`seeact_fixture.py` is an unscored integration check. A local button changes an
input's live value, and the official HTML evaluator reads that value from the
same browser. Its negative task ID is outside the benchmark manifest. Passing
this fixture is not a benchmark success or a completed external comparison.

```bash
python -m unittest test_seeact_adapter
python seeact_fixture.py \
  --seeact-source /absolute/path/to/pinned/SeeAct \
  --vwa-checkout /absolute/path/to/visualwebarena \
  --relay-endpoint http://127.0.0.1:PORT/worker/seeact_fixture/v1/chat/completions \
  --credential-file /absolute/path/to/local/client-credential \
  --output /absolute/path/to/new/fixture-directory
```

The planned Forum comparison retains all 170 original task IDs, ten decisions
per task, and a 1280×720 viewport. A fresh, identically initialized forum is
required for each method/task, including the 18 tasks whose original config
requests reset. New Base and SeeAct controls must share this reset protocol.
Historical Base/CausalAct scores retain their own protocol and cannot be
silently substituted as matched controls. The isolated factory must pass
startup, authentication, and disk checks before scored tasks begin. Until those
checks and the complete valid comparison finish, main-table entries stay
pending. The official 910-task coverage and full CausalAct comparison remain
separate incomplete work.

The original manifest contains 86 text-only tasks and 84 tasks with 89 reference
images. Resolve the benchmark's relative image paths against its checkout.
For the matched comparison, retain original files and freeze a shared, metadata-free
PNG representation of the decoded first frame, preserving dimensions and alpha.
This follows the existing Base's still-image loading behavior for GIF/WebP and
supplies identical reference inputs to both executors. Record original, canonical,
and transport hashes. Every completed planning request must contain all expected
reference-image hashes; evaluator requests are audited separately. Missing files
or dropped references invalidate the attempt rather than producing task zeroes.
`freeze_forum_inputs.py --tasks /absolute/path/to/tasks --checkout
/absolute/path/to/visualwebarena --output /absolute/path/to/new/input-directory`
creates this manifest, preserves the original files, and verifies decoded pixel
hashes before any evaluation starts.

`base_forum_adapter.py` calls the existing Base prompt, parser, actions, and
official evaluator, with its ten-decision budget and 2.5-second action wait.
Its legacy GPT name selects an existing prompt implementation only; every actual
response must be Claude. An exception swallowed by the native runner cannot
become a valid zero: exactly one completed official evaluation and valid model
calls are required. Genuine negative and fractional official scores are retained.
Model decisions are counted separately from the runner's automatic budget stop.

`forum_pair_queue.py --protocol /absolute/path/to/frozen/protocol.json` resumes
only missing method/task pairs. Its protocol freezes sources, original tasks,
reference inputs, seed image, step budget, environment, retry budget, and a fixed
counterbalanced method order. Each method/task receives a fresh owner-labeled
container and the same official login procedure. Factory initialization satisfies
all reset-required tasks; it never resets the original running forum. New browser
contexts redirect legacy absolute navigation URLs with HTTP 307 and proxy legacy
subresources/API requests to their own environment, preserving request methods
and bodies without contacting the old service. This guard is installed before
navigation for both executors, including the benchmark's `reddit.com` prompt
aliases; routing errors remain invalid attempts. At most two
new workers run, with admission respecting the shared limit of three Forum/OS
workers, host memory, and disk floors. The existing shared relay keeps its global
limits. Terminal inference/environment errors retain their logs, remain unscored,
and use bounded retries/backoff. Valid score bytes are hash-protected and never
rerun. Reports retain per-method denominators and the valid paired intersection;
the full report is written only after all 170 pairs have valid scores.

The reference-image fixture can exercise either executor with `--method base`
or `--method seeact` plus `--reference-image-check`. It remains outside every
benchmark denominator. The matched queue does not replace historical scores or
establish a same-initial-state comparison with historical CausalAct controls.

`prepare_forum_seed.py` starts only an owner-labeled container from the pinned
image, checks readiness and disk space, then stops PostgreSQL cleanly before
committing a new seed image. It requires copy-on-write storage and refuses
unreviewed data volumes, an existing preparation container, or an existing seed
tag. Cleanup verifies ownership. The HTTP readiness probe changes visit state,
so both methods require the same new seed; historical controls keep their own
environment record. Preparation creates no benchmark scores.
