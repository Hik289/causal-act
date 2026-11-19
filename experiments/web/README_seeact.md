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
header without resizing, recompressing, or changing any image byte. It also
makes the upstream browser helper honor headless execution and accepts the
benchmark's existing authentication state. These are recorded environment and
transport adaptations. Native prompts, parser rules, and actions are unchanged.
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

The original manifest contains 86 text-only tasks and 84 tasks with reference
images. This adapter currently rejects reference-image tasks explicitly rather
than dropping their input. They remain pending an input-preserving adaptation;
an eventual text-only run must be labeled an 86-task subset. Screenshot support
does not establish support for a separate task reference image.

`prepare_forum_seed.py` starts only an owner-labeled container from the pinned
image, checks readiness and disk space, then stops PostgreSQL cleanly before
committing a new seed image. It requires copy-on-write storage and refuses
unreviewed data volumes, an existing preparation container, or an existing seed
tag. Cleanup verifies ownership. The HTTP readiness probe changes visit state,
so both methods require the same new seed; historical controls keep their own
environment record. Preparation creates no benchmark scores.
