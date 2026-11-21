# Claude interface for native AndroidWorld M3A

`claude_m3a.py` implements AndroidWorld's text and multimodal inference
interfaces using an explicitly supplied shared, paced Claude endpoint. It does
not start a proxy, emulator, task queue, training run or benchmark evaluation.

The native integration is pinned to AndroidWorld
[`55ddaf5edc6f6c25673aed343907d5b9dca66f84`](https://github.com/google-research/android_world/tree/55ddaf5edc6f6c25673aed343907d5b9dca66f84).
`make_native_wrapper` imports the real native interfaces and binds their
`array_to_jpeg_bytes` encoder. Native M3A retains its prompts, screenshot/SOM
order, Reason/Action parser, JSONAction conversion, execution and default
timing. Input arrays use the original JPEG encoder without resizing. This is
an adapted Claude configuration, distinct from a reported GPT/Gemini baseline.

The requested model is `anthropic.claude-opus-5-5`; successful responses must
identify that model. The output budget is 1,000 tokens. Sampling uses the
existing relay defaults: neither a forwarded temperature nor a seed is claimed.
The safety-classifier result is unknown (`None`), matching an API that supplies
no corresponding classification.

## Runtime binding

Callers must verify the deployed paced service and bind its endpoint, source
receipt, credential-file reference and limits in their private run protocol.
The wrapper accepts explicit loopback `/worker/android_.../v1/chat/completions`
routes and supplies no default API URL or GPT fallback. Existing shared limits
and worker ownership apply; this adapter creates no inference slots.

```python
import os
from pathlib import Path
from claude_m3a import make_native_wrapper
from android_world.agents import m3a

llm = make_native_wrapper(
    endpoint=os.environ["CAUSALACT_ANDROID_PACED_ENDPOINT"],
    credential=Path(os.environ["CAUSALACT_ANDROID_CREDENTIAL_FILE"]).read_text().strip(),
    audit_dir=Path(os.environ["CAUSALACT_ANDROID_AUDIT_DIR"]),
    max_attempts=1,
)
# env must already satisfy the frozen native environment/task protocol.
agent = m3a.M3A(env, llm)
```

The default is one attempt. An explicit allowance may permit up to three, with
one declared delay per retry. Service `Retry-After` is never shortened; outer
queues must account for these attempts rather than multiply retry allowances.
No request follows an HTTP redirect.

Private audits retain the request body, exact image/input hashes, raw successful
response, bounded invalid-response bytes and per-attempt events. Authorization
headers are not logged. Keep audits, credentials, original images and private
protocols outside the published checkout.

## Failure and scoring behavior

Provider, connection, empty-response, schema and model-identity failures raise
`InvalidModelCall`. This also escapes native M3A's **summary** phase, preventing
a failed model request from quietly becoming a task zero. Callers must retain
the attempt and record `score=null` under the frozen invalid-attempt policy.

Nonempty malformed actions, infeasibility and nonempty length-limited model
outputs remain with the native parser and task scorer. They are not discarded
or retried based on task outcome.

## Validation and remaining scope

```sh
python -m unittest discover -s experiments/mobile -p 'test_*.py' -v
```

The ten offline wire/error tests make no provider or device calls. Separately,
native integration must exercise the real pinned M3A action/summary interfaces
and confirm that transport errors propagate. A synthetic live transport probe
is still preparation and contributes zero benchmark or transfer scores.

Formal E7 evaluation uses the frozen-task lifecycle below and still requires a
controlled agent/Refiner runner, training/validation checkpoint selection and a
frozen held-out task/seed protocol. This interface alone produces no E7 result.

## Frozen task lifecycle

`frozen_task.py` restores the frozen snapshot archive before calling the
supplied native task initializer. The caller owns the emulator, task identity,
parameters and split. It does not select tasks, launch a model, evaluate success
or supply retry allowances.

The host archive must match its registered SHA. Restoration verifies file
contents, types, UID/GID, permission bits and link targets; timestamps are not
compared. A matching existing snapshot is verified without writing. Different
snapshot storage is exported and renamed for preservation before installing the
frozen archive, so stale extra files cannot survive as an overlay. This helper
does not clear real application data; any application reset belongs to the
unchanged native initializer or teardown.

`frozen_native_task` calls native teardown after the body and also attempts
cleanup after partial initialization. Provider/body errors propagate if teardown
also fails. The pinned native implementation can swallow failed snapshot loads
as warnings; these initialization/teardown warnings are captured and classified
as invalid environment preparation before scoring, rather than task zeros.
Audit directories are created once, and existing evidence is never overwritten.

```python
from frozen_task import NativeSnapshotDevice, frozen_native_task

# task and archive identity must already be bound by the private run protocol.
env.reset(go_home=True)
device = NativeSnapshotDevice(env.controller, adb_path=adb_path, serial=serial)
with frozen_native_task(
    task, env, device=device, archive=archive,
    expected_sha256=archive_sha256, audit_dir=private_audit_dir,
):
    observation = env.get_state(wait_to_stabilize=True)
```

Eleven additional offline checks cover corruption, metadata changes, stale
files, native warning handling and simultaneous provider/cleanup failures. A
separate native development preflight restored 821 frozen members before
`ClockTimerEntry` initialization, read a real observation and completed native
teardown. It called neither an agent nor the success evaluator. Clock is reserved
for development and must be excluded from future E7 held-out application sets.
The frozen host archive remained unchanged. This preflight exercises the missing
snapshot path; preservation of a different existing snapshot is covered by the
offline checks.

The combined mobile checks number 21. The original Claude interface tests are
unchanged. Formal E7 still needs the controlled agent/Refiner runner,
training/validation data and checkpoint selection, and a frozen held-out
protocol. Preparation validity is separate from task success and provider
availability.
