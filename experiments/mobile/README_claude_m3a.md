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

## Controlled development Base attempt

`native_attempt.py` connects the existing frozen-task lifecycle to the original
`android_world.episode_runner.run_episode` and native success evaluator. It
implements one development Base attempt, without an emulator launcher, retry,
Refiner method, training policy or formal-transfer runner. Its fourteen new
offline checks are separate from the earlier 21 mobile checks.

A separate preflight checked the actual imported native source bytes and Base
binding without execution. Four new scripted cases exercised the unchanged
native episode runner: valid zero, budget exhaustion, provider exception and
cleanup exception. They initialized no real task, launched no emulator and
made no model request. Their scripted scores are not AndroidWorld results.

Before initialization, a SHA-bound private protocol must match the task family,
params and seed, apps, goal, snapshot archive, actual imported native sources,
native step budget/start-screen setting and admitted provider route/retry policy.
The supplied agent must be the unchanged native M3A with the explicit Claude
wrapper and original pauses. Provider readiness must be explicitly true under
its bound evidence hash; current provider-pending evidence prevents execution
before native initialization or model requests.

An attempt directory is created once and cannot be overwritten. The original
episode data is serialized as private write-only evidence. The scorer runs once
after an episode, including budget exhaustion. Valid zeros are retained; a
model, device, scorer or cleanup exception propagates and leaves `score=null`.
A score computed before failed cleanup remains in the evidence but is not
committed as valid. Even successful development attempts contribute zero formal
benchmark or transfer scores. Native lifecycle cleanup remains responsible for
preserving an original exception when teardown also fails.

This entry point does not establish provider recovery. A controlled Refiner
integration, training/validation data and checkpoint selection, and frozen
held-out applications/tasks/seeds are still required for E7.

## Native device-error boundary

The pinned M3A catches exceptions from `env.execute_action` and returns a failed
step. This includes some device failures, which could otherwise reach the
success evaluator after the episode budget is exhausted.

`environment_guard.py`, used by the development Base entry, temporarily observes
the owned agent's environment and step calls. Typed OS/HTTP/subprocess timeout,
ADB-controller and gRPC errors, plus non-OK responses raised by the pinned ADB
`check_ok` helper, stop the episode before scoring with
`InvalidEnvironmentCall`. Their original cause and private failed-step evidence
are retained, and the attempt has `score=null`. The original environment and
agent step are restored on exit. The guard neither retries nor calls a model.

Action `ValueError` and unclassified `RuntimeError` retain native behavior;
message text alone does not determine validity. This boundary does not claim
to diagnose every possible device failure. The native source, prompts, action
values, screenshot encoding and default pauses remain unchanged. A new private
development protocol must additionally freeze the actual imported environment
source hashes; prior preparation protocols and receipts remain immutable.

Ten new offline checks cover exception classification, restoration, evidence
permissions and duplicate-attempt protection. Seven additional scripted cases
with the actual pinned M3A and episode runner reproduce the original swallowed
OS error, verify four device-error paths stop before scoring, and preserve
ordinary invalid-action and healthy execution behavior. Native timing defaults
are retained in those cases. All use scripted local responses, with zero real
task initializations, real task scorer calls, device actions, emulator launches,
external model requests or benchmark/transfer scores. The existing provider
readiness gate still blocks live execution before initialization.

## Development Refiner proposal bridge

`refiner_bridge.py` connects an explicitly supplied development policy to the
action-proposal boundary of the pinned native M3A. It changes the returned
Reason/Action text before native parsing. Consequently, execution, the native
summary prompt and action history refer to the same refined action. The
original provider response is retained separately from Refiner provenance.
Summary requests pass through unchanged; keeping a proposal preserves the exact
original response. The policy receives copies of the two native images and
parsed proposal, together with the original goal and prompt. It receives no task
object, scorer or outcome label.

The private protocol must bind the exact native source and the actual supplied
policy source, assets and settings. A changed native phase boundary, an unbound
policy or a formal scope is rejected. Only the owned agent is wrapped, and its
inference interface and step are restored after errors. Existing attempt and
provider guards still belong to the outer runner. This hook does not initialize
a task, launch an emulator, add a model request or provide retry allowances.

Malformed nonempty Base and Refiner actions retain native parser behavior.
Provider and policy failures propagate without a fallback; evidence is private
and cannot overwrite an earlier bridge directory. An agent-generated complete
status is not substituted for the native success evaluator.

Eleven new offline checks cover proposal/history consistency, input isolation,
summary forwarding, failure propagation and evidence preservation. Separate
scripted cases use the actual pinned M3A and episode runner, including changed
actions, complete status with a valid zero, malformed actions, device failure
and local provider failures. They initialize no real task or emulator and make
no external model request. Their scripted scores are not benchmark results.

The bridge supplies a development hook. The controlled pair entry below adds
orchestration; a trained mobile Refiner, training/validation checkpoint selection
and frozen held-out task/seed evaluation remain required for E7.

## Controlled development pairs

`controlled_pair.py` runs one pre-registered development Base/Refiner pair in a
frozen arm order. Both arms require fresh native task and agent objects, with
identical task parameters, seed, goal, native settings and snapshot archive. The
actual implementation files, native protocol and policy-bridge protocol are
bound by SHA before either arm initializes a task. Provider-pending evidence
blocks the public entry before initialization or model requests. The caller
supplies the owned environment and policy; there is no emulator launcher,
provider fallback, learned policy, training procedure or formal-transfer queue.

Each arm has exactly one attempt. An exclusive pair lock prevents concurrent
controllers. Resume verifies the original protocol, result and recursively
hashed evidence, preserves both valid zeros and invalid attempts, and executes
only an arm whose attempt has never started. An interrupted arm without a
committed receipt requires inspection rather than a new attempt. Earlier
reports are retained under separate filenames. A provider, device or cleanup
failure remains invalid with `score=null`; the original exception propagates.

The effective pair denominator is one only when both native scores are valid.
Otherwise the paired difference is null, while any valid arm remains visible.
All development receipts have zero benchmark and transfer scores. The Refiner
arm is explicitly identified in its native attempt record.

Sixteen new offline checks cover immutable scores, invalid-arm preservation,
resume, concurrency and identity gates. Separate scripted pairs exercise the
pinned native M3A and episode runner; fixture scores are preparation evidence.
These do not establish online provider recovery or a trained Refiner. E7 still
requires training/validation selection, frozen held-out applications/tasks/seeds
and actual controlled live execution before any formal result is reported.
