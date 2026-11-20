# UI-TARS adapted Forum evaluation

The native prompt and parser run through an isolated Chromium adapter and the
unchanged official Forum evaluator. A frozen 170-task queue is deferred until
WebVoyager finishes its own scores and releases its evaluation slot. Scripted
fixtures are preparation, excluded from every benchmark denominator. Actual
UI-TARS transport must pass before formal scoring starts.

Synthetic reference fixtures use the actual processor's patch factor and
minimum-pixel metadata. The original 17×19 fixtures were rejected before
inference; their three invalid transport attempts remain archived. A bound
environment repair can replace only the synthetic preflight images and grant
one persistent additional preflight attempt. It preserves the benchmark
protocol, real task references, service, generation settings and formal-case
retry budgets. HTTP failures retain their status and a bounded error body.

The native source is [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS),
pinned at `582f3a7ea5d285ee8ed9e2e84048d1ab01453c49`. Its unmodified
`codes/ui_tars/action_parser.py` performs Qwen2.5-VL postprocessing. The local
model service uses fp16 across two GPUs and thumbnails each image to a maximum
side of640. A1280×720 screenshot becomes640×360, then the deployed processor's
`image_grid_thw=[1,26,46]` establishes a644×364 model coordinate frame. This
geometry comes from source and processor metadata, without task targets or
benchmark scores.

The frozen local UI client sends canonical inline PNG bytes. Claude controls
use the shared relay's JPEG encoding for images above 3,700,000 bytes. The
inherited protocol field describing that relay applies to the Claude controls;
it does not describe local UI transport. Source-reference hashes must match
the common canonical files, while each model's actual encoded-image hashes
are audited separately. Native image encoding and processing differences
remain part of the reported comparison.

`uitars_contract.py` runs with `CUDA_VISIBLE_DEVICES` empty. It refuses a
thumbnail cap that disagrees with the service source, checks native-parser and
processor geometry, records source/configuration hashes and streams hashes of
local checkpoint files. The weight manifest identifies files on disk; it does
not prove the running service loaded those exact bytes. The script never calls
the service, loads model weights, changes its image policy or modifies a live
evaluation source. Existing E8 single-image and S2 frame protocols stay separate.

`uitars_contract_fixture.py` uses the actual pinned native parser and a real
Chromium page to click a synthetic center button and check retained DOM state.
A second scripted prediction outside the frame must remain outside after
mapping; coordinates are not clamped. The fixture makes zero model requests,
creates no benchmark score and is excluded from every experimental denominator.

`uitars_adapter.py` uses the pinned `COMPUTER_USE_DOUBAO` prompt and native
parser. It keeps all task reference images and three screenshot-history steps,
with the current screen last. Every request records image hashes, the actual
served model, response and usage. The deployed service puts image placeholders
before the text within each message, thumbnails every image, and uses fp16,
deterministic generation and at most 512 new tokens. That service is reused
without restarting or changing it.

Actions execute through an explicit browser whitelist. Pointer coordinates
remain unclamped; Unicode typing uses the isolated browser clipboard; Ctrl+L
operates an HTTP address buffer. Native vertical five-tick scrolls map to 600
pixels, horizontal scrolling retains the pinned converter's no-op, drag takes
one second, and wait takes five seconds. A common 2.5-second observation wait,
1280×720 viewport and ten-decision budget are frozen adaptations. Model action
syntax failures stop the agent and receive official evaluation of the retained
state. Transport, environment or model-identity failures remain invalid attempts.

`uitars_adapter_fixture.py` exercises references across the truncated history,
address navigation, coordinate clicks, Unicode submit and the actual official
DOM evaluator. Its positive and negative scripted cases receive 1 and 0;
both have `fixture=True`, zero actual model requests and zero benchmark scores.

`uitars_deferred_queue.py` checks predecessor raw-score identities and hashes,
actual processes, disk and host memory before reserving the released third
Forum lane. It permits three bounded transport attempts with the existing
backoff, never retries to select a better semantic response, and takes a
singleton lock. Formal cases use fresh owned clean-seed environments and the
same tasks and references as Base/SeeAct. Base scores are referenced without
rerunning or duplicating them. UI-TARS planning and Claude evaluation calls
have separate provenance: comparison with Claude Base uses different native
planning models and does not identify a causal intervention effect.

Existing Base/SeeAct/WebVoyager runtime sources, valid positive and negative
scores, and the E8 single-image and S2 frame protocols remain separately
identified. Completion of these fixtures does not complete the external
baseline or the full 910-task benchmark.
# Recovering exhausted native memory errors

A separately bound environment repair can route missing UI-TARS cases to a
verified local service through `runtime_environment`. It requires a hashed
technical receipt, the native model identity, unchanged input and generation
contracts, and zero benchmark scores from the service checks. External URLs,
credentials in endpoints and other planning methods are rejected. The frozen
protocol file and original service are retained.

The observed failure was CUDA allocation inside visual attention. A new service
can use idle GPUs with the same fp16 checkpoint and an explicit module placement
chosen from checkpoint tensor sizes, reserving activation memory on the vision
GPU. Image bytes, thumbnail processing, history, prompts, token limits, decoding
and actions remain fixed. Check a byte-identical successful request and an exact
previously failing multi-image request before admitting formal cases; record
the output comparison without selecting for semantic success. A failed auto
placement check and any invalid inference logs remain preserved.

Resume only specifically registered missing method/task identities, at most one
additional attempt each for this repair. Preserve all valid scores, the original
attempts and technical-check budget. The original queue lock, disk/memory checks,
three-worker limit and reference-control ownership still apply. This repair is
not a same-model causal intervention result or a reason to rerun valid zeroes.
