# Native WebVoyager Forum adaptation

The upstream repository is [MinorJerry/WebVoyager](https://github.com/MinorJerry/WebVoyager),
fixed at `5a7896738c10bfb8b9edccce6bb0e0411f8ae569`. The adapter executes its
`run.py` main loop directly. Its system prompt, set-of-mark extraction, action
parser, context clipping, action execution and waits remain upstream code.
Native source files are hashed and remain unedited.

This configuration uses the existing paced Claude Opus 5.5 transport. It is a
Claude adaptation, not the paper's original GPT-4V setting. Native planning
keeps the upstream 1,000-token budget, the recent three screenshots, fixed box
color, and ten benchmark decisions. The native seed argument is 42; actual
sampling follows the unchanged shared relay, which does not forward that seed.
All task reference images use the same frozen canonical inputs and large-image
transport policy as the new Base/SeeAct controls. Each planning request must
contain every required reference image.

The native browser runs Selenium 4.15.2 with a pinned ChromeDriver116 and the
same Chromium116 binary used by the controls. Its actual inner viewport is
1280×720 at device scale1, explicitly verified before scoring. Upstream's
headless user-agent string remains unchanged. The isolated Python environment
inherits the deployed OpenAI1.3.5, Pillow10.0.1 and Playwright1.37 packages;
these differ from upstream's OpenAI1.1.1/Pillow10.1.0 requirements and are
recorded as environment differences. Browser and driver binaries have SHA256
identities and never download replacements during evaluation.

Every case receives a new owner-labeled container from the same clean Forum
seed as the new Base controls, plus official authentication state. A separate
CDP client services origin routing while the native Selenium loop runs. It is
installed before the first task navigation; legacy links, subresources and API
requests are served by that case's owned environment. Routing errors invalidate
the attempt. The official VWA evaluator attaches to the exact live Selenium
page before browser cleanup, preserving unsaved DOM state.

Native observation extraction failures, uncompleted model requests and missing
official evaluations do not become task zeros. The native PDF Assistant API
has no declared Claude equivalent; encountering it is an invalid attempt,
with original logs retained. Policy action/format mistakes and genuine
negative official scores remain valid when inference and evaluation complete.

`prepare_webvoyager_queue.py` freezes a separate 170-task, one-worker queue.
It requires a passed real-Claude reference/routing/DOM fixture and hash-checked
native/browser/driver sources. It references an existing new-initial-state
Base protocol with identical original tasks, image inputs, seed, viewport,
model and decision budget. It never submits Base tasks. Only already-valid
Base raw files contribute to the paired intersection, and their bytes become
protected alongside new WebVoyager scores. Historical Base/CausalAct scores
remain separate.

Run the prepared queue using `forum_pair_queue.py --protocol` with its absolute
protocol path. Existing valid results are never re-run; invalid attempts use
the original bounded retry/backoff policy. The one new worker plus the existing
two Base/SeeAct workers respects the three-worker Forum/OS host limit, memory
and disk floors, and the shared Bedrock8-inflight/8-starts-per6-seconds gateway.
Per-method and paired denominators are distinct. A full pair report requires
all170 valid intersections; finishing only the external method waits for its
separately owned Base controls without resubmitting them.

`webvoyager_fixture.py --scripted` tests the real native browser loop with zero
model requests. Its real-Claude mode tests reference delivery, native actions,
owned-origin fetches and official evaluation of retained DOM state. The
fixtures use a negative synthetic task ID, create no benchmark `result.json`,
and are excluded from all experimental denominators.
