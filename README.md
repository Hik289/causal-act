<div align="center">
  <h1>CausalAct: Controlled Visual Interventions for Reliable Agent Actions</h1>
  <p>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="docs/reproduction.md"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg"></a>
  </p>
  <p>Research code for testing how changes in a scene affect a visual agent's actions.</p>
</div>

> A visual agent should adjust its action when the task requires it, stay correct
> when unrelated details change, and find the same target when it moves.

<p align="center">
  <img src="assets/causalact-overview.png" width="96%" alt="A forum subscription task with three edits. Changing the subscription state changes whether to click. Changing the background preserves the action. Moving the button changes where to click.">
</p>

## Recent Commit

**November 19, 2026** — Added native SeeAct and existing Base adapters using
audited Claude requests and the official live-page evaluator. A resumable
170-task Forum queue freezes identical reference images and creates a fresh,
owner-checked environment for each method/task. It preserves valid negative
scores, rejects incomplete inference/evaluation, and reports paired denominators.
Browser routing keeps legacy absolute links inside each owned environment.
Thirty-two interface/queue checks and reference-image fixtures cover these
paths; preparation and fixture results are not benchmark scores. See the
[adapter protocol](experiments/web/README_seeact.md).
A recorded runtime repair also retries a navigation-invalidated observation
before any model call, preserving existing scores and the original action budget.

**November 19, 2026** — Added a controlled forum component study with a
forum-level train/validation/test split, 100 held-out cases, backend-verified
edits, and two rounds of separate A, V, and I adapter training. The study includes
ten configurations and records full module parameter counts, validation-selected
checkpoints, raw predictions, and metric denominators. I selects from three
validated edit programs; these component scores do not establish full benchmark
success or cross-environment transfer. See the [study protocol](experiments/training/README.md).

A separate validation-gated replication now audits the action-label sampling
prior, applies one training-derived class-weight correction, and reserves new
forums for the original/adjusted checkpoint comparison. Original scores remain
intact; the followup makes no claim of an improved held-out result.

An inference-only followup also freezes the original Full/A+V checkpoints and
selects V's fusion weight using validation cases only. Its original/calibrated
comparison uses preselected, disjoint forums after a strict validation gate.
The original results remain visible, including any negative or mixed findings;
a zero fusion weight is explicitly a disabled-V setting.
An optional scheduling amendment starts registered followups on idle cards
while preserving original-job priority, memory limits, and the shared limit
of four local GPU workers.
Admission also checks actual GPU compute processes before claiming an idle card.

Agent S2 and S3 also require an explicit UI-TARS 1.5
processed-image frame. The frame is recorded from the deployed processor before
evaluation, and earlier coordinate protocols retain separate results. S2 keeps
native image bytes for OCR. Runtime repairs also cover malformed DAG replies,
desktop readiness, cached evaluator dependencies, and oversized Web images.
Screenshot-only observations preserve the official evaluator's access to
accessibility trees; a worker check verifies that these paths remain separate.
An optional offline install keeps the official evaluator wheel and checks its
version before scoring. An initial-frame check can reject known unrendered
desktops on the actual first model input, after the runner's existing readiness
wait and fresh capture, retaining the original image for audit.
Validation includes twenty-seven unit tests, both Agent S backends, independent
synthetic geometry checks, real OCR, the original Calc file and evaluator
dependency command, and the previously rejected Web images sent to Bedrock.
[Commit history](https://github.com/Hik289/causal-act/commits/main/)

## How it works

Suppose the task is to subscribe to a forum. The agent should click **Subscribe**
if the user has not subscribed yet. If the user is already subscribed, it should
stop. Changing the page background should preserve that decision. Moving the
button should move the click to its new location.

CausalAct turns these cases into training examples and tests.

1. **Change the scene.** Edit the task state, appearance, or layout in the browser or simulator. Check that the edit worked and record which actions are correct.
2. **Train the refiner.** Train a small model to choose a correct action for each version of the scene. The same action can be correct in both versions.
3. **Check the proposed action.** An optional verifier decides when to allow a correction to the base agent's action.

The code includes scripts that create these edits, train the refiner and verifier,
and connect them to the benchmarks. The [research note](docs/research-note.md)
explains the method and the work completed so far.

## Quick start

Clone the repository and run the checks. They use the Python standard library
and work without a running benchmark.

```bash
git clone https://github.com/Hik289/causal-act.git
cd causal-act
bash scripts/check.sh
```

To set up benchmarks, generate data, or train a model, follow the
[reproduction guide](docs/reproduction.md).

## Code and documentation

| Path | Contents |
| :--- | :--- |
| [`experiments/web/`](experiments/web/) | Forum edits, state checks, evaluation metrics, and VisualWebArena runs |
| [`experiments/desktop/`](experiments/desktop/) | OSWorld runs, verifier training, and experiments that replay a base run before testing a correction |
| [`experiments/driving/`](experiments/driving/) | CARLA setup, a base-agent smoke test, and paired Bench2Drive evaluation |
| [`experiments/grounding/`](experiments/grounding/) | ScreenSpot-Pro point evaluation for a vision model and UI-TARS |
| [`integrations/`](integrations/) | Changes needed to connect this code to specific versions of the benchmarks |
| [`scripts/`](scripts/) | Repository checks, setup tools, and an alternative overview diagram |

The original CausalAct code uses the [MIT License](LICENSE). Benchmark code,
patches, and assets are covered by the terms of their source projects.

## Citation

```bibtex
@misc{causalact2026,
  title        = {{CausalAct}: Controlled Visual Interventions for Reliable Agent Actions},
  author       = {{CausalAct Contributors}},
  year         = {2026},
  howpublished = {Research software},
  url          = {https://github.com/Hik289/causal-act}
}
```
