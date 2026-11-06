<div align="center">
  <h1>CausalAct</h1>
  <p><strong>Controlled Visual Interventions for Reliable Agent Actions</strong></p>
  <p>
    <a href="docs/research-note.md"><img alt="Research note" src="https://img.shields.io/badge/Research-Note-245ca7?style=flat-square"></a>
    <a href="#experiment-results"><img alt="Experiment results" src="https://img.shields.io/badge/Experiment-Results-236451?style=flat-square"></a>
    <a href="docs/reproduction.md"><img alt="Reproduction guide" src="https://img.shields.io/badge/Reproduction-Guide-67479c?style=flat-square"></a>
    <a href="https://github.com/Hik289/causal-act/archive/refs/heads/main.zip"><img alt="Download source" src="https://img.shields.io/badge/Download-Source-9b421e?style=flat-square"></a>
  </p>
  <p>Research code for testing when visual agents should revise an action and when they should leave it alone.</p>
</div>

<p align="center">
  <img src="assets/causalact-overview.png" width="100%" alt="CausalAct compares an original forum page with targeted, unrelated, and layout interventions. An environment checker labels valid actions; a refiner learns corrections; a verifier gates whether to apply them.">
</p>

## At a glance

| Question | What this release contains |
| :--- | :--- |
| **Does the agent react to the right evidence?** | Environment-validated subscription interventions, appearance controls, and action metrics. |
| **Can a small refiner fix the base agent?** | Action-refiner and verifier training, online integrations, and both positive and negative outcomes. |
| **Where can I start?** | Run the [offline checks](#quick-start), then follow the [reproduction guide](docs/reproduction.md). |

## Method

1. **Intervene.** Change the application state, appearance, or layout in a controlled environment. Check that the edit actually took effect and label the acceptable actions.
2. **Refine.** Train a lightweight model on the original and edited observations. A relevant change may invalidate the old action; a purely decorative edit should not. A layout edit can move the click without changing its target.
3. **Gate.** Let the base agent act by default. Apply a correction when the gate fires; measure both rescued tasks and tasks that the override hurts.

The released experiments use scripted environment edits. Training a learned editor and running the full multi-round expert iteration loop remain research directions.

## Experiment results

| Setting | Base / control | CausalAct or follow-up | What the comparison supports |
| :--- | ---: | ---: | :--- |
| Forum subscription, both states correct | 0/48 | **48/48** | Held-out forums. Equal-weight supervision on both states also reaches 48/48; the tested data intervention accounts for the gain. |
| VisualWebArena forum, eligible Base tasks | 60/161 (37.3%) | — | Single-site baseline. The full VisualWebArena benchmark was not completed. |
| VisualWebArena forum, live navigation gate | — | **−4.1 pp** | Seven harmed tasks, zero rescued in the override rerun. Offline grounding gains did not transfer. |
| OSWorld, independent matched rollouts | 75/296 (25.3%) | **58/296 (19.6%)** | **−5.7 pp** under a shared threshold and a 12-step budget. |
| OSWorld, passthrough accounting | 76/300 (25.3%) | **92/300 (30.7%)** | **+5.3 pp** after reusing the Base outcome on untouched tasks and counting 18 rescues and two harms under per-domain settings. This is a reconstruction, not a fresh end-to-end rollout. |
| Bench2Drive, Base smoke route | Driving Score 0.00879 | — | One logged route; the vehicle was blocked. No tested CausalAct comparison. |

See the [results audit](docs/results.md) for denominators, provenance, and limitations. The strongest supported finding is the controlled forum result. The desktop results answer different questions and should be reported together.

## Quick start

The offline checks use the Python standard library and do not require a simulator, a model, or credentials.

```bash
git clone https://github.com/Hik289/causal-act.git
cd causal-act
bash scripts/check.sh
python3 scripts/summarize-results.py
```

Live benchmark integration needs separate upstream checkouts and environments. Install the [recorded patches](integrations/) only into clean benchmark checkouts and supply local configuration as described in the [reproduction guide](docs/reproduction.md).

## Repository map

| Path | Contents |
| :--- | :--- |
| [`experiments/web/`](experiments/web/) | Intervention builders, browser state checks, causal metrics, training, and VisualWebArena runners |
| [`experiments/desktop/`](experiments/desktop/) | OSWorld outcome verifiers, replay, gate variants, and task splits |
| [`experiments/driving/`](experiments/driving/) | CARLA setup, route and Base smoke controller |
| [`integrations/`](integrations/) | Patches against the recorded upstream revisions |
| [`results/`](results/) | Aggregate counts and diagnostic metrics |
| [`docs/`](docs/) | Research note, results audit, and reproduction guide |

Model weights, VM images, raw browser and desktop captures, session files, and execution logs are generated or obtained separately. Benchmark assets and patches remain subject to their upstream terms.
