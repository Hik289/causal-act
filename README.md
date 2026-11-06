<div align="center">
  <h1>CausalAct: Controlled Visual Interventions for Reliable Agent Actions</h1>
  <p>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <a href="docs/reproduction.md"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-3776AB.svg"></a>
  </p>
  <p>Research code for testing and improving how visual agents respond to changes in their environment.</p>
</div>

> A visual agent should revise its action when the task state changes, ignore
> irrelevant appearance changes, and follow the same control when it moves.

<p align="center">
  <img src="assets/causalact-overview.png" width="96%" alt="A forum subscription task across four views. The original requires a click; an already-subscribed state requires stopping; a background change keeps the action; moving the control moves the click.">
</p>

## Method

1. **Intervene.** Change the task state, appearance, or layout in a controlled environment. Check that the change took effect and record the acceptable actions.
2. **Refine.** Train a lightweight model on paired observations and their action labels. If both states admit the same correct action, no switch is required.
3. **Gate.** Forward the base agent's proposed action unless the verifier authorizes a correction.

The release includes scripted interventions, refiner and verifier training, and benchmark integrations. See the [research note](docs/research-note.md) for the evaluation protocol and its scope.

## Quick start

The offline checks use the Python standard library. Live experiments need separate upstream environments and local configuration.

```bash
git clone https://github.com/Hik289/causal-act.git
cd causal-act
bash scripts/check.sh
```

See the [reproduction guide](docs/reproduction.md) for benchmark setup, data generation, and training.

## Repository scope

| Path | Contents |
| :--- | :--- |
| [`experiments/web/`](experiments/web/) | Forum interventions, environment checks, metrics, and VisualWebArena runners |
| [`experiments/desktop/`](experiments/desktop/) | OSWorld outcome verifiers, shared-prefix replay, and gate variants |
| [`experiments/driving/`](experiments/driving/) | CARLA setup, development routes, and a Base smoke controller |
| [`integrations/`](integrations/) | Patches against recorded upstream revisions |
| [`scripts/`](scripts/) | Offline checks, benchmark integration, and figure rendering |

The original CausalAct code is released under the [MIT License](LICENSE). Upstream benchmark code, patches, and assets remain subject to their respective terms.

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
