# Setup and experiments

Run the commands below from the repository root. The repository checks need
Python 3.9 or newer. Benchmark runs need their own environments, and the training
scripts expect a CUDA GPU. For CARLA 0.9.15, use the Python version and
dependencies specified by Bench2Drive.

## Run the repository checks

```bash
bash scripts/check.sh
```

This checks the format of intervention records, the action metrics, and the
calculations in the results summary. Running benchmarks or training models
requires the setup below.

## Configure local paths and model access

Create a separate clone for each benchmark. The experiment launcher finds files
relative to this repository, even when you call it from another directory.
Use these environment variables to change the paths or model settings.

| Variable | Default or purpose |
| :--- | :--- |
| `CAUSALACT_RUN_DIR` | `.runs/` stores generated data, screenshots, results, and model checkpoints |
| `CAUSALACT_VWA_DIR` | `.vendor/VisualWebArena` |
| `CAUSALACT_VWA_AUTH` | `.auth/reddit_state.json` inside the VisualWebArena checkout stores the local forum login state |
| `CAUSALACT_OSWORLD_DIR` | `.vendor/OSWorld` |
| `CAUSALACT_BENCH2DRIVE_DIR` | `.vendor/Bench2Drive` |
| `CAUSALACT_CARLA_DIR` | `.vendor/CARLA` |
| `CAUSALACT_BASE_MODEL` | The recorded model label is `gpt-5.6-sol`. Set a model supported by your service. |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Set your own API credential and service URL as needed. The short driving test requires an explicit service URL. |

The original runs used a compatible model service with the recorded label.
Choose a service and model available in your setup. Training and data
generation need the base service only when the selected script calls it.

## VisualWebArena and forum tests

1. Follow the [VisualWebArena setup](https://github.com/web-arena-x/visualwebarena)
   to create a local forum, set the site variables and any required reset
   credentials, and install dependencies. Log in to create a fresh local
   browser session.
2. Check out the commit listed in
   [visualwebarena.revision](../integrations/visualwebarena.revision), then
   apply the CausalAct changes.

   ```bash
   python3 scripts/install-integration.py visualwebarena .vendor/VisualWebArena
   ```

3. Install Playwright and Chromium. Use a disposable forum instance for the
   following commands because they change subscription state. Run the state
   checks (called **oracles** in the code), then generate the targeted examples.

   ```bash
   bash scripts/run-experiment.sh experiments/web/causalact/oracle/validate_oracles.py
   bash scripts/run-experiment.sh experiments/web/causalact/data/build_target_set.py
   ```

   The generated JSONL records include paths to the generated images.

4. In a CUDA environment, install
   [requirements-training.txt](../requirements-training.txt). Compare training
   on original states, training on both states, and training on both states with
   an extra penalty for choosing an action that has become wrong.

   ```bash
   bash scripts/run-experiment.sh experiments/web/causalact/modules/train_a_ltarget.py \
     --mode lact_only --out .runs/web/checkpoints/original-only
   bash scripts/run-experiment.sh experiments/web/causalact/modules/train_a_ltarget.py \
     --mode lact_both --out .runs/web/checkpoints/paired
   bash scripts/run-experiment.sh experiments/web/causalact/modules/train_a_ltarget.py \
     --mode lact_ltarget --out .runs/web/checkpoints/paired-margin
   ```

The [web folder](../experiments/web/) also has scripts for unrelated edits,
navigation examples, model services, and experiment runs. Some older launchers
expect multiple GPUs and the model and benchmark setup used in the original
runs. Set the GPU visibility, task range, and service URLs for your machine.

When comparing how the base and refiner select a target, give them the same
screenshots, numbered marks, and candidate descriptions.

## OSWorld

Follow the [OSWorld setup](https://github.com/xlang-ai/OSWorld) to create a
disposable VM or Docker environment. Check out the commit listed in
[osworld.revision](../integrations/osworld.revision), then apply the patch and
run the short setup check.

```bash
python3 scripts/install-integration.py osworld .vendor/OSWorld
bash scripts/run-experiment.sh experiments/desktop/osworld_smoke.py
```

The [desktop folder](../experiments/desktop/) includes base-agent runs, scripts
that collect screenshots and outcome labels, verifier training, and replay
experiments. It also includes task lists split into groups for separate workers.
Use data from your own runs to train the outcome verifier. Older replay
launchers expect saved base runs and, where used, a running local verifier.
Adjust their settings to your setup.

For independent runs, compare the same task IDs and count both failed tasks that
become successful and successful tasks that become failures. For replay runs,
record the step where the correction starts and keep the saved base run up to
that step. State explicitly when a result is calculated by reusing base outcomes
on tasks where no correction was made. The [results report](results.md) explains
both calculations.

## Bench2Drive and CARLA

Install [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), CARLA 0.9.15,
the matching maps, and the required Python packages. Check out the commit listed
in [bench2drive.revision](../integrations/bench2drive.revision), apply the patch,
and start the simulator before running the base agent.

```bash
python3 scripts/install-integration.py bench2drive .vendor/Bench2Drive
bash scripts/run-experiment.sh experiments/driving/run_b2d.sh base
```

This runs a short test of the base driving agent. Comparing CausalAct driving
corrections requires further implementation and evaluation.

## Files in this release

The release includes experiment code, benchmark patches, recorded result counts,
and the overview PNG. Generate screenshots, login sessions, benchmark caches,
VM disks, model checkpoints, and local logs in your own environment. Configure
model access locally. Each benchmark retains its own terms.
