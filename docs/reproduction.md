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

### Optional Bedrock adapter

The adapter translates Chat Completions requests into Bedrock Messages requests.
Set `AWS_BEARER_TOKEN_BEDROCK` to your Bedrock credential,
`CAUSALACT_BEDROCK_MODEL` to the model ID, and `OPENAI_API_KEY` to a separate
local client credential. Set `AWS_REGION` or `CAUSALACT_BEDROCK_PROFILE` if your
Bedrock configuration requires them. The adapter accepts up to eight
simultaneous upstream requests and binds to loopback.

```bash
python3 experiments/services/bedrock_relay.py .runs/bedrock
```

In the shell that launches the experiments, set `OPENAI_BASE_URL` to
`http://127.0.0.1:18822/v1` and use the same local client credential.
OSWorld and VisualWebArena can use their existing planner label with this
adapter, which sends every request to the configured Bedrock model.

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

### Run the 170-task forum comparison

Prepare the locally configured forum task JSON using the VisualWebArena setup.
The preparation command preserves the recorded task selection and order. Start
the outcome verifier with a trained web checkpoint. The same verifier server
used for desktop accepts a different checkpoint through `CIV_VCKPT`.

```bash
bash scripts/run-experiment.sh experiments/web/prepare_forum_pair.py \
  .vendor/VisualWebArena/config_files/vwa/test_reddit.json .runs/forum/tasks

CIV_VCKPT=/path/to/web/checkpoint CIV_VPORT=18834 \
  PYTHONPATH=experiments python3 experiments/desktop/v_server.py
```

In a separate shell with VisualWebArena and model access configured, run both
arms. Keep `REDDIT` and the local browser session configured for your checkout.

```bash
CIV_WEB_VGATE_URL=http://127.0.0.1:18834/pfail \
  bash scripts/run-experiment.sh experiments/web/run_forum_pair.sh \
  .runs/forum/tasks .runs/forum/comparison
```

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

### Run the fixed desktop replay comparisons

Start `experiments/desktop/v_server.py` in the verifier environment with
`CIV_VCKPT` set to the verifier checkpoint directory. The
[recorded checkpoint](../experiments/desktop/paired_replay/recorded_checkpoint.json)
identifies the weights selected in the original run. In the OSWorld environment,
run the timing comparison or select `--protocol calibration` for the earlier
three-arm comparison.

```bash
bash scripts/run-experiment.sh experiments/desktop/paired_replay/run_protocol.py \
  "$PWD/.runs/desktop/timing" --checkpoint /path/to/checkpoint/model.safetensors
```

Both use fixed 24-task manifests and a 12-step budget. The timing comparison
uses a separate held-out manifest and follows the recorded Base action schedule
up to the first intervention. Reuse the output directory to resume tasks
without a score. The report counts only tasks with scores in every arm.

The `run_agent_s2.py` and `run_agent_s3.py` entry points use the corresponding
Agent S package and OSWorld runner arguments. Set `CAUSALACT_GROUNDER_URL` to
a compatible UI-TARS service, and `CIV_S2_MEMORY_DIR` to a writable directory
for the S2 worker. Use separate result directories and memory folders for each
shard.

## Bench2Drive and CARLA

Install [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), CARLA 0.9.15,
the matching maps, and the required Python packages. Check out the commit listed
in [bench2drive.revision](../integrations/bench2drive.revision), apply the patch,
and start the simulator before running the base agent.

```bash
python3 scripts/install-integration.py bench2drive .vendor/Bench2Drive
bash scripts/run-experiment.sh experiments/driving/run_b2d.sh base
```

The command above runs the original short Base smoke test. To run the
paired protocol, activate the Bench2Drive environment and generate the splits
from the official route files.

```bash
python3 experiments/driving/paired/split.py \
  .vendor/Bench2Drive/leaderboard/data .runs/driving/splits

bash scripts/run-experiment.sh experiments/driving/paired/run.py \
  .runs/driving/splits .runs/driving/comparison
```

The paired runner starts its own simulator and uses a fixed controller in
both arms. The CausalAct arm adds a second model query to check stopping
evidence. The runner starts the disjoint 100-route holdout after all 20
validation routes have valid pairs and the mean paired Driving Score improves.
The recorded split includes the route IDs and XML hashes. Set
`CARLA_GRAPHICS_ADAPTER`, `CARLA_PORT`, or `CARLA_TM_PORT` to select local
simulator resources. Run this protocol only in CARLA.

## ScreenSpot-Pro

The two evaluators report point hits against the annotated bounding boxes,
grouped by target area. Install Pillow and `huggingface_hub` for the download
and the generic vision-model evaluator. The UI-TARS evaluator also needs the
model's image processor and its Transformers dependencies.

```bash
python3 experiments/grounding/download.py .runs/screenspot
bash scripts/run-experiment.sh experiments/grounding/point_eval.py .runs/screenspot
bash scripts/run-experiment.sh experiments/grounding/uitars_eval.py \
  /path/to/ui-tars-processor .runs/screenspot/dataset .runs/screenspot/uitars
```

Set `CAUSALACT_GROUNDER_URL` for the UI-TARS endpoint and
`CAUSALACT_GROUNDER_KEY` if it requires a credential. The UI-TARS evaluation
uses a 640-pixel maximum image edge. Both scripts keep their point predictions
locally and allow API errors to be retried.

## Files in this release

The release includes experiment code, benchmark patches, recorded result counts,
and the overview PNG. Generate screenshots, login sessions, benchmark caches,
VM disks, model checkpoints, and local logs in your own environment. Configure
model access locally. Each benchmark retains its own terms.
