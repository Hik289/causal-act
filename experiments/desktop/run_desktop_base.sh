#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -eo pipefail
cd ${CAUSALACT_OSWORLD_DIR}
:
:
export SCREEN_WIDTH=1280
export SCREEN_HEIGHT=720
python run.py \
  --provider_name docker \
  --headless \
  --observation_type screenshot \
  --action_space pyautogui \
  --model "$CAUSALACT_BASE_MODEL" \
  --domain chrome \
  --max_steps 15 \
  --result_dir ${CAUSALACT_RUN_DIR}/desktop_base_results
