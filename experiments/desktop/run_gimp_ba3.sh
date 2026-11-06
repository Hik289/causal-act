#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -eo pipefail
cd ${CAUSALACT_OSWORLD_DIR}
:
export SCREEN_WIDTH=1280 SCREEN_HEIGHT=720
export CIV_LOOPBREAK=1 CIV_STREAK=3
export CIV_AUDIT=${CAUSALACT_RUN_DIR}/gimp_ba3_audit.log
python run_gimp_lb.py --provider_name docker --headless --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 15 --result_dir ${CAUSALACT_RUN_DIR}/desktop_ba3_gimp_results
