#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
export CIV_OUTCOME_GATE=1 CIV_CAP=2 CIV_AUDIT=${CAUSALACT_RUN_DIR}/gimp_og_audit.log
export CUDA_VISIBLE_DEVICES=2
exec python run_gimp_og.py --provider_name docker --headless --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 --result_dir ${CAUSALACT_RUN_DIR}/gimp_og_results
