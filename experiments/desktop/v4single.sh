#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
export CIV_REPLAY=1 CIV_STREAK=4 CIV_CAP=3
export CIV_BASE_ROOT=${CAUSALACT_RUN_DIR}/desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol
export CIV_AUDIT=${CAUSALACT_RUN_DIR}/v4_audit_single.log
export CUDA_VISIBLE_DEVICES=0
exec python run_gimp_v4.py --provider_name docker --headless --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 --result_dir ${CAUSALACT_RUN_DIR}/gimp_v4_results --test_all_meta_path shards/v4_single.json
