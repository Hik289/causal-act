#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
W=$1
export CIV_LOOPBREAK=1 CIV_STREAK=4 CIV_CAP=3
export CIV_AUDIT=${CAUSALACT_RUN_DIR}/v3_audit_w${W}.log
:
export CUDA_VISIBLE_DEVICES=$((W % 4))
exec python run_gimp_v3.py --provider_name docker --headless --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 --result_dir ${CAUSALACT_RUN_DIR}/gimp_v3_results --test_all_meta_path shards/og_gimp_${W}.json
