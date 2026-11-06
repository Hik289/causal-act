#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}
RES=${CAUSALACT_RUN_DIR}/gimp_v4_results/pyautogui/screenshot/gpt-5.6-sol/gimp
cd "$ROOT"
# remove partial (no result.txt) dirs so they re-run
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
sleep 2
: # Process and container cleanup is managed by the operator.
for w in 0 1 2 3 4 5; do
  CIV_REPLAY=1 CIV_STREAK=4 CIV_CAP=3 \
  CIV_BASE_ROOT=${CAUSALACT_RUN_DIR}/desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol \
  CIV_AUDIT=${CAUSALACT_RUN_DIR}/v4_audit_w${w}.log \
  CUDA_VISIBLE_DEVICES=$((w % 4)) \
  nohup python run_gimp_v4.py \
    --provider_name docker --headless --observation_type screenshot \
    --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 \
    --result_dir ${CAUSALACT_RUN_DIR}/gimp_v4_results \
    --test_all_meta_path shards/v4r_${w}.json \
    > logs/v4r_w${w}.log 2>&1 &
  echo "w$w pid $!"
  sleep 10
done
echo "alive:$(pgrep -f run_gimp_v4|wc -l)"
