#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
mkdir -p logs shards
for i in 0 1 2 3 4 5; do
  export CIV_LOOPBREAK=1 CIV_CAP=2
  export CIV_AUDIT=${CAUSALACT_RUN_DIR}/gimp_bav2_audit_w${i}.log
:
  export SCREEN_WIDTH=1280 SCREEN_HEIGHT=720
  nohup python run_gimp_lb2.py --provider_name docker --headless \
    --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" \
    --domain gimp --max_steps 15 \
    --result_dir ${CAUSALACT_RUN_DIR}/desktop_bav2_gimp_results \
    --test_all_meta_path shards/gimp_shard_${i}.json \
    > logs/gimp_bav2_w${i}.log 2>&1 &
  echo "launched w$i pid $!"
  sleep 4
done
echo "ALL LAUNCHED"
wait
