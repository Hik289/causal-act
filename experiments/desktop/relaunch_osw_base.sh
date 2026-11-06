#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
sleep 2
for w in 0 1 2 3 4 5 6 7; do
  CIV_LOOPBREAK=0 \
  nohup python run_gimp_lb2.py --provider_name docker --headless \
    --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" \
    --domain all --max_steps 12 --result_dir ${CAUSALACT_RUN_DIR}/osw_base_full \
    --test_all_meta_path shards/osw_base_shard_${w}.json \
    > logs/osw_base_r${w}.log 2>&1 &
  echo "w$w pid $!"
  sleep 3
done
echo "ALL RELAUNCHED workers=$(pgrep -f run_gimp_lb2|wc -l)"
