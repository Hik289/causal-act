#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
for w in 0 1 2 3 4 5 6 7 8 9; do
  CIV_LOOPBREAK=1 CIV_CAP=2 CIV_AUDIT=${CAUSALACT_RUN_DIR}/ba_weak_audit_w${w}.log \
  nohup python run_gimp_lb2.py --provider_name docker --headless \
    --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" \
    --domain all --max_steps 12 --result_dir ${CAUSALACT_RUN_DIR}/osw_ba_weak_full \
    --test_all_meta_path shards/ba_weak_shard_${w}.json \
    > logs/ba_weak_w${w}.log 2>&1 &
  echo "w$w pid $!"; sleep 3
done
echo "ALL 10 BA WEAK LAUNCHED procs=$(pgrep -f run_gimp_lb2|wc -l)"
