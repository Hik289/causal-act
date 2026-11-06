#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/reddit_full_base
LOG_DIR=$VWA_DIR/logs/reddit_full
cd "$VWA_DIR"
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980 SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
export causalact_USAGE_LOG="$RESULT_DIR/usage.jsonl"
# finer sub-ranges over the 3 slow chunks; get_unfinished skips done tasks
RANGES=("0 18" "18 36" "36 53" "106 123" "123 141" "141 159" "159 176" "176 193" "193 210")
i=10
for r in "${RANGES[@]}"; do
  set -- $r; S=$1; E=$2
  nohup bash -c "until python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $S --test_end_idx $E --result_dir '$RESULT_DIR' --test_config_base_dir=config_files/vwa/test_reddit --model \"$CAUSALACT_BASE_MODEL\" --action_set_tag som --observation_type image_som --max_steps 30; do echo crashed; sleep 2; done; echo ADDW_${i}_DONE" > "$LOG_DIR/addw_${i}.log" 2>&1 &
  echo "added worker $i: [$S $E) pid=$!"
  i=$((i+1))
  sleep 2
done
