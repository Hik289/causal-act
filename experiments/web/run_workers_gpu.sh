#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/reddit_full_base
LOG_DIR=$VWA_DIR/logs/reddit_full
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980 SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
export causalact_USAGE_LOG="$RESULT_DIR/usage.jsonl"
# 4 worker 各绑一张 GPU 跑 BLIP-2 (避免 CPU 超订); 覆盖全 0-210, get_unfinished 跳过已完成 81 个
RANGES=("0 53" "53 106" "106 159" "159 210")
g=0
for r in "${RANGES[@]}"; do
  set -- $r; S=$1; E=$2
  CUDA_VISIBLE_DEVICES=$g nohup bash -c "until python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $S --test_end_idx $E --result_dir '$RESULT_DIR' --test_config_base_dir=config_files/vwa/test_reddit --model \"$CAUSALACT_BASE_MODEL\" --action_set_tag som --observation_type image_som --max_steps 30; do echo crashed; sleep 2; done; echo GPUW_${g}_DONE" > "$LOG_DIR/gpuw_${g}.log" 2>&1 &
  echo "gpu worker $g: [$S $E) CUDA=$g pid=$!"
  g=$((g+1))
  sleep 3
done
