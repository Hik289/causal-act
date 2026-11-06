#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# 4 并行 worker 切分 210 task (镜像官方 wa_parallel_run.sh 的 index-split 思路, 但不用 tmux)。
# BLIP-2 captioner 跑 CPU (page_image_query 的 28 task 需要; 458GB RAM 够多 worker)。
set -o pipefail
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/reddit_full_base
LOG_DIR=$VWA_DIR/logs/reddit_full
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
:
export SHOPPING=http://127.0.0.1:7770
export REDDIT=http://127.0.0.1:9999
export WIKIPEDIA=http://127.0.0.1:8888
export HOMEPAGE=http://127.0.0.1:4399
:
:
export causalact_USAGE_LOG="$RESULT_DIR/usage.jsonl"
# 确保 config 已生成
python scripts/generate_test_data.py > "$LOG_DIR/gen_test_data.log" 2>&1
N=210
WORKERS=4
CHUNK=$(( (N + WORKERS - 1) / WORKERS ))   # 53
echo "=== launch $WORKERS workers, chunk=$CHUNK, total=$N ===" | tee "$LOG_DIR/launch.log"
for ((w=0; w<WORKERS; w++)); do
  START=$(( w * CHUNK ))
  END=$(( START + CHUNK ))
  if [ "$END" -gt "$N" ]; then END=$N; fi
  if [ "$START" -ge "$N" ]; then break; fi
  echo "worker $w: [$START, $END)" | tee -a "$LOG_DIR/launch.log"
  # until-loop: 单 worker crash 自动重启 (官方做法), 已完成 task 会被 skip
  nohup bash -c "until python run.py \
      --viewport_width 1280 --viewport_height 720 \
      --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
      --test_start_idx $START --test_end_idx $END \
      --result_dir '$RESULT_DIR' \
      --test_config_base_dir=config_files/vwa/test_reddit \
      --model "$CAUSALACT_BASE_MODEL" --action_set_tag som --observation_type image_som \
      --max_steps 30; do echo 'crashed, retry' >&2; sleep 2; done; echo WORKER_${w}_DONE" \
    > "$LOG_DIR/worker_${w}.log" 2>&1 &
  echo "  pid=$!"
  sleep 2
done
echo "all workers launched" | tee -a "$LOG_DIR/launch.log"
