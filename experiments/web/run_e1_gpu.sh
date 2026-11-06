#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
MODE=$1   # ba | base
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/e1_${MODE}
LOG_DIR=$VWA_DIR/logs/e1_${MODE}
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
if [ "$MODE" = "ba" ]; then export CIV_REFINER_URL=http://127.0.0.1:18900/refine; fi
# 3 worker, 各绑 GPU 0/1/2 跑 BLIP-2 eval captioner (避免 CPU 超订); refiner 在 GPU3
RANGES=("0 70" "70 140" "140 210")
g=0
for r in "${RANGES[@]}"; do
  set -- $r; S=$1; E=$2
  CUDA_VISIBLE_DEVICES=$g nohup bash -c "until python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $S --test_end_idx $E --result_dir '$RESULT_DIR' --test_config_base_dir=config_files/vwa/test_reddit --model \"$CAUSALACT_BASE_MODEL\" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 10; do echo crashed; sleep 2; done; echo DONE_$S" > "$LOG_DIR/w_$S.log" 2>&1 &
  echo "$MODE w[$S $E) GPU$g pid=$!"; g=$((g+1)); sleep 3
done
