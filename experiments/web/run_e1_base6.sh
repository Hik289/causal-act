#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/e1_base
LOG_DIR=$VWA_DIR/logs/e1_base
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
# 6 worker, 2 share each GPU 0/1/2 (BLIP eval 偶发用 GPU, 多数时间等 proxy)
RANGES=("0 35" "35 70" "70 105" "105 140" "140 175" "175 210")
GPUS=(0 1 2 0 1 2)
i=0
for r in "${RANGES[@]}"; do
  set -- $r; S=$1; E=$2; g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup bash -c "until python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $S --test_end_idx $E --result_dir '$RESULT_DIR' --test_config_base_dir=config_files/vwa/test_reddit --model \"$CAUSALACT_BASE_MODEL\" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 10; do echo crashed; sleep 2; done; echo DONE_$S" > "$LOG_DIR/x_$S.log" 2>&1 &
  echo "base w[$S $E) GPU$g pid=$!"; i=$((i+1)); sleep 3
done
