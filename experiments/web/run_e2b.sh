#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# E2b: B+A (nav V-gated) OR B (base) on forum-completable subset (0-210), official SoM, max_steps=10
MODE=$1   # ba | base
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/e2b_${MODE}
LOG_DIR=$VWA_DIR/logs/e2b_${MODE}
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
if [ "$MODE" = "ba" ]; then
  export CIV_NAV_REFINER_URL=http://127.0.0.1:18902/nav_vgate
  export CIV_NAV_AUDIT=$RESULT_DIR/nav_audit.log
fi
# 2 workers on GPU 1,2 (GPU0=nav service, GPU3=old service). BLIP captioner on assigned GPU.
RANGES=("0 105" "105 210")
GPUS=(1 2)
i=0
for r in "${RANGES[@]}"; do
  set -- $r; S=$1; E=$2; g=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$g nohup bash -c "until python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $S --test_end_idx $E --result_dir '$RESULT_DIR' --test_config_base_dir=config_files/vwa/test_reddit --model \"$CAUSALACT_BASE_MODEL\" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 10; do echo crashed; sleep 2; done; echo DONE_$S" >> "$LOG_DIR/w_$S.log" 2>&1 &
  echo "$MODE w[$S $E) GPU$g pid=$!"; i=$((i+1)); sleep 3
done
echo LAUNCHED
