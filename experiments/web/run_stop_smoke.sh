#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/stop_smoke
mkdir -p "$RESULT_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
export CIV_STOP_REFINER_URL=http://127.0.0.1:18903/stop_check
export CIV_STOP_AUDIT=$RESULT_DIR/stop_audit.log
export CIV_LOOP_K=2
for T in 0 1 11 34; do E=$((T+1)); CUDA_VISIBLE_DEVICES=2 python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $T --test_end_idx $E --result_dir "$RESULT_DIR" --test_config_base_dir=config_files/vwa/test_reddit --model "$CAUSALACT_BASE_MODEL" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 10 >> "$RESULT_DIR/run.log" 2>&1; done
echo STOP_SMOKE_DONE >> "$RESULT_DIR/run.log"
