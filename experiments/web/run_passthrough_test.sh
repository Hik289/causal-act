#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/pt_test
mkdir -p "$RESULT_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
export CIV_REFINER_URL=http://127.0.0.1:18900/refine
export CIV_PT_LOG=$RESULT_DIR/pt_actions.log
# navigate tasks (0-3): base never clicks subscribe → refiner must全透传 → all IDENT
CUDA_VISIBLE_DEVICES=1 python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx 0 --test_end_idx 3 --result_dir "$RESULT_DIR" --test_config_base_dir=config_files/vwa/test_reddit --model "$CAUSALACT_BASE_MODEL" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 6
echo PT_DONE
