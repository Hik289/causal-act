#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -eo pipefail
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/reddit_base_full
mkdir -p "$RESULT_DIR"
cd "$VWA_DIR"
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
export NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
:
export SHOPPING=http://127.0.0.1:7770
export REDDIT=http://127.0.0.1:9999
export WIKIPEDIA=http://127.0.0.1:8888
export HOMEPAGE=http://127.0.0.1:4399
:
:
python scripts/generate_test_data.py
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx 0 --test_end_idx 210 \
  --result_dir "$RESULT_DIR" \
  --test_config_base_dir=config_files/vwa/test_reddit \
  --model "$CAUSALACT_BASE_MODEL" \
  --action_set_tag som --observation_type image_som --max_steps 10
