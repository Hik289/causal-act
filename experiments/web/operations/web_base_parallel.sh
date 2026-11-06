#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -eo pipefail
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web_base_traj
mkdir -p "$RESULT_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
export NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena
export REDDIT=http://127.0.0.1:9999
export CLASSIFIEDS=http://127.0.0.1:9980
:
export SHOPPING=http://127.0.0.1:7770
export WIKIPEDIA=http://127.0.0.1:8888
export HOMEPAGE=http://127.0.0.1:4399
:
:
S=$1; E=$2; W=$3
export CUDA_VISIBLE_DEVICES=$((W % 4))
exec python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx $S --test_end_idx $E \
  --result_dir "$RESULT_DIR" \
  --test_config_base_dir=config_files/vwa/test_reddit \
  --model "$CAUSALACT_BASE_MODEL" \
  --action_set_tag som --observation_type image_som --max_steps 10 \
  --eval_captioning_model_device cuda \
  --save_trace_enabled
