#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/e2b_base_override
LOG_DIR=$VWA_DIR/logs/e2b_base_override
mkdir -p "$RESULT_DIR" "$LOG_DIR"
cd "$VWA_DIR"
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache NLTK_DATA=${CAUSALACT_RUN_DIR}/nltk_data
export DATASET=visualwebarena CLASSIFIEDS=http://127.0.0.1:9980
export SHOPPING=http://127.0.0.1:7770 REDDIT=http://127.0.0.1:9999 WIKIPEDIA=http://127.0.0.1:8888 HOMEPAGE=http://127.0.0.1:4399
:
# remaining 13 override tasks split GPU1/GPU2
G1=(9 13 36 114 169 196 198)
G2=(11 34 41 168 194 197)
run_set(){ local gpu=$1; shift; for T in "$@"; do E=$((T+1)); CUDA_VISIBLE_DEVICES=$gpu python run.py --viewport_width 1280 --viewport_height 720 --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json --test_start_idx $T --test_end_idx $E --result_dir "$RESULT_DIR" --test_config_base_dir=config_files/vwa/test_reddit --model "$CAUSALACT_BASE_MODEL" --action_set_tag som --observation_type image_som --eval_captioning_model_device cuda --max_steps 10 >> "$LOG_DIR/base_ov_g${gpu}.log" 2>&1; done; echo "G${gpu}_DONE" >> "$LOG_DIR/base_ov_g${gpu}.log"; }
run_set 1 "${G1[@]}" &
run_set 2 "${G2[@]}" &
wait
echo ALL_DONE
