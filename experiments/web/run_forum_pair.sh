#!/usr/bin/env bash
set -euo pipefail
: "${CAUSALACT_VWA_DIR:?Run with scripts/run-experiment.sh}"
: "${OPENAI_API_KEY:?Set your model credential}"
: "${CIV_WEB_VGATE_URL:?Start the web outcome verifier and set its score URL}"
tasks="$(cd -- "$1" && pwd)"
mkdir -p "$2"
out="$(cd -- "$2" && pwd)"
export CAUSALACT_FORUM_RESUME=1
cd "$CAUSALACT_VWA_DIR"
args=(
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json
  --test_start_idx 0 --test_end_idx "${CAUSALACT_FORUM_COUNT:-170}" --test_config_base_dir "$tasks"
  --model "$CAUSALACT_BASE_MODEL" --action_set_tag som --observation_type image_som --max_steps 10
  --eval_captioning_model_device cpu
)
env -u CIV_WEB_VGATE_URL "${PYTHON:-python3}" -u run.py "${args[@]}" --result_dir "$out/base" >>"$out/base.log" 2>&1
CIV_WEB_VTAU=0.9 CIV_WEB_VAUDIT="$out/gate.log" \
  "${PYTHON:-python3}" -u run.py "${args[@]}" --result_dir "$out/causalact" >>"$out/causalact.log" 2>&1
