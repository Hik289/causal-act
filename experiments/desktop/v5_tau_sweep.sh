#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}; cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
TAU=$1; TAG=$2
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
i=0
for dom in chrome libreoffice_impress os libreoffice_calc; do
  for w in 0 1; do
    SH="shards/v5val_${dom}_${w}.json"
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CIV_VGATE=1 CIV_VTAU=$TAU CIV_VSTREAK=2 CIV_CAP=3 CIV_BASE_ROOT=$BASE \
    CIV_VURL=http://127.0.0.1:18833/pfail \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v5${TAG}_audit_${dom}_${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 3)) \
    nohup python run_gimp_v5.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir ${CAUSALACT_RUN_DIR}/osw_v5${TAG} --test_all_meta_path $SH \
      > logs/v5${TAG}_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i workers tau=$TAU tag=$TAG alive:$(pgrep -f run_gimp_v5|wc -l)"
