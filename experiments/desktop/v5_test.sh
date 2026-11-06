#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}; cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
# FROZEN operating point from validation: tau=0.9, vstreak=2, cap=3, ep2
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import json
test=json.load(open(causalact_location('CAUSALACT_OSWORLD_DIR', 'shards/v5_test.json')))
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
for dom,ids in test.items():
    for w in range(2):
        json.dump({dom: ids[w::2]}, open(f"{root}/shards/v5test_{dom}_{w}.json","w"))
print("test sizes",{k:len(v) for k,v in test.items()})
PY
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
i=0
for dom in chrome libreoffice_impress os libreoffice_calc; do
  for w in 0 1; do
    SH="shards/v5test_${dom}_${w}.json"
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CIV_VGATE=1 CIV_VTAU=0.9 CIV_VSTREAK=2 CIV_CAP=3 CIV_BASE_ROOT=$BASE \
    CIV_VURL=http://127.0.0.1:18833/pfail \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v5test_audit_${dom}_${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 3)) \
    nohup python run_gimp_v5.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir ${CAUSALACT_RUN_DIR}/osw_v5test --test_all_meta_path $SH \
      > logs/v5test_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i TEST workers (tau=0.9 FROZEN) alive:$(pgrep -f run_gimp_v5|wc -l)"
