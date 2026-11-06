#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}
cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
RESROOT=${CAUSALACT_RUN_DIR}/osw_v4
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import json,os
rem=json.load(open(causalact_location('CAUSALACT_OSWORLD_DIR', 'shards/v4md_rem.json')))
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
# per domain, up to 3 workers
for dom,ids in rem.items():
    W=min(3,max(1,len(ids)))
    for w in range(3):
        json.dump({dom: ids[w::W] if w<W else []}, open(f"{root}/shards/v4r2_{dom}_{w}.json","w"))
print("shards built")
PY
: # Process and container cleanup is managed by the operator.
sleep 2
: # Process and container cleanup is managed by the operator.
i=0
for dom in chrome os vs_code libreoffice_writer thunderbird libreoffice_calc libreoffice_impress multi_apps; do
  for w in 0 1 2; do
    SH="shards/v4r2_${dom}_${w}.json"
    [ -s "$SH" ] || continue
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CIV_REPLAY=1 CIV_STREAK=4 CIV_CAP=3 CIV_BASE_ROOT=$BASE \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v4md_audit_${dom}_${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 4)) \
    nohup python run_gimp_v4.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir $RESROOT --test_all_meta_path $SH \
      > logs/v4r2_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i workers alive:$(pgrep -f run_gimp_v4|wc -l)"
