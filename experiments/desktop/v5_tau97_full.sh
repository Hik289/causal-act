#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}; cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
RESROOT=${CAUSALACT_RUN_DIR}/osw_v5t97
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import glob,os,json
base=causalact_location('CAUSALACT_RUN_DIR', 'osw_base_full/pyautogui/screenshot/gpt-5.6-sol')
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
doms="chrome os vs_code libreoffice_writer thunderbird libreoffice_calc libreoffice_impress multi_apps".split()
tot=0
for dom in doms:
    ids=sorted(os.path.basename(x) for x in glob.glob(f"{base}/{dom}/*") if os.path.exists(x+"/result.txt") and os.path.exists(x+"/traj.jsonl"))
    for w in range(3):
        json.dump({dom: ids[w::3]}, open(f"{root}/shards/v5t97_{dom}_{w}.json","w"))
    tot+=len(ids)
print("full",tot)
PY
pgrep -f v_server.py >/dev/null || { nohup bash start_vserver.sh >logs/v_server.log 2>&1 & sleep 40; }
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
i=0
for dom in chrome os vs_code libreoffice_writer thunderbird libreoffice_calc libreoffice_impress multi_apps; do
  for w in 0 1 2; do
    SH="shards/v5t97_${dom}_${w}.json"
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CIV_VGATE=1 CIV_VTAU=0.97 CIV_VSTREAK=2 CIV_CAP=3 CIV_BASE_ROOT=$BASE \
    CIV_VURL=http://127.0.0.1:18833/pfail \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v5t97_audit_${dom}_${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 3)) \
    nohup python run_gimp_v5.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir $RESROOT --test_all_meta_path $SH \
      > logs/v5t97_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i workers tau=0.97 FULL alive:$(pgrep -f run_gimp_v5|wc -l)"
