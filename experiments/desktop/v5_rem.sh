#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}; cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import json
rem=json.load(open(causalact_location('CAUSALACT_OSWORLD_DIR', 'shards/v5full_rem.json')))
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
for dom,ids in rem.items():
    W=min(3,max(1,len(ids)))
    for w in range(3):
        json.dump({dom: ids[w::W] if w<W else []}, open(f"{root}/shards/v5rem_{dom}_{w}.json","w"))
PY
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
# ensure V server alive
pgrep -f v_server.py >/dev/null || { nohup bash ${CAUSALACT_OSWORLD_DIR}/start_vserver.sh >${CAUSALACT_OSWORLD_DIR}/logs/v_server.log 2>&1 & sleep 40; }
i=0
for dom in libreoffice_calc multi_apps chrome os vs_code libreoffice_writer thunderbird libreoffice_impress; do
  for w in 0 1 2; do
    SH="shards/v5rem_${dom}_${w}.json"; [ -s "$SH" ] || continue
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CIV_VGATE=1 CIV_VTAU=0.9 CIV_VSTREAK=2 CIV_CAP=3 CIV_BASE_ROOT=$BASE \
    CIV_VURL=http://127.0.0.1:18833/pfail \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v5full_audit_${dom}_r${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 3)) \
    nohup python run_gimp_v5.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir ${CAUSALACT_RUN_DIR}/osw_v5full --test_all_meta_path $SH \
      > logs/v5rem_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i workers alive:$(pgrep -f run_gimp_v5|wc -l) Vserver:$(pgrep -f v_server.py|wc -l)"
