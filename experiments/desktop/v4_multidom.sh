#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}
cd "$ROOT"
BASE=${CAUSALACT_RUN_DIR}/osw_base_full/pyautogui/screenshot/gpt-5.6-sol
RESROOT=${CAUSALACT_RUN_DIR}/osw_v4
mkdir -p "$RESROOT"
# domains with good base coverage; recoverable-first order
DOMS="chrome os vs_code libreoffice_writer thunderbird libreoffice_calc libreoffice_impress multi_apps"
# build per-(domain,worker) shards: for each domain split its base-completed task ids
python3 - <<PY
from causalact_runtime import location as causalact_location
import glob,os,json
base="$BASE"; root="$ROOT"
doms="$DOMS".split()
WPER=3  # workers per domain
meta={}
for dom in doms:
    ids=[os.path.basename(x) for x in glob.glob(f"{base}/{dom}/*") if os.path.exists(x+"/result.txt") and os.path.exists(x+"/traj.jsonl")]
    for w in range(WPER):
        json.dump({dom: ids[w::WPER]}, open(f"{root}/shards/v4md_{dom}_{w}.json","w"))
    meta[dom]=len(ids)
json.dump(meta, open(f"{root}/shards/v4md_meta.json","w"))
print("domains:",meta)
PY
: # Process and container cleanup is managed by the operator.
sleep 2
: # Process and container cleanup is managed by the operator.
i=0
for dom in $DOMS; do
  for w in 0 1 2; do
    CIV_REPLAY=1 CIV_STREAK=4 CIV_CAP=3 \
    CIV_BASE_ROOT=$BASE \
    CIV_AUDIT=${CAUSALACT_RUN_DIR}/v4md_audit_${dom}_${w}.log \
    CUDA_VISIBLE_DEVICES=$((i % 4)) \
    nohup python run_gimp_v4.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir $RESROOT \
      --test_all_meta_path shards/v4md_${dom}_${w}.json \
      > logs/v4md_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i workers alive:$(pgrep -f run_gimp_v4|wc -l)"
