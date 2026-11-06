#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}; cd "$ROOT"
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
# re-run BASE (no refiner) on validated domains, fresh result dir = seed2
# measures how many tasks flip purely from env nondeterminism (base-vs-base)
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import glob,os,json
base=causalact_location('CAUSALACT_RUN_DIR', 'osw_base_full/pyautogui/screenshot/gpt-5.6-sol')
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
for dom in ["chrome","libreoffice_impress","os","libreoffice_calc"]:
    ids=sorted(os.path.basename(x) for x in glob.glob(f"{base}/{dom}/*") if os.path.exists(x+"/result.txt"))
    for w in range(3): json.dump({dom: ids[w::3]}, open(f"{root}/shards/base2_{dom}_{w}.json","w"))
print("base-seed2 shards built")
PY
i=0
for dom in chrome libreoffice_impress os libreoffice_calc; do
  for w in 0 1 2; do
    SH="shards/base2_${dom}_${w}.json"
    cnt=$(python3 -c "import json;print(len(json.load(open('$SH')).get('$dom',[])))")
    [ "$cnt" -gt 0 ] || continue
    CUDA_VISIBLE_DEVICES=$((i % 3)) \
    nohup python run.py \
      --provider_name docker --headless --observation_type screenshot \
      --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain $dom --max_steps 12 \
      --result_dir ${CAUSALACT_RUN_DIR}/osw_base_seed2 --test_all_meta_path $SH \
      > logs/base2_${dom}_${w}.log 2>&1 &
    i=$((i+1)); sleep 4
  done
done
echo "launched $i base-seed2 workers alive:$(pgrep -f run.py|wc -l)"
