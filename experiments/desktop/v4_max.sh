#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}
cd "$ROOT"
# compute remaining, clean partials, one shard per task for max parallelism
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import glob,os,json,shutil
res=causalact_location('CAUSALACT_RUN_DIR', 'gimp_v4_results/pyautogui/screenshot/gpt-5.6-sol/gimp')
ids=json.load(open(causalact_location('CAUSALACT_OSWORLD_DIR', '')+"/evaluation_examples/test_all.json"))["gimp"]
done={os.path.basename(x) for x in glob.glob(res+"/*") if os.path.exists(x+"/result.txt")}
for d in glob.glob(res+"/*"):
    if not os.path.exists(d+"/result.txt"): shutil.rmtree(d,ignore_errors=True)
rem=[x for x in ids if x not in done]
for i,t in enumerate(rem):
    json.dump({"gimp":[t]},open(f"{causalact_location('CAUSALACT_OSWORLD_DIR', '')}/shards/v4max_{i}.json","w"))
json.dump({"n":len(rem)},open(f"{causalact_location('CAUSALACT_OSWORLD_DIR', '')}/shards/v4max_meta.json","w"))
print("remaining",len(rem))
PY
: # Process and container cleanup is managed by the operator.
sleep 2
: # Process and container cleanup is managed by the operator.
N=$(python3 -c "import json;print(json.load(open('$ROOT/shards/v4max_meta.json'))['n'])")
for w in $(seq 0 $((N-1))); do
  CIV_REPLAY=1 CIV_STREAK=4 CIV_CAP=3 \
  CIV_BASE_ROOT=${CAUSALACT_RUN_DIR}/desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol \
  CIV_AUDIT=${CAUSALACT_RUN_DIR}/v4_audit_max_w${w}.log \
  CUDA_VISIBLE_DEVICES=$((w % 4)) \
  nohup python run_gimp_v4.py \
    --provider_name docker --headless --observation_type screenshot \
    --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 \
    --result_dir ${CAUSALACT_RUN_DIR}/gimp_v4_results \
    --test_all_meta_path shards/v4max_${w}.json \
    > logs/v4max_w${w}.log 2>&1 &
  echo "w$w pid $!"; sleep 9
done
echo "alive:$(pgrep -f run_gimp_v4|wc -l)"
