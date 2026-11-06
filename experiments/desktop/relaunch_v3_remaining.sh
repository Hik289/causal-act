#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -e
ROOT=${CAUSALACT_OSWORLD_DIR}
RES=${CAUSALACT_RUN_DIR}/gimp_v3_results/pyautogui/screenshot/gpt-5.6-sol/gimp
cd "$ROOT"
python3 - <<'PY'
from causalact_runtime import location as causalact_location
import json,os,glob
root=causalact_location('CAUSALACT_OSWORLD_DIR', '')
res=causalact_location('CAUSALACT_RUN_DIR', 'gimp_v3_results/pyautogui/screenshot/gpt-5.6-sol/gimp')
d=json.load(open(root+'/evaluation_examples/test_all.json'))
all_ids=d['gimp']
done={os.path.basename(x) for x in glob.glob(res+'/*') if os.path.exists(x+'/result.txt')}
rem=[x for x in all_ids if x not in done]
N=min(8,max(1,len(rem)))
for i in range(8):
    sub={'gimp':rem[i::N]} if i<N else {'gimp':[]}
    json.dump(sub,open(f'{root}/shards/v3_rem_{i}.json','w'))
json.dump({'done':len(done),'remaining':len(rem),'remaining_ids':rem},open(root+'/logs/v3_relaunch_state.json','w'),indent=2)
print('done',len(done),'remaining',len(rem),'workers',N)
PY
: # Process and container cleanup is managed by the operator.
sleep 2
: # Process and container cleanup is managed by the operator.
N=$(python3 -c "import json;print(min(8,max(1,json.load(open('$ROOT/logs/v3_relaunch_state.json'))['remaining'])))")
for w in $(seq 0 $((N-1))); do
  CIV_LOOPBREAK=1 CIV_STREAK=4 CIV_CAP=3 \
  CIV_AUDIT=${CAUSALACT_RUN_DIR}/v3_rem_audit_w${w}.log \
  nohup python run_gimp_v3.py \
    --provider_name docker --headless --observation_type screenshot \
    --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" --domain gimp --max_steps 12 \
    --result_dir ${CAUSALACT_RUN_DIR}/gimp_v3_results \
    --test_all_meta_path shards/v3_rem_${w}.json \
    > logs/v3_rem_w${w}.log 2>&1 &
  echo "worker $w pid $!"
  sleep 10
done
