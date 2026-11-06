#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# Sharded parallel gimp B+A v2 (precision loop-breaker). N workers, each disjoint task subset.
set -eo pipefail
cd ${CAUSALACT_OSWORLD_DIR}
N=${1:-6}
RESULT=${CAUSALACT_RUN_DIR}/desktop_bav2_gimp_results
mkdir -p logs shards
# build shard meta files
python3 - "$N" <<PY
from causalact_runtime import location as causalact_location
import json, sys
N=int(sys.argv[1])
d=json.load(open("evaluation_examples/test_all.json"))
gimp=d["gimp"]
for i in range(N):
    sub={"gimp": gimp[i::N]}  # round-robin shard
    json.dump(sub, open(f"shards/gimp_shard_{i}.json","w"))
print(f"built {N} shards, sizes:", [len(gimp[i::N]) for i in range(N)])
PY
# launch N workers
for i in $(seq 0 $((N-1))); do
  LOG=logs/gimp_bav2_w${i}.log
  CIV_LOOPBREAK=1 CIV_CAP=2 \
  CIV_AUDIT=${CAUSALACT_RUN_DIR}/gimp_bav2_audit_w${i}.log \
  SCREEN_WIDTH=1280 SCREEN_HEIGHT=720 \
  setsid nohup python run_gimp_lb2.py --provider_name docker --headless \
    --observation_type screenshot --action_space pyautogui --model "$CAUSALACT_BASE_MODEL" \
    --domain gimp --max_steps 15 --result_dir "$RESULT" \
    --test_all_meta_path shards/gimp_shard_${i}.json > "$LOG" 2>&1 < /dev/null &
  echo "launched worker $i -> $LOG"
  sleep 3
done
echo "all $N workers launched"
