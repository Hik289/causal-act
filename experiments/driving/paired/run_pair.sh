#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
routes=$1 out=$2
shift 2
protocol="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
carla="${CAUSALACT_CARLA_DIR:?Run with scripts/run-experiment.sh}"
leader="${CAUSALACT_BENCH2DRIVE_DIR:?Set your Bench2Drive checkout}/leaderboard"
scenario="$CAUSALACT_BENCH2DRIVE_DIR/scenario_runner"
port="${CARLA_PORT:-2000}"
tm_port="${CARLA_TM_PORT:-8000}"
export CAUSALACT_EXTERNAL_CARLA=1
export CARLA_ROOT="$carla" LEADERBOARD_ROOT="$leader" SCENARIO_RUNNER_ROOT="$scenario"
export PYTHONPATH="$carla/PythonAPI/carla:$leader:$scenario${PYTHONPATH:+:$PYTHONPATH}"
for egg in "$carla"/PythonAPI/carla/dist/carla-*.egg; do
  if [ -f "$egg" ]; then export PYTHONPATH="$PYTHONPATH:$egg"; fi
done
mkdir -p "$out"
stop_simulator() {
  kill -TERM -- -"$sim" 2>/dev/null || true
  for attempt in $(seq 1 20); do
    if ! kill -0 -- -"$sim" 2>/dev/null; then break; fi
    sleep .25
  done
  kill -KILL -- -"$sim" 2>/dev/null || true
  wait "$sim" 2>/dev/null || true
}
for mode in "$@"; do
  resume=()
  if [ -s "$out/results_$mode.json" ]; then
    if "${PYTHON:-python3}" - "$out/results_$mode.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))["_checkpoint"]
i, n = c["progress"]
raise SystemExit(not (i == n == len(c["records"]) and all(r["status"] != "Started" for r in c["records"])))
PY
    then continue; fi
    resume=(--resume True)
  fi
  cd "$carla"
  setsid ./CarlaUE4.sh -RenderOffScreen -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-0}" -carla-rpc-port="$port" -nosound \
    >>"$out/carla_$mode.log" 2>&1 &
  sim=$!
  printf '%s\n' "$sim" >"$out/sim.pid"
  trap stop_simulator EXIT
  sleep 55
  kill -0 "$sim"
  cd "$leader"
  export CIV_MODE="$mode" CIV_DRIVE_AUDIT="$out/drive_audit_$mode.log"
  printf '\nStarting %s, routes %s\n' "$mode" "$(basename "$routes")" >>"$out/eval_$mode.log"
  "${PYTHON:-python3}" -u leaderboard/leaderboard_evaluator.py \
    --routes="$routes" --repetitions=1 --track=SENSORS --traffic-manager-seed=0 \
    --checkpoint="$out/results_$mode.json" --agent="$protocol/agent.py" \
    --agent-config=none --debug=0 --host=127.0.0.1 --port="$port" --traffic-manager-port="$tm_port" \
    "${resume[@]}" >>"$out/eval_$mode.log" 2>&1
  stop_simulator
  trap - EXIT
  sleep 5
done
