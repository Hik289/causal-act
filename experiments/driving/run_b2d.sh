#!/usr/bin/env bash
# Bench2Drive single-route Base smoke protocol.
set -e
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
: "${OPENAI_API_KEY:?Supply a local model credential}"
: "${OPENAI_BASE_URL:?Supply a compatible endpoint for the driving smoke}"
mode="${1:-base}"
if [ "$mode" != base ]; then
    printf '%s\n' 'Only the Base smoke protocol is implemented.' >&2
    exit 2
fi
export CARLA_ROOT="$CAUSALACT_CARLA_DIR"
export LEADERBOARD_ROOT="$CAUSALACT_BENCH2DRIVE_DIR/leaderboard"
export SCENARIO_RUNNER_ROOT="$CAUSALACT_BENCH2DRIVE_DIR/scenario_runner"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CIV_MODE=base
export CIV_DRIVE_AUDIT="$CAUSALACT_RUN_DIR/driving/drive_audit_base.log"
cd "$LEADERBOARD_ROOT"
exec "${PYTHON:-python3}" leaderboard/leaderboard_evaluator.py \
  --routes="$LEADERBOARD_ROOT/data/routes_devtest.xml" \
  --routes-subset="${CAUSALACT_ROUTE_INDEX:-0}" --repetitions=1 --track=SENSORS \
  --checkpoint="$CAUSALACT_RUN_DIR/driving/results_base.json" \
  --agent="$LEADERBOARD_ROOT/leaderboard/autoagents/causalact_driving_agent.py" \
  --agent-config=none --debug=0 --port="${CARLA_PORT:-2000}" --traffic-manager-port="${CARLA_TM_PORT:-8000}"
