#!/usr/bin/env bash
# Use an already activated environment for the selected benchmark.
set -e
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CAUSALACT_ROOT="$root"
export CAUSALACT_RUN_DIR="${CAUSALACT_RUN_DIR:-$root/.runs}"
export CAUSALACT_VWA_DIR="${CAUSALACT_VWA_DIR:-$root/.vendor/VisualWebArena}"
export CAUSALACT_OSWORLD_DIR="${CAUSALACT_OSWORLD_DIR:-$root/.vendor/OSWorld}"
export CAUSALACT_BENCH2DRIVE_DIR="${CAUSALACT_BENCH2DRIVE_DIR:-$root/.vendor/Bench2Drive}"
export CAUSALACT_CARLA_DIR="${CAUSALACT_CARLA_DIR:-$root/.vendor/CARLA}"
export CAUSALACT_VWA_AUTH="${CAUSALACT_VWA_AUTH:-$CAUSALACT_VWA_DIR/.auth/reddit_state.json}"
export CAUSALACT_WEB_SCRIPTS="$root/experiments/web"
export CAUSALACT_WEB_AUX="$CAUSALACT_WEB_SCRIPTS/operations"
export CAUSALACT_BASE_MODEL="${CAUSALACT_BASE_MODEL:-gpt-5.6-sol}"
export PYTHONPATH="$root/experiments${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CAUSALACT_RUN_DIR"/{web/data,web/checkpoints,desktop,driving,scratch}
if [ "$#" -eq 0 ]; then
    printf 'Usage: %s <repository-relative script> [args...]\n' "$0" >&2
    exit 2
fi
cd "$root"
script="$1"
shift
case "$script" in
    experiments/web/*) export PYTHONPATH="$CAUSALACT_VWA_DIR:$PYTHONPATH" ;;
    experiments/desktop/*)
        export PYTHONPATH="$CAUSALACT_OSWORLD_DIR:$PYTHONPATH"
        cd "$CAUSALACT_OSWORLD_DIR"
        ;;
    experiments/driving/*) export CIV_DRIVE_MODEL="${CIV_DRIVE_MODEL:-$CAUSALACT_BASE_MODEL}" ;;
esac
case "$script" in
    *.sh) exec bash "$root/$script" "$@" ;;
    *.py) exec "${PYTHON:-python}" "$root/$script" "$@" ;;
    *) printf 'Expected a .sh or .py entry point.\n' >&2; exit 2 ;;
esac
