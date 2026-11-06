#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -x
python -m pip install --upgrade pip
export PIP_CACHE_DIR=${CAUSALACT_RUN_DIR}/pip_cache
cd ${CAUSALACT_OSWORLD_DIR}
pip install -r requirements.txt || { echo REQ_FAIL; exit 2; }
pip install -e . 2>/dev/null
python -c 'import desktop_env; print("desktop_env OK")'
echo OSWORLD_ENV_DONE
