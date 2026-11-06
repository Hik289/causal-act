#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_BENCH2DRIVE_DIR}
pip install -r requirements.txt >${CAUSALACT_RUN_DIR}/driving/b2d_deps.log 2>&1
pip install py-trees==0.8.3 shapely tabulate ephem >>${CAUSALACT_RUN_DIR}/driving/b2d_deps.log 2>&1
echo "B2D_DEPS_DONE rc=$?" >>${CAUSALACT_RUN_DIR}/driving/b2d_deps.log
