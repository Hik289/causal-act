#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_RUN_DIR}/driving
: # Process and container cleanup is managed by the operator.
tar -xzf CARLA_0.9.15_pkg.tar.gz -C CARLA_pkg
echo "EXTRACT_DONE rc=$?"
ls CARLA_pkg/ | head
