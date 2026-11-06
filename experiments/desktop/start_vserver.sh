#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
: # Process and container cleanup is managed by the operator.
cd ${CAUSALACT_OSWORLD_DIR}
export CIV_VPORT=18833 CUDA_VISIBLE_DEVICES=3
exec python v_server.py
