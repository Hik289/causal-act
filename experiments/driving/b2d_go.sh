#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
: # Process and container cleanup is managed by the operator.
: # Process and container cleanup is managed by the operator.
sleep 4
cd ${CAUSALACT_CARLA_DIR}
CUDA_VISIBLE_DEVICES=0 nohup ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 -nosound >${CAUSALACT_RUN_DIR}/driving/carla_server.log 2>&1 &
sleep 40
: # Process and container cleanup is managed by the operator.
cd ${CAUSALACT_RUN_DIR}/driving
nohup bash run_b2d.sh base >b2d_base.log 2>&1 &
echo "launched eval pid $! at $(date +%H:%M:%S)"
