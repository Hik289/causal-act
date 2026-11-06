#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_OSWORLD_DIR}
exec python train_outcome_v2.py --epochs 6 --cap 2000 --out ${CAUSALACT_RUN_DIR}/v_outcome_ckpt
