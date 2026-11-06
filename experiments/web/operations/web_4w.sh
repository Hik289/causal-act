#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_VWA_DIR}
: # Process and container cleanup is managed by the operator.
# 4 workers, one per GPU, splitting 210 into 4 shards (get_unfinished skips done)
for w in 0 1 2 3; do
  S=$((w*53)); E=$((w*53+53)); [ $E -gt 210 ] && E=210
  setsid bash ${CAUSALACT_WEB_AUX}/web_base_parallel.sh $S $E $w >${CAUSALACT_RUN_DIR}/web_base_w${w}.log 2>&1 </dev/null &
  sleep 8
done
sleep 2; echo "web alive:$(pgrep -f 'run.py.*test_reddit'|wc -l)"
