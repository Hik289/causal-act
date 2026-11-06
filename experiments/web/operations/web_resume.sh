#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_VWA_DIR}
for w in 0 1 2 3 4 5; do
  S=$((w*35)); E=$((w*35+35)); [ $E -gt 210 ] && E=210
  setsid bash ${CAUSALACT_WEB_AUX}/web_base_parallel.sh $S $E $w >${CAUSALACT_RUN_DIR}/web_base_w${w}.log 2>&1 </dev/null &
  sleep 3
done
sleep 2; echo "web alive:$(pgrep -f 'run.py.*test_reddit'|wc -l)"
