#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
mkdir -p ${CAUSALACT_RUN_DIR}/web_clean
cd ${CAUSALACT_VWA_DIR}
for w in 0 1 2 3 4 5; do
  S=$((w*35)); E=$((w*35+35)); [ $E -gt 210 ] && E=210
  setsid bash ${CAUSALACT_WEB_AUX}/web_clean.sh $S $E >${CAUSALACT_RUN_DIR}/web_clean_w${w}.log 2>&1 </dev/null &
  sleep 5
done
sleep 2; echo "web-clean alive:$(pgrep -f 'run.py.*test_reddit'|wc -l)"
