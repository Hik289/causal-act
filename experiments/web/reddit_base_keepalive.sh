#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# Keepalive wrapper: relaunch reddit base full until all 210 tasks have render html (VWA get_unfinished dedups)
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/reddit_base_full
LOG=${CAUSALACT_VWA_DIR}/logs/reddit_base_full.log
RUN=${CAUSALACT_WEB_SCRIPTS}/run_reddit_base_full.sh
for attempt in $(seq 1 50); do
  done=$(ls "$RESULT_DIR"/*.html 2>/dev/null | wc -l)
  if [ "$done" -ge 210 ]; then echo "$(date) ALL 210 DONE" >> "$LOG"; break; fi
  echo "$(date) keepalive attempt $attempt, done=$done, (re)launching" >> "$LOG"
  bash "$RUN" >> "$LOG" 2>&1
  sleep 10
done
echo "$(date) keepalive exit, final done=$(ls $RESULT_DIR/*.html 2>/dev/null|wc -l)" >> "$LOG"
