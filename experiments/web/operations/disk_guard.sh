#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# Abort classifieds pull if /data free < 25G to protect shared disk
while docker ps -a --format "{{.Names}}" | grep -q classifieds || pgrep -f "docker compose up" >/dev/null; do
  AVAIL=$(df --output=avail -BG /data | tail -1 | tr -dc "0-9")
  if [ "$AVAIL" -lt 25 ]; then
    echo "$(date) DISK_GUARD_TRIP avail=${AVAIL}G -> stopping classifieds pull" >> ${CAUSALACT_VWA_DIR}/logs/disk_guard.log
: # Process and container cleanup is managed by the operator.
    cd ${CAUSALACT_RUN_DIR}/sites/classifieds_docker_compose && docker compose down 2>&1
    break
  fi
  sleep 20
done
echo "$(date) disk_guard exit avail_now=$(df --output=avail -BG /data | tail -1)" >> ${CAUSALACT_VWA_DIR}/logs/disk_guard.log
