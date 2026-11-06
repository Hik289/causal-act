#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_RUN_DIR}
mkdir -p driving && cd driving
echo "=== clone Bench2Drive repo ==="
if [ ! -d Bench2Drive ]; then
  git clone --depth 1 https://github.com/Thinklab-SJTU/Bench2Drive.git 2>&1 | tail -3
fi
echo "=== download CARLA 0.9.15 (Bench2Drive uses 0.9.15) ==="
if [ ! -f CARLA_0.9.15.tar.gz ] && [ ! -d CARLA_0.9.15 ]; then
  wget -q --show-progress https://tiny.carla.org/carla-0-9-15-linux -O CARLA_0.9.15.tar.gz 2>&1 | tail -2 || \
  wget -q https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz -O CARLA_0.9.15.tar.gz 2>&1 | tail -2
fi
echo "=== CARLA download status ==="
ls -la CARLA_0.9.15.tar.gz 2>/dev/null
df -h /data2 | tail -1
echo "INSTALL_CARLA_STEP1_DONE"
