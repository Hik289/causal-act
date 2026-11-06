#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
cd ${CAUSALACT_CARLA_DIR}
# headless CARLA server on GPU0, port 2000
CUDA_VISIBLE_DEVICES=0 setsid ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 -nosound >${CAUSALACT_RUN_DIR}/driving/carla_server.log 2>&1 </dev/null &
echo "CARLA server launching pid $!"
sleep 45
# test client connection
python - <<'PY'
from causalact_runtime import location as causalact_location
import carla, time
try:
    c=carla.Client('localhost',2000); c.set_timeout(30.0)
    w=c.get_world(); m=c.get_available_maps()
    print("CARLA CLIENT OK: version", c.get_server_version(), "| maps:", len(m))
    print("sample maps:", [x.split('/')[-1] for x in m[:6]])
except Exception as e:
    print("CARLA client err:", repr(e)[:200])
PY
