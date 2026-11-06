#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# CARLA 0.9.15 python API is cp37; Bench2Drive needs carla + deps
WHL=${CAUSALACT_CARLA_DIR}/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
pip install "$WHL" >>${CAUSALACT_RUN_DIR}/driving/mkenv.log 2>&1
pip install numpy pygame networkx distro six >>${CAUSALACT_RUN_DIR}/driving/mkenv.log 2>&1
python -c "import carla; print('carla import OK', carla.__file__)" >>${CAUSALACT_RUN_DIR}/driving/mkenv.log 2>&1
echo "ENV_DONE rc=$?" >>${CAUSALACT_RUN_DIR}/driving/mkenv.log
