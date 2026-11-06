#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -eo pipefail
cd ${CAUSALACT_VWA_DIR}
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
:
export SHOPPING=http://127.0.0.1:7770
export REDDIT=http://127.0.0.1:9999
export WIKIPEDIA=http://127.0.0.1:8888
export HOMEPAGE=http://127.0.0.1:4399
:
:
echo '=== generate_test_data ==='
python scripts/generate_test_data.py 2>&1 | head -8
echo '=== generated reddit configs ==='
ls config_files/vwa/test_reddit/ 2>/dev/null | head -3
ls config_files/vwa/test_reddit/ 2>/dev/null | wc -l
echo '=== auto_login (reddit cookie) ==='
mkdir -p ./.auth
python browser_env/auto_login.py 2>&1 | tail -10
ls -la ./.auth/ 2>&1
echo PREP_DONE
