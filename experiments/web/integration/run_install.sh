#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -x
python --version
pip install -r requirements.txt || { echo PIP_REQ_FAIL; exit 2; }
pip install -e . || { echo PIP_E_FAIL; exit 3; }
python -m playwright install chromium || { echo PW_FAIL; exit 4; }
python -m playwright install-deps chromium 2>/dev/null
echo INSTALL_DONE_RC=0
