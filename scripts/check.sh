#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/experiments${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python3}" experiments/web/causalact/data/test_intervention_schema.py
"${PYTHON:-python3}" experiments/web/causalact/eval/test_causal_metrics.py
"${PYTHON:-python3}" scripts/summarize-results.py
