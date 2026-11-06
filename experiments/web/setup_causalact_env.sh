#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
set -x
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export PIP_CACHE_DIR=${CAUSALACT_RUN_DIR}/pip_cache
pip install 'torch==2.1.2' torchvision --index-url https://download.pytorch.org/whl/cu118 || { echo TORCH_FAIL; exit 2; }
pip install 'transformers==4.49.0' 'accelerate>=0.26' pillow num2words 'datasets>=2.18' || { echo TF_FAIL; exit 3; }
python -c 'import torch,transformers; print("torch",torch.__version__,"cuda",torch.cuda.is_available(),"tf",transformers.__version__)'
echo ENV_DONE_RC=0
