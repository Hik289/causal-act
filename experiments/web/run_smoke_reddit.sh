#!/usr/bin/env bash
# Archived experiment launcher; see docs/reproduction.md.
: "${CAUSALACT_ROOT:?Run with scripts/run-experiment.sh}"
# ⚠️ API key/base_url/site URL 全部进程内注入 (见下), 不写 shell profile。
set -eo pipefail
VWA_DIR=${CAUSALACT_VWA_DIR}
RESULT_DIR=${CAUSALACT_RUN_DIR}/web/smoke_reddit
mkdir -p "$RESULT_DIR"
cd "$VWA_DIR"
# BLIP-2 captioner (image_som 需要) 放 CPU 跑: 2080 Ti 11GB 装不下 blip2-flan-t5-xl(~12GB VRAM);
# SoM 标注是 DOM-based 不需 GPU, base agent 是 API, 故整个 smoke 无需本地 GPU。
export CUDA_VISIBLE_DEVICES=""
# HF 权重缓存放 /data2 (不占 /data)
export HF_HOME=${CAUSALACT_RUN_DIR}/hf_cache
export TRANSFORMERS_CACHE=${CAUSALACT_RUN_DIR}/hf_cache
mkdir -p "$HF_HOME"
# ── 进程内注入 (不落盘到 profile) ─────────────────────────────
export DATASET=visualwebarena
export CLASSIFIEDS=http://127.0.0.1:9980
:
export SHOPPING=http://127.0.0.1:7770
export REDDIT=http://127.0.0.1:9999
export WIKIPEDIA=http://127.0.0.1:8888
export HOMEPAGE=http://127.0.0.1:4399
:
:
# 记录 token usage 供全量预算估算
export causalact_USAGE_LOG="$RESULT_DIR/usage.jsonl"
# 生成 config files (替换 __REDDIT__ 等占位符)
python scripts/generate_test_data.py
# 跑 reddit 前 N 个非-VQA task (SoM 多模态)
START=${1:-0}
END=${2:-3}
python run.py \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --test_start_idx "$START" \
  --test_end_idx "$END" \
  --result_dir "$RESULT_DIR" \
  --test_config_base_dir=config_files/vwa/test_reddit \
  --model "$CAUSALACT_BASE_MODEL" \
  --action_set_tag som \
  --observation_type image_som \
  --max_steps 10
