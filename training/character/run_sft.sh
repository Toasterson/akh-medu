#!/usr/bin/env bash
# White/Yellow/Green-belt character SFT (dojo curriculum, docs/ai/manuals/
# 2026-09-01-babyllm-dojo-curriculum.md) on Apple Silicon via MLX.
#
# Idempotent: safe to re-run. Skips venv creation if it already exists,
# re-installs mlx-lm (cheap, picks up updates), always regenerates the
# converted corpus (deterministic, seed 42), and skips training if a
# finished adapter is already present at ADAPTER_PATH unless FORCE_RETRAIN=1.
#
# Env overrides:
#   BABYLLM_BASE_MODEL   default mlx-community/Qwen3-0.6B-4bit
#   ADAPTER_NAME         default white-belt-v0
#   BATCH_SIZE           default 2
#   MAX_SEQ_LEN          default 1024
#   ITERS                default 1000
#   FORCE_RETRAIN=1      retrain even if the adapter already exists
#
# Extra args after `--` (or any trailing args) are passed through to
# `mlx_lm lora` verbatim, e.g. to retry with a smaller batch/seq length:
#   BATCH_SIZE=1 MAX_SEQ_LEN=768 FORCE_RETRAIN=1 ./run_sft.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV=".venv-mlx"
PY="$VENV/bin/python"

MODEL="${BABYLLM_BASE_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
FALLBACK_MODEL="mlx-community/Qwen3-0.6B-bf16"
ADAPTER_NAME="${ADAPTER_NAME:-white-belt-v0}"
ADAPTER_PATH="adapters/${ADAPTER_NAME}"
DATA_DIR="data"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
ITERS="${ITERS:-1000}"

echo "== step 1/3: uv venv (.venv-mlx, python 3.12) =="
if [ ! -x "$PY" ]; then
    uv venv --python 3.12 "$VENV"
fi
uv pip install --python "$PY" -U mlx-lm

echo "== step 2/3: convert corpus -> ${DATA_DIR}/{train,valid,lineage}.jsonl =="
"$PY" convert_corpus.py

if [ -f "${ADAPTER_PATH}/adapters.safetensors" ] && [ -z "${FORCE_RETRAIN:-}" ]; then
    echo "== step 3/3: adapter already exists at ${ADAPTER_PATH}, skipping =="
    echo "   (set FORCE_RETRAIN=1 to retrain from scratch)"
    exit 0
fi

# Verify the preferred base model resolves on the Hub; fall back to bf16
# if the 4bit repo is missing (mirrors the task's documented fallback).
if ! "$PY" -c "
from huggingface_hub import HfApi
import sys
try:
    HfApi().model_info('${MODEL}')
except Exception as e:
    sys.exit(1)
" >/dev/null 2>&1; then
    echo "base model ${MODEL} not resolvable, falling back to ${FALLBACK_MODEL}"
    MODEL="$FALLBACK_MODEL"
fi

mkdir -p "$(dirname "$ADAPTER_PATH")"

echo "== step 3/3: mlx_lm lora training (model=${MODEL}, batch=${BATCH_SIZE}, seq=${MAX_SEQ_LEN}, iters=${ITERS}) =="
"$PY" -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$DATA_DIR" \
    --fine-tune-type lora \
    --config lora_config.yaml \
    --num-layers 16 \
    --batch-size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --val-batches -1 \
    --learning-rate 1e-4 \
    --steps-per-report 20 \
    --steps-per-eval 100 \
    --max-seq-length "$MAX_SEQ_LEN" \
    --save-every 200 \
    --adapter-path "$ADAPTER_PATH" \
    --mask-prompt \
    --seed 42 \
    "$@"

echo "done. adapter at ${ADAPTER_PATH}"
