#!/usr/bin/env bash
# Fetches LongMemEval and LoCoMo evaluation datasets.
# Do NOT commit the downloaded files to git.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/eval_data"

mkdir -p "$DATA_DIR/longmemeval" "$DATA_DIR/locomo"

echo "Downloading LongMemEval oracle split (bounded retrieval smoke benchmark)..."
curl -fsSL -o "$DATA_DIR/longmemeval/longmemeval_oracle.json" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
    || { echo "LongMemEval download failed — dataset may require HuggingFace authentication"; exit 1; }

echo "Downloading LoCoMo (long conversational memory benchmark)..."
curl -fsSL -o "$DATA_DIR/locomo/locomo.tar.gz" \
    "https://github.com/alext234/locomo/raw/main/data/locomo.tar.gz" \
    || { echo "LoCoMo download failed"; exit 1; }
tar -xzf "$DATA_DIR/locomo/locomo.tar.gz" -C "$DATA_DIR/locomo/"
rm "$DATA_DIR/locomo/locomo.tar.gz"

echo "Datasets ready in $DATA_DIR"
