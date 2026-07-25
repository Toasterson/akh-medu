#!/usr/bin/env bash
# One-shot bootstrap for a fresh NVIDIA (H100) cloud box: verify the toolchain,
# build the CUDA training binaries, and prove the GPU is actually usable before
# you spend hours (or rental money) on a run that was never going to work.
#
#   bash training/scripts/setup-cuda.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
ROOT="$PWD"
echo "=== Phase 51 CUDA setup — repo: $ROOT"

fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 1. toolchain -----------------------------------------------------------
command -v cargo >/dev/null || fail "cargo not found — install Rust (https://rustup.rs)"
echo "rust : $(rustc --version)"

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — no NVIDIA driver on this box"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || fail "nvidia-smi ran but could not query the GPU"

if ! command -v nvcc >/dev/null; then
  echo "WARN: nvcc not on PATH. Burn's CUDA backend needs the CUDA toolkit at build time."
  echo "      If CUDA lives elsewhere, export CUDA_PATH=/usr/local/cuda (or similar) and re-run."
  [ -n "${CUDA_PATH:-}" ] || fail "neither nvcc on PATH nor CUDA_PATH set"
else
  echo "cuda : $(nvcc --version | tail -1)"
fi

# --- 2. build ---------------------------------------------------------------
# `cuda` and `gpu` (ROCm) are mutually exclusive — the crate enforces this with
# a compile_error!, so a stray --features gpu here fails loudly rather than
# silently building the wrong backend.
echo "=== building CUDA training binaries (first build pulls CubeCL: expect 5-15 min)"
cargo build -p grug-train --example train_rule --example eval_rule \
    --features cuda --release

# --- 3. prove the GPU works -------------------------------------------------
# A 60-step toy run on the tiny `spike` model: if the CUDA context, kernels and
# autodiff are wired correctly this finishes in well under a minute and the loss
# moves. Cheap insurance against discovering a broken backend 3 hours in.
echo "=== GPU smoke test (tiny model, 60 steps)"
SMOKE="$ROOT/runs/_cuda-smoke"
rm -rf "$SMOKE"
./target/release/examples/train_rule \
    --logic rule --model spike --typed-slots true --seed 7 \
    --max-level 3 --train-examples 64 --epochs 1 --batch-size 8 \
    --bpe-vocab-size 500 --holdout-examples 8 --eval-every 1000 \
    --ckpt-every 1000 --eval-max-new-tokens 40 --max-seq-len 512 \
    --train-depths 2 --eval-depths 3 \
    --artifacts-dir "$SMOKE" 2>&1 | tail -12

grep -qE "^step +[0-9]+: loss" "$SMOKE/../_cuda-smoke.log" 2>/dev/null || true
echo
echo "=== setup OK."
echo "Next: bash training/scripts/run-350m.sh     # the real run (detached, resumable)"
echo "      bash training/scripts/status.sh p350m # honest live status"
