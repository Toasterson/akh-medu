#!/usr/bin/env bash
# Phase 50 / W1 — illumos portability shakeout, run INSIDE the `builder` VM.
#
# Not part of `vmctl up` provisioning (that stays fast + re-runnable). This
# is the heavy step: build the release feature sets natively on OpenIndiana
# and run the test suite, capturing the FIRST failures for triage. Driven
# from the host by scripts/run-shakeout.sh, which rsyncs the worktree in and
# execs this over SSH.
#
# Runs each step best-effort (does NOT set -e) so one failure doesn't hide
# the others; every step's stdout+stderr is tee'd so the host can pull the
# logs back. CARGO_BUILD_JOBS is bounded because the VM has 4GB RAM and rustc
# LTO/codegen on the candle stack is memory-hungry.
set -uo pipefail

SRC="${1:-$HOME/akh-medu}"
cd "$SRC" || { echo "FATAL: source dir $SRC not found"; exit 2; }

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export CARGO_TARGET_DIR="$SRC/target"   # VM-local target; separate from host
. "$HOME/.cargo/env" 2>/dev/null || true

echo "==================================================================="
echo "illumos shakeout on $(uname -a)"
echo "rustc: $(rustc -Vv 2>&1 | head -1)  CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS"
echo "==================================================================="

run_step() {
    local label="$1"; shift
    echo
    echo ">>> STEP: $label"
    echo ">>> CMD:  $*"
    local start end rc
    start=$(date +%s)
    "$@"
    rc=$?
    end=$(date +%s)
    echo ">>> RESULT: $label rc=$rc ($((end-start))s)"
    return $rc
}

# 1) akh — lean client (ACP only, no candle/ORT). Cheapest; most likely to pass.
run_step "build akh (client)" \
    cargo build --release --no-default-features --features client --bin akh

# 2) akhomed — daemon feature set (server = daemon+acp+postgres, default candle).
run_step "build akhomed (server)" \
    cargo build --release --features server --bin akhomed

# 3) seshd — synthesis only (candle, no onnxruntime).
run_step "build seshd (synthesis)" \
    cargo build --release -p seshat --bin seshd --no-default-features --features synthesis

# 4) full test suite for the server feature set — never run on illumos before.
#    --no-run first so a compile failure is distinguishable from a test failure.
run_step "compile tests (server)" \
    cargo test --features server --no-run

run_step "run tests (server)" \
    cargo test --features server -- --test-threads=2

echo
echo ">>> SHAKEOUT COMPLETE"
