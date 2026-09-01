#!/usr/bin/env bash
# The Phase 51 350M foundation run. Detached, resumable, sequential.
#
#   bash training/scripts/run-350m.sh            # start (or resume) the run
#   bash training/scripts/status.sh p350m        # check on it
#   bash training/scripts/resume.sh p350m        # after a box restart / preemption
#
# Reads every hyperparameter from training/configs/350m.env — edit there, not
# here, so the config stays a single source of truth that lands in git.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT="$PWD"

CONFIG="${CONFIG:-training/configs/350m.env}"
[ -f "$CONFIG" ] || { echo "FAIL: config not found: $CONFIG" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG"

BACKEND_FEATURE="${BACKEND_FEATURE:-cuda}"     # cuda (H100) | gpu (local ROCm)
RUN_DIR="runs/${RUN_NAME}"
LOG="runs/${RUN_NAME}.log"
BIN="./target/release/examples/train_rule"

# --- preflight: never start a run that can't finish -------------------------
# RULE 1: one GPU job at a time. Two concurrent jobs hung the AMD card hard
# enough to need a reboot; don't repeat it on rented hardware.
if pgrep -f "examples/train_rule" >/dev/null 2>&1; then
  echo "FAIL: a train_rule process is already running — GPU jobs must be sequential." >&2
  echo "      Check: bash training/scripts/status.sh ${RUN_NAME}" >&2
  exit 1
fi

[ -x "$BIN" ] || {
  echo "Building ($BACKEND_FEATURE)…"
  cargo build -p grug-train --example train_rule --example eval_rule \
      --features "$BACKEND_FEATURE" --release
}

# The 350M preset must exist, or the run dies after the dataset build.
"$BIN" --help >/dev/null 2>&1 || true
grep -q "p1a_350m" crates/grug-train/examples/train_rule.rs \
  || { echo "FAIL: train_rule has no p1a_350m model arm" >&2; exit 1; }

# --- resume if there's a checkpoint ------------------------------------------
RESUME_ARG=()
if [ -d "$RUN_DIR/checkpoints" ]; then
  LAST="$(ls -1 "$RUN_DIR/checkpoints" 2>/dev/null | sort | tail -1 || true)"
  if [ -n "$LAST" ]; then
    echo "Resuming from $RUN_DIR/checkpoints/$LAST"
    RESUME_ARG=(--resume "$RUN_DIR/checkpoints/$LAST")
  fi
fi

mkdir -p "$RUN_DIR" training/results

echo "=== launching ${RUN_NAME} (${MODEL}, lr ${LR}, ${EPOCHS} epochs) — detached"
setsid nohup "$BIN" \
    --logic "$LOGIC" --arm "$ARM" --model "$MODEL" \
    --typed-slots "$TYPED_SLOTS" --seed "$SEED" \
    --preds-per-cell "$PREDS_PER_CELL" --max-level "$MAX_LEVEL" \
    --guard-prob "$GUARD_PROB" \
    --distractor-facts "$DISTRACTOR_FACTS" --distractor-rules "$DISTRACTOR_RULES" \
    --yes-frac "$YES_FRAC" \
    --train-depths "$TRAIN_DEPTHS" --eval-depths "$EVAL_DEPTHS" \
    --train-examples "$TRAIN_EXAMPLES" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --learning-rate "$LR" \
    --warmup-steps "${WARMUP_STEPS:-0}" --lr-min-ratio "${LR_MIN_RATIO:-0.1}" \
    --bpe-vocab-size "$BPE_VOCAB_SIZE" --holdout-examples "$HOLDOUT" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --eval-every "$EVAL_EVERY" --ckpt-every "$CKPT_EVERY" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --artifacts-dir "$RUN_DIR" \
    "${RESUME_ARG[@]}" \
    > "$LOG" 2>&1 < /dev/null &
disown || true

sleep 8
# RULE 2: verify by process + log mtime, NEVER by log content — the corpus is
# deterministic-seeded, so a stale log from a previous run is byte-identical to
# a fresh one and will happily fool you into thinking a dead run is alive.
if pgrep -f "examples/train_rule" >/dev/null && \
   [ "$(stat -c %Y "$LOG" 2>/dev/null || echo 0)" -gt "$(( $(date +%s) - 120 ))" ]; then
  echo "=== RUNNING (pid $(pgrep -f 'examples/train_rule' | head -1)), log: $LOG"
  echo "    status: bash training/scripts/status.sh ${RUN_NAME}"
else
  echo "FAIL: launch did not take — no live process or log is stale. Check $LOG" >&2
  tail -20 "$LOG" 2>/dev/null
  exit 1
fi
