#!/usr/bin/env bash
# Honest status of a training run — verified against LIVE signals.
#
#   bash training/scripts/status.sh [run-name]     # default: p350m
#
# Why this exists: the corpus is deterministic-seeded, so a *stale* log from a
# previous run is byte-identical to a fresh one. Reading log CONTENT to decide
# "is it running?" will fool you (it fooled us for two hours). This checks the
# process table and the log's mtime instead.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN="${1:-p350m}"
LOG="runs/${RUN}.log"
DIR="runs/${RUN}"
NOW=$(date +%s)

echo "=== ${RUN} @ $(date '+%H:%M:%S')"

# --- liveness (process + mtime, never content) ------------------------------
PID="$(pgrep -f 'examples/train_rule' | head -1 || true)"
if [ -n "$PID" ]; then
  ELAPSED=$(ps -o etimes= -p "$PID" 2>/dev/null | tr -d ' ')
  echo "process : ALIVE (pid $PID, elapsed ${ELAPSED}s)"
else
  echo "process : none running"
fi

if [ -f "$LOG" ]; then
  MTIME=$(stat -c %Y "$LOG")
  AGE=$(( NOW - MTIME ))
  printf 'log     : %s (last write %ss ago)' "$LOG" "$AGE"
  if [ -n "$PID" ] && [ "$AGE" -gt 900 ]; then
    printf '  <-- WARNING: process alive but log stale >15min (hung? long eval?)'
  fi
  echo
else
  echo "log     : $LOG (missing)"
fi

# --- progress ---------------------------------------------------------------
if [ -f "$LOG" ]; then
  echo "step    : $(grep -E '^step ' "$LOG" | tail -1 || echo '(none yet)')"
  echo "--- last evals (in-distribution) ---"
  grep -E 'eval@' "$LOG" | grep 'in-dist' | tail -3 || echo "(no evals yet)"
  echo "--- last evals (length-generalization; chain 0.000 is EXPECTED) ---"
  grep -E 'eval@' "$LOG" | grep 'len-gen' | tail -2 || echo "(none)"
  ERR="$(grep -iE 'panic|fault|hang|exception|NaN' "$LOG" | tail -2 || true)"
  [ -n "$ERR" ] && { echo "--- !! errors !! ---"; echo "$ERR"; }
fi

# --- checkpoints (what you can resume from) ---------------------------------
if [ -d "$DIR/checkpoints" ]; then
  echo "ckpts   : $(ls -1 "$DIR/checkpoints" | wc -l) saved, newest $(ls -1 "$DIR/checkpoints" | sort | tail -1)"
else
  echo "ckpts   : none yet"
fi

echo
echo "resume  : bash training/scripts/resume.sh ${RUN}"
echo "note    : quote the BEST checkpoint, not the final one (high-LR runs dip and recover)."
