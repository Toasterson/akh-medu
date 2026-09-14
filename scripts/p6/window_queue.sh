#!/usr/bin/env bash
# Phase 51 P6 — run local windows one after another on the single GPU (ADR 105/106).
#
#   scripts/p6/window_queue.sh <queue-file>
#
# Queue file: one window per line, `#` comments allowed:
#   <window-dir>|<parent-bundle>|<siblings train.sft.jsonl>|<extra train_knowledge args>
# A window whose `status.json` already says `done` is skipped; one that says `train`/`gate`/`build`
# (a supervisor still running it) is waited for; anything else (missing/failed) is (re)launched via
# `scripts/p6/local_window.sh` in the foreground, so windows never overlap. A failed window stops
# the queue: a named failure is a hard stop, not something to skip past silently (ADR 105).
# Every window posts its own board events; this script only writes `<queue-file>.log`.
set -euo pipefail
QUEUE="${1:?queue file}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
LOG="$QUEUE.log"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

phase_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('phase',''))" "$1/status.json" 2>/dev/null || echo missing; }

while IFS='|' read -r window parent siblings extra; do
  [[ -z "$window" || "$window" == \#* ]] && continue
  case "$(phase_of "$window")" in
    done) say "skip $window (done)"; continue ;;
    build|train|gate|trained)
      say "waiting for running window $window"
      while [[ "$(phase_of "$window")" =~ ^(build|train|gate|trained)$ ]]; do sleep 120; done
      ;;
  esac
  case "$(phase_of "$window")" in
    done) say "$window finished: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['detail'])" "$window/status.json")"; continue ;;
    failed) say "STOP: $window failed — $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['detail'])" "$window/status.json")"; exit 1 ;;
  esac
  say "launch $window from $parent with $siblings $extra"
  # shellcheck disable=SC2086
  P6_BASE_TRAIN="${P6_BASE_TRAIN:-runs/p6-4/base/train.sft.jsonl}" P6_BASE_EVAL="${P6_BASE_EVAL:-runs/p6-4/base/eval.sft.jsonl}" \
    caffeinate -i scripts/p6/local_window.sh "$window" "$parent" "$siblings" $extra || { say "STOP: $window failed"; exit 1; }
  say "$window finished: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['detail'])" "$window/status.json")"
done < "$QUEUE"
say "queue complete"
