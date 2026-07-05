#!/usr/bin/env bash
# Gap-fill: re-import the 8 journals that the LLM skipped (some via hallucinated
# "already idempotent" claims). One file per turn + emphatic no-skip prompt so
# the model cannot batch-skip. memory_write is genuinely idempotent.
set -uo pipefail
AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"; WORK="$HOME/.openclaw/workspace/memory"
LOG="${LOG:-/Users/toasty/akh-medu/scripts/openclaw_import.log}"
TURN_TIMEOUT="${TURN_TIMEOUT:-240}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }

MISSING=(
  2026-05-08.md 2026-05-09.md 2026-05-10.md 2026-05-11.md
  2026-05-12.md 2026-05-13.md
  anima-observability-design.md anima-pipeline-findings.md
)

echo "=== GAP-FILL run $(ts) ===" >> "$LOG"
for name in "${MISSING[@]}"; do
  f="$WORK/$name"
  log ">>> gap: $name"
  printf '%s\n' "Read the file ${f} and then call memory_write exactly once with a concise factual summary of its contents and source set to ${f} (origin openclaw). You MUST actually invoke the memory_write tool — do NOT skip it or claim it already exists; the tool handles idempotency itself. Do this one file only, then stop." \
    | perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $!"' "$TURN_TIMEOUT" \
        "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[1]}; [[ "$rc" == "142" ]] && log "!!! $name TIMED OUT"
  echo "" | tee -a "$LOG"
done
log "=== gap-fill complete ==="
