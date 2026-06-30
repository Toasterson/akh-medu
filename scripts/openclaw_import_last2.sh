#!/usr/bin/env bash
# Final 2: cloud GLM intermittently hangs on read-tool turns. Inline the content
# (small files) so the turn is a single generation -> memory_write, no reads.
set -uo pipefail
AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"; WORK="$HOME/.openclaw/workspace/memory"
LOG="/Users/toasty/akh-medu/scripts/openclaw_import.log"
TT=300
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }

echo "=== LAST2 run $(ts) ===" >> "$LOG"
for name in 2026-05-13.md anima-observability-design.md; do
  f="$WORK/$name"; log ">>> last2(inline): $name"
  CONTENT=$(tr '\n\t' '  ' < "$f")
  printf '%s\n' "Below is the full text of an OpenClaw memory journal at ${f}. Do NOT read any file. Extract the key facts and call the memory_write tool once with a concise summary, source=${f}, origin openclaw. Then stop. TEXT: ${CONTENT}" \
    | perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $!"' "$TT" "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
done
log "=== last2 run complete ==="
