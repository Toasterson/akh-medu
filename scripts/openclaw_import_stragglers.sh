#!/usr/bin/env bash
# Final 3 stragglers. 05-13 + observability: plain read-then-write retry.
# pipeline-findings (62KB): exceeds the 8-round/4KB read budget, so we inline
# the file content into the prompt (newlines->spaces) — the LLM extracts and
# calls memory_write in a single round, no file reads needed.
set -uo pipefail
AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"; WORK="$HOME/.openclaw/workspace/memory"
LOG="/Users/toasty/akh-medu/scripts/openclaw_import.log"
TT=420
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
turn() { # stdin = prompt; runs one headless turn with watchdog
  perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $!"' "$TT" \
    "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
}

echo "=== STRAGGLERS run $(ts) ===" >> "$LOG"

for name in 2026-05-13.md anima-observability-design.md; do
  f="$WORK/$name"; log ">>> straggler(read): $name"
  printf '%s\n' "Read ${f} and then ACTUALLY call the memory_write tool once with a concise factual summary and source=${f} (origin openclaw). Do not claim success without invoking the tool. One file only, then stop." | turn
  echo "" | tee -a "$LOG"
done

# pipeline-findings: inline content, no reads.
f="$WORK/anima-pipeline-findings.md"; log ">>> straggler(inline): anima-pipeline-findings.md"
CONTENT=$(tr '\n\t' '  ' < "$f")
printf '%s\n' "Below is the full text of an OpenClaw memory journal at ${f}. Do NOT read any file. Extract the key facts and call the memory_write tool once with a concise summary, source=${f}, origin openclaw. Then stop. TEXT: ${CONTENT}" | turn
echo "" | tee -a "$LOG"

log "=== stragglers run complete ==="
