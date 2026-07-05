#!/usr/bin/env bash
# Durable re-import (postgres=true now). Inline EVERY file's content into the
# prompt so each turn is a single read-free generation -> writer-tool call:
# avoids both the 8-round/4KB read cap and the cloud-stall-on-read-turn hangs.
# Idempotent; WAL flushes to PG every 2s as it runs.
set -uo pipefail
AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"; WORK="$HOME/.openclaw/workspace"
LOG="${LOG:-/Users/toasty/akh-medu/scripts/openclaw_reimport.log}"
TT="${TURN_TIMEOUT:-300}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }
inline_turn() { # $1=prompt
  printf '%s\n' "$1" | perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $!"' "$TT" \
    "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

echo "=== DURABLE re-import $(ts) ===" >> "$LOG"

# Journals — one inline turn each
JN=()
while IFS= read -r l; do JN+=("$l"); done < <(ls "$WORK"/memory/*.md 2>/dev/null | sort)
log ">>> ${#JN[@]} journals (inline)"
for f in "${JN[@]}"; do
  log ">>> journal: $(basename "$f")"
  C=$(tr '\n\t' '  ' < "$f")
  inline_turn "Below is the full text of an OpenClaw memory journal at ${f}. Do NOT read any file. Extract the key facts and call the memory_write tool once with a concise summary, source=${f}, origin openclaw. Then stop. TEXT: ${C}"
done

# Skills — one inline turn each
SK=()
while IFS= read -r l; do SK+=("$l"); done < <(ls "$WORK"/skills/*/SKILL.md 2>/dev/null | sort)
log ">>> ${#SK[@]} skills (inline)"
for f in "${SK[@]}"; do
  log ">>> skill: $f"
  C=$(tr '\n\t' '  ' < "$f")
  inline_turn "Below is the full text of an OpenClaw skill definition at ${f}. Do NOT read any file. Call the skill_author tool once with the skill's name, a description, and the steps/procedure it documents. Then stop. TEXT: ${C}"
done

log "=== durable re-import complete ==="
