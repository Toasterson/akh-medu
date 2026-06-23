#!/usr/bin/env bash
# OpenClaw self-import driver for Tecton (ADR-076).
#
# Drives the akh's OWN LLM to read its OpenClaw past-life files and write them
# into its knowledge graph via the deterministic writer tools. All writes are
# idempotent (memory_write hashes summary+source; identity_awaken upserts), so
# re-running is safe and resumable.
#
# headless chat reads ONE line per turn and an agentic turn caps at ~8 tool
# rounds, so prompts are single-line and journals are batched small.
set -uo pipefail

AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"
MEM="$HOME/.openclaw/workspace/memory"
SKILLS="$HOME/.openclaw/workspace/skills"
LOG="${LOG:-/Users/toasty/akh-medu/scripts/openclaw_import.log}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }

# send_turn "<single-line prompt>"  — one ACP headless turn, output appended to log
send_turn() {
  local prompt="$1"
  printf '%s\n' "$prompt" | "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
  echo "" | tee -a "$LOG"
}

echo "=== OpenClaw import run $(ts) ===" >> "$LOG"

# ── 1. Identity (idempotent) ────────────────────────────────────────────────
if [[ "${SKIP_IDENTITY:-0}" != "1" ]]; then
  log ">>> identity"
  send_turn "Read ${HOME}/.openclaw/workspace/IDENTITY.md, ${HOME}/.openclaw/workspace/SOUL.md, and ${HOME}/.openclaw/workspace/USER.md, then call identity_awaken with the parsed name, description, and purpose. This is idempotent; proceed without asking."
fi

# ── 2. Journals, batched 3 per turn ─────────────────────────────────────────
if [[ "${SKIP_JOURNALS:-0}" != "1" ]]; then
  JOURNALS=()
  while IFS= read -r line; do JOURNALS+=("$line"); done < <(ls "$MEM"/*.md 2>/dev/null | sort)
  log ">>> ${#JOURNALS[@]} journals"
  batch=()
  flush() {
    [[ ${#batch[@]} -eq 0 ]] && return
    local list; list=$(printf '%s, ' "${batch[@]}"); list=${list%, }
    send_turn "Import these OpenClaw memory journals into your knowledge graph. For EACH file, read it then call memory_write with a concise summary and source set to the file's absolute path (origin openclaw). Files: ${list}. These are idempotent; do all of them, then stop."
    batch=()
  }
  for f in "${JOURNALS[@]}"; do
    batch+=("$f")
    [[ ${#batch[@]} -ge 3 ]] && flush
  done
  flush
fi

# ── 3. Skills, batched 2 per turn ───────────────────────────────────────────
if [[ "${SKIP_SKILLS:-0}" != "1" ]]; then
  SKILLFILES=()
  while IFS= read -r line; do SKILLFILES+=("$line"); done < <(ls "$SKILLS"/*/SKILL.md 2>/dev/null | sort)
  log ">>> ${#SKILLFILES[@]} skills"
  sbatch=()
  sflush() {
    [[ ${#sbatch[@]} -eq 0 ]] && return
    local list; list=$(printf '%s, ' "${sbatch[@]}"); list=${list%, }
    send_turn "Register these OpenClaw skills. For EACH SKILL.md, read it then call skill_author with the skill's name, description, and the steps/procedure it documents. Files: ${list}. Do all of them, then stop."
    sbatch=()
  }
  for f in "${SKILLFILES[@]}"; do
    sbatch+=("$f")
    [[ ${#sbatch[@]} -ge 2 ]] && sflush
  done
  sflush
fi

log "=== import run complete ==="
