#!/usr/bin/env bash
# Resume driver: import a fixed list of remaining OpenClaw journals + skills,
# with a per-turn watchdog so a stalled cloud LLM call can't wedge the run.
# All writes are idempotent, so a re-run of any item is harmless.
set -uo pipefail

AKH="${AKH_BIN:-/Users/toasty/ws/akh-medu/target/release/akh}"
WS="tecton"
WORK="$HOME/.openclaw/workspace"
LOG="${LOG:-/Users/toasty/akh-medu/scripts/openclaw_import.log}"
TURN_TIMEOUT="${TURN_TIMEOUT:-300}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(ts)" "$*" | tee -a "$LOG"; }

# one ACP headless turn with a hard SIGALRM timeout (perl), output to log
send_turn() {
  local prompt="$1"
  printf '%s\n' "$prompt" \
    | perl -e 'alarm shift @ARGV; exec @ARGV or die "exec: $!"' "$TURN_TIMEOUT" \
        "$AKH" -w "$WS" chat --headless 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[1]}
  if [[ "$rc" == "142" ]]; then log "!!! turn TIMED OUT after ${TURN_TIMEOUT}s"; fi
  echo "" | tee -a "$LOG"
}

# Remaining journals (absolute paths)
REM_JOURNALS=(
  "$WORK/memory/anima-observability-design.md"
  "$WORK/memory/anima-pipeline-findings.md"
  "$WORK/memory/anima-pipeline-v2-findings.md"
  "$WORK/memory/archibald-maintenance-design.md"
  "$WORK/memory/cluster-topology.md"
  "$WORK/memory/contacts-design-2026-05-11.md"
  "$WORK/memory/tasks.md"
  "$WORK/memory/tecton-home-improvements-2026-05-22.md"
)

echo "=== resume run $(ts) (timeout ${TURN_TIMEOUT}s) ===" >> "$LOG"

if [[ "${SKIP_JOURNALS:-0}" != "1" ]]; then
  log ">>> ${#REM_JOURNALS[@]} remaining journals"
  batch=()
  flush() {
    [[ ${#batch[@]} -eq 0 ]] && return
    local list; list=$(printf '%s, ' "${batch[@]}"); list=${list%, }
    send_turn "Import these OpenClaw memory journals into your knowledge graph. For EACH file, read it then call memory_write with a concise summary and source set to the file's absolute path (origin openclaw). Files: ${list}. These are idempotent; do all of them, then stop."
    batch=()
  }
  for f in "${REM_JOURNALS[@]}"; do
    batch+=("$f")
    [[ ${#batch[@]} -ge 2 ]] && flush
  done
  flush
fi

if [[ "${SKIP_SKILLS:-0}" != "1" ]]; then
  SKILLFILES=()
  while IFS= read -r line; do SKILLFILES+=("$line"); done < <(ls "$WORK"/skills/*/SKILL.md 2>/dev/null | sort)
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

log "=== resume run complete ==="
