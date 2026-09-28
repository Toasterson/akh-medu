#!/usr/bin/env bash
# Liveness watchdog for one Lilith Runpod run (Anima WI 01a0e270-6021; operator decision 2026-09-28):
# check every INTERVAL (default 15 min) whether the run is still alive and progressing,
# instead of tearing the pod down on a blind timer.
#
#   watchdog.sh <pod-id> <ssh-host> <ssh-port> <run-id> [interval-seconds]
#
# Per check (one ssh call, no pgrep/pkill — a pattern would match the ssh command itself):
#   done     smoke-result.json exists            -> copy run dir back, terminate pod, exit 0
#   dead     pid gone and no result              -> copy what exists, terminate, exit 2
#   stalled  pid alive, no file written for a whole interval AND GPU at 0% -> copy, terminate, exit 3
#   running  otherwise                           -> keep going
#   unreachable twice in a row                   -> terminate, exit 4
set -uo pipefail
POD=$1; HOST=$2; PORT=$3; RUN=$4; INTERVAL=${5:-900}
ROOT=/workspace/lilith; KEY_REF='op://Homelab/Runpod API Key/credential'
OPSA=${OPSA:-$HOME/.local/bin/op-service-account}
DEST=${LILITH_RUNS_DEST:-$HOME/lilith-private/smoke/runs}
IDENTITY=${LILITH_SSH_IDENTITY:-$HOME/.ssh/id_ed25519}; KNOWN=${LILITH_KNOWN_HOSTS:-$HOME/lilith-private/smoke/known_hosts}
SSH=(ssh -i "$IDENTITY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20
     -o UserKnownHostsFile="$KNOWN" -p "$PORT" "root@$HOST")
log() { printf '%s watchdog[%s] %s\n' "$(date -u +%FT%TZ)" "$RUN" "$*"; }

probe() {
  # shellcheck disable=SC2016  # the single-quoted part expands on the pod, by design
  "${SSH[@]}" "RUNDIR=$ROOT/runs/$RUN; INTERVAL=$INTERVAL; "'
    if [ -f "$RUNDIR/smoke-result.json" ]; then echo state=done; exit 0; fi
    pid=""; [ -f "$RUNDIR/smoke.pid" ] && pid=$(cat "$RUNDIR/smoke.pid")
    alive=no; [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=yes
    newest=$(find "$RUNDIR" '"$ROOT"'/smoke-*.out -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
    age=$(( $(date +%s) - ${newest%.*} ))
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "state=probe alive=$alive pid=${pid:-none} progress_age=$age gpu_util=${gpu:-?}"'
}

collect() { mkdir -p "$DEST"; if scp -q -r -i "$IDENTITY" -o IdentitiesOnly=yes \
  -o UserKnownHostsFile="$KNOWN" -P "$PORT" \
  "root@$HOST:$ROOT/runs/$RUN" "$DEST/"; then log "collected $DEST/$RUN"; else log "collect failed"; fi; }

terminate() {
  local key code
  key=$("$OPSA" read "$KEY_REF") || { log "cannot read Runpod key; pod $POD NOT terminated"; return 1; }
  code=$(printf 'Authorization: Bearer %s\n' "$key" | curl -sS -m 60 -o /dev/null -w '%{http_code}' \
         -X DELETE -H @- "https://rest.runpod.io/v1/pods/$POD"); unset key
  log "terminate pod $POD -> HTTP $code ($1)"
}

misses=0
while :; do
  out=$(probe); rc=$?
  if [ $rc -eq 255 ] || [ -z "$out" ]; then
    misses=$((misses+1)); log "unreachable ($misses/2)"
    if [ $misses -ge 2 ]; then terminate "unreachable"; exit 4; fi
    sleep "$INTERVAL"; continue
  fi
  misses=0; log "$out"
  case "$out" in
    *state=done*) collect; terminate "done"; cat "$DEST/$RUN/smoke-result.json" 2>/dev/null; exit 0 ;;
  esac
  alive=$(sed -n 's/.*alive=\([a-z]*\).*/\1/p' <<<"$out")
  age=$(sed -n 's/.*progress_age=\([0-9-]*\).*/\1/p' <<<"$out")
  gpu=$(sed -n 's/.*gpu_util=\([0-9?]*\).*/\1/p' <<<"$out")
  if [ "$alive" != yes ]; then collect; terminate "dead"; exit 2; fi
  if [ "${age:-0}" -ge "$INTERVAL" ] && [ "$gpu" = 0 ]; then collect; terminate "stalled"; exit 3; fi
  sleep "$INTERVAL"
done
