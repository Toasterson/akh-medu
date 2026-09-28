#!/usr/bin/env bash
# Supervisor-side watcher for one Lilith Runpod smoke run (Anima WI 01a0e270-6021).
#
#   scripts/lilith/supervise_pod.sh --pod-id <id> --ssh root@<host> [--ssh-port <port>] \
#       --run-id <run-id> [--identity <key>] [--root /workspace/lilith] [--dest <dir>] \
#       [--poll-seconds 60] [--max-unreachable 5] [--hard-cap-seconds <n>]
#
# Waits for the run's completion marker, <root>/runs/<run-id>/smoke-result.json,
# which run_smoke.sh writes last (after SHA256SUMS and the tarball). It polls over
# ssh and never looks at process names (no pgrep -f): a process list says nothing
# about whether the evidence is complete.
#
# Ends when
#   * the marker exists: copies the run dir to <dest>/<run-id>, checks SHA256SUMS,
#     prints smoke-result.json on stdout; exit 0 (status passed) or 1 (anything else);
#   * the host is unreachable (ssh exit 255) for --max-unreachable consecutive polls:
#     exit 2. The pod is NOT terminated; check it in the Runpod console, since it may
#     still be billing;
#   * --hard-cap-seconds elapses (optional): best-effort copy of whatever the run dir
#     holds, then the pod is terminated through the Runpod REST API
#     (DELETE https://rest.runpod.io/v1/pods/<id>); exit 3, or 4 if the terminate call
#     failed. Passing --hard-cap-seconds is the operator's approval of that terminate
#     (the Runpod rule: billable deletes are explicit operator decisions). The API key is
#     read at call time with
#       op-service-account read 'op://Homelab/Runpod API Key/credential'
#     and handed to curl on stdin (-H @-); it is never written to disk, logged, or put
#     on a command line.
# Usage errors exit 64. <dest> defaults to ~/lilith-private/smoke/runs; an existing
# <dest>/<run-id> is refused up front so evidence is never overwritten.
set -euo pipefail
umask 077

RUNPOD_REST="https://rest.runpod.io/v1"
KEY_REFERENCE="op://Homelab/Runpod API Key/credential"

POD_ID="" SSH_TARGET="" SSH_PORT="" IDENTITY="" RUN_ID=""
ROOT="/workspace/lilith"
DEST="${HOME}/lilith-private/smoke/runs"
POLL_SECONDS=60
MAX_UNREACHABLE=5
HARD_CAP_SECONDS=""

log() { printf 'supervise[%s %s]: %s\n' "${RUN_ID:-?}" "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() {  # die <exit> <code> <message> <help>
  printf 'supervise: error[%s]: %s\n  help: %s\n' "$2" "$3" "$4" >&2
  exit "$1"
}
usage() { die 64 lilith::supervise::usage "$1" "see the header of $0 for the flags"; }

while (( $# )); do
  case "$1" in
    --pod-id) POD_ID="${2:-}"; shift 2 ;;
    --ssh) SSH_TARGET="${2:-}"; shift 2 ;;
    --ssh-port) SSH_PORT="${2:-}"; shift 2 ;;
    --identity) IDENTITY="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --max-unreachable) MAX_UNREACHABLE="${2:-}"; shift 2 ;;
    --hard-cap-seconds) HARD_CAP_SECONDS="${2:-}"; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) usage "unknown argument '$1'" ;;
  esac
done

[[ "$POD_ID" =~ ^[A-Za-z0-9]+$ ]] || usage "--pod-id '$POD_ID' is missing or not a Runpod pod id"
[[ "$SSH_TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.:_-]+$ ]] || usage "--ssh '$SSH_TARGET' is not user@host"
[[ -z "$SSH_PORT" || "$SSH_PORT" =~ ^[0-9]+$ ]] || usage "--ssh-port '$SSH_PORT' is not a port number"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || usage "--run-id '$RUN_ID' is missing or invalid"
[[ "$ROOT" =~ ^/[A-Za-z0-9/._-]+$ ]] || usage "--root '$ROOT' must be an absolute path without shell metacharacters"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || usage "--poll-seconds must be a positive integer"
[[ "$MAX_UNREACHABLE" =~ ^[1-9][0-9]*$ ]] || usage "--max-unreachable must be a positive integer"
[[ -z "$HARD_CAP_SECONDS" || "$HARD_CAP_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
  usage "--hard-cap-seconds must be a positive integer"
[[ -z "$IDENTITY" || -r "$IDENTITY" ]] || usage "--identity '$IDENTITY' is not readable"

REMOTE_RUN="$ROOT/runs/$RUN_ID"
LOCAL_RUN="$DEST/$RUN_ID"
[[ ! -e "$LOCAL_RUN" ]] || die 64 lilith::supervise::dest_exists "$LOCAL_RUN already exists" \
  "evidence is never overwritten; move it aside or pass another --dest"
mkdir -p "$DEST"

# Pods regenerate host keys on every start: pin them for this supervision only.
KNOWN_HOSTS="$(mktemp "${TMPDIR:-/tmp}/lilith-supervise-known-hosts.XXXXXX")"
trap 'rm -f "$KNOWN_HOSTS"' EXIT
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
             -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$KNOWN_HOSTS" -o LogLevel=ERROR)
[[ -z "$SSH_PORT" ]] || SSH_OPTIONS+=(-p "$SSH_PORT")
[[ -z "$IDENTITY" ]] || SSH_OPTIONS+=(-i "$IDENTITY" -o IdentitiesOnly=yes)

# Remote commands are built client-side on purpose from the validated ROOT/RUN_ID above.
# shellcheck disable=SC2029
remote() { ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "$1"; }

# probe: prints present | absent | no_run_dir; returns 255 when the host is unreachable.
probe() {
  remote "if [ -f '$REMOTE_RUN/smoke-result.json' ]; then echo present;
          elif [ -d '$REMOTE_RUN' ]; then echo absent; else echo no_run_dir; fi"
}

copy_back() {  # copy the remote run dir to $LOCAL_RUN via a partial dir; never overwrites
  local partial="$DEST/.partial-$RUN_ID"
  rm -rf "$partial"
  mkdir -p "$partial"
  if remote "tar -C '$ROOT/runs' -cf - '$RUN_ID'" | tar -C "$partial" -xf -; then
    mv "$partial/$RUN_ID" "$LOCAL_RUN"
    rmdir "$partial"
    log "copied $SSH_TARGET:$REMOTE_RUN -> $LOCAL_RUN"
    return 0
  fi
  log "copy of $REMOTE_RUN failed; partial copy left in $partial"
  return 1
}

verify_sums() {
  [[ -f "$LOCAL_RUN/SHA256SUMS" ]] || { log "no SHA256SUMS in $LOCAL_RUN"; return 1; }
  local checker=(sha256sum -c --quiet)
  command -v sha256sum >/dev/null || checker=(shasum -a 256 -c --quiet)
  if (cd "$LOCAL_RUN" && "${checker[@]}" SHA256SUMS >&2); then
    log "SHA256SUMS verified"
  else
    log "SHA256SUMS mismatch in $LOCAL_RUN: treat the copy as damaged evidence"
    return 1
  fi
}

terminate_pod() {  # DELETE the pod; the key lives only in this function's memory
  local key status
  if ! key="$(op-service-account read "$KEY_REFERENCE")" || [[ -z "$key" ]]; then
    log "could not read the Runpod API key from 1Password ($KEY_REFERENCE); pod $POD_ID NOT terminated"
    return 1
  fi
  status="$(printf 'Authorization: Bearer %s\n' "$key" |
            curl -sS -m 60 -o /dev/null -w '%{http_code}' -X DELETE -H @- "$RUNPOD_REST/pods/$POD_ID")" ||
    status="curl_failed"
  key=""
  if [[ "$status" == 2?? ]]; then
    log "pod $POD_ID terminated (HTTP $status)"
    return 0
  fi
  log "terminate of pod $POD_ID failed (HTTP $status); terminate it in the Runpod console now"
  return 1
}

log "watching $SSH_TARGET:$REMOTE_RUN/smoke-result.json (pod $POD_ID, poll ${POLL_SECONDS}s," \
    "unreachable limit $MAX_UNREACHABLE, hard cap ${HARD_CAP_SECONDS:-none})"
START=$SECONDS
misses=0
while :; do
  elapsed=$(( SECONDS - START ))
  if [[ -n "$HARD_CAP_SECONDS" ]] && (( elapsed >= HARD_CAP_SECONDS )); then
    log "hard cap ${HARD_CAP_SECONDS}s reached without smoke-result.json; copying what exists, then terminating"
    copy_back || true
    if terminate_pod; then exit 3; fi
    exit 4
  fi
  state="" rc=0
  state="$(probe)" || rc=$?
  if (( rc == 255 )); then
    misses=$(( misses + 1 ))
    log "host unreachable ($misses/$MAX_UNREACHABLE)"
    if (( misses >= MAX_UNREACHABLE )); then
      log "giving up: $SSH_TARGET unreachable for $misses consecutive polls; pod $POD_ID was NOT terminated" \
          "(check it in the Runpod console; it may still be billing)"
      exit 2
    fi
  elif (( rc != 0 )); then
    misses=0
    log "probe failed on the host (exit $rc); retrying"
  else
    misses=0
    case "$state" in
      present) break ;;
      absent) log "running (no smoke-result.json yet, ${elapsed}s)" ;;
      *) log "no run directory $REMOTE_RUN yet (smoke not started or wrong --run-id/--root)" ;;
    esac
  fi
  wait_for="$POLL_SECONDS"
  if [[ -n "$HARD_CAP_SECONDS" ]]; then
    left=$(( HARD_CAP_SECONDS - (SECONDS - START) ))
    (( left < wait_for )) && wait_for=$(( left > 0 ? left : 0 ))
  fi
  sleep "$wait_for"
done

log "smoke-result.json present; copying the run directory"
copy_back || die 1 lilith::supervise::copy "could not copy $REMOTE_RUN" \
  "rerun this script (the result is already on the volume) or copy it by hand"
verify_sums || true
cat "$LOCAL_RUN/smoke-result.json"
if python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "passed")' \
     "$LOCAL_RUN/smoke-result.json"; then
  exit 0
fi
exit 1
