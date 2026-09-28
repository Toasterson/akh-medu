#!/usr/bin/env bash
# Lilith supervisor status report (Anima WI 01a0e2db-ed79, Lilith C4b; workflow §4).
# Print it at every update and every wait point.
#
#   scripts/lilith/exam/supervisor_status.sh [--no-runpod]
#
# Sections:
#   disk      free space on the Mac (keep >= 20 GB; the runs are the largest files)
#   pods      every Runpod pod via REST (GET https://rest.runpod.io/v1/pods), Lilith pods marked
#   account   credit, spend/h and spend limit via GraphQL (https://api.runpod.io/graphql)
#   recorder  age of the newest C3 recorder snapshot under $LILITH_SNAPSHOT_ROOT
#   runs      active Lilith runs: RUNNING akh-medu-lilith* pods + live watchdog pid files
#
# The Runpod key is read at call time with
#   op-service-account read 'op://Homelab/Runpod API Key/credential'
# held in one shell variable, handed to curl on stdin (-H @-), and unset before exit.
# It is never written to disk, logged, exported, or put on a command line.
#
# Environment:
#   OPSA                   op-service-account path (default ~/.local/bin/op-service-account)
#   LILITH_SNAPSHOT_ROOT   recorder root (default /Volumes/akh-medu-runs/lilith/cos-snapshots)
#   LILITH_WATCHDOG_DIR    watchdog pid files, <run>.pid (default ~/lilith-private/watchdogs)
#   LILITH_RECORDER_STALE  seconds after which the recorder is flagged stale (default 5400)
#   RUNPOD_REST / RUNPOD_GRAPHQL   API base URLs (tests point these at doubles)
# Exit: 0 (report printed, sections that could not be read say why), 64 on a usage error.
set -uo pipefail
umask 077

KEY_REFERENCE="op://Homelab/Runpod API Key/credential"
OPSA="${OPSA:-$HOME/.local/bin/op-service-account}"
RUNPOD_REST="${RUNPOD_REST:-https://rest.runpod.io/v1}"
RUNPOD_GRAPHQL="${RUNPOD_GRAPHQL:-https://api.runpod.io/graphql}"
SNAPSHOT_ROOT="${LILITH_SNAPSHOT_ROOT:-/Volumes/akh-medu-runs/lilith/cos-snapshots}"
WATCHDOG_DIR="${LILITH_WATCHDOG_DIR:-$HOME/lilith-private/watchdogs}"
RECORDER_STALE="${LILITH_RECORDER_STALE:-5400}"
POD_PREFIX="akh-medu-lilith"
USE_RUNPOD=1

while (($#)); do
  case "$1" in
    --no-runpod) USE_RUNPOD=0; shift ;;
    -h | --help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) printf 'supervisor_status: error[lilith::status::usage]: unknown argument %s\n  help: see --help\n' "$1" >&2
       exit 64 ;;
  esac
done
[[ "$RECORDER_STALE" =~ ^[1-9][0-9]*$ ]] || {
  printf 'supervisor_status: error[lilith::status::usage]: LILITH_RECORDER_STALE must be a positive integer\n' >&2
  exit 64
}

echo "=== Lilith status $(date -u +%FT%TZ) ==="
df -h "$HOME" | awk 'NR==2{printf "disk: %s free of %s (%s used)\n", $4, $2, $5}'

KEY=""
cleanup() { KEY=""; unset KEY; }
trap cleanup EXIT

# api <url> [json-body]: prints the response body; the key only ever travels on curl's stdin.
api() {
  if [[ $# -gt 1 ]]; then
    printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$KEY" |
      curl -sS -m 30 --fail-with-body -H @- --data "$2" "$1"
  else
    printf 'Authorization: Bearer %s\n' "$KEY" | curl -sS -m 30 --fail-with-body -H @- "$1"
  fi
}

PODS_JSON=""
if ((USE_RUNPOD)); then
  if ! KEY="$("$OPSA" read "$KEY_REFERENCE" 2>/dev/null)" || [[ -z "$KEY" ]]; then
    KEY=""
    echo "pods: unavailable (cannot read the Runpod key via $OPSA; check the 1Password service account)"
    echo "account: unavailable (no key)"
  else
    if PODS_JSON="$(api "$RUNPOD_REST/pods")"; then
      # shellcheck disable=SC2016  # the single-quoted text is Python source, not shell
      PODS_PREFIX="$POD_PREFIX" python3 -c '
import json, os, sys
prefix = os.environ["PODS_PREFIX"]
try:
    pods = json.load(sys.stdin)
except json.JSONDecodeError:
    print("pods: unreadable REST response"); sys.exit(0)
if isinstance(pods, dict):
    pods = pods.get("pods") or pods.get("data") or []
running = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
burn = sum(float(p.get("costPerHr") or 0) for p in running)
print(f"pods: {len(pods)} total, {len(running)} running, ${burn:.2f}/h while running")
for p in sorted(pods, key=lambda p: str(p.get("name"))):
    mark = "*" if str(p.get("name", "")).startswith(prefix) else " "
    gpu = (p.get("machine") or {}).get("gpuDisplayName") or p.get("gpuTypeId") or (p.get("gpu") or {}).get("displayName") or "?"
    vol = p.get("networkVolumeId") or "-"
    rate = float(p.get("costPerHr") or 0)
    print(" %s %s %s %s %s $%.2f/h volume=%s started=%s" % (mark, p.get("id"), p.get("name"), p.get("desiredStatus"),
                                                           gpu, rate, vol, p.get("lastStartedAt") or "-"))
' <<<"$PODS_JSON"
    else
      echo "pods: unavailable (REST call failed)"
      PODS_JSON=""
    fi
    if ACCOUNT_JSON="$(api "$RUNPOD_GRAPHQL" '{"query":"query { myself { clientBalance currentSpendPerHr spendLimit } }"}')"; then
      # shellcheck disable=SC2016  # the single-quoted text is Python source, not shell
      python3 -c '
import json, sys
try:
    me = (json.load(sys.stdin).get("data") or {}).get("myself") or {}
except json.JSONDecodeError:
    me = {}
if not me:
    print("account: unreadable GraphQL response"); sys.exit(0)
bal, rate = me.get("clientBalance"), me.get("currentSpendPerHr")
line = "account: credit $%.2f, spending $%.3f/h, spend limit %s" % (float(bal or 0), float(rate or 0), me.get("spendLimit"))
if bal and rate and float(rate) > 0:
    line += f", ~{float(bal) / float(rate):.0f} h of credit at this rate"
print(line)
' <<<"$ACCOUNT_JSON"
    else
      echo "account: unavailable (GraphQL call failed)"
    fi
  fi
  cleanup
else
  echo "pods: skipped (--no-runpod)"
  echo "account: skipped (--no-runpod)"
fi

# recorder: newest <UTC-timestamp>/manifest.json under the snapshot root.
if [[ -d "$SNAPSHOT_ROOT" ]]; then
  SNAPSHOT_ROOT="$SNAPSHOT_ROOT" STALE="$RECORDER_STALE" python3 -c '
import os, re
from datetime import datetime, timezone
root, stale = os.environ["SNAPSHOT_ROOT"], int(os.environ["STALE"])
ids = sorted(d for d in os.listdir(root) if re.fullmatch(r"\d{8}T\d{6}Z", d)
             and os.path.isfile(os.path.join(root, d, "manifest.json")))
if not ids:
    print(f"recorder: no snapshots under {root}")
else:
    newest = datetime.strptime(ids[-1], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    age = int((datetime.now(timezone.utc) - newest).total_seconds())
    flag = "STALE" if age > stale else "ok"
    print(f"recorder: {len(ids)} snapshots, newest {ids[-1]} ({age // 60} min ago) {flag}")
'
else
  echo "recorder: snapshot root $SNAPSHOT_ROOT not mounted"
fi

# runs: RUNNING Lilith pods + watchdog pid files (never pgrep: a pattern matches its own watcher).
active=0
if [[ -n "$PODS_JSON" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    echo "run: pod $line"
    active=$((active + 1))
  done < <(PODS_PREFIX="$POD_PREFIX" python3 -c '
import json, os, sys
try:
    pods = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)
if isinstance(pods, dict):
    pods = pods.get("pods") or pods.get("data") or []
for p in pods:
    if str(p.get("name", "")).startswith(os.environ["PODS_PREFIX"]) and p.get("desiredStatus") == "RUNNING":
        print("%s (%s) RUNNING" % (p.get("name"), p.get("id")))
' <<<"$PODS_JSON")
fi
if [[ -d "$WATCHDOG_DIR" ]]; then
  for pidfile in "$WATCHDOG_DIR"/*.pid; do
    [[ -e "$pidfile" ]] || continue
    run="$(basename "$pidfile" .pid)"
    pid="$(head -1 "$pidfile" 2>/dev/null)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "run: watchdog $run alive (pid $pid)"
      active=$((active + 1))
    else
      echo "run: watchdog $run NOT running (pid file ${pid:-empty}); collect its log and check the pod"
    fi
  done
fi
echo "active runs: $active"
