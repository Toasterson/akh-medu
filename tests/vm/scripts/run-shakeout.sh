#!/usr/bin/env bash
# Phase 50 / W1 — host-side driver for the illumos portability shakeout.
#
# rsyncs this akh-medu worktree into the `builder` VM (excluding target/ and
# .git to keep it light) and runs builder-shakeout.sh there over SSH,
# tee'ing the whole transcript to tests/vm/logs/shakeout.log. Reads the
# builder VM's forwarded SSH port + generated key from vmctl's state file,
# same as smoke.sh.
#
# LONG-RUNNING (native candle build on 2 vCPU / 4GB illumos = tens of
# minutes, maybe more). Run detached:
#   setsid nohup ./scripts/run-shakeout.sh > logs/shakeout-driver.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"      # tests/vm
REPO="$(cd "$HERE/../.." && pwd)"                             # worktree root
VMS_JSON="${HOME}/.local/share/vmctl/vms.json"
SSH_USER="vmctl"
LOG="$HERE/logs/shakeout.log"
mkdir -p "$HERE/logs"

port=$(python3 -c "import json;print(json.load(open('$VMS_JSON'))['builder']['ssh_host_port'])")
wd=$(python3 -c "import json;print(json.load(open('$VMS_JSON'))['builder']['work_dir'])")
key="$wd/id_ed25519_generated"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i "$key" -p "$port")
TARGET="${SSH_USER}@127.0.0.1"

echo "[run-shakeout] rsyncing worktree -> builder:~/akh-medu (port $port)"
rsync -az --delete \
    --exclude 'target/' --exclude '.git/' --exclude 'tests/vm/logs/' \
    --exclude 'tests/vm/.keys/' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$REPO/" "${TARGET}:akh-medu/" 2>&1 | tee -a "$LOG" \
    || { echo "[run-shakeout] rsync FAILED (see $LOG)"; exit 1; }

echo "[run-shakeout] uploading + running builder-shakeout.sh"
scp "${SSH_OPTS[@]}" "$HERE/scripts/builder-shakeout.sh" "${TARGET}:builder-shakeout.sh" 2>&1 | tee -a "$LOG"
ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x builder-shakeout.sh && CARGO_BUILD_JOBS=2 ./builder-shakeout.sh \$HOME/akh-medu" 2>&1 | tee -a "$LOG"

echo "[run-shakeout] done — transcript in $LOG"
