#!/usr/bin/env bash
# Phase 50 / W1 — vmctl up smoke test.
#
# Brings up the exec1/pg/builder fleet declared in VMFile.kdl and asserts
# the provisioned state via non-interactive SSH:
#   exec1   -> akh-exec user exists + is a real (non-root) account
#   pg      -> postgres accepts a connection, pgvector extension present
#   builder -> rustc/cargo on PATH and runnable
#
# `vmctl ssh` (the CLI subcommand) is always interactive (execs the system
# `ssh` binary with no way to pass a remote command through vmctl itself —
# see crates/vmctl/src/commands/ssh.rs), so this script talks to the guests
# directly: it reads vmctl's own state file (~/.local/share/vmctl/vms.json)
# for each VM's forwarded SSH port and per-VM generated key
# (<work_dir>/id_ed25519_generated), then runs plain `ssh user@127.0.0.1
# -p <port> <command>` itself. That's the documented, stable way to script
# non-interactive commands against a vmctl-managed VM until vmctl grows a
# `ssh <name> -- <cmd>` passthrough.
#
# This is a long-running operation (image pull already cached separately,
# but boot + cloud-init + provisioning can take several minutes per VM).
# Run it detached if you might hit a session limit:
#   setsid nohup ./smoke.sh > logs/smoke.log 2>&1 &
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VMS_JSON="${HOME}/.local/share/vmctl/vms.json"
SSH_USER="vmctl"
FAIL=0

log() { printf '[smoke] %s\n' "$*"; }
pass() { printf '[smoke] PASS: %s\n' "$*"; }
bad() { printf '[smoke] FAIL: %s\n' "$*" >&2; FAIL=1; }

remote() {
    # remote <vm-name> <command...>
    local vm="$1"; shift
    local port key
    port=$(python3 -c "
import json
with open('${VMS_JSON}') as f:
    data = json.load(f)
print(data['$vm']['ssh_host_port'])
" 2>/dev/null)
    if [ -z "$port" ] || [ "$port" = "None" ]; then
        bad "$vm: could not read ssh_host_port from $VMS_JSON"
        return 1
    fi
    key=$(python3 -c "
import json
with open('${VMS_JSON}') as f:
    data = json.load(f)
print(data['$vm']['work_dir'])
" 2>/dev/null)/id_ed25519_generated
    if [ ! -f "$key" ]; then
        bad "$vm: generated key not found at $key"
        return 1
    fi
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=10 -i "$key" -p "$port" "${SSH_USER}@127.0.0.1" "$@"
}

log "gen-keys.sh (idempotent)"
bash scripts/gen-keys.sh

log "vmctl up --file VMFile.kdl (this pulls/boots/provisions all 3 VMs)"
if ! ~/.cargo/bin/vmctl up --file VMFile.kdl; then
    bad "vmctl up did not exit 0"
    exit 1
fi

log "--- exec1: akh-exec user ---"
if remote exec1 "id akh-exec && test \$(id -u akh-exec) -ne 0"; then
    pass "exec1: akh-exec exists and is non-root"
else
    bad "exec1: akh-exec check failed"
fi
if remote exec1 "svcs -H -o state network/ssh | grep -q online"; then
    pass "exec1: network/ssh SMF service online"
else
    bad "exec1: network/ssh service not online"
fi

log "--- pg: postgres + pgvector ---"
if remote pg "sudo -u postgres psql -tAc 'select 1'" | grep -q '^1$'; then
    pass "pg: postgres accepts connections"
else
    bad "pg: postgres connection check failed"
fi
if remote pg "sudo -u postgres psql -d akh_vm_smoke -tAc \"select extname from pg_extension where extname='vector'\"" | grep -q vector; then
    pass "pg: pgvector extension installed"
else
    bad "pg: pgvector extension check failed"
fi

log "--- builder: rust toolchain ---"
if remote builder "rustc --version"; then
    pass "builder: rustc runs"
else
    bad "builder: rustc check failed"
fi
if remote builder "cargo --version"; then
    pass "builder: cargo runs"
else
    bad "builder: cargo check failed"
fi

if [ "$FAIL" -eq 0 ]; then
    log "ALL CHECKS PASSED"
else
    log "ONE OR MORE CHECKS FAILED (see FAIL lines above)"
fi
exit "$FAIL"
