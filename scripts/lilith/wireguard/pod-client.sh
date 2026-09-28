#!/usr/bin/env bash
# Lilith C6a pod-side WireGuard client.
#
# Runs inside the ephemeral Runpod pod. Reads its entire configuration from
# environment variables injected by the supervisor as Runpod secrets — the
# exact values `peer.py add` printed to stdout once. That bundle (and the
# private key inside it) is never written into the pod image or onto the
# shared network volume; it exists only as process-start secrets for this
# one pod session.
#
# Fails closed: if the capability check, the tunnel client itself, or any
# allowlisted-service health check fails, this script exits non-zero and
# never reports the pod ready. There is no silent "tunnel down, proceed
# anyway" path.
#
# Required env:
#   LILITH_WG_PRIVATE_KEY   pod's own WireGuard private key (ephemeral, minted by `peer.py add`)
#   LILITH_WG_ADDRESS       pod's tunnel address, e.g. "10.73.0.17/32"
#   LILITH_WG_HUB_PUBKEY    hub's public key
#   LILITH_WG_HUB_ENDPOINT  hub's public endpoint, "<host>:<port>"
#   LILITH_WG_TUNNEL_SUBNET the whole tunnel subnet, e.g. "10.73.0.0/24"
#   LILITH_WG_ONETUN_CMDS   newline-separated onetun invocations (peer.py bundle: onetun_commands)
#   LILITH_WG_SERVICES      newline-separated "name local_port protocol" triples to health-check
#
# Optional env:
#   LILITH_WG_CLIENT             force "onetun" | "wireproxy" | "kernel" | "auto" (default: auto)
#   LILITH_WG_WIREPROXY_CONF     rendered wireproxy.conf text (only used if LILITH_WG_CLIENT=wireproxy,
#                                 or auto falls back to wireproxy)
#   LILITH_WG_KERNEL_CONF        rendered wg-quick conf text, private key inline (only used if
#                                 LILITH_WG_CLIENT=kernel, or auto detects real kernel capability)
#   LILITH_WG_HEALTHCHECK_TIMEOUT_S  per-attempt TCP connect timeout in seconds (default: 5)
#   LILITH_WG_HEALTHCHECK_RETRIES    attempts before failing a service (default: 6)
#   LILITH_WG_HOSTS              Variant C: one /etc/hosts line ("127.0.0.1 seshat.lab.home
#                                 board.lab.home") so HTTP clients send the right Host header
#                                 through the lab-gateway forward. Appended once if missing.
#
# NOTE on onetun's --private-key: the rendered onetun commands read the key
# via ordinary shell expansion of $LILITH_WG_PRIVATE_KEY at the point this
# script execs them, so the key appears in this pod's own argv for the
# lifetime of the onetun process (visible only inside this single-tenant
# container via ps/proc). Before first real use, check `onetun --help` for
# a same-effect environment-variable flag that avoids this; that is a
# hardening step to verify against the deployed onetun version, not
# something this script can assume.

set -euo pipefail

CLIENT_MODE="${LILITH_WG_CLIENT:-auto}"
TIMEOUT_S="${LILITH_WG_HEALTHCHECK_TIMEOUT_S:-5}"
RETRIES="${LILITH_WG_HEALTHCHECK_RETRIES:-6}"

log() { printf '[pod-client] %s\n' "$*" >&2; }
fail() { log "FATAL: $*"; exit 1; }

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    fail "required env var $name is not set"
  fi
}

for var in LILITH_WG_PRIVATE_KEY LILITH_WG_ADDRESS LILITH_WG_HUB_PUBKEY \
           LILITH_WG_HUB_ENDPOINT LILITH_WG_TUNNEL_SUBNET LILITH_WG_SERVICES; do
  require_env "$var"
done

CHILD_PIDS=()
cleanup() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# A real capability probe, not just a device-node check: Runpod pods often
# expose /dev/net/tun as a device node without granting the NET_ADMIN
# capability actually needed to use it, so also probe for the privilege
# with a harmless dummy-interface add/delete.
detect_kernel_capable() {
  [ -c /dev/net/tun ] || return 1
  command -v wg-quick >/dev/null 2>&1 || command -v wireguard-go >/dev/null 2>&1 || return 1
  if ip link add __lilith_probe type dummy >/dev/null 2>&1; then
    ip link del __lilith_probe >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

start_kernel() {
  command -v wg-quick >/dev/null 2>&1 || fail "kernel mode requested but wg-quick is not installed"
  require_env LILITH_WG_KERNEL_CONF
  install -d -m 0700 /etc/wireguard
  printf '%s' "$LILITH_WG_KERNEL_CONF" > /etc/wireguard/wg0.conf
  chmod 0600 /etc/wireguard/wg0.conf
  wg-quick up wg0
  log "kernel WireGuard up on wg0"
}

start_wireproxy() {
  command -v wireproxy >/dev/null 2>&1 || fail "wireproxy mode requested but wireproxy is not installed"
  require_env LILITH_WG_WIREPROXY_CONF
  local conf
  conf="$(mktemp /tmp/lilith-wireproxy.XXXXXX.conf)"
  printf '%s' "$LILITH_WG_WIREPROXY_CONF" > "$conf"
  chmod 0600 "$conf"
  wireproxy -c "$conf" &
  CHILD_PIDS+=("$!")
  log "wireproxy started (pid $!) from $conf"
}

start_onetun() {
  command -v onetun >/dev/null 2>&1 || fail "onetun mode requested but onetun is not installed"
  require_env LILITH_WG_ONETUN_CMDS
  local count=0 cmd
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    count=$((count + 1))
    log "starting onetun process #$count"
    eval "$cmd" &
    CHILD_PIDS+=("$!")
  done <<< "$LILITH_WG_ONETUN_CMDS"
  [ "$count" -gt 0 ] || fail "LILITH_WG_ONETUN_CMDS was empty — nothing to forward"
}

# Variant C: forwarded HTTP virtual hosts resolve to the local forward.
if [ -n "${LILITH_WG_HOSTS:-}" ]; then
  if grep -qxF "$LILITH_WG_HOSTS" /etc/hosts 2>/dev/null; then
    :
  elif [ -w /etc/hosts ]; then
    printf '%s\n' "$LILITH_WG_HOSTS" >> /etc/hosts
    log "mapped forwarded virtual hosts: $LILITH_WG_HOSTS"
  else
    fail "LILITH_WG_HOSTS is set but /etc/hosts is not writable; send explicit Host headers instead"
  fi
fi

case "$CLIENT_MODE" in
  kernel)
    start_kernel
    ;;
  wireproxy)
    start_wireproxy
    ;;
  onetun)
    start_onetun
    ;;
  auto)
    if detect_kernel_capable; then
      log "kernel WireGuard capability detected — using kernel mode"
      start_kernel
    elif command -v onetun >/dev/null 2>&1; then
      log "no kernel WireGuard capability — using onetun (userspace, preferred fallback)"
      start_onetun
    elif command -v wireproxy >/dev/null 2>&1; then
      log "no kernel WireGuard capability and no onetun — using wireproxy (userspace)"
      start_wireproxy
    else
      fail "no usable WireGuard client found: no kernel capability, no onetun, no wireproxy"
    fi
    ;;
  *)
    fail "unknown LILITH_WG_CLIENT=$CLIENT_MODE (expected one of: auto onetun wireproxy kernel)"
    ;;
esac

# Give the tunnel a moment to complete its first handshake before probing.
sleep 2

log "health-checking every allowlisted service (fail closed on any miss)..."
FAILED=0
while IFS=' ' read -r name port protocol; do
  [ -z "$name" ] && continue
  if [ "$protocol" = "udp" ]; then
    log "SKIP $name (127.0.0.1:$port/udp) — no reliable connect-style check for UDP; verify manually"
    continue
  fi
  ok=0
  attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    if timeout "$TIMEOUT_S" bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null; then
      exec 3<&- 3>&- 2>/dev/null || true
      ok=1
      break
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  if [ "$ok" -eq 1 ]; then
    log "OK   $name (127.0.0.1:$port)"
  else
    log "FAIL $name (127.0.0.1:$port) unreachable after $RETRIES attempts"
    FAILED=1
  fi
done <<< "$LILITH_WG_SERVICES"

if [ "$FAILED" -ne 0 ]; then
  fail "one or more allowlisted services are unreachable through the tunnel"
fi

log "all TCP allowlisted services reachable; tunnel is up"
wait
