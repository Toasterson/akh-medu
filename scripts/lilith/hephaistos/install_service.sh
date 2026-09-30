#!/usr/bin/env bash
# Install and start lilith-serve.service on hephaistos (ADR 121). Idempotent.
#
#   bash scripts/lilith/hephaistos/install_service.sh              # install, enable, start, wait for /health
#   bash scripts/lilith/hephaistos/install_service.sh --no-start   # install only
#   bash scripts/lilith/hephaistos/install_service.sh --rotate-key # new API key, then restart
#
# Needs a verified llama-server build ($PREFIX/current, build_server.sh) and a verified converted
# model ($STATE/models/current, convert_parent.sh). Creates, once: the system user `lilith-serve`
# (groups render, video), /etc/lilith-serve/server.env from server.env.example (never overwritten),
# and /etc/lilith-serve/api-key (48 random bytes, 0600, owned by the service user; generated here and
# written through a pipe: it is never on a command line, in the unit or in the environment). The unit
# restarts the server only when the unit, the env file or the installed run_server.sh changed.
# Refusals (fail closed): a bind address that is not on this host or is 0.0.0.0, a port already in
# use by something else, a model whose provenance does not verify, the unit failing to reach /health.
set -euo pipefail
umask 022
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

START=1
ROTATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start) START=0; shift ;;
    --rotate-key) ROTATE=1; shift ;;
    -h | --help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die lilith::heph::usage "unknown argument '$1'" "see --help" ;;
  esac
done

UNIT_NAME=lilith-serve.service
UNIT_DEST="${LILITH_UNIT_DIR:-/etc/systemd/system}/$UNIT_NAME"
ENV_FILE="$ETC/server.env"
KEY_FILE="$ETC/api-key"
APPLIED="$STATE/applied.sha256"

[[ -x "$PREFIX/current/bin/llama-server" ]] || die lilith::heph::not_built "$PREFIX/current/bin/llama-server is missing" \
  "run scripts/lilith/hephaistos/build_server.sh first"
"${TOOLS[@]}" check-model "$STATE/models/current" >/dev/null 2>&1 || die lilith::heph::no_model \
  "no verified converted model at $STATE/models/current" "run scripts/lilith/hephaistos/convert_parent.sh first"
if [[ "${LILITH_NO_SYSTEMD:-0}" != 1 ]]; then
  need_cmd systemctl "this script installs a systemd unit"
fi

# ---- user and directories -----------------------------------------------------------------------
if ! id "$SVC_USER" >/dev/null 2>&1; then
  log "creating the system user $SVC_USER"
  as_root useradd --system --home-dir "$STATE" --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
fi
for group in render video; do
  if command -v getent >/dev/null && getent group "$group" >/dev/null; then as_root usermod -a -G "$group" "$SVC_USER"; fi
done
SVC_GROUP="$(id -gn "$SVC_USER")"
as_root install -d -m 755 "$PREFIX" "$PREFIX/bin" "$STATE" "$STATE/models"
as_root install -d -m 750 -o "$SVC_USER" -g "$SVC_GROUP" "$STATE/cache"
as_root install -d -m 750 -g "$SVC_GROUP" "$ETC"

# ---- env file (created once) --------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  as_root install -m 644 "$HEPH_DIR/server.env.example" "$ENV_FILE"
  log "created $ENV_FILE from server.env.example (edit it, then restart the unit)"
fi
env_value() { # env_value KEY: the value in server.env
  sed -n "s/^$1=//p" "$ENV_FILE" | tail -n1
}
HOST="$(env_value LILITH_LLAMA_HOST)"
PORT="$(env_value LILITH_LLAMA_PORT)"
case "$HOST" in
  "" | 0.0.0.0 | ::) die lilith::heph::bind "LILITH_LLAMA_HOST is '${HOST}' in $ENV_FILE" \
    "set it to the LAN address (192.168.1.114); the endpoint is plain HTTP, so it must not bind every interface" ;;
esac
if [[ "${LILITH_NO_SYSTEMD:-0}" != 1 ]]; then
  ip -4 -o addr show 2>/dev/null | grep "inet ${HOST}/" >/dev/null || die lilith::heph::bind \
    "$HOST is not an address of this host" "ip -4 addr; set LILITH_LLAMA_HOST in $ENV_FILE to this host's LAN address"
  if ss -H -ltn "sport = :$PORT" 2>/dev/null | grep . >/dev/null; then
    systemctl is-active --quiet "$UNIT_NAME" || die lilith::heph::port "port $PORT is already in use" \
      "ss -ltnp 'sport = :$PORT'; stop what holds it or change LILITH_LLAMA_PORT in $ENV_FILE"
  fi
fi

# ---- API key (created once; --rotate-key replaces it) -------------------------------------------
if [[ "$ROTATE" == 1 || ! -s "$KEY_FILE" ]]; then
  log "generating the API key ($KEY_FILE, 0600, owned by $SVC_USER)"
  # The key travels through a pipe into a file that is already 0600 and owned by the service user;
  # it is on no command line and is never printed.
  as_root rm -f "$KEY_FILE.new"
  as_root sh -c 'umask 077 && : > "$1" && chmod 600 "$1"' _ "$KEY_FILE.new"
  as_root chown "$SVC_USER:$SVC_GROUP" "$KEY_FILE.new"
  python3 -c 'import secrets,sys;sys.stdout.write(secrets.token_urlsafe(48)+"\n")' | as_root tee "$KEY_FILE.new" >/dev/null
  as_root mv -f "$KEY_FILE.new" "$KEY_FILE"
  KEY_CHANGED=1
else
  KEY_CHANGED=0
fi
[[ "$(as_root stat -c '%a %U' "$KEY_FILE" 2>/dev/null || echo unknown)" == "600 $SVC_USER" ]] \
  || [[ "${LILITH_NO_SYSTEMD:-0}" == 1 ]] || die lilith::heph::key "$KEY_FILE is not mode 600 owned by $SVC_USER" \
  "sudo chown $SVC_USER:$SVC_GROUP $KEY_FILE && sudo chmod 600 $KEY_FILE"

# ---- run script and unit --------------------------------------------------------------------------
as_root install -m 755 "$HEPH_DIR/run_server.sh" "$PREFIX/bin/run_server.sh"
UNIT_TMP="$(mktemp)"
trap 'rm -f "$UNIT_TMP"' EXIT
sed -e "s#^User=.*#User=$SVC_USER#" -e "s#^Group=.*#Group=$SVC_GROUP#" \
  -e "s#/opt/lilith-serve/bin/run_server.sh#$PREFIX/bin/run_server.sh#" \
  -e "s#/etc/lilith-serve/#$ETC/#g" -e "s#/var/lib/lilith-serve/cache#$STATE/cache#" \
  "$HEPH_DIR/$UNIT_NAME" > "$UNIT_TMP"
as_root install -d -m 755 "$(dirname "$UNIT_DEST")"
as_root install -m 644 "$UNIT_TMP" "$UNIT_DEST"

APPLIED_NOW="$( { cat "$UNIT_TMP" "$PREFIX/bin/run_server.sh"; as_root cat "$ENV_FILE"; readlink -f "$PREFIX/current" 2>/dev/null || true; \
  readlink -f "$STATE/models/current" 2>/dev/null || true; } | "${TOOLS[@]}" sha256 -)"
APPLIED_THEN="$(as_root cat "$APPLIED" 2>/dev/null || true)"

if [[ "${LILITH_NO_SYSTEMD:-0}" == 1 ]]; then
  log "LILITH_NO_SYSTEMD=1: unit installed at $UNIT_DEST, systemctl not run"
  exit 0
fi
as_root systemctl daemon-reload
as_root systemctl enable "$UNIT_NAME" >/dev/null 2>&1
if [[ "$START" == 1 ]]; then
  if ! systemctl is-active --quiet "$UNIT_NAME"; then
    log "starting $UNIT_NAME"
    as_root systemctl start "$UNIT_NAME"
  elif [[ "$APPLIED_NOW" != "$APPLIED_THEN" || "$KEY_CHANGED" == 1 ]]; then
    log "the unit, env, build, model or key changed: restarting $UNIT_NAME"
    as_root systemctl restart "$UNIT_NAME"
  else
    log "$UNIT_NAME is running and nothing changed"
  fi
  printf '%s\n' "$APPLIED_NOW" | as_root tee "$APPLIED" >/dev/null
  # Wait for /health: model load takes from 20 seconds (page cache warm) to a few minutes.
  for _ in $(seq 1 180); do
    if curl -fsS -m 3 -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; then
      log "healthy: http://$HOST:$PORT/health"
      log "API key fingerprint (sha256, first 8 hex): $(as_root cat "$KEY_FILE" | "${TOOLS[@]}" sha256 - | cut -c1-8)"
      log "next: python3 scripts/lilith/hephaistos/verify.py --base-url http://$HOST:$PORT --key-file <key file>"
      exit 0
    fi
    systemctl is-failed --quiet "$UNIT_NAME" && break
    sleep 5
  done
  as_root journalctl -u "$UNIT_NAME" -n 40 --no-pager >&2 || true
  die lilith::heph::health "$UNIT_NAME did not become healthy at http://$HOST:$PORT/health" \
    "read the journal above; run_server.sh prints the exact refusal (missing model, key mode, bind address); GPU busy with another job is the other usual cause (rocm-smi)"
fi
log "installed; start it with: sudo systemctl start $UNIT_NAME"
