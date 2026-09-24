#!/bin/bash
# Phase 53f (ADR 113): cut the Roman akhs over to the full akhomed built from main.
#
# Runs CUTOVER.md steps 2-8 in order and stops at the first failure. Run it as
# root (`sudo`), because the Roman core, runtimes, sandbox, executor and events
# are system LaunchDaemons. Every step that does not need root runs as the
# deployment owner (`sudo -u $OWNER`).
#
#   sudo deploy/roman-mini/cutover.sh <REPO>        # REPO = clean origin/main checkout, already built (CUTOVER.md step 1)
#   sudo DRY_RUN=1 deploy/roman-mini/cutover.sh <REPO>   # print the plan only
#
# Rollback: CUTOVER.md "Rollback" (the backup directory is printed at step 3).
set -euo pipefail

REPO="${1:?usage: cutover.sh <clean origin/main checkout with target/release built>}"
OWNER="${OWNER:-toasty}"
OWNER_UID="$(id -u "$OWNER")"
OWNER_HOME="$(dscl . -read "/Users/$OWNER" NFSHomeDirectory | awk '{print $2}')"
ROOT="${ROOT:-$OWNER_HOME/Library/Application Support/roman-akhs}"
BIN_DIR="${CARGO_TARGET_DIR:-$REPO/target}/release"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BK="$ROOT/backups/phase53-$TS"
GUI="gui/$OWNER_UID"
AGENTS="$OWNER_HOME/Library/LaunchAgents"
DAEMONS=/Library/LaunchDaemons
QUIET_TIMEOUT_SECS="${QUIET_TIMEOUT_SECS:-1200}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
run() { log "+ $*"; [ "$DRY_RUN" = 1 ] || "$@"; }
# Commands that do not need root run as the deployment owner.
owner() { if [ "$(id -u)" = 0 ]; then sudo -u "$OWNER" -H "$@"; else "$@"; fi; }
as_owner() { run owner "$@"; }
die() { log "ABORT: $*"; exit 1; }

[ "$(id -u)" = 0 ] || [ "$DRY_RUN" = 1 ] || die "run with sudo"
trap 'log "FAILED at line $LINENO. Admission may be paused and the core stopped; the backup is $BK. Follow CUTOVER.md Rollback."' ERR

# --- preconditions ------------------------------------------------------------
log "preconditions"
[ -d "$ROOT" ] || die "no deployment root at $ROOT"
[ -z "$(owner git -C "$REPO" status --porcelain)" ] || die "$REPO is not clean"
owner git -C "$REPO" fetch -q origin
SHA="$(owner git -C "$REPO" rev-parse --short=12 HEAD)"
owner git -C "$REPO" merge-base --is-ancestor HEAD origin/main || die "HEAD $SHA is not on origin/main"
log "provenance: $REPO is clean and HEAD $SHA is on origin/main"
for b in akhomed akh-acp akheventd akhsessd akhrtd akhsbxd akhexecd akh-trigger; do
  [ -x "$BIN_DIR/$b" ] || die "missing $BIN_DIR/$b (run CUTOVER.md step 1)"
done

# --- step 2: pause admission and wait for quiet --------------------------------
log "step 2: pause admission"
run launchctl bootout system/dev.roman-akhs.events || true
run launchctl bootout "$GUI/dev.roman-akhs.observer" || true
run launchctl bootout "$GUI/dev.roman-akhs.laya-controller" || true
if [ "$DRY_RUN" != 1 ]; then
  deadline=$(( $(date +%s) + QUIET_TIMEOUT_SECS ))
  while :; do
    running="$(curl -s -H "Authorization: Bearer $(cat "$ROOT/secrets/operator.key")" \
      'http://127.0.0.1:18431/v1/sessions?compact=true' | python3 -c \
      'import json,sys; s=json.load(sys.stdin); r=[x.get("id") for x in s if x.get("status")=="running"]; print(len(r))')"
    [ "$running" = 0 ] && break
    [ "$(date +%s)" -lt "$deadline" ] || die "$running turn(s) still running after ${QUIET_TIMEOUT_SECS}s; admission stays paused, re-run later"
    log "waiting for $running running turn(s)"; sleep 15
  done
  log "quiet"
fi

# --- step 3: stop the core and back it up ------------------------------------
log "step 3: stop core, back up to $BK"
run launchctl bootout system/dev.roman-akhs.core || true
as_owner mkdir -p "$BK"
as_owner chmod 700 "$BK"
as_owner sqlite3 "$ROOT/state/core/akh.sqlite3" ".backup '$BK/akh.sqlite3'"
if [ "$DRY_RUN" != 1 ]; then
  [ "$(owner sqlite3 "$BK/akh.sqlite3" 'pragma integrity_check;')" = ok ] || die "backup integrity check failed"
fi
as_owner sh -c "shasum -a 256 '$ROOT/state/core/akh.sqlite3' '$BK/akh.sqlite3' > '$BK/SHA256SUMS'"
as_owner cp -Rp "$ROOT/core-config" "$ROOT/config" "$BK/"
run cp -p "$DAEMONS"/dev.roman-akhs.*.plist "$BK/"
as_owner sh -c "cp -p '$AGENTS'/dev.roman-akhs.*.plist '$BK/'"
for f in t3_entrypoint.py roman-janus daemon_credentials.py observer.py; do as_owner cp -p "$ROOT/bin/$f" "$BK/"; done

# --- step 4: install binaries and render the full-akh configuration ----------
log "step 4: render configuration and install binaries from $SHA"
as_owner python3 "$REPO/deploy/roman-mini/configure_core.py" --root "$ROOT" --binaries "$BIN_DIR" --repo "$REPO" --no-fetch
as_owner python3 -c 'import tomllib,sys;[tomllib.load(open(p,"rb")) for p in sys.argv[1:]];print("toml ok")' \
  "$ROOT/core-config/akh-medu/config.toml" "$ROOT"/core-config/akh-medu/workspaces/{janus,minerva,vesta}.toml
for b in akheventd akhsessd akhrtd akhsbxd akhexecd akh-trigger; do
  as_owner cp -p "$ROOT/bin/$b" "$BK/$b.pre-phase53"
  as_owner install -m 755 "$BIN_DIR/$b" "$ROOT/bin/$b"
done
as_owner sh -c "shasum -a 256 '$ROOT'/bin/{akheventd,akhsessd,akhrtd,akhsbxd,akhexecd,akh-trigger} > '$ROOT/evidence/services-main-$SHA.sha256'"
for r in janus minerva vesta; do as_owner install -m 700 "$REPO/deploy/roman-mini/t3_entrypoint.py" "$ROOT/bin/roman-$r"; done
as_owner cp -p "$REPO/deploy/roman-mini/observer.py" "$ROOT/bin/observer.py"

# --- step 4 wire order + step 5: readers first, then the core, akhexecd last --
log "step 5: restart readers (akhsessd, akhsbxd), start the full core"
run launchctl kickstart -k "$GUI/dev.roman-akhs.sessions"
run launchctl kickstart -k system/dev.roman-akhs.sandbox
run install -m 0644 -o root -g wheel "$ROOT/launchd/dev.roman-akhs.core.plist" "$DAEMONS/dev.roman-akhs.core.plist"
run launchctl bootstrap system "$DAEMONS/dev.roman-akhs.core.plist"
if [ "$DRY_RUN" != 1 ]; then
  for _ in $(seq 1 60); do curl -sf http://127.0.0.1:18430/healthz >/dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:18430/healthz >/dev/null || die "core /healthz not up after 300s (see $ROOT/logs/core.err.log; rollback per CUTOVER.md)"
  pgrep -f "akhomed-main-$SHA" >/dev/null || die "running core is not akhomed-main-$SHA"
fi
log "core up: akhomed-main-$SHA"
for label in runtime runtime-minerva runtime-vesta; do run launchctl kickstart -k "system/dev.roman-akhs.$label"; done
run launchctl kickstart -k system/dev.roman-akhs.executor

# --- step 7: reactor seeds ------------------------------------------------------
log "step 7: apply and enable reactor seeds"
as_owner python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT"
as_owner python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT" --enable

# --- step 8: re-target events and resume admission ---------------------------
log "step 8: re-target events, resume admission"
as_owner python3 "$REPO/deploy/roman-mini/retarget_events.py" "$ROOT"
run launchctl bootstrap system "$DAEMONS/dev.roman-akhs.events.plist"
run launchctl bootstrap "$GUI" "$AGENTS/dev.roman-akhs.observer.plist"
run launchctl bootstrap "$GUI" "$AGENTS/dev.roman-akhs.laya-controller.plist"

log "cutover complete: akhomed-main-$SHA; backup $BK"
log "next: CUTOVER.md step 6 checks, step 9 verification, step 10 hygiene"
