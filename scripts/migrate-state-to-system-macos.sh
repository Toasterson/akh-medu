#!/bin/bash
# Move an akhomed's state out of a user's home and into the macOS system
# locations, for a daemon deployment.
#
# WHY
#
# akhomed now runs as a system LaunchDaemon (root), but its paths still resolved
# through $HOME, so the daemon read and wrote a *user's* tree. That is wrong in
# three separate ways:
#
#   * Files the daemon writes there are root-owned, inside a directory the user
#     owns. The user can read them (0644) but cannot rewrite them, so running
#     the `akh` CLI as themselves hits permission errors on write. This already
#     happened to ~/.local/state/akh-medu/run/akhomed.pid.
#   * A system service's lifetime gets coupled to an account. Delete or rename
#     the user and the daemon's state goes with it.
#   * On macOS a user's home is inside per-user privacy (TCC) and FileVault
#     scope. That is the same class of problem that made the LaunchAgent
#     unable to reach the LAN Postgres in the first place.
#
# WHAT MOVES (see AkhPaths::system_daemon in src/paths.rs — this script and that
# function must agree, and there is a test asserting the layout)
#
#   ~/.config/akh-medu       -> /Library/Application Support/akh-medu   (COPIED)
#   ~/.local/share/akh-medu  -> /Library/Application Support/akh-medu/data (MOVED)
#   ~/.local/state/akh-medu  -> /var/db/akh-medu                        (MOVED)
#   (pid/socket)             -> /var/run/akh-medu                       (fresh)
#
# Config is *copied*, not moved: it is small, and leaving the user's copy in
# place keeps a working fallback and a trivial rollback. Note the consequence —
# after this, the daemon reads the system copy, so edit that one. Data and state
# are moved, because 15 GB of models has no business being duplicated.
#
# Both source and destination are on the same APFS volume, so the moves are
# renames: instant, and they need no free space.
#
# Usage:  sudo scripts/migrate-state-to-system-macos.sh [--user <name>] [--dry-run]
set -euo pipefail

LABEL="dev.akh-medu.akhomed"
SUPPORT="/Library/Application Support/akh-medu"
SYS_DATA="$SUPPORT/data"
SYS_STATE="/var/db/akh-medu"
SYS_RUN="/var/run/akh-medu"
SYS_CACHE="/Library/Caches/akh-medu"

MIG_USER="${SUDO_USER:-}"
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --user) MIG_USER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" -eq 0 ] || { echo "ERROR: must run as root (sudo)" >&2; exit 1; }
[ -n "$MIG_USER" ] || { echo "ERROR: no user; pass --user <name>" >&2; exit 1; }
id "$MIG_USER" >/dev/null 2>&1 || { echo "ERROR: no such user: $MIG_USER" >&2; exit 1; }

USER_HOME=$(eval echo "~$MIG_USER")
OLD_CONFIG="$USER_HOME/.config/akh-medu"
OLD_DATA="$USER_HOME/.local/share/akh-medu"
OLD_STATE="$USER_HOME/.local/state/akh-medu"

run() { if [ "$DRY_RUN" = 1 ]; then echo "  DRY-RUN: $*"; else "$@"; fi; }

echo "migrating akhomed state to the macOS system locations"
echo "  user  : $MIG_USER ($USER_HOME)"
[ "$DRY_RUN" = 1 ] && echo "  MODE  : dry run — nothing will be changed"

# --- 1. the daemon must not be running ---------------------------------------
# It holds these directories open and rewrites state (WAL, snapshots, PID) on a
# timer. Moving them underneath a live daemon is how you get a half-written
# workspace.
if launchctl print "system/$LABEL" >/dev/null 2>&1; then
  echo "  stopping daemon"
  run launchctl bootout "system/$LABEL" 2>/dev/null || true
  for _ in $(seq 1 50); do
    launchctl print "system/$LABEL" >/dev/null 2>&1 || break
    sleep 0.2
  done
  if [ "$DRY_RUN" = 0 ] && launchctl print "system/$LABEL" >/dev/null 2>&1; then
    echo "ERROR: daemon still loaded after bootout; aborting before touching state" >&2
    exit 1
  fi
else
  echo "  daemon not loaded"
fi

# --- 2. refuse to clobber an existing migration ------------------------------
# Re-running must not silently merge a stale system tree with a live user one.
for d in "$SYS_DATA" "$SYS_STATE"; do
  if [ -e "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null || true)" ]; then
    echo "ERROR: $d already exists and is not empty." >&2
    echo "       This looks like a previous migration. Refusing to merge trees." >&2
    echo "       Inspect it, then either keep it (skip this script) or remove it." >&2
    exit 1
  fi
done

# --- 3. create the system tree ----------------------------------------------
echo "  creating system directories"
run install -d -o root -g wheel -m 0755 "$SUPPORT"
run install -d -o root -g wheel -m 0755 "$SYS_CACHE"
run install -d -o root -g wheel -m 0755 "$SYS_RUN"
# /var/db already exists on macOS and carries SIP's `restricted` flag, so
# `install -d` on it fails with "Operation not permitted" even as root — it
# tries to chown/chmod an existing system directory. We only ever need it to
# exist, and it always does; creating it is for the non-macOS path.
[ -d /var/db ] || run install -d -o root -g wheel -m 0755 /var/db

# --- 4. config: copy ---------------------------------------------------------
if [ -d "$OLD_CONFIG" ]; then
  echo "  copying config  ($(du -sh "$OLD_CONFIG" 2>/dev/null | cut -f1)) -> $SUPPORT"
  # -a preserves modes, which matters: certs/ is 0600 and must stay that way.
  run rsync -a "$OLD_CONFIG"/ "$SUPPORT"/
  run chown -R root:wheel "$SUPPORT"
else
  echo "  no config at $OLD_CONFIG — skipping"
fi

# --- 5. data: move -----------------------------------------------------------
if [ -d "$OLD_DATA" ]; then
  echo "  moving data     ($(du -sh "$OLD_DATA" 2>/dev/null | cut -f1)) -> $SYS_DATA"
  run mv "$OLD_DATA" "$SYS_DATA"
  run chown -R root:wheel "$SYS_DATA"
else
  echo "  no data at $OLD_DATA — creating empty"
  run install -d -o root -g wheel -m 0755 "$SYS_DATA"
fi

# --- 6. state: move ----------------------------------------------------------
if [ -d "$OLD_STATE" ]; then
  echo "  moving state    ($(du -sh "$OLD_STATE" 2>/dev/null | cut -f1)) -> $SYS_STATE"
  run mv "$OLD_STATE" "$SYS_STATE"
  run chown -R root:wheel "$SYS_STATE"
  # The old run/ dir came along; the daemon writes a fresh PID into /var/run.
  run rm -rf "$SYS_STATE/run"
else
  echo "  no state at $OLD_STATE — creating empty"
  run install -d -o root -g wheel -m 0755 "$SYS_STATE"
fi

# --- 7. permissions ----------------------------------------------------------
# The PID file must be world-readable so an unprivileged `akh` can discover the
# daemon (it carries a pid and endpoints, no secrets). Private key material must
# not be.
if [ "$DRY_RUN" = 0 ]; then
  chmod 0755 "$SYS_RUN"
  if [ -d "$SUPPORT/certs" ]; then
    chmod 0700 "$SUPPORT/certs"
    find "$SUPPORT/certs" -type f -exec chmod 0600 {} +
    echo "  certs re-secured: 0700 dir / 0600 files, root:wheel"
  fi
fi

echo
if [ "$DRY_RUN" = 1 ]; then
  echo "dry run complete — nothing changed."
  exit 0
fi

cat <<SUMMARY
migration complete.

  config   $SUPPORT              (copied; edit THIS one from now on)
  data     $SYS_DATA
  state    $SYS_STATE
  runtime  $SYS_RUN
  cache    $SYS_CACHE

The originals that were COPIED are still in place:
  $OLD_CONFIG

The originals that were MOVED are gone. To roll back:
  sudo launchctl bootout system/$LABEL
  sudo mv "$SYS_DATA" "$OLD_DATA"
  sudo mv "$SYS_STATE" "$OLD_STATE"
  sudo chown -R $MIG_USER "$OLD_DATA" "$OLD_STATE"
  # then re-run install-launchdaemon-macos.sh from a commit without AKH_SYSTEM_PATHS

Next: re-run the installer so the daemon uses the system layout.
  sudo scripts/install-launchdaemon-macos.sh --user $MIG_USER
SUMMARY
