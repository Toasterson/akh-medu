#!/bin/bash
# Install akhomed as a macOS *system* LaunchDaemon with a privilege-dropped
# executor.
#
# WHY A DAEMON AND NOT AN AGENT
#
# akhomed ran as a per-user LaunchAgent (`~/Library/LaunchAgents`). Two things
# are wrong with that for a machine acting as an executor host:
#
#   1. **Local Network TCC.** macOS gates LAN access per-executable, per-user.
#      A LaunchAgent-spawned akhomed is silently denied — the LAN Postgres
#      connect fails with `No route to host` while the identical URL works from
#      Terminal — so the akh falls back to `postgres=false` and loses shared
#      truth. The grant is also lost on every rebuild, because the binary's
#      identity changes. The system domain is not subject to per-user Local
#      Network TCC, so a LaunchDaemon fixes this permanently rather than until
#      the next `cargo build`.
#   2. **It only runs while a GUI user is logged in.** An executor host has to
#      keep working across logout and reboot.
#
# WHY THE EXECUTOR MUST DROP PRIVILEGE
#
# A LaunchDaemon runs as root. Without `[exec] kind = "local-drop"` that would
# hand root to every `shell_exec` the agent issues. This script therefore
# configures the drop at the same time it installs the daemon — the two changes
# are only safe together, so they ship together.
#
# WHAT GETS INSTALLED (all root-owned; a root daemon must never exec anything
# writable by a non-root user, or that user can trivially become root)
#
#   /usr/local/libexec/akhomed              the binary        root:wheel 0755
#   /usr/local/libexec/akhomed-daemon.sh    the wrapper       root:wheel 0755
#   /usr/local/etc/akh-medu/daemon.env      secrets           root:wheel 0600
#   /Library/LaunchDaemons/dev.akh-medu.akhomed.plist         root:wheel 0644
#
# Note the binary is *copied* to /usr/local/libexec rather than referenced in
# the build tree. The old LaunchAgent exec'd
# `~/ws/akh-medu/target/release/akhomed`, which meant a `git checkout` or
# `cargo clean` silently changed — or deleted — what the service would run on
# next restart. It had in fact been crash-looping for three weeks because that
# path did not exist.
#
# Re-run this script after every rebuild to publish a new binary.
#
# Usage:  sudo scripts/install-launchdaemon-macos.sh [--user <exec-user>]
set -euo pipefail

LABEL="dev.akh-medu.akhomed"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
LIBEXEC="/usr/local/libexec"
ETC="/usr/local/etc/akh-medu"
BIN_SRC="${BIN_SRC:-$(cd "$(dirname "$0")/.." && pwd)/target/release/akhomed}"

# The user commands run as, and whose config/state the daemon uses. Defaults to
# the owner of the invoking sudo session, which is almost always right.
EXEC_USER="${SUDO_USER:-}"
# Workspaces the daemon brings up at boot. Kept as a flag rather than hardcoded
# because relocating an akh between hosts (ADR 049) means changing exactly this
# list on two machines, and editing an installer to move an akh is how the two
# ends drift apart.
AUTO_START="${AKH_AUTO_START:-thoth-mac,gemma-mac}"
while [ $# -gt 0 ]; do
  case "$1" in
    --user) EXEC_USER="$2"; shift 2 ;;
    --autostart) AUTO_START="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root (use sudo) — installing into /Library and /usr/local" >&2
  exit 1
fi
if [ -z "$EXEC_USER" ] || [ "$EXEC_USER" = "root" ]; then
  echo "ERROR: refusing to run the executor as root. Pass --user <name>." >&2
  echo "       local-drop exists to execute as a *less* privileged principal." >&2
  exit 1
fi
if ! id "$EXEC_USER" >/dev/null 2>&1; then
  echo "ERROR: exec user '$EXEC_USER' does not exist (akh-medu never creates accounts)" >&2
  exit 1
fi
if [ ! -x "$BIN_SRC" ]; then
  echo "ERROR: no akhomed binary at $BIN_SRC" >&2
  echo "       build it first: cargo build --release --features server --bin akhomed" >&2
  exit 1
fi

USER_HOME=$(eval echo "~$EXEC_USER")
echo "installing $LABEL"
echo "  exec user : $EXEC_USER (uid $(id -u "$EXEC_USER"), gid $(id -g "$EXEC_USER"))"
echo "  home      : $USER_HOME"
echo "  binary    : $BIN_SRC"

# --- stop whatever is running now -------------------------------------------
# Both domains: the LaunchAgent this replaces, and any earlier daemon.
if launchctl print "system/$LABEL" >/dev/null 2>&1; then
  echo "  unloading existing daemon"
  launchctl bootout "system/$LABEL" 2>/dev/null || true
  # `bootout` returns before the service is actually gone. Bootstrapping into a
  # domain that still holds the label fails with the famously unhelpful
  # `Bootstrap failed: 5: Input/output error`, and because the old daemon has
  # already stopped by then, the machine is left with no akhomed at all. Wait
  # for the label to actually disappear.
  for _ in $(seq 1 50); do
    launchctl print "system/$LABEL" >/dev/null 2>&1 || break
    sleep 0.2
  done
  if launchctl print "system/$LABEL" >/dev/null 2>&1; then
    echo "ERROR: '$LABEL' is still loaded 10s after bootout; refusing to continue" >&2
    echo "       inspect with: sudo launchctl print system/$LABEL" >&2
    exit 1
  fi
fi
AGENT_PLIST="$USER_HOME/Library/LaunchAgents/${LABEL}.plist"
if [ -f "$AGENT_PLIST" ]; then
  EXEC_UID=$(id -u "$EXEC_USER")
  if launchctl print "gui/$EXEC_UID/$LABEL" >/dev/null 2>&1; then
    echo "  unloading superseded LaunchAgent (gui/$EXEC_UID)"
    launchctl bootout "gui/$EXEC_UID/$LABEL" 2>/dev/null || true
  fi
  mv "$AGENT_PLIST" "${AGENT_PLIST}.superseded-by-daemon"
  echo "  LaunchAgent plist renamed to ${AGENT_PLIST##*/}.superseded-by-daemon"
fi

# --- binary ------------------------------------------------------------------
install -d -o root -g wheel -m 0755 "$LIBEXEC"
install -o root -g wheel -m 0755 "$BIN_SRC" "$LIBEXEC/akhomed"

# Codesign with a stable identity if one is available. Unlike the LaunchAgent
# this is not load-bearing for LAN access any more (the system domain is not
# subject to per-user Local Network TCC), but a stable identity still keeps the
# binary's identity constant across rebuilds for anything else that keys off it.
CODESIGN_ID="${AKH_CODESIGN_ID:-Apple Development: toasterson@gmail.com (QPJC8WM3MN)}"
if security find-identity -v -p codesigning 2>/dev/null | grep -qF "$CODESIGN_ID"; then
  codesign --force --sign "$CODESIGN_ID" --identifier "$LABEL" "$LIBEXEC/akhomed" 2>/dev/null \
    && echo "  signed with: $CODESIGN_ID" \
    || echo "  WARNING: codesign failed; continuing unsigned" >&2
else
  echo "  note: signing identity not found; installing ad-hoc-signed binary"
fi

# --- secrets -----------------------------------------------------------------
# Copied out of the user's ~/env-file into root-owned storage. The daemon must
# not source a file the exec user can write: that user could inject DYLD_* and
# become root.
install -d -o root -g wheel -m 0755 "$ETC"
if [ -f "$USER_HOME/env-file" ]; then
  install -o root -g wheel -m 0600 "$USER_HOME/env-file" "$ETC/daemon.env"
  echo "  secrets   : copied from $USER_HOME/env-file (re-run after rotating keys)"
else
  [ -f "$ETC/daemon.env" ] || install -o root -g wheel -m 0600 /dev/null "$ETC/daemon.env"
  echo "  secrets   : no ~/env-file found; $ETC/daemon.env left as-is"
fi

# --- layout: system or home? -------------------------------------------------
# Prefer the system layout (AkhPaths::system_daemon), but only when the state is
# actually there. Setting AKH_SYSTEM_PATHS=1 against empty system directories
# would start the daemon with no workspaces and quietly mint fresh ones, while
# the real state sat unreferenced in the user's home — a data-loss-shaped
# failure that looks like "the akh forgot everything".
SUPPORT="/Library/Application Support/akh-medu"
if [ -f "$SUPPORT/config.toml" ]; then
  AKH_PATHS_MODE="system"
  DAEMON_HOME="/var/root"
  echo "  layout    : system ($SUPPORT)"
else
  AKH_PATHS_MODE="home"
  DAEMON_HOME="$USER_HOME"
  echo "  layout    : home ($USER_HOME) — state still lives in the user's tree"
  echo "              run scripts/migrate-state-to-system-macos.sh to move it"
fi

# --- wrapper -----------------------------------------------------------------
cat > "$LIBEXEC/akhomed-daemon.sh" <<WRAPPER
#!/bin/bash
# Root-owned launcher for the akhomed LaunchDaemon. Generated by
# scripts/install-launchdaemon-macos.sh — edit that, not this.
set -a
[ -f "$ETC/daemon.env" ] && . "$ETC/daemon.env"
set +a

# The per-akh config resolves \`api_key = "env:OLLAMA_CLOUD_KEY"\`.
export OLLAMA_CLOUD_KEY="\${OLLAMA_CLOUD_KEY:-\${OLLAMA_KEY:-}}"

# Path layout, resolved by the installer (see AkhPaths in src/paths.rs):
#
#   system — AKH_SYSTEM_PATHS=1 selects the macOS daemon locations
#            (/Library/Application Support/akh-medu, /var/db, /var/run) and HOME
#            is root's own. Nothing the daemon owns lives in a user's tree.
#
#   home   — the pre-migration arrangement: paths resolve through a user's HOME.
#            Works, but the daemon leaves root-owned files in that user's
#            directories, which they then cannot rewrite.
#
# Child processes are unaffected either way: the local-drop executor sets HOME
# to the *exec principal's* home for every command it runs, so builds still see
# the user's caches and dotfiles.
export AKH_PATHS_MODE="$AKH_PATHS_MODE"
[ "\$AKH_PATHS_MODE" = "system" ] && export AKH_SYSTEM_PATHS=1
export HOME="$DAEMON_HOME"

exec "$LIBEXEC/akhomed"
WRAPPER
chown root:wheel "$LIBEXEC/akhomed-daemon.sh"
chmod 0755 "$LIBEXEC/akhomed-daemon.sh"

# --- plist -------------------------------------------------------------------
# No UserName key: the daemon runs as root deliberately, and akhomed's own
# local-drop executor lowers privilege per command. ProcessType Background asks
# the scheduler not to treat it as interactive.
cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$LIBEXEC/akhomed-daemon.sh</string>
    </array>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>StandardOutPath</key>
    <string>/var/log/akhomed.out.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/akhomed.err.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>AKH_LLM_MODEL</key>
        <string>gemma3-4b</string>
        <key>AKH_NLU_LLM</key>
        <string>0</string>
        <key>AKH_AUTO_START</key>
        <string>$AUTO_START</string>
        <!-- The executor backends log under the akh::backend::* target, not
             akh_medu::backend::* (see build_local_drop_backend in
             src/bin/akhomed.rs). Naming the crate path here instead silently
             filters out the one line that tells an operator which principal
             the akh executes as.
             NOTE: this heredoc is unquoted so it can interpolate $LABEL and
             $LIBEXEC, which means backticks here would be run as commands and
             would corrupt the plist. Keep this comment backtick-free. -->
        <key>RUST_LOG</key>
        <string>akhomed=info,akhomed_store=info,akh_medu=warn,akh_medu::engine=error,akh::backend=info</string>
    </dict>

    <key>SoftResourceLimits</key>
    <dict>
        <key>NumberOfFiles</key>
        <integer>10240</integer>
    </dict>
</dict>
</plist>
PLISTEOF
chown root:wheel "$PLIST"
chmod 0644 "$PLIST"

# Validate before loading. The heredoc above is unquoted (it interpolates
# $LABEL and $LIBEXEC), so any shell-active character that sneaks into it —
# a backtick in a comment did exactly this once — silently produces malformed
# XML. launchctl's response to that is `Bootstrap failed: 5: Input/output
# error`, which names neither the file nor the problem, and by then the old
# daemon has already been unloaded. Catch it here, while the operator still
# has a useful error and the previous state is recoverable.
if ! plutil -lint "$PLIST" >/dev/null 2>&1; then
  echo "ERROR: generated plist is not valid XML — refusing to load it" >&2
  plutil -lint "$PLIST" >&2 || true
  echo "       (the daemon is currently unloaded; fix this script and re-run)" >&2
  exit 1
fi

touch /var/log/akhomed.out.log /var/log/akhomed.err.log
chown root:wheel /var/log/akhomed.out.log /var/log/akhomed.err.log

# --- load --------------------------------------------------------------------
if ! launchctl bootstrap system "$PLIST"; then
  echo "ERROR: launchctl bootstrap failed. Common causes:" >&2
  echo "  - the label is still loaded (see the bootout wait above)" >&2
  echo "  - the plist is not root-owned, or is group/world-writable" >&2
  echo "  - $LIBEXEC/akhomed-daemon.sh is not executable" >&2
  echo "Diagnose with: sudo launchctl print system/$LABEL" >&2
  exit 1
fi
launchctl enable "system/$LABEL" 2>/dev/null || true

# Bootstrap succeeding only means launchd accepted the job. Confirm it actually
# came up rather than crash-looping — the LaunchAgent this replaces spent three
# weeks "installed" while exiting 78 on every spawn.
sleep 2
if launchctl print "system/$LABEL" 2>/dev/null | grep -q "state = running"; then
  echo "  daemon state: running"
else
  echo "  WARNING: bootstrapped but not yet running — check /var/log/akhomed.err.log" >&2
  launchctl print "system/$LABEL" 2>/dev/null | grep -E "state =|last exit" >&2 || true
fi

echo
echo "loaded. verify with:"
echo "  sudo launchctl print system/$LABEL | grep -E 'state|pid|last exit'"
echo "  tail -f /var/log/akhomed.out.log"
echo
echo "REMAINING MANUAL STEP: set the executor to drop privilege, in each akh's config:"
echo "  [exec]"
echo "  kind = \"local-drop\""
echo "  user = \"$EXEC_USER\""
echo "  workdir = \"$USER_HOME/ws\""
echo "Until that is set, [exec] defaults to in-process and commands run as ROOT."
