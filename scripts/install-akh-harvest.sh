#!/bin/sh
# Install the transcript harvester on this machine (akh-medu ADR 119).
#
#   sh scripts/install-akh-harvest.sh --machine <name> --bundle <certs.tgz> [--binary <path>|--build] [--interval 15m]
#
# What it does: unpacks the machine's mTLS client certs, writes the env file,
# installs the harvester binary to ~/.local/bin, installs a timer for the
# platform (systemd --user on Linux, launchd agent on macOS, SMF on illumos),
# enables it, and runs one harvest immediately so you can see the summary line.
# Prerequisites done by the operator (docs/ops/transcript-harvest.md, "Adding a
# machine"): PG role harvest_<machine> with INSERT, and a v3 client cert with
# CN=harvest_<machine>, packed as <bundle> containing ca.crt, tls.crt, tls.key.
set -eu
MACHINE=""; BUNDLE=""; BINARY=""; BUILD=0; INTERVAL="15m"
while [ $# -gt 0 ]; do case "$1" in
  --machine) MACHINE=$2; shift 2;; --bundle) BUNDLE=$2; shift 2;; --binary) BINARY=$2; shift 2;;
  --build) BUILD=1; shift;; --interval) INTERVAL=$2; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[ -n "$MACHINE" ] && [ -n "$BUNDLE" ] || { echo "usage: $0 --machine <name> --bundle <certs.tgz> [--binary <path>|--build]" >&2; exit 2; }
OS=$(uname -s); CFG="$HOME/.config/akh-transcripts"; CERTS="$CFG/certs"; BIN="$HOME/.local/bin/akh-harvest"
umask 077; mkdir -p "$CERTS" "$HOME/.local/bin" "$HOME/.local/state/akh-transcripts"
tar xzf "$BUNDLE" -C "$CERTS"; chmod 600 "$CERTS"/*.key
for f in ca.crt tls.crt tls.key; do [ -s "$CERTS/$f" ] || { echo "bundle is missing $f" >&2; exit 1; }; done
CN=$(openssl x509 -in "$CERTS/tls.crt" -noout -subject | sed 's/.*CN *= *//')
[ "$CN" = "harvest_$MACHINE" ] || { echo "cert CN is '$CN' but the role must be 'harvest_$MACHINE' — pg_hba clientcert=verify-full maps CN to role" >&2; exit 1; }
if [ "$BUILD" = 1 ]; then command -v cargo >/dev/null || { echo "--build needs cargo" >&2; exit 1; }; cargo build --release -p akh-transcripts --bin akh-harvest; BINARY=target/release/akh-harvest; fi
[ -n "$BINARY" ] && [ -x "$BINARY" ] || { echo "need --binary <path to akh-harvest> or --build" >&2; exit 1; }
install -m 0755 "$BINARY" "$BIN"
# harvest-pg's server cert SANs cover harvest-pg.lab.home, not the LAN IP; rustls verifies the name.
getent hosts harvest-pg.lab.home >/dev/null 2>&1 || dscacheutil -q host -a name harvest-pg.lab.home 2>/dev/null | grep -q ip_address || \
  echo "NOTE: harvest-pg.lab.home does not resolve here — add '192.168.1.60 harvest-pg.lab.home' to /etc/hosts (needs sudo) before the first run."
cat > "$CFG/harvest.env" <<ENV
# akh-harvest on $MACHINE — insert-only role over mTLS to harvest-pg (ADR 108).
AKH_TRANSCRIPTS_DATABASE_URL='postgres://harvest_$MACHINE@harvest-pg.lab.home:5432/transcripts?sslmode=verify-full&sslrootcert=$CERTS/ca.crt&sslcert=$CERTS/tls.crt&sslkey=$CERTS/tls.key'
AKH_TRANSCRIPTS_MACHINE=$MACHINE
AKH_TRANSCRIPTS_STATE_DIR=$HOME/.local/state/akh-transcripts
RUST_LOG=akh_transcripts=info
ENV
case "$OS" in
  Linux)
    U="$HOME/.config/systemd/user"; mkdir -p "$U"
    printf '[Unit]\nDescription=akh-harvest — push agent-session transcripts to harvest-pg (ADR 108)\n\n[Service]\nType=oneshot\nEnvironmentFile=%%h/.config/akh-transcripts/harvest.env\nExecStart=%%h/.local/bin/akh-harvest --once\nNice=10\n' > "$U/akh-harvest.service"
    printf '[Unit]\nDescription=Run akh-harvest every %s\n\n[Timer]\nOnBootSec=3min\nOnUnitActiveSec=%s\nRandomizedDelaySec=60\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n' "$INTERVAL" "$INTERVAL" > "$U/akh-harvest.timer"
    systemctl --user daemon-reload; systemctl --user enable --now akh-harvest.timer; systemctl --user start akh-harvest.service
    journalctl --user -u akh-harvest.service -n 5 --no-pager -o cat | grep "harvest complete" || true ;;
  Darwin)
    SECS=$(echo "$INTERVAL" | awk '/m$/{sub("m","");print $1*60} /h$/{sub("h","");print $1*3600} /^[0-9]+$/{print $1}')
    P="$HOME/Library/LaunchAgents/dev.akh-medu.akh-harvest.plist"; mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
    ENVXML=$(sed -n 's/^\([A-Z_]*\)=\(.*\)$/<key>\1<\/key><string>\2<\/string>/p' "$CFG/harvest.env" | sed "s/^'//; s/'$//; s/<string>'/<string>/; s/'<\/string>/<\/string>/")
    cat > "$P" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>dev.akh-medu.akh-harvest</string>
  <key>ProgramArguments</key><array><string>$BIN</string><string>--once</string></array>
  <key>EnvironmentVariables</key><dict>$ENVXML</dict>
  <key>StartInterval</key><integer>${SECS:-900}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/akh-harvest.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/akh-harvest.log</string>
</dict></plist>
PL
    launchctl bootout "gui/$(id -u)" "$P" 2>/dev/null || true; launchctl bootstrap "gui/$(id -u)" "$P"; launchctl kickstart -k "gui/$(id -u)/dev.akh-medu.akh-harvest"
    sleep 5; grep "harvest complete" "$HOME/Library/Logs/akh-harvest.log" | tail -1 || true ;;
  SunOS)
    SECS=$(echo "$INTERVAL" | awk '/m$/{sub("m","");print $1*60} /h$/{sub("h","");print $1*3600} /^[0-9]+$/{print $1}')
    M="$CFG/akh-harvest.xml"
    cat > "$M" <<SMF
<?xml version="1.0"?>
<!DOCTYPE service_bundle SYSTEM "/usr/share/lib/xml/dtd/service_bundle.dtd.1">
<service_bundle type="manifest" name="akh-harvest"><service name="application/akh-harvest" type="service" version="1">
  <create_default_instance enabled="true"/><single_instance/>
  <dependency name="network" grouping="require_all" restart_on="none" type="service"><service_fmri value="svc:/milestone/network:default"/></dependency>
  <method_context><method_credential user="$(id -un)"/><method_environment>
$(sed -n "s/^\([A-Z_]*\)='\{0,1\}\(.*[^']\)'\{0,1\}$/    <envvar name='\1' value='\2'\/>/p" "$CFG/harvest.env")
  </method_environment></method_context>
  <exec_method type="method" name="start" exec="/bin/sh -c 'while :; do $BIN --once; sleep ${SECS:-900}; done &amp;'" timeout_seconds="60"/>
  <exec_method type="method" name="stop" exec=":kill" timeout_seconds="60"/>
  <property_group name="startd" type="framework"><propval name="duration" type="astring" value="contract"/></property_group>
</service></service_bundle>
SMF
    echo "SMF manifest written to $M — import with: pfexec svccfg import $M && svcs akh-harvest" ;;
  *) echo "unsupported OS $OS: env and binary installed, schedule it yourself with: $BIN --once" ;;
esac
echo "installed akh-harvest for machine '$MACHINE' ($OS); env: $CFG/harvest.env"
