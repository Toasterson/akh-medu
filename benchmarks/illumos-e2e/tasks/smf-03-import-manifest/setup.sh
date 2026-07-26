#!/bin/sh
# smf-03-import-manifest/setup.sh -- runs as root on exec1 before the task.
# Idempotent: installs a working daemon script but NO manifest -- the akh
# must author the manifest XML itself and bring the service under SMF.
set -e

mkdir -p /opt/akh-e2e

cat > /opt/akh-e2e/echo-daemon.sh <<'INNER'
#!/bin/sh
# Trivial long-running daemon: appends a timestamp to a log file every
# second, forever. Used to prove a supervised process is actually alive
# (not just that some SMF instance claims "online").
LOG=/var/tmp/akh-e2e-echo.log
: > "$LOG"
while :; do
    date +%s >> "$LOG"
    sleep 1
done
INNER
chmod 755 /opt/akh-e2e/echo-daemon.sh

# Make sure any stale log from a previous run of this task doesn't make the
# "recent activity" check in verify.sh pass without the akh doing anything.
rm -f /var/tmp/akh-e2e-echo.log

echo "setup: /opt/akh-e2e/echo-daemon.sh installed, no manifest present"
