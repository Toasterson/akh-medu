#!/bin/sh
# smf-03-import-manifest/verify.sh -- deterministic end-state check.
FMRI="svc:/site/akh-echo:default"
LOG=/var/tmp/akh-e2e-echo.log

state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI -> $state"
if [ "$state" != "online" ]; then
    echo "FAIL: $FMRI is '$state', expected 'online'"
    exit 1
fi

if [ ! -f "$LOG" ]; then
    echo "FAIL: $LOG does not exist -- daemon does not appear to be running"
    exit 1
fi

now=$(date +%s)
mtime=$(perl -e 'print((stat($ARGV[0]))[9])' "$LOG" 2>/dev/null || echo 0)
if [ "$mtime" -eq 0 ]; then
    # perl may be unavailable; fall back to a coarse check that the file
    # has grown by re-checking line count twice a second apart.
    n1=$(wc -l < "$LOG")
    sleep 2
    n2=$(wc -l < "$LOG")
    echo "checked: $LOG line count $n1 -> $n2 over 2s (perl mtime unavailable)"
    if [ "$n2" -le "$n1" ]; then
        echo "FAIL: $LOG is not growing -- daemon does not appear to be alive"
        exit 1
    fi
    echo "PASS: $FMRI online and $LOG actively growing"
    exit 0
fi

age=$((now - mtime))
echo "checked: $LOG last modified ${age}s ago"
if [ "$age" -gt 10 ]; then
    echo "FAIL: $LOG has not been written to in ${age}s -- daemon does not appear to be alive"
    exit 1
fi

echo "PASS: $FMRI online and $LOG actively growing"
exit 0
