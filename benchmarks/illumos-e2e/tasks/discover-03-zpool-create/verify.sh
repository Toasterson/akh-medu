#!/bin/sh
# discover-03-zpool-create/verify.sh -- deterministic end-state check.
POOL=akhtank
DS=akhtank/data

health=$(zpool list -H -o health "$POOL" 2>/dev/null || echo "MISSING")
echo "checked: zpool list -H -o health $POOL -> $health"
if [ "$health" != "ONLINE" ]; then
    echo "FAIL: pool '$POOL' health='$health', expected 'ONLINE'"
    exit 1
fi

comp=$(zfs get -H -o value compression "$DS" 2>/dev/null || echo "MISSING")
quota=$(zfs get -H -o value quota "$DS" 2>/dev/null || echo "MISSING")
mnt=$(zfs get -H -o value mountpoint "$DS" 2>/dev/null || echo "MISSING")
echo "checked: $DS compression='$comp' quota='$quota' mountpoint='$mnt'"

fail=0
if [ "$comp" != "lz4" ]; then
    echo "FAIL: compression='$comp', expected 'lz4'"
    fail=1
fi
if [ "$quota" != "50M" ]; then
    echo "FAIL: quota='$quota', expected '50M'"
    fail=1
fi
if [ "$mnt" != "/akh-data" ]; then
    echo "FAIL: mountpoint='$mnt', expected '/akh-data'"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "PASS: $POOL/$DS matches required properties"
exit 0
