#!/bin/sh
# smf-02-enable-configure/verify.sh -- deterministic end-state check.
FMRI="svc:/site/akh-greeter:default"
EXPECTED="buenas-akh"

state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI -> $state"
if [ "$state" != "online" ]; then
    echo "FAIL: $FMRI is '$state', expected 'online'"
    exit 1
fi

greeting=$(svcprop -p app/greeting "$FMRI" 2>/dev/null || echo "MISSING")
echo "checked: svcprop -p app/greeting $FMRI -> $greeting"
if [ "$greeting" != "$EXPECTED" ]; then
    echo "FAIL: app/greeting is '$greeting', expected '$EXPECTED'"
    exit 1
fi

echo "PASS: $FMRI online with app/greeting=$EXPECTED"
exit 0
