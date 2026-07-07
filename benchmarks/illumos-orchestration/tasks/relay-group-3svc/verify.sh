#!/bin/sh
# relay-group-3svc/verify.sh -- deterministic end-state check, root, after
# the task. Exit 0 = pass (all three services fixed), non-zero = fail. The
# akh never sees this file. Echoes each check so a failed run is
# diagnosable from stdout alone.
FMRI_A="svc:/site/relay-a:default"
FMRI_B="svc:/site/relay-b:default"
FMRI_C="svc:/site/relay-c:default"
EXPECTED_GREETING="relay-c-ready"

fail=0

state_a=$(svcs -H -o state "$FMRI_A" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI_A -> $state_a"
if [ "$state_a" != "online" ]; then
    echo "FAIL: $FMRI_A is '$state_a', expected 'online'"
    fail=1
fi

state_b=$(svcs -H -o state "$FMRI_B" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI_B -> $state_b"
if [ "$state_b" != "online" ]; then
    echo "FAIL: $FMRI_B is '$state_b', expected 'online'"
    fail=1
fi

state_c=$(svcs -H -o state "$FMRI_C" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI_C -> $state_c"
if [ "$state_c" != "online" ]; then
    echo "FAIL: $FMRI_C is '$state_c', expected 'online'"
    fail=1
fi

greeting=$(svcprop -p app/greeting "$FMRI_C" 2>/dev/null || echo "MISSING")
echo "checked: svcprop -p app/greeting $FMRI_C -> $greeting"
if [ "$greeting" != "$EXPECTED_GREETING" ]; then
    echo "FAIL: app/greeting on $FMRI_C is '$greeting', expected '$EXPECTED_GREETING'"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "FAIL: relay-group-3svc not fully fixed"
    exit 1
fi

echo "PASS: relay-a, relay-b, relay-c all online, relay-c app/greeting=$EXPECTED_GREETING"
exit 0
