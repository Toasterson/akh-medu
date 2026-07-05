#!/bin/sh
# smf-01-clear-maintenance/verify.sh -- deterministic end-state check.
# Exit 0 = pass, non-zero = fail. The akh never sees this file.
FMRI="svc:/site/akh-relay:default"

state=$(svcs -H -o state "$FMRI" 2>/dev/null || echo missing)
echo "checked: svcs -H -o state $FMRI -> $state"

if [ "$state" != "online" ]; then
    echo "FAIL: $FMRI is '$state', expected 'online'"
    exit 1
fi

echo "PASS: $FMRI is online"
exit 0
