#!/bin/sh
# c-03-libscf-smf-state/verify.sh -- deterministic end-state check.
SRC=/root/svcstate.c
BIN=/root/svcstate
FMRI="svc:/site/akh-guard:default"

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC does not exist"
    exit 1
fi

if grep -qE 'system\(|popen\(|exec[lv]p?\(' "$SRC"; then
    echo "FAIL: $SRC shells out (system/popen/exec*) instead of using libscf directly"
    exit 1
fi

if ! grep -q 'scf_simple_prop_get' "$SRC"; then
    echo "FAIL: $SRC does not appear to call scf_simple_prop_get"
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN is not present/executable -- did it compile?"
    exit 1
fi

if command -v ldd >/dev/null 2>&1; then
    if ! ldd "$BIN" 2>/dev/null | grep -qi 'libscf'; then
        echo "FAIL: $BIN is not linked against libscf"
        exit 1
    fi
fi

got=$("$BIN" "$FMRI" 2>/tmp/c03-stderr | tr -d '[:space:]')
want=$(svcs -H -o state "$FMRI" 2>/dev/null | tr -d '[:space:]')
echo "checked: $BIN $FMRI -> '$got', ground truth (svcs -H -o state) -> '$want'"

if [ -z "$want" ] || [ "$got" != "$want" ]; then
    echo "FAIL: svcstate printed '$got', expected '$want'"
    cat /tmp/c03-stderr 2>/dev/null
    exit 1
fi

if [ "$want" != "maintenance" ]; then
    echo "WARN: ground truth state is '$want', not the expected 'maintenance' -- setup.sh may not have converged; treating output match as sufficient"
fi

echo "PASS: svcstate matches live restarter state"
exit 0
