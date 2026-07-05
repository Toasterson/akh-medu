#!/bin/sh
# c-02-libscf-read-prop/verify.sh -- deterministic end-state check.
SRC=/root/readgreeting.c
BIN=/root/readgreeting
FMRI="svc:/site/akh-note:default"

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

got=$("$BIN" 2>/tmp/c02-stderr | tr -d '[:space:]')
want=$(svcprop -p config/greeting "$FMRI" 2>/dev/null | tr -d '[:space:]')
echo "checked: $BIN -> '$got', ground truth (svcprop config/greeting) -> '$want'"

if [ -z "$want" ] || [ "$got" != "$want" ]; then
    echo "FAIL: readgreeting printed '$got', expected '$want'"
    cat /tmp/c02-stderr 2>/dev/null
    exit 1
fi

echo "PASS: readgreeting matches live config/greeting value"
exit 0
