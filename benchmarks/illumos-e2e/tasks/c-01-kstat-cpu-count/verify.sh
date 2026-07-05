#!/bin/sh
# c-01-kstat-cpu-count/verify.sh -- deterministic end-state check.
SRC=/root/cpucount.c
BIN=/root/cpucount

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC does not exist"
    exit 1
fi

if grep -qE 'system\(|popen\(|exec[lv]p?\(' "$SRC"; then
    echo "FAIL: $SRC shells out (system/popen/exec*) instead of using libkstat directly"
    exit 1
fi

if ! grep -q 'kstat_open' "$SRC" || ! grep -q 'kstat_lookup' "$SRC"; then
    echo "FAIL: $SRC does not appear to call kstat_open/kstat_lookup"
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN is not present/executable -- did it compile?"
    exit 1
fi

if command -v ldd >/dev/null 2>&1; then
    if ! ldd "$BIN" 2>/dev/null | grep -qi 'libkstat'; then
        echo "FAIL: $BIN is not linked against libkstat"
        exit 1
    fi
fi

got=$("$BIN" 2>/tmp/c01-stderr | tr -d '[:space:]')
want=$(kstat -p -m cpu_info 2>/dev/null | awk -F: '{print $1":"$2}' | sort -u | wc -l | tr -d '[:space:]')
echo "checked: $BIN -> '$got', ground truth (kstat -p -m cpu_info instances) -> '$want'"

if [ -z "$got" ] || [ "$got" != "$want" ]; then
    echo "FAIL: cpucount printed '$got', expected '$want'"
    cat /tmp/c01-stderr 2>/dev/null
    exit 1
fi

echo "PASS: cpucount matches live cpu_info kstat instance count"
exit 0
