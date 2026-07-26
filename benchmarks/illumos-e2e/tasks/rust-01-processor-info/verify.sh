#!/bin/sh
# rust-01-processor-info/verify.sh -- deterministic end-state check.
SRC=/root/cpuclock.rs
BIN=/root/cpuclock

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC does not exist"
    exit 1
fi

if grep -qE 'Command::new|std::process' "$SRC"; then
    echo "FAIL: $SRC shells out (std::process::Command) instead of calling processor_info(2) directly"
    exit 1
fi

if ! grep -q 'processor_info' "$SRC"; then
    echo "FAIL: $SRC does not appear to call processor_info"
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN is not present/executable -- did it compile?"
    exit 1
fi

got=$("$BIN" 2>/tmp/rust01-stderr | tr -d '[:space:]')
want=$(kstat -p cpu_info:0:cpu_info0:clock_MHz 2>/dev/null | awk '{print $2}' | tr -d '[:space:]')
echo "checked: $BIN -> '$got', ground truth (kstat clock_MHz) -> '$want'"

if [ -z "$got" ] || [ "$got" != "$want" ]; then
    echo "FAIL: cpuclock printed '$got', expected '$want'"
    cat /tmp/rust01-stderr 2>/dev/null
    exit 1
fi

echo "PASS: cpuclock matches live processor_info/kstat ground truth"
exit 0
