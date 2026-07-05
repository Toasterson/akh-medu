#!/bin/sh
# rust-02-door-roundtrip/verify.sh -- deterministic end-state check.
SRC=/root/doorping.rs
BIN=/root/doorping

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC does not exist"
    exit 1
fi

if grep -qE 'Command::new|std::process' "$SRC"; then
    echo "FAIL: $SRC shells out (std::process::Command) instead of using the Doors API directly"
    exit 1
fi

for sym in door_create door_call door_return; do
    if ! grep -q "$sym" "$SRC"; then
        echo "FAIL: $SRC does not appear to call $sym"
        exit 1
    fi
done

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN is not present/executable -- did it compile?"
    exit 1
fi

# No `timeout(1)` dependency (GNU coreutils, not guaranteed present under
# /usr/bin on illumos) -- run in the background with a manual watchdog so a
# deadlocked door call can't hang the whole grading run.
"$BIN" >/tmp/rust02-stdout 2>/tmp/rust02-stderr &
pid=$!
n=0
while kill -0 "$pid" 2>/dev/null && [ "$n" -lt 10 ]; do
    sleep 1
    n=$((n + 1))
done
if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null
    echo "FAIL: doorping did not terminate within 10s (possible deadlock in the door call)"
    exit 1
fi
wait "$pid" 2>/dev/null

got=$(cat /tmp/rust02-stdout)
echo "checked: $BIN -> '$got'"

if [ "$got" != "ping-pong" ]; then
    echo "FAIL: doorping printed '$got', expected 'ping-pong'"
    cat /tmp/rust02-stderr 2>/dev/null
    exit 1
fi

echo "PASS: door round trip produced the expected transformed payload"
exit 0
