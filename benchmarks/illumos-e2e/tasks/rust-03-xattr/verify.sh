#!/bin/sh
# rust-03-xattr/verify.sh -- deterministic end-state check.
SRC=/root/xattrdemo.rs
BIN=/root/xattrdemo
TARGET=/var/tmp/akh-e2e-target.txt

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC does not exist"
    exit 1
fi

if grep -qiE 'setxattr|getxattr|xattr::' "$SRC"; then
    echo "FAIL: $SRC uses Linux setxattr/getxattr or the xattr crate, neither of which exists on illumos"
    exit 1
fi

if ! grep -qE 'O_XATTR|attropen' "$SRC"; then
    echo "FAIL: $SRC does not appear to use O_XATTR/attropen"
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN is not present/executable -- did it compile?"
    exit 1
fi

got=$("$BIN" 2>/tmp/rust03-stderr | tr -d '\n')
echo "checked: $BIN -> '$got'"
if [ "$got" != "hello-xattr" ]; then
    echo "FAIL: xattrdemo printed '$got', expected 'hello-xattr'"
    cat /tmp/rust03-stderr 2>/dev/null
    exit 1
fi

# Independent cross-check: read the attribute back with the shell-level
# runat(1) tool, without going through the akh's own program at all.
if command -v runat >/dev/null 2>&1; then
    cross=$(runat "$TARGET" cat note 2>/dev/null | tr -d '\n')
    echo "checked (independent, via runat(1)): note -> '$cross'"
    if [ "$cross" != "hello-xattr" ]; then
        echo "FAIL: runat cross-check read '$cross', expected 'hello-xattr' -- attribute may not actually be persisted"
        exit 1
    fi
else
    echo "WARN: runat(1) not found -- skipping independent cross-check"
fi

echo "PASS: xattr round trip matches expected content"
exit 0
