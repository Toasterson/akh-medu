#!/bin/sh
# rbac-03-restrict-privileges/verify.sh -- deterministic end-state check.

# Sanity: the account must still be usable at all.
sane=$(su - batchrunner -c 'echo alive' 2>/tmp/rbac03-sane-err)
echo "checked: su - batchrunner -c 'echo alive' -> '$sane'"
if [ "$sane" != "alive" ]; then
    echo "FAIL: batchrunner account appears broken (cannot even echo)"
    cat /tmp/rbac03-sane-err 2>/dev/null
    exit 1
fi

# Static: the effective privilege set for a fresh batchrunner shell must
# not list proc_info.
eset=$(su - batchrunner -c 'ppriv -v $$' 2>/tmp/rbac03-ppriv-err | awk '/^E:/{print; exit}')
echo "checked: ppriv -v \$\$ (E line) -> '$eset'"
if echo "$eset" | grep -qw 'proc_info'; then
    echo "FAIL: batchrunner's effective privilege set still contains proc_info: $eset"
    exit 1
fi
if [ -z "$eset" ]; then
    echo "FAIL: could not read batchrunner's effective privilege set at all"
    cat /tmp/rbac03-ppriv-err 2>/dev/null
    exit 1
fi

# Functional: without proc_info, batchrunner must be denied access to a
# /proc entry it does not own (pid 1, owned by root).
su - batchrunner -c 'dd if=/proc/1/psinfo of=/dev/null bs=1 count=1' >/tmp/rbac03-out 2>/tmp/rbac03-err
rc=$?
echo "checked: su - batchrunner -c 'dd if=/proc/1/psinfo ...' exit=$rc"
if [ "$rc" -eq 0 ]; then
    echo "FAIL: batchrunner could still read /proc/1/psinfo -- proc_info does not appear to have been removed"
    exit 1
fi

echo "PASS: batchrunner's default privilege set excludes proc_info and the boundary holds functionally"
exit 0
