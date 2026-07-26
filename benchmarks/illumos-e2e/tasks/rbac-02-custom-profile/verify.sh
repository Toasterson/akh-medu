#!/bin/sh
# rbac-02-custom-profile/verify.sh -- deterministic end-state check.
LOG=/var/adm/akh-e2e-secure.log

# Positive: with the granted profile via pfexec, auditor can read the log.
su - auditor -c "pfexec /usr/bin/tail $LOG" >/tmp/rbac02-out 2>/tmp/rbac02-err
rc=$?
echo "checked: su - auditor -c 'pfexec tail $LOG' exit=$rc"
if [ "$rc" -ne 0 ] || [ ! -s /tmp/rbac02-out ]; then
    echo "FAIL: auditor could not read $LOG via pfexec (rc=$rc)"
    cat /tmp/rbac02-err 2>/dev/null
    exit 1
fi

# Negative: without pfexec, auditor's own uid still cannot read the log --
# the grant must be scoped to the profile, not a blanket permission change.
su - auditor -c "/usr/bin/tail $LOG" >/tmp/rbac02-out2 2>/tmp/rbac02-err2
rc2=$?
echo "checked: su - auditor -c 'tail $LOG' (no pfexec) exit=$rc2"
if [ "$rc2" -eq 0 ]; then
    echo "FAIL: auditor could read $LOG WITHOUT pfexec -- privilege is not scoped to the profile (file may have been made world-readable instead)"
    exit 1
fi

echo "PASS: auditor can read $LOG only via the granted profile, not directly"
exit 0
