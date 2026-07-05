#!/bin/sh
# rbac-02-custom-profile/setup.sh -- runs as root on exec1 before the task.
# Idempotent: creates a plain user with no profiles, and a log file only
# root can read, so the profile grant is functionally verifiable.
set -e

if ! id auditor >/dev/null 2>&1; then
    useradd -m -s /bin/sh auditor
fi
usermod -P "" auditor 2>/dev/null || true

# A log only root can read -- auditor must go through the granted profile
# to see it.
echo "root-only log line $(date)" > /var/adm/akh-e2e-secure.log
chmod 600 /var/adm/akh-e2e-secure.log
chown root:root /var/adm/akh-e2e-secure.log

# Remove any leftover profile from a previous run of this task.
grep -v '^Log Auditor:' /etc/security/prof_attr > /etc/security/prof_attr.tmp 2>/dev/null || true
[ -f /etc/security/prof_attr.tmp ] && mv /etc/security/prof_attr.tmp /etc/security/prof_attr
grep -v '^Log Auditor:' /etc/security/exec_attr > /etc/security/exec_attr.tmp 2>/dev/null || true
[ -f /etc/security/exec_attr.tmp ] && mv /etc/security/exec_attr.tmp /etc/security/exec_attr

echo "setup: auditor exists with no profiles; /var/adm/akh-e2e-secure.log is root-only (0600)"
