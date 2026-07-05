#!/bin/sh
# rbac-03-restrict-privileges/setup.sh -- runs as root on exec1 before the
# task. Idempotent: creates a service-account-style user with the illumos
# default (full "basic") privilege set, which includes proc_info -- the
# privilege that lets a process inspect /proc entries for processes it
# does not own, regardless of that process's actual owner.
set -e

if ! id batchrunner >/dev/null 2>&1; then
    useradd -m -s /bin/sh batchrunner
fi
usermod -K defaultpriv=basic batchrunner 2>/dev/null || true

echo "setup: batchrunner exists with the default 'basic' privilege set (includes proc_info)"
