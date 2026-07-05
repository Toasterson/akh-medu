#!/bin/sh
# rbac-01-create-role/setup.sh -- runs as root on exec1 before the task.
# Idempotent: creates a plain, unprivileged user with no roles/profiles.
set -e

if ! id opuser >/dev/null 2>&1; then
    useradd -m -s /bin/sh opuser
fi

# Make sure any role/profile a previous run of this task created is gone,
# so the task always starts from "opuser has no roles, netops does not
# exist".
usermod -R "" opuser 2>/dev/null || true
userdel -r netops 2>/dev/null || true

echo "setup: opuser exists with no roles; netops role absent"
