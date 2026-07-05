#!/bin/sh
# rbac-01-create-role/verify.sh -- deterministic end-state check.

entry=$(grep '^netops:' /etc/user_attr 2>/dev/null || echo "")
echo "checked: /etc/user_attr netops entry -> '$entry'"
if [ -z "$entry" ]; then
    echo "FAIL: no 'netops' entry in /etc/user_attr -- role was not created"
    exit 1
fi

if ! echo "$entry" | grep -q 'type=role'; then
    echo "FAIL: 'netops' entry exists but is not type=role"
    exit 1
fi

profs=$(profiles netops 2>/dev/null || echo "")
echo "checked: profiles netops -> $profs"
if ! echo "$profs" | grep -qi 'Network Management'; then
    echo "FAIL: role 'netops' is not carrying the 'Network Management' profile"
    exit 1
fi

user_roles=$(roles opuser 2>/dev/null || echo "")
echo "checked: roles opuser -> $user_roles"
if ! echo "$user_roles" | grep -qw 'netops'; then
    echo "FAIL: user 'opuser' is not assigned the 'netops' role"
    exit 1
fi

echo "PASS: netops role exists with Network Management profile and is assigned to opuser"
exit 0
