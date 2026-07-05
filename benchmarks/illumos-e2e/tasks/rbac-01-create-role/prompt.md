# Delegate network administration without giving out root

The user `opuser` on this host needs to be able to perform network
configuration tasks (the kind of thing covered by the built-in
"Network Management" rights profile), but should not otherwise have
broad root access, and there is no `sudo`/`sudoers` on this system to
reach for.

Set this up properly using this host's native privilege model: create a
role named exactly `netops` that carries the "Network Management" rights
profile, and grant the `opuser` user the ability to assume that role.
