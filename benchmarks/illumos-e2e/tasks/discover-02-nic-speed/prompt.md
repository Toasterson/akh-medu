# Report the primary NIC's link state and speed

This host has no `ethtool` and no `ip` command -- find the native illumos
way to do this instead.

Identify this host's primary network interface (the one carrying its
default route / primary IP), then report its link state and speed.

Write exactly three lines to `/var/tmp/akh-nic-report.txt` in this form:

    iface=<name>
    state=<up|down>
    speed_mbps=<n>

where `<name>` is the illumos link name (e.g. `net0`, `e1000g0`), `<up|down>`
is the link state, and `<n>` is the link speed in Mbps as a bare integer.
