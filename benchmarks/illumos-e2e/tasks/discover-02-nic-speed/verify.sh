#!/bin/sh
# discover-02-nic-speed/verify.sh -- deterministic end-state check.
REPORT=/var/tmp/akh-nic-report.txt

if [ ! -f "$REPORT" ]; then
    echo "FAIL: $REPORT does not exist"
    exit 1
fi

iface=$(awk -F= '/^iface=/{print $2}' "$REPORT" | tr -d '[:space:]')
got_state=$(awk -F= '/^state=/{print $2}' "$REPORT" | tr -d '[:space:]')
got_speed=$(awk -F= '/^speed_mbps=/{print $2}' "$REPORT" | tr -d '[:space:]')
echo "checked: report iface='$iface' state='$got_state' speed_mbps='$got_speed'"

if [ -z "$iface" ]; then
    echo "FAIL: no iface= line in $REPORT"
    exit 1
fi

if ! dladm show-link -p -o link 2>/dev/null | grep -qx "$iface"; then
    echo "FAIL: '$iface' is not a link known to dladm show-link"
    exit 1
fi

want_state=$(dladm show-linkprop -c -o value -p state "$iface" 2>/dev/null | tr -d '[:space:]')
want_speed_raw=$(dladm show-linkprop -c -o value -p speed "$iface" 2>/dev/null | tr -d '[:space:]')
# speed property value is a bare Mbps integer already (e.g. "1000"); some
# drivers may report it with no suffix, others with a "G"/"M" suffix on
# `dladm show-link` (not show-linkprop) -- show-linkprop's speed value is
# documented as the numeric Mbps value.
want_speed=$(echo "$want_speed_raw" | tr -d 'Mm')
echo "ground truth: dladm state='$want_state' speed_mbps='$want_speed'"

if [ "$got_state" != "$want_state" ]; then
    echo "FAIL: state='$got_state', expected '$want_state'"
    exit 1
fi

if [ "$got_speed" != "$want_speed" ]; then
    echo "FAIL: speed_mbps='$got_speed', expected '$want_speed'"
    exit 1
fi

echo "PASS: nic report for '$iface' matches dladm ground truth"
exit 0
