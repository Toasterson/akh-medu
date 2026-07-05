#!/bin/sh
# discover-01-cpu-info/verify.sh -- deterministic end-state check.
REPORT=/var/tmp/akh-cpu-report.txt

if [ ! -f "$REPORT" ]; then
    echo "FAIL: $REPORT does not exist"
    exit 1
fi

got_cpus=$(awk -F= '/^cpus=/{print $2}' "$REPORT" | tr -d '[:space:]')
got_clock=$(awk -F= '/^clock_mhz=/{print $2}' "$REPORT" | tr -d '[:space:]')
echo "checked: report contents cpus='$got_cpus' clock_mhz='$got_clock'"

want_cpus=$(psrinfo | wc -l | tr -d '[:space:]')
want_clock=$(kstat -p cpu_info:0:cpu_info0:clock_MHz 2>/dev/null | awk '{print $2}' | tr -d '[:space:]')
echo "ground truth: psrinfo count='$want_cpus' kstat clock_MHz='$want_clock'"

if [ -z "$got_cpus" ] || [ "$got_cpus" != "$want_cpus" ]; then
    echo "FAIL: cpus='$got_cpus', expected '$want_cpus'"
    exit 1
fi

if [ -z "$got_clock" ] || [ "$got_clock" != "$want_clock" ]; then
    echo "FAIL: clock_mhz='$got_clock', expected '$want_clock'"
    exit 1
fi

echo "PASS: cpu report matches psrinfo/kstat ground truth"
exit 0
