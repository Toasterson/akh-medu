#!/bin/sh
# discover-01-cpu-info/setup.sh -- runs as root on exec1 before the task.
set -e
mkdir -p /var/tmp
rm -f /var/tmp/akh-cpu-report.txt
echo "setup: nothing to prepare, report file cleared"
