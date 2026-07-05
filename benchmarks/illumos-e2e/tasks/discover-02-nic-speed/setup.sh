#!/bin/sh
# discover-02-nic-speed/setup.sh -- runs as root on exec1 before the task.
set -e
mkdir -p /var/tmp
rm -f /var/tmp/akh-nic-report.txt
echo "setup: nothing to prepare, report file cleared"
