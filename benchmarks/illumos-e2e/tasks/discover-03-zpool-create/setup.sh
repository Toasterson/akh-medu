#!/bin/sh
# discover-03-zpool-create/setup.sh -- runs as root on exec1 before the task.
# Idempotent: destroys any leftover akhtank pool from a previous run of
# this task, then creates a blank 200MB file to use as a vdev (no spare
# physical/virtual disk is assumed to be attached to the VM).
set -e

zpool destroy akhtank 2>/dev/null || true
rm -f /var/tmp/akh-e2e-vdev.img
mkfile 200m /var/tmp/akh-e2e-vdev.img
rm -rf /akh-data 2>/dev/null || true

echo "setup: /var/tmp/akh-e2e-vdev.img (200MB) ready, akhtank pool absent"
