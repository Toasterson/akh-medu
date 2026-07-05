#!/bin/sh
# rust-03-xattr/setup.sh -- runs as root on exec1 before the task.
set -e
mkdir -p /var/tmp
echo "base file content, unrelated to the attribute" > /var/tmp/akh-e2e-target.txt
if ! command -v rustc >/dev/null 2>&1; then
    echo "setup: WARNING -- rustc not found on PATH; the akh may need to install/locate a Rust toolchain first"
fi
echo "setup: /var/tmp/akh-e2e-target.txt ready, no extended attribute on it yet"
