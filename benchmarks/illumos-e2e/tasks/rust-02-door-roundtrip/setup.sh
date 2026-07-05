#!/bin/sh
# rust-02-door-roundtrip/setup.sh -- runs as root on exec1 before the task.
set -e
if ! command -v rustc >/dev/null 2>&1; then
    echo "setup: WARNING -- rustc not found on PATH; the akh may need to install/locate a Rust toolchain first"
fi
echo "setup: nothing else to prepare"
