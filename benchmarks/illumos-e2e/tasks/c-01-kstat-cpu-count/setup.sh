#!/bin/sh
# c-01-kstat-cpu-count/setup.sh -- runs as root on exec1 before the task.
# Best-effort: ensure a C compiler is present. Package names on OpenIndiana
# Hipster drift with the gcc version currently published; try a few
# plausible FMRIs and continue regardless (the akh may need to finish this
# step itself if none of these match what's actually published).
set -e

if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    echo "setup: a C compiler is already present"
    exit 0
fi

for pkg in developer/gcc developer/gcc-13 developer/gcc-12 developer/build-essential; do
    pkg install -v "$pkg" 2>/dev/null && { echo "setup: installed $pkg"; exit 0; }
done

echo "setup: WARNING -- could not confirm a C compiler; the akh may need to install one (pkg search gcc)"
exit 0
