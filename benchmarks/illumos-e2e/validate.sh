#!/bin/sh
# benchmarks/illumos-e2e/validate.sh
#
# Standalone, dependency-free static validation of the task corpus: no
# cargo/rust toolchain required (unlike
# benchmarks/harness-core/tests/illumos_e2e_task_set.rs, which does deeper
# structural validation of meta.toml and calls hash_task_set_dir). Run this
# from anywhere; it locates the tasks/ directory relative to this script.
#
# Checks:
#   1. Every task directory has all four required files.
#   2. Every setup.sh/verify.sh passes `sh -n` (and `dash -n` if available).
#   3. Every meta.toml has the required top-level keys present (a grep-based
#      sanity check -- the harness-core test does real TOML parsing).
#   4. At least 15 tasks exist, and at least 3 are marked as version traps.
#
# Exit 0 = all checks passed. Non-zero = at least one check failed (details
# printed to stdout as they're found).

set -u
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASKS_DIR="$SCRIPT_DIR/tasks"
fail=0

if [ ! -d "$TASKS_DIR" ]; then
    echo "FAIL: $TASKS_DIR does not exist"
    exit 1
fi

task_count=0
trap_count=0

for dir in "$TASKS_DIR"/*/; do
    [ -d "$dir" ] || continue
    task=$(basename "$dir")
    task_count=$((task_count + 1))

    for req in meta.toml setup.sh prompt.md verify.sh; do
        if [ ! -f "$dir$req" ]; then
            echo "FAIL: $task is missing $req"
            fail=1
        fi
    done

    for script in setup.sh verify.sh; do
        f="$dir$script"
        [ -f "$f" ] || continue
        if ! sh -n "$f" >/tmp/akh-e2e-validate-err.$$ 2>&1; then
            echo "FAIL: $task/$script does not pass 'sh -n':"
            sed 's/^/    /' /tmp/akh-e2e-validate-err.$$
            fail=1
        fi
        if command -v dash >/dev/null 2>&1; then
            if ! dash -n "$f" >/tmp/akh-e2e-validate-err.$$ 2>&1; then
                echo "FAIL: $task/$script does not pass 'dash -n' (strict POSIX -- check for bashisms):"
                sed 's/^/    /' /tmp/akh-e2e-validate-err.$$
                fail=1
            fi
        fi
        rm -f /tmp/akh-e2e-validate-err.$$
    done

    meta="$dir/meta.toml"
    if [ -f "$meta" ]; then
        for key in 'id' 'family' 'description' 'timeout_secs' 'expected_manpages' 'mode' '\[version_trap\]'; do
            if ! grep -qE "^${key}[[:space:]]*=" "$meta" && ! grep -qE "^${key}" "$meta"; then
                echo "FAIL: $task/meta.toml is missing required key: $key"
                fail=1
            fi
        done
        if grep -q '^present = true' "$meta"; then
            trap_count=$((trap_count + 1))
            if ! grep -q '^note' "$meta"; then
                echo "FAIL: $task/meta.toml has version_trap.present = true but no note"
                fail=1
            fi
        fi
    fi
done

echo "checked: $task_count task directories, $trap_count marked as version traps"

if [ "$task_count" -lt 15 ]; then
    echo "FAIL: expected at least 15 tasks, found $task_count"
    fail=1
fi

if [ "$trap_count" -lt 3 ]; then
    echo "FAIL: expected at least 3 version-trap tasks, found $trap_count"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "PASS: all static checks passed"
fi

exit "$fail"
