#!/bin/sh
# relay-group-3svc/validate.sh -- static lint, run from a dev machine (not
# on exec1). Shellchecks setup.sh/verify.sh if shellcheck is present;
# no-ops (with a note) otherwise so this never blocks a run that lacks the
# tool.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

if ! command -v shellcheck >/dev/null 2>&1; then
    echo "validate: shellcheck not found on PATH, skipping lint"
    exit 0
fi

status=0
for f in "$DIR/setup.sh" "$DIR/verify.sh"; do
    echo "validate: shellcheck $f"
    if ! shellcheck -s sh "$f"; then
        status=1
    fi
done

exit "$status"
