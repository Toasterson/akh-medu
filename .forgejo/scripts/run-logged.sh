#!/bin/sh
# Preserve complete failing diagnostics without letting successful compiler
# output throttle execution on the remote per-line log transport.
set -eu
label=$1
shift
log=$(mktemp "${TMPDIR:-/tmp}/akh-ci.XXXXXX")
trap 'rm -f "$log"' EXIT HUP INT TERM
started=$(date +%s)
if "$@" >"$log" 2>&1; then
  elapsed=$(($(date +%s) - started))
  printf '%s passed in %ss\n' "$label" "$elapsed"
  # Test totals are evidence; individual successes and compiler progress are not.
  sed -n '/^test result:/p' "$log"
else
  status=$?
  cat "$log"
  printf '%s failed (exit %s)\n' "$label" "$status" >&2
  exit "$status"
fi
