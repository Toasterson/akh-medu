#!/usr/bin/env bash
# Resume a run from its newest checkpoint — the on-and-off training workhorse.
# Cloud boxes get preempted, rentals lapse, you stop for the night: this picks
# up where the last checkpoint left off.
#
#   bash training/scripts/resume.sh [run-name]    # default: p350m
#
# (run-350m.sh already auto-resumes if checkpoints exist; this is the explicit
# form, and the one to use after a reboot or preemption.)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN="${1:-p350m}"
DIR="runs/${RUN}"

[ -d "$DIR/checkpoints" ] || { echo "FAIL: no checkpoints in $DIR — start with run-350m.sh" >&2; exit 1; }
LAST="$(ls -1 "$DIR/checkpoints" | sort | tail -1)"
[ -n "$LAST" ] || { echo "FAIL: checkpoints dir is empty" >&2; exit 1; }

if pgrep -f "examples/train_rule" >/dev/null; then
  echo "FAIL: a train_rule process is already running — GPU jobs must be sequential." >&2
  exit 1
fi

echo "Resuming ${RUN} from ${LAST}"
exec bash training/scripts/run-350m.sh
