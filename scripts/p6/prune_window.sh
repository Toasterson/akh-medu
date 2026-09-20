#!/usr/bin/env bash
# Phase 51 P6 — reclaim disk from a FINISHED window: keep only the checkpoint that
# candidate-model.json references (model + Adam + meta), delete the intermediate ones.
#   scripts/p6/prune_window.sh <window-dir>
# Refuses to touch a window whose status.json is not `done` or `failed`, and never
# deletes the referenced checkpoint, gate evidence, logs or the sibling mixture.
set -euo pipefail
WINDOW="${1:?window dir}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
phase="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('phase',''))" "$WINDOW/status.json")"
case "$phase" in done|failed) ;; *) echo "refusing: $WINDOW is in phase '$phase'"; exit 2 ;; esac
ART="$WINDOW/artifacts"
keep="$(python3 -c "import json,sys,os; print(os.path.basename(json.load(open(sys.argv[1]))['checkpoint_dir']))" "$ART/candidate-model.json" 2>/dev/null || true)"
[ -n "$keep" ] || { echo "no candidate-model.json in $ART — keeping the newest checkpoint instead"; keep="$(ls "$ART/checkpoints" | tail -1)"; }
before="$(du -sh "$ART/checkpoints" | cut -f1)"
for d in "$ART"/checkpoints/*/; do
  name="$(basename "$d")"
  [ "$name" = "$keep" ] && continue
  rm -rf "$d"
done
echo "$WINDOW: kept $keep, checkpoints $before -> $(du -sh "$ART/checkpoints" | cut -f1)"
