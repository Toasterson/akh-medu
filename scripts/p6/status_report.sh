#!/usr/bin/env bash
# Phase 51 P6 — one-screen status of the local window queue, printed at every update.
#   scripts/p6/status_report.sh [runs-root]   (default runs/p6-4)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
ROOT="${1:-runs/p6-4}"
echo "=== P6 status $(date -u +%FT%TZ) ==="
df -h / | awk 'NR==2{printf "disk: %s free of %s (%s used)\n", $4, $2, $5}'
if [ -d runs ]; then
  printf "runs/ total %s — " "$(du -sh runs 2>/dev/null | cut -f1)"
  du -sh runs/* 2>/dev/null | sort -rh | head -4 | awk '{printf "%s %s; ", $1, $2}'; echo
fi
[ -d "$ROOT" ] && { printf "%s: " "$ROOT"; du -sh "$ROOT"/window-* 2>/dev/null | sort -rh | head -5 | awk '{printf "%s %s; ", $1, $2}'; echo; }
QLOG="$(ls "$ROOT"/queue-*.txt.log 2>/dev/null | head -1)"
if pgrep -f window_queue.sh >/dev/null; then echo "queue: alive — $(tail -1 "$QLOG" 2>/dev/null)"; else echo "queue: NOT RUNNING — $(tail -1 "$QLOG" 2>/dev/null)"; fi
pgrep -f 'examples/train_knowledge' >/dev/null && echo "trainer: alive" || echo "trainer: idle"
pgrep -f 'eval_tool_contract_exam|eval_knowledge_exam' >/dev/null && echo "gate: running"
for w in "$ROOT"/window-*/; do
  [ -f "$w/status.json" ] || continue
  name="$(basename "$w")"
  phase="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('phase','?'))" "$w/status.json")"
  step="$(grep -E '^step ' "$w/window.log" 2>/dev/null | tail -1 | awk '{print $2}')"
  ev="$(grep -E 'eval@' "$w/window.log" 2>/dev/null | tail -1 | sed 's/^ *//')"
  ck="$(ls "$w/artifacts/checkpoints" 2>/dev/null | tail -1)"
  nck="$(ls "$w/artifacts/checkpoints" 2>/dev/null | wc -l | tr -d ' ')"
  line="$name: $phase"
  [ -n "$step" ] && line="$line step $step"
  [ -n "$ck" ] && line="$line ckpt $ck (${nck} kept)"
  [ -n "$ev" ] && line="$line | $ev"
  if [ "$phase" = done ] || [ "$phase" = failed ]; then
    line="$line | $(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('detail','')[:140])" "$w/status.json")"
  fi
  echo "$line"
done
