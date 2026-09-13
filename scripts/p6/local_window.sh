#!/usr/bin/env bash
# Phase 51 P6.4 — one supervised local remediation window + gate on the Mac (ADR 105/106/107).
#
# Runs `train_knowledge --remediate-from-bundle` (f32, Metal, recompute attention) from an
# uncertified candidate on base + engine-generated siblings, then reconstructs a candidate
# bundle from the new checkpoint and scores the unchanged quarantined sentinels
# (tools 1,272 + knowledge 8×100). Every phase writes `status.json` and posts a
# `coordination/status@v1` event to the Agent Board run thread; nothing wakes a chat.
#
#   scripts/p6/local_window.sh <window-dir> <parent-bundle> <siblings.jsonl> [extra train_knowledge args…]
#
# Required env (or ~/env-file): AGENT_BOARD_* for board_request.py. Optional:
#   P6_BOARD_THREAD (default: contents of /tmp/board-thread-id)
#   P6_LR (default 4e-6), P6_EPOCHS (1), P6_BATCH (1), P6_SEED (62000)
#   P6_BASE_TRAIN / P6_BASE_EVAL / P6_CONTRACTS (default: runs/checkpoints/local-curricula-v3-8k/sft/…)
#   P6_SENTINELS (default: runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/sentinels)
#   P6_SKIP_GATE=1 to stop after training.
set -euo pipefail

WINDOW="${1:?window dir}"; PARENT="${2:?parent candidate bundle}"; SIBLINGS="${3:?siblings train.sft.jsonl}"; shift 3
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
CURR="${P6_BASE_DIR:-runs/checkpoints/local-curricula-v3-8k/sft}"
BASE_TRAIN="${P6_BASE_TRAIN:-$CURR/tools.train.sft.jsonl}"
BASE_EVAL="${P6_BASE_EVAL:-$CURR/tools.eval.sft.jsonl}"
CONTRACTS="${P6_CONTRACTS:-$CURR/contracts.json}"
SENTINELS="${P6_SENTINELS:-runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/sentinels}"
LR="${P6_LR:-4e-6}"; EPOCHS="${P6_EPOCHS:-1}"; BATCH="${P6_BATCH:-1}"; SEED="${P6_SEED:-62000}"
FEATURES="${P6_FEATURES:-metal,grad_checkpointing}"
BOARD="$HOME/.claude/skills/agent-board-coordination/scripts/board_request.py"
THREAD="${P6_BOARD_THREAD:-$(cat /tmp/board-thread-id 2>/dev/null || true)}"
mkdir -p "$WINDOW"
LOG="$WINDOW/window.log"; STATUS="$WINDOW/status.json"
[ -f "$HOME/env-file" ] && { set -a; # shellcheck disable=SC1090
  source "$HOME/env-file" >/dev/null 2>&1 || true; set +a; }

status() { # phase detail
  python3 - "$STATUS" "$1" "$2" <<'EOF'
import json, sys, datetime
p, phase, detail = sys.argv[1:4]
try: d = json.load(open(p))
except Exception: d = {"schema_version": 1, "history": []}
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
d.update({"phase": phase, "detail": detail, "updated_at": now})
d["history"].append({"at": now, "phase": phase, "detail": detail})
json.dump(d, open(p, "w"), indent=2)
EOF
  echo "[$(date -u +%FT%TZ)] $1: $2" | tee -a "$LOG"
}

board() { # state summary detail
  [ -n "$THREAD" ] || return 0
  python3 - "$1" "$2" "$3" "$WINDOW" > /tmp/p6-board-msg.json <<'EOF'
import json, re, sys, time
state, summary, detail, window = sys.argv[1:5]
print(json.dumps({"message_type": "coordination/status@v1",
  "payload": {"state": state, "detail": detail},
  "summary": summary, "retention": {"kind": "durable"}, "reply_to": None,
  "idempotency_key": "status:p6-local-window:" + re.sub(r"[^A-Za-z0-9._-]+", "-", window).strip("-") + f":{state}:{int(time.time())}"}))
EOF
  python3 "$BOARD" post "/v1/threads/$THREAD/messages" @/tmp/p6-board-msg.json >/dev/null 2>&1 || echo "board post failed (non-fatal)" | tee -a "$LOG"
}

fail() { status failed "$1"; board blocked "P6 local window FAILED: $WINDOW" "$1"; exit 1; }
trap 'fail "aborted at line $LINENO"' ERR

status build "cargo build --release -p grug-train --features $FEATURES --example train_knowledge --example eval_tool_contract_exam --example eval_knowledge_exam"
cargo build --release -p grug-train --features "$FEATURES" --example train_knowledge --example eval_tool_contract_exam --example eval_knowledge_exam >> "$LOG" 2>&1
BIN=target/release/examples

status train "remediate from $PARENT with $SIBLINGS lr=$LR epochs=$EPOCHS batch=$BATCH"
board working "P6 local window started: $WINDOW" "train_knowledge --remediate-from-bundle $PARENT + $SIBLINGS (lr $LR, $EPOCHS epoch, batch $BATCH, $FEATURES)"
ART="$WINDOW/artifacts"; mkdir -p "$ART"
until mkdir /tmp/p6-gpu.lock 2>/dev/null; do sleep 30; done
trap 'rmdir /tmp/p6-gpu.lock 2>/dev/null; fail "aborted at line $LINENO"' ERR
"$BIN/train_knowledge" \
  --contracts "$CONTRACTS" --train-jsonl "$BASE_TRAIN" --eval-jsonl "$BASE_EVAL" \
  --supplemental-jsonl "$SIBLINGS" \
  --artifacts-dir "$ART" --model small_8k --epochs "$EPOCHS" --batch-size "$BATCH" \
  --learning-rate "$LR" --seed "$SEED" \
  --remediate-from-bundle "$PARENT" "$@" >> "$LOG" 2>&1
rmdir /tmp/p6-gpu.lock 2>/dev/null || true
trap 'fail "aborted at line $LINENO"' ERR
status trained "candidate-model.json at $ART"

if [ "${P6_SKIP_GATE:-0}" = "1" ]; then status done "training only (P6_SKIP_GATE=1)"; board working "P6 local window trained: $WINDOW" "gate skipped by request"; exit 0; fi

status gate "sentinel gate on $ART (tools 1272 + knowledge 8x100)"
GATE="$WINDOW/gate"; mkdir -p "$GATE"
until mkdir /tmp/p6-gpu.lock 2>/dev/null; do sleep 30; done
trap 'rmdir /tmp/p6-gpu.lock 2>/dev/null; fail "aborted at line $LINENO"' ERR
"$BIN/eval_tool_contract_exam" --bundle "$ART" --exam "$SENTINELS/all-tools.json" \
  --evidence-out "$GATE/all-tools.evidence.json" --predictions-out "$GATE/all-tools.predictions.json" \
  --observations-out "$GATE/all-tools.observations.jsonl" --decode-reports-out "$GATE/all-tools.decode-reports.json" \
  --max-new-tokens 768 --decoding constrained >> "$LOG" 2>&1
for exam in "$SENTINELS"/knowledge-*.json; do
  name="$(basename "$exam" .json)"
  "$BIN/eval_knowledge_exam" --bundle "$ART" --exam "$exam" \
    --evidence-out "$GATE/$name.evidence.json" --predictions-out "$GATE/$name.predictions.json" >> "$LOG" 2>&1 || echo "knowledge exam $name failed" >> "$LOG"
done
rmdir /tmp/p6-gpu.lock 2>/dev/null || true
trap 'fail "aborted at line $LINENO"' ERR

SUMMARY="$(python3 - "$GATE/all-tools.observations.jsonl" <<'EOF'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
p=sum(r['pass_count'] for r in rows); n=sum(r['sample_count'] for r in rows)
cat=sum(r['catastrophic_count'] for r in rows)
blockers=[f"{r['family'].split('.')[-1]} {r['pass_count']}/{r['sample_count']}{' cat'+str(r['catastrophic_count']) if r['catastrophic_count'] else ''}" for r in rows if r['pass_count']/r['sample_count']<0.85 or r['catastrophic_count']/r['sample_count']>0.01]
print(f"tools {p}/{n}, catastrophic {cat}; blockers: {', '.join(blockers) or 'none'}")
EOF
)"
status done "$SUMMARY"
board working "P6 local window gated: $WINDOW" "$SUMMARY — evidence in $GATE"
echo "$SUMMARY"
