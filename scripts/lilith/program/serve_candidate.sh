#!/usr/bin/env bash
# Serve one training-program stage adapter with vLLM for an on-policy round, on the TRAINING pod.
# Thin wrapper over scripts/lilith/serve/serve_pod.sh's --enable-lora / --lora-modules path: it
# writes the lilith/serve-candidate@1 selection that serve_pod.sh reads through
# LILITH_SERVE_CANDIDATE_SELECTION (serve_adapter.py candidate), pins the adapter by sha256,
# starts vLLM detached and waits for it to be healthy. Nothing is activated or certified.
#
#   printf '%s\n' "$KEY" | bash serve_candidate.sh start --run-id ID --stage S --main-sha SHA [--root /workspace/lilith]
#   bash serve_candidate.sh stop   [--root /workspace/lilith]
#   bash serve_candidate.sh status [--root /workspace/lilith]
#
# The API key (>= 32 chars, chosen by the Mac supervisor per round) arrives on stdin, is exported to the
# serve_pod.sh process only and is never written to disk or argv; vLLM enforces it.
#
# Reaching it: vLLM listens on :8000 (LILITH_SERVE_PORT). With the pod's HTTP port 8000 exposed, the
# proxy URL is https://<pod-id>-8000.proxy.runpod.net/v1 (OpenAI-compatible; model name `lilith`).
# The Runpod proxy answers 403 to Python-urllib's default User-Agent: Python clients must send
# `User-Agent: curl/8.7.1`.
#
# GPU sharing: the trainer (poll mode) keeps the 12B parent resident while it waits, so vLLM defaults to
# LILITH_SERVE_GPU_MEMORY_UTILIZATION=0.45 and a 32K context. If it cannot fit next to the trainer, run the
# trainer with --onpolicy-wait-mode exit (it checkpoints and exits 75 so the whole GPU is free) and re-start it
# with --resume after the archive is delivered. The self-stop backstop of serve_pod.sh is disabled here
# (LILITH_SERVE_POD_HARD_CAP_SECONDS=0): the host supervisor owns the pod stop, and a backstop would kill training.
set -euo pipefail
umask 077

die() {  # die <code> <message> <help>
  printf 'serve_candidate: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ACTION="${1:-}"
[[ -n "$ACTION" ]] || die lilith::candidate::usage "no action" "use start, stop or status"
shift

ROOT=/workspace/lilith RUN_ID="" STAGE="" MAIN_SHA=""
while (( $# )); do
  case "$1" in
    --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
    --run-id) RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
    --stage) STAGE="${2:?--stage needs a value}"; shift 2 ;;
    --main-sha) MAIN_SHA="${2:?--main-sha needs a value}"; shift 2 ;;
    *) die lilith::candidate::usage "unknown argument $1" "see the header of serve_candidate.sh" ;;
  esac
done
if [[ "$ROOT" != /workspace/* && "${LILITH_ALLOW_NON_WORKSPACE:-0}" != 1 ]]; then
  die lilith::candidate::root "root $ROOT is not on the /workspace volume" "a pod stop wipes everything else"
fi
SERVE="$ROOT/serve"
PY="${LILITH_SERVE_PYTHON:-$ROOT/env/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"

case "$ACTION" in
  start)
    [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$ ]] || die lilith::candidate::usage "--run-id is missing or unsafe" "pass the manifest run_id"
    [[ "$STAGE" =~ ^(white|yellow|green|brown|black)$ ]] || die lilith::candidate::usage "--stage '$STAGE' is not a belt stage" "white, yellow, green, brown or black"
    [[ "$MAIN_SHA" =~ ^[0-9a-f]{40}$ ]] || die lilith::candidate::usage "--main-sha must be a 40-hex commit id" "the payload's main SHA (serve_pod.sh verifies provenance against it)"
    ADAPTER_DIR="$ROOT/program/$RUN_ID/stages/$STAGE/adapter"
    [[ -f "$ADAPTER_DIR/adapter_model.safetensors" && -f "$ADAPTER_DIR/adapter_config.json" ]] ||
      die lilith::candidate::adapter "no saved stage adapter at $ADAPTER_DIR" "the trainer writes it before onpolicy-needed.json"
    IFS= read -r KEY || true
    (( ${#KEY} >= 32 )) || die lilith::candidate::api_key "no API key (>= 32 chars) on stdin" "pipe the per-round key to stdin; it is never an argument"
    mkdir -p "$SERVE"
    chmod 700 "$SERVE"
    if [[ -f "$SERVE/serve.pid" ]]; then
      read -r other_host other_pid < "$SERVE/serve.pid" || true
      if [[ "${other_host:-}" == "$(hostname)" && -n "${other_pid:-}" ]] && kill -0 "$other_pid" 2>/dev/null; then
        die lilith::candidate::running "vLLM already serves (pid $other_pid)" "run stop first"
      fi
    fi
    SELECTION="$SERVE/candidate-selection.json"
    "$PY" - "$ADAPTER_DIR" "$RUN_ID" "$STAGE" "$SELECTION" <<'PY'
import json, sys
from pathlib import Path
adapter, run_id, stage, out = sys.argv[1:5]
receipt = json.loads((Path(adapter) / 'lilith-adapter-receipt.json').read_text())
rank = int(json.loads((Path(adapter) / 'adapter_config.json').read_text())['r'])
allowed = (1, 8, 16, 32, 64, 128, 256, 320, 512)
record = {'schema': 'lilith/serve-candidate@1', 'run_id': run_id, 'stage': stage, 'adapter_dir': adapter,
          'adapter_model_sha256': receipt['files_sha256']['adapter_model.safetensors'], 'served_model_name': 'lilith',
          'max_lora_rank': next(r for r in allowed if rank <= r)}
Path(out).write_text(json.dumps(record, indent=2) + '\n')
PY
    rm -f "$SERVE/status.json"
    mkdir -p "$SERVE/logs"
    export LILITH_ROOT="$ROOT" LILITH_EXPECTED_MAIN_SHA="$MAIN_SHA" LILITH_SERVE_CANDIDATE_SELECTION="$SELECTION"
    export LILITH_SERVE_POD_HARD_CAP_SECONDS=0 LILITH_SERVE_SKIP_PID1_ENV=1
    export LILITH_SERVE_GPU_MEMORY_UTILIZATION="${LILITH_SERVE_GPU_MEMORY_UTILIZATION:-0.45}"
    export LILITH_SERVE_MAX_MODEL_LEN="${LILITH_SERVE_MAX_MODEL_LEN:-32768}"
    export LILITH_SERVE_API_KEY="$KEY"
    unset KEY
    nohup setsid bash "$PAYLOAD/scripts/lilith/serve/serve_pod.sh" > "$SERVE/candidate-start.log" 2>&1 < /dev/null &
    unset LILITH_SERVE_API_KEY
    TIMEOUT=$(( ${LILITH_SERVE_HEALTH_TIMEOUT_SECONDS:-1500} + 120 ))
    deadline=$(( SECONDS + TIMEOUT ))
    while (( SECONDS < deadline )); do
      if [[ -f "$SERVE/status.json" ]]; then
        state="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state",""))' "$SERVE/status.json")"
        case "$state" in
          ready) cat "$SERVE/status.json"; exit 0 ;;
          failed | exited | stopped) cat "$SERVE/status.json" >&2; die lilith::candidate::start "serve_pod.sh reported $state" "see $SERVE/candidate-start.log and $SERVE/logs/" ;;
        esac
      fi
      sleep 5
    done
    die lilith::candidate::health "vLLM was not ready within ${TIMEOUT}s" "see $SERVE/candidate-start.log; run stop before retrying"
    ;;
  stop)
    if [[ ! -f "$SERVE/serve.pid" ]]; then echo '{"stopped": true, "was_running": false}'; exit 0; fi
    read -r host pid < "$SERVE/serve.pid" || true
    if [[ -z "${pid:-}" ]] || ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$SERVE/serve.pid"; echo '{"stopped": true, "was_running": false}'; exit 0
    fi
    # serve_pod.sh was started under setsid: it leads a process group holding vLLM and its workers.
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
    kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pgid" 2>/dev/null || true; sleep 2; fi
    kill -0 "$pid" 2>/dev/null && die lilith::candidate::stop "serve_pod.sh (pid $pid) survived SIGKILL" "the GPU is still held; inspect the pod"
    rm -f "$SERVE/serve.pid"
    echo '{"stopped": true, "was_running": true}'
    ;;
  status)
    if [[ -f "$SERVE/status.json" ]]; then cat "$SERVE/status.json"; else echo '{"state": "absent"}'; fi
    ;;
  *) die lilith::candidate::usage "unknown action $ACTION" "use start, stop or status" ;;
esac
