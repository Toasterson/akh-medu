#!/usr/bin/env bash
# Lilith serve pod entrypoint (Anima WI 01a0e270-314e, Lilith C2 Stage 0; ADR 116 Decision 7).
# Runs ON the Runpod pod `akh-medu-lilith-serve` at every pod start (the pod's start command
# launches it from /workspace/lilith/serve/current/, see lilith_serve.py START_COMMAND), or by
# hand over ssh after `lilith_serve.py install-payload --launch`.
#
#   bash /workspace/lilith/serve/current/scripts/lilith/serve/serve_pod.sh
#
# Steps (every step fails closed; state lands in $ROOT/serve/sessions/<session>/):
#   1. provenance   payload files vs source-manifest.json vs $ROOT/serve/current.json main_sha
#                   (written by the Mac supervisor after its merge-base check on origin/main)
#   2. api key      LILITH_SERVE_API_KEY (Runpod pod env / secret) -> VLLM_API_KEY; never argv
#   3. bootstrap    runpod/bootstrap.sh --check; an env built from another lock, a missing or
#                   half-built env, or a broken ninja is moved aside and rebuilt from the
#                   payload's lock (bootstrap.sh install). Parent/driver failures are refusals.
#   4. selection    serve_adapter.py: frozen parent by default; the LoRA adapter under
#                   $ROOT/adapters/active/ only with a C4b-certified active.json. Exception for the
#                   training program: LILITH_SERVE_CANDIDATE_SELECTION=<lilith/serve-candidate@1 json>
#                   serves one sha256-pinned UNCERTIFIED stage adapter (serve_adapter.py candidate)
#   5. vLLM        OpenAI server on :$LILITH_SERVE_PORT (8000), served model name `lilith`,
#                   64K context, bf16, KV cache dtype LILITH_SERVE_KV_CACHE_DTYPE (auto), native
#                   tool calling: --enable-auto-tool-choice --tool-call-parser <serve_tools.py
#                   resolve: LILITH_SERVE_TOOL_PARSER, default auto = the registered gemma4 parser>
#                   --chat-template <the parent's pinned chat_template.jinja>. No parser, no serving.
#                   reasoning: --reasoning-parser <serve_tools.py resolve-reasoning:
#                   LILITH_SERVE_REASONING_PARSER, default auto = the registered gemma4 parser,
#                   'none' disables it>, resolved the same way but non-fatal when nothing fits —
#                   the pinned template opens a thinking channel whenever tools or a system
#                   message are present; without a reasoning parser it leaks into `content`.
#   6. health       /health 200, /v1/models lists `lilith` with the key AND answers 401
#                   without it (the key is enforced), then status.json state=ready
#   7. backstop     after LILITH_SERVE_POD_HARD_CAP_SECONDS (7800 = 2 h 10 min) the pod stops
#                   itself with its pod-scoped RUNPOD_API_KEY, in case the Mac idle-watch died
#
# Logs: $ROOT/serve/logs/vllm-<date>.log through log_redact.py — timestamps, the uvicorn
# access line (method, path, status) and vLLM stats; never prompt or response bodies. vLLM's
# own request logger is off at the source as well.
#
# Environment (all optional except the key):
#   LILITH_SERVE_API_KEY (required, >= 32 chars)   LILITH_ROOT (/workspace/lilith)
#   LILITH_SERVE_PORT (8000)                       LILITH_SERVE_MAX_MODEL_LEN (65536)
#   LILITH_SERVE_KV_CACHE_DTYPE auto|fp8|fp8_e4m3|fp8_e5m2 (auto = the model dtype, bf16).
#     fp8 halves KV memory; Gemma 4's KV is small (0.60 GB/sequence at 32K bf16, see the local
#     GPU qualification plan §2a), so auto is right on the H100. int8 is not a vLLM KV cache
#     dtype; int8 / TurboQuant-class KV belongs to the local-card lane (C6b), not this pod.
#   LILITH_SERVE_GPU_MEMORY_UTILIZATION (0.90)     LILITH_SERVE_ADAPTER auto|parent (auto)
#   LILITH_SERVE_HEALTH_TIMEOUT_SECONDS (1500)     LILITH_SERVE_POD_HARD_CAP_SECONDS (7800; 0 = off)
#   LILITH_SERVE_TOOL_PARSER (auto)                LILITH_SERVE_CHAT_TEMPLATE ($ROOT/parent/chat_template.jinja)
#   LILITH_SERVE_REASONING_PARSER auto|none|<name> (auto; non-fatal — see step 5)
#   LILITH_EXPECTED_MAIN_SHA (overrides current.json)  LILITH_ALLOW_UNMERGED=1 (recorded deviation)
#   Numerics: LILITH_NUMERICS_SDPA_BACKEND (efficient) / LILITH_NUMERICS_SDPA_MATH_FALLBACK (1),
#     the recorded CUDA default (C6 round 3). They govern torch SDPA only; vLLM picks its own
#     attention kernels (flashinfer/triton) and is not covered by the C6 cache control.
#   Test hooks: LILITH_SERVE_BOOTSTRAP, LILITH_SERVE_VLLM, LILITH_SERVE_PYTHON,
#     LILITH_SERVE_RUNPOD_REST, LILITH_ALLOW_NON_WORKSPACE=1.
set -euo pipefail
umask 077

# An ssh session (install-payload --launch) does not inherit the container env that Runpod set
# at create time; PID 1 has it. Import only our own settings and the pod identity, without eval.
if [[ -r /proc/1/environ && "${LILITH_SERVE_SKIP_PID1_ENV:-0}" != 1 ]]; then
  while IFS= read -r -d '' entry; do
    name="${entry%%=*}"
    case "$name" in
      LILITH_SERVE_* | LILITH_ROOT | LILITH_CUDA_LANE | RUNPOD_POD_ID | RUNPOD_API_KEY)
        [[ -n "${!name:-}" ]] || export "$name=${entry#*=}" ;;
    esac
  done < /proc/1/environ
  unset entry name
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT="${LILITH_ROOT:-/workspace/lilith}"
SERVE="$ROOT/serve"
LOGS="$SERVE/logs"
PORT="${LILITH_SERVE_PORT:-8000}"
MAX_LEN="${LILITH_SERVE_MAX_MODEL_LEN:-65536}"
KV_DTYPE="${LILITH_SERVE_KV_CACHE_DTYPE:-auto}"
GPU_UTIL="${LILITH_SERVE_GPU_MEMORY_UTILIZATION:-0.90}"
ADAPTER_MODE="${LILITH_SERVE_ADAPTER:-auto}"
HEALTH_TIMEOUT="${LILITH_SERVE_HEALTH_TIMEOUT_SECONDS:-1500}"
HARD_CAP="${LILITH_SERVE_POD_HARD_CAP_SECONDS:-7800}"
BOOTSTRAP="${LILITH_SERVE_BOOTSTRAP:-$PAYLOAD/scripts/lilith/runpod/bootstrap.sh}"
case "${LILITH_CUDA_LANE:-cu130}" in   # same lane as bootstrap.sh; cu130 = <root>/env (default), cu128 = <root>/env-cu128
  cu130) ENV_NAME=env ;;
  cu128) ENV_NAME=env-cu128 ;;
  *) printf 'serve_pod: error[lilith::serve::lane]: unknown LILITH_CUDA_LANE=%s\n  help: use cu130 (default) or cu128\n' "$LILITH_CUDA_LANE" >&2
     exit 1 ;;
esac
ENV_DIR="$ROOT/$ENV_NAME"
VLLM="${LILITH_SERVE_VLLM:-$ENV_DIR/bin/vllm}"
PY="${LILITH_SERVE_PYTHON:-$ENV_DIR/bin/python}"
RUNPOD_REST="${LILITH_SERVE_RUNPOD_REST:-https://rest.runpod.io/v1}"
SERVED_NAME=lilith

die() {  # die <code> <message> <help>
  printf 'serve_pod: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  if [[ -n "${SESSION_DIR:-}" ]]; then write_status failed "$1: $2"; fi
  exit 1
}
log() { printf '%s serve_pod: %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

# ------------------------------------------------------------------ validation
if [[ "$ROOT" != /workspace/* && "${LILITH_ALLOW_NON_WORKSPACE:-0}" != 1 ]]; then
  die lilith::serve::root "root $ROOT is not on the /workspace volume" \
    "a pod stop wipes everything else; set LILITH_ROOT under /workspace"
fi
[[ "$PORT" =~ ^[0-9]{2,5}$ ]] || die lilith::serve::usage "LILITH_SERVE_PORT '$PORT' is not a port" "use 8000"
[[ "$MAX_LEN" =~ ^[1-9][0-9]*$ ]] || die lilith::serve::usage "LILITH_SERVE_MAX_MODEL_LEN '$MAX_LEN' is not a positive integer" "use 65536"
case "$KV_DTYPE" in
  auto | fp8 | fp8_e4m3 | fp8_e5m2) ;;
  *) die lilith::serve::kv_dtype "KV cache dtype '$KV_DTYPE' is not supported here" \
       "choose auto (default), fp8, fp8_e4m3 or fp8_e5m2; int8/TurboQuant KV is the local-card lane (C6b)" ;;
esac
[[ "$GPU_UTIL" =~ ^0\.[0-9]+$ ]] || die lilith::serve::usage "LILITH_SERVE_GPU_MEMORY_UTILIZATION '$GPU_UTIL' is not in (0,1)" "use 0.90"
case "$ADAPTER_MODE" in auto | parent) ;; *)
  die lilith::serve::usage "LILITH_SERVE_ADAPTER '$ADAPTER_MODE' is not auto|parent" "unset it for auto" ;;
esac
[[ "$HEALTH_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die lilith::serve::usage "LILITH_SERVE_HEALTH_TIMEOUT_SECONDS must be a positive integer" "use 1500"
[[ "$HARD_CAP" =~ ^[0-9]+$ ]] || die lilith::serve::usage "LILITH_SERVE_POD_HARD_CAP_SECONDS must be an integer" "use 7800, or 0 to disable"

mkdir -p "$LOGS" "$SERVE/sessions"
chmod 700 "$SERVE" "$LOGS"
SESSION="s-$(date -u +%Y%m%dT%H%M%SZ)-$$"
SESSION_DIR="$SERVE/sessions/$SESSION"
mkdir -p "$SESSION_DIR"
[[ -x "$PY" ]] || PY="$(command -v python3)"

write_status() {  # write_status <state> [detail]; the supervisor reads status.json over ssh
  local tmp="$SERVE/.status.json.$$"
  "$PY" - "$1" "${2:-}" "$SESSION" "$PORT" > "$tmp" <<'PY'
import datetime, json, sys
state, detail, session, port = sys.argv[1:5]
print(json.dumps({'schema': 'lilith/serve-status@1', 'state': state, 'detail': detail, 'session': session,
                  'port': int(port), 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
PY
  mv "$tmp" "$SERVE/status.json"
  cp "$SERVE/status.json" "$SESSION_DIR/status.json"
}

# Single instance per host: the pod start command and a manual --launch must not race.
PIDFILE="$SERVE/serve.pid"
HOST_ID="$(hostname)"
if [[ -f "$PIDFILE" ]]; then
  read -r other_host other_pid < "$PIDFILE" || true
  if [[ "${other_host:-}" == "$HOST_ID" && -n "${other_pid:-}" ]] && kill -0 "$other_pid" 2>/dev/null; then
    die lilith::serve::running "serve_pod.sh already runs on this pod (pid $other_pid)" \
      "check $SERVE/status.json; stop that process first to restart vLLM"
  fi
fi
printf '%s %s\n' "$HOST_ID" "$$" > "$PIDFILE"
VLLM_PID=""
BACKSTOP_PID=""
# shellcheck disable=SC2329  # invoked by the EXIT trap
cleanup() {
  if [[ -n "$BACKSTOP_PID" ]]; then kill "$BACKSTOP_PID" 2>/dev/null || true; fi
  if [[ -n "$VLLM_PID" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
  rm -f "$PIDFILE"
}
trap cleanup EXIT
trap 'write_status stopped "signal"; exit 143' TERM INT
write_status starting "session $SESSION"
log "session $SESSION (root $ROOT, payload $PAYLOAD)"

# ------------------------------------------------------------------ 1. provenance
EXPECTED="${LILITH_EXPECTED_MAIN_SHA:-}"
if [[ -z "$EXPECTED" && -f "$SERVE/current.json" ]]; then
  EXPECTED="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("main_sha",""))' "$SERVE/current.json")"
fi
ALLOW=()
[[ "${LILITH_ALLOW_UNMERGED:-0}" == 1 ]] && ALLOW=(--allow-unmerged)
"$PY" "$PAYLOAD/scripts/lilith/runpod/verify_payload.py" verify --root "$PAYLOAD" \
  --manifest "$PAYLOAD/source-manifest.json" --expected-sha "$EXPECTED" \
  --output "$SESSION_DIR/provenance.json" ${ALLOW[@]+"${ALLOW[@]}"} ||
  die lilith::serve::provenance "payload provenance failed (see $SESSION_DIR/provenance.json)" \
    "install a payload built by make_payload.sh from origin/main with lilith_serve.py install-payload"

# ------------------------------------------------------------------ 2. api key
KEY="${LILITH_SERVE_API_KEY:-}"
unset LILITH_SERVE_API_KEY
(( ${#KEY} >= 32 )) || die lilith::serve::api_key "LILITH_SERVE_API_KEY is missing or shorter than 32 characters" \
  "the pod env must carry the key from 1Password item lilith-serve-api-key (lilith_serve.py create sets it)"
export VLLM_API_KEY="$KEY"
unset KEY

# ------------------------------------------------------------------ 3. bootstrap
export LILITH_ROOT="$ROOT" PATH="$ENV_DIR/bin:$PATH" HF_HOME="$ROOT/.hf-home" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONNOUSERSITE=1
export LILITH_NUMERICS_SDPA_BACKEND="${LILITH_NUMERICS_SDPA_BACKEND:-efficient}"
export LILITH_NUMERICS_SDPA_MATH_FALLBACK="${LILITH_NUMERICS_SDPA_MATH_FALLBACK:-1}"
BOOT_ACTION=reused
if ! bash "$BOOTSTRAP" --check > "$SESSION_DIR/bootstrap-check.json" 2> "$SESSION_DIR/bootstrap-check.err"; then
  code="$(sed -n 's/.*error\[\(lilith::bootstrap::[a-z_]*\)\].*/\1/p' "$SESSION_DIR/bootstrap-check.err" | head -n1)"
  case "$code" in
    lilith::bootstrap::env_lock | lilith::bootstrap::ninja | lilith::bootstrap::env_versions)
      aside="$ENV_DIR.old-$(date -u +%Y%m%dT%H%M%SZ)"
      log "bootstrap check: $code; moving $ENV_DIR to $aside and rebuilding from the payload lock"
      mv "$ENV_DIR" "$aside" || die lilith::serve::bootstrap "could not move $ENV_DIR aside" "move it by hand and restart"
      BOOT_ACTION="rebuilt ($code; old env at $aside)" ;;
    lilith::bootstrap::env_missing | lilith::bootstrap::env_incomplete)
      BOOT_ACTION="built ($code)" ;;
    *)
      die lilith::serve::bootstrap "bootstrap --check failed with ${code:-an unknown error} (see $SESSION_DIR/bootstrap-check.err)" \
        "parent, driver and CUDA failures are not auto-repaired; fix the volume or recreate the pod on a host that meets the lane ($LILITH_CUDA_LANE, default cu130 = CUDA 13, cu128 = CUDA 12.8)" ;;
  esac
  bash "$BOOTSTRAP" >> "$SESSION_DIR/bootstrap-install.log" 2>&1 ||
    die lilith::serve::bootstrap "bootstrap install failed (see $SESSION_DIR/bootstrap-install.log)" \
      "inspect the log; a lock that no longer resolves needs a new payload from origin/main"
  bash "$BOOTSTRAP" --check > "$SESSION_DIR/bootstrap-check.json" 2>> "$SESSION_DIR/bootstrap-check.err" ||
    die lilith::serve::bootstrap "bootstrap --check still fails after the rebuild" "see $SESSION_DIR/bootstrap-check.err"
fi
[[ -x "$PY" ]] || PY="$ENV_DIR/bin/python"
log "bootstrap: $BOOT_ACTION"

# ------------------------------------------------------------------ 4. selection
if [[ -n "${LILITH_SERVE_CANDIDATE_SELECTION:-}" ]]; then
  # Training-program on-policy rounds (scripts/lilith/program/serve_candidate.sh): an UNCERTIFIED stage
  # adapter under $ROOT/program/ is served on the training pod only, pinned by sha256. Never the C2 serve pod.
  "$PY" "$SCRIPT_DIR/serve_adapter.py" candidate --selection "$LILITH_SERVE_CANDIDATE_SELECTION" --root "$ROOT" \
    > "$SESSION_DIR/selection.json" 2> "$SESSION_DIR/selection.err" ||
    die lilith::serve::candidate "$(head -n1 "$SESSION_DIR/selection.err")" "$(sed -n 's/^  help: //p' "$SESSION_DIR/selection.err" | head -n1)"
else
  "$PY" "$SCRIPT_DIR/serve_adapter.py" select --registry "$ROOT/adapters/active" --mode "$ADAPTER_MODE" \
    > "$SESSION_DIR/selection.json" 2> "$SESSION_DIR/selection.err" ||
    die lilith::serve::adapter "$(head -n1 "$SESSION_DIR/selection.err")" "$(sed -n 's/^  help: //p' "$SESSION_DIR/selection.err" | head -n1)"
fi
sel() { "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
          "$SESSION_DIR/selection.json" "$1"; }
MODE="$(sel mode)"

# ------------------------------------------------------------------ 5. vLLM
HELP_FILE="$SESSION_DIR/vllm-serve-help.txt"
"$VLLM" serve --help=all > "$HELP_FILE" 2>/dev/null || "$VLLM" serve --help > "$HELP_FILE" 2>/dev/null || true
HELP="$(cat "$HELP_FILE")"
# Tool calling is mandatory: the parser is resolved against the installed vLLM's registry.
if ! "$PY" "$SCRIPT_DIR/serve_tools.py" resolve --requested "${LILITH_SERVE_TOOL_PARSER:-auto}" \
     --help-file "$HELP_FILE" --chat-template "${LILITH_SERVE_CHAT_TEMPLATE:-$ROOT/parent/chat_template.jinja}" \
     > "$SESSION_DIR/tools.json" 2> "$SESSION_DIR/tools.err"; then
  first="$(head -n1 "$SESSION_DIR/tools.err")"
  code="${first#\[}"; code="${code%%\]*}"
  die "${code:-lilith::serve::tool_parser}" "${first#*] }" "$(sed -n 's/^  help: //p' "$SESSION_DIR/tools.err" | head -n1)"
fi
tool() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$SESSION_DIR/tools.json" "$1"; }
TOOL_PARSER="$(tool parser)"
CHAT_TEMPLATE="$(tool chat_template)"
log "tool calling: parser $TOOL_PARSER ($(tool reason)); chat template $CHAT_TEMPLATE"

# Reasoning-channel capture is non-fatal (see the step-5 header comment): resolution never dies
# on "auto, nothing registered" the way tool-call resolution does, only on an explicit name that
# isn't registered — that IS a refusal, same [code] message / help: shape as tools.err above.
if ! "$PY" "$SCRIPT_DIR/serve_tools.py" resolve-reasoning --requested "${LILITH_SERVE_REASONING_PARSER:-auto}" \
     > "$SESSION_DIR/reasoning.json" 2> "$SESSION_DIR/reasoning.err"; then
  first="$(head -n1 "$SESSION_DIR/reasoning.err")"
  code="${first#\[}"; code="${code%%\]*}"
  die "${code:-lilith::serve::reasoning_parser}" "${first#*] }" "$(sed -n 's/^  help: //p' "$SESSION_DIR/reasoning.err" | head -n1)"
fi
reasoning() { "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
          "$SESSION_DIR/reasoning.json" "$1"; }
REASONING_PARSER="$(reasoning parser)"
log "reasoning: $([[ -n "$REASONING_PARSER" ]] && echo "parser $REASONING_PARSER" || echo "disabled") ($(reasoning reason))"

ARGS=(serve "$ROOT/parent" --tokenizer "$ROOT/parent" --host 0.0.0.0 --port "$PORT"
      --dtype bfloat16 --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
      --kv-cache-dtype "$KV_DTYPE" --limit-mm-per-prompt '{"image": 0, "audio": 0}'
      --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --chat-template "$CHAT_TEMPLATE")
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ "$MODE" == adapter ]]; then
  ARGS+=(--served-model-name "$SERVED_NAME-parent" --enable-lora --max-loras 1
         --max-lora-rank "$(sel max_lora_rank)" --lora-modules "$SERVED_NAME=$(sel adapter_dir)")
else
  ARGS+=(--served-model-name "$SERVED_NAME")
fi
# Request bodies stay out of the logs: vLLM >= 0.10 logs requests only with
# --enable-log-requests (never passed); older releases need --disable-log-requests.
if grep -q -- '--enable-log-requests' <<<"$HELP"; then
  REQUEST_LOGGING="off (vLLM default; --enable-log-requests not passed)"
elif grep -q -- '--disable-log-requests' <<<"$HELP"; then
  ARGS+=(--disable-log-requests)
  REQUEST_LOGGING="off (--disable-log-requests)"
else
  REQUEST_LOGGING="unknown flag set; log_redact.py filters bodies"
fi

"$PY" - "$SESSION_DIR/session.json" "$SESSION" "$MODE" "$KV_DTYPE" "$MAX_LEN" "$PORT" "$REQUEST_LOGGING" \
  "$BOOT_ACTION" "$HARD_CAP" "$SESSION_DIR" "${ARGS[@]}" <<'PY'
import datetime, json, sys
from pathlib import Path
out, session, mode, kv, max_len, port, logging, boot, cap, sdir, *args = sys.argv[1:]
sdir = Path(sdir)
load = lambda name: json.loads((sdir / name).read_text()) if (sdir / name).is_file() else None
record = {'schema': 'lilith/serve-session@1', 'session': session, 'served_model_name': 'lilith', 'mode': mode,
          'selection': load('selection.json'), 'tool_calling': load('tools.json'),
          'reasoning': load('reasoning.json'), 'provenance': {k: (load('provenance.json') or {}).get(k)
          for k in ('status', 'expected_main_sha', 'manifest_sha256', 'problems')},
          'bootstrap': {'action': boot, 'check': load('bootstrap-check.json')},
          'kv_cache_dtype': kv, 'max_model_len': int(max_len), 'port': int(port), 'request_logging': logging,
          'numerics': {'label': 'sdpa=efficient+math', 'scope': 'torch SDPA only; vLLM selects its own attention '
                       'kernels and is outside the C6 cache control'},
          'pod_hard_cap_seconds': int(cap), 'vllm_args': args,
          'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
Path(out).write_text(json.dumps(record, indent=2) + '\n')
PY

log "starting vLLM ($MODE, kv $KV_DTYPE, ctx $MAX_LEN, port $PORT)"
"$VLLM" "${ARGS[@]}" > >("$PY" "$SCRIPT_DIR/log_redact.py" --dir "$LOGS" --prefix vllm) 2>&1 &
VLLM_PID=$!

# ------------------------------------------------------------------ 7. backstop (armed before health)
# shellcheck disable=SC2329  # invoked by the backstop subshell
self_stop() {
  log "on-pod hard cap ${HARD_CAP}s reached: stopping pod ${RUNPOD_POD_ID:-<unknown>}"
  write_status stopping "pod hard cap ${HARD_CAP}s"
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    if command -v runpodctl >/dev/null && { runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; } >&2; then
      return 0
    fi
    if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
      local code
      code="$(printf 'Authorization: Bearer %s\n' "$RUNPOD_API_KEY" |
              curl -sS -m 60 -o /dev/null -w '%{http_code}' -X POST -H @- "$RUNPOD_REST/pods/$RUNPOD_POD_ID/stop" || true)"
      log "self-stop via REST -> HTTP $code"
      [[ "$code" == 2?? ]] && return 0
    fi
  fi
  log "self-stop unavailable (no runpodctl / RUNPOD_API_KEY / RUNPOD_POD_ID); killing vLLM so the GPU idles — the pod still bills until stopped"
  kill "$VLLM_PID" 2>/dev/null || true
}
if (( HARD_CAP > 0 )); then
  ( sleep "$HARD_CAP"; self_stop ) &
  BACKSTOP_PID=$!
fi

# ------------------------------------------------------------------ 6. health
deadline=$(( SECONDS + HEALTH_TIMEOUT ))
until curl -fsS -m 5 -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; do
  kill -0 "$VLLM_PID" 2>/dev/null || die lilith::serve::vllm_exited "vLLM exited during startup" \
    "see $LOGS/vllm-$(date -u +%F).log"
  (( SECONDS < deadline )) || die lilith::serve::health_timeout "vLLM not healthy after ${HEALTH_TIMEOUT}s" \
    "see $LOGS/vllm-$(date -u +%F).log; raise LILITH_SERVE_HEALTH_TIMEOUT_SECONDS only if the load was progressing"
  sleep 5
done
models="$(printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" | curl -fsS -m 10 -H @- "http://127.0.0.1:$PORT/v1/models")" ||
  die lilith::serve::models "GET /v1/models with the key failed" "see the vLLM log"
"$PY" -c 'import json,sys; ids=[m["id"] for m in json.loads(sys.argv[1])["data"]]; sys.exit(0 if "lilith" in ids else 1)' "$models" ||
  die lilith::serve::models "/v1/models does not list 'lilith'" "check --served-model-name / --lora-modules in session.json"
unauth="$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" || true)"
[[ "$unauth" == 401 ]] || die lilith::serve::auth "/v1/models answered HTTP $unauth without the API key" \
  "the proxy URL is public; never serve without an enforced key. Check VLLM_API_KEY handling in this vLLM version"
write_status ready "vLLM healthy; model lilith ($MODE)"
log "ready: http://0.0.0.0:$PORT (model $SERVED_NAME, $MODE)"

rc=0
wait "$VLLM_PID" || rc=$?
write_status exited "vLLM exited rc=$rc"
log "vLLM exited rc=$rc"
exit "$rc"
