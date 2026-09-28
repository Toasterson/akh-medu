#!/usr/bin/env bash
# Lilith C6 H100 smoke (Anima WI 01a0e270-6021): one-shot, fail-closed, <= 2 GPU-hours.
#
#   LILITH_EXPECTED_MAIN_SHA=<40-hex main commit> bash <payload>/scripts/lilith/runpod/run_smoke.sh <run-id>
#
# The pod receives only the payload built by make_payload.sh (never the whole
# repository). Sequence, each step bounded by the remaining budget:
#   provenance (payload files vs source-manifest.json vs LILITH_EXPECTED_MAIN_SHA)
#   -> bootstrap --check -> GPU identity -> data pin
#   -> lilith_training_cache_control.py on CUDA with resident KV
#   -> memory probe 8K/16K/32K -> reference micro-run 8K/16K (+ generation tok/s)
#   -> cold reload of the 8K adapter in a fresh process
#   -> smoke-result.json, then SHA256SUMS (last), then the results tarball.
#
# Refuses an existing run directory (one-shot; resuming is a new run id). The EXIT
# trap always writes the result, sums and tarball, including on failure.
# Environment (recorded in smoke-result.json):
#   LILITH_EXPECTED_MAIN_SHA (required), LILITH_SOURCE_MANIFEST (default <payload>/source-manifest.json),
#   LILITH_ROOT (default /workspace/lilith), LILITH_SMOKE_BUDGET_SECONDS (7200),
#   LILITH_SMOKE_SYNTHETIC=1 (no private subset: recorded deviation),
#   LILITH_SMOKE_ALLOW_GPU=1 (non-H100),
#   LILITH_ALLOW_UNMERGED=1 (provenance failures recorded as unverified_allowed instead of stopping).
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # payload root (or a full checkout)
ROOT="${LILITH_ROOT:-/workspace/lilith}"
BUDGET="${LILITH_SMOKE_BUDGET_SECONDS:-7200}"
RUN_ID="${1:-}"

die() {
  printf 'run_smoke: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  die lilith::smoke::usage "run id '${RUN_ID}' is missing or invalid" "pass e.g. c6-smoke-20260928a"
if [[ "$ROOT" != /workspace/* && "${LILITH_ALLOW_NON_WORKSPACE:-0}" != 1 ]]; then
  die lilith::smoke::root "root $ROOT is not on the /workspace volume" "a pod stop wipes everything else"
fi
RUN="$ROOT/runs/$RUN_ID"
[[ ! -e "$RUN" ]] || die lilith::smoke::exists "$RUN already exists" \
  "runs are one-shot; choose a new run id (a stopped run is never resumed in place)"
mkdir -p "$ROOT/runs"
mkdir "$RUN" || die lilith::smoke::exists "could not claim $RUN" "another smoke claimed it; choose a new run id"

# ------------------------------------------------------------- claimed run dir
unset HF_TOKEN
PY="$ROOT/env/bin/python"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$ROOT/.hf-home"
export CUBLAS_WORKSPACE_CONFIG=":4096:8" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
MARKER="lilith-c6-smoke:$RUN_ID"
START="$(date +%s)"
STEPS="$RUN/steps.tsv"
printf '%s\n' "$$" > "$RUN/smoke.pid"
printf '%s\n' "$MARKER" > "$RUN/smoke.marker"
printf 'step\tstatus\texit_code\tseconds\n' > "$STEPS"
exec 3>&1 4>&2                                   # original stdout/stderr for after the log closes
exec > >(tee -a "$RUN/smoke.log") 2>&1
TEE_PID=$!
echo "$MARKER started $(date -u +%FT%TZ) budget=${BUDGET}s"

DATA="$ROOT/data/sft-subset-v1.jsonl"
DATA_ARGS=(--data "$DATA")
if [[ "${LILITH_SMOKE_SYNTHETIC:-0}" == 1 ]]; then DATA_ARGS=(--synthetic-data); fi

finish() {
  local rc=$?
  set +e
  local py="$PY"; [[ -x "$py" ]] || py=python3
  "$py" - "$RUN" "$RUN_ID" "$rc" "$START" "$BUDGET" <<'PY'
import csv, json, os, sys, time
from pathlib import Path
run, run_id, rc, start, budget = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
steps = list(csv.DictReader((run / 'steps.tsv').open(), delimiter='\t'))
def load(name):
    path = run / name
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return {'unreadable': name}
micro = load('microrun.json') or {}
provenance = load('provenance.json') or {}
result = {'schema': 'lilith/c6-smoke-result@1', 'run_id': run_id,
          'status': 'passed' if rc == 0 and all(s['status'] == 'ok' for s in steps) else 'failed',
          'exit_code': rc, 'steps': steps, 'wall_clock_seconds': time.time() - start,
          'budget_seconds': budget, 'quality_claim': False, 'release_selected': False,
          'knobs': {k: os.environ.get(k) for k in ('LILITH_EXPECTED_MAIN_SHA', 'LILITH_SOURCE_MANIFEST',
                                                   'LILITH_SMOKE_SYNTHETIC', 'LILITH_SMOKE_ALLOW_GPU',
                                                   'LILITH_ALLOW_UNMERGED', 'LILITH_SMOKE_BUDGET_SECONDS')},
          'provenance': {k: provenance.get(k) for k in ('status', 'expected_main_sha', 'manifest_sha256',
                                                        'allow_unmerged', 'problems')},
          'cache_control_status': (load('cache-control.json') or {}).get('status'),
          'memory_probe_status': (load('memory-probe.json') or {}).get('status'),
          'memory_probe': {k: {f: v.get(f) for f in ('status', 'peak_reserved_bytes', 'headroom_fraction',
                                                      'steady_state_tokens_per_second')}
                           for k, v in ((load('memory-probe.json') or {}).get('contexts') or {}).items()},
          'microrun_status': micro.get('status'), 'microrun_is_reference': micro.get('is_reference'),
          'microrun_contexts': {k: {f: v.get(f) for f in ('status', 'train_tokens_per_second', 'peak_reserved_bytes',
                                                          'seconds')} | {'adapter_tensor_sha256':
                                    (v.get('adapter') or {}).get('tensor_sha256')}
                                for k, v in (micro.get('contexts') or {}).items()},
          'generation': {f: (micro.get('generation') or {}).get(f)
                         for f in ('backend', 'tokens_per_second', 'vllm_error')},
          'cold_reload_status': (load('cold-reload.json') or {}).get('status')}
(run / 'smoke-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'exit_code': rc}))
PY
  echo "$MARKER finished rc=$rc; packaging $RUN/results-$RUN_ID.tar.gz"
  rm -f "$RUN/smoke.pid"
  # Close the log so its last line is on disk before it is hashed.
  exec 1>&3 2>&4
  local waited=0
  while kill -0 "$TEE_PID" 2>/dev/null && (( waited < 50 )); do sleep 0.2; waited=$(( waited + 1 )); done
  # SHA256SUMS is the last file written inside the run dir; only the tarball follows.
  (cd "$RUN" && find . -type f ! -name 'results-*.tar.gz' ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum > SHA256SUMS)
  tar -C "$ROOT/runs" -czf "$RUN/results-$RUN_ID.tar.gz" \
    --exclude="$RUN_ID/results-$RUN_ID.tar.gz" --exclude='*/adapter_model.safetensors' "$RUN_ID"
  printf '%s results=%s rc=%s\n' "$MARKER" "$RUN/results-$RUN_ID.tar.gz" "$rc"
}
trap finish EXIT

remaining() { echo $(( BUDGET - ($(date +%s) - START) )); }

step() {  # step <name> <command...>; external commands are bounded by the remaining budget
  local name="$1"; shift
  local left; left="$(remaining)"
  if (( left < 120 )); then
    printf '%s\tbudget_exhausted\t-\t0\n' "$name" >> "$STEPS"
    echo "budget exhausted before step $name"; exit 1
  fi
  echo "== step $name (remaining ${left}s)"
  local t0; t0="$(date +%s)"
  local rc=0
  if declare -F "$1" >/dev/null; then
    # In-shell checks run with set -e suspended (|| context): every command in
    # them carries an explicit `|| return 1`, so nothing can pass by falling through.
    "$@" || rc=$?
  else
    timeout --signal=TERM --kill-after=60 "$left" "$@" || rc=$?
  fi
  local status=ok
  if (( rc == 124 )); then status=timeout; elif (( rc != 0 )); then status=failed; fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$(( $(date +%s) - t0 ))" >> "$STEPS"
  if (( rc != 0 )); then echo "step $name $status (exit $rc)"; exit 1; fi
}

provenance() {
  local manifest="${LILITH_SOURCE_MANIFEST:-$REPO/source-manifest.json}"
  local py="$PY"; [[ -x "$py" ]] || py=python3          # verify_payload.py is stdlib-only
  local allow=()
  if [[ "${LILITH_ALLOW_UNMERGED:-0}" == 1 ]]; then allow=(--allow-unmerged); fi
  "$py" "$SCRIPT_DIR/verify_payload.py" verify --root "$REPO" --manifest "$manifest" \
    --expected-sha "${LILITH_EXPECTED_MAIN_SHA:-}" --output "$RUN/provenance.json" \
    ${allow[@]+"${allow[@]}"} || return 1
}

gpu_identity() {
  nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader > "$RUN/gpu.csv" || return 1
  cat "$RUN/gpu.csv" || return 1
  if ! grep -q 'H100' "$RUN/gpu.csv" && [[ "${LILITH_SMOKE_ALLOW_GPU:-0}" != 1 ]]; then
    echo "not an H100; the C6 reference is defined on 1x H100 80GB (LILITH_SMOKE_ALLOW_GPU=1 to override)" >&2
    return 1
  fi
}

data_pin() {
  if [[ "${LILITH_SMOKE_SYNTHETIC:-0}" == 1 ]]; then
    echo "synthetic data requested: recorded deviation, loss curve not comparable"; return 0
  fi
  "$PY" - "$DATA" "$REPO/training/lilith/transcripts-20260922/sft-subset-v1-manifest.json" <<'PY' || return 1
import hashlib, json, sys
from pathlib import Path
data, manifest = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
if not data.is_file():
    sys.exit(f'{data} missing: transfer the admitted subset or set LILITH_SMOKE_SYNTHETIC=1')
observed = hashlib.sha256(data.read_bytes()).hexdigest()
if observed != manifest['private_subset_sha256']:
    sys.exit(f'{data} sha256 {observed} != manifest {manifest["private_subset_sha256"]}')
print('subset pinned', observed)
PY
}

step provenance provenance
step bootstrap-check bash -c "bash '$SCRIPT_DIR/bootstrap.sh' --check > '$RUN/bootstrap-check.json'"
step gpu gpu_identity
step data-pin data_pin
step cache-control "$PY" "$REPO/scripts/session-platform/lilith_training_cache_control.py" \
  --parent "$ROOT/parent" --output "$RUN/cache-control.json" --cache-offloading false
step memory-probe "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode memory-probe \
  --parent "$ROOT/parent" --output "$RUN/memory-probe.json"
# Leave 15 minutes for generation measurement, cold reload and packaging.
MICRO_WALL=$(( $(remaining) - 900 ))
step microrun "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode microrun --parent "$ROOT/parent" \
  "${DATA_ARGS[@]}" --adapter-root "$RUN/adapters" --output "$RUN/microrun.json" \
  --max-wall-seconds "$MICRO_WALL"
step cold-reload "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode cold-reload --parent "$ROOT/parent" \
  --adapter-dir "$RUN/adapters/ctx-8192" --probe "$RUN/adapters/ctx-8192.probe.json" \
  --output "$RUN/cold-reload.json"
echo "$MARKER all steps passed"
