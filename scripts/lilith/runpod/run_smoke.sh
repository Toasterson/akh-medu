#!/usr/bin/env bash
# Lilith C6 H100 smoke (Anima WI 01a0e270-6021): one-shot, fail-closed, <= 2 GPU-hours.
#
#   LILITH_EXPECTED_MAIN_SHA=<40-hex main commit> bash <payload>/scripts/lilith/runpod/run_smoke.sh \
#       [--only <step>[,<step>...]] <run-id>
#
# --only runs the gates plus the named steps (data-pin always runs; it is cheap):
#   cache-control, cache-diagnostic, memory-probe, microrun, cold-reload, vllm-generation
# e.g. --only cache-control,cache-diagnostic,vllm-generation skips the 58-minute micro-run.
# Still fail-closed: a named step whose prerequisite is not ok (or not selected) is
# recorded as skipped_dependency and fails the run; unselected steps get no row and
# smoke-result.json records the selection under "only".
#
# The pod receives only the payload built by make_payload.sh (never the whole
# repository). Steps, each started only when the global budget has >= 120s left
# and bounded by the remaining budget:
#   gates (a failure stops the run): provenance (payload files vs source-manifest.json
#     vs LILITH_EXPECTED_MAIN_SHA) -> bootstrap --check -> GPU identity
#   data pin
#   cache-control: lilith_training_cache_control.py on CUDA with resident KV
#   cache-diagnostic: lilith_cache_path_diagnostic.py, only when cache-control failed
#   memory-probe 8K/16K/32K                      (independent of cache-control)
#   microrun 8K/16K (+ generation tok/s)         (needs data-pin)
#   cold-reload of the 8K adapter, fresh process (needs microrun)
#   vllm-generation: strict vLLM LoRA generation check, no HF fallback (needs an adapter:
#     this run's microrun, or LILITH_GENERATION_ADAPTER_DIR)
# Every step runs with $LILITH_ROOT/env/bin first on PATH, so console scripts of the
# env (ninja for the vLLM/flashinfer JIT, and the rest) resolve to the pinned env.
# A failed non-gate step does not stop the independent steps after it; a step whose
# prerequisite failed is recorded as skipped_dependency. The overall status is
# FAILED when any step is not ok, and smoke-result.json lists every failed step.
#
# Completion marker: smoke-result.json is the LAST file written. The EXIT trap
# (also on failure) closes the log, writes SHA256SUMS (covering the result), packs
# results-<run-id>.tar.gz outside the run directory and moves it in, and only then
# renames the staged result into place. Supervisors wait for smoke-result.json to
# exist (scripts/lilith/supervise_pod.sh); never on process names or pgrep -f.
#
# Refuses an existing run directory (one-shot; resuming is a new run id).
# Environment (recorded in smoke-result.json):
#   LILITH_EXPECTED_MAIN_SHA (required), LILITH_SOURCE_MANIFEST (default <payload>/source-manifest.json),
#   LILITH_ROOT (default /workspace/lilith), LILITH_SMOKE_BUDGET_SECONDS (7200),
#   LILITH_SMOKE_SYNTHETIC=1 (no private subset: recorded deviation),
#   LILITH_SMOKE_ALLOW_GPU=1 (non-H100),
#   LILITH_ALLOW_UNMERGED=1 (provenance failures recorded as unverified_allowed instead of stopping),
#   LILITH_DIAGNOSTIC_BUDGET_SECONDS (1800; pass budget of the cache diagnostic, model load excluded;
#     the step is capped at budget + 300 s),
#   LILITH_GENERATION_ADAPTER_DIR (adapter for vllm-generation when microrun is not selected).
# Numerics (lilith_numerics.py; applies to every step below that runs GPU compute:
# cache-control, memory-probe, microrun, cold-reload, vllm-generation's HF fallback).
# Unset (the recorded default, lilith_numerics.platform_default()) = EFFICIENT SDPA
# with MATH fallback on CUDA — this smoke always runs on CUDA, so that is what
# "unset" now means; ROCm/CPU callers of lilith_numerics elsewhere keep PyTorch's
# own default. C6 round 2/3 (2026-09-28) found the true CUDA default
# nondeterministic on the cache control and confirmed EFFICIENT+MATH restores
# bit-exact resident-vs-offload equivalence; see lilith_numerics.py and
# training/results/lilith-c6-smoke-20260928-RESULT.md. Every run records the
# requested and observed settings under "numerics" in its result JSON.
#   LILITH_NUMERICS_SDPA_BACKEND=default|math|efficient|flash|cudnn,
#   LILITH_NUMERICS_SDPA_MATH_FALLBACK=1, LILITH_NUMERICS_TF32=default|on|off,
#   LILITH_NUMERICS_BF16_REDUCTION=default|on|off, LILITH_NUMERICS_DETERMINISTIC=1.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # payload root (or a full checkout)
ROOT="${LILITH_ROOT:-/workspace/lilith}"
BUDGET="${LILITH_SMOKE_BUDGET_SECONDS:-7200}"

die() {
  printf 'run_smoke: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}

SELECTABLE="cache-control cache-diagnostic memory-probe microrun cold-reload vllm-generation"
ONLY=""
if [[ "${1:-}" == --only ]]; then
  ONLY="${2:-}"
  shift 2 || die lilith::smoke::usage "--only needs a step list" "e.g. --only cache-control,cache-diagnostic"
  [[ -n "$ONLY" ]] || die lilith::smoke::usage "--only needs a step list" "choose from: $SELECTABLE"
  IFS=, read -r -a ONLY_STEPS <<< "$ONLY"
  for wanted in "${ONLY_STEPS[@]}"; do
    [[ " $SELECTABLE " == *" $wanted "* ]] ||
      die lilith::smoke::usage "--only step '$wanted' is unknown" "choose from: $SELECTABLE"
  done
fi
RUN_ID="${1:-}"
selected() {  # selected <step>: true in a full run, or when named in --only
  [[ -z "$ONLY" || ",$ONLY," == *",$1,"* ]]
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
# Console scripts of the pinned env (ninja for the vLLM/flashinfer JIT) come first.
export PATH="$ROOT/env/bin:$PATH"
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

STAGE="$ROOT/runs/.stage-$RUN_ID"   # outside the run dir: the result and tarball are built here

finish() {
  local rc=$?
  set +e
  local py="$PY"; [[ -x "$py" ]] || py=python3
  mkdir -p "$STAGE"
  LILITH_SMOKE_ONLY="$ONLY" "$py" - "$RUN" "$RUN_ID" "$rc" "$START" "$BUDGET" "$STAGE/smoke-result.json" <<'PY'
import csv, json, os, sys, time
from pathlib import Path
run, run_id, rc, start, budget = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
staged = Path(sys.argv[6])
steps = list(csv.DictReader((run / 'steps.tsv').open(), delimiter='\t'))
failed_steps = [s['step'] for s in steps if s['status'] != 'ok']
def load(name):
    path = run / name
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except json.JSONDecodeError:
        return {'unreadable': name}
micro = load('microrun.json') or {}
provenance = load('provenance.json') or {}
diagnostic = load('cache-diagnostic.json') or {}
result = {'schema': 'lilith/c6-smoke-result@1', 'run_id': run_id,
          'status': 'passed' if rc == 0 and steps and not failed_steps else 'failed',
          'exit_code': rc, 'steps': steps, 'failed_steps': failed_steps,
          'wall_clock_seconds': time.time() - start,
          'budget_seconds': budget, 'quality_claim': False, 'release_selected': False,
          'completion_marker': 'smoke-result.json is written last; SHA256SUMS covers it',
          'tarball': f'results-{run_id}.tar.gz',
          'only': os.environ.get('LILITH_SMOKE_ONLY') or None,
          'vllm_generation': {f: (load('vllm-generation.json') or {}).get(f)
                              for f in ('status', 'backend', 'tokens_per_second', 'vllm_error', 'error')},
          'knobs': {k: os.environ.get(k) for k in ('LILITH_EXPECTED_MAIN_SHA', 'LILITH_SOURCE_MANIFEST',
                                                   'LILITH_SMOKE_SYNTHETIC', 'LILITH_SMOKE_ALLOW_GPU',
                                                   'LILITH_ALLOW_UNMERGED', 'LILITH_SMOKE_BUDGET_SECONDS',
                                                   'LILITH_DIAGNOSTIC_BUDGET_SECONDS',
                                                   'LILITH_GENERATION_ADAPTER_DIR',
                                                   'LILITH_NUMERICS_SDPA_BACKEND',
                                                   'LILITH_NUMERICS_SDPA_MATH_FALLBACK', 'LILITH_NUMERICS_TF32',
                                                   'LILITH_NUMERICS_BF16_REDUCTION',
                                                   'LILITH_NUMERICS_DETERMINISTIC')},
          'provenance': {k: provenance.get(k) for k in ('status', 'expected_main_sha', 'manifest_sha256',
                                                        'allow_unmerged', 'problems')},
          'cache_control_status': (load('cache-control.json') or {}).get('status'),
          'cache_control_numerics': ((load('cache-control.json') or {}).get('numerics') or {}).get('label'),
          'cache_diagnostic_status': diagnostic.get('status'),
          'cache_diagnostic_verdict': {k: (diagnostic.get('verdict') or {}).get(k)
                                       for k in ('cause', 'restoring_variations_confirmed_tokenwise',
                                                 'restoring_variations_chunked_only',
                                                 'recommended_numerics_flags', 'fix_requires_engine_change',
                                                 'next_step')} if diagnostic.get('verdict') else None,
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
staged.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'exit_code': rc, 'failed_steps': failed_steps}))
PY
  if [[ ! -s "$STAGE/smoke-result.json" ]]; then   # the writer itself failed: still leave a marker
    printf '{"schema": "lilith/c6-smoke-result@1", "run_id": "%s", "status": "failed", "exit_code": %s, "error": "result writer failed"}\n' \
      "$RUN_ID" "$rc" > "$STAGE/smoke-result.json"
  fi
  echo "$MARKER finished rc=$rc; packaging $RUN/results-$RUN_ID.tar.gz"
  rm -f "$RUN/smoke.pid"
  # Close the log so its last line is on disk before it is hashed and packed.
  exec 1>&3 2>&4
  local waited=0
  while kill -0 "$TEE_PID" 2>/dev/null && (( waited < 150 )); do sleep 0.2; waited=$(( waited + 1 )); done
  # SHA256SUMS covers every run file plus the staged result under its final name.
  { (cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum)
    (cd "$STAGE" && sha256sum ./smoke-result.json); } > "$STAGE/SHA256SUMS"
  mv "$STAGE/SHA256SUMS" "$RUN/SHA256SUMS"
  # Pack from outside the run dir (no "file changed as we read it"); the staged
  # result is stored as <run-id>/smoke-result.json.
  local rename=(--transform "s,^\.stage-$RUN_ID/,$RUN_ID/,")
  [[ "$(tar --version 2>/dev/null)" == *"GNU tar"* ]] || rename=(-s ",^\.stage-$RUN_ID/,$RUN_ID/,")
  if tar -C "$ROOT/runs" -czf "$STAGE/results.tar.gz" "${rename[@]}" \
       --exclude='*/adapter_model.safetensors' "$RUN_ID" ".stage-$RUN_ID/smoke-result.json"; then
    mv "$STAGE/results.tar.gz" "$RUN/results-$RUN_ID.tar.gz"
  else
    rm -f "$STAGE/results.tar.gz"
    echo "$MARKER packaging failed; the run directory is the evidence" >&2
  fi
  # Completion marker: the result is renamed into place last (same filesystem: atomic).
  mv "$STAGE/smoke-result.json" "$RUN/smoke-result.json"
  rmdir "$STAGE" 2>/dev/null
  printf '%s results=%s rc=%s\n' "$MARKER" "$RUN/results-$RUN_ID.tar.gz" "$rc"
}
trap finish EXIT

remaining() { echo $(( BUDGET - ($(date +%s) - START) )); }

step_status() {  # last recorded status of a step, or empty
  awk -F'\t' -v name="$1" '$1 == name { status = $2 } END { print status }' "$STEPS"
}

record_step() {  # record_step <name> <status> <exit_code> <seconds>
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$STEPS"
}

needs() {  # needs <step> <prerequisite...>: true when every prerequisite is ok, else records a skip
  local name="$1"; shift
  local prerequisite
  for prerequisite in "$@"; do
    if [[ "$(step_status "$prerequisite")" != ok ]]; then
      record_step "$name" skipped_dependency - 0
      echo "== step $name skipped: needs $prerequisite (status: $(step_status "$prerequisite"))"
      return 1
    fi
  done
}

# step [--cap SECONDS] <name> <command...>
# Records one steps.tsv row and returns the command's status; never exits, so the
# caller decides: `step ... || exit 1` for gates, `step ... || true` otherwise.
# External commands are bounded by min(remaining budget, cap).
step() {
  local cap=""
  if [[ "$1" == --cap ]]; then cap="$2"; shift 2; fi
  local name="$1"; shift
  local left; left="$(remaining)"
  if (( left < 120 )); then
    record_step "$name" budget_exhausted - 0
    echo "budget exhausted before step $name (remaining ${left}s)"; return 1
  fi
  local limit="$left"
  if [[ -n "$cap" ]] && (( cap < limit )); then limit="$cap"; fi
  echo "== step $name (remaining ${left}s, limit ${limit}s)"
  local t0; t0="$(date +%s)"
  local rc=0
  if declare -F "$1" >/dev/null; then
    # In-shell checks run with set -e suspended (|| context): every command in
    # them carries an explicit `|| return 1`, so nothing can pass by falling through.
    "$@" || rc=$?
  else
    timeout --signal=TERM --kill-after=60 "$limit" "$@" || rc=$?
  fi
  local status=ok
  if (( rc == 124 )); then status=timeout; elif (( rc != 0 )); then status=failed; fi
  record_step "$name" "$status" "$rc" "$(( $(date +%s) - t0 ))"
  if (( rc != 0 )); then echo "step $name $status (exit $rc)"; fi
  return "$rc"
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

NUMERICS_ARGS=()
[[ -z "${LILITH_NUMERICS_SDPA_BACKEND:-}" ]] || NUMERICS_ARGS+=(--numerics-sdpa-backend "$LILITH_NUMERICS_SDPA_BACKEND")
[[ "${LILITH_NUMERICS_SDPA_MATH_FALLBACK:-0}" != 1 ]] || NUMERICS_ARGS+=(--numerics-sdpa-math-fallback)
[[ -z "${LILITH_NUMERICS_TF32:-}" ]] || NUMERICS_ARGS+=(--numerics-tf32 "$LILITH_NUMERICS_TF32")
[[ -z "${LILITH_NUMERICS_BF16_REDUCTION:-}" ]] ||
  NUMERICS_ARGS+=(--numerics-bf16-reduced-precision-reduction "$LILITH_NUMERICS_BF16_REDUCTION")
[[ "${LILITH_NUMERICS_DETERMINISTIC:-0}" != 1 ]] || NUMERICS_ARGS+=(--numerics-deterministic)
DIAGNOSTIC_BUDGET="${LILITH_DIAGNOSTIC_BUDGET_SECONDS:-1800}"
[[ "$DIAGNOSTIC_BUDGET" =~ ^[0-9]+$ ]] ||
  die lilith::smoke::usage "LILITH_DIAGNOSTIC_BUDGET_SECONDS='$DIAGNOSTIC_BUDGET' is not a whole number" "unset it or pass seconds"
[[ -z "$ONLY" ]] || echo "$MARKER --only $ONLY (gates and data-pin always run)"

# Gates: nothing after them is meaningful without them.
step provenance provenance || exit 1
step bootstrap-check bash -c "bash '$SCRIPT_DIR/bootstrap.sh' --check > '$RUN/bootstrap-check.json'" || exit 1
step gpu gpu_identity || exit 1

step data-pin data_pin || true
if selected cache-control; then
  step cache-control "$PY" "$REPO/scripts/session-platform/lilith_training_cache_control.py" \
    --parent "$ROOT/parent" --output "$RUN/cache-control.json" --cache-offloading false \
    ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
fi
# The diagnostic follows a failed control; named alone in --only it runs unconditionally.
if selected cache-diagnostic && { [[ -n "$ONLY" ]] && ! selected cache-control ||
                                  [[ "$(step_status cache-control)" != ok ]]; }; then
  # The report carries a provisional verdict after every case, so even the cap leaves one.
  step --cap $(( DIAGNOSTIC_BUDGET + 300 )) cache-diagnostic \
    "$PY" "$REPO/scripts/session-platform/lilith_cache_path_diagnostic.py" \
    --parent "$ROOT/parent" --output "$RUN/cache-diagnostic.json" \
    --budget-seconds "$DIAGNOSTIC_BUDGET" || true
fi
if selected memory-probe; then
  step memory-probe "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode memory-probe \
    --parent "$ROOT/parent" --output "$RUN/memory-probe.json" \
    ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
fi
# Leave 15 minutes for generation measurement, cold reload and packaging.
MICRO_WALL=$(( $(remaining) - 900 ))
if selected microrun && needs microrun data-pin; then
  step microrun "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode microrun --parent "$ROOT/parent" \
    "${DATA_ARGS[@]}" --adapter-root "$RUN/adapters" --output "$RUN/microrun.json" \
    --max-wall-seconds "$MICRO_WALL" ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
fi
if selected cold-reload && needs cold-reload microrun; then
  step cold-reload "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode cold-reload --parent "$ROOT/parent" \
    --adapter-dir "$RUN/adapters/ctx-8192" --probe "$RUN/adapters/ctx-8192.probe.json" \
    --output "$RUN/cold-reload.json" ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
fi
if selected vllm-generation; then
  GENERATION_ADAPTER="${LILITH_GENERATION_ADAPTER_DIR:-}"
  if [[ -n "$GENERATION_ADAPTER" ]]; then
    if [[ -d "$GENERATION_ADAPTER" ]]; then
      step vllm-generation "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode generate --parent "$ROOT/parent" \
        --adapter-dir "$GENERATION_ADAPTER" --generation-backend vllm \
        --output "$RUN/vllm-generation.json" ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
    else
      record_step vllm-generation skipped_dependency - 0
      echo "== step vllm-generation skipped: LILITH_GENERATION_ADAPTER_DIR=$GENERATION_ADAPTER is not a directory"
    fi
  elif needs vllm-generation microrun; then
    step vllm-generation "$PY" "$SCRIPT_DIR/reference_microrun.py" --mode generate --parent "$ROOT/parent" \
      --adapter-dir "$RUN/adapters/ctx-8192" --generation-backend vllm \
      --output "$RUN/vllm-generation.json" ${NUMERICS_ARGS[@]+"${NUMERICS_ARGS[@]}"} || true
  fi
fi

FAILED_STEPS="$(awk -F'\t' 'NR > 1 && $2 != "ok" { printf "%s%s", sep, $1; sep = " " }' "$STEPS")"
if [[ -n "$FAILED_STEPS" ]]; then
  echo "$MARKER FAILED steps: $FAILED_STEPS"
  exit 1
fi
echo "$MARKER all steps passed"
