#!/usr/bin/env bash
# Lilith C6 H100 smoke (Anima WI 01a0e270-6021): one-shot, fail-closed, <= 2 GPU-hours.
#
#   bash scripts/lilith/runpod/run_smoke.sh <run-id>
#
# Sequence (each step bounded by the remaining budget):
#   provenance -> bootstrap --check -> GPU identity -> data pin
#   -> lilith_training_cache_control.py on CUDA with resident KV
#   -> memory probe 8K/16K/32K -> reference micro-run 8K/16K (+ generation tok/s)
#   -> cold reload of the 8K adapter in a fresh process
#   -> SHA256SUMS + results tarball under /workspace/lilith/runs/<run-id>/
#
# Refuses an existing run directory (one-shot; resuming is a new run id). The EXIT
# trap always writes smoke-result.json and the tarball, including on failure.
# Environment knobs (all recorded in smoke-result.json):
#   LILITH_ROOT (default /workspace/lilith), LILITH_SMOKE_BUDGET_SECONDS (7200),
#   LILITH_SMOKE_SYNTHETIC=1 (no private subset: recorded deviation),
#   LILITH_SMOKE_ALLOW_GPU=1 (non-H100), LILITH_ALLOW_DIRTY=1 (dirty checkout),
#   LILITH_ALLOW_UNMERGED=1 (HEAD not on origin/main; e.g. validating this PR before merge).
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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
exec > >(tee -a "$RUN/smoke.log") 2>&1
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
result = {'schema': 'lilith/c6-smoke-result@1', 'run_id': run_id,
          'status': 'passed' if rc == 0 and all(s['status'] == 'ok' for s in steps) else 'failed',
          'exit_code': rc, 'steps': steps, 'wall_clock_seconds': time.time() - start,
          'budget_seconds': budget, 'quality_claim': False, 'release_selected': False,
          'knobs': {k: os.environ.get(k) for k in ('LILITH_SMOKE_SYNTHETIC', 'LILITH_SMOKE_ALLOW_GPU',
                                                   'LILITH_ALLOW_DIRTY', 'LILITH_ALLOW_UNMERGED',
                                                   'LILITH_SMOKE_BUDGET_SECONDS')},
          'provenance': load('provenance.json'), 'cache_control_status': (load('cache-control.json') or {}).get('status'),
          'memory_probe': {k: {f: v.get(f) for f in ('status', 'peak_reserved_bytes', 'headroom_fraction')}
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
  (cd "$RUN" && find . -type f ! -name 'results-*.tar.gz' ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum > SHA256SUMS)
  tar -C "$ROOT/runs" -czf "$RUN/results-$RUN_ID.tar.gz" \
    --exclude="$RUN_ID/results-$RUN_ID.tar.gz" --exclude='*/adapter_model.safetensors' "$RUN_ID"
  echo "$MARKER finished rc=$rc results=$RUN/results-$RUN_ID.tar.gz"
  rm -f "$RUN/smoke.pid"
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
    "$@" || rc=$?          # short in-shell checks (provenance, GPU identity, data pin)
  else
    timeout --signal=TERM --kill-after=60 "$left" "$@" || rc=$?
  fi
  local status=ok; (( rc == 0 )) || status=failed; (( rc == 124 )) && status=timeout
  printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$(( $(date +%s) - t0 ))" >> "$STEPS"
  (( rc == 0 )) || { echo "step $name $status (exit $rc)"; exit 1; }
}

provenance() {
  local sha dirty ancestor
  sha="$(git -C "$REPO" rev-parse HEAD)"
  dirty="$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
  if [[ "$dirty" != 0 && "${LILITH_ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "checkout $REPO is dirty ($dirty files); smoke code must be a clean origin/main commit" >&2
    return 1
  fi
  if git -C "$REPO" rev-parse --verify -q origin/main >/dev/null; then
    if git -C "$REPO" merge-base --is-ancestor "$sha" origin/main; then ancestor=true; else ancestor=false; fi
    if [[ "$ancestor" == false && "${LILITH_ALLOW_UNMERGED:-0}" != 1 ]]; then
      echo "HEAD $sha is not on origin/main; pod code runs from main SHAs (ADR 116 provenance)" >&2
      return 1
    fi
  else
    ancestor=unverifiable
  fi
  "$PY" - "$RUN/provenance.json" "$sha" "$dirty" "$ancestor" "$SCRIPT_DIR" "$REPO" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, sha, dirty, ancestor, scripts, repo = sys.argv[1:7]
names = ['bootstrap.sh', 'run_smoke.sh', 'lilith_lora_trainer.py', 'reference_microrun.py', 'requirements-cuda.txt']
sources = {name: hashlib.sha256((Path(scripts) / name).read_bytes()).hexdigest() for name in names}
for name in ('lilith_training_engine.py', 'lilith_training_cache_control.py', 'lilith_rl.py'):
    sources[name] = hashlib.sha256((Path(repo) / 'scripts/session-platform' / name).read_bytes()).hexdigest()
Path(out).write_text(json.dumps({'git_sha': sha, 'dirty_tracked_files': int(dirty),
                                 'ancestor_of_origin_main': ancestor, 'sources_sha256': sources}, indent=2) + '\n')
PY
}

gpu_identity() {
  nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader | tee "$RUN/gpu.csv"
  if ! grep -q 'H100' "$RUN/gpu.csv" && [[ "${LILITH_SMOKE_ALLOW_GPU:-0}" != 1 ]]; then
    echo "not an H100; the C6 reference is defined on 1x H100 80GB (LILITH_SMOKE_ALLOW_GPU=1 to override)" >&2
    return 1
  fi
}

data_pin() {
  if [[ "${LILITH_SMOKE_SYNTHETIC:-0}" == 1 ]]; then
    echo "synthetic data requested: recorded deviation, loss curve not comparable"; return 0
  fi
  "$PY" - "$DATA" "$REPO/training/lilith/transcripts-20260922/sft-subset-v1-manifest.json" <<'PY'
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
