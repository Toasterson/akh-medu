#!/usr/bin/env bash
# Idempotent Lilith Runpod pod bootstrap (Anima WI 01a0e270-6021, slice C6).
#
#   bash scripts/lilith/runpod/bootstrap.sh            # install / verify, idempotent
#   bash scripts/lilith/runpod/bootstrap.sh --check    # verify only, JSON summary on stdout
#
# Layout (everything on the network volume; a pod stop wipes /root):
#   /workspace/lilith/{parent,gguf,env,data,bin,adapters,runs}
#   env/versions.json      exact package versions + lock sha256
#   parent/receipt.json    pinned google/gemma-4-12B-it revision + verified sha256 pins
#   paths.json             lilith/paths@1 for the runtime-training config/stack/worker
#
# HF_TOKEN is read from the environment for the gated parent download only. It is
# never written to disk, echoed, or passed on the command line.
#
# Refusals (fail closed): pin mismatch, an env built from a different lock, an
# unreceipted parent directory, no CUDA device (unless LILITH_ALLOW_NO_CUDA=1),
# a root outside /workspace (unless LILITH_ALLOW_NON_WORKSPACE=1).
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ROOT="${LILITH_ROOT:-/workspace/lilith}"
UV_VERSION="0.12.8"
PYTHON_VERSION="3.12"
LOCK="$SCRIPT_DIR/requirements-cuda.txt"
PARENT_REPO="google/gemma-4-12B-it"
PARENT_REVISION="5926caa4ec0cac5cbfadaf4077420520de1d5205"
WEIGHTS_SHA256="5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
CONFIG_SHA256="478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9"
TOKENIZER_SHA256="cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
SUBSET_MANIFEST="$REPO/training/lilith/transcripts-20260922/sft-subset-v1-manifest.json"

die() {
  printf 'bootstrap: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}
log() { printf 'bootstrap: %s\n' "$*" >&2; }

MODE=install
case "${1:-}" in
  "" | --install) ;;
  --check) MODE=check ;;
  *) die lilith::bootstrap::usage "unknown argument '$1'" "use no argument (install) or --check" ;;
esac

if [[ "$ROOT" != /workspace/* && "${LILITH_ALLOW_NON_WORKSPACE:-0}" != 1 ]]; then
  die lilith::bootstrap::root "root $ROOT is not on the /workspace volume" \
    "a pod stop wipes everything else; set LILITH_ROOT under /workspace"
fi

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
LOCK_SHA256="$(sha256_of "$LOCK")"
ENV_PY="$ROOT/env/bin/python"
export UV_PYTHON_INSTALL_DIR="$ROOT/bin/python"
export UV_NO_MODIFY_PATH=1
export HF_HOME="$ROOT/.hf-home"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------- verification
verify_parent() {  # $1 = parent dir; prints the receipt artifacts JSON on success
  "$ENV_PY" - "$1" "$WEIGHTS_SHA256" "$CONFIG_SHA256" "$TOKENIZER_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path
parent, weights, config, tokenizer = Path(sys.argv[1]), *sys.argv[2:5]
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
artifacts = []
for path in sorted(parent.rglob('*')):
    relative = path.relative_to(parent)
    if not path.is_file() or relative.parts[0] == '.cache' or relative.name == 'receipt.json':
        continue
    artifacts.append({'name': str(relative), 'bytes': path.stat().st_size, 'sha256': sha(path)})
observed = {a['name']: a['sha256'] for a in artifacts}
pins = {'model.safetensors': weights, 'config.json': config, 'tokenizer.json': tokenizer}
bad = {name: observed.get(name) for name, pin in pins.items() if observed.get(name) != pin}
if bad:
    print(json.dumps({'status': 'pin_mismatch', 'observed': bad, 'expected': pins}), file=sys.stderr)
    sys.exit(3)
print(json.dumps({'artifacts': artifacts, 'pins': pins}))
PY
}

check_env() {
  [[ -x "$ENV_PY" ]] || die lilith::bootstrap::env_missing "$ENV_PY does not exist" "run bootstrap.sh without --check"
  [[ -f "$ROOT/env/.lilith-lock-sha256" && "$(cat "$ROOT/env/.lilith-lock-sha256")" == "$LOCK_SHA256" ]] ||
    die lilith::bootstrap::env_lock "env was not built from $LOCK (sha $LOCK_SHA256)" \
      "move it aside (mv $ROOT/env $ROOT/env.old-\$(date +%s)) and rerun bootstrap.sh"
  [[ -f "$ROOT/env/versions.json" ]] || die lilith::bootstrap::env_versions "env/versions.json missing" \
    "the env build did not finish; move it aside and rerun"
}

write_versions() {
  LOCK_SHA256="$LOCK_SHA256" LOCK_PATH="$LOCK" "$ENV_PY" - "$ROOT/env/versions.json" <<'PY'
import importlib.metadata as metadata, json, os, platform, subprocess, sys
packages = ['torch', 'transformers', 'peft', 'trl', 'bitsandbytes', 'vllm', 'accelerate', 'safetensors',
            'tokenizers', 'huggingface_hub', 'jsonschema', 'triton', 'flashinfer-python', 'nvidia-cuda-runtime']
versions = {}
for name in packages:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
import torch
record = {'schema': 'lilith/env-versions@1', 'python': sys.version, 'platform': platform.platform(),
          'packages': versions, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
          'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'lock_sha256': os.environ['LOCK_SHA256'], 'lock': os.environ['LOCK_PATH']}
try:
    record['nvidia_smi'] = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
        check=True, capture_output=True, text=True).stdout.strip()
except (OSError, subprocess.CalledProcessError) as error:
    record['nvidia_smi'] = 'unavailable: ' + type(error).__name__
with open(sys.argv[1], 'x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps({'cuda_available': record['cuda_available'], 'torch_cuda': record['torch_cuda']}))
PY
}

require_cuda() {
  if [[ "${LILITH_ALLOW_NO_CUDA:-0}" == 1 ]]; then return; fi
  "$ENV_PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' ||
    die lilith::bootstrap::no_cuda "torch $("$ENV_PY" -c 'import torch; print(torch.__version__)') sees no CUDA device" \
      "the lock uses CUDA 13 wheels (driver >= 580); pick a pod host with a newer driver"
}

# --------------------------------------------------------------------- check
if [[ "$MODE" == check ]]; then
  check_env
  require_cuda
  [[ -f "$ROOT/parent/receipt.json" ]] || die lilith::bootstrap::parent_missing "no parent receipt" \
    "run bootstrap.sh with HF_TOKEN set to download the pinned parent"
  verify_parent "$ROOT/parent" >/dev/null ||
    die lilith::bootstrap::parent_pin "parent files differ from the pins" "never train on a drifted parent; re-download"
  "$ENV_PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(json.dumps({'status': 'ok', 'root': str(root),
                  'versions': json.loads((root / 'env/versions.json').read_text())['packages'],
                  'parent_receipt': json.loads((root / 'parent/receipt.json').read_text())['revision']}))
PY
  exit 0
fi

# ------------------------------------------------------------------- install
mkdir -p "$ROOT"/{gguf,data,bin,adapters,runs,private}
chmod 700 "$ROOT/private"

# 1. uv (pinned) into the volume.
UV="$ROOT/bin/uv"
if [[ ! -x "$UV" ]] || ! "$UV" --version | grep -q " $UV_VERSION"; then
  log "installing uv $UV_VERSION into $ROOT/bin"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$ROOT/bin" sh >&2
  "$UV" --version | grep -q " $UV_VERSION" || die lilith::bootstrap::uv "uv $UV_VERSION did not install" \
    "check network access to astral.sh"
fi

# 2. Pinned CUDA venv; never mutated in place once built.
if [[ -d "$ROOT/env" ]]; then
  check_env
  log "env already built from lock $LOCK_SHA256; reusing"
else
  log "creating $ROOT/env (python $PYTHON_VERSION) from $LOCK"
  "$UV" venv --python "$PYTHON_VERSION" "$ROOT/env" >&2
  if ! "$UV" pip install --python "$ENV_PY" --require-hashes --no-cache -r "$LOCK" >&2; then
    rm -rf "$ROOT/env"
    die lilith::bootstrap::install "package installation from the lock failed" "inspect the log above; the partial env was removed"
  fi
  write_versions >&2
  printf '%s\n' "$LOCK_SHA256" > "$ROOT/env/.lilith-lock-sha256"
fi
require_cuda

# 3. Pinned parent.
if [[ -f "$ROOT/parent/receipt.json" ]]; then
  verify_parent "$ROOT/parent" >/dev/null ||
    die lilith::bootstrap::parent_pin "receipted parent no longer matches the pins" \
      "move $ROOT/parent aside and rerun to re-download; never train on it"
  log "parent verified against pins; reusing"
elif [[ -e "$ROOT/parent" ]]; then
  die lilith::bootstrap::parent_unknown "$ROOT/parent exists without a receipt" \
    "move it aside (its provenance is unknown) and rerun"
else
  [[ -n "${HF_TOKEN:-}" ]] || die lilith::bootstrap::hf_token "HF_TOKEN is not set" \
    "export HF_TOKEN for the gated $PARENT_REPO download (it is never written to disk)"
  PARTIAL="$ROOT/parent.partial"
  log "downloading $PARENT_REPO@$PARENT_REVISION into $PARTIAL"
  "$ENV_PY" - "$PARENT_REPO" "$PARENT_REVISION" "$PARTIAL" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, revision, target = sys.argv[1:4]
snapshot_download(repo_id=repo, revision=revision, local_dir=target, token=os.environ['HF_TOKEN'],
                  allow_patterns=['*.json', '*.jinja', '*.safetensors', '*.model', '*.txt'])
PY
  unset HF_TOKEN
  ARTIFACTS="$(verify_parent "$PARTIAL")" ||
    die lilith::bootstrap::parent_pin "downloaded parent does not match the pins" \
      "left in $PARTIAL for forensics; do not use it"
  mv "$PARTIAL" "$ROOT/parent"
  ARTIFACTS="$ARTIFACTS" "$ENV_PY" - "$ROOT/parent" "$PARENT_REPO" "$PARENT_REVISION" "$SUBSET_MANIFEST" <<'PY'
import datetime, hashlib, importlib.metadata, json, os, sys
from pathlib import Path
parent, repo, revision, manifest = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
verified = json.loads(os.environ['ARTIFACTS'])
template = hashlib.sha256((parent / 'chat_template.jinja').read_bytes()).hexdigest() \
    if (parent / 'chat_template.jinja').exists() else None
expected_template = json.loads(manifest.read_text())['chat_template_sha256'] if manifest.exists() else None
receipt = {'schema': 'lilith/parent-receipt@1', 'repo': repo, 'revision': revision, 'cache': str(parent),
           'artifacts': verified['artifacts'], 'pins': verified['pins'], 'pins_verified': True,
           'chat_template_sha256': template,
           'chat_template_matches_sft_subset_manifest': template == expected_template,
           'huggingface_hub': importlib.metadata.version('huggingface_hub'),
           'downloaded_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with open(parent / 'receipt.json', 'x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
PY
  log "parent verified and receipted"
fi

# 4. paths.json for the runtime-training config/stack/worker (written once).
if [[ ! -f "$ROOT/paths.json" ]]; then
  "$ENV_PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = {'schema': 'lilith/paths@1', 'parent_dir': str(root / 'parent'),
         'transcript_dataset': str(root / 'data/sft-subset-v1.jsonl'),
         'templates_dir': str(root / 'data/stack-templates'), 'venv_python': str(root / 'env/bin/python'),
         'runtime_bin_dir': str(root / 'bin'), 'reviewer_runtime_binary': str(root / 'bin/akhrtd'),
         'private_root': str(root / 'private'), 'output_root': str(root / 'runs'),
         'cache_offloading': False}
with open(root / 'paths.json', 'x') as stream:
    json.dump(paths, stream, indent=2)
    stream.write('\n')
PY
  log "wrote $ROOT/paths.json (H100: resident KV cache)"
fi
log "ready: $ROOT (lock $LOCK_SHA256, parent $PARENT_REVISION)"
