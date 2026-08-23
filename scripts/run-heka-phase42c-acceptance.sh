#!/usr/bin/env bash
# Provision an exact-name disposable pgvector database and real MiniLM runtime,
# then run the ignored Phase 42c production-path acceptance test. Database
# creation, migration, queries, and cleanup are performed by the Rust/SeaORM
# test; this wrapper only owns the loopback container and downloaded files.
set -euo pipefail

IMAGE="${HEKA_PHASE42C_PGVECTOR_IMAGE:-pgvector/pgvector:pg16}"
if command -v uuidgen >/dev/null 2>&1; then
  RESOURCE_SUFFIX="$(uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]')"
elif command -v openssl >/dev/null 2>&1; then
  RESOURCE_SUFFIX="$(openssl rand -hex 16)"
else
  echo "uuidgen or openssl is required for a unique resource name" >&2
  exit 1
fi
RESOURCE_SUFFIX="${RESOURCE_SUFFIX:0:16}"
if [[ ! "$RESOURCE_SUFFIX" =~ ^[0-9a-f]{16}$ ]]; then
  echo "failed to generate a safe disposable resource suffix" >&2
  exit 1
fi
RESOURCE_ID="heka_phase42c_${RESOURCE_SUFFIX}"
CONTAINER="heka-phase42c-${RESOURCE_SUFFIX}"
CONTAINER_STARTED=0
TEMP_BASE="${TMPDIR:-/tmp}"
TEMP_BASE="${TEMP_BASE%/}"
TEMP_ROOT="$(mktemp -d "$TEMP_BASE/heka-phase42c.XXXXXX")"

cleanup() {
  if [[ "$CONTAINER_STARTED" == 1 ]]; then
    case "$CONTAINER" in
      heka-phase42c-[0-9a-f]*) docker rm -f -- "$CONTAINER" >/dev/null 2>&1 || true ;;
      *) echo "refusing to clean unexpected container name" >&2 ;;
    esac
  fi
  case "$TEMP_ROOT" in
    "$TEMP_BASE"/heka-phase42c.*) rm -rf -- "$TEMP_ROOT" ;;
    *) echo "refusing to clean unexpected temporary path" >&2 ;;
  esac
}
trap cleanup EXIT

docker run --detach \
  --name "$CONTAINER" \
  --publish 127.0.0.1::5432 \
  --tmpfs /var/lib/postgresql/data:rw,uid=999,gid=999,mode=0700,size=1024m \
  --user 999:999 \
  --env PGDATA=/var/lib/postgresql/data/pgdata \
  --entrypoint /bin/sh \
  "$IMAGE" \
  -ceu 'initdb -D "$PGDATA" -U heka --encoding=UTF8 --auth-local=trust --auth-host=trust >/tmp/initdb.log; exec postgres -D "$PGDATA" -c listen_addresses="*"' \
  >/dev/null
CONTAINER_STARTED=1

ready=0
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" pg_isready -U heka -d postgres >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  docker logs --tail 80 "$CONTAINER" >&2
  echo "disposable PostgreSQL did not become ready" >&2
  exit 1
fi

PORT_BINDING="$(docker port "$CONTAINER" 5432/tcp | head -n 1)"
PORT="${PORT_BINDING##*:}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "could not resolve the disposable loopback port" >&2
  exit 1
fi

MODEL_DIR="${HEKA_PHASE42C_MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  MODEL_DIR="$TEMP_ROOT/model"
  mkdir -p "$MODEL_DIR"
  curl --fail --location --retry 3 \
    --max-filesize 134217728 \
    --output "$MODEL_DIR/model.onnx" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
  curl --fail --location --retry 3 \
    --max-filesize 8388608 \
    --output "$MODEL_DIR/tokenizer.json" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
fi
if [[ ! -f "$MODEL_DIR/model.onnx" || ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "MiniLM model.onnx and tokenizer.json are required" >&2
  exit 1
fi

DATASET="${HEKA_PHASE42C_LONGMEMEVAL_DATASET:-}"
if [[ -z "$DATASET" ]]; then
  DATASET="$TEMP_ROOT/longmemeval_oracle.json"
  curl --fail --location --retry 3 \
    --max-filesize 67108864 \
    --output "$DATASET" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
fi
if [[ ! -f "$DATASET" ]]; then
  echo "official LongMemEval oracle dataset is required" >&2
  exit 1
fi

if [[ -z "${ORT_DYLIB_PATH:-}" ]]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) ORT_ASSET="onnxruntime-osx-arm64-1.23.2.tgz" ;;
    Darwin-x86_64) ORT_ASSET="onnxruntime-osx-x86_64-1.23.2.tgz" ;;
    Linux-x86_64) ORT_ASSET="onnxruntime-linux-x64-1.23.2.tgz" ;;
    Linux-aarch64|Linux-arm64) ORT_ASSET="onnxruntime-linux-aarch64-1.23.2.tgz" ;;
    *) echo "unsupported platform for ONNX Runtime; set ORT_DYLIB_PATH" >&2; exit 1 ;;
  esac
  ORT_ARCHIVE="$TEMP_ROOT/$ORT_ASSET"
  curl --fail --location --retry 3 \
    --max-filesize 268435456 \
    --output "$ORT_ARCHIVE" \
    "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/$ORT_ASSET"
  mkdir -p "$TEMP_ROOT/onnxruntime"
  tar -xzf "$ORT_ARCHIVE" -C "$TEMP_ROOT/onnxruntime"
  while IFS= read -r candidate; do
    ORT_DYLIB_PATH="$candidate"
    break
  done < <(find "$TEMP_ROOT/onnxruntime" -type f \( -name 'libonnxruntime.so*' -o -name 'libonnxruntime*.dylib' \) | sort)
fi
if [[ -z "${ORT_DYLIB_PATH:-}" || ! -f "$ORT_DYLIB_PATH" ]]; then
  echo "ONNX Runtime dynamic library is unavailable" >&2
  exit 1
fi

export ORT_DYLIB_PATH
export HEKA_PHASE42C_LIVE=1
export HEKA_PHASE42C_RESOURCE_ID="$RESOURCE_ID"
export HEKA_PHASE42C_ADMIN_DATABASE_URL="postgres://heka@127.0.0.1:${PORT}/postgres?sslmode=disable"
export HEKA_PHASE42C_MODEL_DIR="$MODEL_DIR"
export HEKA_PHASE42C_LONGMEMEVAL_DATASET="$DATASET"

cargo test -p heka --lib \
  eval::longmemeval::live_tests::phase42c_live_acceptance \
  -- --ignored --exact --nocapture
