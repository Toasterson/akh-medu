#!/usr/bin/env bash
# Provision an exact-name disposable pgvector database and real MiniLM runtime,
# then run the ignored Phase 42c production-path acceptance test. Database
# creation, migration, queries, and cleanup are performed by the Rust/SeaORM
# test; this wrapper only owns the loopback container and downloaded files.
# The container must be supplied by immutable digest. Model, tokenizer, dataset,
# and ONNX Runtime inputs must have their corresponding HEKA_PHASE42C_*_SHA256
# values; downloads additionally require immutable 40-hex repository revisions.
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEKA_PHASE42C_TARGET_DIR="$REPO_ROOT/target"
IMAGE="${HEKA_PHASE42C_PGVECTOR_IMAGE:-}"
if [[ ! "$IMAGE" =~ ^[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[0-9a-f]{64}$ ]]; then
  echo "HEKA_PHASE42C_PGVECTOR_IMAGE must be an immutable image@sha256:digest reference" >&2
  exit 1
fi

sha256_file() {
  local path="$1"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$path" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$path" | awk '{print $1}'
  else
    echo "shasum or sha256sum is required for artifact verification" >&2
    return 1
  fi
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$label expected SHA-256 must be 64 lowercase hexadecimal characters" >&2
    return 1
  fi
  local actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch" >&2
    return 1
  fi
}
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
DB_ENV_FILE="$TEMP_ROOT/postgres.env"
PHASE42C_DB_PASSWORD=""

cleanup() {
  local exit_status="$?"
  local evidence_root=""
  unset HEKA_PHASE42C_ADMIN_DATABASE_URL
  PHASE42C_DB_PASSWORD=""
  unset PHASE42C_DB_PASSWORD
  if [[ "$exit_status" != 0 && "$CONTAINER_STARTED" == 1 ]]; then
    if evidence_root="$(mktemp -d "$TEMP_BASE/heka-phase42c-evidence.XXXXXX")"; then
      chmod 700 "$evidence_root"
      docker logs "$CONTAINER" >"$evidence_root/container.log" 2>&1 || true
      chmod 600 "$evidence_root/container.log" 2>/dev/null || true
      echo "preserved exact disposable-container logs at $evidence_root/container.log" >&2
    else
      echo "could not create private failure-evidence directory" >&2
    fi
  fi
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
  return "$exit_status"
}
trap cleanup EXIT

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required to generate disposable PostgreSQL credentials" >&2
  exit 1
fi
PHASE42C_DB_PASSWORD="$(openssl rand -hex 32)"
if [[ ! "$PHASE42C_DB_PASSWORD" =~ ^[0-9a-f]{64}$ ]]; then
  echo "failed to generate a safe disposable PostgreSQL password" >&2
  exit 1
fi
printf 'HEKA_PHASE42C_DB_PASSWORD=%s\n' "$PHASE42C_DB_PASSWORD" >"$DB_ENV_FILE"
chmod 600 "$DB_ENV_FILE"

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "refusing to reuse an existing disposable container name" >&2
  exit 1
fi
CONTAINER_STARTED=1
docker run --detach \
  --name "$CONTAINER" \
  --publish 127.0.0.1::5432 \
  --tmpfs /var/lib/postgresql/data:rw,uid=999,gid=999,mode=0700,size=1024m \
  --user 999:999 \
  --env PGDATA=/var/lib/postgresql/data/pgdata \
  --env-file "$DB_ENV_FILE" \
  --entrypoint /bin/sh \
  "$IMAGE" \
  -ceu 'umask 077; printf "%s\n" "$HEKA_PHASE42C_DB_PASSWORD" >/tmp/heka-postgres.pw; initdb -D "$PGDATA" -U heka --encoding=UTF8 --auth-local=scram-sha-256 --auth-host=scram-sha-256 --pwfile=/tmp/heka-postgres.pw >/tmp/initdb.log; rm -f /tmp/heka-postgres.pw; unset HEKA_PHASE42C_DB_PASSWORD; printf "%s\n" "host all all 0.0.0.0/0 scram-sha-256" "host all all ::/0 scram-sha-256" >>"$PGDATA/pg_hba.conf"; exec postgres -D "$PGDATA" -c listen_addresses="*"' \
  >/dev/null
rm -f -- "$DB_ENV_FILE"

ready=0
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" pg_isready -U heka -d postgres >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "disposable PostgreSQL did not become ready" >&2
  exit 1
fi

PORT_BINDING="$(docker port "$CONTAINER" 5432/tcp | head -n 1)"
PORT="${PORT_BINDING##*:}"
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "could not resolve the disposable loopback port" >&2
  exit 1
fi
if ! command -v nc >/dev/null 2>&1; then
  echo "nc is required for the bounded host-side PostgreSQL readiness probe" >&2
  exit 1
fi
host_ready=0
for _ in $(seq 1 60); do
  if nc -z -w 1 127.0.0.1 "$PORT" >/dev/null 2>&1; then
    host_ready=1
    break
  fi
  sleep 1
done
if [[ "$host_ready" != 1 ]]; then
  echo "disposable PostgreSQL did not accept TCP connections on its published loopback port" >&2
  exit 1
fi

MODEL_DIR="${HEKA_PHASE42C_MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  MODEL_REVISION="${HEKA_PHASE42C_MODEL_REVISION:-}"
  if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "HEKA_PHASE42C_MODEL_REVISION must be an immutable 40-hex commit when downloading the model" >&2
    exit 1
  fi
  MODEL_DIR="$TEMP_ROOT/models/all-MiniLM-L6-v2"
  mkdir -p "$MODEL_DIR"
  curl --fail --location --retry 3 \
    --max-filesize 134217728 \
    --output "$MODEL_DIR/model.onnx" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$MODEL_REVISION/onnx/model.onnx"
  curl --fail --location --retry 3 \
    --max-filesize 8388608 \
    --output "$MODEL_DIR/tokenizer.json" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$MODEL_REVISION/tokenizer.json"
fi
if [[ ! -f "$MODEL_DIR/model.onnx" || ! -f "$MODEL_DIR/tokenizer.json" ]]; then
  echo "MiniLM model.onnx and tokenizer.json are required" >&2
  exit 1
fi
verify_sha256 "$MODEL_DIR/model.onnx" "${HEKA_PHASE42C_MODEL_ONNX_SHA256:-}" "MiniLM model.onnx"
verify_sha256 "$MODEL_DIR/tokenizer.json" "${HEKA_PHASE42C_TOKENIZER_SHA256:-}" "MiniLM tokenizer.json"

DATASET="${HEKA_PHASE42C_LONGMEMEVAL_DATASET:-}"
if [[ -z "$DATASET" ]]; then
  DATASET_REVISION="${HEKA_PHASE42C_DATASET_REVISION:-}"
  if [[ ! "$DATASET_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "HEKA_PHASE42C_DATASET_REVISION must be an immutable 40-hex commit when downloading the dataset" >&2
    exit 1
  fi
  DATASET="$TEMP_ROOT/longmemeval_s_cleaned.json"
  curl --fail --location --retry 3 \
    --max-filesize 536870912 \
    --output "$DATASET" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/$DATASET_REVISION/longmemeval_s_cleaned.json"
fi
if [[ ! -f "$DATASET" ]]; then
  echo "official LongMemEval cleaned small dataset is required" >&2
  exit 1
fi
verify_sha256 "$DATASET" "${HEKA_PHASE42C_DATASET_SHA256:-}" "LongMemEval cleaned small dataset"

ORT_DOWNLOADED=0
if [[ -z "${ORT_DYLIB_PATH:-}" ]]; then
  ORT_DOWNLOADED=1
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
  verify_sha256 "$ORT_ARCHIVE" "${HEKA_PHASE42C_ORT_ARCHIVE_SHA256:-}" "ONNX Runtime archive"
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
if [[ -n "${HEKA_PHASE42C_ORT_DYLIB_SHA256:-}" ]]; then
  verify_sha256 "$ORT_DYLIB_PATH" "$HEKA_PHASE42C_ORT_DYLIB_SHA256" "ONNX Runtime dynamic library"
elif [[ "$ORT_DOWNLOADED" != 1 ]]; then
  echo "HEKA_PHASE42C_ORT_DYLIB_SHA256 is required for a caller-supplied ONNX Runtime library" >&2
  exit 1
fi

export ORT_DYLIB_PATH
export HEKA_PHASE42C_LIVE=1
export HEKA_PHASE42C_RESOURCE_ID="$RESOURCE_ID"
export HEKA_PHASE42C_ADMIN_DATABASE_URL="postgres://heka:${PHASE42C_DB_PASSWORD}@127.0.0.1:${PORT}/postgres?sslmode=disable"
export HEKA_PHASE42C_MODEL_DIR="$MODEL_DIR"
export HEKA_PHASE42C_LONGMEMEVAL_DATASET="$DATASET"

CARGO_TARGET_DIR="$HEKA_PHASE42C_TARGET_DIR" cargo build \
  --manifest-path "$REPO_ROOT/Cargo.toml" \
  -p heka --bin hekad --bin heka-capture
HEKAD_BIN="$HEKA_PHASE42C_TARGET_DIR/debug/hekad"
HEKA_CAPTURE_BIN="$HEKA_PHASE42C_TARGET_DIR/debug/heka-capture"
if [[ ! -x "$HEKAD_BIN" ]]; then
  echo "workspace hekad acceptance binary is unavailable" >&2
  exit 1
fi
if [[ ! -x "$HEKA_CAPTURE_BIN" ]]; then
  echo "workspace heka-capture acceptance binary is unavailable" >&2
  exit 1
fi
export HEKA_PHASE42C_HEKAD_BIN="$HEKAD_BIN"
export HEKA_PHASE42C_HEKA_CAPTURE_BIN="$HEKA_CAPTURE_BIN"

CARGO_TARGET_DIR="$HEKA_PHASE42C_TARGET_DIR" cargo test \
  --manifest-path "$REPO_ROOT/Cargo.toml" \
  -p heka --lib \
  eval::longmemeval::live_tests::phase42c_live_acceptance \
  -- --ignored --exact --nocapture
