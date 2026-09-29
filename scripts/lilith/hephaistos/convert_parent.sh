#!/usr/bin/env bash
# Convert Lilith's pinned parent to a Q8_0 GGUF for llama-server and publish it on the runs share
# (ADR 121). Idempotent and resumable.
#
#   bash scripts/lilith/hephaistos/convert_parent.sh
#   bash scripts/lilith/hephaistos/convert_parent.sh --force          # reconvert even if a valid GGUF exists
#   bash scripts/lilith/hephaistos/convert_parent.sh --no-publish     # do not write the result to the share
#   bash scripts/lilith/hephaistos/convert_parent.sh --keep-stage     # keep the 24 GB staged parent
#
# What it does, in order (each step is skipped when its result already verifies):
#   1. A valid converted model is already installed under $STATE/models/<name>: done.
#   2. A valid one is on the share ($NFS/hephaistos-20260927/models/<name>): copy it, verify its
#      sha256 against its PROVENANCE.json, install it. No conversion.
#   3. Otherwise convert: verify the parent on the share against the pins in pins.json (sha256 of
#      all seven files, size first), stage it locally, convert with the pinned llama.cpp
#      converter to Q8_0, write PROVENANCE.json (source pins, converter commit, output sha256),
#      publish to the share (GGUF first, PROVENANCE.json last: it is the commit marker), read the
#      published GGUF back and verify it, install it locally.
# Refusals (fail closed): any parent file that differs from its pin (never converted), a
# provenance that names another parent revision or converter, a hash mismatch on any copy.
set -euo pipefail
umask 022
# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

FORCE=0
PUBLISH=1
KEEP_STAGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --no-publish) PUBLISH=0; shift ;;
    --keep-stage) KEEP_STAGE=1; shift ;;
    -h | --help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die lilith::heph::usage "unknown argument '$1'" "see --help" ;;
  esac
done

REVISION="$(pin parent.revision)"
QUANT="$(pin gguf.quant)"
GGUF_NAME="$(pin gguf.filename)"
COMMIT="$(pin llama_cpp.commit)"
NAME="lilith-parent-${REVISION:0:8}-${QUANT}-llamacpp-${COMMIT:0:12}"
LOCAL_OUT="$STATE/models/$NAME"
STAGE_DIR="$WORK/parent-${REVISION:0:8}"
OUT_WORK="$WORK/out-$NAME"
SRC="$WORK/llama.cpp"
VENV="$WORK/convert-venv"

python3 - <<'PY' || die lilith::heph::python "python3 is older than 3.10" "sudo apt-get install -y python3 python3-venv"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

mkdir -p "$WORK"
RUNS="$(nfs_root)"
NFS_PARENT="$RUNS/$(pin parent.nfs_relative_dir)"
NFS_OUT="$RUNS/$(pin gguf.nfs_relative_parent)/$NAME"

install_local() { # install_local DIR: DIR holds the GGUF, PROVENANCE.json and chat template; verify then install
  local dir="$1"
  "${TOOLS[@]}" check-model "$dir" >/dev/null
  as_root install -d -m 755 "$STATE" "$STATE/models" "$LOCAL_OUT"
  if [[ "$dir" != "$LOCAL_OUT" ]]; then
    local file
    for file in "$GGUF_NAME" chat_template.jinja PROVENANCE.json; do
      as_root install -m 644 "$dir/$file" "$LOCAL_OUT/$file.partial"
      as_root mv -f "$LOCAL_OUT/$file.partial" "$LOCAL_OUT/$file"
    done
  fi
  "${TOOLS[@]}" check-model "$LOCAL_OUT" >/dev/null
  as_root ln -sfn "$LOCAL_OUT" "$STATE/models/current"
  log "installed $LOCAL_OUT; $STATE/models/current -> $NAME"
}

# 1. already installed?
if [[ "$FORCE" == 0 ]] && "${TOOLS[@]}" check-model "$LOCAL_OUT" >/dev/null 2>&1; then
  as_root ln -sfn "$LOCAL_OUT" "$STATE/models/current"
  log "$NAME is already installed and verified; nothing to do"
  exit 0
fi

# 2. already converted on the share?
if [[ "$FORCE" == 0 ]] && nfs_run test -f "$NFS_OUT/PROVENANCE.json"; then
  log "found a converted model on the share: $NFS_OUT"
  need_free_gib "$WORK" 20 "copying the converted model"
  REUSE="$WORK/reuse-$NAME"
  rm -rf "$REUSE"
  mkdir -p "$REUSE"
  nfs_run cat "$NFS_OUT/PROVENANCE.json" > "$REUSE/PROVENANCE.json"
  read -r GGUF_SHA GGUF_SIZE < <(python3 - "$REUSE/PROVENANCE.json" <<'PY'
import json, sys
out = json.load(open(sys.argv[1]))["output"]
print(out["sha256"], out["size"])
PY
  )
  read -r TEMPLATE_SHA TEMPLATE_SIZE < <("${TOOLS[@]}" pins file chat_template.jinja)
  if nfs_run cat "$NFS_OUT/chat_template.jinja" | "${TOOLS[@]}" stage --dst "$REUSE/chat_template.jinja" \
      --sha256 "$TEMPLATE_SHA" --size "$TEMPLATE_SIZE" >/dev/null \
    && nfs_run cat "$NFS_OUT/$GGUF_NAME" | "${TOOLS[@]}" stage --dst "$REUSE/$GGUF_NAME" \
      --sha256 "$GGUF_SHA" --size "$GGUF_SIZE" >/dev/null \
    && install_local "$REUSE"; then
    rm -rf "$REUSE"
    exit 0
  fi
  rm -rf "$REUSE"
  log "the model on the share failed verification; converting from the parent instead (it is replaced on publish)"
fi

# 3. convert
need_free_gib "$WORK" 50 "staging the parent (24 GB), the converted model (13 GB) and the conversion env"
nfs_run test -d "$NFS_PARENT" || die lilith::heph::parent_missing "$NFS_PARENT is not readable" \
  "check the runs share is mounted ($NFS_MOUNT) and that hephaistos-20260927/ is on it; access is as uid $NFS_UID"

log "staging the pinned parent (google/gemma-4-12B-it@${REVISION:0:8}) into $STAGE_DIR"
mkdir -p "$STAGE_DIR"
for file in $(python3 -c 'import json,sys;print(" ".join(sorted(json.load(open(sys.argv[1]))["parent"]["files"])))' "$PINS"); do
  if "${TOOLS[@]}" verify-dir "$STAGE_DIR" --files "$file" >/dev/null 2>&1; then
    log "  $file: staged and verified"
    continue
  fi
  read -r sha size < <("${TOOLS[@]}" pins file "$file")
  log "  $file: copying $size bytes from the share (verified while copying)"
  nfs_run cat "$NFS_PARENT/$file" | "${TOOLS[@]}" stage --dst "$STAGE_DIR/$file" --sha256 "$sha" --size "$size" >/dev/null
done
# Everything pinned is present; one final full check so a receipt exists for the provenance.
"${TOOLS[@]}" verify-dir "$STAGE_DIR" >/dev/null

ensure_llama_source "$SRC"
REQUIREMENTS="$SRC/requirements/requirements-convert_hf_to_gguf.txt"
[[ -f "$REQUIREMENTS" ]] || die lilith::heph::source "$REQUIREMENTS is missing from the pinned llama.cpp" \
  "delete $SRC and rerun"
PYTHON="${LILITH_CONVERT_PYTHON:-python3}"
if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating the conversion venv $VENV ($($PYTHON --version 2>&1))"
  "$PYTHON" -m venv "$VENV" || die lilith::heph::venv "could not create a venv with $PYTHON" \
    "sudo apt-get install -y python3-venv, or set LILITH_CONVERT_PYTHON to a python >= 3.10 with venv"
fi
REQ_SHA="$(python3 "$HEPH_DIR/heph_tools.py" sha256 "$REQUIREMENTS")"
TRANSFORMERS="$(pin llama_cpp.converter_transformers)"
if [[ ! -f "$VENV/.requirements-$REQ_SHA-transformers-$TRANSFORMERS" ]]; then
  log "installing the converter's requirements (CPU torch; this is the largest download)"
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q -r "$REQUIREMENTS" \
    || die lilith::heph::pip "pip could not install $REQUIREMENTS" \
         "check network access to pypi.org and download.pytorch.org; a python older than 3.10 cannot"
  # llama.cpp pins transformers 4.57.6, which cannot read the parent's tokenizer_config.json
  # (pins.json llama_cpp.converter_transformers_reason). Replace it with the verified version.
  "$VENV/bin/python" -m pip install --disable-pip-version-check -q "transformers==$TRANSFORMERS" \
    || die lilith::heph::pip "pip could not install transformers==$TRANSFORMERS" "check network access to pypi.org"
  touch "$VENV/.requirements-$REQ_SHA-transformers-$TRANSFORMERS"
fi
PIP_FREEZE_SHA="$("$VENV/bin/python" -m pip freeze | "${TOOLS[@]}" sha256 -)"

rm -rf "$OUT_WORK"
mkdir -p "$OUT_WORK"
OUT_GGUF="$OUT_WORK/$GGUF_NAME"
CONVERT_COMMAND="python convert_hf_to_gguf.py <parent> --outtype $QUANT --model-name lilith-parent --outfile $GGUF_NAME"
log "converting to $QUANT (10-30 minutes; CPU only)"
"$VENV/bin/python" "$SRC/convert_hf_to_gguf.py" "$STAGE_DIR" --outtype "$QUANT" --model-name lilith-parent \
  --outfile "$OUT_GGUF" > "$WORK/convert-$NAME.log" 2>&1 || die lilith::heph::convert \
  "convert_hf_to_gguf.py failed" "see $WORK/convert-$NAME.log (tail -n 60); the pinned converter registers Gemma4UnifiedForConditionalGeneration"

[[ "$(head -c 4 "$OUT_GGUF")" == GGUF ]] || die lilith::heph::convert "$OUT_GGUF is not a GGUF file" \
  "see $WORK/convert-$NAME.log"
python3 - "$OUT_GGUF" <<'PY' || die lilith::heph::convert "the converted file has an implausible size for a Q8_0 12B model" "see $WORK/convert-$NAME.log; expected 11-15 GB"
import os, sys
size = os.path.getsize(sys.argv[1])
sys.exit(0 if 11e9 <= size <= 15e9 else 1)
PY
cp "$STAGE_DIR/chat_template.jinja" "$OUT_WORK/chat_template.jinja"
"${TOOLS[@]}" provenance --gguf "$OUT_GGUF" --template "$OUT_WORK/chat_template.jinja" \
  --out "$OUT_WORK/PROVENANCE.json" --parent-dir "$STAGE_DIR" --llama-commit "$(git -C "$SRC" rev-parse HEAD)" \
  --convert-command "$CONVERT_COMMAND" --pip-freeze-sha256 "$PIP_FREEZE_SHA" >/dev/null
"${TOOLS[@]}" check-model "$OUT_WORK" >/dev/null
log "converted and verified: $(python3 -c 'import json,sys;o=json.load(open(sys.argv[1]))["output"];print(o["size"],"bytes, sha256",o["sha256"])' "$OUT_WORK/PROVENANCE.json")"

if [[ "$PUBLISH" == 1 ]]; then
  log "publishing to $NFS_OUT (the share writes at about 11 MB/s: ~20 minutes for 13 GB)"
  # An older, invalid PROVENANCE.json must not survive an interrupted publish: remove it first.
  nfs_run rm -f "$NFS_OUT/PROVENANCE.json" "$NFS_OUT/SHA256SUMS"
  nfs_write_file "$OUT_GGUF" "$NFS_OUT/$GGUF_NAME"
  nfs_write_file "$OUT_WORK/chat_template.jinja" "$NFS_OUT/chat_template.jinja"
  PUBLISHED_SHA="$(nfs_run cat "$NFS_OUT/$GGUF_NAME" | "${TOOLS[@]}" sha256 -)"
  WANT_SHA="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["output"]["sha256"])' "$OUT_WORK/PROVENANCE.json")"
  [[ "$PUBLISHED_SHA" == "$WANT_SHA" ]] || die lilith::heph::publish \
    "the GGUF read back from the share has sha256 $PUBLISHED_SHA, expected $WANT_SHA" \
    "rerun with --force; PROVENANCE.json was not published, so the bad copy is not trusted"
  ( cd "$OUT_WORK" && python3 - "$GGUF_NAME" chat_template.jinja <<'PY' ) > "$OUT_WORK/SHA256SUMS"
import hashlib, sys
for name in sys.argv[1:]:
    digest = hashlib.sha256()
    with open(name, "rb") as handle:
        while chunk := handle.read(1 << 23):
            digest.update(chunk)
    print(f"{digest.hexdigest()}  {name}")
PY
  nfs_write_file "$OUT_WORK/SHA256SUMS" "$NFS_OUT/SHA256SUMS"
  nfs_write_file "$OUT_WORK/PROVENANCE.json" "$NFS_OUT/PROVENANCE.json"
  log "published $NFS_OUT (GGUF read back and verified)"
fi

install_local "$OUT_WORK"
rm -rf "$OUT_WORK"
if [[ "$KEEP_STAGE" == 0 ]]; then
  rm -rf "$STAGE_DIR"
  log "removed the staged parent (the share is the source of truth)"
fi
