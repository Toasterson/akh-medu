#!/usr/bin/env bash
# Build the minimal Lilith pod payload from a clean origin/main commit (run on the Mac).
#
#   bash scripts/lilith/runpod/make_payload.sh [output-dir]     # default <repo>/target/lilith-payload
#
# The file list is verify_payload.PAYLOAD_FILES: it includes the program trainer
# (scripts/lilith/program/), the pilot's history/ guards, lilith_lora_trainer.py and the
# session-platform lilith_adapter_mechanics/numerics modules, so one payload SHA covers a program run.
#
# Output: lilith-payload-main-<sha12>.tar.gz (exactly verify_payload.PAYLOAD_FILES at
# their repo paths + source-manifest.json) and source-manifest-main-<sha12>.json.
# Refuses a dirty tracked tree, a HEAD that is not on origin/main, or an existing output.
# On the pod: extract, then run with LILITH_EXPECTED_MAIN_SHA=<printed main_sha>.
set -euo pipefail
umask 022
export COPYFILE_DISABLE=1   # macOS tar: no AppleDouble ._* entries in the payload

die() {
  printf 'make_payload: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
OUT_DIR="${1:-$REPO/target/lilith-payload}"

git -C "$REPO" fetch -q origin main ||
  die lilith::payload::fetch "cannot fetch origin/main" "check network and credentials; provenance needs the real main"
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD)"
MAIN_SHA="$(git -C "$REPO" rev-parse origin/main)"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] ||
  die lilith::payload::dirty "tracked files in $REPO are modified" "build the payload from a clean checkout of origin/main"
git -C "$REPO" merge-base --is-ancestor "$HEAD_SHA" "$MAIN_SHA" ||
  die lilith::payload::not_main "HEAD $HEAD_SHA is not on origin/main ($MAIN_SHA)" \
    "merge the PR first, then: git checkout --detach origin/main"

SHORT="${HEAD_SHA:0:12}"
TARBALL="$OUT_DIR/lilith-payload-main-$SHORT.tar.gz"
MANIFEST_COPY="$OUT_DIR/source-manifest-main-$SHORT.json"
[[ ! -e "$TARBALL" && ! -e "$MANIFEST_COPY" ]] ||
  die lilith::payload::exists "$TARBALL already exists" "payloads are write-once; remove it deliberately or reuse it"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# The file list and the manifest builder come from the committed HEAD, never the worktree.
git -C "$REPO" show "$HEAD_SHA:scripts/lilith/runpod/verify_payload.py" > "$STAGE/.verify_payload.py"
FILES=()
while IFS= read -r line; do FILES+=("$line"); done < <(python3 "$STAGE/.verify_payload.py" list)
git -C "$REPO" archive --format=tar "$HEAD_SHA" -- "${FILES[@]}" | tar -x -C "$STAGE" ||
  die lilith::payload::archive "a payload file is not in commit $HEAD_SHA" "every PAYLOAD_FILES entry must be committed on main"
rm "$STAGE/.verify_payload.py"
python3 "$STAGE/scripts/lilith/runpod/verify_payload.py" build --root "$STAGE" --main-sha "$HEAD_SHA" \
  --output "$STAGE/source-manifest.json"

mkdir -p "$OUT_DIR"
tar -C "$STAGE" -czf "$TARBALL" source-manifest.json "${FILES[@]}"
cp "$STAGE/source-manifest.json" "$MANIFEST_COPY"
sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
printf '{"main_sha": "%s", "tarball": "%s", "tarball_sha256": "%s", "manifest": "%s", "manifest_sha256": "%s", "files": %d}\n' \
  "$HEAD_SHA" "$TARBALL" "$(sha "$TARBALL")" "$MANIFEST_COPY" "$(sha "$MANIFEST_COPY")" "${#FILES[@]}"
