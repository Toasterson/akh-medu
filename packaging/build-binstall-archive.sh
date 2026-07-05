#!/bin/sh
#
# Build a cargo-binstall-compatible archive for the lean `akh` client.
#
# Produces (matching [package.metadata.binstall] in Cargo.toml):
#   <outdir>/akh-<version>-<target>.tar.gz          (the `akh` binary at the root)
#   <outdir>/akh-<version>-<target>.tar.gz.sha256
#
# binstall's bin-dir is "{ bin }{ binary-ext }", i.e. it expects the binary at
# the archive root named `akh` — so we stage exactly that, nothing else.
#
# Usage:
#   packaging/build-binstall-archive.sh --version 0.7.0 \
#       --target x86_64-unknown-linux-gnu --bin target/release/akh [--outdir dist]

set -eu

VERSION=""
TARGET=""
BIN=""
OUTDIR="dist"

while [ $# -gt 0 ]; do
	case "$1" in
	--version) VERSION=$2; shift 2 ;;
	--target) TARGET=$2; shift 2 ;;
	--bin) BIN=$2; shift 2 ;;
	--outdir) OUTDIR=$2; shift 2 ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

[ -n "$VERSION" ] || { echo "--version is required" >&2; exit 2; }
[ -n "$TARGET" ] || { echo "--target is required" >&2; exit 2; }
[ -n "$BIN" ] || { echo "--bin is required" >&2; exit 2; }
[ -f "$BIN" ] || { echo "binary not found: $BIN" >&2; exit 1; }

NAME="akh-${VERSION}-${TARGET}"
mkdir -p "$OUTDIR"
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/akh-binstall.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT

cp "$BIN" "$STAGE/akh"
chmod 0755 "$STAGE/akh"

( cd "$STAGE" && tar czf - akh ) > "$OUTDIR/$NAME.tar.gz"

# Portable sha256 across Linux / macOS / illumos.
sha_file="$OUTDIR/$NAME.tar.gz.sha256"
if command -v sha256sum >/dev/null 2>&1; then
	( cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
elif command -v shasum >/dev/null 2>&1; then
	( cd "$OUTDIR" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
else
	( cd "$OUTDIR" && digest -a sha256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
fi

echo "==> $OUTDIR/$NAME.tar.gz"
cat "$sha_file"
