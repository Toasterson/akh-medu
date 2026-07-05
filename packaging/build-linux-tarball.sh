#!/bin/sh
#
# Assemble a relocatable Linux release tarball from already-built binaries.
#
# In CI the binaries are *extracted from the freshly built docker images*
# (single-binary builds that fit the runner's memory budget) rather than
# recompiled — see .forgejo/workflows/release.yml. Run locally by pointing
# --bindir at a `target/release` directory.
#
# Produces:
#   <outdir>/akh-medu-<version>-<arch>-linux.tar.gz
#   <outdir>/akh-medu-<version>-<arch>-linux.tar.gz.sha256
#
# Usage:
#   packaging/build-linux-tarball.sh \
#       --version 0.7.0 --bindir target/release [--libdir /usr/local/lib] \
#       [--arch x86_64] [--outdir dist]

set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)

VERSION=""
BINDIR=""
LIBDIR=""
ARCH=$(uname -m)
OUTDIR="$REPO/dist"

while [ $# -gt 0 ]; do
	case "$1" in
	--version) VERSION=$2; shift 2 ;;
	--bindir) BINDIR=$2; shift 2 ;;
	--libdir) LIBDIR=$2; shift 2 ;;
	--arch) ARCH=$2; shift 2 ;;
	--outdir) OUTDIR=$2; shift 2 ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

[ -n "$VERSION" ] || { echo "--version is required" >&2; exit 2; }
[ -n "$BINDIR" ] || { echo "--bindir is required" >&2; exit 2; }

NAME="akh-medu-${VERSION}-${ARCH}-linux"
STAGE="$OUTDIR/$NAME"

echo "==> Staging $NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/systemd" "$STAGE/config"

for b in akh akhomed seshd; do
	if [ -f "$BINDIR/$b" ]; then
		cp "$BINDIR/$b" "$STAGE/bin/$b"
		chmod 0755 "$STAGE/bin/$b"
	else
		echo "    WARNING: $BINDIR/$b not found — skipping" >&2
	fi
done

if [ -n "$LIBDIR" ] && ls "$LIBDIR"/libonnxruntime.so* >/dev/null 2>&1; then
	mkdir -p "$STAGE/lib"
	cp "$LIBDIR"/libonnxruntime.so* "$STAGE/lib/"
	echo "    bundled onnxruntime from $LIBDIR"
fi

cp "$REPO/contrib/systemd/system/akhomed.service" "$STAGE/systemd/"
cp "$REPO/contrib/systemd/system/seshd.service" "$STAGE/systemd/"
# SMF manifests/methods too: OS-agnostic text consumed by install.sh's illumos
# fallback (the supported illumos path is the IPS .p5p, but this keeps a
# hand-built illumos tarball functional).
mkdir -p "$STAGE/smf"
cp "$REPO/contrib/smf/akhomed.xml" "$REPO/contrib/smf/seshd.xml" "$STAGE/smf/"
cp "$REPO/contrib/smf/akhomed.method" "$STAGE/smf/akhomed.method"
cp "$REPO/contrib/smf/seshd.method" "$STAGE/smf/seshd.method"
cp "$REPO/contrib/config/seshat.toml.sample" "$STAGE/config/"
cp "$REPO/packaging/install.sh" "$STAGE/install.sh"
chmod 0755 "$STAGE/install.sh"
cp "$REPO/LICENSE" "$STAGE/" 2>/dev/null || true
cp "$REPO/README.md" "$STAGE/" 2>/dev/null || true
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

echo "==> Creating tarball"
( cd "$OUTDIR" && tar czf "$NAME.tar.gz" "$NAME" )
( cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
rm -rf "$STAGE"

echo "==> $OUTDIR/$NAME.tar.gz"
cat "$OUTDIR/$NAME.tar.gz.sha256"
