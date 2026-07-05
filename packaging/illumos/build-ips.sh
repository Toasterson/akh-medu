#!/bin/sh
#
# Build an IPS (pkg(5)) package for akh-medu and emit a self-contained .p5p
# archive (and optionally publish to a remote pkg repo).
#
# Runs on illumos with the standard pkg(5) toolchain (pkgsend, pkgmogrify,
# pkgdepend, pkgrepo, pkgrecv, pkgfmt, pkglint). Point --bindir at a directory
# holding the freshly built `akh`, `akhomed`, and `seshd` binaries.
#
# Produces:
#   <outdir>/akh-medu-<version>.p5p           (install: pkg install -g <file> akh-medu)
#   <outdir>/akh-medu-<version>.p5p.sha256
#
# Usage:
#   packaging/illumos/build-ips.sh --version 0.7.0 --bindir target/release \
#       [--outdir dist] [--publisher akh-medu] [--branch 0] [--repo-uri URI]
#
# Account UID/GID defaults can be overridden via env (AKH_UID/AKH_GID/
# SESHAT_UID/SESHAT_GID) if they collide with existing accounts.

set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)

VERSION=""
BINDIR=""
OUTDIR="$REPO/dist"
PUBLISHER="akh-medu"
BRANCH="0"
REPO_URI=""

AKH_UID=${AKH_UID:-911}
AKH_GID=${AKH_GID:-911}
SESHAT_UID=${SESHAT_UID:-912}
SESHAT_GID=${SESHAT_GID:-912}

while [ $# -gt 0 ]; do
	case "$1" in
	--version) VERSION=$2; shift 2 ;;
	--bindir) BINDIR=$2; shift 2 ;;
	--outdir) OUTDIR=$2; shift 2 ;;
	--publisher) PUBLISHER=$2; shift 2 ;;
	--branch) BRANCH=$2; shift 2 ;;
	--repo-uri) REPO_URI=$2; shift 2 ;;
	*) echo "unknown option: $1" >&2; exit 2 ;;
	esac
done

[ -n "$VERSION" ] || { echo "--version is required" >&2; exit 2; }
[ -n "$BINDIR" ] || { echo "--bindir is required" >&2; exit 2; }

WORK=$(mktemp -d /tmp/akh-medu-ips.XXXXXX)
PROTO="$WORK/proto"
trap 'rm -rf "$WORK"' EXIT

echo "==> Staging proto area"
mkdir -p \
	"$PROTO/opt/akh-medu/bin" \
	"$PROTO/usr/bin" \
	"$PROTO/lib/svc/manifest/application/akh-medu" \
	"$PROTO/lib/svc/method" \
	"$PROTO/etc/akh-medu" \
	"$PROTO/var/lib/akh-medu/state" \
	"$PROTO/var/lib/akh-medu/seshat/models" \
	"$PROTO/var/cache/akh-medu"

for b in akh akhomed seshd; do
	[ -f "$BINDIR/$b" ] || { echo "missing binary: $BINDIR/$b" >&2; exit 1; }
	cp "$BINDIR/$b" "$PROTO/opt/akh-medu/bin/$b"
	chmod 0555 "$PROTO/opt/akh-medu/bin/$b"
done

ln -s ../../opt/akh-medu/bin/akh "$PROTO/usr/bin/akh"

cp "$REPO/contrib/smf/akhomed.xml" "$PROTO/lib/svc/manifest/application/akh-medu/akhomed.xml"
cp "$REPO/contrib/smf/seshd.xml"   "$PROTO/lib/svc/manifest/application/akh-medu/seshd.xml"
chmod 0444 "$PROTO"/lib/svc/manifest/application/akh-medu/*.xml

cp "$REPO/contrib/smf/akhomed.method" "$PROTO/lib/svc/method/akhomed"
cp "$REPO/contrib/smf/seshd.method"   "$PROTO/lib/svc/method/seshd"
chmod 0555 "$PROTO"/lib/svc/method/akhomed "$PROTO"/lib/svc/method/seshd

cp "$REPO/contrib/config/seshat.toml.sample" "$PROTO/etc/akh-medu/seshat.toml.sample"
# Live config delivered with preserve=true (see the .mog) so upgrades keep edits.
cp "$REPO/contrib/config/seshat.toml.sample" "$PROTO/etc/akh-medu/seshat.toml"
chmod 0644 "$PROTO"/etc/akh-medu/seshat.toml*

echo "==> Generating base manifest"
pkgsend generate "$PROTO" | pkgfmt > "$WORK/base.p5m"

echo "==> Applying transforms"
pkgmogrify \
	-D VERSION="$VERSION" \
	-D BRANCH="$BRANCH" \
	-D PUBLISHER="$PUBLISHER" \
	-D AKH_UID="$AKH_UID" -D AKH_GID="$AKH_GID" \
	-D SESHAT_UID="$SESHAT_UID" -D SESHAT_GID="$SESHAT_GID" \
	"$WORK/base.p5m" "$REPO/packaging/illumos/akh-medu.p5m.mog" \
	| pkgfmt > "$WORK/akh-medu.p5m"

echo "==> Resolving dependencies"
pkgdepend generate -md "$PROTO" "$WORK/akh-medu.p5m" > "$WORK/akh-medu.dep.p5m"
pkgdepend resolve -m "$WORK/akh-medu.dep.p5m"
FINAL="$WORK/akh-medu.dep.p5m.res"

echo "==> Linting (non-fatal)"
pkglint "$FINAL" || echo "    pkglint reported issues (see above) — continuing"

echo "==> Publishing to staging repo"
pkgrepo create "$WORK/repo"
pkgrepo set -s "$WORK/repo" publisher/prefix="$PUBLISHER"
# Two payload roots: the proto (file contents) and the repo root (for LICENSE).
pkgsend -s "$WORK/repo" publish -d "$PROTO" -d "$REPO" "$FINAL"

mkdir -p "$OUTDIR"
ARCHIVE="$OUTDIR/akh-medu-${VERSION}.p5p"
rm -f "$ARCHIVE"
echo "==> Creating archive $ARCHIVE"
pkgrecv -s "$WORK/repo" -a -d "$ARCHIVE" '*'

if command -v sha256sum >/dev/null 2>&1; then
	( cd "$OUTDIR" && sha256sum "akh-medu-${VERSION}.p5p" > "akh-medu-${VERSION}.p5p.sha256" )
else
	( cd "$OUTDIR" && digest -a sha256 "akh-medu-${VERSION}.p5p" > "akh-medu-${VERSION}.p5p.sha256" )
fi

if [ -n "$REPO_URI" ]; then
	echo "==> Republishing to remote repo $REPO_URI"
	pkgrecv -s "$WORK/repo" -d "$REPO_URI" '*'
fi

echo "==> Done:"
echo "    $ARCHIVE"
echo "    install with: pkg install -g $ARCHIVE akh-medu"
