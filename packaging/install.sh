#!/bin/sh
#
# akh-medu install helper.
#
# Run from the root of an unpacked release tarball. Detects the OS and installs
# the binaries, service definitions, dedicated service accounts, and runtime
# directories. Idempotent — safe to re-run for upgrades.
#
#   Linux   : binaries -> /usr/local/bin, systemd units -> /etc/systemd/system
#   illumos : binaries -> /opt/akh-medu/bin, SMF manifests imported via svccfg
#             (on illumos, prefer `pkg install -g akh-medu-*.p5p akh-medu`;
#              this path is the tarball fallback)
#
# Does NOT start the services — it prints the enable command so you stay in
# control of when they come up.
#
# Usage: ./install.sh [--prefix DIR] [--no-users] [--help]

set -eu

PROG=${0##*/}
SRC=$(cd "$(dirname "$0")" && pwd)

PREFIX=""
CREATE_USERS=1

usage() {
	cat <<EOF
$PROG — install akh-medu from an unpacked release tarball.

Options:
  --prefix DIR   Install binaries under DIR/bin (Linux only; default /usr/local).
  --no-users     Do not create the akh/seshat service accounts.
  --help         Show this help.
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--prefix) PREFIX=${2:?--prefix needs a value}; shift 2 ;;
	--prefix=*) PREFIX=${1#*=}; shift ;;
	--no-users) CREATE_USERS=0; shift ;;
	-h|--help) usage; exit 0 ;;
	*) echo "$PROG: unknown option: $1" >&2; usage >&2; exit 2 ;;
	esac
done

if [ "$(id -u)" -ne 0 ]; then
	echo "$PROG: must run as root (try: sudo ./install.sh)" >&2
	exit 1
fi

OS=$(uname -s)

# --- shared helpers ---------------------------------------------------------

ensure_dir() {
	# ensure_dir <path> <owner> <mode>
	[ -d "$1" ] || mkdir -p "$1"
	[ -n "${2:-}" ] && chown "$2" "$1" 2>/dev/null || true
	[ -n "${3:-}" ] && chmod "$3" "$1" 2>/dev/null || true
}

install_file() {
	# install_file <src> <dest> <mode>
	cp "$1" "$2"
	chmod "$3" "$2"
}

# --- Linux ------------------------------------------------------------------

install_linux() {
	prefix=${PREFIX:-/usr/local}
	bindir="$prefix/bin"
	libdir="$prefix/lib/akh-medu"

	echo "==> Installing akh-medu (Linux) under $prefix"
	ensure_dir "$bindir"
	for b in akh akhomed seshd; do
		if [ -f "$SRC/bin/$b" ]; then
			install_file "$SRC/bin/$b" "$bindir/$b" 0755
			echo "    installed $bindir/$b"
		fi
	done

	if ls "$SRC"/lib/libonnxruntime.so* >/dev/null 2>&1; then
		ensure_dir "$libdir"
		cp "$SRC"/lib/libonnxruntime.so* "$libdir/"
		echo "    installed onnxruntime -> $libdir"
	fi

	if [ "$CREATE_USERS" -eq 1 ]; then
		create_user_linux akh /var/lib/akh-medu
		create_user_linux seshat /var/lib/akh-medu/seshat
	fi

	ensure_dir /etc/akh-medu root:root 0755
	ensure_dir /var/lib/akh-medu akh:akh 0750
	ensure_dir /var/lib/akh-medu/state akh:akh 0750
	ensure_dir /var/cache/akh-medu akh:akh 0750
	ensure_dir /var/lib/akh-medu/seshat seshat:seshat 0750
	ensure_dir /var/lib/akh-medu/seshat/models seshat:seshat 0750

	if [ -f "$SRC/config/seshat.toml.sample" ]; then
		if [ ! -f /etc/akh-medu/seshat.toml ]; then
			install_file "$SRC/config/seshat.toml.sample" /etc/akh-medu/seshat.toml 0640
			chown root:seshat /etc/akh-medu/seshat.toml 2>/dev/null || true
			echo "    wrote /etc/akh-medu/seshat.toml (edit the database URL)"
		else
			install_file "$SRC/config/seshat.toml.sample" /etc/akh-medu/seshat.toml.sample 0644
			echo "    /etc/akh-medu/seshat.toml exists; wrote .sample alongside"
		fi
	fi

	for unit in akhomed seshd; do
		if [ -f "$SRC/systemd/$unit.service" ]; then
			install_file "$SRC/systemd/$unit.service" "/etc/systemd/system/$unit.service" 0644
			echo "    installed /etc/systemd/system/$unit.service"
		fi
	done

	if command -v systemctl >/dev/null 2>&1; then
		systemctl daemon-reload || true
		echo
		echo "Done. Enable services with:"
		echo "    systemctl enable --now akhomed"
		echo "    systemctl enable --now seshd     # after editing /etc/akh-medu/seshat.toml"
	else
		echo
		echo "Done. systemd not detected — start the binaries directly or wire your init."
	fi
}

create_user_linux() {
	# create_user_linux <name> <home>
	name=$1; home=$2
	if ! getent group "$name" >/dev/null 2>&1; then
		groupadd -r "$name"
	fi
	if ! getent passwd "$name" >/dev/null 2>&1; then
		useradd -r -g "$name" -d "$home" -s /usr/sbin/nologin \
			-c "akh-medu service account" "$name"
		echo "    created user $name"
	fi
}

# --- illumos ----------------------------------------------------------------

install_illumos() {
	bindir=/opt/akh-medu/bin

	echo "==> Installing akh-medu (illumos) under /opt/akh-medu"
	echo "    (note: 'pkg install -g akh-medu-*.p5p akh-medu' is the preferred path)"
	ensure_dir "$bindir"
	for b in akh akhomed seshd; do
		if [ -f "$SRC/bin/$b" ]; then
			install_file "$SRC/bin/$b" "$bindir/$b" 0555
			echo "    installed $bindir/$b"
		fi
	done

	# Symlink the CLI onto the default PATH.
	[ -f "$bindir/akh" ] && ln -sf "$bindir/akh" /usr/bin/akh

	if [ "$CREATE_USERS" -eq 1 ]; then
		create_user_illumos akh /var/lib/akh-medu
		create_user_illumos seshat /var/lib/akh-medu/seshat
	fi

	ensure_dir /etc/akh-medu root:root 0755
	ensure_dir /var/lib/akh-medu akh:akh 0750
	ensure_dir /var/lib/akh-medu/state akh:akh 0750
	ensure_dir /var/cache/akh-medu akh:akh 0750
	ensure_dir /var/lib/akh-medu/seshat seshat:seshat 0750
	ensure_dir /var/lib/akh-medu/seshat/models seshat:seshat 0750

	if [ -f "$SRC/config/seshat.toml.sample" ] && [ ! -f /etc/akh-medu/seshat.toml ]; then
		install_file "$SRC/config/seshat.toml.sample" /etc/akh-medu/seshat.toml 0640
		chown root:seshat /etc/akh-medu/seshat.toml 2>/dev/null || true
		echo "    wrote /etc/akh-medu/seshat.toml (edit the database URL)"
	fi

	# SMF method scripts + manifests. This tarball path is a fallback; the
	# supported illumos install is the IPS package (pkg install -g *.p5p).
	if [ ! -f "$SRC/smf/akhomed.xml" ]; then
		echo "    WARNING: no SMF manifests in this tarball (smf/ missing)." >&2
		echo "    Installed binaries only. For services, use the IPS package:" >&2
		echo "        pkg install -g akh-medu-*.p5p akh-medu" >&2
		return 0
	fi

	ensure_dir /lib/svc/method
	for m in akhomed seshd; do
		if [ -f "$SRC/smf/$m.method" ]; then
			install_file "$SRC/smf/$m.method" "/lib/svc/method/$m" 0555
		fi
	done

	ensure_dir /lib/svc/manifest/application/akh-medu
	for x in akhomed seshd; do
		if [ -f "$SRC/smf/$x.xml" ]; then
			install_file "$SRC/smf/$x.xml" "/lib/svc/manifest/application/akh-medu/$x.xml" 0444
			svccfg import "/lib/svc/manifest/application/akh-medu/$x.xml" && \
				echo "    imported svc:/application/akh-medu/$x"
		fi
	done

	echo
	echo "Done. Enable services with:"
	echo "    svcadm enable akhomed"
	echo "    svcadm enable seshd     # after editing /etc/akh-medu/seshat.toml"
}

create_user_illumos() {
	# create_user_illumos <name> <home>
	name=$1; home=$2
	if ! getent group "$name" >/dev/null 2>&1; then
		groupadd "$name"
	fi
	if ! getent passwd "$name" >/dev/null 2>&1; then
		useradd -g "$name" -d "$home" -s /bin/false \
			-c "akh-medu service account" "$name"
		echo "    created user $name"
	fi
}

# --- dispatch ---------------------------------------------------------------

case "$OS" in
Linux)  install_linux ;;
SunOS)  install_illumos ;;
*)      echo "$PROG: unsupported OS '$OS' (Linux or illumos/SunOS only)" >&2; exit 1 ;;
esac
