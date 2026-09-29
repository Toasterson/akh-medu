# shellcheck shell=bash
# shellcheck disable=SC2034 # the variables here are used by the scripts that source this file
# Shared settings and helpers for the hephaistos serving scripts (ADR 121). Sourced, never run.
#
# Everything a script needs to know about the box lives here, each value overridable from the
# environment. The defaults are hephaistos: Ubuntu 22.04, ROCm, RX 7900 XTX (gfx1100), passwordless
# sudo for `toasty`, the akh-medu-runs NFS share from 192.168.1.81.

HEPH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROG="${PROG:-$(basename "${0:-heph}")}"
PINS="${LILITH_PINS_FILE:-$HEPH_DIR/pins.json}"   # LILITH_PINS_FILE: tests only; a real run uses the pins in the repo
TOOLS=(python3 "$HEPH_DIR/heph_tools.py" --pins "$PINS")

PREFIX="${LILITH_SERVE_PREFIX:-/opt/lilith-serve}"        # llama.cpp builds + run_server.sh (root-owned)
STATE="${LILITH_SERVE_STATE:-/var/lib/lilith-serve}"      # models, shader cache (service user's home)
ETC="${LILITH_SERVE_ETC:-/etc/lilith-serve}"              # server.env (0644), api-key (0600)
SVC_USER="${LILITH_SERVE_USER:-lilith-serve}"
WORK="${LILITH_WORK_DIR:-$HOME/lilith-serve-work}"        # operator-owned: source, build, venv, staged parent

# The runs share. `setpriv` = the share's top directory is 0700 uid 501 (the Mac user) and root is
# squashed, so every read and write from hephaistos (uid 1000) runs as uid 501 (memory note
# hephaistos-access-and-nfs-share). `direct` = this user can already use the mount (the Mac).
NFS_EXPORT="${LILITH_NFS_EXPORT:-192.168.1.81:/export/toast/akh-medu-runs}"
NFS_MOUNT="${LILITH_NFS_MOUNT:-/mnt/akh-medu-runs}"
NFS_ACCESS="${LILITH_NFS_ACCESS:-setpriv}"
NFS_UID="${LILITH_NFS_UID:-501}"
NFS_GID="${LILITH_NFS_GID:-20}"

die() { # die CODE MESSAGE HELP
  printf '%s: error[%s]: %s\n  help: %s\n' "$PROG" "$1" "$2" "$3" >&2
  exit 1
}
log() { printf '%s: %s\n' "$PROG" "$*" >&2; }

# LILITH_NO_SUDO=1 runs everything as the calling user (tests and a Mac dry run; the prefix and
# state directories must then be writable by that user).
as_root() {
  if [[ "${LILITH_NO_SUDO:-0}" == 1 || ${EUID:-$(id -u)} -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

need_cmd() { # need_cmd COMMAND HINT
  command -v "$1" >/dev/null 2>&1 || die lilith::heph::tool "'$1' is not installed" "$2"
}

pin() { "${TOOLS[@]}" pins get "$1"; }

free_gib() { # free_gib PATH: whole GiB free on the filesystem holding PATH (nearest existing parent)
  python3 - "$1" <<'PY'
import os, shutil, sys
path = sys.argv[1]
while not os.path.exists(path):
    path = os.path.dirname(path) or "/"
print(shutil.disk_usage(path).free // 2**30)
PY
}

need_free_gib() { # need_free_gib PATH GIB WHAT   (LILITH_MIN_FREE_GIB overrides GIB: tests, or a box you know)
  local have want="${LILITH_MIN_FREE_GIB:-$2}"
  have="$(free_gib "$1")"
  set -- "$1" "$want" "$3"
  [[ "$have" -ge "$2" ]] || die lilith::heph::disk "$3 needs $2 GiB free under $1, only $have GiB is" \
    "free space there, or point LILITH_WORK_DIR / LILITH_SERVE_STATE at a bigger filesystem"
}

# ---- the runs share ----------------------------------------------------------------------------

nfs_root() { # print the mountpoint of the runs share, mounting it when it is not mounted
  local mountpoint=""
  if [[ "$NFS_ACCESS" == direct ]]; then
    [[ -d "$NFS_MOUNT" ]] || die lilith::heph::nfs "$NFS_MOUNT is not a directory" \
      "mount the runs share there, or set LILITH_NFS_MOUNT"
    printf '%s\n' "$NFS_MOUNT"
    return
  fi
  need_cmd findmnt "install util-linux"
  mountpoint="$(findmnt -rn -S "$NFS_EXPORT" -o TARGET 2>/dev/null | head -n1 || true)"
  if [[ -z "$mountpoint" ]]; then
    need_cmd mount.nfs "sudo apt-get install -y nfs-common"
    as_root install -d -m 755 "$NFS_MOUNT"
    log "mounting $NFS_EXPORT at $NFS_MOUNT"
    as_root mount -t nfs -o vers=4,hard,timeo=600 "$NFS_EXPORT" "$NFS_MOUNT" \
      || die lilith::heph::nfs "could not mount $NFS_EXPORT at $NFS_MOUNT" \
           "check that 192.168.1.81 is up and exports it to this host (showmount -e 192.168.1.81)"
    mountpoint="$NFS_MOUNT"
  fi
  printf '%s\n' "$mountpoint"
}

nfs_run() { # nfs_run COMMAND...: run as the share's owner
  if [[ "$NFS_ACCESS" == direct ]]; then
    "$@"
  else
    need_cmd setpriv "install util-linux"
    as_root setpriv --reuid="$NFS_UID" --regid="$NFS_GID" --clear-groups \
      --inh-caps=+dac_read_search --ambient-caps=+dac_read_search "$@"
  fi
}

nfs_write_file() { # nfs_write_file SRC DST: copy a local file to the share atomically (partial + rename)
  local src="$1" dst="$2"
  nfs_run mkdir -p "$(dirname "$dst")"
  # shellcheck disable=SC2016 # $1 is expanded by the inner shell on purpose
  nfs_run sh -c 'cat > "$1.partial" && mv -f "$1.partial" "$1"' _ "$dst" <"$src" \
    || die lilith::heph::nfs_write "could not write $dst on the share" \
         "check the mount is read-write and has space; nothing was renamed into place"
}

# ---- llama.cpp source ---------------------------------------------------------------------------

ensure_llama_source() { # ensure_llama_source DIR: DIR is a clean checkout of exactly the pinned commit
  local dir="$1" repo commit head
  repo="$(pin llama_cpp.repo)"
  commit="$(pin llama_cpp.commit)"
  need_cmd git "sudo apt-get install -y git"
  mkdir -p "$dir"
  if [[ ! -d "$dir/.git" ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$repo"
  fi
  head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$head" != "$commit" ]]; then
    log "fetching llama.cpp $commit"
    git -C "$dir" fetch -q --depth 1 origin "$commit" \
      || die lilith::heph::source "could not fetch llama.cpp $commit from $repo" \
           "check network access to github.com; the commit is the $(pin llama_cpp.tag) release"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
  head="$(git -C "$dir" rev-parse HEAD)"
  [[ "$head" == "$commit" ]] || die lilith::heph::source "checkout is at $head, not the pinned $commit" \
    "delete $dir and rerun"
  [[ -z "$(git -C "$dir" status --porcelain)" ]] || die lilith::heph::source \
    "$dir has local changes; the build must be the pinned source" "delete $dir and rerun"
}
