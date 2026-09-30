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

ROCM_ROOT="${LILITH_ROCM_DIR:-/opt/rocm}"                 # ROCm lives here but its bin/ is not on the default PATH
GCC_LIB_DIR="${LILITH_GCC_LIB_DIR:-/usr/lib/gcc/$(uname -m)-linux-gnu}"   # one directory per installed GCC major
CXX_INCLUDE_DIR="${LILITH_CXX_INCLUDE_DIR:-/usr/include/c++}"            # libstdc++-<N>-dev puts its headers in <N>/
VULKAN_INCLUDE_DIR="${LILITH_VULKAN_INCLUDE_DIR:-/usr/include/vulkan}"

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

# ---- toolchain probes ---------------------------------------------------------------------------

add_rocm_to_path() { # put $ROCM_ROOT/bin on PATH when it exists and is not already there
  [[ -d "$ROCM_ROOT/bin" ]] || return 0
  case ":$PATH:" in *":$ROCM_ROOT/bin:"*) ;; *) PATH="$ROCM_ROOT/bin:$PATH"; export PATH ;; esac
}

rocm_version() { # print the installed ROCm version (for example 6.3.4-76), or "none"
  local root="$ROCM_ROOT"
  if command -v hipconfig >/dev/null 2>&1; then root="$(hipconfig -R 2>/dev/null || echo "$ROCM_ROOT")"; fi
  cat "$root/.info/version" 2>/dev/null || cat "$ROCM_ROOT/.info/version" 2>/dev/null || echo none
}

highest_gcc_major() { # print the highest installed GCC major version (ROCm's clang prefers it), or nothing
  local entry
  for entry in "$GCC_LIB_DIR"/*; do
    [[ -d "$entry" ]] && basename "$entry"
  done 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -n 1
}

vulkan_headers_patch() { # print N of the installed Vulkan headers' VK_HEADER_VERSION N, or nothing
  grep -h '^#define VK_HEADER_VERSION ' "$VULKAN_INCLUDE_DIR/vulkan_core.h" 2>/dev/null | awk '{print $3}' | head -n 1
}

# ---- Vulkan SDK (jammy has no glslc and old headers) ---------------------------------------------

vulkan_system_sufficient() { # 0 when the system has glslc AND headers at least the pinned minimum patch
  local patch min
  command -v glslc >/dev/null 2>&1 || return 1
  patch="$(vulkan_headers_patch)"
  min="$(pin vulkan_sdk.system_headers_min_patch)"
  [[ -n "$patch" && "$patch" -ge "$min" ]]
}

VULKAN_SDK_VERSION=""   # set by use_vulkan_sdk; empty = the system toolchain was used
VULKAN_SDK_SHA256=""

use_vulkan_sdk() { # use_vulkan_sdk DOWNLOAD(0|1): unpack the pinned LunarG SDK under $WORK and source its env
  local download="$1" version url sha root marker tarball staging
  version="$(pin vulkan_sdk.version)"
  url="$(pin vulkan_sdk.url)"
  sha="$(pin vulkan_sdk.sha256)"
  root="$WORK/vulkan-sdk"
  marker="$root/$version.sha256"
  if [[ ! -f "$root/$version/setup-env.sh" || "$(cat "$marker" 2>/dev/null || true)" != "$sha" ]]; then
    [[ "$download" == 1 ]] || die lilith::heph::vulkan_sdk       "the system has no glslc or its Vulkan headers are older than 1.3.$(pin vulkan_sdk.system_headers_min_patch), and the pinned Vulkan SDK $version is not unpacked under $root"       "run: bash $0 --backend vulkan --install-deps (downloads the pinned SDK from sdk.lunarg.com, verifies its sha256, builds with it from the work directory)"
    [[ "${LILITH_HOST_ARCH:-$(uname -m)}" == x86_64 ]] || die lilith::heph::vulkan_sdk "the LunarG SDK tarball is x86_64 only, this host is ${LILITH_HOST_ARCH:-$(uname -m)}"       "install glslc and current Vulkan headers from your distribution instead"
    need_cmd curl "sudo apt-get install -y curl"
    need_cmd tar "sudo apt-get install -y tar xz-utils"
    need_free_gib "$WORK" 4 "the Vulkan SDK download and unpacking"
    mkdir -p "$root"
    tarball="$root/vulkan_sdk-$version.tar.xz"
    log "downloading the pinned Vulkan SDK $version"
    curl -fsSL --retry 3 -o "$tarball.partial" "$url" || die lilith::heph::vulkan_sdk \
      "could not download $url" "check network access to sdk.lunarg.com; the URL is pinned in pins.json (vulkan_sdk.url)"
    if [[ "$(python3 "$HEPH_DIR/heph_tools.py" sha256 "$tarball.partial")" != "$sha" ]]; then
      rm -f "$tarball.partial"
      die lilith::heph::vulkan_sdk "the downloaded Vulkan SDK does not match its pinned sha256 $sha" \
        "do not use it; retry once, and if it repeats LunarG replaced the file: stop and report (pins.json vulkan_sdk)"
    fi
    staging="$root/.unpack-$version"
    rm -rf "${staging:?}" "${root:?}/$version"
    mkdir -p "$staging"
    tar -xf "$tarball.partial" -C "$staging" || die lilith::heph::vulkan_sdk "could not unpack the Vulkan SDK" \
      "needs xz: sudo apt-get install -y xz-utils; then delete $root and rerun"
    rm -f "$tarball.partial"
    [[ -f "$staging/$version/setup-env.sh" ]] || die lilith::heph::vulkan_sdk \
      "the unpacked SDK has no $version/setup-env.sh" "the tarball layout changed; stop and report"
    mv "$staging/$version" "$root/$version"
    rm -rf "$staging"
    printf '%s\n' "$sha" > "$marker"
  fi
  # setup-env.sh is written for interactive shells: it reads variables it never set.
  set +u
  # shellcheck disable=SC1090,SC1091 # the SDK's own script, present only after the download
  source "$root/$version/setup-env.sh"
  set -u
  [[ -n "${VULKAN_SDK:-}" ]] || die lilith::heph::vulkan_sdk "sourcing $root/$version/setup-env.sh did not set VULKAN_SDK" \
    "delete $root/$version and rerun"
  command -v glslc >/dev/null 2>&1 || die lilith::heph::vulkan_sdk "the SDK at $VULKAN_SDK provides no glslc" \
    "delete $root/$version and rerun"
  VULKAN_SDK_VERSION="$version"
  VULKAN_SDK_SHA256="$sha"
  log "building with the pinned Vulkan SDK $version ($VULKAN_SDK)"
}
