#!/usr/bin/env bash
# Phase 50 / W1c -- offline transform of the openindiana-rust-ci image.
#
# WHY THIS EXISTS
# ---------------
# The published `openindiana-rust-ci` OI Hipster 2026.04 image is a broken
# build. Booting it under vmctl reaches multi-user but never becomes
# SSH-reachable. Root cause (2026-07-06, this session) is THREE independent
# image-BUILD defects -- not a seed/cloud-init/provisioner problem as earlier
# sessions assumed:
#
#   1. /usr/lib/sysconfig/sysconfig is an x86-64 GNU/Linux ELF
#      (interpreter /lib64/ld-linux-x86-64.so.2). illumos cannot exec it;
#      the SMF start method dies from SIGKILL every boot and
#      svc:/system/sysconfig:default goes to maintenance. No login user is
#      ever created -> SSH auth fails.
#
#   2. The guest has NO IDE/ATAPI cdrom driver: the running device tree shows
#      only isa@1 + the three virtio devices (vioblk/vioif/viorand), no
#      pci-ide/ata node. vmctl attaches the NoCloud CIDATA seed as an IDE
#      cdrom (crates/vm-manager/src/backends/qemu.rs, `if=ide,media=cdrom`),
#      so the seed -- and the per-run SSH key inside it -- is unreadable by
#      the guest.
#
#   3. /sbin/dladm, /sbin/ipadm and /sbin/flowadm are MISSING. Only the
#      SUNWcs compat symlinks (usr/sbin/dladm -> ../../sbin/dladm) are
#      installed; the package that ships the actual command binaries
#      (system/network core-os) was never installed by the recipe. Without
#      dladm/ipadm the datalink + IP layer cannot be configured, so
#      svc:/network/physical:default fails (`/sbin/dladm: not found`,
#      `cannot plumb net0: Could not open DLPI link`) and the guest never
#      gets an IP -> the SSH host-forward has nothing to reach.
#
# WHAT THIS SCRIPT FIXES (defects #1 and part of #2's fallout)
# -----------------------------------------------------------
# It mounts the image offline (qemu-nbd + `zpool import`, no boot needed) and:
#   * replaces the Linux-ELF sysconfig with a native /sbin/sh reimplementation
#     (scripts/illumos-image-fix/sysconfig-native) that reads the CIDATA seed
#     IF it is ever readable, creates the seed's login user(s) + SSH keys,
#     grants them the "Primary Administrator" RBAC profile, sets hostname, and
#     brings up DHCP -- and always exits 0 so the SMF instance goes online;
#   * drops a sudo -> pfexec shim at /usr/bin/sudo (the image ships no sudo(8),
#     but the vmctl provisioner scripts + smoke.sh call `sudo`);
#   * installs a corrected /etc/svc/profile/site/ipmp-disable.xml so
#     svc:/network/ipmp:default (fatal on a single-NIC guest) is disabled;
#   * bakes declarative /etc/hostname.{net0,vioif0} + /etc/dhcp.{net0,vioif0}
#     so net-physical would DHCP the primary NIC on its own.
#
# WHAT IT CANNOT FIX (defect #3 -- needs an image REBUILD)
# -------------------------------------------------------
# The missing /sbin/dladm, /sbin/ipadm, /sbin/flowadm binaries CANNOT be
# recovered offline: the image carries no IPS file-content cache, and the
# 2025.04 OI image on hand ships a different (non-ABI-compatible) build of
# libdladm/libipadm/dlmgmtd. So after this transform the guest STILL will not
# get networking and is STILL not SSH-reachable. The real fix is to rebuild
# openindiana-rust-ci with the network-commands package installed (and a
# native-illumos sysconfig, and a seed bus illumos can read) -- see
# refraction-forger / solstice-ci recipes. This script exists so that the
# non-networking fixes are reproducible and so a rebuilt image only has to
# solve #3.
#
# USAGE:  sudo-capable, needs qemu-nbd + zfs on the host.
#     tests/vm/scripts/bake-illumos-fixed-image.sh \
#         [SRC_IMAGE] [DST_IMAGE]
#   defaults:
#     SRC = ~/.local/share/vmctl/images/openindiana-rust-ci.qcow2
#     DST = ~/.local/share/vmctl/images/openindiana-rust-ci-fixed.qcow2
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/illumos-image-fix"

SRC="${1:-$HOME/.local/share/vmctl/images/openindiana-rust-ci.qcow2}"
DST="${2:-$HOME/.local/share/vmctl/images/openindiana-rust-ci-fixed.qcow2}"
NBD="${NBD_DEV:-/dev/nbd0}"
POOL="forgebuild_0197087a"          # the OI-rust-ci root pool name
MNT="$(mktemp -d)"

log() { printf '[bake] %s\n' "$*"; }

cleanup() {
    sync || true
    sudo zfs umount "$POOL/ROOT/be-1" 2>/dev/null || true
    sudo zpool export "$POOL" 2>/dev/null || true
    sudo qemu-nbd -d "$NBD" 2>/dev/null || true
    rmdir "$MNT" 2>/dev/null || true
}
trap cleanup EXIT

[ -f "$SRC" ] || { echo "source image not found: $SRC" >&2; exit 1; }

log "copying $SRC -> $DST (standalone, ~7GB)"
qemu-img convert -f qcow2 -O qcow2 "$SRC" "$DST"

log "loading nbd + attaching $DST on $NBD"
sudo modprobe nbd max_part=16
sudo qemu-nbd -c "$NBD" -f qcow2 "$DST"
sleep 1; sudo partprobe "$NBD" 2>/dev/null || true; sleep 1

log "importing pool $POOL (altroot $MNT)"
sudo zpool import -f -N -R "$MNT" "$POOL"
sudo zfs set mountpoint=/ "$POOL/ROOT/be-1"
sudo zfs mount "$POOL/ROOT/be-1"

# 1. native sysconfig replacement (keep the Linux ELF as .bak for forensics)
if [ ! -f "$MNT/usr/lib/sysconfig/sysconfig.linux-elf.bak" ]; then
    sudo cp "$MNT/usr/lib/sysconfig/sysconfig" \
            "$MNT/usr/lib/sysconfig/sysconfig.linux-elf.bak"
fi
sudo cp "$FIX/sysconfig-native" "$MNT/usr/lib/sysconfig/sysconfig"
sudo chown root:bin "$MNT/usr/lib/sysconfig/sysconfig"
sudo chmod 555 "$MNT/usr/lib/sysconfig/sysconfig"
log "installed native sysconfig provisioner"

# 2. sudo -> pfexec shim
sudo cp "$FIX/sudo" "$MNT/usr/bin/sudo"
sudo chown root:bin "$MNT/usr/bin/sudo"
sudo chmod 555 "$MNT/usr/bin/sudo"
log "installed sudo->pfexec shim"

# 3. corrected ipmp-disable site profile
sudo mkdir -p "$MNT/etc/svc/profile/site"
sudo cp "$FIX/ipmp-disable.xml" "$MNT/etc/svc/profile/site/ipmp-disable.xml"
sudo chown root:sys "$MNT/etc/svc/profile/site/ipmp-disable.xml"
sudo chmod 444 "$MNT/etc/svc/profile/site/ipmp-disable.xml"
log "installed ipmp-disable site profile"

# 4. declarative DHCP on the primary NIC (both likely names)
for ifc in net0 vioif0; do
    sudo sh -c ": > '$MNT/etc/hostname.$ifc'"
    sudo sh -c ": > '$MNT/etc/dhcp.$ifc'"
    sudo chmod 644 "$MNT/etc/hostname.$ifc" "$MNT/etc/dhcp.$ifc"
done
log "baked /etc/hostname.* + /etc/dhcp.* for net0/vioif0"

log "done. NOTE: this image is still NOT SSH-reachable -- defect #3"
log "(missing /sbin/dladm,ipadm,flowadm) blocks networking until a rebuild."
