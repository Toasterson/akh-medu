#!/usr/bin/env bash
# Phase 50 / W1d -- offline bake of the openindiana-hipster-cloud (OI
# 2025.04) image so it gives a REPRODUCIBLE SSH login independent of
# whatever the cloud-init seed / svc:/system/illumos/{metadata,useragent}
# provisioning agents do at boot.
#
# WHY THIS EXISTS
# ----------------
# docs/ai/2026-07-06-phase50-w1c-image-defects.md root-caused the
# openindiana-rust-ci image as an unfixable botched build (missing
# dladm/ipadm). openindiana-hipster-cloud.img (already on the host,
# ~/.local/share/vmctl/images/) is a COMPLETE OI build: real dladm/ipadm
# AND a real ata/ahci IDE driver (both confirmed present by mounting the
# image offline with qemu-nbd + `zpool import`, no boot -- see the W1d
# report). But its real provisioning agents (illumos's own
# svc:/system/illumos/metadata + svc:/system/illumos/useragent, a genuine
# Rust reimplementation with a cloud-init-shaped `CloudConfig`/`UserConfig`
# deserializer confirmed via DWARF debug info) have three properties that
# make them a bad SSH channel to depend on for a bounded test:
#   1. sshd_config ships `PermitRootLogin no` and neither agent touches it,
#      so even if a root SSH key were installed via meta-data, sshd itself
#      refuses the login.
#   2. The image ships NO sudo(8) at all (confirmed absent), same as
#      rust-ci -- but unlike rust-ci this is fixable with the same
#      sudo->pfexec shim from illumos-image-fix/.
#   3. useragent's CloudConfig `users:` parser (confirmed via DWARF: struct
#      UserConfig { name, expire_date, gecos, homedir, primary_group,
#      groups, lock_passwd, inactive, passwd, no_create_home, no_user_group,
#      ssh_authorized_keys, system, shell }) has NO field for granting sudo
#      or an RBAC profile -- a user created purely from cloud-init user-data
#      gets SSH key auth but zero administrative rights, which the
#      exec1/pg/builder provisioners need (they all call `sudo`).
#
# So rather than keep reverse-engineering the exact seed/meta-data schema
# these agents expect (the CloudConfig fields ARE cloud-init-compatible and
# vmctl's default `#cloud-config` generation -- tests/vm/../cloudinit.rs
# build_cloud_config() -- likely mostly works, as a bonus), this script
# bakes a login channel that does NOT depend on the seed being read at all:
# a static root SSH key + a PermitRootLogin flip + the sudo shim, all
# applied offline. cloud-init is kept wired in VMFile.kdl for hostname/
# bonus-user purposes but nothing on the SSH-reachability critical path
# depends on it succeeding.
#
# METHOD (reused from bake-illumos-fixed-image.sh / W1c): qemu-nbd attaches
# the qcow2 image as a block device with NO boot; `zpool import -R <altroot>`
# imports the guest's root pool at an altroot so all edits land under a
# temp directory, never the host's real /. This is offline and non-invasive
# -- the same method W1c used to read the rust-ci defects' SMF logs.
#
# USAGE:  sudo-capable, needs qemu-nbd + zfs on the host.
#     tests/vm/scripts/bake-hipster-cloud-image.sh \
#         [SRC_IMAGE] [DST_IMAGE] [ROOT_PUBKEY]
#   defaults:
#     SRC        = ~/.local/share/vmctl/images/openindiana-hipster-cloud.img (raw)
#     DST        = ~/.local/share/vmctl/images/openindiana-hipster-cloud-fixed.qcow2
#     ROOT_PUBKEY = tests/vm/.keys/root_ed25519.pub (run gen-keys.sh first
#                   if it or the matching .keys/root_ed25519 don't exist --
#                   this script does NOT generate the keypair itself, so the
#                   private half is never at risk of being silently
#                   regenerated out from under a running VMFile setup)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX="$HERE/illumos-image-fix"

SRC="${1:-$HOME/.local/share/vmctl/images/openindiana-hipster-cloud.img}"
DST="${2:-$HOME/.local/share/vmctl/images/openindiana-hipster-cloud-fixed.qcow2}"
ROOT_PUBKEY="${3:-$HERE/.keys/root_ed25519.pub}"
NBD="${NBD_DEV:-/dev/nbd1}"
MNT="$(mktemp -d)"
POOL=""  # discovered dynamically below (this image's pool is "oirpool",
         # but don't hardcode -- discover it the same way `zpool import -d`
         # would report it, in case a future image uses a different name)

log() { printf '[bake-hipster] %s\n' "$*"; }

cleanup() {
    sync || true
    if [ -n "$POOL" ]; then
        sudo zfs umount "$POOL/export/home" 2>/dev/null || true
        sudo zfs umount "$POOL/export" 2>/dev/null || true
        sudo zfs umount "$POOL/ROOT"/* 2>/dev/null || true
        sudo zpool export "$POOL" 2>/dev/null || true
    fi
    sudo qemu-nbd -d "$NBD" 2>/dev/null || true
    rmdir "$MNT" 2>/dev/null || true
}
trap cleanup EXIT

[ -f "$SRC" ] || { echo "source image not found: $SRC" >&2; exit 1; }
[ -f "$ROOT_PUBKEY" ] || {
    echo "root pubkey not found: $ROOT_PUBKEY -- run: ssh-keygen -t ed25519 -N '' -f tests/vm/.keys/root_ed25519" >&2
    exit 1
}

log "converting $SRC (raw) -> $DST (qcow2, standalone)"
qemu-img convert -f raw -O qcow2 "$SRC" "$DST"

log "loading nbd + attaching $DST on $NBD"
sudo modprobe nbd max_part=16
sudo qemu-nbd -c "$NBD" -f qcow2 "$DST"
sleep 1; sudo partprobe "$NBD" 2>/dev/null || true; sleep 1

log "discovering root pool on $NBD"
IMPORT_OUT="$(sudo zpool import -d /dev 2>&1)" || true
POOL="$(printf '%s\n' "$IMPORT_OUT" | awk '/^ *pool:/{print $2; exit}')"
[ -n "$POOL" ] || { echo "could not discover an importable pool on $NBD:\n$IMPORT_OUT" >&2; exit 1; }
log "found pool: $POOL"

log "importing $POOL (altroot $MNT)"
sudo zpool import -f -N -R "$MNT" "$POOL"
# Pick the actual boot environment DATASET under <pool>/ROOT, not the
# <pool>/ROOT container itself (which has canmount=off and can't mount).
BE="$(sudo zfs list -H -o name -r "$POOL/ROOT" | grep "^$POOL/ROOT/" | head -1)"
[ -n "$BE" ] || { echo "no boot environment found under $POOL/ROOT" >&2; exit 1; }
sudo zfs set mountpoint=/ "$BE"
sudo zfs mount "$BE"
# Mount the rest (/export etc.) best-effort -- the container dataset
# oirpool/ROOT has canmount=off and `zfs mount -a` returns non-zero for it,
# which is harmless: only the BE (mounted above) matters for the bake.
sudo zfs mount -a 2>/dev/null || true

# 1. Static root SSH key -- the login channel that does NOT depend on the
#    cloud-init seed or useragent ever running successfully.
sudo mkdir -p "$MNT/root/.ssh"
sudo cp "$ROOT_PUBKEY" "$MNT/root/.ssh/authorized_keys"
sudo chmod 700 "$MNT/root/.ssh"
sudo chmod 600 "$MNT/root/.ssh/authorized_keys"
sudo chown -R 0:0 "$MNT/root/.ssh"
log "installed root authorized_keys from $ROOT_PUBKEY"

# 2. sshd_config: PermitRootLogin no -> prohibit-password (key-only, no
#    password auth for root; PasswordAuthentication for other users is
#    untouched).
sudo cp "$MNT/etc/ssh/sshd_config" "$MNT/etc/ssh/sshd_config.orig-w1d.bak"
sudo sed -i 's/^PermitRootLogin no$/PermitRootLogin prohibit-password/' "$MNT/etc/ssh/sshd_config"
grep -q '^PermitRootLogin prohibit-password$' "$MNT/etc/ssh/sshd_config" \
    && log "sshd_config: PermitRootLogin -> prohibit-password" \
    || { echo "FAILED to patch sshd_config PermitRootLogin" >&2; exit 1; }

# 3. sudo -> pfexec shim (image ships no sudo(8); the exec1/pg/builder
#    provisioner scripts + smoke.sh all call `sudo`). Harmless when the SSH
#    user is already root: pfexec is a no-op passthrough for uid 0.
sudo cp "$FIX/sudo" "$MNT/usr/bin/sudo"
sudo chown root:bin "$MNT/usr/bin/sudo"
sudo chmod 555 "$MNT/usr/bin/sudo"
log "installed sudo->pfexec shim"

# 4. Declarative DHCP on the primary NIC -- THE networking fix (W1d root
#    cause). This image's root pool is named `oirpool`, not `rpool`, and
#    illumos's metadata agent (/usr/lib/metadata) fatally aborts on its very
#    first real step -- `zpool list rpool` -> "cannot open 'rpool': no such
#    pool" -- BEFORE it ever configures the network. It's the metadata agent
#    (not NWAM, which is inactive in this cloud image) that would otherwise
#    run `ipadm create-addr -T dhcp` on vioif0. Result: a first boot leaves
#    ipadm.conf empty, no /etc/hostname.*, no DHCP lease -- vioif0 has a
#    datalink but NO IP, so sshd (which DOES start) never answers the QEMU
#    host-forward ("Failed getting banner"). These two empty files make
#    svc:/network/physical:default DHCP vioif0 on its own at boot, entirely
#    independent of the (aborting) metadata agent. Both likely NIC names are
#    covered. See docs/ai/2026-07-06-phase50-w1c-image-defects.md (W1d).
for ifc in net0 vioif0; do
    sudo sh -c ": > '$MNT/etc/hostname.$ifc'"
    sudo sh -c ": > '$MNT/etc/dhcp.$ifc'"
    sudo chmod 644 "$MNT/etc/hostname.$ifc" "$MNT/etc/dhcp.$ifc"
    sudo chown root:root "$MNT/etc/hostname.$ifc" "$MNT/etc/dhcp.$ifc"
done
log "baked declarative /etc/hostname.{net0,vioif0} + /etc/dhcp.{net0,vioif0} (DHCP bypass for the pool-name metadata abort)"

log "done. $DST is ready; point VMFile.kdl's vm blocks at it with"
log "  ssh { user \"root\"; private-key \"$(basename "$(dirname "$ROOT_PUBKEY")")/$(basename "${ROOT_PUBKEY%.pub}")\" }"
