#!/usr/bin/env bash
# Phase 50 / exec1 — direct QEMU launcher for the toolchain-baked OpenIndiana
# guest, bypassing vmctl entirely.
#
# WHY NOT vmctl: vmctl's QEMU backend hardcodes `-machine q35,accel=kvm`
# (see ~/ws/nebula/vm-manager/crates/vm-manager/src/backends/qemu.rs). On
# q35's PCIe root complex, illumos's virtio-blk driver (vioblk) cannot map
# the modern virtio PCI capability and the kernel panics trying to mount
# root ZFS. The qemu-illumos-shim/ wrapper in this directory works around
# that for vmctl-driven flows by rewriting q35->pc on the fly, but this
# script instead invokes qemu directly with `-machine pc,accel=kvm` from the
# start — no shim needed. Everything else here mirrors vmctl's proven flag
# shape (OVMF pflash pair, virtio-blk via drive0/device split, user-mode net
# with hostfwd) so the guest sees the same device topology it was
# provisioned under.
#
# WHY 6G RAM: `-m 6144` (6GB). The gcc 14.3 + rustc 1.96 `pkg update` /
# toolchain bake OOMed the guest at the default 4GB — see
# docs/ai/plans/2026-07-05-phase50-feature-test-benchmarks.md. Don't go
# below 6144 unless you've confirmed the workload is lighter.
#
# KNOWN BOOT QUIRK #1 — UEFI shell instead of autoboot:
#   OVMF sometimes drops to the EFI Interactive Shell (`Shell>`) instead of
#   walking the guest's boot order into the OpenIndiana loader. If that
#   happens, drive the serial console (see below) and type, followed by
#   Enter after each line:
#     FS0:
#     \EFI\Boot\bootx64.efi
#   Give OVMF 15-30s after power-on before concluding it's stuck; capture
#   the console log first (tail -f "$CONSOLE_LOG").
#
# KNOWN BOOT QUIRK #2 — NIC enumerates as net0, DHCP config expects vioif0:
#   The baked guest's DHCP config lives at /etc/hostname.vioif0 +
#   /etc/dhcp.vioif0 (illumos legacy device-name style). If GLDv3 assigns
#   the modern generic datalink name `net0` to the vioif NIC instead
#   (`dladm show-link` will confirm), those files never get applied and you
#   see on boot:
#     Failed to plumb IPv4 interface(s): net0
#   and SSH is unreachable. Fix over the serial console (root login, try
#   empty password, then "solaris", then "openindiana"):
#     ipadm create-if vioif0 2>/dev/null; ipadm create-if net0 2>/dev/null
#     ipadm create-addr -T dhcp vioif0/dhcp 2>/dev/null || \
#       ipadm create-addr -T dhcp net0/dhcp
#   or persist it by creating /etc/hostname.net0 + /etc/dhcp.net0 (copy the
#   vioif0 ones) and rebooting. If serial login is impossible, shut qemu
#   down and fix it offline via qemu-nbd against the overlay's zpool the way
#   tests/vm/scripts/bake-hipster-cloud-image.sh does (read that script
#   first — remember to `zpool export` before disconnecting nbd).
#
# The guest metadata agent's `cannot open 'rpool'` error at boot is KNOWN
# and harmless — the pool is actually named `oirpool`, not `rpool`.
#
# USAGE:
#   tests/vm/scripts/launch-exec1-pc6g.sh              # launch (daemonized)
#   tests/vm/scripts/launch-exec1-pc6g.sh status        # check if running
#   tests/vm/scripts/launch-exec1-pc6g.sh console       # attach to serial via socat (Ctrl+] to detach... actually Ctrl+C to quit socat)
#   tests/vm/scripts/launch-exec1-pc6g.sh type "FS0:"   # send a line to the EFI shell over serial
#   tests/vm/scripts/launch-exec1-pc6g.sh stop          # graceful ACPI powerdown via monitor, falls back to kill
#
# SSH once booted:
#   ssh -F /dev/null -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
#       -i tests/vm/.keys/root_ed25519 -p 10042 root@127.0.0.1

set -euo pipefail

VM_NAME="exec1"
VM_DIR="${HOME}/.local/share/vmctl/vms/${VM_NAME}"
OVMF_CODE="/usr/share/OVMF/OVMF_CODE.fd"
EFIVARS="${VM_DIR}/efivars.fd"
OVERLAY="${VM_DIR}/overlay.qcow2"
CONSOLE_LOG="${VM_DIR}/console.log"
SERIAL_SOCK="/tmp/exec1-serial.sock"
MONITOR_SOCK="/tmp/exec1-mon.sock"
QMP_SOCK="/tmp/exec1-qmp.sock"
PIDFILE="${VM_DIR}/qemu.pid"
SSH_PORT=10042
MEM_MB=6144
SMP=2
MAC="52:54:00:ae:c1:01"

QEMU_BIN="$(command -v qemu-system-x86_64)"

die() { echo "launch-exec1-pc6g: $*" >&2; exit 1; }

cmd_start() {
    [ -f "$OVERLAY" ] || die "overlay not found at $OVERLAY (do NOT recreate it — it's toolchain-baked; check the path)"
    [ -f "$OVMF_CODE" ] || die "OVMF_CODE not found at $OVMF_CODE"
    [ -f "$EFIVARS" ] || die "efivars.fd not found at $EFIVARS (pairs with $OVMF_CODE)"

    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "already running: pid $(cat "$PIDFILE")"
        exit 0
    fi

    for s in "$SERIAL_SOCK" "$MONITOR_SOCK" "$QMP_SOCK"; do
        [ -S "$s" ] && rm -f "$s"
    done

    "$QEMU_BIN" \
        -name "$VM_NAME" \
        -machine pc,accel=kvm \
        -cpu host \
        -smp "$SMP" \
        -m "$MEM_MB" \
        -nodefaults \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -drive if=pflash,format=raw,file="$EFIVARS" \
        -drive file="$OVERLAY",format=qcow2,if=none,id=drive0,discard=unmap \
        -device virtio-blk-pci,drive=drive0 \
        -device virtio-rng-pci \
        -netdev user,id=net0,hostfwd=tcp:127.0.0.1:${SSH_PORT}-:22 \
        -device virtio-net-pci,netdev=net0,mac="$MAC" \
        -chardev socket,id=serial0,path="$SERIAL_SOCK",server=on,wait=off,logfile="$CONSOLE_LOG" \
        -serial chardev:serial0 \
        -monitor unix:"$MONITOR_SOCK",server,nowait \
        -qmp unix:"$QMP_SOCK",server,nowait \
        -display none \
        -daemonize \
        -pidfile "$PIDFILE"

    sleep 1
    echo "started: pid $(cat "$PIDFILE" 2>/dev/null || echo '?')"
    echo "serial:  $SERIAL_SOCK  (socat - UNIX-CONNECT:$SERIAL_SOCK to attach)"
    echo "monitor: $MONITOR_SOCK"
    echo "qmp:     $QMP_SOCK"
    echo "ssh:     ssh -F /dev/null -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -i tests/vm/.keys/root_ed25519 -p ${SSH_PORT} root@127.0.0.1"
    echo "console log: $CONSOLE_LOG"
}

cmd_status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "running: pid $(cat "$PIDFILE")"
    else
        echo "not running"
    fi
}

cmd_console() {
    echo "attaching to serial console at $SERIAL_SOCK — Ctrl+C to detach (does not stop the VM)" >&2
    socat -,raw,echo=0,escape=0x1d UNIX-CONNECT:"$SERIAL_SOCK"
}

cmd_type() {
    local line="$1"
    printf '%s\r\n' "$line" | socat - UNIX-CONNECT:"$SERIAL_SOCK"
}

cmd_stop() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "system_powerdown" | socat - UNIX-CONNECT:"$MONITOR_SOCK" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "stopped"; return 0; }
            sleep 1
        done
        echo "ACPI powerdown timed out, sending SIGTERM"
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
    else
        echo "not running"
    fi
}

case "${1:-start}" in
    start) cmd_start ;;
    status) cmd_status ;;
    console) cmd_console ;;
    type) shift; cmd_type "$*" ;;
    stop) cmd_stop ;;
    *) die "unknown subcommand: $1 (use start|status|console|type|stop)" ;;
esac
