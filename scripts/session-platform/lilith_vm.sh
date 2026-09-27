#!/usr/bin/env bash
set -euo pipefail

VM_NAME=lilith-illumos-20260922
STATE_DIR=${LILITH_VM_STATE_DIR:-/home/toasty/.local/share/akh-medu/lilith-illumos-20260922}
BASE_IMAGE=${LILITH_VM_BASE_IMAGE:-$STATE_DIR/golden.qcow2}
EVIDENCE_DIR=/home/toasty/ws/agi/akh-medu/training/lilith/native-20260922
SSH_PORT=${LILITH_VM_SSH_PORT:-10115}
PID_FILE="$STATE_DIR/qemu.pid"

init_vm() {
  mkdir -p "$STATE_DIR"
  if [[ ! -f "$STATE_DIR/overlay.qcow2" ]]; then
    qemu-img create -f qcow2 -F qcow2 -b "$BASE_IMAGE" "$STATE_DIR/overlay.qcow2"
  fi
  if [[ ! -f "$STATE_DIR/efivars.fd" ]]; then
    cp /usr/share/OVMF/OVMF_VARS.fd "$STATE_DIR/efivars.fd"
  fi
}

is_running() {
  [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start_vm() {
  init_vm
  if is_running; then
    echo "$VM_NAME already running (pid $(cat "$PID_FILE"))"
    return
  fi
  rm -f "$STATE_DIR/qmp.sock"
  qemu-system-x86_64 \
    -name "$VM_NAME" -enable-kvm -machine pc,accel=kvm -cpu host \
    -smp 2 -m 4096M -nodefaults \
    -qmp "unix:$STATE_DIR/qmp.sock,server=on,wait=off" \
    -chardev "file,id=serial0,path=$STATE_DIR/console.log,append=on" -serial chardev:serial0 \
    -display none -device virtio-rng-pci \
    -drive "file=$STATE_DIR/overlay.qcow2,format=qcow2,if=none,id=drive0,discard=unmap" \
    -device virtio-blk-pci,drive=drive0,disable-modern=on,disable-legacy=off \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd \
    -drive "if=pflash,format=raw,file=$STATE_DIR/efivars.fd" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$SSH_PORT-:22" \
    -device virtio-net-pci,netdev=net0,mac=52:54:4c:49:4c:49,disable-modern=on,disable-legacy=off \
    -daemonize -pidfile "$PID_FILE"
  echo "$VM_NAME started (pid $(cat "$PID_FILE"), ssh 127.0.0.1:$SSH_PORT)"
}

stop_vm() {
  if ! is_running; then
    echo "$VM_NAME is stopped"
    return
  fi
  printf '%s\n%s\n' '{"execute":"qmp_capabilities"}' '{"execute":"system_powerdown"}' \
    | socat - "UNIX-CONNECT:$STATE_DIR/qmp.sock" >/dev/null
  for _ in $(seq 1 30); do
    is_running || { echo "$VM_NAME stopped"; return; }
    sleep 1
  done
  echo "$VM_NAME did not stop within 30 seconds" >&2
  return 1
}

status_vm() {
  if is_running; then
    echo "$VM_NAME running pid=$(cat "$PID_FILE") ssh=127.0.0.1:$SSH_PORT"
  else
    echo "$VM_NAME stopped ssh=127.0.0.1:$SSH_PORT"
  fi
  qemu-img info --force-share --backing-chain "$STATE_DIR/overlay.qcow2"
}

case "${1:-}" in
  init) init_vm ;;
  start) start_vm ;;
  stop) stop_vm ;;
  status) status_vm ;;
  *) echo "usage: $0 {init|start|stop|status}" >&2; exit 2 ;;
esac
