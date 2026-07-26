#!/usr/bin/env bash
# Phase 50 / W1 — generate the akh-exec SSH keypair used to test the 48c
# russh remote executor against exec1 (ADR 086 auth deviation: plain
# provisioned keys, not 48d OIDC->cert). Idempotent; run before `vmctl up`
# since the exec1 `file` provisioner uploads .keys/akh-exec_ed25519.pub at
# boot time. Keys are gitignored — never commit them.
#
# Phase 50 / W1d — also generates .keys/root_ed25519, the STATIC key baked
# directly into openindiana-hipster-cloud-fixed.qcow2's /root/.ssh/authorized_keys
# by scripts/bake-hipster-cloud-image.sh, and referenced by VMFile.kdl's
# `ssh { user "root"; private-key ".keys/root_ed25519" }` blocks + smoke.sh.
# This is the primary SSH-reachability channel (see VMFile.kdl's exec1
# comment for why) — unlike the auto-generated per-VM cloud-init keypair,
# it must be stable across `vmctl up` runs because it's baked into the
# image offline, not injected at boot, so this script never regenerates it
# once present.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.keys"
mkdir -p "$DIR"
chmod 700 "$DIR"

KEY="$DIR/akh-exec_ed25519"
if [ -f "$KEY" ]; then
    echo "[gen-keys] $KEY already exists, skipping"
else
    ssh-keygen -t ed25519 -N '' -C "akh-exec@phase50-w1" -f "$KEY"
    echo "[gen-keys] generated $KEY / $KEY.pub"
fi

ROOT_KEY="$DIR/root_ed25519"
if [ -f "$ROOT_KEY" ]; then
    echo "[gen-keys] $ROOT_KEY already exists, skipping"
else
    ssh-keygen -t ed25519 -N '' -C "root@phase50-w1d" -f "$ROOT_KEY"
    echo "[gen-keys] generated $ROOT_KEY / $ROOT_KEY.pub"
    echo "[gen-keys] NOTE: run scripts/bake-hipster-cloud-image.sh to bake $ROOT_KEY.pub into the image"
fi
