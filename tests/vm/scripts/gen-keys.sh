#!/usr/bin/env bash
# Phase 50 / W1 — generate the akh-exec SSH keypair used to test the 48c
# russh remote executor against exec1 (ADR 086 auth deviation: plain
# provisioned keys, not 48d OIDC->cert). Idempotent; run before `vmctl up`
# since the exec1 `file` provisioner uploads .keys/akh-exec_ed25519.pub at
# boot time. Keys are gitignored — never commit them.
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
