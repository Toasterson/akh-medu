#!/usr/bin/env bash
# Phase 50 / W1 — exec1 provisioner.
#
# Creates the unprivileged `akh-exec` user that the Phase-48 executor
# (48b LocalDrop / 48c russh remote) targets, and makes sure sshd is up so
# the host can reach this guest over real SSH (ADR 086 auth deviation: plain
# provisioned keys, not OIDC->cert — 48d stays deferred/production-only).
#
# Idempotent: safe to re-run via `vmctl provision --name exec1`.
set -uo pipefail

log() { printf '[exec1-provision] %s\n' "$*"; }

USERNAME="akh-exec"
HOME_DIR="/export/home/${USERNAME}"

log "illumos release: $(uname -v 2>/dev/null || true)"

# --- sshd -------------------------------------------------------------
# OpenIndiana ships openssh as the system ssh service (network/ssh, not
# network/ssh/keydist). Make sure it's enabled so the host can connect for
# both provisioning (already true, we're running over it) and the 48c
# executor tests.
if command -v svcadm >/dev/null 2>&1; then
    sudo svcadm enable -rs network/ssh 2>&1 || log "WARN: could not enable network/ssh (may already be enabled)"
    svcs network/ssh 2>&1 || true
fi

# --- akh-exec unprivileged user ----------------------------------------
if id "$USERNAME" >/dev/null 2>&1; then
    log "user $USERNAME already exists"
else
    log "creating user $USERNAME"
    sudo mkdir -p /export/home
    # -m creates the home dir; illumos useradd defaults to /bin/sh.
    sudo useradd -m -d "$HOME_DIR" -s /bin/bash "$USERNAME" \
        || sudo useradd -m -d "$HOME_DIR" -s /bin/sh "$USERNAME"
    # Lock the password (key-only auth); illumos passwd -N sets NP (no login
    # via password) without locking the account outright like -l does.
    sudo passwd -N "$USERNAME" 2>&1 || true
fi

sudo mkdir -p "${HOME_DIR}/.ssh"
if [ -f /tmp/akh-exec.pub ]; then
    sudo cp /tmp/akh-exec.pub "${HOME_DIR}/.ssh/authorized_keys"
    sudo rm -f /tmp/akh-exec.pub
    log "installed authorized_keys for $USERNAME from uploaded pubkey"
else
    log "WARN: /tmp/akh-exec.pub not found; akh-exec has no authorized_keys yet"
fi
sudo chmod 700 "${HOME_DIR}/.ssh"
sudo chmod 600 "${HOME_DIR}/.ssh/authorized_keys" 2>/dev/null || true
sudo chown -R "${USERNAME}:${USERNAME}" "${HOME_DIR}/.ssh" 2>&1 || true

# --- RBAC confinement posture (ADR 082 §confinement) --------------------
# akh-exec should be a genuinely unprivileged illumos account: no profiles,
# no sudo rights, so 48b LocalDrop's "OS is the sole security boundary" claim
# is testable. Report what it can/can't do rather than asserting further
# privileges here.
log "akh-exec profiles: $(sudo profiles "$USERNAME" 2>&1 | tr '\n' ';' || true)"

log "exec1 provisioning done"
