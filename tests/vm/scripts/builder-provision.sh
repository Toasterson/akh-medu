#!/usr/bin/env bash
# Phase 50 / W1 — `builder` VM provisioner.
#
# openindiana-rust-ci already ships a Rust toolchain (that's the point of the
# image); this provisioner just verifies it and best-effort installs the
# native toolchain bits akh-medu's default feature set needs to compile
# (candle-backend pulls in `cc`-driven C/C++ builds for tokenizers'
# esaxx-rs). No repo checkout here — that's the W1 shakeout step, run
# separately (large, and we want it re-runnable without re-provisioning).
set -uo pipefail

log() { printf '[builder-provision] %s\n' "$*"; }

log "illumos release: $(uname -v 2>/dev/null || true)"
log "rustc: $(rustc -Vv 2>&1 || echo 'NOT FOUND')"
log "cargo: $(cargo -V 2>&1 || echo 'NOT FOUND')"

sudo pkg refresh 2>&1 || log "WARN: pkg refresh failed (publisher unreachable?)"

for pkg in \
    developer/gcc-13 developer/gcc-14 \
    developer/build/gnu-make \
    developer/build/pkg-config \
    developer/versioning/git \
    archiver/gnu-tar \
    web/curl \
    web/ca-certificates \
    library/zlib \
    developer/clang \
; do
    sudo pkg install -v "$pkg" 2>&1 || true
done

if ! command -v rustc >/dev/null 2>&1; then
    log "WARN: rustc not on PATH; installing via rustup as a fallback"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 || true
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env" 2>/dev/null || true
fi

log "final toolchain check:"
rustc -Vv 2>&1 || log "FAIL: rustc still not available"
cargo -V 2>&1 || log "FAIL: cargo still not available"
cc --version 2>&1 | head -1 || log "WARN: no C compiler on PATH"

log "builder provisioning done"
