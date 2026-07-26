#!/usr/bin/env bash
# Phase 50 / W1 — `pg` VM provisioner.
#
# Installs postgres + pgvector on OpenIndiana so Phase-38 relocation/WAL/
# split-brain tests get a real network-boundary postgres (see ADR 086 §1).
# Best-effort + defensive: package names/paths are discovered at runtime
# rather than hardcoded, because the exact IPS catalog on the
# openindiana-rust-ci image was not known in advance (documented in the W1
# shakeout report if any step below fails).
set -uo pipefail

log() { printf '[pg-provision] %s\n' "$*"; }
fail() { printf '[pg-provision] FAIL: %s\n' "$*" >&2; }

log "illumos release: $(uname -v 2>/dev/null || true)"

# --- discover + install postgresql via IPS ------------------------------
sudo pkg refresh 2>&1 || log "WARN: pkg refresh failed (publisher unreachable?)"

# W1d: OpenIndiana ships PostgreSQL as `database/postgres-<NN>` (NOT
# `postgresql-<NN>` — that earlier guess produced a false "no package
# found" FAIL even though DNS + the catalog were fine). The <NN> suffix
# encodes the version: modern majors are 10..17, while the legacy trains
# are 82/83/.../96 (= 8.2 .. 9.6). Prefer the newest MODERN major (suffix
# in 10..79) so a plain numeric sort doesn't pick "96" (9.6) over "12".
PG_SUFFIX=$(pkg list -a 2>/dev/null | awk '{print $1}' \
    | grep -E '^database/postgres-[0-9]+$' \
    | sed 's#.*/postgres-##' \
    | awk '$1 >= 10 && $1 < 80' \
    | sort -n | tail -1)
if [ -z "${PG_SUFFIX:-}" ]; then
    # No modern major in the catalog; fall back to the newest legacy train.
    PG_SUFFIX=$(pkg list -a 2>/dev/null | awk '{print $1}' \
        | grep -E '^database/postgres-[0-9]+$' \
        | sed 's#.*/postgres-##' | sort -n | tail -1)
fi

if [ -z "${PG_SUFFIX:-}" ]; then
    fail "no database/postgres-<NN> package found in the IPS catalog (pkg list -a | grep postgres for the real name)"
    pkg list -a 2>/dev/null | grep -i postgres || true
    exit 1
fi
PG_PKG="postgres-${PG_SUFFIX}"

log "installing database/${PG_PKG} (+ /developer for pg_config, /library)"
sudo pkg install -v "database/${PG_PKG}" 2>&1 || {
    fail "pkg install database/${PG_PKG} failed"
    exit 1
}

# Postgres server dev headers (pg_config) + runtime libs needed to build
# pgvector. On OI these are subpackages of the versioned postgres package.
for cand in \
    "database/${PG_PKG}/developer" \
    "database/${PG_PKG}/library" \
    "developer/build/gnu-make" \
    "developer/gcc-13" "developer/gcc-14" \
; do
    sudo pkg install -v "$cand" 2>&1 || true
done

PG_CONFIG=$(find /usr/postgres /usr/gnu -maxdepth 3 -name pg_config 2>/dev/null | sort -V | tail -1)
if [ -z "${PG_CONFIG:-}" ]; then
    fail "pg_config not found under /usr/postgres after install; cannot build pgvector"
else
    log "using pg_config: $PG_CONFIG ($($PG_CONFIG --version 2>&1))"
fi

# --- enable the SMF service ---------------------------------------------
PG_FMRI=$(svcs -aH -o fmri 2>/dev/null | grep -i postgresql | head -1)
if [ -n "${PG_FMRI:-}" ]; then
    log "enabling $PG_FMRI"
    sudo svcadm enable -rs "$PG_FMRI" 2>&1 || fail "svcadm enable $PG_FMRI failed"
    svcs "$PG_FMRI" 2>&1 || true
else
    fail "no postgresql SMF service found (svcs -a | grep postgres); the package may need manual initdb"
fi

# --- pgvector ------------------------------------------------------------
if [ -n "${PG_CONFIG:-}" ]; then
    if [ ! -d /tmp/pgvector ]; then
        git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector 2>&1 || fail "git clone pgvector failed (no network egress to github.com?)"
    fi
    if [ -d /tmp/pgvector ]; then
        ( cd /tmp/pgvector && make PG_CONFIG="$PG_CONFIG" 2>&1 && sudo make PG_CONFIG="$PG_CONFIG" install 2>&1 ) \
            || fail "pgvector build/install failed"
    fi
fi

# --- smoke: create a test db + CREATE EXTENSION vector -------------------
sleep 2
sudo -u postgres psql -c "SELECT version();" 2>&1 || fail "psql -c 'SELECT version()' failed — postgres may not be accepting connections yet"
sudo -u postgres psql -c "CREATE DATABASE akh_vm_smoke;" 2>&1 || log "createdb akh_vm_smoke: non-fatal (may already exist)"
sudo -u postgres psql -d akh_vm_smoke -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1 || fail "CREATE EXTENSION vector failed"

log "pg provisioning done"
