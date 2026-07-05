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

PG_PKG=$(pkg list -a 2>/dev/null | awk '{print $1}' \
    | grep -E '^(database/)?postgresql-[0-9]+$' \
    | sed 's#.*/##' \
    | sort -t- -k2 -n -u \
    | tail -1)

if [ -z "${PG_PKG:-}" ]; then
    fail "no database/postgresql-<N> package found in the IPS catalog (pkg list -a | grep postgres for the real name)"
    pkg list -a 2>/dev/null | grep -i postgres || true
    exit 1
fi

log "installing database/${PG_PKG}"
sudo pkg install -v "database/${PG_PKG}" 2>&1 || {
    fail "pkg install database/${PG_PKG} failed"
    exit 1
}

# Postgres dev headers (pg_config, server headers) needed to build pgvector.
sudo pkg install -v "database/${PG_PKG%%-*}-${PG_PKG#*-}" 2>&1 || true
for cand in "database/postgresql-dev" "database/postgresql-${PG_PKG#postgresql-}-dev" "developer/build/gnu-make" "developer/gcc-13" "developer/gcc-14"; do
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
