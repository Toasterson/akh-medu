#!/usr/bin/env bash
# Scheduled fetch for the Chief-of-Staff guard's dedicated ancestry clone
# (ADR 116 Decision 1 revision 4; src/agent/chief_of_staff/policy.rs
# `GitAncestry`).
#
# `GitAncestry::ancestry` runs `git -C <repository> rev-parse --verify` and
# `git -C <repository> merge-base --is-ancestor <sha> <main-ref>` against
# whatever refs already exist in that clone -- it never fetches. Left alone,
# the clone's `refs/remotes/origin/main` goes stale and a legitimately
# main-landed deploy/restart SHA looks like a non-ancestor: every such
# operator-class action is refused (`CosError::NonMainArtifact`). This script
# is the missing scheduled fetch.
#
# Usage:
#   ancestry_fetch.sh <clone-dir> [remote-url]
#
# <clone-dir> must match the `[chief_of_staff].repository` path in the akh's
# config. If it does not exist yet, this script bootstraps it as a bare
# mirror clone (bare: the guard only ever reads refs and commit objects
# through `git -C`, never a working tree, and a bare repo cannot be
# accidentally left dirty by something else touching the directory).
#
# A portable mkdir-based lock at "<clone-dir>.fetch.lock" guards against two
# fetches overlapping (macOS ships no flock(1); mkdir is atomic on every
# filesystem this runs on). A lock whose owning pid is no longer alive is
# treated as stale and cleared automatically.
set -euo pipefail
umask 077

CLONE_DIR=${1:?"usage: ancestry_fetch.sh <clone-dir> [remote-url]"}
REMOTE_URL=${2:-https://code.aopc.cloud/toasterson/akh-medu.git}
MAIN_REF=${3:-main}
CLONE_DIR=${CLONE_DIR%/}
LOCK_DIR="${CLONE_DIR}.fetch.lock"

log() { printf '%s ancestry_fetch[%s] %s\n' "$(date -u +%FT%TZ)" "$$" "$*"; }

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  local owner=""
  [ -f "$LOCK_DIR/pid" ] && owner=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    log "another fetch is already running (pid $owner); skipping this run"
    return 1
  fi
  log "clearing a stale lock (owner pid ${owner:-unknown} is not running)"
  rm -rf "$LOCK_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  log "could not acquire the lock even after clearing a stale one; skipping this run"
  return 1
}

release_lock() {
  # Only release a lock this process actually holds -- a lock acquired by a
  # concurrent run after we gave up must never be removed by us.
  if [ "${LOCK_HELD:-0}" = 1 ]; then
    rm -rf "$LOCK_DIR"
  fi
}

LOCK_HELD=0
if acquire_lock; then
  LOCK_HELD=1
else
  exit 0   # a live fetch already owns the lock; not an error, just a skip
fi
trap release_lock EXIT

bootstrap_if_missing() {
  if [ -d "$CLONE_DIR" ]; then
    return 0
  fi
  log "bootstrapping a bare mirror clone at $CLONE_DIR from $REMOTE_URL"
  mkdir -p "$(dirname "$CLONE_DIR")"
  git clone --bare "$REMOTE_URL" "$CLONE_DIR"
}

ensure_origin_remote() {
  # `git clone --bare` does not configure a remote named "origin" (that is
  # only done for a non-bare clone), and an operator-restored clone might
  # have none either -- add or repair it before the fetch that needs it.
  local current
  current=$(git -C "$CLONE_DIR" remote get-url origin 2>/dev/null || true)
  if [ -z "$current" ]; then
    log "configuring remote 'origin' -> $REMOTE_URL"
    git -C "$CLONE_DIR" remote add origin "$REMOTE_URL"
  elif [ "$current" != "$REMOTE_URL" ]; then
    log "remote 'origin' points at $current, expected $REMOTE_URL; repairing"
    git -C "$CLONE_DIR" remote set-url origin "$REMOTE_URL"
  fi
}

bootstrap_if_missing

if ! git -C "$CLONE_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  log "refusing: $CLONE_DIR exists but is not a git repository"
  exit 1
fi

ensure_origin_remote

log "fetching origin/$MAIN_REF into $CLONE_DIR"
if git -C "$CLONE_DIR" fetch --prune origin "+refs/heads/${MAIN_REF}:refs/remotes/origin/${MAIN_REF}"; then
  sha=$(git -C "$CLONE_DIR" rev-parse --verify --quiet "refs/remotes/origin/${MAIN_REF}" || echo unknown)
  log "fetch ok; refs/remotes/origin/${MAIN_REF} -> $sha"
  exit 0
fi

log "fetch FAILED; the guard's ancestry check keeps using the last successful fetch until this recovers"
exit 1
