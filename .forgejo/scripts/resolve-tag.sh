#!/bin/sh
# Print the release tag (e.g. v0.7.0) on stdout, or empty if none resolves.
#
# Companion to resolve-branch.sh — Solstice (Forgejo Actions) does not reliably
# populate GITHUB_REF / GITHUB_REF_NAME, and `${{ inputs.* }}` / `${{ secrets.* }}`
# substitution inside step `env:` blocks is inconsistent. Resolution order:
#   1. RELEASE_TAG env (set this for manual / workflow_dispatch runs)
#   2. GITHUB_REF_NAME (when it looks like a tag)
#   3. GITHUB_REF (strip refs/tags/)
#   4. SOLSTICE_* tag envs, if present
#   5. `git describe --tags --exact-match` on the checked-out commit
# Never fails — prints empty string if nothing resolves.
set -eu

case "${RELEASE_TAG:-}" in
	v*) printf '%s' "$RELEASE_TAG"; exit 0 ;;
esac

case "${GITHUB_REF_NAME:-}" in
	v*) printf '%s' "$GITHUB_REF_NAME"; exit 0 ;;
esac

case "${GITHUB_REF:-}" in
	refs/tags/*) printf '%s' "${GITHUB_REF#refs/tags/}"; exit 0 ;;
esac

for var in SOLSTICE_REF_NAME SOLSTICE_TAG SOLSTICE_REF; do
	val=$(eval "printf '%s' \"\${$var:-}\"")
	case "$val" in
		refs/tags/*) printf '%s' "${val#refs/tags/}"; exit 0 ;;
		v*) printf '%s' "$val"; exit 0 ;;
	esac
done

if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
	t=$(git describe --tags --exact-match 2>/dev/null || true)
	case "$t" in
		v*) printf '%s' "$t"; exit 0 ;;
	esac
fi

printf ''
