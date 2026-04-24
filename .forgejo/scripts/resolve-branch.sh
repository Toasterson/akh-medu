#!/bin/sh
# Print the current branch name on stdout, or empty if none can be resolved.
#
# Motivation: Solstice (Forgejo Actions runner) does not reliably populate
# GITHUB_REF / GITHUB_REF_NAME — previous workflow runs that gated branch-
# tag emission on those env vars silently emitted only the short-sha tag.
#
# Priority order:
#   1. GITHUB_REF_NAME (GitHub Actions + newer Forgejo)
#   2. GITHUB_REF (strip refs/heads/…)
#   3. Common SOLSTICE_* envs, in case Solstice grows them
#   4. `git name-rev --name-only HEAD` on the checked-out repo
#      (Solstice checks out the sha as detached HEAD, with origin/<branch>
#      pointing at the same commit — name-rev finds that ref).
#
# Never fails — prints empty string if nothing resolves. Callers should
# decide what to do with an empty result.

set -eu

if [ -n "${GITHUB_REF_NAME:-}" ]; then
    printf '%s' "$GITHUB_REF_NAME"
    exit 0
fi

case "${GITHUB_REF:-}" in
    refs/heads/*)
        printf '%s' "${GITHUB_REF#refs/heads/}"
        exit 0
        ;;
esac

for var in SOLSTICE_REF_NAME SOLSTICE_BRANCH SOLSTICE_HEAD_REF; do
    val=$(eval "printf '%s' \"\${$var:-}\"")
    if [ -n "$val" ]; then
        printf '%s' "$val"
        exit 0
    fi
done

if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
    val=$(git name-rev --name-only --no-undefined HEAD 2>/dev/null \
        | sed 's|^remotes/origin/||; s|~.*$||; s|\^.*$||') || val=""
    if [ -n "$val" ] && [ "$val" != "undefined" ]; then
        printf '%s' "$val"
        exit 0
    fi
fi

# No resolver matched — print empty, caller decides.
printf ''
