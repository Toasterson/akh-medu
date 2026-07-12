#!/bin/sh
#
# Minimal, dependency-free Forgejo Release helper (no jq/python — runs on the
# Linux *and* illumos runners). Talks to the Forgejo REST API.
#
# Subcommands:
#   ensure <tag> [name] [body]   Get-or-create the release for <tag>; print its id.
#   upload <tag> <file> [name]   Ensure the release, then (re)upload <file> as an
#                                asset, replacing any existing asset of that name.
#
# Configuration via environment:
#   FORGEJO_API     default https://code.aopc.cloud/api/v1
#   FORGEJO_OWNER   default toasterson
#   FORGEJO_REPO    default akh-medu
#   FORGEJO_TOKEN   PAT with write:repository (falls back to RELEASE_TOKEN /
#                   REGISTRY_TOKEN / GITHUB_TOKEN)

set -eu

API=${FORGEJO_API:-https://code.aopc.cloud/api/v1}
OWNER=${FORGEJO_OWNER:-toasterson}
REPO=${FORGEJO_REPO:-akh-medu}
TOKEN=${FORGEJO_TOKEN:-${RELEASE_TOKEN:-${REGISTRY_TOKEN:-${GITHUB_TOKEN:-}}}}

if [ -z "$TOKEN" ]; then
	echo "forgejo-release: no API token (set FORGEJO_TOKEN)" >&2
	exit 1
fi

AUTH="Authorization: token $TOKEN"
BASE="$API/repos/$OWNER/$REPO"

# Extract the first top-level "id": value from a JSON blob.
# Two illumos/POSIX grep pitfalls that GNU grep tolerates (both bit the
# illumos runner): a final line without a trailing newline is ignored, and
# lines longer than LINE_MAX (~2k; the release JSON is ~4k) never match.
# Terminate the stream and split it into short lines before grepping.
first_id() {
	{ cat; echo; } | tr ',{' '\n\n' | grep -o '"id":[0-9]*' | head -1 | grep -o '[0-9][0-9]*'
}

ensure_release() {
	tag=$1
	name=${2:-$tag}
	body=${3:-"Automated release $tag."}

	# Existing release for this tag?
	existing=$(curl -fsS -H "$AUTH" "$BASE/releases/tags/$tag" 2>/dev/null || true)
	if [ -n "$existing" ]; then
		printf '%s' "$existing" | first_id
		return 0
	fi

	# Create it. Escape the body's quotes/backslashes for JSON embedding.
	# The bare assignment must not abort under `set -e` on a transient 5xx or a
	# concurrent-create 409 — fall back to re-fetching the release by tag.
	esc_body=$(printf '%s' "$body" | sed 's/\\/\\\\/g; s/"/\\"/g')
	created=$(curl -fsS -X POST -H "$AUTH" -H 'Content-Type: application/json' \
		-d "{\"tag_name\":\"$tag\",\"name\":\"$name\",\"body\":\"$esc_body\",\"draft\":false,\"prerelease\":false}" \
		"$BASE/releases" 2>/dev/null) || created=""
	if [ -z "$created" ]; then
		# Maybe created concurrently by another job, or a transient error — retry the GET.
		created=$(curl -fsS -H "$AUTH" "$BASE/releases/tags/$tag" 2>/dev/null || true)
	fi
	if [ -z "$created" ]; then
		echo "forgejo-release: could not create or fetch release for $tag" >&2
		return 1
	fi
	printf '%s' "$created" | first_id
}

upload_asset() {
	tag=$1
	file=$2
	name=${3:-$(basename "$file")}

	[ -f "$file" ] || { echo "forgejo-release: file not found: $file" >&2; exit 1; }

	rel_id=$(ensure_release "$tag")
	[ -n "$rel_id" ] || { echo "forgejo-release: could not resolve release id for $tag" >&2; exit 1; }

	# Replace an existing asset of the same name (idempotent re-runs).
	old_id=$({ curl -fsS -H "$AUTH" "$BASE/releases/$rel_id/assets" 2>/dev/null; echo; } \
		| tr '}' '\n' | grep "\"name\":\"$name\"" | { cat; echo; } | tr ',{' '\n\n' | grep -o '"id":[0-9]*' | head -1 | grep -o '[0-9][0-9]*' || true)
	if [ -n "$old_id" ]; then
		curl -fsS -X DELETE -H "$AUTH" "$BASE/releases/$rel_id/assets/$old_id" >/dev/null || true
		echo "    replaced existing asset $name"
	fi

	curl -fsS -X POST -H "$AUTH" \
		-F "attachment=@$file;filename=$name" \
		"$BASE/releases/$rel_id/assets?name=$name" >/dev/null
	echo "    uploaded $name to release $tag (id $rel_id)"
}

cmd=${1:-}
case "$cmd" in
ensure) shift; ensure_release "$@" ;;
upload) shift; upload_asset "$@" ;;
*) echo "usage: $0 {ensure <tag> [name] [body] | upload <tag> <file> [name]}" >&2; exit 2 ;;
esac
