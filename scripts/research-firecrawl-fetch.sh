#!/usr/bin/env bash
# research-firecrawl-fetch.sh — scrape a URL (incl. PDFs) via Firecrawl, emit clean markdown.
#
# Used by the deep-research workflow's fetch stage in place of the built-in WebFetch,
# which fails to extract content from many academic PDFs. Firecrawl parses PDFs by default.
#
# Usage:   scripts/research-firecrawl-fetch.sh <url>
# Key:     read from $FIRECRAWL_API_KEY, else from ~/.config/firecrawl/key (chmod 600).
# Output:  clean markdown on stdout; diagnostics on stderr; nonzero exit on failure.
#
# The key is never echoed. Prefer the keyfile so the secret stays out of shell history
# and process listings.
set -euo pipefail

URL="${1:-}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <url>" >&2
  exit 2
fi

KEY="${FIRECRAWL_API_KEY:-}"
if [[ -z "$KEY" && -r "$HOME/.config/firecrawl/key" ]]; then
  KEY="$(< "$HOME/.config/firecrawl/key")"
fi
if [[ -z "$KEY" ]]; then
  echo "error: no Firecrawl API key (set \$FIRECRAWL_API_KEY or write ~/.config/firecrawl/key)" >&2
  exit 3
fi

# v2/scrape: markdown only, follow main content, parse PDFs (default true), generous timeout.
RESP="$(curl -sS -m 90 -X POST "https://api.firecrawl.dev/v2/scrape" \
  -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -d "$(URL="$URL" python3 -c 'import json,os; print(json.dumps({"url":os.environ["URL"],"formats":["markdown"],"onlyMainContent":True,"timeout":85000}))')" \
  )" || { echo "error: curl to Firecrawl failed for $URL" >&2; exit 4; }

# Parse: print data.markdown on success, surface the API error otherwise.
# RESP is passed via env (not stdin — stdin here is the python program itself).
RESP="$RESP" URL="$URL" python3 <<'PY'
import json, os, sys
raw = os.environ["RESP"]
try:
    obj = json.loads(raw)
except Exception as e:
    sys.stderr.write(f"error: non-JSON response from Firecrawl: {e}\n{raw[:500]}\n")
    sys.exit(5)
if not obj.get("success", False):
    sys.stderr.write(f"error: Firecrawl reported failure for {os.environ['URL']}: {obj.get('error') or obj}\n")
    sys.exit(6)
md = (obj.get("data") or {}).get("markdown") or ""
if not md.strip():
    sys.stderr.write(f"error: Firecrawl returned empty markdown for {os.environ['URL']}\n")
    sys.exit(7)
sys.stdout.write(md)
PY
