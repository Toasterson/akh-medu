#!/usr/bin/env bash
# smoke-test.sh — Quick verification that akhomed is serving all endpoints.
#
# Usage:
#   ./scripts/smoke-test.sh [PORT]   (default: 8200)
#
# Runs ~20 HTTP endpoint checks against a running akhomed and reports
# PASS/FAIL for each. Exit code is the number of failures.

set -euo pipefail

PORT="${1:-8200}"
BASE="http://127.0.0.1:${PORT}"
WS="${AKH_WORKSPACE:-default}"

PASS=0
FAIL=0
TOTAL=0

check() {
    local label="$1"
    local method="$2"
    local url="$3"
    local body="${4:-}"
    TOTAL=$((TOTAL + 1))

    local status
    if [ "$method" = "GET" ]; then
        status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")
    else
        status=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
            -X "$method" -H "Content-Type: application/json" \
            -d "${body:-{}}" "$url" 2>/dev/null || echo "000")
    fi

    if [ "$status" -ge 200 ] && [ "$status" -lt 400 ]; then
        printf "  PASS  %-40s %s\n" "$label" "$status"
        PASS=$((PASS + 1))
    else
        printf "  FAIL  %-40s %s\n" "$label" "$status"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== akh-medu smoke test (${BASE}, workspace: ${WS}) ==="
echo ""

# Core endpoints
check "health"                 GET  "${BASE}/health"
check "config"                 GET  "${BASE}/config"
check "workspaces"             GET  "${BASE}/workspaces"

# Workspace-scoped endpoints
check "workspace info"         GET  "${BASE}/workspaces/${WS}/info"
check "workspace status"       GET  "${BASE}/workspaces/${WS}/status"
check "symbols"                GET  "${BASE}/workspaces/${WS}/symbols"
check "triples"                GET  "${BASE}/workspaces/${WS}/triples"
check "goals"                  GET  "${BASE}/workspaces/${WS}/goals"
check "daemon status"          GET  "${BASE}/workspaces/${WS}/daemon"

# Awaken
check "awaken status"          GET  "${BASE}/workspaces/${WS}/awaken/status"

# Seeds
check "seeds list"             GET  "${BASE}/workspaces/${WS}/seeds"

# PIM
check "PIM inbox"              GET  "${BASE}/workspaces/${WS}/pim/inbox"

# Calendar
check "calendar events"        GET  "${BASE}/workspaces/${WS}/cal/events"

# Preferences
check "preferences"            GET  "${BASE}/workspaces/${WS}/preferences"

# Causal
check "causal schemas"         GET  "${BASE}/workspaces/${WS}/causal/schemas"

# Library
check "library list"           GET  "${BASE}/workspaces/${WS}/library"

# Triggers
check "triggers"               GET  "${BASE}/workspaces/${WS}/triggers"

# Compartments
check "compartments"           GET  "${BASE}/workspaces/${WS}/compartments"

echo ""
echo "=== Results: ${PASS}/${TOTAL} passed, ${FAIL} failed ==="

exit "$FAIL"
