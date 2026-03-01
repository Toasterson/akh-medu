#!/usr/bin/env bash
# stability-monitor.sh — Background loop that logs daemon health to CSV.
#
# Usage:
#   ./scripts/stability-monitor.sh [PORT] [INTERVAL_SECS]
#   # Ctrl+C to stop. Output: stability-YYYY-MM-DD.csv
#
# Logs: timestamp, RSS (KB), VSZ (KB), symbols, triples, cycles, goals
# Run overnight before leaving — check for unbounded RSS growth.

set -euo pipefail

PORT="${1:-8200}"
INTERVAL="${2:-300}"  # default: 5 minutes
BASE="http://127.0.0.1:${PORT}"
WS="${AKH_WORKSPACE:-default}"
OUTFILE="stability-$(date +%Y-%m-%d).csv"

echo "Monitoring akhomed (${BASE}) every ${INTERVAL}s → ${OUTFILE}"
echo "Press Ctrl+C to stop."
echo ""

# Write CSV header.
echo "timestamp,rss_kb,vsz_kb,symbols,triples,cycles,goals,running" > "$OUTFILE"

while true; do
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Get process stats (RSS, VSZ) from the PID.
    pid=$(pgrep -f "akhomed" | head -1 || true)
    if [ -n "$pid" ]; then
        # macOS ps uses different column names; this works on both Linux and macOS.
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
        vsz=$(ps -o vsz= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
    else
        rss=0
        vsz=0
    fi

    # Get daemon status from HTTP.
    json=$(curl -s --max-time 5 "${BASE}/workspaces/${WS}/daemon" 2>/dev/null || echo "{}")
    symbols=$(echo "$json" | grep -o '"kg_symbols":[0-9]*' | grep -o '[0-9]*' || echo "0")
    triples=$(echo "$json" | grep -o '"kg_triples":[0-9]*' | grep -o '[0-9]*' || echo "0")
    cycles=$(echo "$json" | grep -o '"total_cycles":[0-9]*' | grep -o '[0-9]*' || echo "0")
    goals=$(echo "$json" | grep -o '"active_goals":[0-9]*' | grep -o '[0-9]*' || echo "0")
    running=$(echo "$json" | grep -o '"running":true' | head -1 || echo "false")
    [ -n "$running" ] && running="true" || running="false"

    echo "${ts},${rss},${vsz},${symbols},${triples},${cycles},${goals},${running}" >> "$OUTFILE"
    printf "%s  RSS=%sKB  sym=%s  tri=%s  cyc=%s  goals=%s\n" "$ts" "$rss" "$symbols" "$triples" "$cycles" "$goals"

    sleep "$INTERVAL"
done
