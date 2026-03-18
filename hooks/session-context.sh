#!/bin/bash
# SessionStart hook: query akh-medu KG status and inject context summary.
# Connects to the MCP HTTP endpoint, initializes a session, and calls status.
# If the server is unreachable, exits silently with a note.
set -euo pipefail

BASE="http://127.0.0.1:8200/mcp"

# Try to initialize an MCP session
INIT_RESPONSE=$(curl -s --max-time 5 -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"hook","version":"1.0"}}}' \
  -D /dev/stderr 2>/tmp/akh-hook-headers.txt) || {
  echo '{"systemMessage": "akh-medu daemon is offline — KG context unavailable this session."}'
  exit 0
}

SESSION=$(grep -i "mcp-session-id" /tmp/akh-hook-headers.txt 2>/dev/null | tr -d '\r' | awk '{print $2}')

if [ -z "$SESSION" ]; then
  echo '{"systemMessage": "akh-medu daemon is offline — KG context unavailable this session."}'
  exit 0
fi

# Send initialized notification
curl -s --max-time 3 -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null 2>&1 || true

# Call status tool
STATUS=$(curl -s --max-time 5 -X POST "$BASE" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"status","arguments":{}}}' \
  2>/dev/null | grep "^data: {" | sed 's/^data: //' | head -1)

if [ -z "$STATUS" ]; then
  echo '{"systemMessage": "akh-medu is running but status query failed."}'
  exit 0
fi

# Extract the text content from the MCP response
TEXT=$(echo "$STATUS" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d['result']['content'][0]['text'])
except:
    print('KG status unavailable')
" 2>/dev/null)

echo "{\"systemMessage\": \"akh-medu KG context: ${TEXT}\"}"
exit 0
