#!/bin/bash
# Stop hook: approve stopping and note any KG-worthy insights.
# Currently just approves — consolidation logic can be added later
# once we have fine-grained MCP tools for batch triple assertion.
echo '{"decision": "approve", "reason": "session complete"}'
exit 0
