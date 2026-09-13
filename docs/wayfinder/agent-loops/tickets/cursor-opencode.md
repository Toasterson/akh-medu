---
type: research
status: open
assignee: null
anima_id: 01a0878c-30a2-7022-942f-078aaf065f19
legacy_anima_id: 01a086ef-8121-7a33-b729-1ddb28b6c15d
legacy_status: cancelled
plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_assignment: verified native member
blocked_by: []
---

# Cursor and OpenCode integration contracts and execution control

## Question

Which current official interfaces can implement AgentLoop for Cursor and OpenCode? Compare server/SDK, CLI, ACP, custom tools/MCP, subscription authentication, streamed events, approvals, cancellation, resume and reliable exclusion of native execution. Do not assume that supporting MCP gives full execution interception or that OpenCode can reuse another vendor subscription. Cite primary sources and explicitly record unknown capabilities. No live model calls or credential reads. Capture assets/cursor-opencode.md.

Research branch: `research/akh-agent-loops-cursor-opencode`.

## Research findings available

Cursor has ACP and an official SDK Bridge with local tool allowlisting and custom callbacks; restrictions must be reapplied on resume. OpenCode has server/ACP integration, but pinned source registers MCP by directory/name rather than session identity. Treat dedicated instances as a candidate until isolation tests pass. OpenCode documents ChatGPT access but rejects Claude subscription plugins. Neither adapter has proven exclusive Akh/SSH execution.

[Full research asset](../assets/cursor-opencode.md).

Research commit: `3740efb4eb526ffc650c57c7696657fe9223d5cd` on `research/akh-agent-loops-cursor-opencode`.

Status: findings are ingested and linked to this Work Item. Anima rejects draft → done without promotion to ready. Retain the planning draft; do not rerun this completed research or promote it merely to close it.

## Tracker membership

Canonical item: [Cursor and OpenCode integration contracts and execution control](https://anima.wegmueller.it/work-items/01a0878c-30a2-7022-942f-078aaf065f19).
Member of [the agent-loop Wayfinder Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615). The previous
orphan record remains cancelled; this replacement is an analysis item in draft.

[Native research Document](https://anima.wegmueller.it/documents/01a0878d-94d0-7100-8258-812d3dba8eb3).
