---
type: research
status: open
assignee: null
anima_id: 01a0878c-304f-79f2-b81a-e2d583ee5735
legacy_anima_id: 01a086ef-80ab-76f1-a356-37e6c23a4db7
legacy_status: cancelled
plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_assignment: verified native member
blocked_by: []
---

# Codex and Claude integration contracts and subscription boundaries

## Question

Which current official interfaces support user-owned subscription authentication, Akh custom tools, cancellation, approvals, session resume and disabling native execution? For each claim cite primary sources and version/date. Separate technical interoperability, documented support and unresolved restrictions. Can all filesystem, command, hook, MCP and subagent execution remain under Akh control without modifying provider authentication? No live model calls or credential reads are needed. Capture findings in assets/codex-claude.md.

Research branch: `research/akh-agent-loops-codex-claude`.

## Research findings available

Codex App Server and Claude SDK expose host tool callbacks. Codex hooks expressly do not form a complete execution boundary; Claude tool exclusion, strict MCP and aliases are promising but untested together. User sign-in to unmodified Claude Code and SDK product authentication have distinct published guidance. No exclusive Akh/SSH conformance claim.

[Full research asset](../assets/codex-claude.md).

Research commit: `2ab0b7704b53b8698d8a13d05dfbf013ced813d5` on `research/akh-agent-loops-codex-claude`.

Status: findings are ingested and linked to this Work Item. Anima rejects draft → done without promotion to ready. Retain the planning draft; do not rerun this completed research or promote it merely to close it.

## Tracker membership

Canonical item: [Codex and Claude integration contracts and subscription boundaries](https://anima.wegmueller.it/work-items/01a0878c-304f-79f2-b81a-e2d583ee5735).
Member of [the agent-loop Wayfinder Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615). The previous
orphan record remains cancelled; this replacement is an analysis item in draft.

[Native research Document](https://anima.wegmueller.it/documents/01a0878d-9451-7182-b97c-bf3470f924fc).
