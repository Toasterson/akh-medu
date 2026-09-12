---
type: grilling
status: open
assignee: null
anima_id: 01a0878c-325d-74d3-9f45-1b47e9c6c044
legacy_anima_id: 01a086ef-83a6-77f3-89c8-fc80bb2f9cfe
legacy_status: cancelled
plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_assignment: verified native member
blocked_by: ["execution-boundary", "loop-contract", "context-continuity", "client-contract", "loop-strategies"]
---

# Migration path and provider conformance gates

## Question

What dependency-ordered implementation work follows the locked decisions? Preserve existing internal behavior, ACP clients, permissions, workspace consistency, evidence gates and configuration compatibility. Set provider-specific gates for concurrent session isolation, SSH execution, interrupted operations, restart/reconnect, and auth/usage errors. Keep unsupported adapters discoverable but unavailable until evidence passes.

## Tracker membership

Canonical item: [Migration path and provider conformance gates](https://anima.wegmueller.it/work-items/01a0878c-325d-74d3-9f45-1b47e9c6c044).
Member of [the agent-loop Wayfinder Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615). The previous
orphan record remains cancelled; this replacement is an analysis item in draft.
