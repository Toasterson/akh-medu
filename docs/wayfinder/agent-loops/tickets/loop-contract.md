---
type: grilling
status: open
assignee: null
anima_id: 01a0878c-315a-73b3-bc8c-d16e006934de
legacy_anima_id: 01a086ef-81f4-7512-a4e0-062e0ea473d8
legacy_status: cancelled
plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_assignment: verified native member
blocked_by: ["session-ownership", "execution-boundary"]
---

# AgentLoop contract and lifecycle events

## Question

What minimal object-safe Rust contract represents a complete loop session while leaving ChatBackend below InternalLoop and Backend below tools? Specify commands/events for start, submit, interrupt, approvals/input, finish, reconnect and close; typed outcomes/capabilities; backpressure and single-turn ownership. Account for existing synchronous tools and thread-local session context without assuming async callbacks preserve it.

## Tracker membership

Canonical item: [AgentLoop contract and lifecycle events](https://anima.wegmueller.it/work-items/01a0878c-315a-73b3-bc8c-d16e006934de).
Member of [the agent-loop Wayfinder Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615). The previous
orphan record remains cancelled; this replacement is an analysis item in draft.
