---
type: grilling
status: open
assignee: null
anima_id: 01a0878c-2fde-7ac1-9806-ea1a74edc799
legacy_anima_id: 01a086ef-8032-7e63-be25-1574988ed23d
legacy_status: cancelled
plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_assignment: verified native member
blocked_by: []
---

# Session ownership and loop selection

## Question

Is loop selection per Akh or per session, and which session properties remain stable across reconnects? Distinguish reconnecting a client, resuming a provider thread, and handing off to another loop. What are the client controller and observer rules?

## Accepted selection decision

The user chose: **“One loop fixed for each Akh.”**

Loop implementation selection belongs to the Akh configuration. All sessions
of that Akh use that implementation, with separate conversation and provider
session state. A client cannot select another loop for an individual session.
Different Akhs may use different implementations.

## Remaining lifecycle question

When changing an Akh's loop, must all its sessions be closed first?
Recommendation pending user: yes, so one Akh never has sessions running different
loop implementations at the same time. Reconnect, controller and observer rules
remain covered by the loop-contract and client-contract decisions. This ticket
remains open; its selection subdecision is accepted.

## Tracker membership

Canonical item: [Session ownership and loop selection](https://anima.wegmueller.it/work-items/01a0878c-2fde-7ac1-9806-ea1a74edc799).
Member of [the agent-loop Wayfinder Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615). The previous
orphan record remains cancelled; this replacement is an analysis item in draft.
