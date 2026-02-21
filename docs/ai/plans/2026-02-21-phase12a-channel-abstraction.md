# Phase 12a — Communication Channel Abstraction

> Status: **Complete**
> Date: 2026-02-21

## Goal

Unify all agent interaction surfaces (TUI, headless REPL, WebSocket) behind a common
`CommChannel` trait with an OCapN-inspired capability model. Insert the abstraction
transparently without changing observable behavior.

## Design

- **`CommChannel` trait** (Send, not Sync) — `channel_id`, `channel_kind`, `capabilities`,
  `try_receive` (non-blocking), `send`, `is_connected`
- **`ChannelKind`** — Operator / Trusted / Social / Public
- **`ChannelCapabilities`** — 10 boolean flags + optional rate limit. Factory methods per kind.
  `require(action, channel_id, kind)` for capability gating.
- **`ChannelRegistry`** — HashMap-based, enforces exactly-one-operator invariant
- **`OperatorChannel`** — wraps existing `MessageSink` with `InboundHandle` (cloneable
  producer for the UI event loop)
- **Message types** — `InboundMessage` (channel_id, sender: InterlocutorId, content:
  MessageContent, timestamp), `OutboundMessage` (ResponseContent wrapping Vec<AkhMessage>,
  provenance, confidence, constraint_check: ConstraintCheckStatus)
- **Agent integration** — `channel_registry` field, `setup_operator_channel()` method,
  `drain_inbound()` method

## Files

| File | Action | Lines |
|------|--------|-------|
| `src/agent/channel.rs` | New | ~310 |
| `src/agent/channel_message.rs` | New | ~240 |
| `src/agent/operator_channel.rs` | New | ~200 |
| `src/agent/error.rs` | Modified | +5 |
| `src/agent/mod.rs` | Modified | +10 |
| `src/agent/agent.rs` | Modified | +45 |
| `src/message.rs` | Modified | +5 |
| `src/tui/mod.rs` | Modified | +25 |
| `src/main.rs` | Modified | +10 |

## Tests

- `channel.rs`: 16 tests (capability presets, require pass/fail, registry invariants,
  drain, channel_ids, kind display)
- `channel_message.rs`: 11 tests (InterlocutorId, classify delegation, OutboundMessage
  round-trip, builders, ConstraintCheckStatus)
- `operator_channel.rs`: 8 tests (FIFO, send through sink, InboundHandle, cloneable,
  metadata, pending count)

## Future (deferred to 12b–12g)

- `ResponseContent::Grounded` variant (12b)
- `ConstraintCheckStatus::Passed/Failed` variants (12c)
- `MessageContent::GoalProposal/Activity/Reaction` variants (12d/12e)
- WebSocket channel implementation (12e)
- Social channel with per-interlocutor microtheories (12d)
