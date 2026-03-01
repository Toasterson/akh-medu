# ADR 008 — Communication Channel Abstraction

> Date: 2026-02-21
> Status: Accepted
> Phase: 12a

## Context

The TUI (`src/tui/mod.rs`), headless chat (`src/main.rs`), and WebSocket
(`src/bin/akhomed.rs`) each interact with the agent through different ad-hoc
code paths. There is no shared protocol layer and no capability model
differentiating operators from external actors. Phase 12 needs a unified
abstraction for all interaction surfaces.

## Decision

Introduce a `CommChannel` trait with an OCapN-inspired capability model:

1. **`CommChannel` trait** (Send, not Sync) — the unit of interaction.
   Five methods: `channel_id`, `channel_kind`, `capabilities`, `try_receive`
   (non-blocking), `send`, `is_connected`.

2. **`ChannelKind`** — four trust tiers: Operator (full control, singleton),
   Trusted (peer federation), Social (external user), Public (read-only).

3. **`ChannelCapabilities`** — immutable struct with 10 boolean capability
   flags and an optional rate limit. Factory methods produce the correct
   preset for each kind. `require(action)` gates actions at runtime.

4. **`ChannelRegistry`** on the Agent — HashMap-keyed by channel ID.
   Enforces exactly-one-operator invariant at registration time.

5. **`OperatorChannel`** — first concrete implementation. Wraps the existing
   `Arc<dyn MessageSink>` for outbound rendering and provides an
   `InboundHandle` (cloneable, `Arc<Mutex<VecDeque>>`) for the UI event
   loop to enqueue messages.

6. **Message types** — `InboundMessage` and `OutboundMessage` with
   `InterlocutorId`, `MessageContent` (Text/Command), `ResponseContent`
   (wrapping `Vec<AkhMessage>`), provenance, confidence, and a placeholder
   `ConstraintCheckStatus`.

## Alternatives Considered

- **Async channels (tokio mpsc)**: Rejected because the OODA loop is
  synchronous and adding an async runtime would be a large cross-cutting
  change. `try_receive` + `VecDeque` is sufficient.

- **Single enum instead of trait**: Would couple all transport details
  together and make the registry less extensible. Trait object dispatch
  is the right cost for open extensibility.

- **No capability model**: Could add capabilities later, but inserting
  them now (immutable, factory-based) is zero-cost and prevents
  security gaps when social/public channels arrive in 12d–12e.

## Consequences

- All interaction surfaces can be treated uniformly by the agent.
- The operator invariant prevents accidental privilege escalation.
- The `InboundHandle` pattern preserves the existing synchronous event loop.
- Future phases (12b–12g) extend `MessageContent`, `ResponseContent`, and
  `ConstraintCheckStatus` without breaking the trait interface.
- The `CommChannel` trait is `Send` but not `Sync`, matching the ownership
  model where the registry holds channels and lends `&mut` borrows.
