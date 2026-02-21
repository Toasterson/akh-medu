# ADR 012 — Federation via Oxifed

> Date: 2026-02-21
> Status: Accepted
> Phase: 12e

## Context

After Phase 12d established per-interlocutor social modeling, the agent existed
only in the terminal. It could not participate in federated social networks
where neurosymbolic interaction becomes most interesting. Oxifed is a
multi-domain ActivityPub federation platform that handles all AP protocol
details (HTTP signatures, WebFinger, inbox/outbox, delivery). Akh-medu
needs to integrate as an application inside the oxifed ecosystem.

## Decision

Introduce an `OxifedChannel` implementing `CommChannel` with `ChannelKind::Social`,
communicating with oxifed through its AMQP message bus and REST admin API.
The entire module is feature-gated under `--features oxifed`.

### Architecture

1. **App-in-ecosystem model** — akh-medu never touches ActivityPub directly.
   Oxifed's `domainservd` handles HTTP endpoints, signatures, WebFinger.
   Oxifed's `publisherd` handles federation delivery. Akh-medu is a consumer
   of oxifed's AMQP bus.

2. **AMQP integration** via `lapin`:
   - **Consumer task**: Background tokio task subscribes to the actor's inbox
     queue (bound to `oxifed.incoming.process` exchange), parses messages,
     converts to `InboundMessage`, and pushes to a sync `VecDeque`.
   - **Publisher task**: Background tokio task reads from an `mpsc` channel
     and publishes to `oxifed.internal.publish` exchange for outbound delivery.

3. **Serde-compatible message types** — defined locally to avoid depending on
   the full oxifed crate (which pulls mongodb, ring, axum). JSON serialization
   is the contract. Types match oxifed's `messaging` module format.

4. **Activity bridges**:
   - `incoming_object_to_inbound()`: Extracts text from Note/Article objects,
     strips HTML tags, creates `MessageContent::Text`.
   - `incoming_activity_to_inbound()`: Maps Create→Text, Follow→Command,
     Like/Announce→Command, Undo→Command. Unknown types are silently dropped.
   - `outbound_to_note()`: Linearizes `OutboundMessage` (both `Messages` and
     `Grounded` variants) to a `NoteCreate` payload.

5. **Constraint-check gating** — `send()` checks `ConstraintCheckStatus::is_passed()`
   before publishing. Failed checks are silently dropped (Social channel
   suppression from Phase 12c).

6. **Non-blocking sync interface** — `try_receive()` pops from the VecDeque
   without blocking. `send()` pushes to the mpsc channel. The Agent's sync
   OODA loop is never blocked by AMQP I/O.

### Feature gate

```toml
[features]
oxifed = ["daemon", "lapin", "deadpool-lapin", "reqwest"]
```

The `daemon` dependency provides tokio for background tasks.

## Alternatives Considered

- **Depend on the oxifed crate directly**: Rejected — it pulls mongodb,
  ring, axum, and many other heavy dependencies. Defining compatible serde
  structs locally keeps the dependency chain minimal.

- **Async CommChannel trait**: Rejected — the Agent's OODA loop is synchronous
  by design (ADR 007). Non-blocking polling via `try_receive()` is sufficient.
  Background tokio tasks bridge the async AMQP operations.

- **REST-only integration (no AMQP)**: Would require polling the admin API
  for new messages, introducing latency. AMQP provides real-time delivery
  with publisher confirms.

- **Direct ActivityPub implementation**: Massive scope — HTTP signatures,
  WebFinger, actor model, delivery, retries. Oxifed already handles all of
  this. The app-in-ecosystem model is the right abstraction level.

## Consequences

- Akh-medu can join the fediverse as an oxifed actor (e.g., `@akh@domain.example`).
- All outbound messages go through the Phase 12c constraint pipeline before
  federation — no ungrounded or low-confidence claims leak to social channels.
- The oxifed feature is opt-in; default builds have zero AMQP/async overhead.
- Interlocutor profiles (Phase 12d) are created for Follow events from the
  fediverse, enabling the theory-of-mind model for remote actors.
- Future work: timeline consumption (reading the home timeline), profile
  synchronization from KG state, and agent-to-agent structured communication
  (Phase 12g) over the same AMQP transport.
