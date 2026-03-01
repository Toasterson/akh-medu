# Phase 12e — Federation via Oxifed

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12d (social KG), Phase 12c (constraint checking)

## Objective

Integrate akh-medu as an application inside the oxifed ActivityPub federation
system. The akh communicates with oxifed through its AMQP message bus (inbox
consumer + outbox publisher) and REST admin API. All ActivityPub protocol
handling (HTTP signatures, WebFinger, delivery) is oxifed's responsibility.

## Deliverables

### New files

1. **`src/agent/oxifed.rs`** (~700 lines, feature-gated under `oxifed`)
   - `OxifedError` miette diagnostic enum
   - `OxifedConfig` — AMQP URL, admin API URL, domain, actor, token
   - Serde-compatible oxifed message types (11 struct types in `OxifedMessage` enum)
   - AMQP exchange/queue constants matching oxifed
   - `incoming_object_to_inbound()` — Note/Article content with HTML stripping
   - `incoming_activity_to_inbound()` — Create/Follow/Like/Announce/Undo
   - `outbound_to_note()` — linearizes `OutboundMessage` to `NoteCreate`
   - `OxifedChannel` implementing `CommChannel` (ChannelKind::Social)
     - Background AMQP consumer task → sync inbox VecDeque
     - mpsc outbound channel → background publisher task
     - Constraint-check gating in `send()`
   - `OxifedInboundHandle` for test injection
   - 16 unit tests

### Modified files

2. **`Cargo.toml`** — `lapin`, `deadpool-lapin`, `reqwest` as optional deps;
   `oxifed` feature flag (depends on `daemon`)
3. **`src/agent/mod.rs`** — `#[cfg(feature = "oxifed")]` module + re-exports
4. **`src/agent/error.rs`** — `Oxifed` transparent variant (cfg-gated)

## Verification

- `cargo build` — default features compile, no new warnings
- `cargo build --features oxifed` — compiles with oxifed feature
- `cargo test --lib` — 1082 tests pass (default features)
- `cargo test --lib --features oxifed` — 1098 tests pass (16 new)
