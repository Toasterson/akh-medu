# Phase 12g — Multi-Agent Communication

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12a (channel abstraction), Phase 12d (interlocutor registry)

## Objective

Add structured agent-to-agent communication with OCapN-inspired capability
tokens. Agents communicate via `AgentProtocolMessage` variants carried as
`MessageContent::AgentMessage` through the existing `CommChannel` infrastructure.
Capability tokens scope what each agent may do (query, assert, propose goals,
subscribe, view provenance).

## Deliverables

### New files

1. **`src/agent/multi_agent.rs`** (~820 lines including tests)
   - `MultiAgentError` miette diagnostic enum (5 variants)
   - `CapabilityScope` enum (6 variants: QueryAll, QueryTopics, AssertTopics, ProposeGoals, Subscribe, ViewProvenance)
   - `CapabilityToken` struct with scoped permissions, expiry, revocation
   - `AgentProtocolMessage` enum (10 structured message types)
   - `InterlocutorKind` enum (Human, Agent)
   - `TokenRegistry` with pair indexing and message validation
   - Trust bootstrap functions (`initial_trust_for_agent`, `should_promote_trust`)
   - 21 unit tests

### Modified files

2. **`src/agent/mod.rs`** — module declaration + re-exports
3. **`src/agent/error.rs`** — `MultiAgent` transparent variant
4. **`src/agent/channel_message.rs`** — `MessageContent::AgentMessage` variant, `InboundMessage::classify()` dispatch
5. **`src/agent/nlp.rs`** — `UserIntent::AgentProtocol` variant
6. **`src/agent/channel.rs`** — `can_propose_goals` capability flag in `ChannelCapabilities`
7. **`src/agent/interlocutor.rs`** — `kind: InterlocutorKind` field on `InterlocutorProfile`, `is_agent()` method
8. **`src/agent/agent.rs`** — `token_registry` field on `Agent`, accessor methods
9. **`src/tui/mod.rs`** — `AgentProtocol` intent handling
10. **`src/main.rs`** — `AgentProtocol` intent handling in headless chat

## Verification

- `cargo build` — no new warnings
- `cargo test --lib` — 1122 tests pass (22 new)
- `cargo test --lib --features oxifed` — 1138 tests pass
