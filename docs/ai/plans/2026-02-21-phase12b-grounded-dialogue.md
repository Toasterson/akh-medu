# Phase 12b — Grounded Operator Dialogue

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12a (channel abstraction)

## Objective

Add KG-grounded query responses with provenance, configurable detail levels,
and conversation state tracking to the operator channel.

## Deliverables

### New files

1. **`src/agent/conversation.rs`** (~450 lines)
   - `ResponseDetail` enum (Concise/Normal/Full) with `from_str_loose` parser
   - `Speaker` enum (Operator/Agent)
   - `ConversationTurn` struct with speaker, text, resolved entities, timestamp
   - `ConversationState` with bounded turn history, active referents, detail level
   - `GroundedResponse` with prose, supporting triples, confidence, provenance
   - `GroundedTriple` with labels, confidence, derivation tag
   - `ground_query()` pipeline: resolve → collect triples → filter metadata →
     collect provenance → synthesize → return `GroundedResponse`
   - 16 unit tests

### Modified files

2. **`src/agent/channel_message.rs`** — `ResponseContent::Grounded` variant,
   `OutboundMessage::grounded()` constructor, updated `to_akh_messages()`

3. **`src/agent/agent.rs`** — `conversation_state` field, initialization in
   `new()` and `resume()`, accessor methods, `set_response_detail()`

4. **`src/agent/nlp.rs`** — `UserIntent::SetDetail` variant, "set detail" and
   "detail" prefix recognition, 3 new unit tests

5. **`src/agent/mod.rs`** — module declaration and re-exports for conversation types

6. **`src/tui/mod.rs`** — grounded-first query path in `process_input_local`,
   `SetDetail` intent handler, updated help text

7. **`src/main.rs`** — grounded-first query path in headless chat, `SetDetail` handler

8. **`docs/ai/architecture.md`** — Phase 12b status, module count bump

9. **`docs/ai/decisions/009-grounded-dialogue.md`** — ADR for the design

## Verification

- `cargo build` — compiles with no new warnings
- `cargo test --lib` — 1055 tests pass (17 new)
- Pre-existing integration test failures (missing `access_timestamps`/
  `expected_effects` fields) are unrelated
