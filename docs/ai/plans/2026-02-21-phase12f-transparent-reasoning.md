# Phase 12f — Transparent Reasoning and Explanations

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12b (grounded dialogue), Phase 12c (constraint checking)

## Objective

Wire the provenance ledger into conversational "why?" queries so the agent
can explain itself. Build a provenance-to-prose pipeline that walks derivation
chains and renders them as human-readable explanations.

## Deliverables

### New files

1. **`src/agent/explain.rs`** (~500 lines)
   - `ExplainError` miette diagnostic enum (EntityNotFound, NoProvenance, Provenance)
   - `ExplanationQuery` enum: Why, How, WhatKnown, HowConfident, WhatChanged
   - `ExplanationQuery::parse()` — NL recognition from user input
   - `DerivationNode` struct for recursive provenance trees
   - `build_derivation_tree()` — recursive provenance walk with cycle detection
   - `render_derivation_tree()` — indented hierarchy with box-drawing connectors
   - `render_derivation_prose()` — concise comma-separated format
   - `derivation_kind_prose()` — human-readable strings for all 48 DerivationKind variants
   - `explain_entity()` — derivation tree + known facts
   - `explain_known()` — enumerate triples with provenance
   - `explain_confidence()` — aggregate stats + evidence breakdown
   - `explain_changes()` — KG diff since timestamp
   - `execute_query()` — dispatch to appropriate function
   - 18 unit tests

### Modified files

2. **`src/agent/mod.rs`** — module declaration + re-exports
3. **`src/agent/error.rs`** — `Explain` transparent variant
4. **`src/agent/nlp.rs`** — `UserIntent::Explain` variant, parse check before Query
5. **`src/tui/mod.rs`** — Explain intent handling in `process_input_local()`
6. **`src/main.rs`** — Explain intent handling in headless chat

## Verification

- `cargo build` — no new warnings
- `cargo test --lib` — 1100 tests pass (18 new)
- `cargo test --lib --features oxifed` — 1116 tests pass
