# Phase 12c — Pre-Communication Constraint Checking

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12b (grounded dialogue)

## Objective

Add a six-check constraint pipeline that validates outbound messages before
emission, with configurable per-channel-kind behavior (emit/annotate/suppress).

## Deliverables

### New files

1. **`src/agent/constraint_check.rs`** (~600 lines)
   - `ConstraintCheckError` miette diagnostic enum
   - `ConstraintViolation` enum: Contradiction, BelowConfidence, RateLimitExceeded, SensitivityBreach
   - `ConstraintWarning` enum: Ungrounded, LowRelevance, LowConfidence
   - `CheckOutcome` struct: passed, violations, warnings
   - `CommunicationBudget`: sliding-window rate tracking with cooldown
   - `ConstraintConfig` + `ConfidenceThresholds`: per-channel-kind thresholds
   - `SensitivityLevel` enum (Public/Low/Medium/High/Private) with channel kind mapping
   - `ConstraintChecker`: stateful checker with 6-check pipeline
   - `EmissionDecision` enum and `emission_decision()` function
   - 15 unit tests

### Modified files

2. **`src/agent/conversation.rs`** — `GroundedTriple` enriched with SymbolId
   fields, renamed label/confidence/derivation fields

3. **`src/agent/channel_message.rs`** — `ConstraintCheckStatus` evolved with
   Passed/Failed variants, `from_outcome()`, `is_passed()`

4. **`src/agent/agent.rs`** — `constraint_checker` field, initialization,
   `check_and_wrap_grounded()` method, accessor methods

5. **`src/agent/error.rs`** — `ConstraintCheck` transparent variant

6. **`src/agent/mod.rs`** — module declaration + re-exports

7. **`src/tui/mod.rs`** — constraint-checked grounded query path with
   operator annotation

8. **`src/main.rs`** — constraint-checked headless query path

## Verification

- `cargo build` — compiles with no new warnings
- `cargo test --lib` — 1072 tests pass (17 new)
