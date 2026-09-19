# ContextManager materialization: where the seam goes

Scoping note for the next slice, written 2026-09-19 against `d99b442b`. No code
changed. This answers the placement question the handoff flagged, before any
implementation, because the wrong placement creates the second cognitive core
the ledger forbids.

## Current split

`crates/akh-session-services/src/internal.rs` states its own ownership: "Model
and tool loop owned by akhrtd. Cognitive context comes from akhomed." In
practice the runtime does more than that today:

- `crates/akh-session-services/src/native_window.rs` selects the prompt view. It
  drops whole turns oldest-first until the rendered prompt fits the budget,
  pins unmarked system messages, keeps the newest user turn with its tool
  traffic, and errors with `capacity` if the pinned remainder still overflows.
  It returns `dropped_messages` and `prompt_tokens`, nothing else.
- `crates/akh-session-services/src/native_recall.rs` decides which recalled
  spans become `neural_recall` system messages ahead of a user turn.
- `src/agent/context/manager.rs` holds the real seam in akhomed: `assemble`,
  `commit` and `stats`, with `ContextSection`, `DropReason`, `TokenBudget`,
  `TurnOutcome` and `ContextWindowStats`, and shipped strategies
  (`PassThrough`, `Passive`, `Compacting`, `Virtual`) wrapped in
  `SafetyEnforced` so eviction can never drop a safety-pinned span.

So eviction and materialization are currently decided twice: a rich strategy in
the core, and an independent oldest-first heuristic in the runtime that the core
never sees and cannot learn from. `commit` is never called for a native turn.

## Decision

The ContextManager stays in akhomed and becomes authoritative for native turns.
akhrtd keeps no eviction policy of its own.

- akhomed assembles the turn, including which recalled spans are materialized,
  and hands akhrtd a message list that already fits the budget.
- akhrtd keeps `native_window` only as a defensive capacity check: if what it
  receives does not fit the real tokenizer count, it fails the turn with
  `capacity` rather than silently dropping cognitive content. A runtime that
  quietly discards turns is a second policy.
- akhomed calls `commit` with the turn outcome so `Compacting` and `Virtual`
  learn from native turns exactly as they do from other backings.
- `ContextWindowStats` becomes the single reported accounting for native turns,
  replacing the bare `dropped_messages` integer.

The budget must be counted with the model's own tokenizer, not the `len/4`
heuristic: `CandleTokenCounter` already exists in `src/agent/context/manager.rs`
for this, and the runtime's exact count is what the capacity check compares
against.

## Why not the other placement

Moving strategy into akhrtd would put assemble/commit next to the weights and
avoid a round trip. It is rejected: it duplicates the core's policy, gives the
runtime durable-history opinions it must not have, and makes recall quality a
property of the runtime process rather than of the cognitive core. That is the
second Engine the ledger rules out.

## First implementable slice

1. **Done** (`native_window::verify`, wired in `internal_native::completion`).
   The runtime verifies instead of evicting whenever a message carries
   `"context_budgeted": true`, and fails `capacity` with the measured and
   budgeted token counts. Absent the marker it takes the existing evicting
   path, so a core that has not moved yet is unaffected: the five-daemon canary
   reran on HIP with the identical result, four dropped messages and the
   correct post-restart answer. Six `native_window` tests and the 25-test
   runtime suite pass.
2. Emit `ContextWindowStats` for native turns from whichever side owns the
   decision, replacing `dropped_messages` in the native response shape.
3. Call `commit` with a real `TurnOutcome` after each native turn.
4. Rerun the five-daemon canary: its `final_dropped_messages` and
   `max_materialized_spans` assertions are exactly the observable this changes,
   so they are the acceptance gate.

Steps 1-3 need a wire-shape change between akhomed and akhrtd. Keep the existing
payload accepted so a mixed pair still interoperates during the transition, the
same discipline the span annex used.
