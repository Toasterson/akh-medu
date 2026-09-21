# Native prompt window prerequisite

Inspection before combined HTTP/model acceptance found that `akhrtd` still
rendered the complete durable conversation and rejected physical context
overflow. It had no actual eviction path on which to test recall after eviction.

The native completion path now selects a bounded view, using the exact
family-specific completion tokenizer on the fully rendered prompt (including
tool schemas). It reserves the configured output budget and evicts only whole
older user-led turns, including their assistant/tool traffic. Unmarked system
messages and the newest user turn with all its tool traffic remain pinned.
Runtime-marked recalled system notes move with their following user turn.
The complete durable conversation is retained unchanged by selection.

Recall reprompting uses the same selection and output reservation. It may
evict more old turns to fit notes, but if the pinned/current context plus notes
cannot fit, optional recall is skipped. Tokenizer errors propagate; no heuristic
fallback is used. Existing continuation handling detects divergent token
prefixes and freshly prefills; the selector does not splice KV state.

## Tests and limits

The initial whole-turn eviction test failed against a pass-through selector.
Three selector tests now cover whole tool-cycle eviction, system/current-turn
pinning, attached recall blocks, exact-budget pass-through and counter errors.
The existing tiny native CPU fixture was extended to 20 ordinary turns under
a 96-token physical cap. It completes, retains the original whitespace-exact
first user message in durable history, and inserts that fact as a recalled
system block when the fact is no longer visible. The fixture always answers
`ok` and supplies canned recall hits: this proves window/reprompt mechanics,
not semantic answer quality or HTTP transport.
An additional oversized recalled-note case completes normally and verifies
that skipped notes are not inserted into durable history.

The Candle package suite passed with 29 library tests and two artifact/GPU
tests explicitly ignored, plus the enabled binary and doc-test targets.
The exact-model HIP/Vulkan answer gates predate this change and must be rerun.

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features candle-runtime
```

This is a minimal native prompt-view selector, not completed ContextManager
strategy/commit integration, virtual-memory statistics, learned recall or KV
compression. Durable history still has the existing 8 MiB byte ceiling.
Selection currently recounts successively smaller rendered histories; its
performance is not yet benchmarked. The runtime integration and exact counter
live in the pre-existing mixed runtime worktree (`internal.rs`,
`internal_native.rs`, `candle_backend.rs`); do not mistake a standalone selector
checkpoint for a deployable runtime release.

Next: combined isolated native/core/session HTTP acceptance, with eviction
enabled and exact-model answer checks, followed by full ContextManager
materialization/commit and telemetry work. No shared services changed.
