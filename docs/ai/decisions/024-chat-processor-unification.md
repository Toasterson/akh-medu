# ADR 024: Unified ChatProcessor for Input Processing

**Status:** Accepted
**Date:** 2026-02-28
**Context:** Phase 14 (NLU Extension) prerequisite

## Problem

Input processing was duplicated across four locations with inconsistent feature coverage:

1. **TUI `process_input_local()`** (~460 lines) — full chain: grounded dialogue, discourse, conversational classification, persona responses, `handle_unknown_subject`, `escalate_to_goal`
2. **akhomed `process_ws_input()`** (~240 lines) — degraded copy: no conversational classification, no grounded dialogue for queries, no persona responses. "Hello, who are you?" resolved to `Query { subject: "you" }` which hit `engine.resolve_symbol("you")` and returned "Symbol not found."
3. **Headless CLI `Commands::Chat`** (~200 lines) — separate intent dispatch, no NLU pipeline, no freeform handling
4. **`AgentAction::Chat`** (~100 lines) — autonomous goal-per-question, no shared handling

Total: ~1000 lines of near-identical code with divergent behavior.

## Decision

Create a unified `ChatProcessor` struct in `src/chat.rs` that:

- Owns the `NluPipeline` and grammar preference
- Takes `(&str, &mut Agent, &Engine)` and returns `Vec<AkhMessage>`
- Contains ALL intent handlers migrated from the TUI
- Implements NLU-first processing: structural commands skip NLU, everything else tries the NLU pipeline before falling back to `classify_intent()`
- Is used identically by TUI, daemon WS, and headless CLI

### Processing flow

```
Input
  +-- "/" prefix -> caller handles (TUI commands, not ChatProcessor)
  +-- Structural command? -> skip NLU, classify_intent() directly
  +-- Everything else -> NLU pipeline first
       +-- NLU succeeds -> ingest as fact via TextIngestTool
       +-- NLU fails -> classify_intent() fallback
            +-- Query -> ground_query -> discourse -> synthesis -> handle_unknown_subject -> escalate
            +-- Assert -> TextIngestTool
            +-- SetGoal -> add goal + run cycles + synthesize
            +-- Freeform -> classify_conversational
            |    +-- Greeting/Ack/FollowUp/MetaQuestion -> persona response
            |    +-- Unrecognized -> escalate_to_goal
            +-- Other intents -> handle as appropriate
```

### Command merge

`akh agent chat` is deprecated (prints warning, delegates to same code path). `akh chat` gains `--fresh` flag. The agent is always autonomous — no separate `--max-cycles` flag needed (ChatProcessor uses a sensible default of 10).

## Consequences

- **Bug fix:** "Hello, who are you?" now gets a persona greeting instead of "Symbol not found" in remote mode
- **~700 lines removed** from duplicated handlers across TUI, akhomed, and headless CLI
- **Single point of truth** for input processing — all future intent handlers go in `ChatProcessor`
- **NLU-first** processing means structured natural language facts are ingested without needing assertion patterns
- `akh agent chat` still works but prints a deprecation warning

## Files Changed

| File | Action |
|------|--------|
| `src/chat.rs` | Created (~500 lines) |
| `src/lib.rs` | Added `pub mod chat;` |
| `src/message.rs` | Added `to_plain_text()` method |
| `src/tui/mod.rs` | Replaced ~460-line `process_input_local()` with 5-line delegation, removed standalone helpers |
| `src/bin/akhomed.rs` | Deleted `process_ws_input()` (~240 lines), replaced with ChatProcessor |
| `src/main.rs` | Replaced headless chat (~200 lines), deprecated `AgentAction::Chat`, added `--fresh` to `Commands::Chat` |
