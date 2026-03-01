# ADR 025: VSA-Native Dialogue System

**Status:** Accepted
**Date:** 2026-02-28
**Context:** Phase 14 (NLU Extension), extends ADR 024 (Unified ChatProcessor)

## Problem

After ADR 024 unified input processing into `ChatProcessor`, two parallel classification systems remained:

1. **`classify_intent()`** — regex-based pattern matching that maps raw text to `UserIntent` variants (Query, Assert, SetGoal, Freeform, etc.). First-match-wins with no confidence scoring.
2. **`classify_conversational()`** — Lexicon-scored keyword matching that maps freeform text to `ConversationalKind` variants (Greeting, Acknowledgment, FollowUp, MetaQuestion, Unrecognized).

These regex classifiers competed with the NLU pipeline rather than composing with it:

- `classify_intent()` ran *before* NLU for structural commands and *after* NLU as a fallback, creating two classification paths for the same input.
- `classify_conversational()` only ran when `classify_intent()` returned `Freeform`, meaning dialogue acts were second-class citizens discovered only after all other classification failed.
- Neither system produced `AbsTree` nodes, so dialogue acts bypassed the grammar framework entirely — no linearization, no GBNF coverage, no microtheory integration.
- Conversation state was in-memory only (`ConversationState` fields), not persisted in the KG, and not accessible to microtheory-based reasoning.

## Decision

Remove the regex classification layer and make dialogue acts first-class `AbsTree` variants. All user input flows through the NLU pipeline and gets dispatched by `AbsTree` variant:

1. **Extend `AbsTree`** with 7 dialogue-act variants: `Greeting`, `Farewell`, `Acknowledgment`, `FollowUpRequest`, `MetaQuery`, `GoalRequest`, `StructuralCommand`.
2. **Extend the rule parser** (`src/grammar/parser.rs`) with `try_dialogue_act()` that detects dialogue acts using Lexicon word lists before attempting any other parse. This runs at Tier 1 (0 MB, <1ms).
3. **Extend the GBNF grammar** (`src/nlu/abstree.gbnf`) with `dialogue-act` production rules so Tier 3 (Qwen2.5-1.5B) can also produce dialogue-act nodes via constrained decoding.
4. **Create `DialogueManager`** (`src/agent/dialogue.rs`) with KG-backed microtheory state: each channel gets a `dialogue:<channel_id>` microtheory storing active topic, last act type, and turn count as triples.
5. **Rewrite `ChatProcessor::dispatch_abstree()`** to match on `AbsTree` variants directly: dialogue acts route to `DialogueManager`, assertable structures are ingested as facts, queries are grounded, and freeform text is escalated to goals.
6. **Deprecate** `classify_intent()`, `classify_conversational()`, and `ConversationalKind` in `nlp.rs` (marked `#[deprecated]` with migration notes pointing to ADR 025).
7. **Add dialogue-act linearization** to all concrete grammars (narrative, formal, terse, custom) and corresponding `Cat` enum variants.

### Processing flow (replaces ADR 024 flow)

```
Input
  +-- "/" prefix -> caller handles (TUI slash commands)
  +-- Structural command keyword? -> classify_intent() fast-path (legacy, to be removed)
  +-- Everything else -> NLU pipeline
       +-- Tier 1 rule parser: try_dialogue_act() first, then normal parse
       +-- Tier 2/3/4 cascade if Tier 1 fails
       +-- Result: AbsTree node
            +-- Greeting/Farewell/Ack/FollowUp/MetaQuery -> DialogueManager
            +-- GoalRequest -> add goal + run OODA cycles
            +-- StructuralCommand -> handle_structural_command()
            +-- Assertable (Triple, Compound, etc.) -> TextIngestTool
            +-- Freeform -> escalate to autonomous goal
            +-- Other (entity refs, etc.) -> ground_query pipeline
```

### Dialogue state in KG

```
dialogue:operator  dlg:last-act     greeting
dialogue:operator  dlg:active-topic <entity>
dialogue:operator  dlg:turn-count   <count>
dialogue:operator  dlg:interlocutor operator
dialogue:operator  dlg:context-mt   dialogue:operator
```

The microtheory inherits from `mt:general` and is scoped per channel, so multiple concurrent conversations (TUI, WS, federation) maintain independent state.

## Consequences

### Positive

- **Unified pipeline**: All input goes through NLU and produces `AbsTree` nodes. No parallel classification paths.
- **KG-persistent dialogue state**: Active topic, last act, and turn count survive session boundaries and are queryable via SPARQL.
- **Composable with microtheories**: Dialogue state lives in a scoped microtheory, enabling context-dependent reasoning (e.g., different dialogue behavior per interlocutor).
- **Grammar-integrated**: Dialogue acts have `Cat` variants, GBNF rules, and linearization in all concrete grammars. The LLM translator can produce them; the grammars can render them.
- **Language-aware**: `try_dialogue_act()` uses Lexicon word lists, so dialogue detection works across all 5 supported languages without hardcoded strings.

### Negative

- **Deprecated code still present**: `classify_intent()`, `classify_conversational()`, and `ConversationalKind` remain in `nlp.rs` with `#[deprecated]` annotations. Structural command fast-path still calls `classify_intent()`. Full removal is deferred until all callers are migrated.
- **Transition period**: External code using `classify_intent()` directly will see deprecation warnings. Migration path is to use `ChatProcessor::process_input()` instead.

### Partially supersedes ADR 024

ADR 024 established `ChatProcessor` with NLU-first processing but retained `classify_intent()` as the primary dispatcher for structural commands and the fallback for NLU failures. This ADR extends it by making `AbsTree` dispatch the primary routing mechanism and relegating `classify_intent()` to a deprecated fallback.

## Files Changed

| File | Action |
|------|--------|
| `src/grammar/abs.rs` | Added 7 dialogue-act `AbsTree` variants, `Cat` mappings, constructors, tree traversal support |
| `src/grammar/cat.rs` | Added 7 `Cat` enum variants for dialogue acts with Display |
| `src/grammar/parser.rs` | Added `try_dialogue_act()` using Lexicon word lists, inserted as first parse step |
| `src/grammar/narrative.rs` | Added linearization arms for dialogue-act variants |
| `src/grammar/formal.rs` | Added linearization arms for dialogue-act variants |
| `src/grammar/terse.rs` | Added linearization arms for dialogue-act variants |
| `src/grammar/custom.rs` | Added linearization arms for dialogue-act variants |
| `src/nlu/abstree.gbnf` | Added `dialogue-act` production with 7 sub-rules |
| `src/agent/dialogue.rs` | Created: `DialoguePredicates`, `DialogueManager`, KG-backed state |
| `src/agent/agent.rs` | Added `dialogue_manager` field, init, accessors |
| `src/chat.rs` | Rewrote `dispatch_abstree()` to match on dialogue-act variants |
| `src/agent/nlp.rs` | Deprecated `classify_intent()`, `classify_conversational()`, `ConversationalKind` |
