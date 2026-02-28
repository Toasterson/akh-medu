# VSA-Native Dialogue System

> Date: 2026-02-28

- **Status**: Complete
- **Relates to**: Phase 14j-14m (NLU extension), ADR 024 (ChatProcessor unification)
- **Supersedes**: Tier 3a from `2026-02-27-conversation-improvements.md` (dialogue manager)
- **ADR**: `docs/ai/decisions/025-vsa-dialogue-system.md`

## Summary

Replaced the regex-based intent classification (`classify_intent()`, `classify_conversational()`) with NLU-first AbsTree dispatch. Dialogue acts are now first-class AbsTree variants parsed by the rule parser and LLM translator, dispatched directly by `ChatProcessor`, and backed by KG-persistent state via `DialogueManager`.

## Changes

### AbsTree extensions (`src/grammar/abs.rs`)

Added 7 dialogue-act variants to the `AbsTree` enum:
- `Greeting { addressee: Option<Box<AbsTree>> }`
- `Farewell { addressee: Option<Box<AbsTree>> }`
- `Acknowledgment { referent: Option<Box<AbsTree>> }`
- `FollowUpRequest { topic: Option<Box<AbsTree>> }`
- `MetaQuery { about: Box<AbsTree> }`
- `GoalRequest { description: Box<AbsTree> }`
- `StructuralCommand { command: String, args: Vec<String> }`

Added corresponding constructors, `Cat` mappings, tree traversal (`label()`, `collect_labels()`, `depth()`, `node_count()`), and `is_assertable()` exclusion (dialogue acts are not assertable).

### Cat enum (`src/grammar/cat.rs`)

Added 7 `Cat` variants: `Greeting`, `Farewell`, `Acknowledgment`, `FollowUpRequest`, `MetaQuery`, `GoalRequest`, `StructuralCommand` with `Display` implementations.

### Rule parser (`src/grammar/parser.rs`)

Added `try_dialogue_act()` function that runs as the first parse step (before commands and questions). Uses `Lexicon` word lists for language-aware matching:
- Greetings: "hello", "hi", "hey", etc. (all 5 languages via Lexicon)
- Farewells: "goodbye", "bye", "farewell", etc.
- Acknowledgments: "ok", "thanks", "understood", etc.
- Follow-ups: "tell me more", "more", "continue", etc.
- Meta-queries: "who are you", "what can you do", etc.

### GBNF grammar (`src/nlu/abstree.gbnf`)

Added `dialogue-act` production rule with 7 sub-rules (`greeting`, `farewell`, `acknowledgment`, `follow-up-request`, `meta-query`, `goal-request`, `structural-command`) so the Qwen2.5-1.5B LLM translator can produce dialogue-act nodes via GBNF-constrained decoding.

### DialogueManager (`src/agent/dialogue.rs`)

Created a new module with:
- `DialoguePredicates` — 5 well-known relations (`dlg:active-topic`, `dlg:last-act`, `dlg:turn-count`, `dlg:interlocutor`, `dlg:context-mt`)
- `DialogueManager` — per-channel dialogue state backed by KG microtheory (`dialogue:<channel_id>`)
- `dialogue_act_label()` — extract act type label from AbsTree node
- Response generators: `handle_greeting()`, `handle_farewell()`, `handle_ack()`, `handle_meta_query()`, `handle_follow_up()`
- KG state management: `record_turn()`, `set_last_act()`, `set_active_topic()`

### Agent wiring (`src/agent/agent.rs`)

- Added `dialogue_manager: DialogueManager` field to `Agent`
- Initialized with `DialoguePredicates::init(engine)` and channel `"operator"`
- Added `dialogue_manager()` / `dialogue_manager_mut()` accessors

### ChatProcessor rewrite (`src/chat.rs`)

Rewrote `dispatch_abstree()` to match on AbsTree variants:
- `Greeting` / `Farewell` / `Acknowledgment` / `FollowUpRequest` / `MetaQuery` -> DialogueManager handlers
- `GoalRequest` -> add goal + run OODA cycles
- `StructuralCommand` -> handle_structural_command()
- Assertable structures -> TextIngestTool ingestion
- `Freeform` -> escalate to autonomous goal
- Other -> ground_query pipeline

Added `record_turn()` call on every NLU-parsed input to maintain dialogue state.

### Grammar linearization (`src/grammar/narrative.rs`, `formal.rs`, `terse.rs`, `custom.rs`)

Added linearization arms for all 7 dialogue-act variants in each concrete grammar so they can be rendered as natural language output.

### Deprecations (`src/agent/nlp.rs`)

Marked with `#[deprecated(note = "...ADR 025...")]`:
- `classify_intent()` — "Use NLU pipeline -> AbsTree dispatch instead"
- `classify_conversational()` — "Use NLU pipeline's try_dialogue_act() instead"
- `ConversationalKind` enum — "Use AbsTree dialogue-act variants instead"

## Files Changed

| File | Change |
|------|--------|
| `src/grammar/abs.rs` | 7 new AbsTree variants, constructors, Cat mappings, traversal |
| `src/grammar/cat.rs` | 7 new Cat variants with Display |
| `src/grammar/parser.rs` | `try_dialogue_act()` using Lexicon word lists |
| `src/grammar/narrative.rs` | Dialogue-act linearization |
| `src/grammar/formal.rs` | Dialogue-act linearization |
| `src/grammar/terse.rs` | Dialogue-act linearization |
| `src/grammar/custom.rs` | Dialogue-act linearization |
| `src/nlu/abstree.gbnf` | `dialogue-act` production with 7 sub-rules |
| `src/agent/dialogue.rs` | New module: DialoguePredicates, DialogueManager |
| `src/agent/agent.rs` | dialogue_manager field, init, accessors |
| `src/chat.rs` | AbsTree variant dispatch, DialogueManager integration |
| `src/agent/nlp.rs` | Deprecated classify_intent(), classify_conversational(), ConversationalKind |

## Verification

- All dialogue acts detected by rule parser in Tier 1 (<1ms, 0 MB)
- GBNF grammar validates: LLM translator can produce dialogue-act JSON
- ChatProcessor dispatches all 7 dialogue-act variants correctly
- DialogueManager records turns and updates KG state
- Deprecated functions compile with warnings; no callers broken
- All concrete grammars linearize dialogue acts
