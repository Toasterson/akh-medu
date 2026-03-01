# Conversation Capability Improvements

> Date: 2026-02-27

- **Status**: Planned
- **Relates to**: Phase 12b (grounded dialogue), Phase 14j-14m (NLU extension)
- **Depends on**: Conversational dispatch (committed 2026-02-27)
- **Supersedes**: None

## Context

The TUI now has basic conversational ability: greeting dispatch, follow-up
re-query, acknowledgment responses, meta-question routing, and anaphora
resolution from conversation history. All NLP classification uses the
language-aware Lexicon infrastructure (no hardcoded natural language constants).

This plan identifies concrete next-step improvements, organized from
lowest-effort/highest-impact to deepest architectural changes, and shows where
each connects to the upcoming NLU pipeline (Phase 14j-14m).

## Improvement Tiers

### Tier 1 — Immediate improvements (no new dependencies)

#### 1a. Richer conversation state tracking

**Problem**: `ConversationState` tracks `active_topic` (single SymbolId) and
`active_referents` (Vec of SymbolIds), but only the most recent turn's
referents are stored. Multi-turn coreference chains break.

**Solution**:
- Add a `topic_stack: Vec<SymbolId>` to ConversationState for nested topic tracking
  (e.g., "what is a dog?" → "what about its diet?" → "go back" → pops to dog)
- Decay old referents by turn age rather than replacing them entirely
- Track `last_assertion: Option<String>` so follow-ups like "why?" can
  reference the most recent assertion/fact presented

**Files**: `src/agent/conversation.rs`

#### 1b. Conversational Lexicon auto-detection

**Problem**: The TUI constructs `Lexicon::for_language(Language::default())`
which always gives English. A Russian-speaking user gets English greetings
matched but Russian ones missed.

**Solution**:
- Add `Lexicon::detect_language(text: &str) -> Language` using character-set
  heuristics (Cyrillic→Russian, Arabic script→Arabic, etc.) + bigram frequency
- Cache the detected language on `ConversationState` so subsequent turns don't
  re-detect
- Fall through: if detection fails, try all loaded Lexicons and pick the one
  with the highest classification score

**Files**: `src/grammar/lexer.rs`, `src/agent/conversation.rs`, `src/tui/mod.rs`

#### 1c. Graceful disambiguation

**Problem**: When the user says "it" and there's no active topic, the system
returns `SubjectNotFound`. There's no attempt to ask the user what they mean.

**Solution**:
- Add `ConversationalKind::ClarificationRequest` — the agent asks "What are
  you referring to?" instead of escalating to a goal
- Track an `awaiting_clarification: bool` on ConversationState so the next
  input is treated as the resolution

**Files**: `src/agent/nlp.rs`, `src/agent/conversation.rs`, `src/tui/mod.rs`

#### 1d. Persona-aware response templates in KG

**Problem**: `respond_conversational()` has hardcoded English response strings
("Hello. I am {name}.", "You're welcome!", etc.).

**Solution**:
- Store response templates as KG triples: `<persona> has-greeting-template <template>`
- Templates use `{name}`, `{topic}`, `{trait}` placeholders
- Fall back to current hardcoded defaults if templates are absent
- Skill packs can define custom persona responses per language

**Files**: `src/agent/conversation.rs`, new seed pack `seeds/persona-templates.toml`

### Tier 2 — Enhanced NLP (connects to Phase 14j)

#### 2a. Sentiment-aware acknowledgment

**Problem**: All acknowledgments get the same response. "That's terrible" and
"That's great" are treated identically.

**Solution**:
- Add a `sentiment_signal` field to ConversationalKind::Acknowledgment
- Use Lexicon word lists for positive/negative sentiment words
- Adjust response tone: empathetic for negative, encouraging for positive
- Phase 14k's DistilBERT can later provide more accurate sentiment

**Files**: `src/agent/nlp.rs`, `src/grammar/lexer.rs`, `src/agent/conversation.rs`

#### 2b. Intent confidence scoring

**Problem**: `classify_intent()` is first-match-wins with no confidence. An
input that slightly matches a query pattern commits fully to that path.

**Solution**:
- Return `(UserIntent, f32)` from `classify_intent()` where f32 is confidence
- Below a threshold (e.g., 0.4), fall through to `classify_conversational()`
  before declaring Freeform
- Phase 14m's VSA Parse Ranker directly replaces this with learned confidence

**Files**: `src/agent/nlp.rs`, `src/tui/mod.rs`, `src/main.rs`, `src/bin/akhomed.rs`

#### 2c. Ellipsis resolution

**Problem**: "What is a dog?" → "And a cat?" — the "and" + subject implies
the same question frame applied to a new subject. Currently classified as
Freeform.

**Solution**:
- Detect conjunction + bare noun as an elliptical continuation
- Copy the previous turn's question_word and apply to the new subject
- Store `last_query_frame: Option<(QuestionWord, QueryFocus)>` on ConversationState

**Files**: `src/agent/conversation.rs`, `src/agent/nlp.rs`, `src/tui/mod.rs`

### Tier 3 — Architectural (Phase 14j+ dependency)

#### 3a. Multi-turn dialogue manager

**Problem**: Each turn is processed independently. There's no dialogue policy
or state machine governing conversation flow.

**Solution**:
- Define dialogue states: `Open`, `TopicExploration`, `GoalTracking`,
  `Clarification`, `Farewell`
- Transitions triggered by ConversationalKind + intent + conversation history
- The dialogue manager sits above `classify_intent` and modifies routing
  (e.g., in TopicExploration state, bare nouns are treated as follow-up
  queries, not Freeform)
- Phase 14l's LLM fallback provides dialogue act classification for ambiguous
  transitions

**Files**: New module `src/agent/dialogue.rs`, `src/tui/mod.rs`

#### 3b. Grounded multi-sentence responses

**Problem**: Responses are single-fact ("X is a Y") or synthesis summaries.
The agent can't construct multi-sentence coherent paragraphs that weave
multiple facts together.

**Solution**:
- Use the existing `NarrativeSummary` with a new `conversational` grammar
  variant that produces shorter, more interactive prose
- Limit to 2-3 sentences per response in conversational mode
- Use discourse connectives ("Also,", "Furthermore,", "On the other hand,")
  from a Lexicon set

**Files**: `src/grammar/grammars/`, `src/agent/synthesize.rs`, `src/grammar/lexer.rs`

#### 3c. Proactive conversation contributions

**Problem**: The agent only responds; it never initiates. Even when it
discovers something interesting during idle processing, it doesn't bring it
up.

**Solution**:
- Add `ConversationContribution` struct: a message the agent wants to deliver
  when there's a natural pause
- Sources: idle discoveries, completed goals, watch firings, curiosity results
- Deliver during TUI idle ticks if the user hasn't typed in N seconds
- Respect proactivity level from preference system (Phase 13g)

**Files**: `src/agent/conversation.rs`, `src/tui/mod.rs`, `src/agent/idle.rs`

## Integration with NLU Pipeline (14j-14m)

| Improvement | Pre-14j approach | Post-14j upgrade |
|-------------|-----------------|------------------|
| 1b Language detection | Character-set heuristics | Tier 2 DistilBERT detects language as side effect |
| 2a Sentiment | Lexicon word lists | Tier 2 NER provides sentiment entity spans |
| 2b Intent confidence | Heuristic scoring | Tier 4 VSA Parse Ranker provides calibrated confidence |
| 2c Ellipsis resolution | Pattern matching on conjunctions | Tier 1 extended parser handles ellipsis natively |
| 3a Dialogue manager | State machine + heuristics | Tier 3 LLM classifies dialogue acts |
| 3b Multi-sentence | Grammar variant + connectives | Tier 1 extended grammar handles complex sentences |

## Implementation Priority

1. **1a** Topic stack + referent decay (small change, big impact on multi-turn)
2. **1b** Language auto-detection (removes default-English assumption)
3. **1c** Clarification requests (better UX than silent failure)
4. **2c** Ellipsis resolution (natural conversation pattern)
5. **1d** KG-stored response templates (enables multilingual persona)
6. **2a** Sentiment-aware responses (richer interaction)
7. **2b** Intent confidence (smoother classification boundaries)
8. **3a** Dialogue manager (proper conversation structure)
9. **3c** Proactive contributions (agent initiates)
10. **3b** Grounded multi-sentence (richer output)

## Verification

Each improvement should include:
- Unit tests with multilingual input
- Integration test verifying the TUI dispatch path
- No hardcoded natural language constants (all through Lexicon)
- Zero compiler warnings
