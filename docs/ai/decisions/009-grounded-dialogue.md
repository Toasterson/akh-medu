# ADR 009 — Grounded Operator Dialogue

> Date: 2026-02-21
> Status: Accepted
> Phase: 12b

## Context

Phase 12a introduced the `CommChannel` trait and `OperatorChannel`. Queries were
handled through two separate paths (discourse-aware response via grammar module,
or synthesis-based fallback) without structured provenance or configurable detail
levels. The agent's responses had no formal backing in KG triples and no way for
the operator to tune verbosity.

## Decision

Introduce a grounding pipeline and conversation state manager:

1. **`ConversationState`** — bounded ring buffer of `ConversationTurn` entries
   (max 10) with `Speaker` tracking, active referents (recently-discussed
   entities), active topic, grammar preference, and configurable `ResponseDetail`
   (Concise / Normal / Full).

2. **`GroundedResponse`** — structured response backed by `GroundedTriple` entries
   harvested from the KG, with aggregate confidence, provenance IDs, grammar
   archetype, and `render(detail)` method that produces detail-level-appropriate
   prose.

3. **`ground_query(subject, engine, grammar)` pipeline** — resolves the subject
   entity, collects all triples, filters out agent-metadata predicates, extracts
   provenance and derivation tags, synthesizes prose, and returns a
   `GroundedResponse` with full provenance chain.

4. **`ResponseContent::Grounded` variant** on `OutboundMessage` — extends the
   Phase 12a message protocol with rendered prose, grammar, and triple count,
   enabling downstream consumers to distinguish grounded from legacy responses.

5. **`SetDetail` intent** in NLP and channel message classifier — allows operators
   to type "set detail concise/normal/full" or use `/detail <level>` command.

6. **Grounded-first query path** — TUI and headless chat attempt `ground_query`
   before falling back to discourse-aware and synthesis paths. When grounding
   succeeds, the response carries provenance and the turn is recorded in
   conversation state.

## Alternatives Considered

- **Always use discourse path**: Rejected because it has no provenance tracking
  and no structured confidence model. Grounding adds these without removing the
  discourse fallback.

- **LLM-based response generation**: Out of scope — akh-medu is a zero-LLM
  architecture. Grounding is deterministic and interpretable.

- **Store conversation state in KG**: Considered for Phase 12c+ but premature.
  In-memory `ConversationState` with `VecDeque` is sufficient and avoids KG
  pollution with ephemeral data.

## Consequences

- Query responses now carry provenance and confidence when the subject is found.
- Operators can tune verbosity without restarting the session.
- The grounding pipeline is a pure function of KG state — no side effects.
- Conversation state is ephemeral (not persisted across sessions), appropriate
  for Phase 12b. Session persistence can be added when needed.
- The `ResponseContent::Grounded` variant enables downstream rendering decisions
  (e.g., Phase 12f explanation generation).
