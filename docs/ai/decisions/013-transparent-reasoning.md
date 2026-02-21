# ADR 013 — Transparent Reasoning and Explanations

> Date: 2026-02-21
> Status: Accepted
> Phase: 12f

## Context

The provenance ledger (redb-backed, with derived/source/kind indices) records how
every piece of knowledge was derived, but this information was only accessible
programmatically. The agent couldn't answer "why do you believe X?" or "how
confident are you about X?" in natural conversation. Phase 12b's grounded
dialogue showed provenance IDs but didn't produce human-readable explanations.

## Decision

Introduce an `explain` module with a provenance-to-prose pipeline that walks
derivation chains and renders them as structured explanations.

### Architecture

1. **Five explanation query types** — `ExplanationQuery` enum with `parse()`:
   - `Why { subject }` — trace provenance tree for an entity
   - `How { subject }` — same as Why (decision reasoning)
   - `WhatKnown { subject }` — enumerate all triples with provenance
   - `HowConfident { subject }` — aggregate confidence + evidence breakdown
   - `WhatChanged` — KG diff since last session

2. **DerivationNode tree** — recursive structure built by walking
   `Engine::provenance_of()` records. Uses a `HashSet<u64>` visited set
   for cycle detection and `max_depth` cap (default 5) to prevent infinite
   recursion on deep or circular provenance chains.

3. **Two rendering modes**:
   - `render_derivation_tree()` — indented hierarchy with box-drawing
     connectors (for Full detail mode)
   - `render_derivation_prose()` — comma-separated concise format (for
     Normal detail mode)

4. **Comprehensive derivation prose** — `derivation_kind_prose()` covers
   all 48 `DerivationKind` variants with human-readable descriptions.

5. **NLP integration** — `UserIntent::Explain` variant checked before
   `UserIntent::Query` in `classify_intent()`, ensuring "why X?" is
   treated as an explanation request rather than a factual query.

6. **Metadata filtering** — all explanation functions filter out
   agent-internal metadata triples (via `is_metadata_label()`), showing
   only user-facing knowledge.

## Alternatives Considered

- **LLM-based explanations**: Rejected — the system is LLM-free by design.
  Rule-based provenance walking produces deterministic, verifiable explanations.

- **AbsTree linearization**: Considered building DerivationNodes as AbsTree
  nodes and linearizing through the grammar system. Rejected for now —
  explanation rendering is simpler than discourse generation, and direct
  string formatting is clearer. Can be added later if richer linearization
  is needed.

- **Reasoning journal (Winter-inspired)**: Described in the Phase 12f plan
  as optional. Deferred — can be added as structured `AkhMessage::Reasoning`
  entries from the OODA loop in a follow-up phase.

## Consequences

- The agent can explain any piece of its knowledge via natural language
  ("why dogs?", "explain Rust", "how confident are you about X?").
- Explanations are fully grounded in the provenance ledger — no hallucination.
- The `ExplanationQuery::parse()` mechanism is extensible for future query
  types without modifying the NLP classifier.
- The `derivation_kind_prose()` function serves as a single source of truth
  for human-readable derivation descriptions, usable by other modules.
