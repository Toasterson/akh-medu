# ADR 011 — Social Knowledge Graph with Theory of Mind

> Date: 2026-02-21
> Status: Accepted
> Phase: 12d

## Context

After Phase 12c established constraint-checked communication, the agent had no model
of who it was talking to. Every interaction was anonymous — there was no tracking of
trust levels, interests, or what knowledge each person had been exposed to. The agent
needs per-interlocutor profiles to enable social reasoning, personalized responses,
and theory of mind.

## Decision

Introduce an `InterlocutorRegistry` on the Agent that models each interlocutor as a
KG entity with well-known predicates, a personal microtheory for theory of mind, and
VSA interest vectors for similarity discovery.

1. **InterlocutorProfile** — struct containing: string ID, KG SymbolId, channel IDs,
   trust level (mapped to ChannelKind), personal microtheory SymbolId, interest list,
   interaction count, and last interaction timestamp.

2. **InterlocutorPredicates** — 6 well-known KG relations (`interlocutor:has-channel`,
   `interlocutor:has-trust-level`, `interlocutor:has-interest`,
   `interlocutor:last-interaction`, `interlocutor:interaction-count`,
   `interlocutor:has-knowledge-mt`), initialized via `engine.resolve_or_create_relation`.

3. **Registration** — `register()` creates a KG entity, a personal microtheory
   (`mt:{name}-knowledge` with `ContextDomain::Belief`), and stores trust/channel/mt
   triples. Re-registration updates channel list and interaction tracking.

4. **Interest modeling** — `add_interest()` records interest concepts in the KG and
   rebuilds a VSA interest vector by bundling item memory vectors for each concept.
   `find_similar()` performs Hamming similarity search across all interest vectors.
   `interest_overlap()` computes pairwise similarity.

5. **Theory of mind** — `record_knowledge()` stores `interlocutor:knows` triples in
   the interlocutor's personal microtheory (compartment-scoped), representing what
   the agent believes the interlocutor knows.

6. **Trust immutability** — The operator's trust level is always `ChannelKind::Operator`
   and cannot be demoted. `set_trust_level()` returns `OperatorImmutable` error.

7. **Auto-registration** — `Agent::ensure_interlocutor()` is called in both TUI
   `process_inbound_local()` and headless chat on every inbound message, creating
   profiles transparently on first interaction.

## Alternatives Considered

- **Profiles stored entirely in KG**: Rejected for performance — frequent lookups
  during message processing would require many SPARQL queries. The registry provides
  O(1) HashMap access with KG as backing store.

- **Global microtheory for all interlocutors**: Too coarse — per-person microtheories
  via Phase 9a's `create_context()` provide isolation and enable scoped queries about
  what a specific person knows.

- **HNSW-based interest similarity**: Considered for `find_similar()` but the number
  of interlocutors is typically small enough that brute-force Hamming comparison is
  sufficient. HNSW can be added later if the registry grows large.

## Consequences

- Every interlocutor gets a KG entity and a personal microtheory on first message.
- Interest vectors enable discovery of interlocutors with similar interests.
- Trust levels map directly to ChannelKind, which integrates with the existing
  capability model (Phase 12a) and constraint checking (Phase 12c).
- The `record_knowledge()` API is ready for use by response tracking — when the
  agent tells someone a fact, it can record that they now know it.
- Profiles are in-memory only for now; persistence can be added via bincode
  serialization in `persist_session()` if needed.
