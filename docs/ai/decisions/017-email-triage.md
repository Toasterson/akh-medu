# ADR 017: Email Triage & Priority

**Date**: 2026-02-21
**Status**: Accepted
**Phase**: 13c

## Context

Phase 13b provides spam/ham classification. Non-spam emails still need
prioritization: some are urgent, some are informational, some are noise.
We need a triage system that learns from operator behavior and routes
emails to appropriate queues.

## Decision

### Four-Feature Importance Model

Rather than a single classifier, we use a four-dimensional importance score
combining social, content, thread, and label signals:

- **Social (35%)**: Sender reputation based on reply rate, message frequency,
  recency, and relationship category. Uses exponential moving averages for
  smooth adaptation.
- **Content (25%)**: VSA similarity to important/low-priority OnlineHD
  prototypes (same pattern as spam classifier). Neutral (0.5) when untrained.
- **Thread (20%)**: Thread participation heuristics using in_reply_to and
  references.len() as proxies for engagement depth.
- **Label (20%)**: Operator-assigned route as a prior (Important=1.0, Feed=0.6,
  PaperTrail=0.3).

This decomposition is transparent (each component is independently interpretable)
and the weights are configurable.

### Hybrid Persistence

SenderStats HashMap is persisted via `put_meta` (bincode serialized) as the
authoritative source. Key metrics are also synced to KG triples via
`sync_sender_to_kg()` for SPARQL queryability. This dual approach gives us
fast incremental updates (put_meta) while keeping the KG queryable for
reasoning and explanation.

### HEY-Style Screening as Route Return

The triage pipeline returns `EmailRoute::ScreeningQueue` for first-time
senders with no routing assignment. The agent/CLI layer handles the actual
operator approval flow and calls `set_sender_routing()` to resolve it. This
keeps the triage module pure (no communication side-effects) while supporting
the HEY model of "screen before you see".

## Alternatives Considered

1. **Single combined score without decomposition**: Simpler but opaque.
   Users can't understand why an email was routed where it was.

2. **Full thread lookup in triage**: Would require passing thread trees into
   the triage module. Using in_reply_to/references as proxies is sufficient
   for prioritization and keeps the module self-contained.

3. **Storing sender stats only in KG**: Would require KG reads for every
   triage operation. The HashMap gives us O(1) lookups with periodic KG sync.

## Consequences

- Transparent four-feature scoring enables operator understanding and debugging
- Sender reputation builds over time via EMA — no cold-start cliff
- VSA prototypes enable content-aware routing without keyword lists
- ScreeningQueue pattern defers operator approval to a future UI layer
- 7 new KG predicates (sender: namespace) enable SPARQL queries over sender reputation
