# ADR 016 — OnlineHD Spam Classification

**Date:** 2026-02-21
**Status:** Accepted
**Phase:** 13b

## Context

Phase 13a established the email channel subsystem with JMAP/IMAP connectors,
MIME parsing, JWZ threading, and CommChannel integration. Incoming emails need
classification to separate spam from legitimate messages before triage (Phase 13c).

## Decision

Use a **VSA-native OnlineHD approach** combined with Robinson chi-square Bayesian
scoring and deterministic rule overrides:

1. **OnlineHD prototype learning** — maintain spam and ham prototype HyperVecs,
   updated via majority-vote bundling. No gradient descent, no batch retraining.
   Single-pass incremental learning with natural diminishing-contribution property.

2. **Robinson chi-square** — per-token spam/ham probability table with Fisher's
   method for combining the top-15 most informative tokens. Provides complementary
   signal to VSA similarity (catches specific spam vocabulary patterns).

3. **Deterministic overrides** — operator whitelist/blacklist by domain, plus
   automatic ham classification for mailing lists (List-Id header detection).

4. **Combined score** — `0.7 * VSA + 0.3 * Bayesian` with asymmetric thresholds
   (0.55 spam, 0.45 ham) creating an uncertain zone for conservative classification.

## Alternatives Considered

- **Pure Bayesian (SpamAssassin-style)**: Well-proven but doesn't integrate with
  the VSA infrastructure. Would be a parallel system rather than leveraging
  existing hypervector operations.

- **External ML model**: Would require async runtime, model files, additional
  crate dependencies. Violates the sync-only constraint of the agent module.

- **Pure VSA similarity**: Prototype similarity alone lacks the token-level
  granularity that catches specific spam vocabulary. The Bayesian supplement
  provides this without adding complexity.

## Consequences

- **No new crate dependencies** — uses existing VSA primitives (`encode_token`,
  `encode_label`, `bind`, `bundle`, `similarity`) and standard library math.
- **Sync-only** — all operations are synchronous, compatible with OODA loop.
- **Incremental** — classifier improves with each user feedback without retraining.
- **Persistent** — survives agent restarts via bincode + redb meta store.
- **Provenance-tracked** — every classification decision recorded as
  `DerivationKind::SpamClassification` for transparent reasoning.
- **Standalone module** — classifier is called explicitly, not auto-wired into
  EmailChannel polling. Full triage integration is deferred to Phase 13c.
