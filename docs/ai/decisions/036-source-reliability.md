# ADR 036 — Source Reliability & Credibility (Phase 18)

> Date: 2026-03-19
> Status: Accepted
> Phase: 18a-18c
> Depends on: Phase 17 (evidence theory), Phase 12d (interlocutor profiles)

## Context

Phase 17 gave the agent belief intervals for evidence combination, but treats all sources equally. In reality, some sources are more trustworthy than others, and some may be actively deceptive. The agent needs to assess source reliability, evaluate competing hypotheses, and detect manipulation attempts.

## Decision

### 18a — Admiralty Ratings & Bayesian Trust
- NATO Admiralty Code: source reliability (A-F) + information credibility (1-6)
- Bayesian Beta-distribution trust model: competence/benevolence/integrity dimensions
- Auto-rating from trust model, updated via claim verification
- Bridge to DS mass functions via `AdmiraltyRating.to_mass_function()`

### 18b — Analysis of Competing Hypotheses (ACH)
- Heuer's structured analytic technique (disconfirmation focus)
- Consistency matrix: evidence × hypothesis → {Consistent, Neutral, WeaklyInconsistent, StronglyInconsistent, NA}
- Diagnosticity: variance-based measure of evidence's discriminating power
- Ranking by ascending inconsistency (least rejected = most likely)
- Information gap detection for indistinguishable hypotheses

### 18c — Deception & Credibility Analysis
- Multi-signal credibility: factual support, self-consistency, cross-source consistency, bias, uncertainty
- Manipulation marker detection: urgency, flattery, threats, authority abuse
- Deception likelihood scoring (0.0–1.0)
- CredibilityRecommendation: Accept/AcceptWithCaution/SeekCorroboration/Discard/FlagForReview

## Consequences

- The agent can now assess WHO to trust, not just WHAT to believe
- ACH provides structured hypothesis evaluation for ambiguous situations
- Manipulation detection protects against social engineering
- MCP tools: 49 → **51** (rate_source, verify_source_claim)
- `DerivationKind::SourceReliabilityAssessed` (tag 82)
- 30 new tests (14 + 8 + 8)
