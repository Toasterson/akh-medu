# ADR 035 — Dempster-Shafer Evidence Theory (Phase 17)

> Date: 2026-03-19
> Status: Accepted
> Phase: 17
> Depends on: Phase 15 (causal world model — evidence weight from causal strength)

## Context

The agent's confidence system uses single scalar values (0.0–1.0) per triple. This conflates "strongly confirmed" with "no evidence against" — a triple with confidence 0.8 might mean "tested by 5 sources" or "asserted once with no disconfirmation." We need to distinguish confirmed knowledge from mere absence of contradiction, and to flag when sources conflict.

## Decision

Implement Dempster-Shafer (DS) belief functions over a binary frame {True, False} for evidence tracking, alongside the existing scalar confidence system (backward compatible).

### Core Types
- `MassFunction`: m_true + m_false + m_ignorance = 1.0
- `BeliefInterval`: [Bel(T), Pl(T)] where Bel = m_true, Pl = m_true + m_ignorance
- `EvidenceItem`: source_id + mass + timestamp + reliability
- `ClaimAssessment`: combined interval + verdict + conflict degree
- `ClaimVerdict`: WellSupported | Plausible | InsufficientEvidence | LikelyFalse | Conflicting

### Key Algorithms
- **Dempster's Rule**: principled combination of independent evidence sources
- **Reliability discounting**: unreliable sources' evidence pushed toward ignorance
- **Pignistic probability**: BetP(T) = m_true + m_ignorance/2 (bridge to scalar confidence)
- **Conflict detection**: K-factor monitoring with configurable alert/suppress thresholds

### Evidence Fusion Pipeline (17b)
- Automatic conflict alerting when K > 0.5
- Evidence compaction when exceeding max_evidence_per_claim
- `EvidenceFusionConfig` with tunable thresholds
- `ConflictAlert` with `ConflictRecommendation` (InvestigateFurther | FlagForOperator)

### What we chose NOT to do
- **Replace scalar confidence**: The existing `f32` confidence stays as primary interface. DS belief intervals are an optional enrichment via `assess_claim()`.
- **Full Transferable Belief Model (TBM)**: Simpler binary frame is sufficient for our use case. Multi-hypothesis frames would add complexity without clear benefit.
- **Automatic confidence migration**: One-time migration tool is available but not auto-run.

## Consequences

- The agent can now distinguish "confirmed" from "no evidence against" from "conflicting"
- Conflict detection flags unreliable or contradictory information
- MCP tools: 47 → **49** (assess_evidence, add_evidence)
- `DerivationKind::EvidenceCombination` (tag 81)
- 25 new tests
- Backward compatible — existing code sees no change

## Verification

- `cargo check` / `cargo check --features mcp` — clean
- 25 evidence unit tests pass
- Dempster's rule verified: agreeing sources increase belief, vacuous combination is identity
