# Phase 17 — Dempster-Shafer Evidence Theory & Belief Intervals

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 17 (2 sub-phases: 17a–17b)
- **Depends on**: Phase 15 (causal predicates — uses causal strength for evidence weight)
- **Provenance tags**: 63–64

## Goal

Replace the agent's single-valued confidence scores with Dempster-Shafer belief intervals that explicitly represent ignorance. When the agent receives information from multiple sources, it should be able to combine evidence using principled mathematical rules, distinguish "confirmed true" from "no evidence against" from "conflicting evidence," and flag information that could be false.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  17a Belief Intervals  │  (Belief, Plausibility) per triple
                 │  + Mass Functions      │  Frame of discernment: {True, False}
                 │                       │  Dempster's rule for combining sources
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  17b Evidence Fusion   │  Multi-source evidence combination
                 │  Pipeline             │  Conflict detection across sources
                 │                       │  Ignorance-aware confidence propagation
                 │                       │  Integration with existing confidence system
                 └───────────────────────┘
```

## Sub-phases

### 17a — Belief Intervals & Mass Functions (~450 lines)

**New file**: `src/agent/evidence.rs`

**Types**:
```rust
/// A Dempster-Shafer mass function over the frame {True, False}.
///
/// For a binary frame, there are 4 focal elements:
/// - m(True): evidence supporting the proposition
/// - m(False): evidence against the proposition
/// - m({True,False}): ignorance (uncommitted belief)
/// - m(empty): always 0 (by definition)
///
/// Invariant: m_true + m_false + m_ignorance = 1.0
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MassFunction {
    /// Mass assigned to {True} — direct evidence for.
    pub m_true: f32,
    /// Mass assigned to {False} — direct evidence against.
    pub m_false: f32,
    /// Mass assigned to {True, False} — uncommitted (ignorance).
    pub m_ignorance: f32,
}

/// A belief interval [Bel, Pl] derived from a mass function.
///
/// Bel(True) = m_true (lower probability bound)
/// Pl(True)  = m_true + m_ignorance (upper probability bound)
/// The width (Pl - Bel) = m_ignorance measures uncertainty.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BeliefInterval {
    /// Lower bound on probability (belief).
    pub belief: f32,
    /// Upper bound on probability (plausibility).
    pub plausibility: f32,
}

/// Evidence from a single source about a proposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// The source (interlocutor, sensor, derivation) that provided this evidence.
    pub source_id: SymbolId,
    /// Mass function representing this source's evidence.
    pub mass: MassFunction,
    /// When the evidence was received.
    pub timestamp: u64,
    /// Source reliability rating (0.0-1.0), affects combination weight.
    pub source_reliability: f32,
    /// Optional: the derivation that produced this evidence.
    pub provenance_id: Option<u64>,
}

/// Well-known KG predicates for evidence tracking.
pub struct EvidencePredicates {
    pub has_evidence: SymbolId,
    pub evidence_mass_true: SymbolId,
    pub evidence_mass_false: SymbolId,
    pub evidence_mass_ignorance: SymbolId,
    pub evidence_source: SymbolId,
    pub evidence_conflict: SymbolId,
    pub belief_interval_lower: SymbolId,
    pub belief_interval_upper: SymbolId,
}

/// Result of combining evidence from multiple sources.
#[derive(Debug, Clone)]
pub struct CombinationResult {
    /// Combined mass function.
    pub combined: MassFunction,
    /// Combined belief interval.
    pub interval: BeliefInterval,
    /// Degree of conflict between sources (K factor before normalization).
    pub conflict_degree: f32,
    /// Number of sources combined.
    pub source_count: usize,
    /// Which sources conflicted most.
    pub conflicting_pairs: Vec<(SymbolId, SymbolId, f32)>,
}

/// Assessment of a claim's trustworthiness.
#[derive(Debug, Clone)]
pub struct ClaimAssessment {
    /// The proposition being assessed.
    pub claim_entity: SymbolId,
    /// Combined belief interval from all evidence.
    pub interval: BeliefInterval,
    /// Verdict based on the interval.
    pub verdict: ClaimVerdict,
    /// Evidence items considered.
    pub evidence: Vec<EvidenceItem>,
    /// Conflict between sources.
    pub conflict_degree: f32,
    /// Human-readable reasoning.
    pub reasoning: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimVerdict {
    /// High belief, low plausibility gap: well-supported.
    WellSupported,
    /// Moderate belief: plausible but not certain.
    Plausible,
    /// High ignorance: insufficient evidence.
    InsufficientEvidence,
    /// Low belief, evidence against: likely false.
    LikelyFalse,
    /// High conflict: sources disagree significantly.
    Conflicting,
}
```

**Key algorithms**:

**Dempster's Rule of Combination** (for independent sources):
```
m_combined(A) = (1/K) * sum_{B∩C=A} m1(B) * m2(C)
where K = 1 - sum_{B∩C=∅} m1(B) * m2(C)  (normalization)
```

For binary frame {T, F}:
```
K = m1(T)*m2(F) + m1(F)*m2(T)  // conflict mass
m_combined(T) = (m1(T)*m2(T) + m1(T)*m2(Θ) + m1(Θ)*m2(T)) / (1-K)
m_combined(F) = (m1(F)*m2(F) + m1(F)*m2(Θ) + m1(Θ)*m2(F)) / (1-K)
m_combined(Θ) = m1(Θ)*m2(Θ) / (1-K)
```

**Reliability-weighted combination**: Before combining, discount source mass by reliability:
```
m_discounted(T) = reliability * m(T)
m_discounted(F) = reliability * m(F)
m_discounted(Θ) = 1 - reliability + reliability * m(Θ)
```

**MassFunction** methods:
- `new(m_true, m_false) -> Self` — remainder goes to ignorance
- `from_confidence(conf) -> Self` — convert existing confidence score: m_true=conf, m_false=0, m_ignorance=1-conf
- `to_confidence() -> f32` — pignistic probability: m_true + m_ignorance/2
- `belief_interval() -> BeliefInterval`
- `combine(other) -> CombinationResult` — Dempster's rule
- `combine_multiple(masses) -> CombinationResult` — iterative Dempster's rule
- `discount(reliability) -> MassFunction` — reliability-weighted discounting
- `is_vacuous() -> bool` — m_ignorance > 0.99
- `conflict_with(other) -> f32` — K factor

**EvidenceManager** methods:
- `new(engine)` — init predicates
- `add_evidence(claim, item, engine)` — store evidence, update combined interval
- `assess_claim(claim, engine) -> ClaimAssessment` — gather all evidence, combine, verdict
- `conflict_between_sources(s1, s2, engine) -> f32` — average conflict across shared claims
- `most_uncertain_claims(engine, k) -> Vec<ClaimAssessment>` — claims with widest interval
- `migrate_confidence_to_belief(engine)` — one-time migration: existing confidence → MassFunction

**Provenance**: `DerivationKind::EvidenceCombination { source_count, conflict_degree, belief, plausibility }` (tag 63)

**Tests (~18)**:
1. mass_function_invariant
2. mass_function_from_confidence
3. mass_function_to_confidence_pignistic
4. belief_interval_from_mass
5. belief_interval_width_is_ignorance
6. combine_two_agreeing_sources
7. combine_two_conflicting_sources
8. combine_with_vacuous_source
9. combine_multiple_sources
10. discount_by_reliability
11. discount_unreliable_increases_ignorance
12. claim_verdict_well_supported
13. claim_verdict_insufficient_evidence
14. claim_verdict_conflicting
15. claim_verdict_likely_false
16. conflict_degree_zero_for_agreeing
17. conflict_degree_high_for_opposing
18. mass_function_serialization_roundtrip

### 17b — Evidence Fusion Pipeline (~350 lines)

**Input**: EvidenceManager from 17a + existing confidence system + communication channels

**Output**: Automatic evidence collection from all sources, conflict alerting, ignorance-aware reasoning

**Approach**:

1. **Evidence collection hooks**: When the agent receives information from any source (email, operator, federation, KG derivation), automatically create an EvidenceItem with:
   - source_id from the communication channel / derivation chain
   - reliability from InterlocutorProfile trust level or derivation confidence
   - mass from the claim's stated/inferred confidence

2. **Confidence bridge**: The existing `f32` confidence on triples remains as the primary interface. A new `confidence_with_interval(triple, engine)` function returns the full BeliefInterval. The existing `confidence` field stores the pignistic probability (best single-value estimate from the mass function).

3. **Conflict detection**: When Dempster's rule produces K > 0.5 (high conflict), emit a warning and flag the claim for human review. If K > 0.8, suppress the claim from responses (connect to Phase 12c constraint checking).

4. **Ignorance propagation**: When deriving new triples from existing ones, propagate ignorance: if premise has m_ignorance > 0.5, derived triple gets m_ignorance >= 0.5 (ignorance doesn't disappear through inference).

5. **Integration with constraint checking (Phase 12c)**: The `ConfidenceThresholds` per channel kind now operate on belief intervals:
   - Operator channel: emit if Bel > 0.3 (low bar, annotate uncertainty)
   - Trusted channel: emit if Bel > 0.5
   - Social/Public: emit if Bel > 0.7

**Types**:
```rust
/// Configuration for the evidence fusion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceFusionConfig {
    /// Conflict threshold for flagging (K factor).
    pub conflict_alert_threshold: f32,
    /// Conflict threshold for suppression.
    pub conflict_suppress_threshold: f32,
    /// Ignorance propagation factor (how much ignorance carries through inference).
    pub ignorance_propagation: f32,
    /// Maximum evidence items per claim before compaction.
    pub max_evidence_per_claim: usize,
}

/// An evidence conflict alert.
#[derive(Debug, Clone)]
pub struct ConflictAlert {
    pub claim: SymbolId,
    pub conflict_degree: f32,
    pub source_a: SymbolId,
    pub source_b: SymbolId,
    pub mass_a: MassFunction,
    pub mass_b: MassFunction,
    pub recommendation: ConflictRecommendation,
}

#[derive(Debug, Clone, Copy)]
pub enum ConflictRecommendation {
    /// Seek additional evidence.
    InvestigateFurther,
    /// Prefer the more reliable source.
    PreferSource(SymbolId),
    /// Cannot resolve automatically.
    FlagForOperator,
}
```

**Provenance**: `DerivationKind::EvidenceConflictDetected { claim_id_raw, conflict_degree, source_count }` (tag 64)

**Tests (~10)**:
1. evidence_from_email_creates_item
2. evidence_from_operator_high_reliability
3. evidence_from_federation_social_reliability
4. conflict_alert_high_k
5. conflict_suppress_very_high_k
6. ignorance_propagation_through_derivation
7. confidence_bridge_roundtrip
8. belief_interval_threshold_operator
9. belief_interval_threshold_public
10. evidence_compaction_oldest_removed

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New field: `pub(crate) evidence_manager: EvidenceManager`
- Init/restore/persist lifecycle

### Module registry (`src/agent/mod.rs`)
- `pub mod evidence;`
- Re-exports

### Error (`src/agent/error.rs`)
- `Evidence(#[from] super::evidence::EvidenceError)` variant

### Provenance (`src/provenance.rs`)
- Tags 63–64: EvidenceCombination, EvidenceConflictDetected

### Constraint Checking (`src/agent/constraint_check.rs`)
- `confidence_check()` extended to use belief intervals when available
- New warning type: `ConstraintWarning::HighIgnorance { claim, m_ignorance }`

### OODA (`src/agent/ooda.rs`)
- In `orient()`: assess incoming information via evidence pipeline
- Conflict alerts pushed to working memory

### NLP (`src/agent/nlp.rs`)
- `UserIntent::EvidenceQuery { claim }` — "how confident are you about...", "what's the evidence for..."

### CLI (`src/main.rs`)
- `Commands::Evidence { action: EvidenceAction }` with subcommands: Assess { entity }, Conflicts, Uncertain { count }

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~28 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh evidence assess "climate change"` shows belief interval + sources, `akh evidence conflicts` lists conflicting claims, `akh evidence uncertain 5` shows top-5 most uncertain claims
