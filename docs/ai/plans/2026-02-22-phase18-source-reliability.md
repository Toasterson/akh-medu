# Phase 18 — Source Reliability, ACH & Credibility Assessment

> Date: 2026-02-22
> Research: `docs/ai/decisions/020-predictive-planning-epistemic-research.md`

- **Status**: Planned
- **Phase**: 18 (3 sub-phases: 18a–18c)
- **Depends on**: Phase 17 (evidence theory — uses belief intervals for claim assessment), Phase 12d (interlocutor profiles)
- **Provenance tags**: 65–67

## Goal

Give the agent the ability to systematically assess the reliability of information sources, evaluate competing hypotheses using structured intelligence analysis methods, and detect potential deception or manipulation from sources that may have different goals than the agent.

## Architecture Overview

```
                 ┌───────────────────────┐
                 │  18a Admiralty System  │  Source reliability A-F
                 │  + Bayesian Trust     │  Information credibility 1-6
                 │                       │  Bayesian trust updating from outcomes
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  18b Analysis of      │  Hypothesis matrix
                 │  Competing Hypotheses │  Evidence diagnosticity
                 │  (ACH)               │  Disconfirmation-focused evaluation
                 └──────────┬────────────┘
                            │
                 ┌──────────▼────────────┐
                 │  18c Deception &      │  Consistency checking against KG
                 │  Credibility Analysis │  Source behavior anomaly detection
                 │                       │  Credibility signal extraction
                 │                       │  Adversarial intent modeling
                 └───────────────────────┘
```

## Sub-phases

### 18a — Admiralty Ratings & Bayesian Trust Model (~400 lines)

**New file**: `src/agent/source_reliability.rs`

**Types**:
```rust
/// NATO Admiralty Code: source reliability rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceReliability {
    /// A — Completely reliable. No doubt about authenticity, trustworthiness.
    CompletelyReliable,
    /// B — Usually reliable. Minor doubts, generally trustworthy.
    UsuallyReliable,
    /// C — Fairly reliable. Some doubts about authenticity.
    FairlyReliable,
    /// D — Not usually reliable. Significant doubts.
    NotUsuallyReliable,
    /// E — Unreliable. Shown to be unreliable in past.
    Unreliable,
    /// F — Reliability cannot be judged. New or unknown source.
    CannotJudge,
}

/// NATO Admiralty Code: information credibility rating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InformationCredibility {
    /// 1 — Confirmed by other independent sources.
    Confirmed,
    /// 2 — Probably true. Consistent with known information.
    ProbablyTrue,
    /// 3 — Possibly true. Not confirmed or denied.
    PossiblyTrue,
    /// 4 — Doubtful. Inconsistent with known information.
    Doubtful,
    /// 5 — Improbable. Contradicted by known information.
    Improbable,
    /// 6 — Truth cannot be judged. No basis for evaluation.
    CannotJudge,
}

/// Combined Admiralty rating for an information item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmiraltyRating {
    pub source: SourceReliability,
    pub information: InformationCredibility,
}

/// Bayesian trust model: maintains evidence for source competence, benevolence, integrity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustModel {
    /// Prior trust parameters (Beta distribution: alpha, beta for each dimension).
    pub competence: BetaDistribution,
    pub benevolence: BetaDistribution,
    pub integrity: BetaDistribution,
    /// Number of observations used to build this model.
    pub observation_count: u32,
}

/// Beta distribution parameters for Bayesian trust tracking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaDistribution {
    /// Positive evidence count (successes).
    pub alpha: f32,
    /// Negative evidence count (failures).
    pub beta: f32,
}

/// Observation for trust updating.
#[derive(Debug, Clone)]
pub struct TrustObservation {
    /// What dimension this observation informs.
    pub dimension: TrustDimension,
    /// Outcome: positive (correct/helpful) or negative.
    pub positive: bool,
    /// How strongly this observation weighs (0.0–1.0).
    pub weight: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum TrustDimension {
    /// Source provided accurate information.
    Competence,
    /// Source acted in the agent's interest.
    Benevolence,
    /// Source was honest and consistent.
    Integrity,
}

/// Source reliability predicates for KG.
pub struct ReliabilityPredicates {
    pub has_admiralty_source_rating: SymbolId,
    pub has_admiralty_info_rating: SymbolId,
    pub trust_competence: SymbolId,
    pub trust_benevolence: SymbolId,
    pub trust_integrity: SymbolId,
    pub claim_verified: SymbolId,
    pub claim_falsified: SymbolId,
    pub prediction_accuracy: SymbolId,
}
```

**Key algorithms**:

**Bayesian trust update**:
```
For each trust observation:
  if positive: alpha += weight
  else: beta += weight
Expected trust = alpha / (alpha + beta)
Confidence in estimate = (alpha + beta) / (1 + alpha + beta)
```

**Admiralty auto-rating**: Based on trust model:
```
reliability = weighted_average(competence, benevolence, integrity, weights=[0.4, 0.3, 0.3])
A if reliability > 0.9, B if > 0.75, C if > 0.6, D if > 0.4, E if > 0.2, F otherwise
```

**SourceReliabilityManager** methods:
- `new(engine)` — init predicates
- `rate_source(source_id) -> AdmiraltyRating` — compute from trust model
- `rate_information(claim, engine) -> InformationCredibility` — check against KG for confirmation/contradiction
- `update_trust(source_id, observation)` — Bayesian update
- `verify_claim(source_id, claim, actual_truth, engine)` — when claim truth is determined, update source trust
- `most_reliable_sources(k) -> Vec<(SymbolId, AdmiraltyRating)>`
- `least_reliable_sources(k) -> Vec<(SymbolId, AdmiraltyRating)>`
- `evidence_mass_from_rating(rating) -> MassFunction` — convert Admiralty rating to DS mass for evidence combination
- `persist(engine)` / `restore(engine)`

**Integration with InterlocutorProfile (Phase 12d)**: The InterlocutorProfile gains:
- `trust_model: TrustModel` field
- `admiralty_rating() -> AdmiraltyRating` derived method
- Auto-update: when a claim from this interlocutor is verified/falsified, update trust model

**Provenance**: `DerivationKind::SourceReliabilityAssessed { source_id_raw, reliability, credibility }` (tag 65)

**Tests (~12)**:
1. admiralty_source_labels_roundtrip
2. admiralty_info_labels_roundtrip
3. beta_distribution_expected_value
4. bayesian_trust_update_positive
5. bayesian_trust_update_negative
6. bayesian_trust_convergence
7. auto_rating_from_trust_model
8. rate_information_confirmed
9. rate_information_contradicted
10. evidence_mass_from_admiralty
11. trust_model_serialization
12. verify_claim_updates_trust

### 18b — Analysis of Competing Hypotheses (ACH) (~400 lines)

**New file**: `src/agent/ach.rs`

**Input**: Claims/evidence from EvidenceManager (Phase 17) + source ratings from 18a

**Output**: Structured evaluation of competing hypotheses about a situation

**Types**:
```rust
/// An ACH analysis session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchAnalysis {
    /// Unique ID for this analysis.
    pub id: String,
    /// The situation or question being analyzed.
    pub question: String,
    /// Competing hypotheses.
    pub hypotheses: Vec<AchHypothesis>,
    /// Evidence items.
    pub evidence: Vec<AchEvidence>,
    /// Consistency matrix: evidence[i] vs hypothesis[j] -> rating.
    pub matrix: Vec<Vec<ConsistencyRating>>,
    /// Ranked hypotheses (least inconsistent first).
    pub ranking: Vec<(usize, f32)>,
    /// Key diagnostics: evidence that differentiates between hypotheses.
    pub diagnostics: Vec<usize>,
    /// Timestamp.
    pub created_at: u64,
}

/// A hypothesis in an ACH analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchHypothesis {
    /// Human-readable description.
    pub description: String,
    /// Optional: KG entity this hypothesis corresponds to.
    pub entity: Option<SymbolId>,
    /// Weighted inconsistency score (lower = more likely).
    pub inconsistency_score: f32,
}

/// An evidence item in an ACH analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AchEvidence {
    /// Description of the evidence.
    pub description: String,
    /// Source reliability (from Admiralty rating).
    pub source_reliability: SourceReliability,
    /// Diagnosticity: how much this evidence differentiates hypotheses.
    pub diagnosticity: f32,
    /// Optional: KG entity or provenance ID.
    pub source: Option<SymbolId>,
}

/// How consistent an evidence item is with a hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyRating {
    /// Strongly supports (consistent, expected).
    Consistent,
    /// Neutral (neither supports nor contradicts).
    Neutral,
    /// Weakly inconsistent (unexpected but not impossible).
    WeaklyInconsistent,
    /// Strongly inconsistent (contradicts hypothesis).
    StronglyInconsistent,
    /// Not applicable (evidence irrelevant to this hypothesis).
    NotApplicable,
}
```

**Key algorithms**:

**ACH evaluation** (Heuer's method, disconfirmation-focused):
1. For each hypothesis, sum weighted inconsistencies: `score_h = sum(weight(e) * inconsistency(e,h))` where weight depends on source reliability and diagnosticity
2. Rank by ascending inconsistency (most consistent = least rejected)
3. Compute diagnosticity: evidence that has different ratings across hypotheses

**Diagnosticity score**:
```
diagnosticity(e) = variance(consistency_ratings(e, all_hypotheses))
High variance → evidence differentiates hypotheses → more diagnostic
```

**AchManager** methods:
- `create_analysis(question, hypotheses, evidence) -> AchAnalysis`
- `rate_consistency(analysis, evidence_idx, hypothesis_idx, rating)` — fill matrix cell
- `auto_rate_from_kg(analysis, engine)` — automatically rate by checking evidence against KG
- `evaluate(analysis) -> AchAnalysis` — compute rankings and diagnostics
- `sensitivity_analysis(analysis, hypothesis_idx) -> Vec<(usize, f32)>` — how much would each evidence item need to change to alter the ranking?
- `identify_information_gaps(analysis) -> Vec<String>` — missing evidence that would improve discrimination
- `render_matrix(analysis, engine) -> String` — formatted text table

**Provenance**: `DerivationKind::AchAnalysis { hypothesis_count, evidence_count, top_hypothesis }` (tag 66)

**Tests (~12)**:
1. ach_analysis_creation
2. consistency_rating_weights
3. evaluate_ranks_by_inconsistency
4. diagnosticity_high_for_differentiating
5. diagnosticity_low_for_uniform
6. sensitivity_analysis_identifies_pivotal
7. information_gaps_suggest_missing
8. auto_rate_from_kg_uses_triples
9. render_matrix_readable
10. two_hypotheses_one_evidence
11. multiple_hypotheses_complex
12. serialization_roundtrip

### 18c — Deception & Credibility Analysis (~450 lines)

**New file**: `src/agent/credibility.rs`

**Input**: Source reliability (18a) + evidence theory (17) + existing constraint checking (12c)

**Output**: Deception detection, credibility scoring, adversarial intent modeling

**Types**:
```rust
/// Credibility assessment signals (based on 2024 survey of 175 papers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredibilitySignals {
    /// Does the claim have factual support in the KG?
    pub factual_support: f32,
    /// Does the source show bias patterns?
    pub bias_indicator: f32,
    /// Consistency with the source's own past claims.
    pub self_consistency: f32,
    /// Consistency with claims from other reliable sources.
    pub cross_source_consistency: f32,
    /// Does the claim contain hedging/uncertainty markers?
    pub uncertainty_markers: f32,
    /// How recently has this source provided verifiable accurate claims?
    pub recency_of_verified_claims: f32,
    /// Combined credibility score (weighted average).
    pub combined_score: f32,
}

/// Deception indicators detected in a message or claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeceptionIndicators {
    /// Contradicts known facts in the KG.
    pub contradicts_known_facts: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Contradicts the source's own prior claims.
    pub self_contradictions: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Causal claims that are implausible given the causal model.
    pub implausible_causal_claims: Vec<String>,
    /// Appeals to urgency or emotion without evidence.
    pub manipulation_markers: Vec<String>,
    /// Overall deception likelihood (0.0 = certainly honest, 1.0 = certainly deceptive).
    pub deception_likelihood: f32,
    /// Reasoning for the assessment.
    pub reasoning: String,
}

/// Model of what a source might be trying to achieve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialIntentModel {
    /// What the source appears to want the agent to believe.
    pub desired_beliefs: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// What the source appears to want the agent to do.
    pub desired_actions: Vec<SymbolId>,
    /// Estimated alignment with the agent's goals (1.0 = fully aligned, -1.0 = adversarial).
    pub goal_alignment: f32,
    /// Confidence in this intent model.
    pub confidence: f32,
}

/// VSA role vectors for credibility encoding.
pub struct CredibilityRoleVectors {
    pub factual: HyperVec,
    pub consistency: HyperVec,
    pub source_history: HyperVec,
    pub linguistic: HyperVec,
    pub causal_plausibility: HyperVec,
}

/// Result of credibility analysis on a message.
#[derive(Debug, Clone)]
pub struct CredibilityAnalysis {
    pub message_entity: SymbolId,
    pub source_id: SymbolId,
    pub signals: CredibilitySignals,
    pub deception: DeceptionIndicators,
    pub adversarial_intent: Option<AdversarialIntentModel>,
    pub recommendation: CredibilityRecommendation,
}

#[derive(Debug, Clone, Copy)]
pub enum CredibilityRecommendation {
    /// Accept with standard confidence.
    Accept,
    /// Accept but with reduced confidence.
    AcceptWithCaution,
    /// Seek corroboration before accepting.
    SeekCorroboration,
    /// Likely deceptive; discard.
    Discard,
    /// Flag for operator review.
    FlagForReview,
}
```

**CredibilityAnalyzer** methods:
- `new(engine)` — init role vectors
- `analyze_message(msg, source, engine) -> CredibilityAnalysis` — full pipeline
- `check_factual_support(claims, engine) -> f32` — verify claims against KG
- `check_self_consistency(source, claims, engine) -> f32` — compare against source's past claims
- `check_cross_source(claims, engine) -> f32` — compare against other reliable sources
- `detect_manipulation_markers(text) -> Vec<String>` — urgency, flattery, threats, emotional appeal
- `model_adversarial_intent(source, claims, engine) -> AdversarialIntentModel` — infer what the source wants
- `encode_credibility_profile(ops, signals) -> HyperVec` — VSA encoding for similarity-based pattern detection
- `is_similar_to_known_deception(profile, ops, engine) -> f32` — compare against patterns from past deceptive sources

**Key algorithm — adversarial intent modeling**:
1. Collect all claims from the source in the current conversation
2. Identify which of the agent's goals would be affected if these claims were believed
3. Determine whether believing these claims would make the agent act in ways that benefit the source
4. Check if the source has been verified as reliable on unrelated topics but deceptive on goal-relevant ones (selective deception)
5. Compute goal_alignment: positive if claims would help the agent's goals, negative if they'd divert the agent

**Provenance**: `DerivationKind::CredibilityAnalysis { source_id_raw, credibility_score, deception_likelihood }` (tag 67)

**Tests (~12)**:
1. credibility_signals_combined_score
2. factual_support_from_kg
3. self_consistency_check
4. cross_source_consistency
5. manipulation_marker_urgency
6. manipulation_marker_flattery
7. deception_indicator_contradiction
8. adversarial_intent_aligned
9. adversarial_intent_adversarial
10. credibility_recommendation_accept
11. credibility_recommendation_discard
12. encode_credibility_profile_distinct

## Cross-Cutting Changes

### Agent (`src/agent/agent.rs`)
- New fields: `reliability_manager: SourceReliabilityManager`, `credibility_analyzer: CredibilityAnalyzer`
- Optional: `ach_sessions: Vec<AchAnalysis>` (not persistent by default)
- Init/restore/persist lifecycle

### InterlocutorProfile (`src/agent/interlocutor.rs`)
- New field: `trust_model: TrustModel`
- Derived method: `admiralty_rating() -> AdmiraltyRating`
- Auto-update on claim verification

### Module registry (`src/agent/mod.rs`)
- `pub mod source_reliability;`, `pub mod ach;`, `pub mod credibility;`

### Error (`src/agent/error.rs`)
- `Reliability(#[from] super::source_reliability::ReliabilityError)` variant

### Provenance (`src/provenance.rs`)
- Tags 65–67

### OODA (`src/agent/ooda.rs`)
- In `observe()`: run credibility analysis on incoming messages
- In `orient()`: integrate credibility signals into situation assessment
- Low-credibility information gets reduced confidence in working memory

### NLP (`src/agent/nlp.rs`)
- `UserIntent::ReliabilityQuery { source }` — "how reliable is...", "can I trust..."
- `UserIntent::AchCommand { subcommand }` — "analyze competing hypotheses"

### CLI (`src/main.rs`)
- `Commands::Trust { action: TrustAction }` with subcommands: Rate { source }, Verify { source, claim, truth }, Ranking
- `Commands::Ach { action: AchAction }` with subcommands: New { question }, Hypotheses, Evidence, Evaluate, Matrix

## Verification

1. `cargo build` — compiles cleanly
2. `cargo test` — all existing + ~36 new tests pass
3. `cargo clippy` — no new warnings
4. Manual: `akh trust rate "bob"` shows Admiralty rating, `akh ach new "Why did the server crash?"` creates ACH analysis
