//! Source reliability & Bayesian trust model — Phase 18a.
//!
//! NATO Admiralty Code ratings (source reliability A–F, information credibility 1–6)
//! backed by a Bayesian trust model (Beta distributions over competence, benevolence,
//! integrity). Trust updates from verified/falsified claims drive automatic Admiralty
//! rating recalculation.
//!
//! Integration with Phase 17: `evidence_mass_from_rating()` converts Admiralty ratings
//! to Dempster-Shafer mass functions for principled evidence combination.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::symbol::SymbolId;

use super::evidence::MassFunction;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum ReliabilityError {
    #[error("source not found: {id}")]
    #[diagnostic(
        code(akh::agent::reliability::source_not_found),
        help("Add observations for this source via `update_trust()` or `verify_claim()`.")
    )]
    SourceNotFound { id: u64 },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::reliability::engine),
        help("An engine-level error occurred during reliability assessment.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ReliabilityError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

pub type ReliabilityResult<T> = std::result::Result<T, ReliabilityError>;

// ═══════════════════════════════════════════════════════════════════════
// Admiralty Ratings
// ═══════════════════════════════════════════════════════════════════════

/// NATO Admiralty Code: source reliability rating (A–F).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceReliability {
    /// A — No doubt about authenticity, trustworthiness.
    CompletelyReliable,
    /// B — Minor doubts, generally trustworthy.
    UsuallyReliable,
    /// C — Some doubts about authenticity.
    FairlyReliable,
    /// D — Significant doubts.
    NotUsuallyReliable,
    /// E — Shown to be unreliable in past.
    Unreliable,
    /// F — New or unknown source, cannot judge.
    CannotJudge,
}

impl SourceReliability {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::CompletelyReliable => "A",
            Self::UsuallyReliable => "B",
            Self::FairlyReliable => "C",
            Self::NotUsuallyReliable => "D",
            Self::Unreliable => "E",
            Self::CannotJudge => "F",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CompletelyReliable => "completely reliable",
            Self::UsuallyReliable => "usually reliable",
            Self::FairlyReliable => "fairly reliable",
            Self::NotUsuallyReliable => "not usually reliable",
            Self::Unreliable => "unreliable",
            Self::CannotJudge => "cannot judge",
        }
    }

    /// Convert a trust score (0.0–1.0) to an Admiralty source rating.
    pub fn from_trust_score(score: f32) -> Self {
        if score > 0.9 {
            Self::CompletelyReliable
        } else if score > 0.75 {
            Self::UsuallyReliable
        } else if score > 0.6 {
            Self::FairlyReliable
        } else if score > 0.4 {
            Self::NotUsuallyReliable
        } else if score > 0.2 {
            Self::Unreliable
        } else {
            Self::CannotJudge
        }
    }

    /// Convert to a numeric reliability factor for discounting.
    pub fn to_factor(&self) -> f32 {
        match self {
            Self::CompletelyReliable => 0.95,
            Self::UsuallyReliable => 0.80,
            Self::FairlyReliable => 0.65,
            Self::NotUsuallyReliable => 0.45,
            Self::Unreliable => 0.20,
            Self::CannotJudge => 0.50,
        }
    }
}

/// NATO Admiralty Code: information credibility rating (1–6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InformationCredibility {
    /// 1 — Confirmed by other independent sources.
    Confirmed,
    /// 2 — Probably true, consistent with known information.
    ProbablyTrue,
    /// 3 — Possibly true, not confirmed or denied.
    PossiblyTrue,
    /// 4 — Doubtful, inconsistent with known information.
    Doubtful,
    /// 5 — Improbable, contradicted by known information.
    Improbable,
    /// 6 — Truth cannot be judged.
    CannotJudge,
}

impl InformationCredibility {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Confirmed => "1",
            Self::ProbablyTrue => "2",
            Self::PossiblyTrue => "3",
            Self::Doubtful => "4",
            Self::Improbable => "5",
            Self::CannotJudge => "6",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::ProbablyTrue => "probably true",
            Self::PossiblyTrue => "possibly true",
            Self::Doubtful => "doubtful",
            Self::Improbable => "improbable",
            Self::CannotJudge => "cannot judge",
        }
    }

    /// Convert to a confidence factor.
    pub fn to_factor(&self) -> f32 {
        match self {
            Self::Confirmed => 0.95,
            Self::ProbablyTrue => 0.80,
            Self::PossiblyTrue => 0.60,
            Self::Doubtful => 0.35,
            Self::Improbable => 0.15,
            Self::CannotJudge => 0.50,
        }
    }
}

/// Combined Admiralty rating (e.g. "B2" = usually reliable, probably true).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AdmiraltyRating {
    pub source: SourceReliability,
    pub information: InformationCredibility,
}

impl AdmiraltyRating {
    pub fn label(&self) -> String {
        format!("{}{}", self.source.as_label(), self.information.as_label())
    }

    /// Convert to a DS mass function for evidence combination.
    pub fn to_mass_function(&self) -> MassFunction {
        let reliability = self.source.to_factor();
        let credibility = self.information.to_factor();
        // m_true = reliability * credibility
        // m_false = reliability * (1 - credibility) * 0.5 (conservative)
        let m_true = reliability * credibility;
        let m_false = reliability * (1.0 - credibility) * 0.3;
        MassFunction::new(m_true.min(0.95), m_false.min(0.95))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Bayesian Trust Model
// ═══════════════════════════════════════════════════════════════════════

/// A Beta distribution parameterized by (α, β).
///
/// Expected value = α / (α + β).
/// Variance decreases as observation count increases.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BetaDistribution {
    /// Positive evidence count (successes + prior).
    pub alpha: f32,
    /// Negative evidence count (failures + prior).
    pub beta: f32,
}

impl BetaDistribution {
    /// Create with uninformative prior (α=1, β=1).
    pub fn uninformative() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Expected value (mean of the Beta distribution).
    pub fn expected(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Confidence in the estimate (how many observations).
    pub fn confidence(&self) -> f32 {
        (self.alpha + self.beta - 2.0).max(0.0)
            / (1.0 + self.alpha + self.beta - 2.0)
    }

    /// Update with an observation.
    pub fn update(&mut self, positive: bool, weight: f32) {
        let w = weight.clamp(0.0, 5.0);
        if positive {
            self.alpha += w;
        } else {
            self.beta += w;
        }
    }
}

impl Default for BetaDistribution {
    fn default() -> Self {
        Self::uninformative()
    }
}

/// Trust dimension for observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustDimension {
    /// Source provided accurate information.
    Competence,
    /// Source acted in the agent's interest.
    Benevolence,
    /// Source was honest and consistent.
    Integrity,
}

/// Multi-dimensional Bayesian trust model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrustModel {
    pub competence: BetaDistribution,
    pub benevolence: BetaDistribution,
    pub integrity: BetaDistribution,
    pub observation_count: u32,
}

impl Default for TrustModel {
    fn default() -> Self {
        Self {
            competence: BetaDistribution::uninformative(),
            benevolence: BetaDistribution::uninformative(),
            integrity: BetaDistribution::uninformative(),
            observation_count: 0,
        }
    }
}

impl TrustModel {
    /// Weighted trust score across all dimensions.
    ///
    /// Weights: competence=0.4, benevolence=0.3, integrity=0.3.
    pub fn overall_trust(&self) -> f32 {
        self.competence.expected() * 0.4
            + self.benevolence.expected() * 0.3
            + self.integrity.expected() * 0.3
    }

    /// Update a single trust dimension.
    pub fn update(&mut self, dimension: TrustDimension, positive: bool, weight: f32) {
        match dimension {
            TrustDimension::Competence => self.competence.update(positive, weight),
            TrustDimension::Benevolence => self.benevolence.update(positive, weight),
            TrustDimension::Integrity => self.integrity.update(positive, weight),
        }
        self.observation_count += 1;
    }

    /// Derive an Admiralty source reliability rating.
    pub fn admiralty_rating(&self) -> SourceReliability {
        if self.observation_count == 0 {
            return SourceReliability::CannotJudge;
        }
        SourceReliability::from_trust_score(self.overall_trust())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SourceReliabilityManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages source reliability ratings and trust models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceReliabilityManager {
    /// Per-source trust models.
    pub sources: HashMap<u64, TrustModel>,
}

impl SourceReliabilityManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a trust model for a source.
    pub fn get_or_create(&mut self, source_id: SymbolId) -> &mut TrustModel {
        self.sources.entry(source_id.get()).or_default()
    }

    /// Rate a source's reliability.
    pub fn rate_source(&self, source_id: SymbolId) -> AdmiraltyRating {
        let trust = self.sources.get(&source_id.get());
        let source = match trust {
            Some(t) => t.admiralty_rating(),
            None => SourceReliability::CannotJudge,
        };
        AdmiraltyRating {
            source,
            information: InformationCredibility::CannotJudge,
        }
    }

    /// Update trust based on claim verification.
    ///
    /// When a claim from a source is verified as true or false, update
    /// the competence and integrity dimensions.
    pub fn verify_claim(
        &mut self,
        source_id: SymbolId,
        claim_was_true: bool,
    ) {
        let trust = self.get_or_create(source_id);
        // Competence: was the source accurate?
        trust.update(TrustDimension::Competence, claim_was_true, 1.0);
        // Integrity: was the source honest? (true claim = honest)
        trust.update(TrustDimension::Integrity, claim_was_true, 0.5);
    }

    /// Update a specific trust dimension.
    pub fn update_trust(
        &mut self,
        source_id: SymbolId,
        dimension: TrustDimension,
        positive: bool,
        weight: f32,
    ) {
        let trust = self.get_or_create(source_id);
        trust.update(dimension, positive, weight);
    }

    /// Get the most reliable sources (by overall trust).
    pub fn most_reliable(&self, k: usize) -> Vec<(u64, f32, SourceReliability)> {
        let mut sorted: Vec<_> = self
            .sources
            .iter()
            .map(|(&id, t)| (id, t.overall_trust(), t.admiralty_rating()))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }

    /// Get the least reliable sources.
    pub fn least_reliable(&self, k: usize) -> Vec<(u64, f32, SourceReliability)> {
        let mut sorted: Vec<_> = self
            .sources
            .iter()
            .filter(|(_, t)| t.observation_count > 0)
            .map(|(&id, t)| (id, t.overall_trust(), t.admiralty_rating()))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &crate::engine::Engine) -> ReliabilityResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            ReliabilityError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("reliability manager serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:reliability_manager", &bytes)
            .map_err(|e| ReliabilityError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &crate::engine::Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:reliability_manager")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Admiralty labels ──────────────────────────────────────────

    #[test]
    fn admiralty_source_labels_roundtrip() {
        let ratings = [
            SourceReliability::CompletelyReliable,
            SourceReliability::UsuallyReliable,
            SourceReliability::FairlyReliable,
            SourceReliability::NotUsuallyReliable,
            SourceReliability::Unreliable,
            SourceReliability::CannotJudge,
        ];
        let expected = ["A", "B", "C", "D", "E", "F"];
        for (r, l) in ratings.iter().zip(expected.iter()) {
            assert_eq!(r.as_label(), *l);
        }
    }

    #[test]
    fn admiralty_info_labels() {
        let ratings = [
            InformationCredibility::Confirmed,
            InformationCredibility::ProbablyTrue,
            InformationCredibility::PossiblyTrue,
            InformationCredibility::Doubtful,
            InformationCredibility::Improbable,
            InformationCredibility::CannotJudge,
        ];
        let expected = ["1", "2", "3", "4", "5", "6"];
        for (r, l) in ratings.iter().zip(expected.iter()) {
            assert_eq!(r.as_label(), *l);
        }
    }

    #[test]
    fn admiralty_combined_label() {
        let rating = AdmiraltyRating {
            source: SourceReliability::UsuallyReliable,
            information: InformationCredibility::ProbablyTrue,
        };
        assert_eq!(rating.label(), "B2");
    }

    // ── Beta distribution ─────────────────────────────────────────

    #[test]
    fn beta_expected_value() {
        let b = BetaDistribution { alpha: 8.0, beta: 2.0 };
        assert!((b.expected() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn beta_uninformative_prior() {
        let b = BetaDistribution::uninformative();
        assert!((b.expected() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn bayesian_trust_update_positive() {
        let mut b = BetaDistribution::uninformative();
        b.update(true, 1.0);
        assert!(b.expected() > 0.5);
        assert_eq!(b.alpha, 2.0);
    }

    #[test]
    fn bayesian_trust_update_negative() {
        let mut b = BetaDistribution::uninformative();
        b.update(false, 1.0);
        assert!(b.expected() < 0.5);
        assert_eq!(b.beta, 2.0);
    }

    #[test]
    fn bayesian_trust_convergence() {
        let mut b = BetaDistribution::uninformative();
        for _ in 0..100 {
            b.update(true, 1.0);
        }
        // After 100 positive observations, should converge near 1.0.
        assert!(b.expected() > 0.95, "expected high: {}", b.expected());
    }

    // ── TrustModel ────────────────────────────────────────────────

    #[test]
    fn trust_model_default() {
        let t = TrustModel::default();
        assert!((t.overall_trust() - 0.5).abs() < f32::EPSILON);
        assert_eq!(t.admiralty_rating(), SourceReliability::CannotJudge);
    }

    #[test]
    fn auto_rating_from_trust() {
        let mut t = TrustModel::default();
        // Many positive observations → high trust.
        for _ in 0..50 {
            t.update(TrustDimension::Competence, true, 1.0);
            t.update(TrustDimension::Integrity, true, 1.0);
            t.update(TrustDimension::Benevolence, true, 1.0);
        }
        assert_eq!(t.admiralty_rating(), SourceReliability::CompletelyReliable);
    }

    // ── SourceReliabilityManager ──────────────────────────────────

    #[test]
    fn verify_claim_updates_trust() {
        let mut mgr = SourceReliabilityManager::new();
        let src = SymbolId::new(10).unwrap();

        mgr.verify_claim(src, true);
        mgr.verify_claim(src, true);
        mgr.verify_claim(src, true);

        let rating = mgr.rate_source(src);
        assert_ne!(rating.source, SourceReliability::CannotJudge);
        assert_ne!(rating.source, SourceReliability::Unreliable);
    }

    #[test]
    fn admiralty_to_mass_function() {
        let rating = AdmiraltyRating {
            source: SourceReliability::UsuallyReliable,
            information: InformationCredibility::ProbablyTrue,
        };
        let mass = rating.to_mass_function();
        let sum = mass.m_true + mass.m_false + mass.m_ignorance;
        assert!((sum - 1.0).abs() < 1e-5, "mass invariant: sum={sum}");
        assert!(mass.m_true > 0.5, "B2 should have high m_true");
    }

    #[test]
    fn most_reliable_ranking() {
        let mut mgr = SourceReliabilityManager::new();
        let s1 = SymbolId::new(1).unwrap();
        let s2 = SymbolId::new(2).unwrap();

        // s1: very reliable
        for _ in 0..20 {
            mgr.verify_claim(s1, true);
        }
        // s2: unreliable
        for _ in 0..20 {
            mgr.verify_claim(s2, false);
        }

        let most = mgr.most_reliable(2);
        assert_eq!(most[0].0, 1); // s1 should be first
    }

    #[test]
    fn trust_model_serialization() {
        let mut mgr = SourceReliabilityManager::new();
        let src = SymbolId::new(42).unwrap();
        mgr.verify_claim(src, true);

        let bytes = bincode::serialize(&mgr).unwrap();
        let restored: SourceReliabilityManager = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.sources.len(), 1);
    }
}
