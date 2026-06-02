//! Dempster-Shafer evidence theory — Phase 17.
//!
//! Replaces single-valued confidence scores with belief intervals that
//! explicitly model ignorance. Evidence from multiple sources is combined
//! using Dempster's rule, and conflict between sources is detected and
//! flagged.
//!
//! Core concepts:
//! - **MassFunction**: m(True) + m(False) + m(Ignorance) = 1.0
//! - **BeliefInterval**: [Bel(T), Pl(T)] = [m_true, m_true + m_ignorance]
//! - **Dempster's Rule**: principled combination of independent sources
//! - **Discounting**: reliability-weighted evidence attenuation

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum EvidenceError {
    #[error("invalid mass function: components sum to {sum:.3}, expected 1.0")]
    #[diagnostic(
        code(akh::agent::evidence::invalid_mass),
        help("Ensure m_true + m_false + m_ignorance = 1.0")
    )]
    InvalidMass { sum: f32 },

    #[error("complete conflict: Dempster's rule denominator is zero")]
    #[diagnostic(
        code(akh::agent::evidence::total_conflict),
        help("Sources completely contradict each other. Cannot combine.")
    )]
    TotalConflict,

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::evidence::engine),
        help("An engine-level error occurred during evidence reasoning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for EvidenceError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

pub type EvidenceResult<T> = std::result::Result<T, EvidenceError>;

// ═══════════════════════════════════════════════════════════════════════
// MassFunction
// ═══════════════════════════════════════════════════════════════════════

/// A Dempster-Shafer mass function over frame {True, False}.
///
/// Invariant: `m_true + m_false + m_ignorance = 1.0`
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct MassFunction {
    /// Mass assigned to {True} — direct evidence for.
    pub m_true: f32,
    /// Mass assigned to {False} — direct evidence against.
    pub m_false: f32,
    /// Mass assigned to {True, False} — uncommitted (ignorance).
    pub m_ignorance: f32,
}

impl MassFunction {
    /// Create a mass function. Remainder goes to ignorance.
    pub fn new(m_true: f32, m_false: f32) -> Self {
        let m_ignorance = (1.0 - m_true - m_false).max(0.0);
        Self {
            m_true: m_true.clamp(0.0, 1.0),
            m_false: m_false.clamp(0.0, 1.0),
            m_ignorance,
        }
    }

    /// Convert an existing scalar confidence to a mass function.
    ///
    /// Interpretation: confidence is evidence FOR, no evidence against,
    /// remainder is ignorance.
    pub fn from_confidence(conf: f32) -> Self {
        let c = conf.clamp(0.0, 1.0);
        Self {
            m_true: c,
            m_false: 0.0,
            m_ignorance: 1.0 - c,
        }
    }

    /// Convert to pignistic probability (best single-value estimate).
    ///
    /// BetP(True) = m_true + m_ignorance / 2
    pub fn to_confidence(&self) -> f32 {
        self.m_true + self.m_ignorance / 2.0
    }

    /// Derive the belief interval.
    pub fn belief_interval(&self) -> BeliefInterval {
        BeliefInterval {
            belief: self.m_true,
            plausibility: self.m_true + self.m_ignorance,
        }
    }

    /// Is this a vacuous (total ignorance) mass function?
    pub fn is_vacuous(&self) -> bool {
        self.m_ignorance > 0.99
    }

    /// Conflict factor K between this mass and another.
    ///
    /// K = m1(T)*m2(F) + m1(F)*m2(T)
    pub fn conflict_with(&self, other: &MassFunction) -> f32 {
        self.m_true * other.m_false + self.m_false * other.m_true
    }

    /// Combine with another mass function using Dempster's rule.
    pub fn combine(&self, other: &MassFunction) -> EvidenceResult<MassFunction> {
        let k = self.conflict_with(other);
        if k >= 1.0 {
            return Err(EvidenceError::TotalConflict);
        }
        let norm = 1.0 / (1.0 - k);

        let m_true = (self.m_true * other.m_true
            + self.m_true * other.m_ignorance
            + self.m_ignorance * other.m_true)
            * norm;

        let m_false = (self.m_false * other.m_false
            + self.m_false * other.m_ignorance
            + self.m_ignorance * other.m_false)
            * norm;

        let m_ignorance = (self.m_ignorance * other.m_ignorance) * norm;

        Ok(MassFunction {
            m_true,
            m_false,
            m_ignorance,
        })
    }

    /// Combine multiple mass functions iteratively.
    pub fn combine_multiple(masses: &[MassFunction]) -> EvidenceResult<MassFunction> {
        if masses.is_empty() {
            return Ok(MassFunction::vacuous());
        }
        let mut result = masses[0];
        for m in &masses[1..] {
            result = result.combine(m)?;
        }
        Ok(result)
    }

    /// Discount by source reliability.
    ///
    /// Unreliable sources have their evidence attenuated toward ignorance.
    pub fn discount(&self, reliability: f32) -> MassFunction {
        let r = reliability.clamp(0.0, 1.0);
        MassFunction {
            m_true: r * self.m_true,
            m_false: r * self.m_false,
            m_ignorance: 1.0 - r + r * self.m_ignorance,
        }
    }

    /// Total ignorance — no evidence at all.
    pub fn vacuous() -> Self {
        Self {
            m_true: 0.0,
            m_false: 0.0,
            m_ignorance: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BeliefInterval
// ═══════════════════════════════════════════════════════════════════════

/// A belief interval [Bel, Pl] derived from a mass function.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BeliefInterval {
    /// Lower bound: Bel(True) = m_true.
    pub belief: f32,
    /// Upper bound: Pl(True) = m_true + m_ignorance.
    pub plausibility: f32,
}

impl BeliefInterval {
    /// Width of the interval = ignorance.
    pub fn width(&self) -> f32 {
        (self.plausibility - self.belief).max(0.0)
    }

    /// Is this interval narrow enough to be considered "certain"?
    pub fn is_narrow(&self, threshold: f32) -> bool {
        self.width() < threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EvidenceItem
// ═══════════════════════════════════════════════════════════════════════

/// Evidence from a single source about a proposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// The source that provided this evidence.
    pub source_id: SymbolId,
    /// Mass function representing the source's evidence.
    pub mass: MassFunction,
    /// When the evidence was received.
    pub timestamp: u64,
    /// Source reliability rating (0.0–1.0).
    pub source_reliability: f32,
    /// Optional provenance link.
    pub provenance_id: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════
// ClaimAssessment & Verdict
// ═══════════════════════════════════════════════════════════════════════

/// Assessment of a claim's trustworthiness from combined evidence.
#[derive(Debug, Clone)]
pub struct ClaimAssessment {
    /// The proposition being assessed.
    pub claim_entity: SymbolId,
    /// Combined belief interval.
    pub interval: BeliefInterval,
    /// Verdict based on the interval.
    pub verdict: ClaimVerdict,
    /// Evidence items considered.
    pub evidence_count: usize,
    /// Conflict between sources.
    pub conflict_degree: f32,
    /// Human-readable reasoning.
    pub reasoning: String,
}

/// Verdict on a claim's epistemic status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimVerdict {
    /// High belief, low ignorance: well-supported.
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

impl ClaimVerdict {
    /// Determine verdict from a belief interval and conflict degree.
    pub fn from_interval(interval: &BeliefInterval, conflict: f32) -> Self {
        if conflict > 0.5 {
            return Self::Conflicting;
        }
        if interval.belief > 0.7 && interval.width() < 0.2 {
            Self::WellSupported
        } else if interval.belief > 0.4 {
            Self::Plausible
        } else if interval.width() > 0.5 {
            Self::InsufficientEvidence
        } else {
            Self::LikelyFalse
        }
    }

    pub fn as_label(&self) -> &'static str {
        match self {
            Self::WellSupported => "well-supported",
            Self::Plausible => "plausible",
            Self::InsufficientEvidence => "insufficient evidence",
            Self::LikelyFalse => "likely false",
            Self::Conflicting => "conflicting",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConflictAlert (Phase 17b)
// ═══════════════════════════════════════════════════════════════════════

/// An evidence conflict alert for the operator.
#[derive(Debug, Clone)]
pub struct ConflictAlert {
    pub claim: SymbolId,
    pub conflict_degree: f32,
    pub source_a: SymbolId,
    pub source_b: SymbolId,
    pub recommendation: ConflictRecommendation,
}

/// Recommendation for handling a conflict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictRecommendation {
    /// Seek additional evidence.
    InvestigateFurther,
    /// Cannot resolve automatically — flag for operator.
    FlagForOperator,
}

// ═══════════════════════════════════════════════════════════════════════
// EvidenceFusionConfig (Phase 17b)
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for the evidence fusion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceFusionConfig {
    /// K-factor threshold for conflict alerting.
    pub conflict_alert_threshold: f32,
    /// K-factor threshold for suppressing a claim from responses.
    pub conflict_suppress_threshold: f32,
    /// How much ignorance carries through inference (0.0–1.0).
    pub ignorance_propagation: f32,
    /// Maximum evidence items per claim before compaction.
    pub max_evidence_per_claim: usize,
}

impl Default for EvidenceFusionConfig {
    fn default() -> Self {
        Self {
            conflict_alert_threshold: 0.5,
            conflict_suppress_threshold: 0.8,
            ignorance_propagation: 0.7,
            max_evidence_per_claim: 20,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EvidenceManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages evidence collection, combination, and conflict detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvidenceManager {
    /// Evidence per claim: claim_id → list of evidence items.
    pub claims: HashMap<u64, Vec<EvidenceItem>>,
    /// Fusion pipeline configuration.
    pub config: EvidenceFusionConfig,
    /// Conflict alerts generated since last check.
    #[serde(skip)]
    pub alerts: Vec<ConflictAlert>,
}

impl EvidenceManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: EvidenceFusionConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    /// Add evidence for a claim from a source.
    ///
    /// Automatically discounts by source reliability, checks for conflicts,
    /// and compacts if the evidence list exceeds the configured maximum.
    pub fn add_evidence(&mut self, claim_id: SymbolId, item: EvidenceItem) {
        let evidence_list = self.claims.entry(claim_id.get()).or_default();

        // Check for conflict with existing evidence.
        for existing in evidence_list.iter() {
            let k = existing.mass.conflict_with(&item.mass);
            if k > self.config.conflict_alert_threshold {
                let recommendation = if k > self.config.conflict_suppress_threshold {
                    ConflictRecommendation::FlagForOperator
                } else {
                    ConflictRecommendation::InvestigateFurther
                };
                self.alerts.push(ConflictAlert {
                    claim: claim_id,
                    conflict_degree: k,
                    source_a: existing.source_id,
                    source_b: item.source_id,
                    recommendation,
                });
            }
        }

        evidence_list.push(item);

        // Compact if too many items.
        if evidence_list.len() > self.config.max_evidence_per_claim {
            // Remove oldest items, keeping the most recent.
            evidence_list.sort_by_key(|e| e.timestamp);
            let excess = evidence_list.len() - self.config.max_evidence_per_claim;
            evidence_list.drain(..excess);
        }
    }

    /// Assess a claim by combining all evidence.
    pub fn assess_claim(&self, claim_id: SymbolId) -> ClaimAssessment {
        let evidence = self.claims.get(&claim_id.get());
        let evidence_items = match evidence {
            Some(items) if !items.is_empty() => items,
            _ => {
                return ClaimAssessment {
                    claim_entity: claim_id,
                    interval: MassFunction::vacuous().belief_interval(),
                    verdict: ClaimVerdict::InsufficientEvidence,
                    evidence_count: 0,
                    conflict_degree: 0.0,
                    reasoning: "No evidence available".into(),
                };
            }
        };

        // Discount each source by reliability, then combine.
        let discounted: Vec<MassFunction> = evidence_items
            .iter()
            .map(|e| e.mass.discount(e.source_reliability))
            .collect();

        // Track maximum pairwise conflict.
        let mut max_conflict = 0.0_f32;
        for i in 0..discounted.len() {
            for j in (i + 1)..discounted.len() {
                let k = discounted[i].conflict_with(&discounted[j]);
                max_conflict = max_conflict.max(k);
            }
        }

        let combined = MassFunction::combine_multiple(&discounted)
            .unwrap_or_else(|_| MassFunction::vacuous());

        let interval = combined.belief_interval();
        let verdict = ClaimVerdict::from_interval(&interval, max_conflict);

        let reasoning = format!(
            "{} source(s), Bel={:.2}, Pl={:.2}, conflict={:.2} → {}",
            evidence_items.len(),
            interval.belief,
            interval.plausibility,
            max_conflict,
            verdict.as_label()
        );

        ClaimAssessment {
            claim_entity: claim_id,
            interval,
            verdict,
            evidence_count: evidence_items.len(),
            conflict_degree: max_conflict,
            reasoning,
        }
    }

    /// Get the most uncertain claims (widest belief interval).
    pub fn most_uncertain_claims(&self, k: usize) -> Vec<ClaimAssessment> {
        let mut assessments: Vec<ClaimAssessment> = self
            .claims
            .keys()
            .filter_map(|&id| SymbolId::new(id).map(|s| self.assess_claim(s)))
            .collect();
        assessments.sort_by(|a, b| {
            b.interval.width().partial_cmp(&a.interval.width())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        assessments.truncate(k);
        assessments
    }

    /// Drain pending conflict alerts.
    pub fn drain_alerts(&mut self) -> Vec<ConflictAlert> {
        std::mem::take(&mut self.alerts)
    }

    /// Number of claims tracked.
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> EvidenceResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            EvidenceError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("evidence manager serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:evidence_manager", &bytes)
            .map_err(|e| EvidenceError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:evidence_manager")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Provenance
// ═══════════════════════════════════════════════════════════════════════

/// Record provenance for an evidence combination.
pub fn record_evidence_provenance(
    engine: &Engine,
    derived_id: SymbolId,
    source_count: usize,
    conflict_degree: f32,
    belief: f32,
    plausibility: f32,
) -> EvidenceResult<()> {
    let mut record = crate::provenance::ProvenanceRecord::new(
        derived_id,
        DerivationKind::EvidenceCombination {
            source_count: source_count as u32,
            conflict_degree,
            belief,
            plausibility,
        },
    )
    .with_confidence(belief);
    engine
        .store_provenance(&mut record)
        .map_err(|e| EvidenceError::Engine(Box::new(e)))?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── MassFunction ──────────────────────────────────────────────

    #[test]
    fn mass_function_invariant() {
        let m = MassFunction::new(0.6, 0.1);
        let sum = m.m_true + m.m_false + m.m_ignorance;
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn mass_function_from_confidence() {
        let m = MassFunction::from_confidence(0.8);
        assert!((m.m_true - 0.8).abs() < f32::EPSILON);
        assert!((m.m_false - 0.0).abs() < f32::EPSILON);
        assert!((m.m_ignorance - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn mass_function_to_confidence_pignistic() {
        let m = MassFunction::new(0.6, 0.1);
        // BetP = 0.6 + 0.3/2 = 0.75
        assert!((m.to_confidence() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn mass_function_vacuous() {
        let m = MassFunction::vacuous();
        assert!(m.is_vacuous());
        assert!((m.to_confidence() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mass_function_serialization_roundtrip() {
        let m = MassFunction::new(0.5, 0.2);
        let bytes = bincode::serialize(&m).unwrap();
        let restored: MassFunction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(m, restored);
    }

    // ── BeliefInterval ────────────────────────────────────────────

    #[test]
    fn belief_interval_from_mass() {
        let m = MassFunction::new(0.6, 0.1);
        let bi = m.belief_interval();
        assert!((bi.belief - 0.6).abs() < f32::EPSILON);
        assert!((bi.plausibility - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn belief_interval_width_is_ignorance() {
        let m = MassFunction::new(0.4, 0.2);
        let bi = m.belief_interval();
        assert!((bi.width() - m.m_ignorance).abs() < 1e-6);
    }

    // ── Dempster's Rule ───────────────────────────────────────────

    #[test]
    fn combine_two_agreeing_sources() {
        let m1 = MassFunction::new(0.7, 0.0);
        let m2 = MassFunction::new(0.6, 0.0);
        let combined = m1.combine(&m2).unwrap();
        // Both support True → combined belief should be higher.
        assert!(combined.m_true > m1.m_true);
        assert!(combined.m_true > m2.m_true);
        assert!(combined.m_ignorance < m1.m_ignorance);
    }

    #[test]
    fn combine_two_conflicting_sources() {
        let m1 = MassFunction::new(0.8, 0.0);
        let m2 = MassFunction::new(0.0, 0.8);
        let k = m1.conflict_with(&m2);
        assert!(k > 0.5, "should have high conflict: K={k}");
        // Should still combine (K < 1.0).
        let combined = m1.combine(&m2).unwrap();
        let sum = combined.m_true + combined.m_false + combined.m_ignorance;
        assert!((sum - 1.0).abs() < 1e-5, "invariant violated: sum={sum}");
    }

    #[test]
    fn combine_with_vacuous_source() {
        let m1 = MassFunction::new(0.7, 0.1);
        let vac = MassFunction::vacuous();
        let combined = m1.combine(&vac).unwrap();
        // Combining with vacuous should not change the mass.
        assert!((combined.m_true - m1.m_true).abs() < 1e-6);
        assert!((combined.m_false - m1.m_false).abs() < 1e-6);
    }

    #[test]
    fn combine_multiple_sources() {
        let masses = vec![
            MassFunction::new(0.5, 0.0),
            MassFunction::new(0.6, 0.0),
            MassFunction::new(0.4, 0.0),
        ];
        let combined = MassFunction::combine_multiple(&masses).unwrap();
        assert!(combined.m_true > 0.7, "three agreeing sources: {}", combined.m_true);
    }

    // ── Discounting ───────────────────────────────────────────────

    #[test]
    fn discount_by_reliability() {
        let m = MassFunction::new(0.8, 0.0);
        let discounted = m.discount(0.5);
        assert!((discounted.m_true - 0.4).abs() < f32::EPSILON);
        assert!((discounted.m_ignorance - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn discount_unreliable_increases_ignorance() {
        let m = MassFunction::new(0.8, 0.1);
        let discounted = m.discount(0.3);
        assert!(discounted.m_ignorance > m.m_ignorance);
        let sum = discounted.m_true + discounted.m_false + discounted.m_ignorance;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ── ClaimVerdict ──────────────────────────────────────────────

    #[test]
    fn verdict_well_supported() {
        let bi = BeliefInterval { belief: 0.85, plausibility: 0.95 };
        assert_eq!(ClaimVerdict::from_interval(&bi, 0.0), ClaimVerdict::WellSupported);
    }

    #[test]
    fn verdict_insufficient_evidence() {
        let bi = BeliefInterval { belief: 0.1, plausibility: 0.9 };
        assert_eq!(ClaimVerdict::from_interval(&bi, 0.0), ClaimVerdict::InsufficientEvidence);
    }

    #[test]
    fn verdict_conflicting() {
        let bi = BeliefInterval { belief: 0.5, plausibility: 0.8 };
        assert_eq!(ClaimVerdict::from_interval(&bi, 0.7), ClaimVerdict::Conflicting);
    }

    #[test]
    fn verdict_likely_false() {
        let bi = BeliefInterval { belief: 0.1, plausibility: 0.3 };
        assert_eq!(ClaimVerdict::from_interval(&bi, 0.0), ClaimVerdict::LikelyFalse);
    }

    // ── Conflict ──────────────────────────────────────────────────

    #[test]
    fn conflict_degree_zero_for_agreeing() {
        let m1 = MassFunction::new(0.8, 0.0);
        let m2 = MassFunction::new(0.6, 0.0);
        assert!((m1.conflict_with(&m2) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn conflict_degree_high_for_opposing() {
        let m1 = MassFunction::new(0.9, 0.0);
        let m2 = MassFunction::new(0.0, 0.9);
        let k = m1.conflict_with(&m2);
        assert!(k > 0.8, "opposing sources should conflict: K={k}");
    }

    // ── EvidenceManager ───────────────────────────────────────────

    #[test]
    fn evidence_manager_add_and_assess() {
        let mut mgr = EvidenceManager::new();
        let claim = SymbolId::new(1).unwrap();
        let source = SymbolId::new(10).unwrap();

        mgr.add_evidence(claim, EvidenceItem {
            source_id: source,
            mass: MassFunction::new(0.9, 0.0),
            timestamp: 100,
            source_reliability: 1.0,
            provenance_id: None,
        });

        let assessment = mgr.assess_claim(claim);
        assert_eq!(assessment.verdict, ClaimVerdict::WellSupported);
        assert_eq!(assessment.evidence_count, 1);
    }

    #[test]
    fn evidence_manager_conflict_alert() {
        let mut mgr = EvidenceManager::new();
        let claim = SymbolId::new(1).unwrap();
        let s1 = SymbolId::new(10).unwrap();
        let s2 = SymbolId::new(11).unwrap();

        mgr.add_evidence(claim, EvidenceItem {
            source_id: s1,
            mass: MassFunction::new(0.9, 0.0),
            timestamp: 100,
            source_reliability: 1.0,
            provenance_id: None,
        });

        mgr.add_evidence(claim, EvidenceItem {
            source_id: s2,
            mass: MassFunction::new(0.0, 0.9),
            timestamp: 101,
            source_reliability: 1.0,
            provenance_id: None,
        });

        let alerts = mgr.drain_alerts();
        assert!(!alerts.is_empty(), "should have conflict alert");
        assert!(alerts[0].conflict_degree > 0.5);
    }

    #[test]
    fn evidence_manager_no_evidence() {
        let mgr = EvidenceManager::new();
        let claim = SymbolId::new(999).unwrap();
        let assessment = mgr.assess_claim(claim);
        assert_eq!(assessment.verdict, ClaimVerdict::InsufficientEvidence);
    }

    #[test]
    fn evidence_manager_serialization_roundtrip() {
        let mut mgr = EvidenceManager::new();
        let claim = SymbolId::new(1).unwrap();
        mgr.add_evidence(claim, EvidenceItem {
            source_id: SymbolId::new(10).unwrap(),
            mass: MassFunction::new(0.7, 0.1),
            timestamp: 100,
            source_reliability: 0.8,
            provenance_id: None,
        });

        let bytes = bincode::serialize(&mgr).unwrap();
        let restored: EvidenceManager = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.claim_count(), 1);
    }

    #[test]
    fn most_uncertain_claims() {
        let mut mgr = EvidenceManager::new();
        // Claim 1: high certainty.
        mgr.add_evidence(SymbolId::new(1).unwrap(), EvidenceItem {
            source_id: SymbolId::new(10).unwrap(),
            mass: MassFunction::new(0.9, 0.0),
            timestamp: 100,
            source_reliability: 1.0,
            provenance_id: None,
        });
        // Claim 2: high uncertainty.
        mgr.add_evidence(SymbolId::new(2).unwrap(), EvidenceItem {
            source_id: SymbolId::new(11).unwrap(),
            mass: MassFunction::new(0.1, 0.0),
            timestamp: 101,
            source_reliability: 1.0,
            provenance_id: None,
        });

        let uncertain = mgr.most_uncertain_claims(1);
        assert_eq!(uncertain.len(), 1);
        assert_eq!(uncertain[0].claim_entity, SymbolId::new(2).unwrap());
    }

    // ── Confidence bridge ─────────────────────────────────────────

    #[test]
    fn confidence_bridge_roundtrip() {
        let original = 0.75_f32;
        let mass = MassFunction::from_confidence(original);
        let recovered = mass.to_confidence();
        // Pignistic: 0.75 + 0.25/2 = 0.875 ≠ 0.75 (expected — pignistic adds half ignorance)
        assert!(recovered > original);
        // But from_confidence(1.0) should be exact.
        let m1 = MassFunction::from_confidence(1.0);
        assert!((m1.to_confidence() - 1.0).abs() < f32::EPSILON);
    }
}
