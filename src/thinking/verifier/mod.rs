//! Structural verifier (Phase 41c) — the process-reward-model equivalent.
//!
//! Composes **existing** systems into a score over the current
//! [`ThoughtState`]:
//!
//! - [`contradiction::ContradictionVerifier`] — Phase 9l contradiction
//!   detection over the top hypothesis' materialisation.
//! - [`evidence::EvidenceVerifier`] — Phase 17 Dempster-Shafer belief
//!   intervals on the hypotheses.
//! - [`source::SourceReliabilityVerifier`] — Phase 18a Admiralty trust on
//!   contributing sources.
//! - [`convergence::ConvergenceVerifier`] — focus-vector Hamming stability
//!   over a sliding window.
//!
//! It does not introduce a learned model — it cannot lie about the reasoning
//! because the reasoning *is* the trace.

pub mod contradiction;
pub mod convergence;
pub mod evidence;
pub mod source;

use serde::{Deserialize, Serialize};

use crate::agent::evidence::BeliefInterval;
use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::error::ThinkingResult;
use super::state::{ThoughtState, TripleRef};

pub use contradiction::ContradictionVerifier;
pub use convergence::ConvergenceVerifier;
pub use evidence::EvidenceVerifier;
pub use source::SourceReliabilityVerifier;

// ---------------------------------------------------------------------------
// Verdict vocabulary
// ---------------------------------------------------------------------------

/// A single structural reason contributing to the verifier score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerifierReason {
    /// Phase 9l: a contradiction surfaced — score drops, may abort.
    ContradictionDetected { kind: String, locus: TripleRef },
    /// Phase 17: belief interval narrowed — score rises.
    BeliefIntervalNarrowed {
        from: BeliefInterval,
        to: BeliefInterval,
    },
    /// Phase 18: an unreliable source dominates the evidence — score drops.
    UnreliableSourceDominant { source: SymbolId, weight: f32 },
    /// Convergence: focus_vec stable across the window.
    FocusStable { delta: f32, window: u32 },
    /// VOC (Phase 41e): marginal value of the next step below threshold.
    DiminishingReturns { last_voc: f32 },
}

/// Outcome of a verifier evaluation.
#[derive(Debug, Clone)]
pub struct VerifierVerdict {
    /// Aggregate score in `[0, 1]`.
    pub score: f32,
    /// Halt-now signal: the answer is structurally sufficient.
    pub sufficient: bool,
    /// The structural reasons behind the score.
    pub reasons: Vec<VerifierReason>,
}

impl VerifierVerdict {
    /// A neutral verdict (nothing to say).
    pub fn neutral() -> Self {
        Self {
            score: 0.5,
            sufficient: false,
            reasons: Vec::new(),
        }
    }
}

/// A structural scorer over thought states.
pub trait ThoughtVerifier: Send {
    /// Score the state. `&mut self` so verifiers can keep cheap history
    /// (e.g. the previous belief interval).
    fn score(&mut self, engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict>;

    /// Verifier name for traces.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Composite
// ---------------------------------------------------------------------------

/// Weights + thresholds for the composite verifier.
#[derive(Debug, Clone)]
pub struct CompositeConfig {
    pub contradiction_weight: f32,
    pub evidence_weight: f32,
    pub source_weight: f32,
    pub convergence_weight: f32,
    /// Aggregate score at or above this may halt the loop.
    pub sufficiency_threshold: f32,
    /// The top hypothesis must reach at least this belief to be sufficient.
    pub min_belief: f32,
}

impl Default for CompositeConfig {
    fn default() -> Self {
        Self {
            contradiction_weight: 0.30,
            evidence_weight: 0.40,
            source_weight: 0.15,
            convergence_weight: 0.15,
            sufficiency_threshold: 0.75,
            min_belief: 0.60,
        }
    }
}

/// Runs the four sub-verifiers and aggregates their scores.
pub struct CompositeVerifier {
    pub contradiction: ContradictionVerifier,
    pub evidence: EvidenceVerifier,
    pub source: SourceReliabilityVerifier,
    pub convergence: ConvergenceVerifier,
    pub config: CompositeConfig,
}

impl CompositeVerifier {
    /// Default composition, restoring Phase 18 trust state from the engine.
    pub fn new(engine: &Engine, preds: &super::ThinkingPredicates) -> Self {
        Self {
            contradiction: ContradictionVerifier::new(preds.clone()),
            evidence: EvidenceVerifier::default(),
            source: SourceReliabilityVerifier::restore(engine),
            convergence: ConvergenceVerifier::default(),
            config: CompositeConfig::default(),
        }
    }
}

impl ThoughtVerifier for CompositeVerifier {
    fn score(&mut self, engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict> {
        let c = self.contradiction.score(engine, state)?;
        let e = self.evidence.score(engine, state)?;
        let s = self.source.score(engine, state)?;
        let v = self.convergence.score(engine, state)?;

        let cfg = &self.config;
        let total_w = cfg.contradiction_weight
            + cfg.evidence_weight
            + cfg.source_weight
            + cfg.convergence_weight;
        let score = (c.score * cfg.contradiction_weight
            + e.score * cfg.evidence_weight
            + s.score * cfg.source_weight
            + v.score * cfg.convergence_weight)
            / total_w;

        let mut reasons = c.reasons;
        let contradicted = !reasons.is_empty();
        reasons.extend(e.reasons);
        reasons.extend(s.reasons);
        reasons.extend(v.reasons);

        let top_belief = state
            .top_hypothesis()
            .map(|h| h.belief())
            .unwrap_or(0.0);
        let sufficient = !contradicted
            && score >= cfg.sufficiency_threshold
            && top_belief >= cfg.min_belief;

        Ok(VerifierVerdict {
            score: score.clamp(0.0, 1.0),
            sufficient,
            reasons,
        })
    }

    fn name(&self) -> &'static str {
        "composite"
    }
}
