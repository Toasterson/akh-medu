//! Source-reliability sub-verifier: wraps Phase 18a Admiralty trust ratings.

use crate::agent::source_reliability::SourceReliabilityManager;
use crate::engine::Engine;

use super::super::error::ThinkingResult;
use super::super::state::ThoughtState;
use super::{ThoughtVerifier, VerifierReason, VerifierVerdict};

/// Penalises hypotheses whose evidence is dominated by low-reliability
/// sources (Admiralty factor below `unreliable_threshold`).
///
/// Neutral (score 1.0) when the top hypothesis has no tracked sources.
pub struct SourceReliabilityVerifier {
    manager: SourceReliabilityManager,
    /// Admiralty factor below which a source counts as unreliable.
    pub unreliable_threshold: f32,
}

impl SourceReliabilityVerifier {
    /// Restore Phase 18 trust state persisted on the engine.
    pub fn restore(engine: &Engine) -> Self {
        Self {
            manager: SourceReliabilityManager::restore(engine),
            unreliable_threshold: 0.4,
        }
    }

    /// Use an explicit manager (tests, or a live agent's manager snapshot).
    pub fn with_manager(manager: SourceReliabilityManager) -> Self {
        Self {
            manager,
            unreliable_threshold: 0.4,
        }
    }
}

impl ThoughtVerifier for SourceReliabilityVerifier {
    fn score(&mut self, _engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict> {
        let Some(top) = state.top_hypothesis() else {
            return Ok(VerifierVerdict::neutral());
        };
        if top.sources.is_empty() {
            return Ok(VerifierVerdict {
                score: 1.0,
                sufficient: false,
                reasons: Vec::new(),
            });
        }

        // The dominant source is the least reliable one: Dempster combination
        // lets a single poisoned source drag the fused mass, so we score by
        // the weakest contributor.
        let mut worst: Option<(crate::symbol::SymbolId, f32)> = None;
        for &src in &top.sources {
            let factor = self.manager.rate_source(src).source.to_factor();
            if worst.is_none_or(|(_, w)| factor < w) {
                worst = Some((src, factor));
            }
        }
        let (source, weight) = worst.expect("sources non-empty");

        let mut reasons = Vec::new();
        if weight < self.unreliable_threshold {
            reasons.push(VerifierReason::UnreliableSourceDominant { source, weight });
        }

        Ok(VerifierVerdict {
            score: weight.clamp(0.0, 1.0),
            sufficient: false,
            reasons,
        })
    }

    fn name(&self) -> &'static str {
        "source-reliability"
    }
}
