//! Evidence sub-verifier: wraps Phase 17 Dempster-Shafer belief intervals.

use crate::agent::evidence::BeliefInterval;
use crate::engine::Engine;

use super::super::error::ThinkingResult;
use super::super::state::ThoughtState;
use super::{ThoughtVerifier, VerifierReason, VerifierVerdict};

/// Scores the top hypothesis' belief interval and reports when it narrows.
///
/// Score is `Bel × (1 − width)`: high belief with little residual ignorance
/// scores near 1.0; a vacuous interval scores 0.
#[derive(Debug, Default)]
pub struct EvidenceVerifier {
    last_interval: Option<BeliefInterval>,
}

impl ThoughtVerifier for EvidenceVerifier {
    fn score(&mut self, _engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict> {
        let Some(top) = state.top_hypothesis() else {
            self.last_interval = None;
            return Ok(VerifierVerdict {
                score: 0.0,
                sufficient: false,
                reasons: Vec::new(),
            });
        };

        let interval = top.interval();
        let score = (interval.belief * (1.0 - interval.width())).clamp(0.0, 1.0);

        let mut reasons = Vec::new();
        if let Some(prev) = self.last_interval
            && interval.width() + 1e-6 < prev.width()
        {
            reasons.push(VerifierReason::BeliefIntervalNarrowed {
                from: prev,
                to: interval,
            });
        }
        self.last_interval = Some(interval);

        Ok(VerifierVerdict {
            score,
            sufficient: false,
            reasons,
        })
    }

    fn name(&self) -> &'static str {
        "evidence"
    }
}
