//! Convergence sub-verifier: focus-vector stability over a sliding window.

use crate::engine::Engine;

use super::super::error::ThinkingResult;
use super::super::state::ThoughtState;
use super::{ThoughtVerifier, VerifierReason, VerifierVerdict};

/// Tracks the `focus_vec` Hamming delta over a sliding window (the loop
/// records `1 − similarity` into `state.focus_deltas` whenever the focus
/// changes) and signals convergence when every delta in the window is below
/// `delta_threshold`.
#[derive(Debug, Clone)]
pub struct ConvergenceVerifier {
    /// How many recent deltas must be stable.
    pub window: u32,
    /// Normalised Hamming delta below which a step counts as stable.
    pub delta_threshold: f32,
}

impl Default for ConvergenceVerifier {
    fn default() -> Self {
        Self {
            window: 4,
            delta_threshold: 0.02,
        }
    }
}

impl ConvergenceVerifier {
    /// Is the focus stable over the window? Returns the max delta if so.
    pub fn stable(&self, state: &ThoughtState) -> Option<f32> {
        if state.focus_deltas.len() < self.window as usize {
            return None;
        }
        let recent = state.focus_deltas.iter().rev().take(self.window as usize);
        let mut max_delta = 0.0f32;
        for &d in recent {
            if d >= self.delta_threshold {
                return None;
            }
            max_delta = max_delta.max(d);
        }
        Some(max_delta)
    }
}

impl ThoughtVerifier for ConvergenceVerifier {
    fn score(&mut self, _engine: &Engine, state: &ThoughtState) -> ThinkingResult<VerifierVerdict> {
        if let Some(delta) = self.stable(state) {
            return Ok(VerifierVerdict {
                score: 1.0,
                sufficient: false,
                reasons: vec![VerifierReason::FocusStable {
                    delta,
                    window: self.window,
                }],
            });
        }

        // Not (yet) stable: score by how quiet the recent deltas are.
        let n = state.focus_deltas.len().min(self.window as usize);
        if n == 0 {
            return Ok(VerifierVerdict::neutral());
        }
        let mean: f32 = state.focus_deltas.iter().rev().take(n).sum::<f32>() / n as f32;
        Ok(VerifierVerdict {
            score: (1.0 - mean).clamp(0.0, 1.0) * 0.8, // never full marks while unstable
            sufficient: false,
            reasons: Vec::new(),
        })
    }

    fn name(&self) -> &'static str {
        "convergence"
    }
}
