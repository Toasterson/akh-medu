//! Training verification: detects alignment degradation after training.
//!
//! After each training session, the verifier re-probes a set of known concepts
//! through the neural bridge and checks that similarity to grounded vectors
//! hasn't degraded beyond a threshold. If degradation is detected, the training
//! weights should be rolled back.

use std::collections::HashMap;

use crate::engine::Engine;
use crate::symbol::SymbolId;
use crate::vsa::neural_bridge::NeuralVsaBridge;
use crate::vsa::ops::VsaOps;

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Result of a training verification check.
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// All probed concepts maintained acceptable alignment.
    Passed {
        concepts_checked: usize,
        avg_similarity: f32,
    },
    /// One or more concepts degraded beyond threshold.
    Degraded {
        worst_concept: SymbolId,
        worst_delta: f32,
        degraded_count: usize,
        total_checked: usize,
    },
}

impl VerificationResult {
    /// Whether the verification passed.
    pub fn passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }
}

// ---------------------------------------------------------------------------
// TrainingVerifier
// ---------------------------------------------------------------------------

/// Verifies that training hasn't degraded neural bridge alignment.
///
/// Captures baseline similarity scores before training, then compares
/// after training to detect degradation.
pub struct TrainingVerifier {
    /// Baseline similarities captured before training.
    baselines: HashMap<SymbolId, f32>,
    /// Maximum allowed degradation per concept (default: 0.1).
    degradation_threshold: f32,
}

impl TrainingVerifier {
    /// Create a new verifier with the given threshold.
    pub fn new(degradation_threshold: f32) -> Self {
        Self {
            baselines: HashMap::new(),
            degradation_threshold,
        }
    }

    /// Capture baseline similarity scores for all grounded symbols.
    ///
    /// Call this **before** training starts.
    pub fn capture_baselines(
        &mut self,
        bridge: &NeuralVsaBridge,
        engine: &Engine,
    ) {
        let item_memory = engine.item_memory();
        let ops = engine.ops();
        self.baselines.clear();

        for &sym_id in &item_memory.all_symbols() {
            if let Some(grounded) = item_memory.get(sym_id) {
                // Encode via proxy (same as trainer).
                let proxy = super::trainer::proxy_hidden_state_public(sym_id.0 as u64, bridge.hidden_dim());
                let encoded = bridge.encode(&proxy);
                if let Ok(sim) = ops.similarity(&encoded, grounded) {
                    self.baselines.insert(sym_id, sim);
                }
            }
        }
    }

    /// Verify alignment after training.
    ///
    /// Call this **after** training completes but before accepting the weights.
    pub fn verify(
        &self,
        bridge: &NeuralVsaBridge,
        engine: &Engine,
    ) -> VerificationResult {
        if self.baselines.is_empty() {
            return VerificationResult::Passed {
                concepts_checked: 0,
                avg_similarity: 0.0,
            };
        }

        let item_memory = engine.item_memory();
        let ops = engine.ops();
        let mut degraded_count = 0;
        let mut worst_concept = SymbolId(0);
        let mut worst_delta = 0.0f32;
        let mut total_sim = 0.0f32;
        let mut checked = 0;

        for (&sym_id, &baseline_sim) in &self.baselines {
            if let Some(grounded) = item_memory.get(sym_id) {
                let proxy = super::trainer::proxy_hidden_state_public(sym_id.0 as u64, bridge.hidden_dim());
                let encoded = bridge.encode(&proxy);
                if let Ok(current_sim) = ops.similarity(&encoded, grounded) {
                    let delta = baseline_sim - current_sim;
                    total_sim += current_sim;
                    checked += 1;

                    if delta > self.degradation_threshold {
                        degraded_count += 1;
                        if delta > worst_delta {
                            worst_delta = delta;
                            worst_concept = sym_id;
                        }
                    }
                }
            }
        }

        if degraded_count > 0 {
            VerificationResult::Degraded {
                worst_concept,
                worst_delta,
                degraded_count,
                total_checked: checked,
            }
        } else {
            VerificationResult::Passed {
                concepts_checked: checked,
                avg_similarity: if checked > 0 {
                    total_sim / checked as f32
                } else {
                    0.0
                },
            }
        }
    }

    /// Number of baseline concepts captured.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_result_passed() {
        let result = VerificationResult::Passed {
            concepts_checked: 10,
            avg_similarity: 0.75,
        };
        assert!(result.passed());
    }

    #[test]
    fn verification_result_degraded() {
        let result = VerificationResult::Degraded {
            worst_concept: SymbolId(42),
            worst_delta: 0.15,
            degraded_count: 3,
            total_checked: 10,
        };
        assert!(!result.passed());
    }

    #[test]
    fn empty_baselines_pass() {
        let verifier = TrainingVerifier::new(0.1);
        assert_eq!(verifier.baseline_count(), 0);
    }
}
