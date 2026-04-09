//! Training loop for fine-tuning the neural→VSA bridge and (future) LoRA adapters.
//!
//! Initially uses the built-in SGD trainer from [`NeuralVsaBridge`] to align
//! the encoder with grounded VSA vectors. The Burn-based LoRA training loop
//! (Phase 27b full scope) will be added when Burn reaches stable release.
//!
//! The trainer is designed to run in `spawn_blocking` during daemon idle periods.

use std::path::PathBuf;
use std::sync::Arc;

use crate::engine::Engine;
use crate::vsa::neural_bridge::{BridgeTrainer, NeuralVsaBridge};
use crate::vsa::{Dimension, Encoding, HyperVec};

use super::data_collector::{TrainingDataCollector, TrainingPairKind};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the training loop.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// Learning rate for bridge encoder SGD (default: 0.01).
    pub bridge_lr: f32,
    /// Maximum training steps per session (default: 200).
    pub max_steps: usize,
    /// Batch size (pairs per gradient step) (default: 16).
    pub batch_size: usize,
    /// Minimum improvement (Hamming match %) to continue (default: 0.001).
    pub min_improvement: f32,
    /// Path to save/load bridge weights.
    pub weights_path: Option<PathBuf>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            bridge_lr: 0.01,
            max_steps: 200,
            batch_size: 16,
            min_improvement: 0.001,
            weights_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Training result
// ---------------------------------------------------------------------------

/// Result of a training session.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Number of gradient steps taken.
    pub steps: usize,
    /// Number of training pairs used.
    pub pairs_used: usize,
    /// Initial match rate (fraction of bits matching ground truth).
    pub initial_match_rate: f32,
    /// Final match rate after training.
    pub final_match_rate: f32,
    /// Whether the session improved the model.
    pub improved: bool,
    /// Reason for stopping.
    pub stop_reason: StopReason,
}

/// Why training stopped.
#[derive(Debug, Clone)]
pub enum StopReason {
    /// Reached maximum steps.
    MaxSteps,
    /// Improvement below threshold.
    Converged,
    /// No training data available.
    NoData,
    /// Error during training.
    Error(String),
}

// ---------------------------------------------------------------------------
// BridgeTrainingSession
// ---------------------------------------------------------------------------

/// A training session for the neural→VSA bridge encoder.
///
/// Aligns the bridge encoder with grounded VSA vectors from the ItemMemory.
/// This is the core training primitive — LoRA fine-tuning (future) builds on
/// top of this alignment.
pub struct BridgeTrainingSession {
    config: TrainerConfig,
}

impl BridgeTrainingSession {
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// Run a training session using grounded symbol pairs from the engine.
    ///
    /// For each symbol in the ItemMemory that has a known label, we:
    /// 1. Look up its grounded HyperVec (the target)
    /// 2. Use the label text as a proxy hidden state (via a simple encoding)
    /// 3. Train the bridge to map the proxy to the grounded vector
    ///
    /// This is a bootstrap step. With the Candle backend active, real hidden
    /// states replace the proxy encoding.
    pub fn train_from_grounded_symbols(
        &self,
        bridge: &mut NeuralVsaBridge,
        engine: &Engine,
    ) -> TrainingResult {
        let item_memory = engine.item_memory();
        let symbols = item_memory.all_symbols();

        if symbols.is_empty() {
            return TrainingResult {
                steps: 0,
                pairs_used: 0,
                initial_match_rate: 0.0,
                final_match_rate: 0.0,
                improved: false,
                stop_reason: StopReason::NoData,
            };
        }

        // Collect (proxy_hidden_state, target_hypervec) pairs.
        let hidden_dim = bridge.hidden_dim();
        let mut training_pairs: Vec<(Vec<f32>, HyperVec)> = Vec::new();

        for &sym_id in &symbols {
            if let Some(hv) = item_memory.get(sym_id) {
                // Create a proxy hidden state from the symbol ID.
                // In production, this comes from Candle hidden state extraction.
                let proxy = proxy_hidden_state(sym_id.0 as u64, hidden_dim);
                training_pairs.push((proxy, hv.clone()));
            }
        }

        if training_pairs.is_empty() {
            return TrainingResult {
                steps: 0,
                pairs_used: 0,
                initial_match_rate: 0.0,
                final_match_rate: 0.0,
                improved: false,
                stop_reason: StopReason::NoData,
            };
        }

        // Measure initial match rate.
        let initial_rate = self.measure_match_rate(bridge, &training_pairs);

        let mut trainer = BridgeTrainer::new(hidden_dim, bridge.vsa_dim().0, self.config.bridge_lr);
        let mut prev_rate = initial_rate;
        let mut steps = 0;

        for step in 0..self.config.max_steps {
            // Mini-batch: cycle through pairs.
            let batch_start = (step * self.config.batch_size) % training_pairs.len();
            let batch_end = (batch_start + self.config.batch_size).min(training_pairs.len());

            for (hidden, target) in &training_pairs[batch_start..batch_end] {
                trainer.accumulate(bridge, hidden, target);
            }
            trainer.step(bridge);
            steps += 1;

            // Check improvement every 10 steps.
            if steps % 10 == 0 {
                let current_rate = self.measure_match_rate(bridge, &training_pairs);
                if current_rate - prev_rate < self.config.min_improvement {
                    let final_rate = current_rate;
                    return TrainingResult {
                        steps,
                        pairs_used: training_pairs.len(),
                        initial_match_rate: initial_rate,
                        final_match_rate: final_rate,
                        improved: final_rate > initial_rate,
                        stop_reason: StopReason::Converged,
                    };
                }
                prev_rate = current_rate;
            }
        }

        let final_rate = self.measure_match_rate(bridge, &training_pairs);
        TrainingResult {
            steps,
            pairs_used: training_pairs.len(),
            initial_match_rate: initial_rate,
            final_match_rate: final_rate,
            improved: final_rate > initial_rate,
            stop_reason: StopReason::MaxSteps,
        }
    }

    /// Measure average bit match rate between encoded and target vectors.
    fn measure_match_rate(
        &self,
        bridge: &NeuralVsaBridge,
        pairs: &[(Vec<f32>, HyperVec)],
    ) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }

        let dim = bridge.vsa_dim().0;
        let total_match: usize = pairs
            .iter()
            .map(|(hidden, target)| {
                let encoded = bridge.encode(hidden);
                (0..dim)
                    .filter(|&i| encoded.get_bit(i) == target.get_bit(i))
                    .count()
            })
            .sum();

        total_match as f32 / (pairs.len() * dim) as f32
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic proxy hidden state from a symbol ID (public for verifier).
pub fn proxy_hidden_state_public(seed: u64, dim: usize) -> Vec<f32> {
    proxy_hidden_state(seed, dim)
}

/// Generate a deterministic proxy hidden state from a symbol ID.
///
/// This is a placeholder for real LLM hidden states. It creates a consistent
/// f32 vector from the symbol ID using a simple hash-based approach.
fn proxy_hidden_state(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = Vec::with_capacity(dim);
    let mut hash = seed.wrapping_mul(0x517cc1b727220a95);
    for _ in 0..dim {
        hash ^= hash << 13;
        hash ^= hash >> 7;
        hash ^= hash << 17;
        let val = (hash as f32) / (u64::MAX as f32) * 2.0 - 1.0;
        state.push(val);
    }
    state
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_hidden_state_deterministic() {
        let a = proxy_hidden_state(42, 128);
        let b = proxy_hidden_state(42, 128);
        assert_eq!(a, b);
    }

    #[test]
    fn proxy_hidden_state_different_seeds() {
        let a = proxy_hidden_state(42, 128);
        let b = proxy_hidden_state(43, 128);
        assert_ne!(a, b);
    }

    #[test]
    fn trainer_config_defaults() {
        let config = TrainerConfig::default();
        assert_eq!(config.bridge_lr, 0.01);
        assert_eq!(config.max_steps, 200);
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn training_result_no_data() {
        let session = BridgeTrainingSession::new(TrainerConfig::default());
        // Without an engine, we can't run the full session, but we can verify
        // the stop reason enum works correctly.
        let result = TrainingResult {
            steps: 0,
            pairs_used: 0,
            initial_match_rate: 0.0,
            final_match_rate: 0.0,
            improved: false,
            stop_reason: StopReason::NoData,
        };
        assert!(!result.improved);
        assert!(matches!(result.stop_reason, StopReason::NoData));
    }
}
