//! Active inference OODA enhancement — Phase 20.
//!
//! Extends the OODA loop with active inference: the agent predicts observations,
//! computes surprise (prediction error), and selects actions that minimize
//! expected free energy (pragmatic value + epistemic value).
//!
//! ## Sub-phases
//!
//! - **20a**: Generative model, prediction error, free energy computation
//! - **20b**: EFE-based policy selection with adaptive exploration

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph::Triple;
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for the active inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceConfig {
    /// Weight for pragmatic value (goal progress). Default: 0.6.
    pub pragmatic_weight: f32,
    /// Weight for epistemic value (uncertainty reduction). Default: 0.4.
    pub epistemic_weight: f32,
    /// Learning rate for precision adaptation. Default: 0.05.
    pub precision_learning_rate: f32,
    /// Surprise threshold for triggering model updates. Default: 0.3.
    pub surprise_threshold: f32,
    /// Maximum policies to evaluate per decision. Default: 20.
    pub max_policies: usize,
    /// Softmax temperature for policy selection. Default: 1.0.
    pub temperature: f32,
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            pragmatic_weight: 0.6,
            epistemic_weight: 0.4,
            precision_learning_rate: 0.05,
            surprise_threshold: 0.3,
            max_policies: 20,
            temperature: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 20a — Generative Model & Prediction Error
// ═══════════════════════════════════════════════════════════════════════

/// The agent's generative model: predicted observations and preferred states.
#[derive(Debug, Clone, Default)]
pub struct GenerativeModel {
    /// Expected triples: (s, p, o, expected_confidence).
    pub expected_observations: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Whether the causal transition model is initialized.
    pub transition_model_ready: bool,
    /// Preferred goal states.
    pub preferred_states: Vec<(SymbolId, SymbolId, SymbolId)>,
    /// Current belief state: triples with confidence.
    pub belief_state: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
}

/// Prediction error: difference between expected and actual observations.
#[derive(Debug, Clone)]
pub struct PredictionError {
    /// Expected but not observed (false positives of the model).
    pub false_positives: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Observed but not expected (false negatives of the model).
    pub false_negatives: Vec<(SymbolId, SymbolId, SymbolId, f32)>,
    /// Total error magnitude.
    pub total_error: f32,
    /// Surprise: -log P(observations | model). Higher = more unexpected.
    pub surprise: f32,
    /// Model precision: inverse variance of errors. High = reliable model.
    pub precision: f32,
}

/// Free energy decomposition for a candidate action.
#[derive(Debug, Clone)]
pub struct FreeEnergy {
    /// Pragmatic value: expected goal progress (negative distance to goal).
    pub pragmatic_value: f32,
    /// Epistemic value: expected information gain (uncertainty reduction).
    pub epistemic_value: f32,
    /// Combined: -(w_p * pragmatic + w_e * epistemic). Lower = better.
    pub expected_free_energy: f32,
    /// Human-readable reasoning.
    pub reasoning: String,
}

/// Precision weights for different sources of prediction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrecisionWeights {
    /// Per-source precision (how reliable each source's predictions are).
    pub source_precision: HashMap<u64, f32>,
    /// Overall model precision (how well the generative model predicts).
    pub model_precision: f32,
    /// Observation precision (how noisy the observations are).
    pub observation_precision: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// 20b — Policy & Selection
// ═══════════════════════════════════════════════════════════════════════

/// A candidate policy (action sequence) with its EFE evaluation.
#[derive(Debug, Clone)]
pub struct Policy {
    /// Ordered action sequence (1–3 steps).
    pub actions: Vec<SymbolId>,
    /// Free energy decomposition.
    pub efe: FreeEnergy,
    /// Selection probability (from softmax).
    pub selection_probability: f32,
}

/// Result of the active inference decide phase.
#[derive(Debug, Clone)]
pub struct PolicySelectionResult {
    /// The selected policy.
    pub selected: Policy,
    /// All evaluated policies for transparency.
    pub all_policies: Vec<Policy>,
    /// Whether MCTS was used for pruning.
    pub mcts_used: bool,
    /// Human-readable reasoning.
    pub reasoning: String,
}

// ═══════════════════════════════════════════════════════════════════════
// ActiveInferenceEngine
// ═══════════════════════════════════════════════════════════════════════

/// Active inference engine: prediction, surprise, free energy, policy selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceEngine {
    /// Configuration.
    pub config: ActiveInferenceConfig,
    /// Precision weights (adapted over time).
    pub precision: PrecisionWeights,
    /// Cumulative surprise for monitoring.
    pub cumulative_surprise: f32,
    /// Number of inference cycles run.
    pub cycle_count: u64,
}

impl Default for ActiveInferenceEngine {
    fn default() -> Self {
        Self {
            config: ActiveInferenceConfig::default(),
            precision: PrecisionWeights {
                model_precision: 1.0,
                observation_precision: 1.0,
                source_precision: HashMap::new(),
            },
            cumulative_surprise: 0.0,
            cycle_count: 0,
        }
    }
}

impl ActiveInferenceEngine {
    pub fn new(config: ActiveInferenceConfig) -> Self {
        Self {
            config,
            ..Self::default()
        }
    }

    // ─── 20a: Prediction & Surprise ───────────────────────────────

    /// Generate predictions from current belief state.
    ///
    /// Uses recent high-confidence triples as expected observations.
    pub fn generate_predictions(&self, recent_triples: &[Triple]) -> GenerativeModel {
        let expected: Vec<_> = recent_triples
            .iter()
            .filter(|t| t.confidence > 0.5)
            .map(|t| (t.subject, t.predicate, t.object, t.confidence))
            .collect();

        GenerativeModel {
            expected_observations: expected,
            transition_model_ready: true,
            preferred_states: Vec::new(),
            belief_state: recent_triples
                .iter()
                .map(|t| (t.subject, t.predicate, t.object, t.confidence))
                .collect(),
        }
    }

    /// Compute prediction error between expected and actual observations.
    pub fn compute_prediction_error(
        &self,
        model: &GenerativeModel,
        actual: &[Triple],
    ) -> PredictionError {
        // False positives: expected but not observed.
        let actual_set: Vec<(SymbolId, SymbolId, SymbolId)> = actual
            .iter()
            .map(|t| (t.subject, t.predicate, t.object))
            .collect();

        let false_positives: Vec<_> = model
            .expected_observations
            .iter()
            .filter(|(s, p, o, _)| !actual_set.contains(&(*s, *p, *o)))
            .cloned()
            .collect();

        // False negatives: observed but not expected.
        let expected_set: Vec<(SymbolId, SymbolId, SymbolId)> = model
            .expected_observations
            .iter()
            .map(|(s, p, o, _)| (*s, *p, *o))
            .collect();

        let false_negatives: Vec<_> = actual
            .iter()
            .filter(|t| !expected_set.contains(&(t.subject, t.predicate, t.object)))
            .map(|t| (t.subject, t.predicate, t.object, t.confidence))
            .collect();

        let fp_count = false_positives.len() as f32;
        let fn_count = false_negatives.len() as f32;
        let total = model.expected_observations.len().max(1) as f32;

        let total_error = (fp_count + fn_count) / total;
        let surprise = -((1.0 - total_error).max(0.01)).ln();
        let precision = self.precision.model_precision;

        PredictionError {
            false_positives,
            false_negatives,
            total_error,
            surprise,
            precision,
        }
    }

    /// Should the model be updated given the surprise level?
    pub fn should_update_model(&self, surprise: f32) -> bool {
        surprise > self.config.surprise_threshold
    }

    /// Update model precision based on prediction accuracy.
    pub fn update_precision(&mut self, error: &PredictionError) {
        let lr = self.config.precision_learning_rate;
        // Low error → increase precision; high error → decrease.
        let target = 1.0 / (1.0 + error.total_error);
        self.precision.model_precision += lr * (target - self.precision.model_precision);
        self.precision.model_precision = self.precision.model_precision.clamp(0.1, 10.0);
    }

    // ─── 20a: Free Energy Computation ─────────────────────────────

    /// Compute pragmatic value of an action (expected goal progress).
    ///
    /// Higher = more progress toward preferred states.
    pub fn pragmatic_value(
        &self,
        action_effects_assertions: usize,
        action_effects_retractions: usize,
        goal_relevant_assertions: usize,
    ) -> f32 {
        // Net positive effect toward goals.
        let net = goal_relevant_assertions as f32
            + action_effects_assertions as f32 * 0.1
            - action_effects_retractions as f32 * 0.2;
        net.max(-1.0).min(1.0)
    }

    /// Compute epistemic value of an action (expected uncertainty reduction).
    ///
    /// Higher = more information gain.
    pub fn epistemic_value(
        &self,
        uncertain_claims_resolved: usize,
        total_uncertain: usize,
    ) -> f32 {
        if total_uncertain == 0 {
            return 0.0;
        }
        uncertain_claims_resolved as f32 / total_uncertain as f32
    }

    /// Compute expected free energy for a candidate action.
    ///
    /// EFE = -(w_p * pragmatic + w_e * epistemic). Lower EFE = better action.
    pub fn expected_free_energy(
        &self,
        pragmatic: f32,
        epistemic: f32,
    ) -> FreeEnergy {
        let efe = -(self.config.pragmatic_weight * pragmatic
            + self.config.epistemic_weight * epistemic);

        FreeEnergy {
            pragmatic_value: pragmatic,
            epistemic_value: epistemic,
            expected_free_energy: efe,
            reasoning: format!(
                "EFE={efe:.3} (pragmatic={pragmatic:.3}×{:.1} + epistemic={epistemic:.3}×{:.1})",
                self.config.pragmatic_weight, self.config.epistemic_weight
            ),
        }
    }

    // ─── 20b: Policy Selection ────────────────────────────────────

    /// Select a policy using softmax over negative EFE.
    ///
    /// Temperature adapts with model precision: low precision → high temperature
    /// (more exploration), high precision → low temperature (more exploitation).
    pub fn select_policy(&self, policies: &mut [Policy]) -> Option<usize> {
        if policies.is_empty() {
            return None;
        }
        if policies.len() == 1 {
            policies[0].selection_probability = 1.0;
            return Some(0);
        }

        // Adaptive temperature: inversely proportional to model precision.
        let temp = self.config.temperature / self.precision.model_precision.max(0.1);

        // Compute softmax probabilities over negative EFE.
        let efes: Vec<f32> = policies.iter().map(|p| -p.efe.expected_free_energy).collect();
        let max_efe = efes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = efes.iter().map(|&e| ((e - max_efe) / temp).exp()).collect::<Vec<f32>>().iter().sum();

        for (i, policy) in policies.iter_mut().enumerate() {
            policy.selection_probability = ((efes[i] - max_efe) / temp).exp() / exp_sum;
        }

        // Select the highest probability (deterministic for reproducibility).
        policies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.selection_probability
                    .partial_cmp(&b.selection_probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Run one active inference cycle: predict, observe, compute surprise, update.
    pub fn run_cycle(
        &mut self,
        predicted: &GenerativeModel,
        actual_observations: &[Triple],
    ) -> PredictionError {
        let error = self.compute_prediction_error(predicted, actual_observations);
        self.update_precision(&error);
        self.cumulative_surprise += error.surprise;
        self.cycle_count += 1;
        error
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &crate::engine::Engine) -> Result<(), String> {
        let bytes = bincode::serialize(self)
            .map_err(|e| format!("serialize active inference: {e}"))?;
        engine
            .store()
            .put_meta(b"agent:active_inference", &bytes)
            .map_err(|e| format!("persist active inference: {e}"))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &crate::engine::Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:active_inference")
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
    use crate::graph::Triple;
    use crate::symbol::SymbolId;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn make_triple(s: u64, p: u64, o: u64, conf: f32) -> Triple {
        Triple::new(sym(s), sym(p), sym(o)).with_confidence(conf)
    }

    // ── Config ────────────────────────────────────────────────────

    #[test]
    fn config_default_weights() {
        let config = ActiveInferenceConfig::default();
        assert!((config.pragmatic_weight + config.epistemic_weight - 1.0).abs() < f32::EPSILON);
    }

    // ── Generative Model ──────────────────────────────────────────

    #[test]
    fn generative_model_creates_predictions() {
        let engine = ActiveInferenceEngine::default();
        let triples = vec![
            make_triple(1, 2, 3, 0.8),
            make_triple(4, 5, 6, 0.3), // below threshold
        ];
        let model = engine.generate_predictions(&triples);
        assert_eq!(model.expected_observations.len(), 1); // only high-conf
        assert_eq!(model.belief_state.len(), 2); // all triples
    }

    // ── Prediction Error ──────────────────────────────────────────

    #[test]
    fn prediction_error_false_positives() {
        let engine = ActiveInferenceEngine::default();
        let model = GenerativeModel {
            expected_observations: vec![(sym(1), sym(2), sym(3), 0.8)],
            ..Default::default()
        };
        // Actual: empty → everything is a false positive.
        let error = engine.compute_prediction_error(&model, &[]);
        assert_eq!(error.false_positives.len(), 1);
        assert!(error.false_negatives.is_empty());
    }

    #[test]
    fn prediction_error_false_negatives() {
        let engine = ActiveInferenceEngine::default();
        let model = GenerativeModel::default();
        let actual = vec![make_triple(1, 2, 3, 0.9)];
        let error = engine.compute_prediction_error(&model, &actual);
        assert!(error.false_positives.is_empty());
        assert_eq!(error.false_negatives.len(), 1);
    }

    #[test]
    fn prediction_error_total() {
        let engine = ActiveInferenceEngine::default();
        let model = GenerativeModel {
            expected_observations: vec![
                (sym(1), sym(2), sym(3), 0.8),
                (sym(4), sym(5), sym(6), 0.7),
            ],
            ..Default::default()
        };
        let actual = vec![make_triple(1, 2, 3, 0.9)]; // one match, one miss
        let error = engine.compute_prediction_error(&model, &actual);
        assert_eq!(error.false_positives.len(), 1);
        assert!(error.total_error > 0.0);
    }

    #[test]
    fn surprise_computation() {
        let engine = ActiveInferenceEngine::default();
        let model = GenerativeModel {
            expected_observations: vec![(sym(1), sym(2), sym(3), 0.8)],
            ..Default::default()
        };
        let actual = vec![make_triple(1, 2, 3, 0.9)];
        let error = engine.compute_prediction_error(&model, &actual);
        // Perfect prediction → low surprise.
        assert!(error.surprise < 1.0, "surprise={}", error.surprise);
    }

    // ── Precision ─────────────────────────────────────────────────

    #[test]
    fn model_precision_adapts() {
        let mut engine = ActiveInferenceEngine::default();
        engine.precision.model_precision = 0.5; // start low

        // Low error → precision should increase toward target ~0.91.
        let low_error = PredictionError {
            false_positives: vec![],
            false_negatives: vec![],
            total_error: 0.1,
            surprise: 0.1,
            precision: 1.0,
        };
        let before_low = engine.precision.model_precision;
        engine.update_precision(&low_error);
        assert!(
            engine.precision.model_precision > before_low,
            "low error should increase precision: {} vs {}",
            engine.precision.model_precision, before_low
        );

        // High error → precision should decrease.
        engine.precision.model_precision = 2.0; // start high
        let high_error = PredictionError {
            false_positives: vec![(sym(1), sym(2), sym(3), 0.8)],
            false_negatives: vec![(sym(4), sym(5), sym(6), 0.7)],
            total_error: 0.9,
            surprise: 2.0,
            precision: 1.0,
        };
        let before_high = engine.precision.model_precision;
        engine.update_precision(&high_error);
        assert!(
            engine.precision.model_precision < before_high,
            "high error should decrease precision: {} vs {}",
            engine.precision.model_precision, before_high
        );
    }

    // ── Free Energy ───────────────────────────────────────────────

    #[test]
    fn pragmatic_value_toward_goal() {
        let engine = ActiveInferenceEngine::default();
        let pv = engine.pragmatic_value(5, 0, 3);
        assert!(pv > 0.0, "goal-relevant assertions → positive pragmatic");
    }

    #[test]
    fn pragmatic_value_away_from_goal() {
        let engine = ActiveInferenceEngine::default();
        let pv = engine.pragmatic_value(0, 5, 0);
        assert!(pv < 0.0, "retractions without progress → negative pragmatic");
    }

    #[test]
    fn epistemic_value_resolves_uncertainty() {
        let engine = ActiveInferenceEngine::default();
        let ev = engine.epistemic_value(3, 10);
        assert!((ev - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn efe_combines_both() {
        let engine = ActiveInferenceEngine::default();
        let efe = engine.expected_free_energy(0.8, 0.5);
        // EFE = -(0.6*0.8 + 0.4*0.5) = -(0.48 + 0.20) = -0.68
        assert!((efe.expected_free_energy - (-0.68)).abs() < 0.01);
    }

    // ── Policy Selection ──────────────────────────────────────────

    #[test]
    fn softmax_selection() {
        let engine = ActiveInferenceEngine::default();
        let mut policies = vec![
            Policy {
                actions: vec![sym(1)],
                efe: FreeEnergy {
                    pragmatic_value: 0.9,
                    epistemic_value: 0.5,
                    expected_free_energy: -0.74,
                    reasoning: String::new(),
                },
                selection_probability: 0.0,
            },
            Policy {
                actions: vec![sym(2)],
                efe: FreeEnergy {
                    pragmatic_value: 0.1,
                    epistemic_value: 0.1,
                    expected_free_energy: -0.10,
                    reasoning: String::new(),
                },
                selection_probability: 0.0,
            },
        ];

        let selected = engine.select_policy(&mut policies).unwrap();
        // Lower EFE = better → policy 0 should win.
        assert_eq!(selected, 0);
        assert!(policies[0].selection_probability > policies[1].selection_probability);
    }

    #[test]
    fn softmax_single_policy() {
        let engine = ActiveInferenceEngine::default();
        let mut policies = vec![Policy {
            actions: vec![sym(1)],
            efe: FreeEnergy {
                pragmatic_value: 0.5,
                epistemic_value: 0.5,
                expected_free_energy: -0.5,
                reasoning: String::new(),
            },
            selection_probability: 0.0,
        }];

        let selected = engine.select_policy(&mut policies);
        assert_eq!(selected, Some(0));
        assert!((policies[0].selection_probability - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn softmax_empty() {
        let engine = ActiveInferenceEngine::default();
        let mut policies: Vec<Policy> = vec![];
        assert!(engine.select_policy(&mut policies).is_none());
    }

    // ── Full Cycle ────────────────────────────────────────────────

    #[test]
    fn run_cycle_updates_state() {
        let mut engine = ActiveInferenceEngine::default();
        assert_eq!(engine.cycle_count, 0);

        // Model expects (1,2,3) but actual also has (4,5,6) — partial mismatch.
        let model = GenerativeModel {
            expected_observations: vec![(sym(1), sym(2), sym(3), 0.8)],
            ..Default::default()
        };
        let actual = vec![make_triple(1, 2, 3, 0.9), make_triple(4, 5, 6, 0.7)];
        let error = engine.run_cycle(&model, &actual);

        assert_eq!(engine.cycle_count, 1);
        assert!(engine.cumulative_surprise > 0.0, "surprise={}", engine.cumulative_surprise);
        assert!(error.false_negatives.len() == 1); // (4,5,6) unexpected
    }

    // ── Serialization ─────────────────────────────────────────────

    #[test]
    fn serialization_roundtrip() {
        let mut engine = ActiveInferenceEngine::default();
        engine.cycle_count = 42;
        engine.cumulative_surprise = 3.14;

        let bytes = bincode::serialize(&engine).unwrap();
        let restored: ActiveInferenceEngine = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.cycle_count, 42);
        assert!((restored.cumulative_surprise - 3.14).abs() < f32::EPSILON);
    }
}
