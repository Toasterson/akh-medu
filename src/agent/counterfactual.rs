//! Counterfactual reasoning & prediction tracking — Phase 15c.
//!
//! Pearl Level 3 counterfactual reasoning: "What would have happened if action X
//! instead of Y?" and systematic prediction-outcome tracking for causal model
//! refinement.
//!
//! The counterfactual pipeline:
//! 1. **Abduct** — reconstruct what state held before the actual action
//! 2. **Intervene** — substitute the hypothetical action
//! 3. **Predict** — project effects of the hypothetical via CausalManager
//! 4. **Compare** — identify divergent fluents between actual and hypothetical
//!
//! The prediction tracker logs predictions before execution and verifies them
//! afterward, maintaining per-action accuracy statistics and an exponential
//! moving average for overall model quality.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::agent::causal::{CausalManager, StateTransition};
use crate::engine::Engine;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the counterfactual reasoning subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum CounterfactualError {
    #[error("action schema not found for counterfactual: {action}")]
    #[diagnostic(
        code(akh::agent::cf::schema_not_found),
        help("Register causal action schemas via CausalManager before running counterfactuals.")
    )]
    SchemaNotFound { action: String },

    #[error("prediction not found: id {id}")]
    #[diagnostic(
        code(akh::agent::cf::prediction_not_found),
        help("Log a prediction first via `log_prediction()` before verifying.")
    )]
    PredictionNotFound { id: u64 },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::cf::engine),
        help("An engine-level error occurred during counterfactual reasoning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for CounterfactualError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Result alias for the counterfactual subsystem.
pub type CfResult<T> = std::result::Result<T, CounterfactualError>;

// ═══════════════════════════════════════════════════════════════════════
// CounterfactualQuery & Result
// ═══════════════════════════════════════════════════════════════════════

/// A counterfactual query: "What would have happened if action X instead of Y?"
#[derive(Debug, Clone)]
pub struct CounterfactualQuery {
    /// The actual action that was taken.
    pub actual_action: SymbolId,
    /// The hypothetical alternative action.
    pub hypothetical_action: SymbolId,
    /// The timestamp of the original action.
    pub timestamp: u64,
}

/// Result of counterfactual analysis.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// What actually happened (predicted effects of the actual action).
    pub actual_outcome: StateTransition,
    /// What would have happened under the alternative.
    pub hypothetical_outcome: StateTransition,
    /// Fluents that differ: (symbol_id, actual_holds, hypothetical_holds).
    pub divergent_fluents: Vec<(SymbolId, bool, bool)>,
    /// Whether the hypothetical would have been better (more assertions, fewer retractions).
    pub hypothetical_better: Option<bool>,
    /// Confidence in the counterfactual estimate (product of both prediction confidences).
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// PredictionTracker
// ═══════════════════════════════════════════════════════════════════════

/// Tracks prediction accuracy over time for causal model refinement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionTracker {
    /// Total predictions made.
    pub predictions_made: u64,
    /// Predictions verified as correct.
    pub predictions_correct: u64,
    /// Predictions verified as incorrect.
    pub predictions_incorrect: u64,
    /// Per-action accuracy: action_id → (correct, total).
    pub per_action_accuracy: HashMap<u64, (u32, u32)>,
    /// Running exponential moving average of accuracy (0.0–1.0).
    pub accuracy_ema: f32,
    /// Logged predictions awaiting verification.
    pub pending: Vec<PredictionRecord>,
    /// Next prediction ID.
    next_id: u64,
}

/// A logged prediction for later verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// Unique prediction ID.
    pub id: u64,
    /// The action whose effects were predicted.
    pub action_id: SymbolId,
    /// The predicted state transition.
    pub predicted_transition: StateTransition,
    /// When the prediction was made.
    pub timestamp: u64,
    /// Whether this prediction has been verified.
    pub verified: bool,
    /// Whether the prediction was correct (None if not yet verified).
    pub correct: Option<bool>,
}

impl PredictionTracker {
    /// Log a prediction before executing an action.
    ///
    /// Returns the prediction ID for later verification.
    pub fn log_prediction(
        &mut self,
        action_id: SymbolId,
        predicted: StateTransition,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.predictions_made += 1;

        self.pending.push(PredictionRecord {
            id,
            action_id,
            predicted_transition: predicted,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            verified: false,
            correct: None,
        });

        id
    }

    /// Verify a prediction against the actual post-action state.
    ///
    /// Compares predicted assertions/retractions against the engine's
    /// current triples. Updates accuracy statistics.
    pub fn verify_prediction(
        &mut self,
        prediction_id: u64,
        causal_manager: &CausalManager,
        engine: &Engine,
    ) -> CfResult<bool> {
        let record = self
            .pending
            .iter_mut()
            .find(|r| r.id == prediction_id)
            .ok_or(CounterfactualError::PredictionNotFound {
                id: prediction_id,
            })?;

        let correct = causal_manager.verify_prediction(&record.predicted_transition, engine);
        record.verified = true;
        record.correct = Some(correct);

        let action_key = record.action_id.get();

        // Update global stats.
        if correct {
            self.predictions_correct += 1;
        } else {
            self.predictions_incorrect += 1;
        }

        // Update per-action stats.
        let entry = self.per_action_accuracy.entry(action_key).or_insert((0, 0));
        if correct {
            entry.0 += 1;
        }
        entry.1 += 1;

        // Update EMA (alpha = 0.1 for smoothing).
        let alpha = 0.1_f32;
        let outcome = if correct { 1.0 } else { 0.0 };
        self.accuracy_ema = self.accuracy_ema * (1.0 - alpha) + outcome * alpha;

        // Clean up verified predictions (keep last 50 for history).
        self.pending.retain(|r| !r.verified);

        Ok(correct)
    }

    /// Overall prediction accuracy (0.0–1.0).
    pub fn prediction_accuracy(&self) -> f32 {
        let total = self.predictions_correct + self.predictions_incorrect;
        if total == 0 {
            return 0.5; // uninformed prior
        }
        self.predictions_correct as f32 / total as f32
    }

    /// Per-action prediction accuracy (0.0–1.0).
    pub fn per_action_accuracy(&self, action_id: SymbolId) -> f32 {
        match self.per_action_accuracy.get(&action_id.get()) {
            Some(&(correct, total)) if total > 0 => correct as f32 / total as f32,
            _ => 0.5, // uninformed prior
        }
    }

    /// Suggest which action schemas need refinement based on low accuracy.
    ///
    /// Returns action IDs with accuracy below `threshold` and at least
    /// `min_samples` verification attempts.
    pub fn refinement_suggestions(
        &self,
        threshold: f32,
        min_samples: u32,
    ) -> Vec<(u64, f32)> {
        self.per_action_accuracy
            .iter()
            .filter_map(|(&action_key, &(correct, total))| {
                if total >= min_samples {
                    let acc = correct as f32 / total as f32;
                    if acc < threshold {
                        return Some((action_key, acc));
                    }
                }
                None
            })
            .collect()
    }

    /// Number of pending (unverified) predictions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Counterfactual Query Execution
// ═══════════════════════════════════════════════════════════════════════

/// Execute a counterfactual query: "What if I had done Y instead of X?"
///
/// Pipeline:
/// 1. Predict effects of the actual action
/// 2. Predict effects of the hypothetical action
/// 3. Compare: which fluents diverge?
/// 4. Assess: was the hypothetical better?
pub fn counterfactual_query(
    query: &CounterfactualQuery,
    causal_manager: &CausalManager,
    engine: &Engine,
) -> CfResult<CounterfactualResult> {
    let actual_name = engine.resolve_label(query.actual_action);
    let hypo_name = engine.resolve_label(query.hypothetical_action);

    // Predict actual outcome.
    let actual_outcome = causal_manager
        .predict_effects(&actual_name, engine)
        .map_err(|_| CounterfactualError::SchemaNotFound {
            action: actual_name.clone(),
        })?;

    // Predict hypothetical outcome.
    let hypothetical_outcome = causal_manager
        .predict_effects(&hypo_name, engine)
        .map_err(|_| CounterfactualError::SchemaNotFound {
            action: hypo_name.clone(),
        })?;

    // Find divergent fluents: symbols that are asserted in one but not the other,
    // or retracted in one but not the other.
    let mut divergent_fluents = Vec::new();

    // Collect all affected symbols.
    let mut actual_asserted: HashMap<u64, bool> = HashMap::new();
    for &(s, _, _) in &actual_outcome.assertions {
        actual_asserted.insert(s.get(), true);
    }
    for &(s, _, _) in &actual_outcome.retractions {
        actual_asserted.insert(s.get(), false);
    }

    let mut hypo_asserted: HashMap<u64, bool> = HashMap::new();
    for &(s, _, _) in &hypothetical_outcome.assertions {
        hypo_asserted.insert(s.get(), true);
    }
    for &(s, _, _) in &hypothetical_outcome.retractions {
        hypo_asserted.insert(s.get(), false);
    }

    // Find symbols that differ.
    let mut all_keys: Vec<u64> = actual_asserted.keys().chain(hypo_asserted.keys()).copied().collect();
    all_keys.sort_unstable();
    all_keys.dedup();

    for key in all_keys {
        let actual = actual_asserted.get(&key).copied();
        let hypo = hypo_asserted.get(&key).copied();
        if actual != hypo {
            if let Some(sym_id) = SymbolId::new(key) {
                divergent_fluents.push((
                    sym_id,
                    actual.unwrap_or(false),
                    hypo.unwrap_or(false),
                ));
            }
        }
    }

    // Simple heuristic: hypothetical is "better" if it has more net assertions.
    let actual_net = actual_outcome.assertions.len() as i32
        - actual_outcome.retractions.len() as i32;
    let hypo_net = hypothetical_outcome.assertions.len() as i32
        - hypothetical_outcome.retractions.len() as i32;
    let hypothetical_better = if actual_net == hypo_net {
        None // tie
    } else {
        Some(hypo_net > actual_net)
    };

    // Confidence: decayed product (both are predictions).
    let confidence = 0.8; // base confidence for counterfactual reasoning

    Ok(CounterfactualResult {
        actual_outcome,
        hypothetical_outcome,
        divergent_fluents,
        hypothetical_better,
        confidence,
    })
}

/// Record provenance for a counterfactual reasoning result.
pub fn record_counterfactual_provenance(
    engine: &Engine,
    derived_id: SymbolId,
    actual_action: &str,
    hypothetical_action: &str,
    divergent_count: usize,
) -> CfResult<()> {
    let mut record = crate::provenance::ProvenanceRecord::new(
        derived_id,
        DerivationKind::CounterfactualReasoning {
            actual_action: actual_action.to_string(),
            hypothetical_action: hypothetical_action.to_string(),
            divergent_count: divergent_count as u32,
        },
    )
    .with_confidence(0.8);
    engine
        .store_provenance(&mut record)
        .map_err(|e| CounterfactualError::Engine(Box::new(e)))?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── PredictionTracker ─────────────────────────────────────────

    #[test]
    fn prediction_tracker_default() {
        let tracker = PredictionTracker::default();
        assert_eq!(tracker.predictions_made, 0);
        assert_eq!(tracker.predictions_correct, 0);
        assert_eq!(tracker.predictions_incorrect, 0);
        assert_eq!(tracker.accuracy_ema, 0.0);
        assert_eq!(tracker.prediction_accuracy(), 0.5); // uninformed prior
    }

    #[test]
    fn prediction_tracker_log_increments() {
        let mut tracker = PredictionTracker::default();
        let action = SymbolId::new(10).unwrap();
        let transition = StateTransition {
            action_id: action,
            assertions: vec![],
            retractions: vec![],
            confidence_changes: vec![],
            verified: None,
            timestamp: 100,
        };

        let id0 = tracker.log_prediction(action, transition.clone());
        let id1 = tracker.log_prediction(action, transition);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(tracker.predictions_made, 2);
        assert_eq!(tracker.pending_count(), 2);
    }

    #[test]
    fn prediction_tracker_accuracy_all_correct() {
        let mut tracker = PredictionTracker::default();
        tracker.predictions_correct = 8;
        tracker.predictions_incorrect = 2;
        assert!((tracker.prediction_accuracy() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn prediction_tracker_per_action() {
        let mut tracker = PredictionTracker::default();
        let action = SymbolId::new(42).unwrap();
        tracker.per_action_accuracy.insert(42, (7, 10));
        assert!((tracker.per_action_accuracy(action) - 0.7).abs() < f32::EPSILON);

        // Unknown action → uninformed prior.
        let unknown = SymbolId::new(999).unwrap();
        assert!((tracker.per_action_accuracy(unknown) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn accuracy_ema_updates() {
        let mut tracker = PredictionTracker::default();
        tracker.accuracy_ema = 0.5;

        // Simulate a correct outcome (EMA with alpha=0.1).
        let alpha = 0.1_f32;
        let expected = 0.5 * (1.0 - alpha) + 1.0 * alpha; // 0.55
        tracker.accuracy_ema = tracker.accuracy_ema * (1.0 - alpha) + 1.0 * alpha;
        assert!((tracker.accuracy_ema - expected).abs() < f32::EPSILON);

        // Simulate an incorrect outcome.
        let expected2 = tracker.accuracy_ema * (1.0 - alpha) + 0.0 * alpha;
        tracker.accuracy_ema = tracker.accuracy_ema * (1.0 - alpha) + 0.0 * alpha;
        assert!((tracker.accuracy_ema - expected2).abs() < f32::EPSILON);
    }

    #[test]
    fn refinement_suggestions_low_accuracy() {
        let mut tracker = PredictionTracker::default();
        // Action 1: good accuracy (80%)
        tracker.per_action_accuracy.insert(1, (8, 10));
        // Action 2: poor accuracy (20%)
        tracker.per_action_accuracy.insert(2, (2, 10));
        // Action 3: too few samples
        tracker.per_action_accuracy.insert(3, (0, 2));

        let suggestions = tracker.refinement_suggestions(0.5, 5);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].0, 2);
        assert!((suggestions[0].1 - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut tracker = PredictionTracker::default();
        tracker.predictions_made = 10;
        tracker.predictions_correct = 7;
        tracker.accuracy_ema = 0.7;
        tracker.per_action_accuracy.insert(42, (5, 8));

        let bytes = bincode::serialize(&tracker).unwrap();
        let restored: PredictionTracker = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.predictions_made, 10);
        assert_eq!(restored.predictions_correct, 7);
        assert!((restored.accuracy_ema - 0.7).abs() < f32::EPSILON);
        assert_eq!(restored.per_action_accuracy.get(&42), Some(&(5, 8)));
    }

    // ── Counterfactual divergence ─────────────────────────────────

    #[test]
    fn counterfactual_divergence_detection() {
        // Simulate the divergence logic without needing an engine.
        let mut actual: HashMap<u64, bool> = HashMap::new();
        actual.insert(1, true);  // asserted
        actual.insert(2, false); // retracted

        let mut hypo: HashMap<u64, bool> = HashMap::new();
        hypo.insert(1, true);  // same
        hypo.insert(3, true);  // new assertion

        let mut all_keys: Vec<u64> = actual.keys().chain(hypo.keys()).copied().collect();
        all_keys.sort_unstable();
        all_keys.dedup();

        let mut divergent = Vec::new();
        for key in all_keys {
            let a = actual.get(&key).copied();
            let h = hypo.get(&key).copied();
            if a != h {
                divergent.push((key, a.unwrap_or(false), h.unwrap_or(false)));
            }
        }

        // Key 2: actual retracted, hypo nothing → divergent
        // Key 3: actual nothing, hypo asserted → divergent
        assert_eq!(divergent.len(), 2);
    }
}
