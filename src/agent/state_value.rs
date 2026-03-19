//! State encoding & TD value function — Phase 16a.
//!
//! Provides a compact state representation for planning and a temporal-difference
//! (TD) learned value function for evaluating states. The value function maps
//! agent states to expected cumulative rewards, updated via TD(0):
//!
//!   V(s) += α * [r + γ·V(s') − V(s)]
//!
//! State encoding captures active goals + progress, KG fluent count, and an
//! optional VSA fingerprint for similarity-based generalization to unseen states.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;

use super::goal::{Goal, GoalStatus};

// ═══════════════════════════════════════════════════════════════════════
// AgentState
// ═══════════════════════════════════════════════════════════════════════

/// A compact snapshot of agent state for planning and value estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Active goal IDs and their progress estimates (0.0–1.0).
    pub goal_states: Vec<(SymbolId, f32)>,
    /// Number of triples in the KG (proxy for knowledge level).
    pub fluent_count: usize,
    /// VSA fingerprint of the state (for similarity-based generalization).
    #[serde(skip)]
    pub state_vec: Option<HyperVec>,
    /// Timestamp of this snapshot.
    pub timestamp: u64,
}

impl AgentState {
    /// Compute a deterministic hash for value table lookup.
    pub fn state_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash goal IDs and quantized progress.
        for (id, progress) in &self.goal_states {
            id.get().hash(&mut hasher);
            // Quantize to 10 bins for stability.
            let bin = (*progress * 10.0) as u32;
            bin.hash(&mut hasher);
        }
        // Hash fluent count (quantized to 100s).
        (self.fluent_count / 100).hash(&mut hasher);
        hasher.finish()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// StateRoleVectors
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic VSA role vectors for state encoding.
#[derive(Debug, Clone)]
pub struct StateRoleVectors {
    pub goal_progress: HyperVec,
    pub fluent_active: HyperVec,
    pub resource_available: HyperVec,
    pub memory_pressure: HyperVec,
}

impl StateRoleVectors {
    /// Create role vectors from deterministic tokens.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            goal_progress: encode_token(ops, "state-role:goal-progress"),
            fluent_active: encode_token(ops, "state-role:fluent-active"),
            resource_available: encode_token(ops, "state-role:resource-available"),
            memory_pressure: encode_token(ops, "state-role:memory-pressure"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Reward
// ═══════════════════════════════════════════════════════════════════════

/// Multi-factor reward signal after an action.
#[derive(Debug, Clone)]
pub struct Reward {
    /// Goal progress delta (positive = progress, negative = regression).
    pub goal_progress: f32,
    /// Knowledge gained (new triples asserted).
    pub knowledge_gain: f32,
    /// Bonus for accurate predictions.
    pub prediction_accuracy: f32,
    /// Combined scalar reward.
    pub total: f32,
}

impl Reward {
    /// Compute reward from before/after states and prediction accuracy.
    pub fn compute(before: &AgentState, after: &AgentState, prediction_correct: bool) -> Self {
        // Goal progress: sum of progress deltas across all goals.
        let before_progress: f32 = before.goal_states.iter().map(|(_, p)| p).sum();
        let after_progress: f32 = after.goal_states.iter().map(|(_, p)| p).sum();
        let goal_progress = after_progress - before_progress;

        // Knowledge gain: normalized triple count increase.
        let knowledge_gain = if before.fluent_count > 0 {
            (after.fluent_count as f32 - before.fluent_count as f32)
                / before.fluent_count as f32
        } else if after.fluent_count > 0 {
            1.0
        } else {
            0.0
        };

        let prediction_accuracy = if prediction_correct { 0.1 } else { 0.0 };

        // Weighted combination.
        let total = goal_progress * 1.0 + knowledge_gain * 0.3 + prediction_accuracy;

        Reward {
            goal_progress,
            knowledge_gain,
            prediction_accuracy,
            total,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ValueFunction
// ═══════════════════════════════════════════════════════════════════════

/// TD-learned value function mapping states to expected cumulative reward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFunction {
    /// State hash → value estimate.
    pub state_values: HashMap<u64, f32>,
    /// Learning rate (α). Typical: 0.1.
    pub alpha: f32,
    /// Discount factor (γ). Typical: 0.99.
    pub gamma: f32,
    /// Total TD updates performed.
    pub update_count: u64,
}

impl Default for ValueFunction {
    fn default() -> Self {
        Self {
            state_values: HashMap::new(),
            alpha: 0.1,
            gamma: 0.99,
            update_count: 0,
        }
    }
}

impl ValueFunction {
    /// Create with custom hyperparameters.
    pub fn with_params(alpha: f32, gamma: f32) -> Self {
        Self {
            alpha,
            gamma,
            ..Self::default()
        }
    }

    /// Get the value of a state (0.0 for unseen states).
    pub fn state_value(&self, state: &AgentState) -> f32 {
        let hash = state.state_hash();
        self.state_values.get(&hash).copied().unwrap_or(0.0)
    }

    /// TD(0) update: V(s) += α * [r + γ·V(s') − V(s)].
    pub fn td_update(&mut self, state: &AgentState, reward: f32, next_state: &AgentState) {
        let hash = state.state_hash();
        let current_v = self.state_values.get(&hash).copied().unwrap_or(0.0);
        let next_v = self.state_value(next_state);

        let td_error = reward + self.gamma * next_v - current_v;
        let new_v = current_v + self.alpha * td_error;

        self.state_values.insert(hash, new_v);
        self.update_count += 1;
    }

    /// Generalize value for an unseen state using VSA similarity.
    ///
    /// Finds the k-nearest known states (by VSA similarity) and returns
    /// a similarity-weighted average of their values.
    pub fn generalize_value(&self, state: &AgentState, ops: &VsaOps, k: usize) -> f32 {
        let state_vec = match &state.state_vec {
            Some(v) => v,
            None => return self.state_value(state), // fallback to exact lookup
        };

        // If exact match exists, return it.
        let hash = state.state_hash();
        if let Some(&v) = self.state_values.get(&hash) {
            return v;
        }

        // No VSA-indexed states available — would need a separate index.
        // For now, return the average of all known state values as a rough prior.
        if self.state_values.is_empty() {
            return 0.0;
        }

        // Simple average as prior (VSA-indexed nearest-neighbor requires
        // storing state_vecs alongside hashes — planned for 16b).
        let _ = (state_vec, ops, k); // will be used in 16b with HNSW index
        let sum: f32 = self.state_values.values().sum();
        sum / self.state_values.len() as f32
    }

    /// Number of distinct states with value estimates.
    pub fn known_states(&self) -> usize {
        self.state_values.len()
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> Result<(), String> {
        let bytes = bincode::serialize(self)
            .map_err(|e| format!("serialize value function: {e}"))?;
        engine
            .store()
            .put_meta(b"agent:value_function", &bytes)
            .map_err(|e| format!("persist value function: {e}"))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> Self {
        engine
            .store()
            .get_meta(b"agent:value_function")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// State Encoding
// ═══════════════════════════════════════════════════════════════════════

/// Encode current agent state into a compact `AgentState` snapshot.
pub fn encode_state(goals: &[Goal], engine: &Engine) -> AgentState {
    let goal_states: Vec<(SymbolId, f32)> = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .map(|g| {
            // Estimate progress as cycles_worked / (cycles_worked + stall_threshold).
            let progress = if g.cycles_worked > 0 {
                g.cycles_worked as f32 / (g.cycles_worked as f32 + 10.0)
            } else {
                0.0
            };
            (g.symbol_id, progress)
        })
        .collect();

    let fluent_count = engine.all_triples().len();

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    AgentState {
        goal_states,
        fluent_count,
        state_vec: None, // VSA encoding added when needed for generalization
        timestamp,
    }
}

/// Encode state with VSA fingerprint for similarity-based generalization.
pub fn encode_state_with_vsa(
    goals: &[Goal],
    engine: &Engine,
    ops: &VsaOps,
    roles: &StateRoleVectors,
) -> AgentState {
    let mut state = encode_state(goals, engine);

    // Build VSA fingerprint: bundle of role-bound goal progress vectors.
    let mut components = Vec::new();

    // Goal progress component.
    for (goal_id, progress) in &state.goal_states {
        let goal_vec = crate::vsa::encode::encode_symbol(ops, *goal_id);
        if let Ok(bound) = ops.bind(&roles.goal_progress, &goal_vec) {
            components.push(bound);
        }
    }

    // Fluent count component (quantized).
    let fluent_token = format!("state:fluent-count:{}", state.fluent_count / 100);
    let fluent_vec = encode_token(ops, &fluent_token);
    if let Ok(bound) = ops.bind(&roles.fluent_active, &fluent_vec) {
        components.push(bound);
    }

    if !components.is_empty() {
        let refs: Vec<&HyperVec> = components.iter().collect();
        if let Ok(bundled) = ops.bundle(&refs) {
            state.state_vec = Some(bundled);
        }
    }

    state
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(goals: Vec<(u64, f32)>, fluent_count: usize) -> AgentState {
        AgentState {
            goal_states: goals
                .into_iter()
                .filter_map(|(id, p)| SymbolId::new(id).map(|s| (s, p)))
                .collect(),
            fluent_count,
            state_vec: None,
            timestamp: 100,
        }
    }

    // ── state_hash ────────────────────────────────────────────────

    #[test]
    fn state_hash_deterministic() {
        let s1 = make_state(vec![(1, 0.5), (2, 0.8)], 1000);
        let s2 = make_state(vec![(1, 0.5), (2, 0.8)], 1000);
        assert_eq!(s1.state_hash(), s2.state_hash());
    }

    #[test]
    fn state_hash_differs_for_different_states() {
        let s1 = make_state(vec![(1, 0.5)], 1000);
        let s2 = make_state(vec![(1, 0.9)], 1000);
        // Different progress bins → different hash.
        assert_ne!(s1.state_hash(), s2.state_hash());
    }

    // ── Reward ────────────────────────────────────────────────────

    #[test]
    fn reward_goal_progress() {
        let before = make_state(vec![(1, 0.3)], 500);
        let after = make_state(vec![(1, 0.7)], 500);
        let reward = Reward::compute(&before, &after, false);
        assert!(reward.goal_progress > 0.0);
        assert!(reward.total > 0.0);
    }

    #[test]
    fn reward_knowledge_gain() {
        let before = make_state(vec![], 1000);
        let after = make_state(vec![], 1500);
        let reward = Reward::compute(&before, &after, false);
        assert!((reward.knowledge_gain - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn reward_prediction_accuracy_bonus() {
        let before = make_state(vec![], 100);
        let after = make_state(vec![], 100);
        let with = Reward::compute(&before, &after, true);
        let without = Reward::compute(&before, &after, false);
        assert!(with.total > without.total);
    }

    #[test]
    fn reward_combined_score() {
        let before = make_state(vec![(1, 0.0)], 100);
        let after = make_state(vec![(1, 0.5)], 200);
        let reward = Reward::compute(&before, &after, true);
        // goal_progress=0.5, knowledge_gain=1.0, prediction=0.1
        // total = 0.5*1.0 + 1.0*0.3 + 0.1 = 0.9
        assert!((reward.total - 0.9).abs() < 0.01);
    }

    // ── ValueFunction ─────────────────────────────────────────────

    #[test]
    fn value_function_default() {
        let vf = ValueFunction::default();
        assert_eq!(vf.alpha, 0.1);
        assert_eq!(vf.gamma, 0.99);
        assert_eq!(vf.update_count, 0);
        assert!(vf.state_values.is_empty());
    }

    #[test]
    fn state_value_unknown_returns_zero() {
        let vf = ValueFunction::default();
        let state = make_state(vec![(1, 0.5)], 100);
        assert_eq!(vf.state_value(&state), 0.0);
    }

    #[test]
    fn td_update_increases_value() {
        let mut vf = ValueFunction::with_params(0.5, 0.9);
        let s1 = make_state(vec![(1, 0.0)], 100);
        let s2 = make_state(vec![(1, 0.5)], 200);

        // With positive reward, V(s1) should increase.
        vf.td_update(&s1, 1.0, &s2);
        assert!(vf.state_value(&s1) > 0.0);
        assert_eq!(vf.update_count, 1);
    }

    #[test]
    fn td_update_converges() {
        let mut vf = ValueFunction::with_params(0.1, 0.9);
        let s1 = make_state(vec![(1, 0.0)], 100);
        let s2 = make_state(vec![(1, 1.0)], 100);

        // Repeatedly update with constant reward → value should grow.
        for _ in 0..100 {
            vf.td_update(&s1, 1.0, &s2);
        }

        let v = vf.state_value(&s1);
        // V(s1) grows from 0 toward a stable point. With α=0.1, γ=0.9,
        // and V(s2)=0 (never updated), converges near r·α/(1-γ·(1-α)) ≈ 1.0.
        assert!(v > 0.5, "value should have grown: got {v}");
        assert!(v < 2.0, "value should not overshoot: got {v}");
    }

    #[test]
    fn value_function_serialization() {
        let mut vf = ValueFunction::with_params(0.2, 0.95);
        let state = make_state(vec![(1, 0.5)], 100);
        vf.td_update(&state, 1.0, &state);

        let bytes = bincode::serialize(&vf).unwrap();
        let restored: ValueFunction = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.alpha, 0.2);
        assert_eq!(restored.gamma, 0.95);
        assert_eq!(restored.update_count, 1);
        assert_eq!(
            restored.state_value(&state),
            vf.state_value(&state)
        );
    }

    // ── StateRoleVectors ──────────────────────────────────────────

    #[test]
    fn state_role_vectors_distinct() {
        let ops = VsaOps::new(
            crate::simd::best_kernel(),
            crate::vsa::Dimension::TEST,
            crate::vsa::Encoding::Bipolar,
        );
        let roles = StateRoleVectors::new(&ops);
        let vecs = [
            &roles.goal_progress,
            &roles.fluent_active,
            &roles.resource_available,
            &roles.memory_pressure,
        ];
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let sim = ops.similarity(vecs[i], vecs[j]).unwrap_or(0.0);
                assert!(
                    sim < 0.7,
                    "role vectors {i} and {j} too similar: {sim}"
                );
            }
        }
    }
}
