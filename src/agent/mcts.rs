//! Monte Carlo Tree Search planner — Phase 16b.
//!
//! Multi-step look-ahead planning via MCTS with UCT selection. Uses the
//! causal world model (Phase 15a) as the transition function and the TD
//! value function (Phase 16a) for leaf evaluation.
//!
//! Algorithm:
//! ```text
//! for iteration in 0..max_iterations:
//!     node = select(root)              // UCT traversal
//!     if node.visits >= threshold && !terminal:
//!         child = expand(node)         // add child for unexplored action
//!         value = rollout(child)       // simulate forward
//!     else:
//!         value = value_fn(node.state) // learned estimate
//!     backpropagate(path, value)       // update Q/N along path
//! return best_action_sequence(root)
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::causal::CausalManager;
use super::state_value::{AgentState, ValueFunction, encode_state};

// ═══════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for MCTS planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MctsConfig {
    /// Maximum iterations per planning decision.
    pub max_iterations: usize,
    /// Maximum tree depth (action sequence length).
    pub max_depth: usize,
    /// UCT exploration constant (typically √2 ≈ 1.414).
    pub exploration_constant: f32,
    /// Maximum rollout depth for simulation.
    pub rollout_depth: usize,
    /// Minimum visits before expanding a node.
    pub expansion_threshold: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_depth: 5,
            exploration_constant: 1.414,
            rollout_depth: 3,
            expansion_threshold: 2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MctsNode
// ═══════════════════════════════════════════════════════════════════════

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// The agent state at this node.
    pub state: AgentState,
    /// The action that led to this state (None for root).
    pub action: Option<SymbolId>,
    /// Visit count N(s).
    pub visits: u32,
    /// Cumulative value Q(s).
    pub total_value: f32,
    /// Children nodes.
    pub children: Vec<MctsNode>,
    /// Whether this is a terminal state (goal reached or no applicable actions).
    pub terminal: bool,
    /// Depth in the tree.
    pub depth: usize,
}

impl MctsNode {
    /// Create a root node from an initial state.
    pub fn root(state: AgentState) -> Self {
        Self {
            state,
            action: None,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            terminal: false,
            depth: 0,
        }
    }

    /// Average value: Q(s)/N(s).
    pub fn avg_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }

    /// UCT score: Q/N + c * sqrt(ln(N_parent) / N).
    pub fn uct_score(&self, parent_visits: u32, exploration_constant: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY; // always explore unvisited nodes first
        }
        let exploitation = self.avg_value();
        let exploration = exploration_constant
            * ((parent_visits as f32).ln() / self.visits as f32).sqrt();
        exploitation + exploration
    }

    /// Total nodes in this subtree (including self).
    pub fn tree_size(&self) -> usize {
        1 + self.children.iter().map(|c| c.tree_size()).sum::<usize>()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MctsResult
// ═══════════════════════════════════════════════════════════════════════

/// Result of an MCTS planning session.
#[derive(Debug, Clone)]
pub struct MctsResult {
    /// Best action sequence found (by visit count).
    pub best_plan: Vec<SymbolId>,
    /// Expected value of the best plan.
    pub expected_value: f32,
    /// Total iterations performed.
    pub iterations: usize,
    /// Maximum depth reached in the tree.
    pub max_depth_reached: usize,
    /// First-action candidates with (action, avg_value, visits).
    pub first_action_scores: Vec<(SymbolId, f32, u32)>,
    /// Whether the plan reaches a goal state in simulation.
    pub reaches_goal: bool,
}

// ═══════════════════════════════════════════════════════════════════════
// MctsReflection (R-MCTS)
// ═══════════════════════════════════════════════════════════════════════

/// Reflection data for warm-starting MCTS from past experience.
#[derive(Debug, Clone, Default)]
pub struct MctsReflection {
    /// Past successful action sequences for similar states.
    pub successful_sequences: Vec<Vec<SymbolId>>,
    /// Past failed action sequences for similar states.
    pub failed_sequences: Vec<Vec<SymbolId>>,
    /// Warm-start bias: action_id → prior value bonus.
    pub prior_bias: HashMap<u64, f32>,
}

// ═══════════════════════════════════════════════════════════════════════
// MctsPlanner
// ═══════════════════════════════════════════════════════════════════════

/// Monte Carlo Tree Search planner.
pub struct MctsPlanner {
    pub config: MctsConfig,
}

impl MctsPlanner {
    pub fn new(config: MctsConfig) -> Self {
        Self { config }
    }

    /// Run MCTS planning from the given root state.
    ///
    /// Returns the best action sequence found within the iteration budget.
    pub fn plan(
        &self,
        root_state: AgentState,
        causal_manager: &CausalManager,
        value_fn: &ValueFunction,
        engine: &Engine,
        reflection: Option<&MctsReflection>,
    ) -> MctsResult {
        let mut root = MctsNode::root(root_state);
        let mut max_depth_reached = 0;

        // Get applicable actions once for expansion.
        let applicable: Vec<SymbolId> = causal_manager
            .applicable_actions(engine)
            .iter()
            .map(|s| s.action_id)
            .collect();

        if applicable.is_empty() {
            root.terminal = true;
            return MctsResult {
                best_plan: Vec::new(),
                expected_value: 0.0,
                iterations: 0,
                max_depth_reached: 0,
                first_action_scores: Vec::new(),
                reaches_goal: false,
            };
        }

        // Apply reflection warm-start bias.
        if let Some(refl) = reflection {
            self.apply_reflection(&mut root, refl, &applicable);
        }

        for _iter in 0..self.config.max_iterations {
            // Selection: traverse tree using UCT.
            let path = self.select_path(&root);

            // Get the leaf node at the end of the path.
            let leaf_depth = path.len();
            max_depth_reached = max_depth_reached.max(leaf_depth);

            let leaf = self.get_node_mut(&mut root, &path);

            // Expansion: if visited enough and not terminal, expand.
            if leaf.visits >= self.config.expansion_threshold
                && !leaf.terminal
                && leaf.depth < self.config.max_depth
            {
                // Find unexplored actions.
                let explored: Vec<SymbolId> = leaf.children.iter()
                    .filter_map(|c| c.action)
                    .collect();
                let unexplored: Vec<SymbolId> = applicable.iter()
                    .filter(|a| !explored.contains(a))
                    .copied()
                    .collect();

                if let Some(&action) = unexplored.first() {
                    // Create child node with predicted state.
                    let child_state = self.predict_next_state(
                        &leaf.state, action, causal_manager, engine,
                    );
                    let child_depth = leaf.depth + 1;
                    let terminal = child_depth >= self.config.max_depth
                        || self.is_goal_reached(&child_state);

                    leaf.children.push(MctsNode {
                        state: child_state.clone(),
                        action: Some(action),
                        visits: 0,
                        total_value: 0.0,
                        children: Vec::new(),
                        terminal,
                        depth: child_depth,
                    });

                    // Rollout from the new child.
                    let value = self.rollout(
                        &child_state,
                        child_depth,
                        causal_manager,
                        value_fn,
                        engine,
                    );

                    // Backpropagate: update the new child and all ancestors.
                    let child_idx = leaf.children.len() - 1;
                    leaf.children[child_idx].visits += 1;
                    leaf.children[child_idx].total_value += value;

                    // Backpropagate up the path.
                    self.backpropagate(&mut root, &path, value);
                    continue;
                }
            }

            // No expansion possible: evaluate leaf and backpropagate.
            let value = value_fn.state_value(&leaf.state);
            self.backpropagate(&mut root, &path, value);
        }

        // Extract best plan.
        self.extract_result(&root, max_depth_reached)
    }

    // ─── Selection ────────────────────────────────────────────────

    /// Select a path from root to a leaf using UCT.
    fn select_path(&self, root: &MctsNode) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;

        while !current.children.is_empty() && !current.terminal {
            let parent_visits = current.visits.max(1);
            let best_idx = current
                .children
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let sa = a.uct_score(parent_visits, self.config.exploration_constant);
                    let sb = b.uct_score(parent_visits, self.config.exploration_constant);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            path.push(best_idx);
            current = &current.children[best_idx];
        }

        path
    }

    /// Navigate to a node given a path of child indices.
    fn get_node_mut<'a>(&self, root: &'a mut MctsNode, path: &[usize]) -> &'a mut MctsNode {
        let mut current = root;
        for &idx in path {
            current = &mut current.children[idx];
        }
        current
    }

    // ─── Rollout ──────────────────────────────────────────────────

    /// Simulate forward from a state using the causal model.
    ///
    /// Returns the estimated value of the resulting state.
    fn rollout(
        &self,
        state: &AgentState,
        current_depth: usize,
        causal_manager: &CausalManager,
        value_fn: &ValueFunction,
        engine: &Engine,
    ) -> f32 {
        let mut current_state = state.clone();
        let remaining = self.config.rollout_depth.min(
            self.config.max_depth.saturating_sub(current_depth),
        );

        let applicable: Vec<SymbolId> = causal_manager
            .applicable_actions(engine)
            .iter()
            .map(|s| s.action_id)
            .collect();

        for _ in 0..remaining {
            if applicable.is_empty() {
                break;
            }

            // Simple rollout policy: pick first applicable action.
            // (A more sophisticated policy would use utility scoring.)
            let action = applicable[0];
            current_state = self.predict_next_state(
                &current_state, action, causal_manager, engine,
            );

            if self.is_goal_reached(&current_state) {
                return 1.0; // goal reached — maximum value
            }
        }

        // Evaluate the terminal rollout state.
        value_fn.state_value(&current_state)
    }

    // ─── State Prediction ─────────────────────────────────────────

    /// Predict the next state after applying an action.
    fn predict_next_state(
        &self,
        state: &AgentState,
        action: SymbolId,
        causal_manager: &CausalManager,
        engine: &Engine,
    ) -> AgentState {
        let action_name = engine.resolve_label(action);
        let transition = causal_manager.predict_effects(&action_name, engine);

        let mut next = state.clone();
        next.timestamp += 1; // advance time

        if let Ok(t) = transition {
            // Adjust fluent count based on assertions/retractions.
            next.fluent_count = next.fluent_count
                .saturating_add(t.assertions.len())
                .saturating_sub(t.retractions.len());
        }

        next
    }

    /// Simple goal-reached heuristic: all goals have progress > 0.8.
    fn is_goal_reached(&self, state: &AgentState) -> bool {
        !state.goal_states.is_empty()
            && state.goal_states.iter().all(|(_, p)| *p > 0.8)
    }

    // ─── Backpropagation ──────────────────────────────────────────

    /// Backpropagate a value along the selection path.
    fn backpropagate(&self, root: &mut MctsNode, path: &[usize], value: f32) {
        root.visits += 1;
        root.total_value += value;

        let mut current = root;
        for &idx in path {
            current = &mut current.children[idx];
            current.visits += 1;
            current.total_value += value;
        }
    }

    // ─── Reflection (R-MCTS) ─────────────────────────────────────

    /// Apply warm-start bias from reflection data.
    fn apply_reflection(
        &self,
        root: &mut MctsNode,
        reflection: &MctsReflection,
        applicable: &[SymbolId],
    ) {
        // Pre-create children for actions that appear in successful sequences.
        for seq in &reflection.successful_sequences {
            if let Some(&first_action) = seq.first() {
                if applicable.contains(&first_action) {
                    let already_exists = root.children.iter()
                        .any(|c| c.action == Some(first_action));
                    if !already_exists {
                        let mut child = MctsNode {
                            state: root.state.clone(),
                            action: Some(first_action),
                            visits: 1, // virtual visit
                            total_value: 0.5, // positive prior
                            children: Vec::new(),
                            terminal: false,
                            depth: 1,
                        };
                        // Apply bias if available.
                        if let Some(&bias) = reflection.prior_bias.get(&first_action.get()) {
                            child.total_value = bias;
                        }
                        root.children.push(child);
                    }
                }
            }
        }
    }

    // ─── Result Extraction ────────────────────────────────────────

    /// Extract the best plan from the search tree.
    fn extract_result(&self, root: &MctsNode, max_depth_reached: usize) -> MctsResult {
        // First-action scores.
        let first_action_scores: Vec<(SymbolId, f32, u32)> = root
            .children
            .iter()
            .filter_map(|c| {
                c.action.map(|a| (a, c.avg_value(), c.visits))
            })
            .collect();

        // Best plan: follow most-visited children.
        let mut best_plan = Vec::new();
        let mut current = root;
        let mut reaches_goal = false;

        while !current.children.is_empty() {
            let best = current
                .children
                .iter()
                .max_by_key(|c| c.visits);

            match best {
                Some(node) if node.visits > 0 => {
                    if let Some(action) = node.action {
                        best_plan.push(action);
                    }
                    if self.is_goal_reached(&node.state) {
                        reaches_goal = true;
                        break;
                    }
                    current = node;
                }
                _ => break,
            }
        }

        let expected_value = if let Some(first) = root.children.iter()
            .max_by_key(|c| c.visits)
        {
            first.avg_value()
        } else {
            0.0
        };

        MctsResult {
            best_plan,
            expected_value,
            iterations: self.config.max_iterations,
            max_depth_reached,
            first_action_scores,
            reaches_goal,
        }
    }
}

/// Prune an MCTS tree to limit memory usage.
///
/// Keeps only the top `max_children` children by visit count at each level.
pub fn prune_tree(node: &mut MctsNode, max_children: usize) {
    if node.children.len() > max_children {
        node.children.sort_by(|a, b| b.visits.cmp(&a.visits));
        node.children.truncate(max_children);
    }
    for child in &mut node.children {
        prune_tree(child, max_children);
    }
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

    // ── MctsConfig ────────────────────────────────────────────────

    #[test]
    fn mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.max_depth, 5);
        assert!((config.exploration_constant - 1.414).abs() < 0.001);
    }

    // ── MctsNode ──────────────────────────────────────────────────

    #[test]
    fn mcts_node_root() {
        let state = make_state(vec![(1, 0.5)], 100);
        let root = MctsNode::root(state);
        assert_eq!(root.visits, 0);
        assert_eq!(root.depth, 0);
        assert!(root.action.is_none());
    }

    #[test]
    fn mcts_node_avg_value() {
        let mut node = MctsNode::root(make_state(vec![], 0));
        assert_eq!(node.avg_value(), 0.0);

        node.visits = 4;
        node.total_value = 2.0;
        assert!((node.avg_value() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mcts_node_uct_unvisited_is_infinity() {
        let node = MctsNode::root(make_state(vec![], 0));
        let score = node.uct_score(10, 1.414);
        assert!(score.is_infinite());
    }

    #[test]
    fn mcts_node_uct_formula() {
        let mut node = MctsNode::root(make_state(vec![], 0));
        node.visits = 10;
        node.total_value = 5.0; // avg = 0.5

        let parent_visits = 100;
        let c = 1.414;
        let expected_exploitation = 0.5;
        let expected_exploration = c * ((100_f32).ln() / 10.0).sqrt();
        let expected = expected_exploitation + expected_exploration;

        let actual = node.uct_score(parent_visits, c);
        assert!((actual - expected).abs() < 0.01, "UCT: got {actual}, expected {expected}");
    }

    #[test]
    fn mcts_node_tree_size() {
        let mut root = MctsNode::root(make_state(vec![], 0));
        root.children.push(MctsNode::root(make_state(vec![], 0)));
        root.children.push(MctsNode::root(make_state(vec![], 0)));
        root.children[0].children.push(MctsNode::root(make_state(vec![], 0)));
        assert_eq!(root.tree_size(), 4);
    }

    // ── MctsResult ────────────────────────────────────────────────

    #[test]
    fn mcts_result_empty() {
        let result = MctsResult {
            best_plan: vec![],
            expected_value: 0.0,
            iterations: 0,
            max_depth_reached: 0,
            first_action_scores: vec![],
            reaches_goal: false,
        };
        assert!(result.best_plan.is_empty());
        assert!(!result.reaches_goal);
    }

    // ── Pruning ───────────────────────────────────────────────────

    #[test]
    fn prune_tree_limits_children() {
        let mut root = MctsNode::root(make_state(vec![], 0));
        for i in 0..10 {
            let mut child = MctsNode::root(make_state(vec![], 0));
            child.visits = i;
            root.children.push(child);
        }

        prune_tree(&mut root, 3);
        assert_eq!(root.children.len(), 3);
        // Kept the 3 most-visited.
        assert!(root.children.iter().all(|c| c.visits >= 7));
    }

    // ── Reflection ────────────────────────────────────────────────

    #[test]
    fn reflection_warm_start() {
        let config = MctsConfig::default();
        let planner = MctsPlanner::new(config);
        let mut root = MctsNode::root(make_state(vec![], 0));

        let action = SymbolId::new(42).unwrap();
        let reflection = MctsReflection {
            successful_sequences: vec![vec![action]],
            failed_sequences: vec![],
            prior_bias: HashMap::new(),
        };

        planner.apply_reflection(&mut root, &reflection, &[action]);
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].action, Some(action));
        assert_eq!(root.children[0].visits, 1); // virtual visit
    }

    // ── MctsConfig serialization ──────────────────────────────────

    #[test]
    fn mcts_config_serialization() {
        let config = MctsConfig::default();
        let bytes = bincode::serialize(&config).unwrap();
        let restored: MctsConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.max_iterations, 100);
        assert_eq!(restored.max_depth, 5);
    }

    // ── Goal detection ────────────────────────────────────────────

    #[test]
    fn goal_reached_detection() {
        let planner = MctsPlanner::new(MctsConfig::default());

        let not_reached = make_state(vec![(1, 0.5)], 100);
        assert!(!planner.is_goal_reached(&not_reached));

        let reached = make_state(vec![(1, 0.9)], 100);
        assert!(planner.is_goal_reached(&reached));

        let empty = make_state(vec![], 100);
        assert!(!planner.is_goal_reached(&empty)); // no goals = not reached
    }
}
