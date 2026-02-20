//! HTN-style intelligent task decomposition.
//!
//! Replaces the simple comma/"and" splitting in `goal::generate_sub_goal_descriptions()`
//! with a method registry, VSA-based method selection, dependency DAGs,
//! and resource-rational scoring.
//!
//! # Architecture
//!
//! - **DecompositionMethod**: a reusable decomposition template with keyword hints,
//!   subtask templates, ordering constraints, and optional VSA semantic vector.
//! - **MethodRegistry**: stores methods and selects the best match for a goal
//!   via hybrid keyword + VSA scoring.
//! - **TaskTree**: a petgraph DAG of `TaskNode`s connected by `DependencyEdge`s.
//! - **decompose_goal_htn**: orchestrator that selects a method, instantiates it,
//!   builds the DAG, and returns a `DecompositionResult`.

use miette::Diagnostic;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;

use super::goal::{self, Goal};
use super::plan::PlanStep;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors during decomposition.
#[derive(Debug, Error, Diagnostic)]
pub enum DecompositionError {
    #[error("no applicable decomposition method for goal \"{goal_description}\"")]
    #[diagnostic(
        code(akh::agent::decomposition::no_method),
        help("The goal did not match any HTN method (score < 0.2). Falling back to comma-split decomposition.")
    )]
    NoApplicableMethod { goal_description: String },

    #[error("dependency cycle detected in task tree for method \"{method_name}\"")]
    #[diagnostic(
        code(akh::agent::decomposition::cycle),
        help("The ordering constraints in the decomposition method form a cycle. Check the method's ordering pairs.")
    )]
    CycleDetected { method_name: String },

    #[error("decomposition failed: {message}")]
    #[diagnostic(
        code(akh::agent::decomposition::failed),
        help("An internal error occurred during decomposition. Check the inner cause.")
    )]
    Failed { message: String },
}

/// Convenience alias.
pub type DecompositionResult<T> = std::result::Result<T, DecompositionError>;

// ---------------------------------------------------------------------------
// Strategy & Method
// ---------------------------------------------------------------------------

/// High-level decomposition strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// hypothesis → gather evidence → evaluate → conclude
    Research,
    /// define scope → build components → assemble → validate
    Construction,
    /// observe anomaly → hypothesize → test → resolve
    Investigation,
    /// diagnose → plan fix → apply → verify
    Maintenance,
    /// survey → select focus → deep-dive → synthesize
    Exploration,
    /// gather A + gather B → analyze commonalities → differences → conclude
    CompareContrast,
    /// Custom / learned strategy.
    Custom(String),
}

impl std::fmt::Display for DecompositionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Research => write!(f, "research"),
            Self::Construction => write!(f, "construction"),
            Self::Investigation => write!(f, "investigation"),
            Self::Maintenance => write!(f, "maintenance"),
            Self::Exploration => write!(f, "exploration"),
            Self::CompareContrast => write!(f, "compare-contrast"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

/// A template for one subtask within a decomposition method.
#[derive(Debug, Clone)]
pub struct SubtaskTemplate {
    /// Description template. May contain `{goal}` placeholder.
    pub description_template: String,
    /// Priority offset from parent goal (-128..+127).
    pub priority_offset: i8,
    /// Criteria template. May contain `{goal}` placeholder.
    pub criteria_template: String,
}

/// A reusable decomposition method.
#[derive(Debug, Clone)]
pub struct DecompositionMethod {
    /// Human-readable name.
    pub name: String,
    /// Which strategy this implements.
    pub strategy: DecompositionStrategy,
    /// SPARQL precondition (empty = always applicable).
    pub precondition_sparql: String,
    /// Quick keyword pre-filter for scoring.
    pub keyword_hints: Vec<String>,
    /// Ordered subtask templates.
    pub subtask_templates: Vec<SubtaskTemplate>,
    /// Ordering constraints: `(earlier_idx, later_idx)` pairs.
    pub ordering: Vec<(usize, usize)>,
    /// Lazy VSA encoding of the method's semantic profile.
    pub semantic_vector: Option<HyperVec>,
    /// How many times this method has been used.
    pub usage_count: u32,
    /// Empirical success rate [0.0, 1.0].
    pub success_rate: f32,
}

// ---------------------------------------------------------------------------
// TaskTree with Dependency DAG
// ---------------------------------------------------------------------------

/// What kind of node a task tree contains.
#[derive(Debug, Clone)]
pub enum TaskNodeKind {
    /// A sub-goal that will be recursively decomposed.
    SubGoal {
        description: String,
        priority: u8,
        success_criteria: String,
    },
    /// A concrete plan step that executes directly.
    PlanStep(PlanStep),
}

/// A single node in the task tree.
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// What this node represents.
    pub kind: TaskNodeKind,
    /// Estimated OODA cycles to complete.
    pub estimated_cycles: u32,
    /// Why this node exists.
    pub rationale: String,
}

/// Dependency edge types in the task tree DAG.
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyEdge {
    /// B waits for A to finish before starting.
    FinishToStart,
    /// B waits for A to start before starting.
    StartToStart,
}

/// A dependency DAG of task nodes produced by decomposition.
pub struct TaskTree {
    /// The petgraph DAG.
    pub dag: DiGraph<TaskNode, DependencyEdge>,
    /// Which strategy produced this tree.
    pub strategy: DecompositionStrategy,
    /// Which method was used.
    pub method_name: String,
    /// Root nodes (nodes with no incoming edges).
    pub roots: Vec<NodeIndex>,
}

impl std::fmt::Debug for TaskTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskTree")
            .field("strategy", &self.strategy)
            .field("method_name", &self.method_name)
            .field("node_count", &self.dag.node_count())
            .field("edge_count", &self.dag.edge_count())
            .field("roots", &self.roots.len())
            .finish()
    }
}

impl TaskTree {
    /// Produce nodes in topological order via Kahn's algorithm (petgraph).
    ///
    /// Returns `Err` if the DAG contains a cycle.
    pub fn topological_order(&self) -> DecompositionResult<Vec<NodeIndex>> {
        petgraph::algo::toposort(&self.dag, None).map_err(|_| DecompositionError::CycleDetected {
            method_name: self.method_name.clone(),
        })
    }

    /// Compute the critical path — longest path through the DAG by `estimated_cycles`.
    ///
    /// Returns the total estimated cycles of the critical path, and the path's node indices.
    pub fn critical_path(&self) -> DecompositionResult<(u32, Vec<NodeIndex>)> {
        let topo = self.topological_order()?;
        if topo.is_empty() {
            return Ok((0, Vec::new()));
        }

        // Dynamic programming: longest path from each node.
        let mut dist: std::collections::HashMap<NodeIndex, u32> = std::collections::HashMap::new();
        let mut pred: std::collections::HashMap<NodeIndex, Option<NodeIndex>> =
            std::collections::HashMap::new();

        for &node in &topo {
            let node_cost = self.dag[node].estimated_cycles;
            dist.insert(node, node_cost);
            pred.insert(node, None);
        }

        for &node in &topo {
            let current_dist = dist[&node];
            for neighbor in self.dag.neighbors(node) {
                let neighbor_cost = self.dag[neighbor].estimated_cycles;
                let new_dist = current_dist + neighbor_cost;
                if new_dist > dist[&neighbor] {
                    dist.insert(neighbor, new_dist);
                    pred.insert(neighbor, Some(node));
                }
            }
        }

        // Find the node with the maximum distance.
        let (&end_node, &max_dist) = dist.iter().max_by_key(|&(_, &d)| d).unwrap();

        // Reconstruct path.
        let mut path = vec![end_node];
        let mut current = end_node;
        while let Some(Some(prev)) = pred.get(&current) {
            path.push(*prev);
            current = *prev;
        }
        path.reverse();

        Ok((max_dist, path))
    }

    /// Extract sub-goal nodes in topological order.
    pub fn to_sub_goals(&self) -> DecompositionResult<Vec<(String, u8, String)>> {
        let topo = self.topological_order()?;
        let mut sub_goals = Vec::new();

        for idx in topo {
            if let TaskNodeKind::SubGoal {
                ref description,
                priority,
                ref success_criteria,
            } = self.dag[idx].kind
            {
                sub_goals.push((description.clone(), priority, success_criteria.clone()));
            }
        }

        Ok(sub_goals)
    }
}

// ---------------------------------------------------------------------------
// Decomposition output
// ---------------------------------------------------------------------------

/// Result of a goal decomposition.
pub struct DecompositionOutput {
    /// The full task tree DAG.
    pub tree: TaskTree,
    /// Extracted sub-goals in topological order: (description, priority, criteria).
    pub sub_goals: Vec<(String, u8, String)>,
    /// Dependency pairs: (blocker_idx, blocked_idx) into the sub_goals vec.
    pub dependencies: Vec<(usize, usize)>,
    /// Human-readable rationale for the decomposition.
    pub rationale: String,
}

// ---------------------------------------------------------------------------
// MethodRegistry
// ---------------------------------------------------------------------------

/// Registry of decomposition methods with hybrid keyword + VSA selection.
pub struct MethodRegistry {
    methods: Vec<DecompositionMethod>,
}

impl MethodRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
        }
    }

    /// Register a decomposition method.
    pub fn register(&mut self, method: DecompositionMethod) {
        self.methods.push(method);
    }

    /// How many methods are registered.
    pub fn len(&self) -> usize {
        self.methods.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.methods.is_empty()
    }

    /// Select the best method for a goal using hybrid keyword + VSA scoring.
    ///
    /// Returns `None` if no method scores above the 0.2 threshold.
    pub fn select_method(&self, goal: &Goal, engine: &Engine) -> Option<&DecompositionMethod> {
        if self.methods.is_empty() {
            return None;
        }

        let goal_lower = goal.description.to_lowercase();
        let goal_words: Vec<&str> = goal_lower.split_whitespace().collect();

        // Encode goal as a VSA vector for semantic matching.
        let ops = engine.ops();
        let im = engine.item_memory();
        let goal_vec = crate::vsa::grounding::encode_text_as_vector(
            &goal.description,
            engine,
            ops,
            im,
        )
        .ok();

        let mut best_score = 0.0f32;
        let mut best_method: Option<&DecompositionMethod> = None;

        for method in &self.methods {
            // 1. Keyword score: fraction of method keywords found in goal.
            let keyword_score = if method.keyword_hints.is_empty() {
                0.0
            } else {
                let matched = method
                    .keyword_hints
                    .iter()
                    .filter(|kw| goal_words.iter().any(|w| w.contains(kw.as_str())))
                    .count();
                matched as f32 / method.keyword_hints.len() as f32
            };

            // 2. Semantic score: VSA similarity between goal and method vectors.
            let semantic_score = match (&goal_vec, &method.semantic_vector) {
                (Some(gv), Some(mv)) => ops.similarity(gv, mv).unwrap_or(0.5),
                _ => {
                    // Fallback: encode method keywords as vector on the fly.
                    goal_vec
                        .as_ref()
                        .and_then(|gv| {
                            let keywords: Vec<&str> =
                                method.keyword_hints.iter().map(|s| s.as_str()).collect();
                            crate::vsa::grounding::bundle_symbols(engine, ops, im, &keywords)
                                .ok()
                                .and_then(|mv| ops.similarity(gv, &mv).ok())
                        })
                        .unwrap_or(0.5)
                }
            };

            // 3. Success bonus from historical success rate.
            let success_bonus = method.success_rate * 0.1;

            // 4. Complexity penalty: prefer simpler decompositions.
            let complexity_penalty = method.subtask_templates.len() as f32 * 0.05;

            // 5. Total score.
            let total =
                keyword_score * 0.4 + semantic_score * 0.4 + success_bonus - complexity_penalty;

            if total > best_score {
                best_score = total;
                best_method = Some(method);
            }
        }

        // Only return a method if it scores above the threshold.
        if best_score >= 0.2 {
            best_method
        } else {
            None
        }
    }

    /// Get the methods (for serialization / stats).
    pub fn methods(&self) -> &[DecompositionMethod] {
        &self.methods
    }

    /// Get mutable methods (for updating stats).
    pub fn methods_mut(&mut self) -> &mut [DecompositionMethod] {
        &mut self.methods
    }
}

impl Default for MethodRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in methods
// ---------------------------------------------------------------------------

/// Register the 6 built-in decomposition methods.
pub fn register_builtin_methods(registry: &mut MethodRegistry) {
    // 1. Research
    registry.register(DecompositionMethod {
        name: "research".into(),
        strategy: DecompositionStrategy::Research,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "learn".into(),
            "understand".into(),
            "study".into(),
            "discover".into(),
            "research".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Survey existing knowledge about {goal}".into(),
                priority_offset: 0,
                criteria_template: "Gathered overview of {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Identify knowledge gaps in {goal}".into(),
                priority_offset: -5,
                criteria_template: "Gaps documented for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Gather evidence to fill gaps about {goal}".into(),
                priority_offset: -10,
                criteria_template: "Evidence collected for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Synthesize findings about {goal}".into(),
                priority_offset: -15,
                criteria_template: "Synthesis complete for {goal}".into(),
            },
        ],
        ordering: vec![(0, 1), (1, 2), (2, 3)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });

    // 2. Construction
    registry.register(DecompositionMethod {
        name: "construction".into(),
        strategy: DecompositionStrategy::Construction,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "build".into(),
            "create".into(),
            "implement".into(),
            "generate".into(),
            "write".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Define scope and requirements for {goal}".into(),
                priority_offset: 0,
                criteria_template: "Scope defined for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Gather dependencies for {goal}".into(),
                priority_offset: -5,
                criteria_template: "Dependencies identified for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Build components for {goal}".into(),
                priority_offset: -10,
                criteria_template: "Components built for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Validate and test {goal}".into(),
                priority_offset: -15,
                criteria_template: "Validation complete for {goal}".into(),
            },
        ],
        ordering: vec![(0, 1), (1, 2), (2, 3)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });

    // 3. Investigation
    registry.register(DecompositionMethod {
        name: "investigation".into(),
        strategy: DecompositionStrategy::Investigation,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "investigate".into(),
            "debug".into(),
            "diagnose".into(),
            "resolve".into(),
            "fix".into(),
            "why".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Observe and document anomaly in {goal}".into(),
                priority_offset: 0,
                criteria_template: "Anomaly documented for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Hypothesize causes for {goal}".into(),
                priority_offset: -5,
                criteria_template: "Hypotheses formulated for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Test hypotheses about {goal}".into(),
                priority_offset: -10,
                criteria_template: "Hypotheses tested for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Apply fix for {goal}".into(),
                priority_offset: -15,
                criteria_template: "Fix applied for {goal}".into(),
            },
        ],
        ordering: vec![(0, 1), (1, 2), (2, 3)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });

    // 4. Exploration (fan-out: survey → select → deep-dive A + deep-dive B)
    registry.register(DecompositionMethod {
        name: "exploration".into(),
        strategy: DecompositionStrategy::Exploration,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "explore".into(),
            "survey".into(),
            "map".into(),
            "catalog".into(),
            "inventory".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Broad survey of {goal}".into(),
                priority_offset: 0,
                criteria_template: "Overview of {goal} established".into(),
            },
            SubtaskTemplate {
                description_template: "Select focus areas within {goal}".into(),
                priority_offset: -5,
                criteria_template: "Focus areas selected for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Deep-dive into primary aspect of {goal}".into(),
                priority_offset: -10,
                criteria_template: "Primary aspect explored for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Deep-dive into secondary aspect of {goal}".into(),
                priority_offset: -10,
                criteria_template: "Secondary aspect explored for {goal}".into(),
            },
        ],
        // Fan-out: 0→1, 1→2, 1→3 (2 and 3 are parallel).
        ordering: vec![(0, 1), (1, 2), (1, 3)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });

    // 5. CompareContrast (diamond: gather A + gather B → analyze → differences → conclude)
    registry.register(DecompositionMethod {
        name: "compare_contrast".into(),
        strategy: DecompositionStrategy::CompareContrast,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "compare".into(),
            "contrast".into(),
            "versus".into(),
            "vs".into(),
            "difference".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Gather information about first aspect of {goal}".into(),
                priority_offset: 0,
                criteria_template: "First aspect documented for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Gather information about second aspect of {goal}".into(),
                priority_offset: 0,
                criteria_template: "Second aspect documented for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Analyze commonalities in {goal}".into(),
                priority_offset: -5,
                criteria_template: "Commonalities identified for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Identify differences in {goal}".into(),
                priority_offset: -10,
                criteria_template: "Differences documented for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Conclude comparison of {goal}".into(),
                priority_offset: -15,
                criteria_template: "Comparison concluded for {goal}".into(),
            },
        ],
        // Diamond: 0→2, 1→2, 2→3, 3→4.
        ordering: vec![(0, 2), (1, 2), (2, 3), (3, 4)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });

    // 6. CodeImpl
    registry.register(DecompositionMethod {
        name: "code_impl".into(),
        strategy: DecompositionStrategy::Construction,
        precondition_sparql: String::new(),
        keyword_hints: vec![
            "code".into(),
            "function".into(),
            "struct".into(),
            "module".into(),
            "implement".into(),
            "generate".into(),
        ],
        subtask_templates: vec![
            SubtaskTemplate {
                description_template: "Query existing code structure for {goal}".into(),
                priority_offset: 0,
                criteria_template: "Code structure understood for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Define target entity in KG for {goal}".into(),
                priority_offset: -5,
                criteria_template: "Target defined in KG for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Generate code for {goal}".into(),
                priority_offset: -10,
                criteria_template: "Code generated for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Validate generated code compiles for {goal}".into(),
                priority_offset: -15,
                criteria_template: "Code compiles for {goal}".into(),
            },
            SubtaskTemplate {
                description_template: "Fix compilation errors for {goal}".into(),
                priority_offset: -20,
                criteria_template: "All errors resolved for {goal}".into(),
            },
        ],
        ordering: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
        semantic_vector: None,
        usage_count: 0,
        success_rate: 0.5,
    });
}

// ---------------------------------------------------------------------------
// Decomposition algorithm
// ---------------------------------------------------------------------------

/// Instantiate a method's templates for a specific goal, building a TaskTree DAG.
pub fn decompose_with_method(
    goal: &Goal,
    method: &DecompositionMethod,
) -> DecompositionResult<TaskTree> {
    let goal_short = goal
        .description
        .chars()
        .take(40)
        .collect::<String>();

    let mut dag: DiGraph<TaskNode, DependencyEdge> = DiGraph::new();
    let mut node_indices: Vec<NodeIndex> = Vec::new();

    // Create a node for each subtask template.
    for (i, template) in method.subtask_templates.iter().enumerate() {
        let description = template
            .description_template
            .replace("{goal}", &goal_short);
        let criteria = template.criteria_template.replace("{goal}", &goal_short);

        let priority = (goal.priority as i16 + template.priority_offset as i16)
            .clamp(0, 255) as u8;

        let node = TaskNode {
            kind: TaskNodeKind::SubGoal {
                description,
                priority,
                success_criteria: criteria,
            },
            estimated_cycles: 3, // Default estimate.
            rationale: format!("Step {} of {} method", i + 1, method.name),
        };

        node_indices.push(dag.add_node(node));
    }

    // Add ordering edges.
    for &(earlier, later) in &method.ordering {
        if earlier < node_indices.len() && later < node_indices.len() {
            dag.add_edge(
                node_indices[earlier],
                node_indices[later],
                DependencyEdge::FinishToStart,
            );
        }
    }

    // Validate: no cycles.
    if petgraph::algo::toposort(&dag, None).is_err() {
        return Err(DecompositionError::CycleDetected {
            method_name: method.name.clone(),
        });
    }

    // Identify roots (no incoming edges).
    let roots: Vec<NodeIndex> = node_indices
        .iter()
        .copied()
        .filter(|&idx| {
            dag.neighbors_directed(idx, petgraph::Direction::Incoming)
                .next()
                .is_none()
        })
        .collect();

    Ok(TaskTree {
        dag,
        strategy: method.strategy.clone(),
        method_name: method.name.clone(),
        roots,
    })
}

/// Orchestrator: select a method, decompose, return full result.
///
/// Falls back to `generate_sub_goal_descriptions()` if no HTN method matches.
pub fn decompose_goal_htn(
    goal: &Goal,
    engine: &Engine,
    registry: &MethodRegistry,
) -> DecompositionOutput {
    // Try HTN method selection.
    if let Some(method) = registry.select_method(goal, engine) {
        match decompose_with_method(goal, method) {
            Ok(tree) => {
                // Extract sub-goals and dependencies.
                let sub_goals = tree.to_sub_goals().unwrap_or_default();

                // Build dependency map: for each sub-goal pair that has an edge,
                // record (blocker_idx, blocked_idx) based on sub_goals vec indices.
                let topo = tree.topological_order().unwrap_or_default();
                let dependencies = extract_dependencies(&tree, &topo);

                let rationale = format!(
                    "HTN decomposition via '{}' method (strategy: {}), {} sub-goals, {} dependencies",
                    method.name,
                    method.strategy,
                    sub_goals.len(),
                    dependencies.len(),
                );

                return DecompositionOutput {
                    tree,
                    sub_goals,
                    dependencies,
                    rationale,
                };
            }
            Err(DecompositionError::CycleDetected { .. }) => {
                // Fall through to comma-split fallback.
            }
            Err(_) => {
                // Fall through to comma-split fallback.
            }
        }
    }

    // Fallback: existing comma/and splitting.
    let sub_descs = goal::generate_sub_goal_descriptions(&goal.description);
    let sub_goals: Vec<(String, u8, String)> = sub_descs
        .into_iter()
        .map(|(d, p, c)| (d, p, c))
        .collect();

    // Build a trivial sequential TaskTree for the fallback.
    let mut dag: DiGraph<TaskNode, DependencyEdge> = DiGraph::new();
    let mut node_indices: Vec<NodeIndex> = Vec::new();

    for (desc, priority, criteria) in &sub_goals {
        let node = TaskNode {
            kind: TaskNodeKind::SubGoal {
                description: desc.clone(),
                priority: *priority,
                success_criteria: criteria.clone(),
            },
            estimated_cycles: 3,
            rationale: "Comma-split fallback".into(),
        };
        node_indices.push(dag.add_node(node));
    }

    // Sequential ordering for fallback.
    for i in 0..node_indices.len().saturating_sub(1) {
        dag.add_edge(
            node_indices[i],
            node_indices[i + 1],
            DependencyEdge::FinishToStart,
        );
    }

    let roots = node_indices.first().copied().into_iter().collect();

    let tree = TaskTree {
        dag,
        strategy: DecompositionStrategy::Custom("comma-split".into()),
        method_name: "fallback".into(),
        roots,
    };

    DecompositionOutput {
        tree,
        sub_goals,
        dependencies: Vec::new(),
        rationale: "Fallback: comma/and splitting (no HTN method matched)".into(),
    }
}

/// Extract dependency pairs from a TaskTree as (blocker_idx, blocked_idx)
/// into the sub-goal ordering.
fn extract_dependencies(tree: &TaskTree, topo: &[NodeIndex]) -> Vec<(usize, usize)> {
    use petgraph::visit::EdgeRef;

    // Build index: NodeIndex → position in topo order.
    let pos: std::collections::HashMap<NodeIndex, usize> = topo
        .iter()
        .enumerate()
        .map(|(i, &idx)| (idx, i))
        .collect();

    let mut deps = Vec::new();
    for edge in tree.dag.edge_references() {
        if let (Some(&from_pos), Some(&to_pos)) = (pos.get(&edge.source()), pos.get(&edge.target()))
        {
            deps.push((from_pos, to_pos));
        }
    }

    deps
}

// ---------------------------------------------------------------------------
// Method stats serialization (for session persistence)
// ---------------------------------------------------------------------------

/// Serializable snapshot of method usage stats.
#[derive(Debug, Serialize, Deserialize)]
pub struct MethodStats {
    pub name: String,
    pub usage_count: u32,
    pub success_rate: f32,
}

impl MethodRegistry {
    /// Export stats for persistence.
    pub fn export_stats(&self) -> Vec<MethodStats> {
        self.methods
            .iter()
            .map(|m| MethodStats {
                name: m.name.clone(),
                usage_count: m.usage_count,
                success_rate: m.success_rate,
            })
            .collect()
    }

    /// Import stats from a persisted snapshot.
    pub fn import_stats(&mut self, stats: &[MethodStats]) {
        for stat in stats {
            if let Some(method) = self.methods.iter_mut().find(|m| m.name == stat.name) {
                method.usage_count = stat.usage_count;
                method.success_rate = stat.success_rate;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Plan::from_task_tree
// ---------------------------------------------------------------------------

impl super::plan::Plan {
    /// Flatten a TaskTree into a linear Plan by extracting PlanStep nodes
    /// in topological order. SubGoal nodes are skipped (they become separate goals).
    pub fn from_task_tree(tree: &TaskTree, goal_id: SymbolId) -> Option<super::plan::Plan> {
        let topo = tree.topological_order().ok()?;
        let mut steps = Vec::new();

        for idx in &topo {
            if let TaskNodeKind::PlanStep(ref step) = tree.dag[*idx].kind {
                let mut s = step.clone();
                s.index = steps.len();
                steps.push(s);
            }
        }

        if steps.is_empty() {
            return None;
        }

        Some(super::plan::Plan {
            goal_id,
            steps,
            status: super::plan::PlanStatus::Active,
            attempt: 0,
            strategy: format!("HTN {} ({})", tree.method_name, tree.strategy),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::goal::{Goal, GoalStatus};
    use crate::symbol::SymbolId;

    fn make_goal(desc: &str) -> Goal {
        Goal {
            symbol_id: SymbolId::new(1).unwrap(),
            description: desc.into(),
            status: GoalStatus::Active,
            priority: 128,
            success_criteria: "done".into(),
            parent: None,
            children: Vec::new(),
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            blocked_by: Vec::new(),
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        }
    }

    #[test]
    fn topological_order_sequential() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let method = registry
            .methods()
            .iter()
            .find(|m| m.name == "research")
            .unwrap();

        let tree = decompose_with_method(&make_goal("learn about VSA"), method).unwrap();
        let topo = tree.topological_order().unwrap();

        // Research has 4 sequential steps.
        assert_eq!(topo.len(), 4);
    }

    #[test]
    fn critical_path_sequential() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let method = registry
            .methods()
            .iter()
            .find(|m| m.name == "research")
            .unwrap();

        let tree = decompose_with_method(&make_goal("study knowledge graphs"), method).unwrap();
        let (total, path) = tree.critical_path().unwrap();

        // 4 nodes × 3 cycles each = 12 total on critical path.
        assert_eq!(total, 12);
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn keyword_scoring_selects_research() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let goal = make_goal("research and study the topic");
        let method = registry.select_method(&goal, &engine);

        assert!(method.is_some());
        assert_eq!(method.unwrap().name, "research");
    }

    #[test]
    fn keyword_scoring_selects_investigation() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let goal = make_goal("investigate and debug the memory leak");
        let method = registry.select_method(&goal, &engine);

        assert!(method.is_some());
        assert_eq!(method.unwrap().name, "investigation");
    }

    #[test]
    fn fallback_when_no_method_matches() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        // A very generic goal with no matching keywords.
        let goal = make_goal("xyz");
        let method = registry.select_method(&goal, &engine);

        assert!(method.is_none());
    }

    #[test]
    fn no_cycles_in_builtin_methods() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let goal = make_goal("test goal");
        for method in registry.methods() {
            let result = decompose_with_method(&goal, method);
            assert!(
                result.is_ok(),
                "Method '{}' produced a cycle",
                method.name
            );
        }
    }

    #[test]
    fn exploration_fan_out() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let method = registry
            .methods()
            .iter()
            .find(|m| m.name == "exploration")
            .unwrap();

        let tree = decompose_with_method(&make_goal("explore the codebase"), method).unwrap();
        let topo = tree.topological_order().unwrap();

        // 4 nodes, with fan-out after step 2.
        assert_eq!(topo.len(), 4);

        // Roots should be just the first node (broad survey).
        assert_eq!(tree.roots.len(), 1);
    }

    #[test]
    fn compare_contrast_diamond() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let method = registry
            .methods()
            .iter()
            .find(|m| m.name == "compare_contrast")
            .unwrap();

        let tree =
            decompose_with_method(&make_goal("compare A vs B"), method).unwrap();
        let topo = tree.topological_order().unwrap();

        // 5 nodes in diamond shape.
        assert_eq!(topo.len(), 5);

        // Two roots (gather A and gather B are independent).
        assert_eq!(tree.roots.len(), 2);
    }

    #[test]
    fn decompose_goal_htn_uses_method() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let goal = make_goal("research and study knowledge graphs");
        let result = decompose_goal_htn(&goal, &engine, &registry);

        assert!(result.rationale.contains("HTN"));
        assert!(!result.sub_goals.is_empty());
    }

    #[test]
    fn decompose_goal_htn_falls_back() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        // Goal with commas but no matching keywords → fallback to comma-split.
        let goal = make_goal("alpha thing, beta thing, gamma thing");
        let result = decompose_goal_htn(&goal, &engine, &registry);

        assert!(result.rationale.contains("Fallback"));
        assert_eq!(result.sub_goals.len(), 3);
    }

    #[test]
    fn method_stats_roundtrip() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        // Mutate a method's stats.
        registry.methods_mut()[0].usage_count = 10;
        registry.methods_mut()[0].success_rate = 0.8;

        let stats = registry.export_stats();
        assert_eq!(stats[0].usage_count, 10);

        // Create a fresh registry and import.
        let mut fresh = MethodRegistry::new();
        register_builtin_methods(&mut fresh);
        fresh.import_stats(&stats);

        assert_eq!(fresh.methods()[0].usage_count, 10);
        assert!((fresh.methods()[0].success_rate - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn plan_from_task_tree_empty_for_subgoal_only() {
        let mut registry = MethodRegistry::new();
        register_builtin_methods(&mut registry);

        let method = registry
            .methods()
            .iter()
            .find(|m| m.name == "research")
            .unwrap();

        let tree = decompose_with_method(&make_goal("learn about VSA"), method).unwrap();
        let plan =
            super::super::plan::Plan::from_task_tree(&tree, SymbolId::new(1).unwrap());

        // All nodes are SubGoal, not PlanStep, so the Plan should be None.
        assert!(plan.is_none());
    }
}
