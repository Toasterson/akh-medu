//! Skill dependency graph: topological activation ordering and conflict detection.
//!
//! Phase 31a — builds a petgraph DAG from skill manifests, performs topological
//! sort for safe multi-skill activation, detects circular dependencies and
//! conflicts, and resolves loose capability requirements to concrete providers.

use std::collections::HashMap;

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use crate::error::{SkillError, SkillResult};

use super::SkillManifest;

/// Direction of a dependency edge in the graph.
///
/// An edge `A → B` means "A depends on B" (B must be activated first).
type DepEdge = ();

/// A directed acyclic graph of skill dependencies.
///
/// Nodes are skill IDs; edges point from dependent to dependency
/// (`A → B` = "A needs B active before it can activate").
pub struct SkillDependencyGraph {
    graph: DiGraph<String, DepEdge>,
    node_index: HashMap<String, NodeIndex>,
}

impl Default for SkillDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillDependencyGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_index: HashMap::new(),
        }
    }

    /// Ensure a skill node exists; return its `NodeIndex`.
    pub fn add_skill(&mut self, skill_id: &str) -> NodeIndex {
        if let Some(&idx) = self.node_index.get(skill_id) {
            return idx;
        }
        let idx = self.graph.add_node(skill_id.to_string());
        self.node_index.insert(skill_id.to_string(), idx);
        idx
    }

    /// Add a dependency edge: `dependent` → `dependency`.
    ///
    /// Both nodes are created if not already present.
    pub fn add_dependency(&mut self, dependent: &str, dependency: &str) {
        let dep_node = self.add_skill(dependent);
        let req_node = self.add_skill(dependency);
        // Only add if the edge doesn't already exist.
        if !self.graph.contains_edge(dep_node, req_node) {
            self.graph.add_edge(dep_node, req_node, ());
        }
    }

    /// Build a dependency graph from a collection of manifests.
    ///
    /// Adds all skills and their declared `depends` edges.
    pub fn from_manifests<'a>(manifests: impl IntoIterator<Item = &'a SkillManifest>) -> Self {
        let mut g = Self::new();
        for m in manifests {
            g.add_skill(&m.id);
            for dep in &m.depends {
                g.add_dependency(&m.id, dep);
            }
        }
        g
    }

    /// Return the order in which skills should be activated so every skill's
    /// dependencies are activated before it.
    ///
    /// The returned list is ordered from "must be activated first" to last.
    /// Returns `SkillError::CircularDependency` if the graph has a cycle.
    pub fn activation_order(&self) -> SkillResult<Vec<String>> {
        // toposort returns nodes in topological order (dependencies before dependents
        // when edges go dependent → dependency, so reverse).
        match toposort(&self.graph, None) {
            Ok(order) => {
                // Reverse so dependencies come first.
                let skills: Vec<String> = order
                    .into_iter()
                    .rev()
                    .map(|idx| self.graph[idx].clone())
                    .collect();
                Ok(skills)
            }
            Err(cycle) => {
                let node_label = self.graph[cycle.node_id()].clone();
                Err(SkillError::CircularDependency {
                    cycle: vec![node_label],
                })
            }
        }
    }

    /// Return the skills that directly depend on `skill_id` (its reverse dependencies).
    ///
    /// Used during deactivation to warn if dependents are still active.
    pub fn dependents_of(&self, skill_id: &str) -> Vec<String> {
        let Some(&idx) = self.node_index.get(skill_id) else {
            return Vec::new();
        };
        // Edges go dependent → dependency; we want nodes that point TO skill_id.
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|e| self.graph[e.source()].clone())
            .collect()
    }

    /// Return the skill IDs that `skill_id` directly depends on.
    pub fn dependencies_of(&self, skill_id: &str) -> Vec<String> {
        let Some(&idx) = self.node_index.get(skill_id) else {
            return Vec::new();
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|e| self.graph[e.target()].clone())
            .collect()
    }

    /// Check if activating `skill_id` would conflict with any currently-active skills.
    ///
    /// Returns the first conflicting skill ID found, or `None` if there are no conflicts.
    pub fn check_conflicts(
        manifest: &SkillManifest,
        active_skill_ids: &[String],
    ) -> Option<String> {
        for active in active_skill_ids {
            if manifest.conflicts.contains(active) {
                return Some(active.clone());
            }
        }
        None
    }

    /// Resolve capability requirements: for each entry in `requires`, find the
    /// skill IDs that `provide` it.
    ///
    /// Returns `Ok(providers)` — a deduplicated list of skill IDs that together
    /// satisfy all capabilities in `requires`.
    ///
    /// Returns `Err(UnresolvedCapability)` if any capability has no provider.
    pub fn resolve_capabilities<'a>(
        requires: &[String],
        all_manifests: impl IntoIterator<Item = &'a SkillManifest>,
    ) -> SkillResult<Vec<String>> {
        let manifests: Vec<&SkillManifest> = all_manifests.into_iter().collect();
        let mut providers: Vec<String> = Vec::new();

        for cap in requires {
            let found = manifests.iter().any(|m| m.provides.contains(cap));
            if !found {
                return Err(SkillError::UnresolvedCapability {
                    capability: cap.clone(),
                });
            }
            // Collect providers for this capability.
            for m in &manifests {
                if m.provides.contains(cap) && !providers.contains(&m.id) {
                    providers.push(m.id.clone());
                }
            }
        }

        Ok(providers)
    }
}

/// Compute the full activation plan for a target skill given a set of all known
/// manifests and a set of currently active skill IDs.
///
/// Returns an ordered list of skill IDs to activate (dependencies first).
/// Already-active skills are excluded from the returned list.
pub fn compute_activation_plan(
    target_skill_id: &str,
    all_manifests: &HashMap<String, SkillManifest>,
    active_skill_ids: &[String],
) -> SkillResult<Vec<String>> {
    // Collect the transitive closure of dependencies via BFS/DFS.
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stack: Vec<String> = vec![target_skill_id.to_string()];
    let mut relevant: Vec<SkillManifest> = Vec::new();

    while let Some(id) = stack.pop() {
        if visited.contains(&id) {
            continue;
        }
        visited.insert(id.clone());

        if let Some(m) = all_manifests.get(&id) {
            // Check conflicts with already-active skills.
            if let Some(conflict) = SkillDependencyGraph::check_conflicts(m, active_skill_ids) {
                return Err(SkillError::SkillConflict {
                    skill_id: id.clone(),
                    conflict,
                });
            }
            for dep in &m.depends {
                if !visited.contains(dep) {
                    stack.push(dep.clone());
                }
            }
            relevant.push(m.clone());
        }
        // Unknown dependency IDs are ignored here; the caller may choose to error.
    }

    let g = SkillDependencyGraph::from_manifests(relevant.iter());
    let order = g.activation_order()?;

    // Filter out already-active skills.
    let plan: Vec<String> = order
        .into_iter()
        .filter(|id| !active_skill_ids.contains(id))
        .collect();

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(id: &str, depends: &[&str], provides: &[&str], requires: &[&str]) -> SkillManifest {
        SkillManifest {
            id: id.to_string(),
            name: id.to_string(),
            version: "1.0.0".into(),
            description: String::new(),
            domains: Vec::new(),
            weight_size_bytes: 0,
            triples_file: None,
            rules_file: None,
            cli_tools: Vec::new(),
            wasm_tools: Vec::new(),
            tool_config: Default::default(),
            depends: depends.iter().map(|s| s.to_string()).collect(),
            conflicts: Vec::new(),
            provides: provides.iter().map(|s| s.to_string()).collect(),
            requires: requires.iter().map(|s| s.to_string()).collect(),
            remote_url: None,
            curriculum_file: None,
        }
    }

    #[test]
    fn topological_order_simple() {
        // c → b → a  (a must activate first)
        let mut g = SkillDependencyGraph::new();
        g.add_dependency("c", "b");
        g.add_dependency("b", "a");

        let order = g.activation_order().unwrap();
        let pos: HashMap<_, _> = order
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();
        assert!(pos["a"] < pos["b"]);
        assert!(pos["b"] < pos["c"]);
    }

    #[test]
    fn circular_dependency_detected() {
        let mut g = SkillDependencyGraph::new();
        g.add_dependency("a", "b");
        g.add_dependency("b", "c");
        g.add_dependency("c", "a");

        assert!(matches!(
            g.activation_order(),
            Err(SkillError::CircularDependency { .. })
        ));
    }

    #[test]
    fn conflict_detection() {
        let m = manifest("new_skill", &[], &[], &[]);
        let mut m_with_conflict = m.clone();
        m_with_conflict.conflicts = vec!["old_skill".to_string()];

        let active = vec!["old_skill".to_string()];
        let conflict = SkillDependencyGraph::check_conflicts(&m_with_conflict, &active);
        assert_eq!(conflict, Some("old_skill".to_string()));

        let no_conflict = SkillDependencyGraph::check_conflicts(&m, &active);
        assert!(no_conflict.is_none());
    }

    #[test]
    fn capability_resolution() {
        let provider = manifest("physics", &[], &["spatial_reasoning", "kinematics"], &[]);
        let all = vec![provider];

        let providers = SkillDependencyGraph::resolve_capabilities(
            &["spatial_reasoning".to_string()],
            all.iter(),
        )
        .unwrap();
        assert_eq!(providers, vec!["physics"]);
    }

    #[test]
    fn unresolved_capability_errors() {
        let all: Vec<SkillManifest> = vec![];
        let result = SkillDependencyGraph::resolve_capabilities(
            &["quantum_physics".to_string()],
            all.iter(),
        );
        assert!(matches!(
            result,
            Err(SkillError::UnresolvedCapability { .. })
        ));
    }

    #[test]
    fn activation_plan_respects_active_skills() {
        let a = manifest("a", &[], &[], &[]);
        let b = manifest("b", &["a"], &[], &[]);
        let c = manifest("c", &["b"], &[], &[]);

        let all: HashMap<String, SkillManifest> =
            [a, b, c].into_iter().map(|m| (m.id.clone(), m)).collect();

        // "a" is already active; plan for "c" should only include b and c.
        let active = vec!["a".to_string()];
        let plan = compute_activation_plan("c", &all, &active).unwrap();
        assert!(!plan.contains(&"a".to_string()));
        assert!(plan.contains(&"b".to_string()));
        assert!(plan.contains(&"c".to_string()));
    }
}
