//! Project abstraction: groups related goals under a KG-backed microtheory context.
//!
//! Projects are the cross-session grouping layer — an "agenda" of projects with
//! priority ordering persists across sessions and drives goal selection in the OODA loop.
//! Each project is backed by a `ContextDomain::Task` microtheory for scoped reasoning.

use serde::{Deserialize, Serialize};

use crate::compartment::ContextDomain;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolId, SymbolKind};
use crate::vsa::HyperVec;

use super::agent::AgentPredicates;
use super::error::{AgentError, AgentResult};
use super::goal::Goal;

// ---------------------------------------------------------------------------
// Project status
// ---------------------------------------------------------------------------

/// Lifecycle status of a project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectStatus {
    /// Actively being worked on.
    Active,
    /// All goals completed successfully.
    Completed,
    /// Temporarily paused (e.g., waiting for external input).
    Suspended,
    /// No longer relevant; kept for history.
    Archived,
}

impl std::fmt::Display for ProjectStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Completed => write!(f, "completed"),
            Self::Suspended => write!(f, "suspended"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

impl ProjectStatus {
    /// Parse from a string label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "active" => Some(Self::Active),
            "completed" => Some(Self::Completed),
            "suspended" => Some(Self::Suspended),
            "archived" => Some(Self::Archived),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/// A named grouping of related goals, backed by a KG microtheory.
#[derive(Debug, Clone)]
pub struct Project {
    /// The project's entity symbol in the KG.
    pub id: SymbolId,
    /// Human-readable project name.
    pub name: String,
    /// Goals belonging to this project.
    pub goals: Vec<SymbolId>,
    /// Bundled scope vector for VSA-based goal assignment.
    pub scope_vector: Option<HyperVec>,
    /// Current lifecycle status.
    pub status: ProjectStatus,
    /// Optional cycle budget (resource-awareness hook for Phase 11g).
    pub cycle_budget: Option<u32>,
    /// Cycles consumed so far across all sessions.
    pub cycles_consumed: u32,
    /// Cycle at which this project was created.
    pub created_at: u64,
}

impl Project {
    /// Whether the project's budget is more than 80% consumed.
    pub fn budget_warning(&self) -> bool {
        match self.cycle_budget {
            Some(budget) if budget > 0 => {
                self.cycles_consumed as f32 / budget as f32 > 0.8
            }
            _ => false,
        }
    }

    /// Whether the project's budget has been fully consumed.
    pub fn budget_exceeded(&self) -> bool {
        match self.cycle_budget {
            Some(budget) => self.cycles_consumed >= budget,
            None => false,
        }
    }

    /// Consume one cycle from this project's budget.
    pub fn consume_cycle(&mut self) {
        self.cycles_consumed = self.cycles_consumed.saturating_add(1);
    }
}

// ---------------------------------------------------------------------------
// ProjectPredicates
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for project KG triples.
///
/// Separate from `AgentPredicates` to keep project concerns clean.
/// Both use the `agent:` namespace.
#[derive(Debug, Clone)]
pub struct ProjectPredicates {
    /// `agent:contains_goal` — links a project to one of its goals.
    pub contains_goal: SymbolId,
    /// `agent:project_scope` — links a project to a scope concept.
    pub project_scope: SymbolId,
    /// `agent:project_description` — links a project to its description entity.
    pub project_description: SymbolId,
    /// `agent:project_status` — links a project to its status entity.
    pub project_status: SymbolId,
}

impl ProjectPredicates {
    /// Resolve or create all project predicates in the engine.
    pub fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            contains_goal: engine.resolve_or_create_relation("agent:contains_goal")?,
            project_scope: engine.resolve_or_create_relation("agent:project_scope")?,
            project_description: engine.resolve_or_create_relation("agent:project_description")?,
            project_status: engine.resolve_or_create_relation("agent:project_status")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Agenda
// ---------------------------------------------------------------------------

/// Ordered list of projects with priorities, persisted across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agenda {
    /// (project_id, priority) pairs, sorted highest priority first.
    pub projects: Vec<(SymbolId, u8)>,
    /// Currently active project (the one the agent focuses on).
    pub active_project: Option<SymbolId>,
}

impl Agenda {
    /// Create an empty agenda.
    pub fn new() -> Self {
        Self {
            projects: Vec::new(),
            active_project: None,
        }
    }

    /// Add a project with the given priority.
    pub fn add_project(&mut self, project_id: SymbolId, priority: u8) {
        // Remove if already present (update).
        self.projects.retain(|(id, _)| *id != project_id);
        self.projects.push((project_id, priority));
        // Sort by priority descending.
        self.projects.sort_by(|a, b| b.1.cmp(&a.1));
    }

    /// Remove a project from the agenda.
    pub fn remove_project(&mut self, project_id: SymbolId) {
        self.projects.retain(|(id, _)| *id != project_id);
        if self.active_project == Some(project_id) {
            self.active_project = None;
        }
    }

    /// Select the next active project from the agenda.
    ///
    /// Picks the highest-priority project that is `Active` (not completed/suspended/archived).
    /// Returns the selected project ID, or `None` if no suitable project exists.
    pub fn select_active(&mut self, projects: &[Project]) -> Option<SymbolId> {
        let active_id = self
            .projects
            .iter()
            .find(|(id, _)| {
                projects
                    .iter()
                    .any(|p| p.id == *id && p.status == ProjectStatus::Active)
            })
            .map(|(id, _)| *id);

        self.active_project = active_id;
        active_id
    }
}

impl Default for Agenda {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProjectAssignment
// ---------------------------------------------------------------------------

/// Result of assigning a goal to a project via VSA similarity.
#[derive(Debug, Clone)]
pub enum ProjectAssignment {
    /// Goal matches an existing project.
    Existing {
        project_id: SymbolId,
        similarity: f32,
    },
    /// No project matched; a new one should be created.
    NewProjectNeeded,
}

// ---------------------------------------------------------------------------
// Project CRUD
// ---------------------------------------------------------------------------

/// Create a new project backed by a `Task` microtheory in the KG.
///
/// Stores description, status, and scope triples. Optionally computes a
/// scope vector from the given concept labels.
pub fn create_project(
    engine: &Engine,
    name: &str,
    description: &str,
    scope_concepts: &[&str],
    predicates: &ProjectPredicates,
    agent_predicates: &AgentPredicates,
) -> AgentResult<Project> {
    // Create the project entity.
    let label = format!("project:{name}");
    let project_sym = engine.create_symbol(SymbolKind::Entity, &label)?;

    // Create the backing microtheory.
    let _microtheory = engine.create_context(&label, ContextDomain::Task, &[])?;

    // Store description.
    let desc_entity = engine.resolve_or_create_entity(&format!("desc:{description}"))?;
    engine.add_triple(&Triple::new(
        project_sym.id,
        predicates.project_description,
        desc_entity,
    ))?;

    // Store initial status.
    let status_entity = engine.resolve_or_create_entity("active")?;
    engine.add_triple(&Triple::new(
        project_sym.id,
        predicates.project_status,
        status_entity,
    ))?;

    // Store scope concepts as triples and compute scope vector.
    let mut scope_ids = Vec::new();
    for concept in scope_concepts {
        let concept_id = engine.resolve_or_create_entity(concept)?;
        engine.add_triple(&Triple::new(
            project_sym.id,
            predicates.project_scope,
            concept_id,
        ))?;
        scope_ids.push(concept_id);
    }

    let scope_vector = compute_scope_vector(engine, &scope_ids);

    // Store description via agent:has_description for restore compatibility.
    engine.add_triple(&Triple::new(
        project_sym.id,
        agent_predicates.has_description,
        desc_entity,
    ))?;

    // Record provenance.
    let mut prov = ProvenanceRecord::new(
        project_sym.id,
        DerivationKind::ProjectCreated {
            name: name.to_string(),
        },
    )
    .with_confidence(1.0);
    let _ = engine.store_provenance(&mut prov);

    Ok(Project {
        id: project_sym.id,
        name: name.to_string(),
        goals: Vec::new(),
        scope_vector,
        status: ProjectStatus::Active,
        cycle_budget: None,
        cycles_consumed: 0,
        created_at: project_sym.created_at,
    })
}

/// Restore all projects from the KG.
///
/// Scans for entities with `agent:project_status` triples and reconstructs
/// the `Project` struct for each.
pub fn restore_projects(
    engine: &Engine,
    predicates: &ProjectPredicates,
    agent_predicates: &AgentPredicates,
) -> AgentResult<Vec<Project>> {
    let mut projects = Vec::new();

    // Scan all symbols for project-prefixed entities.
    for sym in engine.all_symbols() {
        let label = engine.resolve_label(sym.id);
        if !label.starts_with("project:") {
            continue;
        }

        let triples = engine.triples_from(sym.id);

        // Must have a project_status triple to be a valid project.
        let status_label = triples
            .iter()
            .find(|t| t.predicate == predicates.project_status)
            .map(|t| engine.resolve_label(t.object));

        let status = match status_label.as_deref().and_then(ProjectStatus::from_label) {
            Some(s) => s,
            None => continue, // Not a valid project entity.
        };

        let name = label.strip_prefix("project:").unwrap_or(&label).to_string();

        // Gather goals.
        let goals: Vec<SymbolId> = triples
            .iter()
            .filter(|t| t.predicate == predicates.contains_goal)
            .map(|t| t.object)
            .collect();

        // Gather scope concepts and compute vector.
        let scope_ids: Vec<SymbolId> = triples
            .iter()
            .filter(|t| t.predicate == predicates.project_scope)
            .map(|t| t.object)
            .collect();

        let scope_vector = compute_scope_vector(engine, &scope_ids);

        projects.push(Project {
            id: sym.id,
            name,
            goals,
            scope_vector,
            status,
            cycle_budget: None,
            cycles_consumed: 0,
            created_at: sym.created_at,
        });
    }

    Ok(projects)
}

/// Add a goal to a project by creating a `contains_goal` triple.
pub fn add_goal_to_project(
    engine: &Engine,
    project: &mut Project,
    goal_id: SymbolId,
    predicates: &ProjectPredicates,
) -> AgentResult<()> {
    engine.add_triple(&Triple::new(
        project.id,
        predicates.contains_goal,
        goal_id,
    ))?;
    if !project.goals.contains(&goal_id) {
        project.goals.push(goal_id);
    }
    Ok(())
}

/// Compute project progress as the fraction of goals that are completed.
///
/// Returns a value in `[0.0, 1.0]`. An empty project returns `0.0`.
pub fn project_progress(project: &Project, goals: &[Goal]) -> f32 {
    if project.goals.is_empty() {
        return 0.0;
    }
    let completed = project
        .goals
        .iter()
        .filter(|gid| {
            goals
                .iter()
                .any(|g| g.symbol_id == **gid && matches!(g.status, super::goal::GoalStatus::Completed))
        })
        .count();
    completed as f32 / project.goals.len() as f32
}

/// Update a project's status in the KG.
pub fn update_project_status(
    engine: &Engine,
    project: &mut Project,
    new_status: ProjectStatus,
    predicates: &ProjectPredicates,
) -> AgentResult<()> {
    let status_entity = engine.resolve_or_create_entity(&new_status.to_string())?;
    engine.add_triple(&Triple::new(
        project.id,
        predicates.project_status,
        status_entity,
    ))?;
    project.status = new_status;
    Ok(())
}

// ---------------------------------------------------------------------------
// VSA project scoping
// ---------------------------------------------------------------------------

/// Compute a scope vector by bundling concept hypervectors.
///
/// Returns `None` if there are no concepts or the bundle fails.
pub fn compute_scope_vector(engine: &Engine, scope_concepts: &[SymbolId]) -> Option<HyperVec> {
    if scope_concepts.is_empty() {
        return None;
    }

    let ops = engine.ops();
    let im = engine.item_memory();

    let vecs: Vec<HyperVec> = scope_concepts
        .iter()
        .map(|id| im.get_or_create(ops, *id))
        .collect();

    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs).ok()
}

/// Assign a goal to the best-matching project via VSA similarity.
///
/// Encodes the goal's description symbol as a hypervector and compares it
/// against each project's scope vector. If the best match exceeds `threshold`,
/// returns `Existing`; otherwise `NewProjectNeeded`.
pub fn assign_goal_to_project(
    goal: &Goal,
    projects: &[Project],
    engine: &Engine,
    threshold: f32,
) -> ProjectAssignment {
    if projects.is_empty() {
        return ProjectAssignment::NewProjectNeeded;
    }

    let ops = engine.ops();
    let im = engine.item_memory();
    let goal_vec = im.get_or_create(ops, goal.symbol_id);

    let mut best_sim = 0.0f32;
    let mut best_project: Option<SymbolId> = None;

    for project in projects {
        if let Some(ref scope_vec) = project.scope_vector {
            if let Ok(sim) = ops.similarity(&goal_vec, scope_vec) {
                if sim > best_sim {
                    best_sim = sim;
                    best_project = Some(project.id);
                }
            }
        }
    }

    match best_project {
        Some(project_id) if best_sim >= threshold => ProjectAssignment::Existing {
            project_id,
            similarity: best_sim,
        },
        _ => ProjectAssignment::NewProjectNeeded,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolId;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    // -- ProjectStatus --

    #[test]
    fn project_status_display() {
        assert_eq!(ProjectStatus::Active.to_string(), "active");
        assert_eq!(ProjectStatus::Completed.to_string(), "completed");
        assert_eq!(ProjectStatus::Suspended.to_string(), "suspended");
        assert_eq!(ProjectStatus::Archived.to_string(), "archived");
    }

    #[test]
    fn project_status_roundtrip() {
        for status in [
            ProjectStatus::Active,
            ProjectStatus::Completed,
            ProjectStatus::Suspended,
            ProjectStatus::Archived,
        ] {
            let label = status.to_string();
            assert_eq!(ProjectStatus::from_label(&label), Some(status));
        }
        assert_eq!(ProjectStatus::from_label("unknown"), None);
    }

    // -- Agenda --

    #[test]
    fn agenda_add_and_select() {
        let mut agenda = Agenda::new();
        agenda.add_project(sym(1), 100);
        agenda.add_project(sym(2), 200);

        assert_eq!(agenda.projects.len(), 2);
        // Highest priority first.
        assert_eq!(agenda.projects[0], (sym(2), 200));
        assert_eq!(agenda.projects[1], (sym(1), 100));
    }

    #[test]
    fn agenda_remove() {
        let mut agenda = Agenda::new();
        agenda.add_project(sym(1), 100);
        agenda.add_project(sym(2), 200);
        agenda.active_project = Some(sym(2));

        agenda.remove_project(sym(2));
        assert_eq!(agenda.projects.len(), 1);
        assert_eq!(agenda.active_project, None);
    }

    #[test]
    fn agenda_skips_completed() {
        let mut agenda = Agenda::new();
        agenda.add_project(sym(1), 200);
        agenda.add_project(sym(2), 100);

        let projects = vec![
            Project {
                id: sym(1),
                name: "done".into(),
                goals: vec![],
                scope_vector: None,
                status: ProjectStatus::Completed,
                cycle_budget: None,
                cycles_consumed: 0,
                created_at: 0,
            },
            Project {
                id: sym(2),
                name: "still going".into(),
                goals: vec![],
                scope_vector: None,
                status: ProjectStatus::Active,
                cycle_budget: None,
                cycles_consumed: 0,
                created_at: 0,
            },
        ];

        let selected = agenda.select_active(&projects);
        assert_eq!(selected, Some(sym(2)));
        assert_eq!(agenda.active_project, Some(sym(2)));
    }

    #[test]
    fn agenda_empty() {
        let mut agenda = Agenda::new();
        let selected = agenda.select_active(&[]);
        assert_eq!(selected, None);
        assert_eq!(agenda.active_project, None);
    }

    // -- Project progress --

    #[test]
    fn project_progress_half_done() {
        use super::super::goal::GoalStatus;

        let project = Project {
            id: sym(10),
            name: "test".into(),
            goals: vec![sym(1), sym(2)],
            scope_vector: None,
            status: ProjectStatus::Active,
            cycle_budget: None,
            cycles_consumed: 0,
            created_at: 0,
        };

        let goals = vec![
            Goal {
                symbol_id: sym(1),
                description: "g1".into(),
                status: GoalStatus::Completed,
                priority: 100,
                success_criteria: "done".into(),
                parent: None,
                children: vec![],
                created_at: 0,
                cycles_worked: 5,
                last_progress_cycle: 5,
                source: None,
                blocked_by: vec![],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
            estimated_effort: None,
            },
            Goal {
                symbol_id: sym(2),
                description: "g2".into(),
                status: GoalStatus::Active,
                priority: 100,
                success_criteria: "not yet".into(),
                parent: None,
                children: vec![],
                created_at: 0,
                cycles_worked: 2,
                last_progress_cycle: 2,
                source: None,
                blocked_by: vec![],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
            estimated_effort: None,
            },
        ];

        let progress = project_progress(&project, &goals);
        assert!((progress - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn project_progress_empty() {
        let project = Project {
            id: sym(10),
            name: "empty".into(),
            goals: vec![],
            scope_vector: None,
            status: ProjectStatus::Active,
            cycle_budget: None,
            cycles_consumed: 0,
            created_at: 0,
        };

        assert_eq!(project_progress(&project, &[]), 0.0);
    }

    // -- KG-backed operations --

    #[test]
    fn predicates_init() {
        let engine = test_engine();
        let preds = ProjectPredicates::init(&engine).unwrap();
        // All predicate IDs should be distinct.
        let ids = [
            preds.contains_goal,
            preds.project_scope,
            preds.project_description,
            preds.project_status,
        ];
        for (i, a) in ids.iter().enumerate() {
            for (j, b) in ids.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "predicate {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn create_and_restore_project() {
        let engine = test_engine();
        let preds = ProjectPredicates::init(&engine).unwrap();
        let agent_preds = AgentPredicates::init(&engine).unwrap();

        let project = create_project(
            &engine,
            "test-proj",
            "A test project",
            &["rust", "testing"],
            &preds,
            &agent_preds,
        )
        .unwrap();

        assert_eq!(project.name, "test-proj");
        assert_eq!(project.status, ProjectStatus::Active);
        assert!(project.scope_vector.is_some());

        // Restore should find it.
        let restored = restore_projects(&engine, &preds, &agent_preds).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].name, "test-proj");
        assert_eq!(restored[0].status, ProjectStatus::Active);
    }

    #[test]
    fn add_goal_and_update_status() {
        let engine = test_engine();
        let preds = ProjectPredicates::init(&engine).unwrap();
        let agent_preds = AgentPredicates::init(&engine).unwrap();

        let mut project = create_project(
            &engine,
            "proj2",
            "Second project",
            &[],
            &preds,
            &agent_preds,
        )
        .unwrap();

        // Add a fake goal.
        let goal_sym = engine
            .create_symbol(SymbolKind::Entity, "goal:test")
            .unwrap();
        add_goal_to_project(&engine, &mut project, goal_sym.id, &preds).unwrap();
        assert_eq!(project.goals.len(), 1);

        // Adding the same goal is idempotent.
        add_goal_to_project(&engine, &mut project, goal_sym.id, &preds).unwrap();
        assert_eq!(project.goals.len(), 1);

        // Update status.
        update_project_status(&engine, &mut project, ProjectStatus::Suspended, &preds).unwrap();
        assert_eq!(project.status, ProjectStatus::Suspended);
    }

    // -- VSA scoping --

    #[test]
    fn assign_existing_project() {
        let engine = test_engine();
        let ops = engine.ops();
        let im = engine.item_memory();

        // Create a project with scope vector from "rust" and "code".
        let rust_id = engine.resolve_or_create_entity("rust").unwrap();
        let code_id = engine.resolve_or_create_entity("code").unwrap();
        let scope_vec = compute_scope_vector(&engine, &[rust_id, code_id]);

        let project = Project {
            id: engine
                .create_symbol(SymbolKind::Entity, "project:rust-code")
                .unwrap()
                .id,
            name: "rust-code".into(),
            goals: vec![],
            scope_vector: scope_vec,
            status: ProjectStatus::Active,
            cycle_budget: None,
            cycles_consumed: 0,
            created_at: 0,
        };

        // Create a goal with a symbol that is related to "rust".
        let goal_sym = engine
            .create_symbol(SymbolKind::Entity, "goal:write-rust-code")
            .unwrap();
        let goal = Goal {
            symbol_id: goal_sym.id,
            description: "Write Rust code".into(),
            status: super::super::goal::GoalStatus::Active,
            priority: 100,
            success_criteria: "code compiles".into(),
            parent: None,
            children: vec![],
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            blocked_by: vec![],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        };

        // With a very low threshold, should match.
        let result = assign_goal_to_project(&goal, &[project], &engine, 0.0);
        match result {
            ProjectAssignment::Existing { similarity, .. } => {
                assert!(similarity >= 0.0);
            }
            ProjectAssignment::NewProjectNeeded => {
                panic!("expected Existing with threshold 0.0");
            }
        }
    }

    #[test]
    fn empty_projects_returns_new_needed() {
        let engine = test_engine();
        let goal_sym = engine
            .create_symbol(SymbolKind::Entity, "goal:anything")
            .unwrap();
        let goal = Goal {
            symbol_id: goal_sym.id,
            description: "anything".into(),
            status: super::super::goal::GoalStatus::Active,
            priority: 100,
            success_criteria: "done".into(),
            parent: None,
            children: vec![],
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            blocked_by: vec![],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        };

        let result = assign_goal_to_project(&goal, &[], &engine, 0.6);
        assert!(matches!(result, ProjectAssignment::NewProjectNeeded));
    }

    #[test]
    fn scope_vector_bundle() {
        let engine = test_engine();
        let a = engine.resolve_or_create_entity("alpha").unwrap();
        let b = engine.resolve_or_create_entity("beta").unwrap();

        let vec = compute_scope_vector(&engine, &[a, b]);
        assert!(vec.is_some());

        // Empty input returns None.
        let empty = compute_scope_vector(&engine, &[]);
        assert!(empty.is_none());
    }
}
