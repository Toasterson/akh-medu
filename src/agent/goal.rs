//! Goal system: hierarchical goals stored as KG entities.
//!
//! Goals are Entity symbols in the knowledge graph with well-known
//! relation predicates for status, priority, criteria, and hierarchy.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::agent::AgentPredicates;
use super::error::AgentResult;

/// Status of a goal.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalStatus {
    Pending,
    Active,
    Completed,
    Failed { reason: String },
    Suspended,
}

impl GoalStatus {
    /// Serialize to a string for KG storage.
    pub fn as_label(&self) -> String {
        match self {
            Self::Pending => "pending".into(),
            Self::Active => "active".into(),
            Self::Completed => "completed".into(),
            Self::Failed { reason } => format!("failed:{reason}"),
            Self::Suspended => "suspended".into(),
        }
    }

    /// Parse from a KG label string.
    pub fn from_label(label: &str) -> Self {
        if let Some(reason) = label.strip_prefix("failed:") {
            Self::Failed {
                reason: reason.into(),
            }
        } else {
            match label {
                "pending" => Self::Pending,
                "active" => Self::Active,
                "completed" => Self::Completed,
                "suspended" => Self::Suspended,
                _ => Self::Pending,
            }
        }
    }
}

impl std::fmt::Display for GoalStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Active => write!(f, "active"),
            Self::Completed => write!(f, "completed"),
            Self::Failed { reason } => write!(f, "failed: {reason}"),
            Self::Suspended => write!(f, "suspended"),
        }
    }
}

/// Default number of cycles without progress before a goal is considered stalled.
pub const DEFAULT_STALL_THRESHOLD: u32 = 5;

/// A goal the agent is working toward.
#[derive(Debug, Clone)]
pub struct Goal {
    /// The goal's Entity symbol in the KG.
    pub symbol_id: SymbolId,
    /// Human-readable description.
    pub description: String,
    /// Current status.
    pub status: GoalStatus,
    /// Priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// How to know when done.
    pub success_criteria: String,
    /// Parent goal for hierarchical decomposition.
    pub parent: Option<SymbolId>,
    /// Child sub-goals.
    pub children: Vec<SymbolId>,
    /// When this goal was created (seconds since UNIX epoch).
    pub created_at: u64,
    /// How many OODA cycles have targeted this goal.
    pub cycles_worked: u32,
    /// Last cycle where the goal made meaningful progress (Advanced or Completed).
    pub last_progress_cycle: u64,
}

impl Goal {
    /// Whether this goal is stalled: has been worked on for `threshold` cycles
    /// since it last made progress.
    pub fn is_stalled(&self, current_cycle: u64, threshold: u32) -> bool {
        self.cycles_worked >= threshold
            && current_cycle.saturating_sub(self.last_progress_cycle) >= threshold as u64
    }
}

/// Create a goal and persist it in the knowledge graph.
///
/// If a symbol with the same goal label already exists (e.g. from a previous
/// persisted session), it is reused rather than causing a duplicate-label error.
pub fn create_goal(
    engine: &Engine,
    description: &str,
    priority: u8,
    criteria: &str,
    predicates: &AgentPredicates,
) -> AgentResult<Goal> {
    let label = format!("goal:{}", description.chars().take(40).collect::<String>());
    let goal_id = engine.resolve_or_create_entity(&label)?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Store description.
    let desc_sym = engine
        .resolve_or_create_entity(&format!("desc:{description}"))?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_description, desc_sym));

    // Store status.
    let status_sym = engine
        .resolve_or_create_entity("status:active")?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_status, status_sym));

    // Store priority.
    let priority_sym = engine
        .resolve_or_create_entity(&format!("priority:{priority}"))?;
    let _ = engine.add_triple(&Triple::new(
        goal_id,
        predicates.has_priority,
        priority_sym,
    ));

    // Store criteria.
    let criteria_sym = engine
        .resolve_or_create_entity(&format!("criteria:{criteria}"))?;
    let _ = engine.add_triple(&Triple::new(
        goal_id,
        predicates.has_criteria,
        criteria_sym,
    ));

    Ok(Goal {
        symbol_id: goal_id,
        description: description.into(),
        status: GoalStatus::Active,
        priority,
        success_criteria: criteria.into(),
        parent: None,
        children: Vec::new(),
        created_at: now,
        cycles_worked: 0,
        last_progress_cycle: 0,
    })
}

/// Decompose a parent goal into sub-goals.
pub fn decompose_goal(
    engine: &Engine,
    parent: &mut Goal,
    sub_descriptions: &[(&str, u8, &str)],
    predicates: &AgentPredicates,
) -> AgentResult<Vec<Goal>> {
    let mut children = Vec::new();
    for (desc, priority, criteria) in sub_descriptions {
        let mut child = create_goal(engine, desc, *priority, criteria, predicates)?;
        child.parent = Some(parent.symbol_id);

        // Store parent→child relation.
        let _ = engine.add_triple(&Triple::new(
            parent.symbol_id,
            predicates.child_goal,
            child.symbol_id,
        ));
        let _ = engine.add_triple(&Triple::new(
            child.symbol_id,
            predicates.parent_goal,
            parent.symbol_id,
        ));

        parent.children.push(child.symbol_id);
        children.push(child);
    }
    Ok(children)
}

/// Update a goal's status in the KG.
pub fn update_goal_status(
    engine: &Engine,
    goal: &mut Goal,
    new_status: GoalStatus,
    predicates: &AgentPredicates,
) -> AgentResult<()> {
    let status_label = format!("status:{}", new_status.as_label());
    let status_sym = engine
        .resolve_or_create_entity(&status_label)?;
    let _ = engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_status,
        status_sym,
    ));
    goal.status = new_status;
    Ok(())
}

/// Filter active goals, sorted by priority (highest first).
pub fn active_goals(goals: &[Goal]) -> Vec<&Goal> {
    let mut active: Vec<&Goal> = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .collect();
    active.sort_by(|a, b| b.priority.cmp(&a.priority));
    active
}

/// Restore goals from the knowledge graph.
pub fn restore_goals(engine: &Engine, predicates: &AgentPredicates) -> AgentResult<Vec<Goal>> {
    let mut goals = Vec::new();

    // Find all symbols whose label starts with "goal:".
    for meta in engine.all_symbols() {
        if !meta.label.starts_with("goal:") {
            continue;
        }

        // Skip garbage labels: Rust types, single-word noise, very short descriptions.
        // These leak in when the agent creates goals from type debug output
        // (e.g., "goal:&Goal", "goal:SymbolId", "goal:&mut Goal").
        let description_raw = meta.label.trim_start_matches("goal:");
        if description_raw.len() < 3
            || description_raw.starts_with('&')
            || description_raw.starts_with('(')
            || description_raw.contains("::")
            || (description_raw.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && description_raw.len() < 15
                && description_raw.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        {
            continue;
        }

        let triples = engine.triples_from(meta.id);
        let mut description = description_raw.to_string();
        let mut status = GoalStatus::Pending;
        // Track the SymbolId of the status object so we pick the most recent one
        // (highest SymbolId = most recently created, since IDs are monotonic).
        let mut status_sym_id: Option<SymbolId> = None;
        let mut priority = 128u8;
        let mut criteria = String::new();
        let mut parent = None;
        let mut children = Vec::new();

        for triple in &triples {
            let obj_label = engine.resolve_label(triple.object);
            if triple.predicate == predicates.has_description {
                description = obj_label.trim_start_matches("desc:").to_string();
            } else if triple.predicate == predicates.has_status {
                // Only accept this status if it's more recent (higher SymbolId)
                // than any previously seen status triple for this goal.
                let dominated = status_sym_id.is_some_and(|prev| triple.object > prev);
                if status_sym_id.is_none() || dominated {
                    let status_str = obj_label.trim_start_matches("status:");
                    status = GoalStatus::from_label(status_str);
                    status_sym_id = Some(triple.object);
                }
            } else if triple.predicate == predicates.has_priority {
                if let Some(p) = obj_label.trim_start_matches("priority:").parse::<u8>().ok() {
                    priority = p;
                }
            } else if triple.predicate == predicates.has_criteria {
                criteria = obj_label.trim_start_matches("criteria:").to_string();
            } else if triple.predicate == predicates.child_goal {
                children.push(triple.object);
            } else if triple.predicate == predicates.parent_goal {
                parent = Some(triple.object);
            }
        }

        goals.push(Goal {
            symbol_id: meta.id,
            description,
            status,
            priority,
            success_criteria: criteria,
            parent,
            children,
            created_at: meta.created_at,
            cycles_worked: 0,
            last_progress_cycle: 0,
        });
    }

    Ok(goals)
}

/// Clear all goals from the in-memory goal list.
///
/// Note: The KG does not currently support triple removal, so goal symbols
/// remain in the graph. The validation filter in `restore_goals()` will skip
/// garbage labels on future restores. Use `--fresh` to start from a clean KG.
pub fn clear_goals(goals: &mut Vec<Goal>) {
    goals.clear();
}

/// Generate sub-goal descriptions from a parent goal's description.
///
/// Splits on commas and "and" to find natural sub-tasks. If no natural split
/// is found, generates generic exploration sub-goals.
pub fn generate_sub_goal_descriptions(description: &str) -> Vec<(String, u8, String)> {
    // Try splitting on commas first.
    let parts: Vec<&str> = description
        .split(',')
        .flat_map(|p| p.split(" and "))
        .map(|p| p.trim())
        .filter(|p| p.len() > 3)
        .collect();

    if parts.len() >= 2 {
        return parts
            .into_iter()
            .enumerate()
            .map(|(i, part)| {
                let priority = 200u8.saturating_sub((i as u8) * 10);
                let criteria = format!("Complete: {part}");
                (part.to_string(), priority, criteria)
            })
            .collect();
    }

    // No natural split — generate generic exploration sub-goals.
    vec![
        (
            format!("Query knowledge about: {}", &description.chars().take(30).collect::<String>()),
            200,
            format!("Find relevant triples for: {description}"),
        ),
        (
            format!("Reason about: {}", &description.chars().take(30).collect::<String>()),
            180,
            format!("Apply reasoning to: {description}"),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_status_roundtrip() {
        let statuses = vec![
            GoalStatus::Pending,
            GoalStatus::Active,
            GoalStatus::Completed,
            GoalStatus::Failed {
                reason: "timeout".into(),
            },
            GoalStatus::Suspended,
        ];
        for status in statuses {
            let label = status.as_label();
            let restored = GoalStatus::from_label(&label);
            assert_eq!(restored, status);
        }
    }

    /// Standalone validation matching `restore_goals` filtering logic.
    fn is_valid_goal_label(label: &str) -> bool {
        let desc = label.trim_start_matches("goal:");
        if desc.len() < 3
            || desc.starts_with('&')
            || desc.starts_with('(')
            || desc.contains("::")
            || (desc.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && desc.len() < 15
                && desc.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        {
            return false;
        }
        true
    }

    #[test]
    fn goal_validation_filters_garbage_labels() {
        // Valid goals
        assert!(is_valid_goal_label("goal:describe the VSA module architecture"));
        assert!(is_valid_goal_label("goal:find all dependencies"));
        assert!(is_valid_goal_label("goal:explore code structure"));

        // Garbage: Rust type names
        assert!(!is_valid_goal_label("goal:&Goal"));
        assert!(!is_valid_goal_label("goal:&mut Goal"));
        assert!(!is_valid_goal_label("goal:SymbolId"));
        assert!(!is_valid_goal_label("goal:GoalStatus"));

        // Garbage: module paths
        assert!(!is_valid_goal_label("goal:std::collections::HashMap"));
        assert!(!is_valid_goal_label("goal:agent::goal::Goal"));

        // Garbage: too short
        assert!(!is_valid_goal_label("goal:ab"));
        assert!(!is_valid_goal_label("goal:x"));

        // Garbage: tuple-like
        assert!(!is_valid_goal_label("goal:(String, u8)"));
    }

    #[test]
    fn clear_goals_empties_list() {
        let mut goals = vec![
            Goal {
                symbol_id: SymbolId::new(1).unwrap(),
                description: "test".into(),
                status: GoalStatus::Active,
                priority: 128,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
            },
        ];
        clear_goals(&mut goals);
        assert!(goals.is_empty());
    }

    #[test]
    fn active_goals_sorted_by_priority() {
        let goals = vec![
            Goal {
                symbol_id: SymbolId::new(1).unwrap(),
                description: "low".into(),
                status: GoalStatus::Active,
                priority: 10,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
            },
            Goal {
                symbol_id: SymbolId::new(2).unwrap(),
                description: "high".into(),
                status: GoalStatus::Active,
                priority: 200,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
            },
            Goal {
                symbol_id: SymbolId::new(3).unwrap(),
                description: "completed".into(),
                status: GoalStatus::Completed,
                priority: 255,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
            },
        ];

        let active = active_goals(&goals);
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].priority, 200);
        assert_eq!(active[1].priority, 10);
    }
}
