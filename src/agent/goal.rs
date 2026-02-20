//! Goal system: hierarchical goals stored as KG entities.
//!
//! Goals are Entity symbols in the knowledge graph with well-known
//! relation predicates for status, priority, criteria, and hierarchy.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::agent::AgentPredicates;
use super::error::AgentResult;
use super::priority_reasoning::PriorityVerdict;

/// Status of a goal.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalStatus {
    Pending,
    Active,
    Completed,
    Failed { reason: String },
    Suspended,
    /// Proposed by autonomous goal generation but not yet activated.
    Proposed,
    /// Dormant: low-priority or infeasible, kept for opportunity detection.
    Dormant,
    /// Reformulated: replaced by a simpler version (Phase 11f).
    Reformulated { replacement: SymbolId },
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
            Self::Proposed => "proposed".into(),
            Self::Dormant => "dormant".into(),
            Self::Reformulated { replacement } => format!("reformulated:{}", replacement.get()),
        }
    }

    /// Parse from a KG label string.
    pub fn from_label(label: &str) -> Self {
        if let Some(reason) = label.strip_prefix("failed:") {
            Self::Failed {
                reason: reason.into(),
            }
        } else if let Some(id_str) = label.strip_prefix("reformulated:") {
            if let Ok(id) = id_str.parse::<u64>() {
                if let Some(sym) = SymbolId::new(id) {
                    return Self::Reformulated { replacement: sym };
                }
            }
            Self::Pending
        } else {
            match label {
                "pending" => Self::Pending,
                "active" => Self::Active,
                "completed" => Self::Completed,
                "suspended" => Self::Suspended,
                "proposed" => Self::Proposed,
                "dormant" => Self::Dormant,
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
            Self::Proposed => write!(f, "proposed"),
            Self::Dormant => write!(f, "dormant"),
            Self::Reformulated { replacement } => {
                write!(f, "reformulated → {replacement}")
            }
        }
    }
}

/// How a goal was generated (provenance for autonomous goals).
#[derive(Debug, Clone)]
pub enum GoalSource {
    /// Generated from knowledge gap detection.
    GapDetection {
        gap_entity: SymbolId,
        gap_kind: String,
        severity: f32,
    },
    /// Generated from contradiction detection.
    ContradictionDetected {
        existing: Triple,
        incoming: Triple,
    },
    /// Generated from opportunity detection (reactivating a dormant/failed goal).
    OpportunityDetected {
        reactivated_goal: SymbolId,
        newly_satisfied: String,
    },
    /// Generated because a drive exceeded its threshold.
    DriveExceeded {
        drive: String,
        strength: f32,
    },
    /// Generated because the OODA loop detected a decision impasse.
    ImpasseDetected {
        goal_id: SymbolId,
        impasse_kind: String,
    },
    /// Generated from a reflection insight.
    ReflectionInsight {
        insight: String,
    },
    /// Generated from a world-monitoring watch firing or expectation discrepancy.
    WorldChange {
        watch_name: String,
        discrepancy: String,
    },
}

/// Justification for a goal's existence (Phase 11f: AGM belief revision).
///
/// Four variants with entrenchment levels determining retraction order.
#[derive(Debug, Clone)]
pub enum GoalJustification {
    /// Explicitly requested by the user — never auto-suspended (entrenchment 3).
    UserRequested,
    /// Decomposed from a parent goal — cascades when parent abandoned (entrenchment 2).
    DecomposedFrom { parent: SymbolId },
    /// Inferred from KG state — invalidated when supporting triples retracted (entrenchment 1).
    InferredFromKG { supporting: Vec<SymbolId> },
    /// Default assumption — first to be retracted (entrenchment 0).
    DefaultAssumption { rationale: String },
}

impl GoalJustification {
    /// Entrenchment rank (higher = harder to retract).
    pub fn entrenchment(&self) -> u8 {
        match self {
            Self::UserRequested => 3,
            Self::DecomposedFrom { .. } => 2,
            Self::InferredFromKG { .. } => 1,
            Self::DefaultAssumption { .. } => 0,
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
    /// How this goal was generated (None for externally-created goals).
    pub source: Option<GoalSource>,
    /// Goals that must complete before this goal can be worked on (HTN dependencies).
    pub blocked_by: Vec<SymbolId>,
    /// Argumentation-based priority verdict (Phase 11c).
    pub priority_rationale: Option<PriorityVerdict>,
    /// Why this goal exists — for AGM belief revision (Phase 11f).
    /// `None` for legacy/restored goals.
    pub justification: Option<GoalJustification>,
    /// If this goal was created by reformulating another goal (Phase 11f).
    pub reformulated_from: Option<SymbolId>,
}

impl Goal {
    /// Effective priority: uses argumentation verdict if available, else raw priority.
    pub fn computed_priority(&self) -> u8 {
        self.priority_rationale
            .as_ref()
            .map(|v| v.computed_priority)
            .unwrap_or(self.priority)
    }

    /// Whether this goal is stalled: has been worked on for `threshold` cycles
    /// since it last made progress.
    pub fn is_stalled(&self, current_cycle: u64, threshold: u32) -> bool {
        self.cycles_worked >= threshold
            && current_cycle.saturating_sub(self.last_progress_cycle) >= threshold as u64
    }

    /// Whether this goal is blocked by uncompleted dependencies.
    ///
    /// A goal is blocked if any of its `blocked_by` entries refer to goals
    /// that are not yet `Completed` or `Failed`.
    pub fn is_blocked(&self, all_goals: &[Goal]) -> bool {
        if self.blocked_by.is_empty() {
            return false;
        }
        self.blocked_by.iter().any(|blocker_id| {
            all_goals
                .iter()
                .find(|g| g.symbol_id == *blocker_id)
                .is_some_and(|g| !matches!(g.status, GoalStatus::Completed | GoalStatus::Failed { .. }))
        })
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
    let desc_sym = engine.resolve_or_create_entity(&format!("desc:{description}"))?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_description, desc_sym));

    // Store status.
    let status_sym = engine.resolve_or_create_entity("status:active")?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_status, status_sym));

    // Store priority.
    let priority_sym = engine.resolve_or_create_entity(&format!("priority:{priority}"))?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_priority, priority_sym));

    // Store criteria.
    let criteria_sym = engine.resolve_or_create_entity(&format!("criteria:{criteria}"))?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_criteria, criteria_sym));

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
        source: None,
        blocked_by: Vec::new(),
        priority_rationale: None,
        justification: None,
        reformulated_from: None,
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
    let status_sym = engine.resolve_or_create_entity(&status_label)?;
    let _ = engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_status,
        status_sym,
    ));
    goal.status = new_status;
    Ok(())
}

/// Filter active goals, sorted by computed priority (highest first).
///
/// Uses argumentation-based priority when available, else raw priority.
pub fn active_goals(goals: &[Goal]) -> Vec<&Goal> {
    let mut active: Vec<&Goal> = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .collect();
    active.sort_by(|a, b| b.computed_priority().cmp(&a.computed_priority()));
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
            || (description_raw
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
                && description_raw.len() < 15
                && description_raw
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_uppercase()))
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
        let mut blocked_by = Vec::new();

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
            } else if triple.predicate == predicates.blocked_by {
                blocked_by.push(triple.object);
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
            source: None,
            blocked_by,
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
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
            format!(
                "Query knowledge about: {}",
                &description.chars().take(30).collect::<String>()
            ),
            200,
            format!("Find relevant triples for: {description}"),
        ),
        (
            format!(
                "Reason about: {}",
                &description.chars().take(30).collect::<String>()
            ),
            180,
            format!("Apply reasoning to: {description}"),
        ),
    ]
}

/// Reformulate a goal: create a replacement with relaxed criteria.
///
/// The new goal inherits the original's priority and parent. The original
/// goal's status is set to `Reformulated { replacement: new_id }`.
/// The replacement has `reformulated_from` pointing to the original and
/// a `DecomposedFrom` justification.
pub fn reformulate_goal(
    engine: &Engine,
    original: &mut Goal,
    relaxed_criteria: &str,
    predicates: &AgentPredicates,
) -> AgentResult<Goal> {
    let desc = format!("(reformulated) {}", original.description);
    let mut replacement = create_goal(engine, &desc, original.priority, relaxed_criteria, predicates)?;
    replacement.parent = original.parent;
    replacement.reformulated_from = Some(original.symbol_id);
    replacement.justification = Some(GoalJustification::DecomposedFrom {
        parent: original.symbol_id,
    });

    // Mark original as reformulated.
    update_goal_status(
        engine,
        original,
        GoalStatus::Reformulated {
            replacement: replacement.symbol_id,
        },
        predicates,
    )?;

    Ok(replacement)
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
            GoalStatus::Proposed,
            GoalStatus::Dormant,
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
        assert!(is_valid_goal_label(
            "goal:describe the VSA module architecture"
        ));
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
        let mut goals = vec![Goal {
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
            source: None,
            blocked_by: Vec::new(),
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
        }];
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
                source: None,
                blocked_by: Vec::new(),
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
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
                source: None,
                blocked_by: Vec::new(),
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
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
                source: None,
                blocked_by: Vec::new(),
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
            },
        ];

        let active = active_goals(&goals);
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].priority, 200);
        assert_eq!(active[1].priority, 10);
    }

    #[test]
    fn is_blocked_with_active_blocker() {
        let goals = vec![
            Goal {
                symbol_id: SymbolId::new(1).unwrap(),
                description: "blocker".into(),
                status: GoalStatus::Active,
                priority: 200,
                success_criteria: String::new(),
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
            },
            Goal {
                symbol_id: SymbolId::new(2).unwrap(),
                description: "blocked".into(),
                status: GoalStatus::Active,
                priority: 100,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
                source: None,
                blocked_by: vec![SymbolId::new(1).unwrap()],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
            },
        ];

        // Goal 2 is blocked because goal 1 is still Active.
        assert!(goals[1].is_blocked(&goals));
        // Goal 1 is not blocked.
        assert!(!goals[0].is_blocked(&goals));
    }

    #[test]
    fn is_blocked_cleared_when_blocker_completed() {
        let goals = vec![
            Goal {
                symbol_id: SymbolId::new(1).unwrap(),
                description: "blocker".into(),
                status: GoalStatus::Completed,
                priority: 200,
                success_criteria: String::new(),
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
            },
            Goal {
                symbol_id: SymbolId::new(2).unwrap(),
                description: "blocked".into(),
                status: GoalStatus::Active,
                priority: 100,
                success_criteria: String::new(),
                parent: None,
                children: Vec::new(),
                created_at: 0,
                cycles_worked: 0,
                last_progress_cycle: 0,
                source: None,
                blocked_by: vec![SymbolId::new(1).unwrap()],
                priority_rationale: None,
                justification: None,
                reformulated_from: None,
            },
        ];

        // Goal 2 is NOT blocked because goal 1 is Completed.
        assert!(!goals[1].is_blocked(&goals));
    }

    #[test]
    fn is_blocked_empty_blocked_by() {
        let goals = vec![Goal {
            symbol_id: SymbolId::new(1).unwrap(),
            description: "free".into(),
            status: GoalStatus::Active,
            priority: 128,
            success_criteria: String::new(),
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
        }];

        assert!(!goals[0].is_blocked(&goals));
    }

    #[test]
    fn status_reformulated_roundtrip() {
        let sym = SymbolId::new(42).unwrap();
        let status = GoalStatus::Reformulated { replacement: sym };
        let label = status.as_label();
        assert_eq!(label, "reformulated:42");
        let restored = GoalStatus::from_label(&label);
        assert_eq!(restored, status);
    }

    #[test]
    fn justification_entrenchment_ordering() {
        let user = GoalJustification::UserRequested;
        let decomp = GoalJustification::DecomposedFrom {
            parent: SymbolId::new(1).unwrap(),
        };
        let inferred = GoalJustification::InferredFromKG {
            supporting: vec![],
        };
        let default = GoalJustification::DefaultAssumption {
            rationale: "test".into(),
        };
        assert!(user.entrenchment() > decomp.entrenchment());
        assert!(decomp.entrenchment() > inferred.entrenchment());
        assert!(inferred.entrenchment() > default.entrenchment());
    }
}
