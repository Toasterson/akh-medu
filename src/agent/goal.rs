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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Auto-cancelled by the hygiene pass (stale or exceeded child depth).
    Cancelled { reason: String },
    /// Collapsed into a primary goal by the deduplication hygiene pass.
    Duplicate { primary: SymbolId },
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
            Self::Cancelled { reason } => format!("cancelled:{reason}"),
            Self::Duplicate { primary } => format!("duplicate:{}", primary.get()),
        }
    }

    /// Parse from a KG label string.
    ///
    /// Returns `None` for a label that does not match any recognized status
    /// (including a malformed parameterized form, e.g. `"reformulated:abc"`
    /// with a non-numeric id) — callers that trust the input (KG restore,
    /// where a legacy/corrupt label should degrade gracefully rather than
    /// fail the whole restore) should fall back explicitly with
    /// `.unwrap_or(GoalStatus::Pending)`; callers that surface a client's
    /// literal status string (e.g. the `_akh/goal/update` ACP handler) should
    /// propagate `None` as an error instead of silently defaulting.
    pub fn from_label(label: &str) -> Option<Self> {
        if let Some(reason) = label.strip_prefix("failed:") {
            Some(Self::Failed {
                reason: reason.into(),
            })
        } else if let Some(id_str) = label.strip_prefix("reformulated:") {
            if let Ok(id) = id_str.parse::<u64>()
                && let Some(sym) = SymbolId::new(id)
            {
                return Some(Self::Reformulated { replacement: sym });
            }
            None
        } else if let Some(reason) = label.strip_prefix("cancelled:") {
            Some(Self::Cancelled {
                reason: reason.into(),
            })
        } else if let Some(id_str) = label.strip_prefix("duplicate:") {
            if let Ok(id) = id_str.parse::<u64>()
                && let Some(sym) = SymbolId::new(id)
            {
                return Some(Self::Duplicate { primary: sym });
            }
            None
        } else {
            match label {
                "pending" => Some(Self::Pending),
                "active" => Some(Self::Active),
                "completed" => Some(Self::Completed),
                "suspended" => Some(Self::Suspended),
                "proposed" => Some(Self::Proposed),
                "dormant" => Some(Self::Dormant),
                _ => None,
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
            Self::Reformulated { replacement } => write!(f, "reformulated → {replacement}"),
            Self::Cancelled { reason } => write!(f, "cancelled: {reason}"),
            Self::Duplicate { primary } => write!(f, "duplicate → {primary}"),
        }
    }
}

/// How a goal was generated (provenance for autonomous goals).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Auto-generated after consolidation via VSA competence gap detection.
    ///
    /// Created when a knowledge tier's coverage score falls below 0.70.
    /// Goals with this source start as `Proposed` at priority 64 so humans
    /// can review before promoting them.
    VsaCompetenceGap {
        /// Bloom-derived tier: "foundational", "intermediate", or "advanced".
        tier: String,
        /// Label of the entity or area with insufficient coverage.
        area_name: String,
        /// Coverage score that triggered this goal (< 0.70).
        coverage_score: f32,
    },
}

/// Justification for a goal's existence (Phase 11f: AGM belief revision).
///
/// Four variants with entrenchment levels determining retraction order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Estimated effort from CBR (Phase 11g).
    pub estimated_effort: Option<super::resource::EffortEstimate>,
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
                .is_some_and(|g| !matches!(
                    g.status,
                    GoalStatus::Completed
                        | GoalStatus::Failed { .. }
                        | GoalStatus::Cancelled { .. }
                        | GoalStatus::Duplicate { .. }
                ))
        })
    }
}

/// Strip any internal `goal:` / `desc:` prefixes that a caller may have
/// accidentally embedded in the description text. These prefixes belong on
/// the KG label, not inside the human-readable description itself.
///
/// Examples:
/// - `"goal: investigate Tecton"` → `"investigate Tecton"`
/// - `"desc:my objective"` → `"my objective"`
/// - `"investigate: Tecton"` → unchanged (single colon mid-sentence is fine)
pub fn normalize_goal_description(description: &str) -> &str {
    // Each prefix is tried in order; the first match wins.
    for prefix in ["goal: ", "goal:", "desc: ", "desc:"] {
        if let Some(rest) = description.strip_prefix(prefix) {
            return rest.trim_start();
        }
    }
    description.trim_start()
}

/// Truncate `text` at the last sentence boundary (`.`, `?`, `!`, `\n`) that
/// falls within `max_chars`, falling back to the last word boundary (space),
/// then to a hard character cut. Returns the original string when it is short
/// enough to fit without truncation.
fn truncate_at_boundary(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    // Sentence boundary: last `.`, `?`, `!`, or `\n` within [0, max_chars).
    for i in (0..max_chars).rev() {
        if matches!(chars[i], '.' | '?' | '!' | '\n') {
            return chars[..=i].iter().collect();
        }
    }
    // Word boundary: last space within [0, max_chars).
    for i in (0..max_chars).rev() {
        if chars[i] == ' ' {
            return chars[..i].iter().collect();
        }
    }
    // Hard cut at max_chars.
    chars[..max_chars].iter().collect()
}

/// Create a goal and persist it in the knowledge graph.
///
/// A goal is stored as a **single** entity whose label is
/// `goal:<full description>`. No separate `desc:` entity is created.
/// The description is normalized to strip any accidental `goal:` / `desc:`
/// leading prefixes before the label is formed.
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
    let description = normalize_goal_description(description);
    let label = format!("goal:{}", truncate_at_boundary(description, 120));
    let goal_id = engine.resolve_or_create_entity(&label)?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Remove any stale property triples from a previous generation cycle through
    // the engine so SPARQL stays in sync (resolve_or_create_entity reuses the same
    // goal_id for the same label, so old triples must be cleared first).
    for pred in [predicates.has_description, predicates.has_status, predicates.has_priority] {
        let old: Vec<_> = engine
            .knowledge_graph()
            .triples_from(goal_id)
            .into_iter()
            .filter(|t| t.predicate == pred)
            .collect();
        for o in old {
            let _ = engine.remove_triple(o.subject, o.predicate, o.object);
        }
    }

    // Store status.
    let status_sym = engine.resolve_or_create_entity("status:active")?;
    let _ = engine.add_triple(&Triple::new(goal_id, predicates.has_status, status_sym));
    let status_key = format!("goal_status:{}", goal_id.get());
    let _ = engine.store().put_meta(status_key.as_bytes(), b"active");

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
        estimated_effort: None,
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
    // Remove old status triples through the engine so the SPARQL store stays in sync.
    let old_status_triples: Vec<_> = engine
        .knowledge_graph()
        .triples_from(goal.symbol_id)
        .into_iter()
        .filter(|t| t.predicate == predicates.has_status)
        .collect();
    for old in old_status_triples {
        let _ = engine.remove_triple(old.subject, old.predicate, old.object);
    }

    let status_label = format!("status:{}", new_status.as_label());
    let status_sym = engine.resolve_or_create_entity(&status_label)?;
    let _ = engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_status,
        status_sym,
    ));
    let status_key = format!("goal_status:{}", goal.symbol_id.get());
    let _ = engine.store().put_meta(status_key.as_bytes(), new_status.as_label().as_bytes());
    goal.status = new_status;
    Ok(())
}

/// Remove all KG triples for a goal so it is not restored on the next session.
///
/// Removes subject-predicate-object entries for the goal's known predicates.
/// The caller must remove the goal from the in-memory vec separately.
pub fn delete_goal_from_kg(
    engine: &Engine,
    goal_id: SymbolId,
    predicates: &AgentPredicates,
) -> AgentResult<()> {
    // Remove all outgoing triples through the engine so SPARQL stays in sync.
    let subject_triples: Vec<_> = engine.knowledge_graph().triples_from(goal_id);
    for t in subject_triples {
        let _ = engine.remove_triple(t.subject, t.predicate, t.object);
    }
    // Remove back-references (parent→child, etc.) so hierarchical integrity is maintained.
    for pred in [
        predicates.parent_goal,
        predicates.child_goal,
        predicates.blocked_by,
    ] {
        let back_refs: Vec<_> = engine
            .knowledge_graph()
            .triples_to(goal_id)
            .into_iter()
            .filter(|t| t.predicate == pred)
            .collect();
        for t in back_refs {
            let _ = engine.remove_triple(t.subject, t.predicate, t.object);
        }
    }
    Ok(())
}

/// Update a goal's priority in the KG and in-memory struct.
pub fn update_goal_priority(
    engine: &Engine,
    goal: &mut Goal,
    priority: u8,
    predicates: &AgentPredicates,
) -> AgentResult<()> {
    let old: Vec<_> = engine
        .knowledge_graph()
        .triples_from(goal.symbol_id)
        .into_iter()
        .filter(|t| t.predicate == predicates.has_priority)
        .collect();
    for t in old {
        let _ = engine.remove_triple(t.subject, t.predicate, t.object);
    }
    let priority_sym = engine.resolve_or_create_entity(&format!("priority:{priority}"))?;
    let _ = engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_priority,
        priority_sym,
    ));
    goal.priority = priority;
    goal.priority_rationale = None; // clear argumentation override on manual update
    Ok(())
}

/// Update a goal's success criteria in the KG and in-memory struct.
pub fn update_goal_criteria(
    engine: &Engine,
    goal: &mut Goal,
    criteria: &str,
    predicates: &AgentPredicates,
) -> AgentResult<()> {
    let old: Vec<_> = engine
        .knowledge_graph()
        .triples_from(goal.symbol_id)
        .into_iter()
        .filter(|t| t.predicate == predicates.has_criteria)
        .collect();
    for t in old {
        let _ = engine.remove_triple(t.subject, t.predicate, t.object);
    }
    let criteria_sym = engine.resolve_or_create_entity(&format!("criteria:{criteria}"))?;
    let _ = engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_criteria,
        criteria_sym,
    ));
    goal.success_criteria = criteria.into();
    Ok(())
}

/// Update a goal's description (goal_text) by writing a `has_description`
/// triple to the KG.
///
/// The goal's Entity label (e.g. `goal:investigate Tecton`) is immutable —
/// it was set at creation time and is used as the lookup key by
/// `restore_goals`.  However, `restore_goals` checks for a
/// `has_description` triple first and uses that over the label-derived
/// description when present (lines 587–591).  So we persist the updated
/// description as a `desc:{text}` entity linked via `has_description`,
/// which the restore path will pick up on the next read.
pub fn update_goal_description(
    engine: &Engine,
    goal: &mut Goal,
    description: &str,
    predicates: &AgentPredicates,
) -> AgentResult<()> {
    let normalized = normalize_goal_description(description);
    // Remove any existing has_description triples for this goal. Propagate
    // failures instead of swallowing them: a partially-removed old
    // description followed by a silently-skipped new one would leave the
    // goal in an inconsistent state that `restore_goals` can't distinguish
    // from success.
    let old: Vec<_> = engine
        .knowledge_graph()
        .triples_from(goal.symbol_id)
        .into_iter()
        .filter(|t| t.predicate == predicates.has_description)
        .collect();
    for t in old {
        engine.remove_triple(t.subject, t.predicate, t.object)?;
    }
    // Store the new description as a desc: entity.
    let desc_sym = engine.resolve_or_create_entity(&format!("desc:{normalized}"))?;
    engine.add_triple(&Triple::new(
        goal.symbol_id,
        predicates.has_description,
        desc_sym,
    ))?;
    goal.description = normalized.to_string();
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
    active.sort_by_key(|g| std::cmp::Reverse(g.computed_priority()));
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
        // Normalize the label-derived description (strip stale goal:/desc: prefixes).
        let mut description = normalize_goal_description(description_raw).to_string();
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
                // Backward compat: old sessions stored the full description in a
                // separate `desc:` entity; use that when available.
                let raw = obj_label.trim_start_matches("desc:");
                description = normalize_goal_description(raw).to_string();
            } else if triple.predicate == predicates.has_status {
                // Only accept this status if it's more recent (higher SymbolId)
                // than any previously seen status triple for this goal.
                let dominated = status_sym_id.is_some_and(|prev| triple.object > prev);
                if status_sym_id.is_none() || dominated {
                    let status_str = obj_label.trim_start_matches("status:");
                    // Lenient: a legacy/corrupt label from an older session
                    // should degrade to Pending rather than fail the whole
                    // restore. Unlike the client-facing update path, there is
                    // no caller here to report an error to.
                    status = GoalStatus::from_label(status_str).unwrap_or(GoalStatus::Pending);
                    status_sym_id = Some(triple.object);
                }
            } else if triple.predicate == predicates.has_priority {
                if let Ok(p) = obj_label.trim_start_matches("priority:").parse::<u8>() {
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

        // If the KG didn't supply a status (e.g. SPARQL persistence is disabled
        // and the store was recreated), fall back to the directly-persisted key.
        if status_sym_id.is_none() {
            let key = format!("goal_status:{}", meta.id.get());
            if let Ok(Some(bytes)) = engine.store().get_meta(key.as_bytes()) {
                if let Ok(label) = std::str::from_utf8(&bytes) {
                    // Lenient fallback, same reasoning as above.
                    status = GoalStatus::from_label(label).unwrap_or(GoalStatus::Pending);
                }
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
            estimated_effort: None,
        });
    }

    Ok(goals)
}

/// Persist the current goal list to the durable store as a serialized blob.
///
/// This is the primary goal-persistence mechanism for restarts without
/// `sparql-persist`. The blob is stored under `agent:goals` in redb and
/// takes priority over KG-triple restoration in `restore_goals_from_store`.
pub fn persist_goals(goals: &[Goal], engine: &Engine) -> AgentResult<()> {
    use super::error::AgentError;
    let bytes = bincode::serialize(goals).map_err(|e| AgentError::ConsolidationFailed {
        message: format!("failed to serialize goals: {e}"),
    })?;
    engine
        .store()
        .put_meta(b"agent:goals", &bytes)
        .map_err(|e| AgentError::ConsolidationFailed {
            message: format!("failed to write goals to store: {e}"),
        })?;
    Ok(())
}

/// Try to restore goals from the durable store blob written by `persist_goals`.
///
/// Returns `None` if no persisted blob exists (fresh engine or memory-only).
pub fn restore_goals_from_store(engine: &Engine) -> Option<Vec<Goal>> {
    let bytes = engine.store().get_meta(b"agent:goals").ok()??;
    bincode::deserialize::<Vec<Goal>>(&bytes).ok()
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
    fn truncate_at_boundary_short_string() {
        assert_eq!(truncate_at_boundary("Hello world", 60), "Hello world");
    }

    #[test]
    fn truncate_at_boundary_sentence_end() {
        let s = "Find all dependencies. Then analyse them in detail and more.";
        // First sentence ends at index 22 (the dot), within 40 chars.
        let got = truncate_at_boundary(s, 40);
        assert!(got.ends_with('.'), "expected sentence cut, got: {got:?}");
        assert!(got.len() <= 40);
    }

    #[test]
    fn truncate_at_boundary_word_boundary() {
        let s = "Find relationships without any sentence terminators here extra";
        let got = truncate_at_boundary(s, 40);
        // Should end at a space (word boundary).
        assert!(!got.ends_with(' '));
        assert!(got.len() < 40);
    }

    #[test]
    fn truncate_at_boundary_hard_cut() {
        let s = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz";
        let got = truncate_at_boundary(s, 10);
        assert_eq!(got.len(), 10);
    }

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
            GoalStatus::Cancelled {
                reason: "idle for 20 cycles".into(),
            },
            GoalStatus::Duplicate {
                primary: SymbolId::new(7).unwrap(),
            },
        ];
        for status in statuses {
            let label = status.as_label();
            let restored = GoalStatus::from_label(&label)
                .unwrap_or_else(|| panic!("from_label should recognize its own as_label output: {label}"));
            assert_eq!(restored, status);
        }
    }

    #[test]
    fn from_label_rejects_unknown_status() {
        assert_eq!(GoalStatus::from_label("cancled"), None);
        assert_eq!(GoalStatus::from_label("bogus"), None);
        assert_eq!(GoalStatus::from_label(""), None);
        // Malformed parameterized forms are unknown too, not silently Pending.
        assert_eq!(GoalStatus::from_label("reformulated:not-a-number"), None);
        assert_eq!(GoalStatus::from_label("duplicate:not-a-number"), None);
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
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
            estimated_effort: None,
        }];

        assert!(!goals[0].is_blocked(&goals));
    }

    #[test]
    fn status_reformulated_roundtrip() {
        let sym = SymbolId::new(42).unwrap();
        let status = GoalStatus::Reformulated { replacement: sym };
        let label = status.as_label();
        assert_eq!(label, "reformulated:42");
        let restored = GoalStatus::from_label(&label).expect("reformulated:42 should parse");
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

    #[test]
    fn normalize_goal_description_strips_prefixes() {
        // Leading goal: prefix is stripped.
        assert_eq!(
            normalize_goal_description("goal:investigate Tecton"),
            "investigate Tecton"
        );
        assert_eq!(
            normalize_goal_description("goal: investigate Tecton"),
            "investigate Tecton"
        );
        // Leading desc: prefix is stripped.
        assert_eq!(
            normalize_goal_description("desc:my objective"),
            "my objective"
        );
        assert_eq!(
            normalize_goal_description("desc: my objective"),
            "my objective"
        );
        // A colon mid-sentence is NOT stripped.
        assert_eq!(
            normalize_goal_description("investigate: I am Tecton"),
            "investigate: I am Tecton"
        );
        // Already clean descriptions pass through unchanged.
        assert_eq!(
            normalize_goal_description("find all dependencies"),
            "find all dependencies"
        );
        // Empty-after-strip falls back to empty.
        assert_eq!(normalize_goal_description("goal:"), "");
        // Nested prefix: only the outermost is stripped.
        assert_eq!(
            normalize_goal_description("goal:desc:inner"),
            "desc:inner"
        );
    }

    fn test_engine() -> crate::engine::Engine {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn create_goal_label_uses_full_description() {
        let engine = test_engine();
        let preds = crate::agent::agent::AgentPredicates::init(&engine).unwrap();

        let description = "investigate: I am Tecton, the AI agent who exists to reason";
        let goal = create_goal(&engine, description, 128, "done", &preds).unwrap();

        // The goal entity label must be goal:<full description> with no truncation.
        let label = engine.resolve_label(goal.symbol_id);
        assert_eq!(label, format!("goal:{description}"));

        // The restored description must match exactly.
        assert_eq!(goal.description, description);

        // No separate desc: entity should exist for this goal.
        let all: Vec<_> = engine
            .all_symbols()
            .into_iter()
            .filter(|m| m.label.starts_with("desc:"))
            .collect();
        assert!(
            all.is_empty(),
            "no desc: entities should be created; found: {all:?}"
        );
    }

    #[test]
    fn create_goal_normalizes_prefixed_description() {
        let engine = test_engine();
        let preds = crate::agent::agent::AgentPredicates::init(&engine).unwrap();

        // Caller accidentally passes a description already prefixed with goal:.
        let raw = "goal: investigate who Tecton is";
        let goal = create_goal(&engine, raw, 128, "done", &preds).unwrap();

        // Label should not double-prefix.
        let label = engine.resolve_label(goal.symbol_id);
        assert_eq!(label, "goal:investigate who Tecton is");
        assert_eq!(goal.description, "investigate who Tecton is");
    }
}
