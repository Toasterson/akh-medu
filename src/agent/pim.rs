//! Personal Information Management (PIM) — Phase 13e.
//!
//! GTD workflow, Eisenhower matrix, PARA categorization, task dependency DAG,
//! recurring tasks, and weekly review. PIM tasks are an **overlay on existing
//! Goals**, not a separate entity type.  `PimMetadata` adds GTD state, quadrant,
//! PARA category, contexts, energy, recurrence, and deadlines to existing `Goal`
//! entities via KG predicates in the `pim:` namespace.

use std::collections::HashMap;
use std::fmt;

use miette::Diagnostic;
use petgraph::graph::{DiGraph, NodeIndex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::goal::{Goal, GoalStatus};
use super::reflect::Adjustment;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the PIM subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum PimError {
    #[error("PIM task not found: {goal_id}")]
    #[diagnostic(
        code(akh::agent::pim::task_not_found),
        help("Use `pim add` to add PIM metadata to an existing goal.")
    )]
    TaskNotFound { goal_id: u64 },

    #[error("invalid GTD transition from {from} to {to}")]
    #[diagnostic(
        code(akh::agent::pim::invalid_transition),
        help(
            "Valid transitions: inbox→next|waiting|someday|reference|done, \
             next|waiting→done|someday, someday→next|reference|done"
        )
    )]
    InvalidTransition { from: String, to: String },

    #[error("dependency cycle detected involving task {task_id}")]
    #[diagnostic(
        code(akh::agent::pim::cycle_detected),
        help("Task dependency DAG must be acyclic.")
    )]
    CycleDetected { task_id: u64 },

    #[error("recurrence parse error: {message}")]
    #[diagnostic(
        code(akh::agent::pim::recurrence_parse),
        help("Valid: daily, weekly:mon,wed,fri, monthly:15, yearly:3-21, every:3d")
    )]
    RecurrenceParse { message: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::pim::engine),
        help("Engine-level error during PIM operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for PimError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type PimResult<T> = std::result::Result<T, PimError>;

// ═══════════════════════════════════════════════════════════════════════
// Enums
// ═══════════════════════════════════════════════════════════════════════

/// GTD (Getting Things Done) workflow state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GtdState {
    /// Newly captured, not yet processed.
    Inbox,
    /// Actionable, should be done ASAP.
    Next,
    /// Waiting on external input.
    Waiting,
    /// Deferred — may act on later.
    Someday,
    /// Reference material, not actionable.
    Reference,
    /// Completed.
    Done,
}

impl GtdState {
    /// States reachable from `self`.
    pub fn valid_transitions(&self) -> &[GtdState] {
        match self {
            Self::Inbox => &[
                Self::Next,
                Self::Waiting,
                Self::Someday,
                Self::Reference,
                Self::Done,
            ],
            Self::Next => &[Self::Done, Self::Someday, Self::Waiting],
            Self::Waiting => &[Self::Done, Self::Someday, Self::Next],
            Self::Someday => &[Self::Next, Self::Reference, Self::Done],
            Self::Reference => &[Self::Someday, Self::Done],
            Self::Done => &[], // terminal
        }
    }

    /// Whether `target` is a legal successor.
    pub fn can_transition_to(&self, target: GtdState) -> bool {
        self.valid_transitions().contains(&target)
    }

    /// Serialize to a short label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Inbox => "inbox",
            Self::Next => "next",
            Self::Waiting => "waiting",
            Self::Someday => "someday",
            Self::Reference => "reference",
            Self::Done => "done",
        }
    }

    /// Parse from label (case-insensitive).
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "inbox" => Some(Self::Inbox),
            "next" => Some(Self::Next),
            "waiting" => Some(Self::Waiting),
            "someday" => Some(Self::Someday),
            "reference" | "ref" => Some(Self::Reference),
            "done" => Some(Self::Done),
            _ => None,
        }
    }
}

impl fmt::Display for GtdState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// Eisenhower matrix quadrant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EisenhowerQuadrant {
    /// Urgent + important — do immediately.
    Do,
    /// Important but not urgent — schedule.
    Schedule,
    /// Urgent but not important — delegate.
    Delegate,
    /// Neither — eliminate.
    Eliminate,
}

impl EisenhowerQuadrant {
    /// Classify from urgency and importance scores (0.0–1.0).
    pub fn classify(urgency: f32, importance: f32) -> Self {
        match (urgency >= 0.5, importance >= 0.5) {
            (true, true) => Self::Do,
            (false, true) => Self::Schedule,
            (true, false) => Self::Delegate,
            (false, false) => Self::Eliminate,
        }
    }

    /// Priority bonus applied to goal priority for ordering.
    pub fn priority_bonus(&self) -> i16 {
        match self {
            Self::Do => 40,
            Self::Schedule => 20,
            Self::Delegate => -10,
            Self::Eliminate => -30,
        }
    }

    /// Serialize to label.
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Do => "do",
            Self::Schedule => "schedule",
            Self::Delegate => "delegate",
            Self::Eliminate => "eliminate",
        }
    }

    /// Parse from label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "do" => Some(Self::Do),
            "schedule" => Some(Self::Schedule),
            "delegate" => Some(Self::Delegate),
            "eliminate" => Some(Self::Eliminate),
            _ => None,
        }
    }
}

impl fmt::Display for EisenhowerQuadrant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// PARA categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParaCategory {
    /// Active project with a deadline.
    Project,
    /// Ongoing area of responsibility.
    Area,
    /// Reference material.
    Resource,
    /// Inactive / archived.
    Archive,
}

impl ParaCategory {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Project => "project",
            Self::Area => "area",
            Self::Resource => "resource",
            Self::Archive => "archive",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "project" => Some(Self::Project),
            "area" => Some(Self::Area),
            "resource" => Some(Self::Resource),
            "archive" => Some(Self::Archive),
            _ => None,
        }
    }
}

impl fmt::Display for ParaCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// GTD context — where/how the task can be done.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PimContext(pub String);

impl PimContext {
    pub fn home() -> Self {
        Self("home".into())
    }
    pub fn office() -> Self {
        Self("office".into())
    }
    pub fn computer() -> Self {
        Self("computer".into())
    }
    pub fn phone() -> Self {
        Self("phone".into())
    }
    pub fn errands() -> Self {
        Self("errands".into())
    }
    pub fn anywhere() -> Self {
        Self("anywhere".into())
    }
}

impl fmt::Display for PimContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Energy level required for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnergyLevel {
    Low,
    Medium,
    High,
}

impl EnergyLevel {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" | "med" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

impl fmt::Display for EnergyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// Recurrence pattern (RRULE-lite).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Recurrence {
    /// Every day.
    Daily,
    /// Specific days of week (0=Mon, 6=Sun).
    Weekly(Vec<u8>),
    /// Day of month (1–31).
    Monthly(u8),
    /// Month (1–12) and day (1–31).
    Yearly(u8, u8),
    /// Every N days.
    EveryNDays(u32),
}

impl Recurrence {
    /// Parse from a string label.
    ///
    /// Formats:
    /// - `"daily"`
    /// - `"weekly:mon,wed,fri"`
    /// - `"monthly:15"`
    /// - `"yearly:3-21"` (March 21)
    /// - `"every:3d"`
    pub fn parse(s: &str) -> PimResult<Self> {
        let lower = s.to_lowercase();
        if lower == "daily" {
            return Ok(Self::Daily);
        }
        if let Some(rest) = lower.strip_prefix("weekly:") {
            let days: Result<Vec<u8>, _> = rest
                .split(',')
                .map(|d| match d.trim() {
                    "mon" => Ok(0),
                    "tue" => Ok(1),
                    "wed" => Ok(2),
                    "thu" => Ok(3),
                    "fri" => Ok(4),
                    "sat" => Ok(5),
                    "sun" => Ok(6),
                    other => Err(PimError::RecurrenceParse {
                        message: format!("unknown day: {other}"),
                    }),
                })
                .collect();
            return Ok(Self::Weekly(days?));
        }
        if let Some(rest) = lower.strip_prefix("monthly:") {
            let day: u8 = rest.trim().parse().map_err(|_| PimError::RecurrenceParse {
                message: format!("invalid day of month: {rest}"),
            })?;
            if !(1..=31).contains(&day) {
                return Err(PimError::RecurrenceParse {
                    message: format!("day of month out of range: {day}"),
                });
            }
            return Ok(Self::Monthly(day));
        }
        if let Some(rest) = lower.strip_prefix("yearly:") {
            let parts: Vec<&str> = rest.trim().split('-').collect();
            if parts.len() != 2 {
                return Err(PimError::RecurrenceParse {
                    message: format!("yearly format: month-day, got: {rest}"),
                });
            }
            let month: u8 = parts[0].parse().map_err(|_| PimError::RecurrenceParse {
                message: format!("invalid month: {}", parts[0]),
            })?;
            let day: u8 = parts[1].parse().map_err(|_| PimError::RecurrenceParse {
                message: format!("invalid day: {}", parts[1]),
            })?;
            if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
                return Err(PimError::RecurrenceParse {
                    message: format!("month/day out of range: {month}-{day}"),
                });
            }
            return Ok(Self::Yearly(month, day));
        }
        if let Some(rest) = lower.strip_prefix("every:") {
            let rest = rest.trim();
            let n_str = rest.strip_suffix('d').unwrap_or(rest);
            let n: u32 = n_str.parse().map_err(|_| PimError::RecurrenceParse {
                message: format!("invalid day count: {n_str}"),
            })?;
            if n == 0 {
                return Err(PimError::RecurrenceParse {
                    message: "every:0d is not valid".into(),
                });
            }
            return Ok(Self::EveryNDays(n));
        }
        Err(PimError::RecurrenceParse {
            message: format!("unrecognized recurrence pattern: {s}"),
        })
    }

    /// Compute the next occurrence after `after_ts` (unix timestamp in seconds).
    pub fn next_occurrence(&self, after_ts: u64) -> u64 {
        const DAY_SECS: u64 = 86_400;
        match self {
            Self::Daily => after_ts + DAY_SECS,
            Self::EveryNDays(n) => after_ts + (*n as u64) * DAY_SECS,
            Self::Weekly(days) => {
                // Find next matching weekday.
                // Approximate: days since epoch mod 7 (epoch was Thursday = 3).
                let current_dow = ((after_ts / DAY_SECS) + 3) % 7; // 0=Mon
                let mut min_delta = u64::MAX;
                for &d in days {
                    let d = d as u64;
                    let delta = if d > current_dow {
                        d - current_dow
                    } else {
                        7 - current_dow + d
                    };
                    // At least 1 day in the future.
                    let delta = if delta == 0 { 7 } else { delta };
                    if delta < min_delta {
                        min_delta = delta;
                    }
                }
                if min_delta == u64::MAX {
                    min_delta = 7;
                }
                after_ts + min_delta * DAY_SECS
            }
            Self::Monthly(day) => {
                // Approximate: add ~30 days, snap to target day.
                after_ts + 30 * DAY_SECS + ((*day as u64).saturating_sub(1)) * DAY_SECS
            }
            Self::Yearly(month, day) => {
                // Approximate: add ~365 days.
                let _ = (month, day); // used for pattern; simplified calculation
                after_ts + 365 * DAY_SECS
            }
        }
    }

    /// Serialize to label.
    pub fn as_label(&self) -> String {
        match self {
            Self::Daily => "daily".into(),
            Self::Weekly(days) => {
                let names: Vec<&str> = days
                    .iter()
                    .map(|d| match d {
                        0 => "mon",
                        1 => "tue",
                        2 => "wed",
                        3 => "thu",
                        4 => "fri",
                        5 => "sat",
                        6 => "sun",
                        _ => "?",
                    })
                    .collect();
                format!("weekly:{}", names.join(","))
            }
            Self::Monthly(day) => format!("monthly:{day}"),
            Self::Yearly(month, day) => format!("yearly:{month}-{day}"),
            Self::EveryNDays(n) => format!("every:{n}d"),
        }
    }

    /// Parse from label (alias for `parse`).
    pub fn from_label(s: &str) -> PimResult<Self> {
        Self::parse(s)
    }
}

impl fmt::Display for Recurrence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_label())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PimMetadata
// ═══════════════════════════════════════════════════════════════════════

/// PIM overlay metadata attached to an existing Goal entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimMetadata {
    /// Goal symbol this metadata annotates.
    pub goal_id: SymbolId,
    /// GTD workflow state.
    pub gtd_state: GtdState,
    /// Urgency score (0.0–1.0).
    pub urgency: f32,
    /// Importance score (0.0–1.0).
    pub importance: f32,
    /// Eisenhower quadrant (derived from urgency + importance).
    pub quadrant: EisenhowerQuadrant,
    /// PARA category.
    pub para: Option<ParaCategory>,
    /// GTD contexts (where/how the task can be done).
    pub contexts: Vec<PimContext>,
    /// Energy level required.
    pub energy: Option<EnergyLevel>,
    /// Estimated time in minutes.
    pub time_estimate_minutes: Option<u32>,
    /// Deadline (unix timestamp).
    pub deadline: Option<u64>,
    /// Recurrence pattern.
    pub recurrence: Option<Recurrence>,
    /// Next due date (unix timestamp).
    pub next_due: Option<u64>,
    /// When last completed (unix timestamp).
    pub last_completed: Option<u64>,
}

// ═══════════════════════════════════════════════════════════════════════
// PimPredicates — 14 well-known relations in the `pim:` namespace
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG predicates for PIM metadata.
#[derive(Debug, Clone)]
pub struct PimPredicates {
    pub gtd_state: SymbolId,
    pub context: SymbolId,
    pub energy: SymbolId,
    pub time_estimate: SymbolId,
    pub urgency: SymbolId,
    pub importance: SymbolId,
    pub para_category: SymbolId,
    pub deadline: SymbolId,
    pub quadrant: SymbolId,
    pub blocked_by: SymbolId,
    pub blocks: SymbolId,
    pub recurrence: SymbolId,
    pub next_due: SymbolId,
    pub last_done: SymbolId,
}

impl PimPredicates {
    /// Resolve or create all PIM predicates.
    pub fn init(engine: &Engine) -> PimResult<Self> {
        Ok(Self {
            gtd_state: engine.resolve_or_create_relation("pim:gtd-state")?,
            context: engine.resolve_or_create_relation("pim:context")?,
            energy: engine.resolve_or_create_relation("pim:energy")?,
            time_estimate: engine.resolve_or_create_relation("pim:time-estimate")?,
            urgency: engine.resolve_or_create_relation("pim:urgency")?,
            importance: engine.resolve_or_create_relation("pim:importance")?,
            para_category: engine.resolve_or_create_relation("pim:para-category")?,
            deadline: engine.resolve_or_create_relation("pim:deadline")?,
            quadrant: engine.resolve_or_create_relation("pim:standard")?,
            blocked_by: engine.resolve_or_create_relation("pim:blocked-by")?,
            blocks: engine.resolve_or_create_relation("pim:blocks")?,
            recurrence: engine.resolve_or_create_relation("pim:recurrence")?,
            next_due: engine.resolve_or_create_relation("pim:next-due")?,
            last_done: engine.resolve_or_create_relation("pim:last-done")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PimRoleVectors — VSA role vectors for priority encoding
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic role hypervectors for encoding PIM priority profiles.
pub struct PimRoleVectors {
    pub urgency: HyperVec,
    pub importance: HyperVec,
    pub energy: HyperVec,
    pub context: HyperVec,
    pub deadline_proximity: HyperVec,
    pub gtd_state: HyperVec,
}

impl PimRoleVectors {
    /// Create role vectors via deterministic token encoding.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            urgency: encode_token(ops, "pim-role:urgency"),
            importance: encode_token(ops, "pim-role:importance"),
            energy: encode_token(ops, "pim-role:energy"),
            context: encode_token(ops, "pim-role:context"),
            deadline_proximity: encode_token(ops, "pim-role:deadline-proximity"),
            gtd_state: encode_token(ops, "pim-role:gtd-state"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PimManager
// ═══════════════════════════════════════════════════════════════════════

/// Serializable representation of the dependency DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableDag {
    /// Node symbols.
    nodes: Vec<SymbolId>,
    /// Edges as (source_index, target_index, edge_data) into `nodes`.
    edges: Vec<(usize, usize, DependencyEdge)>,
}

impl SerializableDag {
    fn from_dag(
        dag: &DiGraph<SymbolId, DependencyEdge>,
        node_index: &HashMap<u64, NodeIndex>,
    ) -> Self {
        use petgraph::visit::EdgeRef;
        let mut nodes = Vec::new();
        let mut idx_map: HashMap<NodeIndex, usize> = HashMap::new();

        // Build index-stable node list.
        for (&raw_id, &ni) in node_index {
            let pos = nodes.len();
            nodes.push(SymbolId::new(raw_id).unwrap());
            idx_map.insert(ni, pos);
        }

        let mut edges = Vec::new();
        for edge in dag.edge_references() {
            if let (Some(&src), Some(&tgt)) = (idx_map.get(&edge.source()), idx_map.get(&edge.target())) {
                edges.push((src, tgt, *edge.weight()));
            }
        }

        Self { nodes, edges }
    }

    fn to_dag(&self) -> (DiGraph<SymbolId, DependencyEdge>, HashMap<u64, NodeIndex>) {
        let mut dag = DiGraph::new();
        let mut node_index = HashMap::new();
        let mut idx_map: Vec<NodeIndex> = Vec::new();

        for &sym in &self.nodes {
            let ni = dag.add_node(sym);
            node_index.insert(sym.get(), ni);
            idx_map.push(ni);
        }

        for &(src, tgt, ref edge) in &self.edges {
            if src < idx_map.len() && tgt < idx_map.len() {
                dag.add_edge(idx_map[src], idx_map[tgt], *edge);
            }
        }

        (dag, node_index)
    }
}

/// Manages PIM metadata for goals, dependency DAG, and VSA priority encoding.
pub struct PimManager {
    metadata: HashMap<u64, PimMetadata>,
    dep_dag: DiGraph<SymbolId, DependencyEdge>,
    node_index: HashMap<u64, NodeIndex>,
    role_vectors: Option<PimRoleVectors>,
    predicates: Option<PimPredicates>,
}

impl Serialize for PimManager {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("PimManager", 2)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.serialize_field("dag", &SerializableDag::from_dag(&self.dep_dag, &self.node_index))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for PimManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct PimManagerData {
            metadata: HashMap<u64, PimMetadata>,
            dag: SerializableDag,
        }

        let data = PimManagerData::deserialize(deserializer)?;
        let (dep_dag, node_index) = data.dag.to_dag();
        Ok(Self {
            metadata: data.metadata,
            dep_dag,
            node_index,
            role_vectors: None,
            predicates: None,
        })
    }
}

impl Default for PimManager {
    fn default() -> Self {
        Self {
            metadata: HashMap::new(),
            dep_dag: DiGraph::new(),
            node_index: HashMap::new(),
            role_vectors: None,
            predicates: None,
        }
    }
}

impl fmt::Debug for PimManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PimManager")
            .field("task_count", &self.metadata.len())
            .field("dep_edges", &self.dep_dag.edge_count())
            .finish()
    }
}

/// Dependency edge in the PIM task DAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyEdge {
    /// Blocked task must wait for blocker to finish.
    FinishToStart,
}

impl PimManager {
    /// Create a new PIM manager, initializing predicates and role vectors.
    pub fn new(engine: &Engine) -> PimResult<Self> {
        let predicates = PimPredicates::init(engine)?;
        let role_vectors = PimRoleVectors::new(engine.ops());
        Ok(Self {
            metadata: HashMap::new(),
            dep_dag: DiGraph::new(),
            node_index: HashMap::new(),
            role_vectors: Some(role_vectors),
            predicates: Some(predicates),
        })
    }

    /// Ensure predicates and role vectors are initialized (for post-deserialization).
    pub fn ensure_init(&mut self, engine: &Engine) -> PimResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(PimPredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(PimRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Number of PIM-tracked tasks.
    pub fn task_count(&self) -> usize {
        self.metadata.len()
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Add PIM metadata overlay to an existing goal.
    pub fn add_task(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        gtd: GtdState,
        urgency: f32,
        importance: f32,
    ) -> PimResult<()> {
        let quadrant = EisenhowerQuadrant::classify(urgency, importance);
        let meta = PimMetadata {
            goal_id,
            gtd_state: gtd,
            urgency: urgency.clamp(0.0, 1.0),
            importance: importance.clamp(0.0, 1.0),
            quadrant,
            para: None,
            contexts: Vec::new(),
            energy: None,
            time_estimate_minutes: None,
            deadline: None,
            recurrence: None,
            next_due: None,
            last_completed: None,
        };
        self.sync_to_kg(engine, goal_id, &meta)?;
        self.record_provenance(engine, goal_id, &meta);
        self.metadata.insert(goal_id.get(), meta);

        // Ensure DAG node exists.
        if !self.node_index.contains_key(&goal_id.get()) {
            let idx = self.dep_dag.add_node(goal_id);
            self.node_index.insert(goal_id.get(), idx);
        }

        Ok(())
    }

    /// Get metadata for a goal (immutable).
    pub fn get_metadata(&self, goal_id: u64) -> Option<&PimMetadata> {
        self.metadata.get(&goal_id)
    }

    /// Get metadata for a goal (mutable).
    pub fn get_metadata_mut(&mut self, goal_id: u64) -> Option<&mut PimMetadata> {
        self.metadata.get_mut(&goal_id)
    }

    // ── GTD transitions ──────────────────────────────────────────────

    /// Transition a task's GTD state, validating the transition.
    pub fn transition_gtd(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        new_state: GtdState,
    ) -> PimResult<()> {
        // Validate transition.
        let current_state = self
            .metadata
            .get(&goal_id.get())
            .ok_or(PimError::TaskNotFound {
                goal_id: goal_id.get(),
            })?
            .gtd_state;

        if !current_state.can_transition_to(new_state) {
            return Err(PimError::InvalidTransition {
                from: current_state.to_string(),
                to: new_state.to_string(),
            });
        }

        // Apply.
        let meta = self.metadata.get_mut(&goal_id.get()).unwrap();
        meta.gtd_state = new_state;
        if new_state == GtdState::Done {
            meta.last_completed = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
        }

        let meta_snapshot = meta.clone();
        self.sync_to_kg(engine, goal_id, &meta_snapshot)?;
        self.record_provenance(engine, goal_id, &meta_snapshot);
        Ok(())
    }

    // ── Eisenhower ───────────────────────────────────────────────────

    /// Update urgency/importance and recompute quadrant.
    pub fn update_eisenhower(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        urgency: f32,
        importance: f32,
    ) -> PimResult<()> {
        {
            let meta = self
                .metadata
                .get_mut(&goal_id.get())
                .ok_or(PimError::TaskNotFound {
                    goal_id: goal_id.get(),
                })?;
            meta.urgency = urgency.clamp(0.0, 1.0);
            meta.importance = importance.clamp(0.0, 1.0);
            meta.quadrant = EisenhowerQuadrant::classify(meta.urgency, meta.importance);
        }
        let snapshot = self.metadata[&goal_id.get()].clone();
        self.sync_to_kg(engine, goal_id, &snapshot)?;
        Ok(())
    }

    // ── PARA ─────────────────────────────────────────────────────────

    /// Set PARA category.
    pub fn set_para(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        category: ParaCategory,
    ) -> PimResult<()> {
        {
            let meta = self
                .metadata
                .get_mut(&goal_id.get())
                .ok_or(PimError::TaskNotFound {
                    goal_id: goal_id.get(),
                })?;
            meta.para = Some(category);
        }
        let snapshot = self.metadata[&goal_id.get()].clone();
        self.sync_to_kg(engine, goal_id, &snapshot)?;
        Ok(())
    }

    // ── Context ──────────────────────────────────────────────────────

    /// Add a GTD context.
    pub fn add_context(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        context: PimContext,
    ) -> PimResult<()> {
        {
            let meta = self
                .metadata
                .get_mut(&goal_id.get())
                .ok_or(PimError::TaskNotFound {
                    goal_id: goal_id.get(),
                })?;
            if !meta.contexts.contains(&context) {
                meta.contexts.push(context);
            }
        }
        let snapshot = self.metadata[&goal_id.get()].clone();
        self.sync_to_kg(engine, goal_id, &snapshot)?;
        Ok(())
    }

    // ── Energy ───────────────────────────────────────────────────────

    /// Set energy level.
    pub fn set_energy(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        energy: EnergyLevel,
    ) -> PimResult<()> {
        {
            let meta = self
                .metadata
                .get_mut(&goal_id.get())
                .ok_or(PimError::TaskNotFound {
                    goal_id: goal_id.get(),
                })?;
            meta.energy = Some(energy);
        }
        let snapshot = self.metadata[&goal_id.get()].clone();
        self.sync_to_kg(engine, goal_id, &snapshot)?;
        Ok(())
    }

    // ── Filtering ────────────────────────────────────────────────────

    /// Filter tasks available for the given context and energy level.
    ///
    /// Returns goal IDs of tasks that are GTD Next, match the context (or have
    /// no context restriction), and require at most the given energy.
    pub fn available_tasks(
        &self,
        context: Option<&PimContext>,
        energy: Option<EnergyLevel>,
        goals: &[Goal],
    ) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| m.gtd_state == GtdState::Next)
            .filter(|m| {
                // Context filter: match if task has no contexts or context matches.
                context.map_or(true, |ctx| m.contexts.is_empty() || m.contexts.contains(ctx))
            })
            .filter(|m| {
                // Energy filter: task energy ≤ available energy.
                match (m.energy, energy) {
                    (None, _) | (_, None) => true,
                    (Some(EnergyLevel::Low), _) => true,
                    (Some(EnergyLevel::Medium), Some(EnergyLevel::Medium | EnergyLevel::High)) => {
                        true
                    }
                    (Some(EnergyLevel::High), Some(EnergyLevel::High)) => true,
                    _ => false,
                }
            })
            .filter(|m| {
                // Must not be blocked (all blockers must be Done).
                !self.is_blocked(m.goal_id, goals)
            })
            .map(|m| m.goal_id)
            .collect()
    }

    /// Check if a task is blocked (any blocker in DAG is not Done).
    fn is_blocked(&self, goal_id: SymbolId, goals: &[Goal]) -> bool {
        use petgraph::visit::EdgeRef;
        let Some(&idx) = self.node_index.get(&goal_id.get()) else {
            return false;
        };
        for edge in self.dep_dag.edges_directed(idx, petgraph::Direction::Incoming) {
            let blocker_sym = self.dep_dag[edge.source()];
            // Check if blocker is Done in PIM metadata or completed as a Goal.
            let pim_done = self
                .metadata
                .get(&blocker_sym.get())
                .map_or(false, |m| m.gtd_state == GtdState::Done);
            let goal_done = goals
                .iter()
                .find(|g| g.symbol_id == blocker_sym)
                .map_or(false, |g| matches!(g.status, GoalStatus::Completed));
            if !pim_done && !goal_done {
                return true;
            }
        }
        false
    }

    // ── Dependency DAG ───────────────────────────────────────────────

    /// Add a dependency: `blocker` must finish before `blocked` can start.
    pub fn add_dependency(
        &mut self,
        engine: &Engine,
        blocker: SymbolId,
        blocked: SymbolId,
    ) -> PimResult<()> {
        // Ensure both nodes exist.
        let blocker_idx = *self
            .node_index
            .entry(blocker.get())
            .or_insert_with(|| self.dep_dag.add_node(blocker));
        let blocked_idx = *self
            .node_index
            .entry(blocked.get())
            .or_insert_with(|| self.dep_dag.add_node(blocked));

        // Add tentative edge.
        let edge = self
            .dep_dag
            .add_edge(blocker_idx, blocked_idx, DependencyEdge::FinishToStart);

        // Cycle check.
        if petgraph::algo::is_cyclic_directed(&self.dep_dag) {
            self.dep_dag.remove_edge(edge);
            return Err(PimError::CycleDetected {
                task_id: blocked.get(),
            });
        }

        // Record in KG.
        if let Some(preds) = &self.predicates {
            let _ = engine.add_triple(&Triple::new(
                blocked,
                preds.blocked_by,
                blocker,
            ));
            let _ = engine.add_triple(&Triple::new(
                blocker,
                preds.blocks,
                blocked,
            ));
        }

        Ok(())
    }

    /// Remove a dependency edge.
    pub fn remove_dependency(&mut self, blocker: SymbolId, blocked: SymbolId) {
        let Some(&blocker_idx) = self.node_index.get(&blocker.get()) else {
            return;
        };
        let Some(&blocked_idx) = self.node_index.get(&blocked.get()) else {
            return;
        };
        if let Some(edge) = self.dep_dag.find_edge(blocker_idx, blocked_idx) {
            self.dep_dag.remove_edge(edge);
        }
    }

    /// Topological order of all tasks (Kahn's algorithm via petgraph).
    pub fn topological_order(&self) -> PimResult<Vec<SymbolId>> {
        petgraph::algo::toposort(&self.dep_dag, None)
            .map(|nodes| nodes.into_iter().map(|idx| self.dep_dag[idx]).collect())
            .map_err(|cycle| PimError::CycleDetected {
                task_id: self.dep_dag[cycle.node_id()].get(),
            })
    }

    /// Critical path: longest path through the DAG by estimated time.
    pub fn critical_path(&self, _goals: &[Goal]) -> PimResult<Vec<SymbolId>> {
        let topo = petgraph::algo::toposort(&self.dep_dag, None).map_err(|cycle| {
            PimError::CycleDetected {
                task_id: self.dep_dag[cycle.node_id()].get(),
            }
        })?;

        if topo.is_empty() {
            return Ok(Vec::new());
        }

        // DP: longest path by estimated minutes (default 30).
        let mut dist: HashMap<NodeIndex, u32> = HashMap::new();
        let mut pred: HashMap<NodeIndex, Option<NodeIndex>> = HashMap::new();
        for &node in &topo {
            let sym = self.dep_dag[node];
            let est = self
                .metadata
                .get(&sym.get())
                .and_then(|m| m.time_estimate_minutes)
                .unwrap_or(30);
            dist.insert(node, est);
            pred.insert(node, None);
        }

        use petgraph::visit::EdgeRef;
        for &node in &topo {
            let node_dist = dist[&node];
            for edge in self
                .dep_dag
                .edges_directed(node, petgraph::Direction::Outgoing)
            {
                let target = edge.target();
                let target_sym = self.dep_dag[target];
                let target_est = self
                    .metadata
                    .get(&target_sym.get())
                    .and_then(|m| m.time_estimate_minutes)
                    .unwrap_or(30);
                let new_dist = node_dist + target_est;
                if new_dist > dist[&target] {
                    dist.insert(target, new_dist);
                    pred.insert(target, Some(node));
                }
            }
        }

        // Find the end node with the longest distance.
        let end_node = *dist.iter().max_by_key(|&(_, d)| d).map(|(n, _)| n).unwrap();

        // Trace back.
        let mut path = vec![self.dep_dag[end_node]];
        let mut current = end_node;
        while let Some(Some(prev)) = pred.get(&current) {
            path.push(self.dep_dag[*prev]);
            current = *prev;
        }
        path.reverse();
        Ok(path)
    }

    /// Tasks that are unblocked and in GTD Next state.
    pub fn ready_tasks(&self, goals: &[Goal]) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| m.gtd_state == GtdState::Next)
            .filter(|m| !self.is_blocked(m.goal_id, goals))
            .map(|m| m.goal_id)
            .collect()
    }

    // ── Recurrence ───────────────────────────────────────────────────

    /// Set recurrence pattern.
    pub fn set_recurrence(
        &mut self,
        engine: &Engine,
        goal_id: SymbolId,
        recurrence: Recurrence,
    ) -> PimResult<()> {
        {
            let meta = self
                .metadata
                .get_mut(&goal_id.get())
                .ok_or(PimError::TaskNotFound {
                    goal_id: goal_id.get(),
                })?;
            meta.recurrence = Some(recurrence);
        }
        let snapshot = self.metadata[&goal_id.get()].clone();
        self.sync_to_kg(engine, goal_id, &snapshot)?;
        Ok(())
    }

    /// Process completed recurring tasks: mark Done, compute next occurrence,
    /// and create a new goal for the next instance.
    pub fn process_recurring_completions(
        &mut self,
        engine: &Engine,
        goals: &mut Vec<Goal>,
        agent_predicates: &super::agent::AgentPredicates,
    ) -> PimResult<Vec<SymbolId>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut new_goals = Vec::new();

        // Collect completed recurring task IDs.
        let completed_recurring: Vec<(u64, Recurrence, String)> = self
            .metadata
            .iter()
            .filter(|(_, m)| m.gtd_state == GtdState::Done && m.recurrence.is_some())
            .map(|(&id, m)| {
                let recurrence = m.recurrence.clone().unwrap();
                let desc = goals
                    .iter()
                    .find(|g| g.symbol_id.get() == id)
                    .map(|g| g.description.clone())
                    .unwrap_or_else(|| format!("recurring-{id}"));
                (id, recurrence, desc)
            })
            .collect();

        for (old_id, recurrence, description) in completed_recurring {
            let next_due = recurrence.next_occurrence(now);

            // Create a new goal for the next occurrence.
            let criteria = format!("{description} completed");
            let new_goal =
                super::goal::create_goal(engine, &description, 128, &criteria, agent_predicates)
                    .map_err(|e| PimError::Engine(Box::new(
                        crate::error::AkhError::Store(crate::error::StoreError::Serialization {
                            message: format!("goal creation failed: {e}"),
                        }),
                    )))?;
            let new_sym = new_goal.symbol_id;
            goals.push(new_goal);

            // Add PIM metadata for the new occurrence.
            self.add_task(engine, new_sym, GtdState::Next, 0.5, 0.5)?;
            if let Some(new_meta) = self.metadata.get_mut(&new_sym.get()) {
                new_meta.recurrence = Some(recurrence);
                new_meta.next_due = Some(next_due);
            }

            // Clear recurrence from the completed instance so it doesn't trigger again.
            if let Some(old_meta) = self.metadata.get_mut(&old_id) {
                old_meta.recurrence = None;
            }

            new_goals.push(new_sym);
        }

        Ok(new_goals)
    }

    // ── Queries ──────────────────────────────────────────────────────

    /// Tasks with next_due before the given timestamp.
    pub fn overdue_tasks(&self, deadline_ts: u64) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| {
                m.gtd_state != GtdState::Done
                    && m.next_due.map_or(false, |due| due < deadline_ts)
            })
            .map(|m| m.goal_id)
            .collect()
    }

    /// Tasks by GTD state.
    pub fn tasks_by_gtd_state(&self, state: GtdState) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| m.gtd_state == state)
            .map(|m| m.goal_id)
            .collect()
    }

    /// Tasks by Eisenhower quadrant.
    pub fn tasks_by_quadrant(&self, q: EisenhowerQuadrant) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| m.quadrant == q)
            .map(|m| m.goal_id)
            .collect()
    }

    /// Tasks by PARA category.
    pub fn tasks_by_para(&self, cat: ParaCategory) -> Vec<SymbolId> {
        self.metadata
            .values()
            .filter(|m| m.para == Some(cat))
            .map(|m| m.goal_id)
            .collect()
    }

    // ── VSA priority encoding ────────────────────────────────────────

    /// Encode priority profile as a HyperVec via role-filler binding.
    pub fn encode_priority(&self, ops: &VsaOps, goal_id: u64) -> Option<HyperVec> {
        let meta = self.metadata.get(&goal_id)?;
        let roles = self.role_vectors.as_ref()?;

        // Urgency filler: bucket into low/medium/high.
        let urgency_bucket = if meta.urgency < 0.33 {
            "low"
        } else if meta.urgency < 0.67 {
            "medium"
        } else {
            "high"
        };
        let urgency_filler = encode_token(ops, urgency_bucket);

        // Importance filler.
        let importance_bucket = if meta.importance < 0.33 {
            "low"
        } else if meta.importance < 0.67 {
            "medium"
        } else {
            "high"
        };
        let importance_filler = encode_token(ops, importance_bucket);

        // Energy filler.
        let energy_label = meta.energy.unwrap_or(EnergyLevel::Medium).as_label();
        let energy_filler = encode_token(ops, energy_label);

        // GTD state filler.
        let gtd_filler = encode_token(ops, meta.gtd_state.as_label());

        // Bind role-filler pairs.
        let pairs: Vec<HyperVec> = [
            ops.bind(&roles.urgency, &urgency_filler),
            ops.bind(&roles.importance, &importance_filler),
            ops.bind(&roles.energy, &energy_filler),
            ops.bind(&roles.gtd_state, &gtd_filler),
        ]
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();

        if pairs.is_empty() {
            return None;
        }

        let refs: Vec<&HyperVec> = pairs.iter().collect();
        ops.bundle(&refs).ok()
    }

    /// Find tasks with similar priority profiles to the given task.
    pub fn find_similar_priority(
        &self,
        ops: &VsaOps,
        goal_id: u64,
        _goals: &[Goal],
        top_k: usize,
    ) -> Vec<(SymbolId, f32)> {
        let Some(query_vec) = self.encode_priority(ops, goal_id) else {
            return Vec::new();
        };

        let mut scores: Vec<(SymbolId, f32)> = self
            .metadata
            .keys()
            .filter(|&&id| id != goal_id)
            .filter_map(|&id| {
                let vec = self.encode_priority(ops, id)?;
                let sim = ops.similarity(&query_vec, &vec).ok()?;
                Some((self.metadata[&id].goal_id, sim))
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    // ── KG sync ──────────────────────────────────────────────────────

    /// Write PIM metadata to KG triples.
    fn sync_to_kg(
        &self,
        engine: &Engine,
        goal_id: SymbolId,
        meta: &PimMetadata,
    ) -> PimResult<()> {
        let Some(preds) = &self.predicates else {
            return Ok(());
        };

        // GTD state.
        let gtd_entity = engine.resolve_or_create_entity(&format!("pim-gtd:{}", meta.gtd_state))?;
        let _ = engine.add_triple(&Triple::new(
            goal_id,
            preds.gtd_state,
            gtd_entity,
        ));

        // Quadrant.
        let quad_entity =
            engine.resolve_or_create_entity(&format!("pim-quadrant:{}", meta.quadrant))?;
        let _ = engine.add_triple(&Triple::new(
            goal_id,
            preds.quadrant,
            quad_entity,
        ));

        // Urgency/Importance as label entities.
        let urgency_entity =
            engine.resolve_or_create_entity(&format!("pim-urgency:{:.2}", meta.urgency))?;
        let _ = engine.add_triple(&Triple::new(
            goal_id,
            preds.urgency,
            urgency_entity,
        ));

        let importance_entity =
            engine.resolve_or_create_entity(&format!("pim-importance:{:.2}", meta.importance))?;
        let _ = engine.add_triple(&Triple::new(
            goal_id,
            preds.importance,
            importance_entity,
        ));

        // Optional fields.
        if let Some(para) = meta.para {
            let para_entity =
                engine.resolve_or_create_entity(&format!("pim-para:{}", para))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.para_category,
                para_entity,
            ));
        }

        if let Some(energy) = meta.energy {
            let energy_entity =
                engine.resolve_or_create_entity(&format!("pim-energy:{}", energy))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.energy,
                energy_entity,
            ));
        }

        for ctx in &meta.contexts {
            let ctx_entity =
                engine.resolve_or_create_entity(&format!("pim-context:{}", ctx.0))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.context,
                ctx_entity,
            ));
        }

        if let Some(deadline) = meta.deadline {
            let deadline_entity =
                engine.resolve_or_create_entity(&format!("pim-deadline:{deadline}"))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.deadline,
                deadline_entity,
            ));
        }

        if let Some(ref recurrence) = meta.recurrence {
            let recur_entity =
                engine.resolve_or_create_entity(&format!("pim-recur:{}", recurrence.as_label()))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.recurrence,
                recur_entity,
            ));
        }

        if let Some(next_due) = meta.next_due {
            let due_entity =
                engine.resolve_or_create_entity(&format!("pim-next-due:{next_due}"))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.next_due,
                due_entity,
            ));
        }

        if let Some(time_est) = meta.time_estimate_minutes {
            let est_entity =
                engine.resolve_or_create_entity(&format!("pim-time-est:{time_est}"))?;
            let _ = engine.add_triple(&Triple::new(
                goal_id,
                preds.time_estimate,
                est_entity,
            ));
        }

        Ok(())
    }

    /// Record PIM provenance for a managed task.
    fn record_provenance(&self, engine: &Engine, goal_id: SymbolId, meta: &PimMetadata) {
        let mut record = ProvenanceRecord::new(
            goal_id,
            DerivationKind::PimTaskManaged {
                goal_id_raw: goal_id.get(),
                gtd_state: meta.gtd_state.to_string(),
                quadrant: meta.quadrant.to_string(),
            },
        );
        let _ = engine.store_provenance(&mut record);
    }

    // ── Persistence ──────────────────────────────────────────────────

    /// Persist PIM state to the engine's durable store.
    pub fn persist(&self, engine: &Engine) -> PimResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| PimError::Engine(Box::new(
            crate::error::AkhError::Store(crate::error::StoreError::Serialization {
                message: format!("failed to serialize PIM manager: {e}"),
            }),
        )))?;
        engine
            .store()
            .put_meta(b"agent:pim_manager", &bytes)
            .map_err(|e| PimError::Engine(Box::new(crate::error::AkhError::Store(e))))?;
        Ok(())
    }

    /// Restore PIM state from the engine's durable store.
    pub fn restore(engine: &Engine) -> PimResult<Self> {
        let bytes = engine
            .store()
            .get_meta(b"agent:pim_manager")
            .map_err(|e| PimError::Engine(Box::new(crate::error::AkhError::Store(e))))?
            .ok_or(PimError::TaskNotFound { goal_id: 0 })?;
        let mut manager: Self =
            bincode::deserialize(&bytes).map_err(|e| PimError::Engine(Box::new(
                crate::error::AkhError::Store(crate::error::StoreError::Serialization {
                    message: format!("failed to deserialize PIM manager: {e}"),
                }),
            )))?;
        manager.ensure_init(engine)?;
        Ok(manager)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// GTD Weekly Review
// ═══════════════════════════════════════════════════════════════════════

/// Result of a GTD weekly review.
#[derive(Debug, Clone)]
pub struct GtdReviewResult {
    /// Inbox items older than 7 days.
    pub stale_inbox: Vec<SymbolId>,
    /// Waiting items (may be unblocked).
    pub waiting_items: Vec<SymbolId>,
    /// Someday items that could promote to Next.
    pub someday_candidates: Vec<SymbolId>,
    /// Tasks with next_due in the past.
    pub overdue: Vec<SymbolId>,
    /// Projects with no active Next actions.
    pub stalled_projects: Vec<SymbolId>,
    /// Next items older than 2 review cycles (14 days).
    pub migration_candidates: Vec<SymbolId>,
    /// Recommended adjustments.
    pub adjustments: Vec<Adjustment>,
    /// Human-readable summary.
    pub summary: String,
}

/// Run a GTD weekly review.
pub fn gtd_weekly_review(
    pim: &PimManager,
    goals: &[Goal],
    projects: &[super::project::Project],
    current_ts: u64,
) -> GtdReviewResult {
    const SEVEN_DAYS: u64 = 7 * 86_400;
    const FOURTEEN_DAYS: u64 = 14 * 86_400;

    let stale_inbox: Vec<SymbolId> = pim
        .metadata
        .values()
        .filter(|m| m.gtd_state == GtdState::Inbox)
        .filter(|m| {
            goals
                .iter()
                .find(|g| g.symbol_id == m.goal_id)
                .map_or(false, |g| current_ts.saturating_sub(g.created_at) > SEVEN_DAYS)
        })
        .map(|m| m.goal_id)
        .collect();

    let waiting_items: Vec<SymbolId> = pim.tasks_by_gtd_state(GtdState::Waiting);

    let someday_candidates: Vec<SymbolId> = pim.tasks_by_gtd_state(GtdState::Someday);

    let overdue = pim.overdue_tasks(current_ts);

    // Stalled projects: projects whose goals have no PIM Next tasks.
    let stalled_projects: Vec<SymbolId> = projects
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                super::project::ProjectStatus::Active
            )
        })
        .filter(|p| {
            !p.goals.iter().any(|gid| {
                pim.get_metadata(gid.get())
                    .map_or(false, |m| m.gtd_state == GtdState::Next)
            })
        })
        .map(|p| p.id)
        .collect();

    // Migration candidates: Next tasks created > 14 days ago.
    let migration_candidates: Vec<SymbolId> = pim
        .metadata
        .values()
        .filter(|m| m.gtd_state == GtdState::Next)
        .filter(|m| {
            goals
                .iter()
                .find(|g| g.symbol_id == m.goal_id)
                .map_or(false, |g| {
                    current_ts.saturating_sub(g.created_at) > FOURTEEN_DAYS
                })
        })
        .map(|m| m.goal_id)
        .collect();

    // Build adjustments.
    let mut adjustments = Vec::new();

    for &id in &overdue {
        if let Some(goal) = goals.iter().find(|g| g.symbol_id == id) {
            adjustments.push(Adjustment::IncreasePriority {
                goal_id: id,
                from: goal.priority,
                to: goal.priority.saturating_add(20),
                reason: "overdue task — priority boost".into(),
            });
        }
    }

    for &id in &migration_candidates {
        adjustments.push(Adjustment::SuggestAbandon {
            goal_id: id,
            reason: "stale Next task — consider moving to Someday or abandoning".into(),
        });
    }

    let summary = format!(
        "GTD review: {} stale inbox, {} waiting, {} someday, {} overdue, \
         {} stalled projects, {} migration candidates, {} adjustments",
        stale_inbox.len(),
        waiting_items.len(),
        someday_candidates.len(),
        overdue.len(),
        stalled_projects.len(),
        migration_candidates.len(),
        adjustments.len(),
    );

    GtdReviewResult {
        stale_inbox,
        waiting_items,
        someday_candidates,
        overdue,
        stalled_projects,
        migration_candidates,
        adjustments,
        summary,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ActionItemGoalSpec bridge (from Phase 13d email extraction)
// ═══════════════════════════════════════════════════════════════════════

/// Create goals from `ActionItemGoalSpec` and add PIM metadata.
///
/// Returns the SymbolIds of the newly created goals.
#[cfg(feature = "email")]
pub fn action_items_to_pim_tasks(
    pim: &mut PimManager,
    engine: &Engine,
    specs: &[crate::email::extract::ActionItemGoalSpec],
    agent_predicates: &super::agent::AgentPredicates,
) -> PimResult<Vec<SymbolId>> {
    let mut created = Vec::new();

    for spec in specs {
        let goal = super::goal::create_goal(
            engine,
            &spec.description,
            spec.priority,
            &spec.criteria,
            agent_predicates,
        )
        .map_err(|e| PimError::Engine(Box::new(
            crate::error::AkhError::Store(crate::error::StoreError::Serialization {
                            message: format!("goal creation failed: {e}"),
                        }),
        )))?;

        let urgency = if spec.priority >= 7 { 0.8 } else { 0.4 };
        let importance = 0.5;

        pim.add_task(engine, goal.symbol_id, GtdState::Inbox, urgency, importance)?;
        created.push(goal.symbol_id);
    }

    Ok(created)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── GtdState ─────────────────────────────────────────────────────

    #[test]
    fn gtd_inbox_can_transition_to_next() {
        assert!(GtdState::Inbox.can_transition_to(GtdState::Next));
    }

    #[test]
    fn gtd_inbox_can_transition_to_done() {
        assert!(GtdState::Inbox.can_transition_to(GtdState::Done));
    }

    #[test]
    fn gtd_done_is_terminal() {
        assert!(!GtdState::Done.can_transition_to(GtdState::Next));
        assert!(!GtdState::Done.can_transition_to(GtdState::Inbox));
    }

    #[test]
    fn gtd_next_cannot_go_to_inbox() {
        assert!(!GtdState::Next.can_transition_to(GtdState::Inbox));
    }

    #[test]
    fn gtd_from_label_roundtrip() {
        for state in [
            GtdState::Inbox,
            GtdState::Next,
            GtdState::Waiting,
            GtdState::Someday,
            GtdState::Reference,
            GtdState::Done,
        ] {
            assert_eq!(GtdState::from_label(state.as_label()), Some(state));
        }
    }

    // ── EisenhowerQuadrant ───────────────────────────────────────────

    #[test]
    fn eisenhower_classify_do() {
        assert_eq!(
            EisenhowerQuadrant::classify(0.8, 0.9),
            EisenhowerQuadrant::Do
        );
    }

    #[test]
    fn eisenhower_classify_schedule() {
        assert_eq!(
            EisenhowerQuadrant::classify(0.2, 0.8),
            EisenhowerQuadrant::Schedule
        );
    }

    #[test]
    fn eisenhower_classify_delegate() {
        assert_eq!(
            EisenhowerQuadrant::classify(0.7, 0.3),
            EisenhowerQuadrant::Delegate
        );
    }

    #[test]
    fn eisenhower_classify_eliminate() {
        assert_eq!(
            EisenhowerQuadrant::classify(0.1, 0.2),
            EisenhowerQuadrant::Eliminate
        );
    }

    #[test]
    fn eisenhower_priority_bonus_ordering() {
        assert!(EisenhowerQuadrant::Do.priority_bonus() > EisenhowerQuadrant::Schedule.priority_bonus());
        assert!(EisenhowerQuadrant::Schedule.priority_bonus() > EisenhowerQuadrant::Delegate.priority_bonus());
        assert!(EisenhowerQuadrant::Delegate.priority_bonus() > EisenhowerQuadrant::Eliminate.priority_bonus());
    }

    // ── Recurrence ───────────────────────────────────────────────────

    #[test]
    fn recurrence_parse_daily() {
        assert_eq!(Recurrence::parse("daily").unwrap(), Recurrence::Daily);
    }

    #[test]
    fn recurrence_parse_weekly() {
        let rec = Recurrence::parse("weekly:mon,wed,fri").unwrap();
        assert_eq!(rec, Recurrence::Weekly(vec![0, 2, 4]));
    }

    #[test]
    fn recurrence_parse_monthly() {
        let rec = Recurrence::parse("monthly:15").unwrap();
        assert_eq!(rec, Recurrence::Monthly(15));
    }

    #[test]
    fn recurrence_parse_yearly() {
        let rec = Recurrence::parse("yearly:3-21").unwrap();
        assert_eq!(rec, Recurrence::Yearly(3, 21));
    }

    #[test]
    fn recurrence_parse_every_n_days() {
        let rec = Recurrence::parse("every:3d").unwrap();
        assert_eq!(rec, Recurrence::EveryNDays(3));
    }

    #[test]
    fn recurrence_parse_invalid() {
        assert!(Recurrence::parse("biweekly").is_err());
    }

    #[test]
    fn recurrence_daily_next_occurrence() {
        let ts = 1_700_000_000u64;
        assert_eq!(Recurrence::Daily.next_occurrence(ts), ts + 86_400);
    }

    #[test]
    fn recurrence_every_3d_next() {
        let ts = 1_700_000_000u64;
        assert_eq!(
            Recurrence::EveryNDays(3).next_occurrence(ts),
            ts + 3 * 86_400
        );
    }

    #[test]
    fn recurrence_roundtrip_label() {
        let rec = Recurrence::Weekly(vec![0, 4]);
        let label = rec.as_label();
        let parsed = Recurrence::from_label(&label).unwrap();
        assert_eq!(parsed, rec);
    }

    // ── PimPredicates ────────────────────────────────────────────────

    #[test]
    fn pim_predicates_init() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let preds = PimPredicates::init(&engine);
        assert!(preds.is_ok());
        let preds = preds.unwrap();
        assert_ne!(preds.gtd_state, preds.context);
    }

    #[test]
    fn pim_predicates_all_distinct() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let preds = PimPredicates::init(&engine).unwrap();
        let ids = [
            preds.gtd_state,
            preds.context,
            preds.energy,
            preds.time_estimate,
            preds.urgency,
            preds.importance,
            preds.para_category,
            preds.deadline,
            preds.quadrant,
            preds.blocked_by,
            preds.blocks,
            preds.recurrence,
            preds.next_due,
            preds.last_done,
        ];
        let set: std::collections::HashSet<u64> = ids.iter().map(|id| id.get()).collect();
        assert_eq!(set.len(), ids.len(), "all predicates must be distinct");
    }

    // ── PimManager CRUD ──────────────────────────────────────────────

    #[test]
    fn pim_manager_add_and_get() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal = super::super::goal::create_goal(
            &engine,
            "test task",
            128,
            "task done",
            &agent_preds,
        )
        .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Inbox, 0.5, 0.5)
            .unwrap();

        let meta = pim.get_metadata(goal.symbol_id.get());
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.gtd_state, GtdState::Inbox);
        assert_eq!(meta.quadrant, EisenhowerQuadrant::Do);
    }

    #[test]
    fn pim_manager_transition_gtd() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal = super::super::goal::create_goal(
            &engine,
            "test task 2",
            128,
            "task done",
            &agent_preds,
        )
        .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Inbox, 0.3, 0.7)
            .unwrap();

        // Valid transition.
        assert!(pim
            .transition_gtd(&engine, goal.symbol_id, GtdState::Next)
            .is_ok());
        assert_eq!(
            pim.get_metadata(goal.symbol_id.get()).unwrap().gtd_state,
            GtdState::Next
        );

        // Invalid transition.
        assert!(pim
            .transition_gtd(&engine, goal.symbol_id, GtdState::Inbox)
            .is_err());
    }

    #[test]
    fn pim_manager_update_eisenhower() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal = super::super::goal::create_goal(
            &engine,
            "eisenhower test",
            128,
            "done",
            &agent_preds,
        )
        .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Next, 0.8, 0.8)
            .unwrap();
        assert_eq!(
            pim.get_metadata(goal.symbol_id.get()).unwrap().quadrant,
            EisenhowerQuadrant::Do
        );

        pim.update_eisenhower(&engine, goal.symbol_id, 0.2, 0.8)
            .unwrap();
        assert_eq!(
            pim.get_metadata(goal.symbol_id.get()).unwrap().quadrant,
            EisenhowerQuadrant::Schedule
        );
    }

    #[test]
    fn pim_manager_set_para() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal =
            super::super::goal::create_goal(&engine, "para test", 128, "done", &agent_preds)
                .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Inbox, 0.5, 0.5)
            .unwrap();
        pim.set_para(&engine, goal.symbol_id, ParaCategory::Project)
            .unwrap();

        assert_eq!(
            pim.get_metadata(goal.symbol_id.get()).unwrap().para,
            Some(ParaCategory::Project)
        );
    }

    #[test]
    fn pim_manager_add_context() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal =
            super::super::goal::create_goal(&engine, "ctx test", 128, "done", &agent_preds)
                .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.add_context(&engine, goal.symbol_id, PimContext::computer())
            .unwrap();
        pim.add_context(&engine, goal.symbol_id, PimContext::office())
            .unwrap();
        // Duplicate should not add again.
        pim.add_context(&engine, goal.symbol_id, PimContext::computer())
            .unwrap();

        let meta = pim.get_metadata(goal.symbol_id.get()).unwrap();
        assert_eq!(meta.contexts.len(), 2);
    }

    #[test]
    fn pim_manager_set_energy() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();
        let goal =
            super::super::goal::create_goal(&engine, "energy test", 128, "done", &agent_preds)
                .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.set_energy(&engine, goal.symbol_id, EnergyLevel::Low)
            .unwrap();

        assert_eq!(
            pim.get_metadata(goal.symbol_id.get()).unwrap().energy,
            Some(EnergyLevel::Low)
        );
    }

    #[test]
    fn pim_manager_task_not_found() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let pim = PimManager::new(&engine).unwrap();
        assert!(pim.get_metadata(99999).is_none());
    }

    #[test]
    fn pim_manager_task_count() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        assert_eq!(pim.task_count(), 0);

        let g1 = super::super::goal::create_goal(&engine, "t1", 128, "done", &agent_preds)
            .unwrap();
        pim.add_task(&engine, g1.symbol_id, GtdState::Inbox, 0.5, 0.5)
            .unwrap();

        assert_eq!(pim.task_count(), 1);
    }

    // ── Dependency DAG ───────────────────────────────────────────────

    #[test]
    fn dep_dag_add_and_topo() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let g1 =
            super::super::goal::create_goal(&engine, "first", 128, "done", &agent_preds).unwrap();
        let g2 = super::super::goal::create_goal(&engine, "second", 128, "done", &agent_preds)
            .unwrap();

        pim.add_task(&engine, g1.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.add_task(&engine, g2.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();

        pim.add_dependency(&engine, g1.symbol_id, g2.symbol_id)
            .unwrap();

        let topo = pim.topological_order().unwrap();
        let pos1 = topo.iter().position(|&s| s == g1.symbol_id).unwrap();
        let pos2 = topo.iter().position(|&s| s == g2.symbol_id).unwrap();
        assert!(pos1 < pos2, "blocker must come before blocked");
    }

    #[test]
    fn dep_dag_cycle_rejected() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let g1 =
            super::super::goal::create_goal(&engine, "a", 128, "done", &agent_preds).unwrap();
        let g2 =
            super::super::goal::create_goal(&engine, "b", 128, "done", &agent_preds).unwrap();

        pim.add_task(&engine, g1.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.add_task(&engine, g2.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();

        pim.add_dependency(&engine, g1.symbol_id, g2.symbol_id)
            .unwrap();

        // Reverse edge should create a cycle.
        assert!(pim
            .add_dependency(&engine, g2.symbol_id, g1.symbol_id)
            .is_err());
    }

    #[test]
    fn dep_dag_remove_dependency() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let g1 =
            super::super::goal::create_goal(&engine, "x", 128, "done", &agent_preds).unwrap();
        let g2 =
            super::super::goal::create_goal(&engine, "y", 128, "done", &agent_preds).unwrap();

        pim.add_task(&engine, g1.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.add_task(&engine, g2.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        pim.add_dependency(&engine, g1.symbol_id, g2.symbol_id)
            .unwrap();

        assert_eq!(pim.dep_dag.edge_count(), 1);
        pim.remove_dependency(g1.symbol_id, g2.symbol_id);
        assert_eq!(pim.dep_dag.edge_count(), 0);
    }

    // ── Weekly review ────────────────────────────────────────────────

    #[test]
    fn weekly_review_finds_overdue() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let goal = super::super::goal::create_goal(
            &engine,
            "overdue task",
            128,
            "done",
            &agent_preds,
        )
        .unwrap();

        pim.add_task(&engine, goal.symbol_id, GtdState::Next, 0.5, 0.5)
            .unwrap();
        if let Some(meta) = pim.get_metadata_mut(goal.symbol_id.get()) {
            meta.next_due = Some(1_000_000); // Way in the past.
        }

        let now = 1_700_000_000u64;
        let review = gtd_weekly_review(&pim, &[goal], &[], now);
        assert!(!review.overdue.is_empty());
    }

    #[test]
    fn weekly_review_summary_not_empty() {
        let pim = PimManager::default();
        let review = gtd_weekly_review(&pim, &[], &[], 1_700_000_000);
        assert!(!review.summary.is_empty());
    }

    // ── VSA encoding ─────────────────────────────────────────────────

    #[test]
    fn vsa_encode_priority_produces_vector() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let goal =
            super::super::goal::create_goal(&engine, "vsa test", 128, "done", &agent_preds)
                .unwrap();
        pim.add_task(&engine, goal.symbol_id, GtdState::Next, 0.7, 0.3)
            .unwrap();

        let vec = pim.encode_priority(engine.ops(), goal.symbol_id.get());
        assert!(vec.is_some());
    }

    #[test]
    fn vsa_similar_priority_returns_results() {
        let engine = Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        }).unwrap();
        let mut pim = PimManager::new(&engine).unwrap();
        let agent_preds = super::super::agent::AgentPredicates::init(&engine).unwrap();

        let g1 =
            super::super::goal::create_goal(&engine, "task A", 128, "done", &agent_preds).unwrap();
        let g2 =
            super::super::goal::create_goal(&engine, "task B", 128, "done", &agent_preds).unwrap();

        pim.add_task(&engine, g1.symbol_id, GtdState::Next, 0.8, 0.8)
            .unwrap();
        pim.add_task(&engine, g2.symbol_id, GtdState::Next, 0.8, 0.8)
            .unwrap();

        let similar = pim.find_similar_priority(engine.ops(), g1.symbol_id.get(), &[], 5);
        assert!(!similar.is_empty());
    }

    // ── PARA label roundtrip ─────────────────────────────────────────

    #[test]
    fn para_from_label_roundtrip() {
        for cat in [
            ParaCategory::Project,
            ParaCategory::Area,
            ParaCategory::Resource,
            ParaCategory::Archive,
        ] {
            assert_eq!(ParaCategory::from_label(cat.as_label()), Some(cat));
        }
    }

    // ── EnergyLevel label roundtrip ──────────────────────────────────

    #[test]
    fn energy_from_label_roundtrip() {
        for level in [EnergyLevel::Low, EnergyLevel::Medium, EnergyLevel::High] {
            assert_eq!(EnergyLevel::from_label(level.as_label()), Some(level));
        }
    }
}
