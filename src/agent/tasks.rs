//! Structured tasks (ADR 081 slice 6) — the deliberate exception to
//! append-only capture.
//!
//! Per [ADR 081](../../docs/ai/decisions/081-akh-memory-schema-typing.md):
//! "Tasks are the deliberate exception to append-only: an LLM-driven
//! structured task tool (grammar-constrained, schema-validated) whose store
//! is authoritative — no markdown parser." A `Task` is a regular KG `Entity`
//! symbol (no new `SymbolKind` variant) carrying well-known `task:`-namespaced
//! predicates, so it rides the existing WAL/snapshot machinery for free —
//! exactly like `CalendarEvent` (`calendar.rs`) and `Contact` (`contact.rs`).
//!
//! ## KG shape: the future spine
//!
//! The plan calls for `create` to project "to the episodic future-spine /
//! calendar / KG commitment," offering two shapes: a separate scheduled-future
//! episode entity, or the task entity itself carrying the date predicate. This
//! module picks **the task entity itself**: on `create_task`, the new `Task`
//! entity is *also* tagged with [`AgentPredicates::memory_type`] →
//! `episodic_memory` and [`AgentPredicates::has_tag`] → `tag:task` /
//! `tag:future` (when it carries a due or scheduled date), with a
//! `has_summary` excerpt of the title. That makes the commitment
//! immediately visible to the existing generic recall path
//! (`memory::recall_episodes` / `memory::recall_recent_episodes`) without
//! minting a second, redundant KG node for one commitment — recall surfaces
//! "Task: buy groceries" the same way it surfaces a past episode. The `task:`
//! predicates (state, priority, due, scheduled, recurrence, …) live
//! alongside on the same symbol and are what this module's own
//! `list`/`get`/`update`/`complete` operations read and write; the
//! `AgentPredicates` triples are only ever written once, at creation, purely
//! for cross-cutting recall.
//!
//! `complete_task` is different: it mints a genuinely **past**
//! `EpisodicEntry`-shaped entity (`episode:task-complete:{id}`, following the
//! `memory::persist_episode` / `memory_write` convention) recording that the
//! task was finished, and links the task to it via `has_episode`. If the task
//! carries a [`super::pim::Recurrence`], completion also spawns the next
//! occurrence as a fresh `Task` entity (new symbol, `spawned_from` linking
//! back to the completed one), so the store never mutates a recurring task's
//! due date in place — each occurrence is its own auditable entity.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::agent::AgentPredicates;
use super::pim::{PimError, Recurrence};

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the structured task subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum TaskError {
    #[error("task title must not be empty")]
    #[diagnostic(
        code(akh::agent::task::empty_title),
        help("Provide a non-empty `title` string.")
    )]
    EmptyTitle,

    #[error("invalid date for `{field}`: \"{value}\" (expected YYYY-MM-DD)")]
    #[diagnostic(
        code(akh::agent::task::invalid_date),
        help("Dates must be ISO-8601 calendar dates, e.g. \"2026-07-15\".")
    )]
    InvalidDate { field: String, value: String },

    #[error("task not found: {symbol_id}")]
    #[diagnostic(
        code(akh::agent::task::not_found),
        help("Use the `list` operation to see existing task ids, or `create` to make a new one.")
    )]
    NotFound { symbol_id: u64 },

    #[error("task \"{title}\" is already {state} — cannot complete it again")]
    #[diagnostic(
        code(akh::agent::task::already_terminal),
        help(
            "Only an Open task can be completed. Create a new task, or `update` a cancelled \
             task back to \"open\" first."
        )
    )]
    AlreadyTerminal { title: String, state: &'static str },

    #[error("invalid status \"{value}\": expected \"open\" or \"cancelled\"")]
    #[diagnostic(
        code(akh::agent::task::invalid_status),
        help(
            "`update` only toggles between \"open\" and \"cancelled\"; call the `complete` \
             operation to finish a task (it also handles recurrence)."
        )
    )]
    InvalidStatus { value: String },

    #[error("invalid priority \"{value}\": expected \"low\", \"medium\", or \"high\"")]
    #[diagnostic(
        code(akh::agent::task::invalid_priority),
        help("Priority must be one of: low, medium, high.")
    )]
    InvalidPriority { value: String },

    #[error("malformed task arguments: {message}")]
    #[diagnostic(
        code(akh::agent::task::bad_input),
        help(
            "Check the tool schema: `create` needs `title`; `update`/`complete` need `id` \
             (the task's symbol id, as returned by `create` or `list`)."
        )
    )]
    BadInput { message: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Recurrence(#[from] PimError),

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::task::engine),
        help("An engine-level error occurred during a task operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for TaskError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type TaskResult<T> = std::result::Result<T, TaskError>;

// ═══════════════════════════════════════════════════════════════════════
// Priority
// ═══════════════════════════════════════════════════════════════════════

/// Task priority — a small closed set, not a free-form score (unlike
/// `PimMetadata`'s continuous urgency/importance — tasks are simpler than
/// goals by design).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskPriority {
    Low,
    #[default]
    Medium,
    High,
}

impl TaskPriority {
    /// Parse from a case-insensitive string.
    pub fn parse(s: &str) -> TaskResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" | "normal" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(TaskError::InvalidPriority {
                value: other.to_string(),
            }),
        }
    }

    /// Stable label for KG storage.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Lifecycle — completion data only exists on `Done`, by construction
// ═══════════════════════════════════════════════════════════════════════

/// A task's lifecycle state. `Done` and `Cancelled` carry the data that only
/// makes sense once the transition happened — there is no way to construct a
/// "done task with no completion timestamp/episode" at the type level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskLifecycle {
    /// Not yet finished or abandoned.
    Open,
    /// Finished. Carries the past `EpisodicEntry` this completion minted and
    /// when it happened — see the module doc's "KG shape" section.
    Done {
        completed_at: u64,
        episode_id: SymbolId,
    },
    /// Abandoned without completing.
    Cancelled { cancelled_at: u64 },
}

// ═══════════════════════════════════════════════════════════════════════
// Task
// ═══════════════════════════════════════════════════════════════════════

/// A structured task, stored as a KG `Entity` symbol.
#[derive(Debug, Clone)]
pub struct Task {
    /// Stable id: the task's own `Entity` symbol. Never changes across
    /// `update`.
    pub symbol_id: SymbolId,
    pub title: String,
    pub lifecycle: TaskLifecycle,
    pub priority: TaskPriority,
    /// 📅 due date (unix seconds, day granularity — midnight UTC).
    pub due: Option<u64>,
    /// ⏳ scheduled date (unix seconds, day granularity).
    pub scheduled: Option<u64>,
    /// 🔁 recurrence rule. Present on both the template and every spawned
    /// occurrence, so the chain can keep extending itself.
    pub recurrence: Option<Recurrence>,
    /// When this task entity was created (symbol creation timestamp).
    pub created_at: u64,
    /// Optional note this task was captured from/for.
    pub source_note: Option<SymbolId>,
    /// The prior occurrence this task was spawned from, if any (recurrence chain).
    pub spawned_from: Option<SymbolId>,
}

impl Task {
    /// Whether this task is still open (neither done nor cancelled).
    pub fn is_open(&self) -> bool {
        matches!(self.lifecycle, TaskLifecycle::Open)
    }
}

/// Render a [`Recurrence`] as the short human phrase used in the emoji line
/// (e.g. `every week`, `every 3 days`) — distinct from `Recurrence::as_label`,
/// which renders the machine-parseable form (`weekly:mon,wed`).
fn recurrence_human(r: &Recurrence) -> String {
    match r {
        Recurrence::Daily => "every day".to_string(),
        Recurrence::EveryNDays(1) => "every day".to_string(),
        Recurrence::EveryNDays(n) => format!("every {n} days"),
        Recurrence::Weekly(days) if days.as_slice().is_empty() || days.as_slice().len() >= 7 => {
            "every week".to_string()
        }
        Recurrence::Weekly(days) => {
            let names: Vec<&str> = days
                .as_slice()
                .iter()
                .map(|d| match d {
                    0 => "Mon",
                    1 => "Tue",
                    2 => "Wed",
                    3 => "Thu",
                    4 => "Fri",
                    5 => "Sat",
                    _ => "Sun",
                })
                .collect();
            format!("every week on {}", names.join(","))
        }
        Recurrence::Monthly(day) => format!("every month on the {day}"),
        Recurrence::Yearly(month, day) => format!("every year on {month:02}-{day:02}"),
    }
}

/// Boundary rendering: Obsidian-Tasks emoji-line syntax
/// (`- [ ] title ⏳ 2026-07-10 📅 2026-07-12 🔁 every week`). This is a
/// **rendering only** — nothing in this crate parses this syntax back; the
/// structured store (this module) is the sole source of truth (ADR 081).
impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let checkbox = match self.lifecycle {
            TaskLifecycle::Open => ' ',
            TaskLifecycle::Done { .. } => 'x',
            TaskLifecycle::Cancelled { .. } => '-',
        };
        write!(f, "- [{checkbox}] {}", self.title)?;
        if let Some(ts) = self.scheduled {
            write!(f, " ⏳ {}", format_date(ts))?;
        }
        if let Some(ts) = self.due {
            write!(f, " 📅 {}", format_date(ts))?;
        }
        if let Some(rec) = &self.recurrence {
            write!(f, " 🔁 {}", recurrence_human(rec))?;
        }
        if let TaskLifecycle::Done { completed_at, .. } = &self.lifecycle {
            write!(f, " ✅ {}", format_date(*completed_at))?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Predicates — `task:` namespace
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG predicates for the task subsystem, following the
/// `CalendarPredicates`/`PimPredicates`/`ContactPredicates` idiom: a small
/// domain-owned set resolved once and reused, rather than growing the
/// generic [`AgentPredicates`] without bound. The `AgentPredicates` fields a
/// `Task` also carries (`memory_type`, `has_tag`, `has_summary`, `learned`)
/// are threaded through explicitly wherever they're needed — see the module
/// doc's "KG shape" section for why.
#[derive(Debug, Clone)]
pub struct TaskPredicates {
    /// Marks a symbol as a task (object is the `task:marker` entity); the
    /// enumeration anchor for `list_tasks`.
    pub task_type: SymbolId,
    pub has_title: SymbolId,
    pub has_state: SymbolId,
    pub has_priority: SymbolId,
    pub has_due: SymbolId,
    pub has_scheduled: SymbolId,
    pub has_recurrence: SymbolId,
    pub has_source_note: SymbolId,
    /// Completed task → its minted `EpisodicEntry`-shaped completion episode.
    pub has_episode: SymbolId,
    /// A spawned recurrence occurrence → the task it was spawned from.
    pub spawned_from: SymbolId,
}

impl TaskPredicates {
    /// Resolve or create all task predicates.
    pub fn init(engine: &Engine) -> TaskResult<Self> {
        Ok(Self {
            task_type: engine.resolve_or_create_relation("task:type")?,
            has_title: engine.resolve_or_create_relation("task:has-title")?,
            has_state: engine.resolve_or_create_relation("task:has-state")?,
            has_priority: engine.resolve_or_create_relation("task:has-priority")?,
            has_due: engine.resolve_or_create_relation("task:has-due")?,
            has_scheduled: engine.resolve_or_create_relation("task:has-scheduled")?,
            has_recurrence: engine.resolve_or_create_relation("task:has-recurrence")?,
            has_source_note: engine.resolve_or_create_relation("task:has-source-note")?,
            has_episode: engine.resolve_or_create_relation("task:has-episode")?,
            spawned_from: engine.resolve_or_create_relation("task:spawned-from")?,
        })
    }
}

/// Label of the marker entity every task links to via `task_type`, and the
/// enumeration anchor `list_tasks` walks backwards from.
const TASK_MARKER_LABEL: &str = "task:marker";

// ═══════════════════════════════════════════════════════════════════════
// Date helpers — no `chrono` (it's feature-gated behind `calendar`; this
// module must stay unconditional). Howard Hinnant's `civil_from_days` /
// `days_from_civil` algorithms (public-domain, proleptic Gregorian). A
// private copy of `civil_from_days` already exists in `agentic_chat.rs` for
// the same "chrono is optional" reason; duplicating the ~15-line pure
// function here (plus its inverse) keeps this module decoupled rather than
// reaching into an unrelated module for an implementation detail.
// ═══════════════════════════════════════════════════════════════════════

/// Days-since-epoch → civil (year, month, day). `days` is `0` for 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Civil (year, month, day) → days-since-epoch. Inverse of [`civil_from_days`].
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Format a unix-seconds timestamp as an ISO-8601 calendar date (`YYYY-MM-DD`).
fn format_date(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Parse an ISO-8601 calendar date (`YYYY-MM-DD`) into unix seconds at
/// midnight UTC. `field` names the argument for the diagnostic.
fn parse_date(field: &str, s: &str) -> TaskResult<u64> {
    let bad = || TaskError::InvalidDate {
        field: field.to_string(),
        value: s.to_string(),
    };
    let parts: Vec<&str> = s.trim().split('-').collect();
    let [y, m, d] = parts[..] else { return Err(bad()) };
    let y: i64 = y.parse().map_err(|_| bad())?;
    let m: u32 = m.parse().map_err(|_| bad())?;
    let d: u32 = d.parse().map_err(|_| bad())?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || y < 1970 {
        return Err(bad());
    }
    let days = days_from_civil(y, m, d);
    if days < 0 {
        return Err(bad());
    }
    Ok(days as u64 * 86_400)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Process-wide counter guaranteeing unique task labels even when two tasks
/// are created with the same title in the same second (the timestamp alone
/// is not enough on coarse clocks).
static TASK_SEQ: AtomicU64 = AtomicU64::new(0);

/// Lowercase a string into a compact label fragment (mirrors `contact.rs`'s
/// `slugify` / `memory_write.rs`'s `slug`).
fn slugify(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let collapsed: String = cleaned
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if collapsed.len() > 40 {
        collapsed[..40].to_string()
    } else if collapsed.is_empty() {
        "task".to_string()
    } else {
        collapsed
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Create
// ═══════════════════════════════════════════════════════════════════════

/// Arguments for [`create_task`].
#[derive(Debug, Clone, Default)]
pub struct NewTask {
    pub title: String,
    pub due: Option<u64>,
    pub scheduled: Option<u64>,
    pub recurrence: Option<Recurrence>,
    pub priority: TaskPriority,
    pub source_note: Option<SymbolId>,
}

/// Create a new task. Also projects it onto the episodic future-spine (see
/// the module doc) when it carries a due or scheduled date, so generic
/// recall surfaces the commitment.
pub fn create_task(
    engine: &Engine,
    preds: &TaskPredicates,
    agent_preds: &AgentPredicates,
    new: NewTask,
) -> TaskResult<Task> {
    if new.title.trim().is_empty() {
        return Err(TaskError::EmptyTitle);
    }

    let seq = TASK_SEQ.fetch_add(1, Ordering::Relaxed);
    let label = format!("task:{}:{}-{seq}", slugify(&new.title), now_secs());
    let symbol_id = engine.create_symbol(crate::symbol::SymbolKind::Entity, &label)?.id;

    let marker = engine.resolve_or_create_entity(TASK_MARKER_LABEL)?;
    engine.add_triple(&Triple::new(symbol_id, preds.task_type, marker))?;

    let title_sym = engine.resolve_or_create_entity(&format!("literal:{}", new.title))?;
    engine.add_triple(&Triple::new(symbol_id, preds.has_title, title_sym))?;

    let state_sym = engine.resolve_or_create_entity("task:state:open")?;
    engine.add_triple(&Triple::new(symbol_id, preds.has_state, state_sym))?;

    let priority_sym =
        engine.resolve_or_create_entity(&format!("task:priority:{}", new.priority.as_label()))?;
    engine.add_triple(&Triple::new(symbol_id, preds.has_priority, priority_sym))?;

    if let Some(due) = new.due {
        let due_sym = engine.resolve_or_create_entity(&format!("date:{}", format_date(due)))?;
        engine.add_triple(&Triple::new(symbol_id, preds.has_due, due_sym))?;
    }
    if let Some(scheduled) = new.scheduled {
        let sched_sym =
            engine.resolve_or_create_entity(&format!("date:{}", format_date(scheduled)))?;
        engine.add_triple(&Triple::new(symbol_id, preds.has_scheduled, sched_sym))?;
    }
    if let Some(rec) = &new.recurrence {
        let rec_sym = engine.resolve_or_create_entity(&format!("recur:{}", rec.as_label()))?;
        engine.add_triple(&Triple::new(symbol_id, preds.has_recurrence, rec_sym))?;
    }
    if let Some(note) = new.source_note {
        engine.add_triple(&Triple::new(symbol_id, preds.has_source_note, note))?;
    }

    // Episodic future-spine projection (module doc): tag as episodic memory
    // so generic recall finds it, without minting a second KG node.
    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
    engine.add_triple(&Triple::new(symbol_id, agent_preds.memory_type, mem_type_sym))?;
    let task_tag = engine.resolve_or_create_entity("tag:task")?;
    engine.add_triple(&Triple::new(symbol_id, agent_preds.has_tag, task_tag))?;
    if new.due.is_some() || new.scheduled.is_some() {
        let future_tag = engine.resolve_or_create_entity("tag:future")?;
        engine.add_triple(&Triple::new(symbol_id, agent_preds.has_tag, future_tag))?;
    }
    let excerpt = if new.title.len() > 80 { &new.title[..80] } else { &new.title };
    let summary_sym = engine.resolve_or_create_entity(&format!("summary:{excerpt}"))?;
    engine.add_triple(&Triple::new(symbol_id, agent_preds.has_summary, summary_sym))?;

    let mut prov = ProvenanceRecord::new(
        symbol_id,
        DerivationKind::AgentConsolidation {
            reason: format!("task created: {}", new.title),
            relevance_score: 1.0,
        },
    )
    .with_confidence(1.0);
    let _ = engine.store_provenance(&mut prov);

    reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Update
// ═══════════════════════════════════════════════════════════════════════

/// Status a task can be moved to via `update`. `Done` is deliberately not a
/// member — only [`complete_task`] can finish a task (it also handles
/// recurrence and the completion episode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateStatus {
    Open,
    Cancelled,
}

impl UpdateStatus {
    pub fn parse(s: &str) -> TaskResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "cancelled" | "canceled" => Ok(Self::Cancelled),
            other => Err(TaskError::InvalidStatus {
                value: other.to_string(),
            }),
        }
    }
}

/// Fields to change on [`update_task`]. Each `Some` replaces the field;
/// `None` leaves it unchanged. There is no way to *clear* an optional field
/// (due/scheduled/recurrence) in this slice — see the final report's
/// "deferred" list.
#[derive(Debug, Clone, Default)]
pub struct TaskUpdate {
    pub title: Option<String>,
    pub due: Option<u64>,
    pub scheduled: Option<u64>,
    pub recurrence: Option<Recurrence>,
    pub priority: Option<TaskPriority>,
    pub status: Option<UpdateStatus>,
}

/// Update fields on an existing task without changing its `symbol_id`.
pub fn update_task(
    engine: &Engine,
    preds: &TaskPredicates,
    symbol_id: SymbolId,
    update: TaskUpdate,
) -> TaskResult<Task> {
    let current = reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })?;

    if let Some(title) = &update.title {
        if title.trim().is_empty() {
            return Err(TaskError::EmptyTitle);
        }
        replace_single(engine, symbol_id, preds.has_title, &format!("literal:{title}"))?;
    }
    if let Some(due) = update.due {
        replace_single(engine, symbol_id, preds.has_due, &format!("date:{}", format_date(due)))?;
    }
    if let Some(scheduled) = update.scheduled {
        replace_single(
            engine,
            symbol_id,
            preds.has_scheduled,
            &format!("date:{}", format_date(scheduled)),
        )?;
    }
    if let Some(rec) = &update.recurrence {
        replace_single(
            engine,
            symbol_id,
            preds.has_recurrence,
            &format!("recur:{}", rec.as_label()),
        )?;
    }
    if let Some(priority) = update.priority {
        replace_single(
            engine,
            symbol_id,
            preds.has_priority,
            &format!("task:priority:{}", priority.as_label()),
        )?;
    }
    if let Some(status) = update.status {
        if matches!(current.lifecycle, TaskLifecycle::Done { .. }) {
            return Err(TaskError::AlreadyTerminal {
                title: current.title.clone(),
                state: "done",
            });
        }
        match status {
            UpdateStatus::Open => {
                replace_single(engine, symbol_id, preds.has_state, "task:state:open")?;
            }
            UpdateStatus::Cancelled => {
                // Encode the timestamp in the state object's own label
                // (`task:state:cancelled:{ts}`) so no extra predicate is needed.
                let with_ts = format!("task:state:cancelled:{}", now_secs());
                replace_single(engine, symbol_id, preds.has_state, &with_ts)?;
            }
        }
    }

    reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })
}

/// Retract the current triple(s) for `predicate` on `symbol_id` (if any) and
/// assert a new one pointing at `resolve_or_create_entity(new_label)`.
fn replace_single(
    engine: &Engine,
    symbol_id: SymbolId,
    predicate: SymbolId,
    new_label: &str,
) -> TaskResult<()> {
    let existing: Vec<SymbolId> = engine
        .triples_from(symbol_id)
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .map(|t| t.object)
        .collect();
    let new_obj = engine.resolve_or_create_entity(new_label)?;
    for obj in existing {
        if obj != new_obj {
            let _ = engine.remove_triple(symbol_id, predicate, obj);
        }
    }
    engine.add_triple(&Triple::new(symbol_id, predicate, new_obj))?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Complete
// ═══════════════════════════════════════════════════════════════════════

/// Result of [`complete_task`].
#[derive(Debug, Clone)]
pub struct CompleteOutcome {
    /// The now-`Done` task.
    pub task: Task,
    /// The past `EpisodicEntry`-shaped completion episode.
    pub episode_id: SymbolId,
    /// The freshly spawned next occurrence, if the task was recurring.
    pub next_occurrence: Option<Task>,
}

/// Mark a task done: mints a past completion episode and, if the task
/// recurs, spawns the next occurrence.
pub fn complete_task(
    engine: &Engine,
    preds: &TaskPredicates,
    agent_preds: &AgentPredicates,
    symbol_id: SymbolId,
) -> TaskResult<CompleteOutcome> {
    let current = reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })?;

    match &current.lifecycle {
        TaskLifecycle::Done { .. } => {
            return Err(TaskError::AlreadyTerminal {
                title: current.title.clone(),
                state: "done",
            });
        }
        TaskLifecycle::Cancelled { .. } => {
            return Err(TaskError::AlreadyTerminal {
                title: current.title.clone(),
                state: "cancelled",
            });
        }
        TaskLifecycle::Open => {}
    }

    let completed_at = now_secs();

    // 1. Mint the past completion episode (memory::persist_episode /
    //    memory_write.rs convention): a fresh Entity, tagged episodic_memory,
    //    summarized, provenance-tracked.
    let episode_label = format!("episode:task-complete:{}:{completed_at}", symbol_id.get());
    let episode_id = engine
        .create_symbol(crate::symbol::SymbolKind::Entity, &episode_label)?
        .id;

    let summary_text = format!("Completed: {}", current.title);
    let excerpt = if summary_text.len() > 80 { &summary_text[..80] } else { &summary_text };
    let summary_sym = engine.resolve_or_create_entity(&format!("summary:{excerpt}"))?;
    engine.add_triple(&Triple::new(episode_id, agent_preds.has_summary, summary_sym))?;

    let complete_tag = engine.resolve_or_create_entity("tag:task_complete")?;
    engine.add_triple(&Triple::new(episode_id, agent_preds.has_tag, complete_tag))?;

    let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
    engine.add_triple(&Triple::new(episode_id, agent_preds.memory_type, mem_type_sym))?;

    engine.add_triple(&Triple::new(episode_id, agent_preds.learned, symbol_id))?;

    let mut prov = ProvenanceRecord::new(
        episode_id,
        DerivationKind::AgentConsolidation {
            reason: format!("task_complete:{}", symbol_id.get()),
            relevance_score: 1.0,
        },
    )
    .with_sources(vec![symbol_id])
    .with_confidence(1.0);
    let _ = engine.store_provenance(&mut prov);

    // 2. Flip the task to Done: state + link to the completion episode.
    replace_single(
        engine,
        symbol_id,
        preds.has_state,
        &format!("task:state:done:{completed_at}"),
    )?;
    engine.add_triple(&Triple::new(symbol_id, preds.has_episode, episode_id))?;

    let task = reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })?;

    // 3. Spawn the next occurrence if recurring.
    let next_occurrence = match &current.recurrence {
        Some(rec) => {
            let anchor = current.due.or(current.scheduled).unwrap_or(completed_at);
            let next_due = current.due.map(|_| rec.next_occurrence(anchor));
            let next_scheduled = current.scheduled.map(|_| rec.next_occurrence(anchor));
            let spawned = create_task(
                engine,
                preds,
                agent_preds,
                NewTask {
                    title: current.title.clone(),
                    due: next_due,
                    scheduled: next_scheduled,
                    recurrence: Some(rec.clone()),
                    priority: current.priority,
                    source_note: current.source_note,
                },
            )?;
            engine.add_triple(&Triple::new(spawned.symbol_id, preds.spawned_from, symbol_id))?;
            Some(reconstruct_task(engine, preds, spawned.symbol_id)?.unwrap_or(spawned))
        }
        None => None,
    };

    Ok(CompleteOutcome {
        task,
        episode_id,
        next_occurrence,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// List / Get
// ═══════════════════════════════════════════════════════════════════════

/// Filter for [`list_tasks`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TaskListFilter {
    #[default]
    Open,
    Done,
    Cancelled,
    All,
}

impl TaskListFilter {
    pub fn parse(s: &str) -> TaskResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "done" => Ok(Self::Done),
            "cancelled" | "canceled" => Ok(Self::Cancelled),
            "all" => Ok(Self::All),
            other => Err(TaskError::BadInput {
                message: format!(
                    "unknown filter \"{other}\": expected open, done, cancelled, or all"
                ),
            }),
        }
    }

    fn matches(self, lifecycle: &TaskLifecycle) -> bool {
        match self {
            Self::All => true,
            Self::Open => matches!(lifecycle, TaskLifecycle::Open),
            Self::Done => matches!(lifecycle, TaskLifecycle::Done { .. }),
            Self::Cancelled => matches!(lifecycle, TaskLifecycle::Cancelled { .. }),
        }
    }
}

/// List tasks matching `filter`, most-recently-created first.
pub fn list_tasks(
    engine: &Engine,
    preds: &TaskPredicates,
    filter: TaskListFilter,
) -> TaskResult<Vec<Task>> {
    let marker = match engine.resolve_symbol(TASK_MARKER_LABEL) {
        Ok(id) => id,
        Err(_) => return Ok(Vec::new()),
    };
    let incoming = engine.triples_to(marker);
    let mut tasks: Vec<Task> = incoming
        .iter()
        .filter(|t| t.predicate == preds.task_type)
        .filter_map(|t| reconstruct_task(engine, preds, t.subject).ok().flatten())
        .filter(|task| filter.matches(&task.lifecycle))
        .collect();
    tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.symbol_id.get().cmp(&a.symbol_id.get())));
    Ok(tasks)
}

/// Fetch a single task by id.
pub fn get_task(engine: &Engine, preds: &TaskPredicates, symbol_id: SymbolId) -> TaskResult<Task> {
    reconstruct_task(engine, preds, symbol_id)?.ok_or(TaskError::NotFound {
        symbol_id: symbol_id.get(),
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Reconstruction
// ═══════════════════════════════════════════════════════════════════════

/// Reconstruct a [`Task`] from its KG triples. Returns `Ok(None)` (not an
/// error) when `symbol_id` is not a task at all — mirrors
/// `memory::reconstruct_episode`'s "no such episode" convention.
fn reconstruct_task(
    engine: &Engine,
    preds: &TaskPredicates,
    symbol_id: SymbolId,
) -> TaskResult<Option<Task>> {
    let triples = engine.triples_from(symbol_id);
    if triples.is_empty() || !triples.iter().any(|t| t.predicate == preds.task_type) {
        return Ok(None);
    }

    let mut title = String::new();
    let mut state_label = "task:state:open".to_string();
    let mut priority = TaskPriority::default();
    let mut due = None;
    let mut scheduled = None;
    let mut recurrence = None;
    let mut source_note = None;
    let mut episode_id = None;
    let mut spawned_from = None;

    for t in &triples {
        let p = t.predicate;
        if p == preds.has_title {
            let raw = engine.resolve_label(t.object);
            title = raw.strip_prefix("literal:").unwrap_or(&raw).to_string();
        } else if p == preds.has_state {
            state_label = engine.resolve_label(t.object);
        } else if p == preds.has_priority {
            let raw = engine.resolve_label(t.object);
            if let Some(rest) = raw.strip_prefix("task:priority:") {
                priority = TaskPriority::parse(rest).unwrap_or_default();
            }
        } else if p == preds.has_due {
            let raw = engine.resolve_label(t.object);
            if let Some(date) = raw.strip_prefix("date:") {
                due = parse_date("due", date).ok();
            }
        } else if p == preds.has_scheduled {
            let raw = engine.resolve_label(t.object);
            if let Some(date) = raw.strip_prefix("date:") {
                scheduled = parse_date("scheduled", date).ok();
            }
        } else if p == preds.has_recurrence {
            let raw = engine.resolve_label(t.object);
            if let Some(rule) = raw.strip_prefix("recur:") {
                recurrence = Recurrence::from_label(rule).ok();
            }
        } else if p == preds.has_source_note {
            source_note = Some(t.object);
        } else if p == preds.has_episode {
            episode_id = Some(t.object);
        } else if p == preds.spawned_from {
            spawned_from = Some(t.object);
        }
    }

    // Parse "task:state:{open|done:{ts}|cancelled:{ts}}".
    let rest = state_label.strip_prefix("task:state:").unwrap_or("open");
    let lifecycle = if let Some(ts_str) = rest.strip_prefix("done:") {
        let completed_at = ts_str.parse().unwrap_or(0);
        TaskLifecycle::Done {
            completed_at,
            episode_id: episode_id.unwrap_or(symbol_id),
        }
    } else if let Some(ts_str) = rest.strip_prefix("cancelled:") {
        TaskLifecycle::Cancelled {
            cancelled_at: ts_str.parse().unwrap_or(0),
        }
    } else {
        TaskLifecycle::Open
    };

    let created_at = engine
        .get_symbol_meta(symbol_id)
        .map(|m| m.created_at)
        .unwrap_or(0);

    Ok(Some(Task {
        symbol_id,
        title,
        lifecycle,
        priority,
        due,
        scheduled,
        recurrence,
        created_at,
        source_note,
        spawned_from,
    }))
}

// ═══════════════════════════════════════════════════════════════════════
// Public date parsing (used by the tool layer to turn "YYYY-MM-DD" args
// into the `u64` the domain types carry).
// ═══════════════════════════════════════════════════════════════════════

/// Parse a `"YYYY-MM-DD"` string into unix seconds (public wrapper of the
/// internal date parser, for the tool layer).
pub fn parse_task_date(field: &str, s: &str) -> TaskResult<u64> {
    parse_date(field, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn preds(engine: &Engine) -> (TaskPredicates, AgentPredicates) {
        (
            TaskPredicates::init(engine).unwrap(),
            AgentPredicates::init(engine).unwrap(),
        )
    }

    #[test]
    fn date_roundtrip() {
        for (y, m, d) in [(1970, 1, 1), (2026, 7, 15), (2000, 2, 29), (2023, 11, 14)] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d));
        }
        assert_eq!(format_date(0), "1970-01-01");
        assert_eq!(parse_date("due", "1970-01-01").unwrap(), 0);
        assert_eq!(format_date(parse_date("due", "2026-07-15").unwrap()), "2026-07-15");
    }

    #[test]
    fn invalid_date_is_a_diagnostic() {
        let err = parse_date("due", "not-a-date").unwrap_err();
        assert!(matches!(err, TaskError::InvalidDate { .. }));
        let err2 = parse_date("due", "2026-13-40").unwrap_err();
        assert!(matches!(err2, TaskError::InvalidDate { .. }));
    }

    #[test]
    fn create_then_list_round_trip() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);

        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Buy groceries".into(),
                due: Some(parse_date("due", "2026-07-15").unwrap()),
                scheduled: None,
                recurrence: None,
                priority: TaskPriority::High,
                source_note: None,
            },
        )
        .unwrap();

        assert_eq!(task.title, "Buy groceries");
        assert!(task.is_open());
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.due.map(format_date), Some("2026-07-15".to_string()));

        let listed = list_tasks(&engine, &tp, TaskListFilter::Open).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].symbol_id, task.symbol_id);

        // Also discoverable via the generic episodic recall path (future-spine).
        let recent = crate::agent::memory::recall_recent_episodes(&engine, &ap, 10).unwrap();
        assert!(recent.iter().any(|e| e.symbol_id == task.symbol_id));
    }

    #[test]
    fn create_rejects_empty_title() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let err = create_task(&engine, &tp, &ap, NewTask::default()).unwrap_err();
        assert!(matches!(err, TaskError::EmptyTitle));
    }

    #[test]
    fn update_changes_fields_without_changing_id() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Draft report".into(),
                priority: TaskPriority::Low,
                ..Default::default()
            },
        )
        .unwrap();
        let id_before = task.symbol_id;

        let updated = update_task(
            &engine,
            &tp,
            task.symbol_id,
            TaskUpdate {
                title: Some("Draft final report".into()),
                priority: Some(TaskPriority::High),
                due: Some(parse_date("due", "2026-08-01").unwrap()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(updated.symbol_id, id_before);
        assert_eq!(updated.title, "Draft final report");
        assert_eq!(updated.priority, TaskPriority::High);
        assert_eq!(updated.due.map(format_date), Some("2026-08-01".to_string()));

        // Re-updating due date doesn't leave the old one behind.
        let updated2 = update_task(
            &engine,
            &tp,
            task.symbol_id,
            TaskUpdate {
                due: Some(parse_date("due", "2026-09-01").unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(updated2.due.map(format_date), Some("2026-09-01".to_string()));
        let due_triples: Vec<_> = engine
            .triples_from(task.symbol_id)
            .into_iter()
            .filter(|t| t.predicate == tp.has_due)
            .collect();
        assert_eq!(due_triples.len(), 1, "stale due triple must be retracted");
    }

    #[test]
    fn update_status_parse_rejects_done() {
        let err = UpdateStatus::parse("done").unwrap_err();
        assert!(matches!(err, TaskError::InvalidStatus { .. }));
    }

    #[test]
    fn complete_mints_episode_and_flips_state() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Deploy service X".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let outcome = complete_task(&engine, &tp, &ap, task.symbol_id).unwrap();
        assert!(matches!(outcome.task.lifecycle, TaskLifecycle::Done { .. }));
        assert!(outcome.next_occurrence.is_none(), "non-recurring task spawns nothing");

        // The completion episode exists, is tagged, and is recallable.
        let recent = crate::agent::memory::recall_recent_episodes(&engine, &ap, 10).unwrap();
        assert!(recent.iter().any(|e| e.symbol_id == outcome.episode_id));
        assert!(engine.resolve_label(outcome.episode_id).starts_with("episode:task-complete:"));

        // Completing again is an error, not a corrupt double-completion.
        let err = complete_task(&engine, &tp, &ap, task.symbol_id).unwrap_err();
        assert!(matches!(err, TaskError::AlreadyTerminal { state: "done", .. }));
    }

    #[test]
    fn recurrence_spawns_exactly_one_next_occurrence() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let due = parse_date("due", "2026-07-01").unwrap();
        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Weekly sync".into(),
                due: Some(due),
                // `Recurrence::Weekly(vec![])` does not round-trip through its own
                // `as_label`/`from_label` (pim.rs encodes it as the bare "weekly:"
                // label, which `parse` then rejects as "unknown day: \"\"" — a
                // pre-existing quirk in `pim::Recurrence`, out of scope here); use
                // `Daily` so this test exercises the KG round-trip, not that bug.
                recurrence: Some(Recurrence::Daily),
                ..Default::default()
            },
        )
        .unwrap();

        let outcome = complete_task(&engine, &tp, &ap, task.symbol_id).unwrap();
        let next = outcome.next_occurrence.expect("recurring task must spawn a next occurrence");
        assert_ne!(next.symbol_id, task.symbol_id);
        assert_eq!(next.title, "Weekly sync");
        assert!(next.due.unwrap() > due, "next occurrence's due date must advance");
        assert_eq!(next.spawned_from, Some(task.symbol_id));

        // Exactly one next occurrence exists in the store.
        let open = list_tasks(&engine, &tp, TaskListFilter::Open).unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].symbol_id, next.symbol_id);
    }

    #[test]
    fn cancel_via_update_then_relist() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Obsolete idea".into(),
                ..Default::default()
            },
        )
        .unwrap();

        let cancelled = update_task(
            &engine,
            &tp,
            task.symbol_id,
            TaskUpdate {
                status: Some(UpdateStatus::Cancelled),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(cancelled.lifecycle, TaskLifecycle::Cancelled { .. }));

        let open = list_tasks(&engine, &tp, TaskListFilter::Open).unwrap();
        assert!(open.is_empty());
        let cancelled_list = list_tasks(&engine, &tp, TaskListFilter::Cancelled).unwrap();
        assert_eq!(cancelled_list.len(), 1);
    }

    #[test]
    fn render_matches_expected_emoji_syntax() {
        let engine = test_engine();
        let (tp, ap) = preds(&engine);
        let task = create_task(
            &engine,
            &tp,
            &ap,
            NewTask {
                title: "Ship the release".into(),
                due: Some(parse_date("due", "2026-07-12").unwrap()),
                recurrence: Some(Recurrence::Daily),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            task.to_string(),
            "- [ ] Ship the release 📅 2026-07-12 🔁 every day"
        );

        let outcome = complete_task(&engine, &tp, &ap, task.symbol_id).unwrap();
        let rendered = outcome.task.to_string();
        assert!(rendered.starts_with("- [x] Ship the release 📅 2026-07-12 🔁 every day ✅ "));
    }

    #[test]
    fn get_task_not_found_is_a_diagnostic() {
        let engine = test_engine();
        let (tp, _ap) = preds(&engine);
        let bogus = SymbolId::new(999_999).unwrap();
        let err = get_task(&engine, &tp, bogus).unwrap_err();
        assert!(matches!(err, TaskError::NotFound { .. }));
    }
}
