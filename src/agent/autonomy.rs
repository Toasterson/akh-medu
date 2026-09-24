//! Budgeted autonomy for the background daemon (Phase 53b, ADR 113 decision 6).
//!
//! ADR 113 makes every akh fully featured: all background tasks run by
//! default. What bounds their cost is a per-akh daily budget, not
//! compiling features out. This module holds the types for that:
//!
//! - [`DaemonTask`]: the typed registry of every daemon task. It replaces
//!   the old free-string `KNOWN_TASK_NAMES`. The `[autonomy] tasks` keys
//!   stay the same strings ([`DaemonTask::as_str`]).
//! - [`AutonomyBudget`]: the `[autonomy.budget]` limits. The defaults are
//!   250 000 tokens and 40 turns per UTC day.
//! - [`BudgetLedger`] and [`AutonomyGovernor`]: the per-akh daily spend,
//!   persisted in the engine's durable metadata under [`BUDGET_LEDGER_KEY`].
//!   An engine without a durable store (in-memory, tests) keeps the ledger in
//!   memory only.
//! - [`DaemonTaskReport`]: the per-run telemetry record that every daemon
//!   task produces, logged under target `akh::autonomy` and exported as the
//!   `akh_daemon_task_*` metrics.
//!
//! # What a "turn" is
//!
//! One turn is one admitted LLM-cost unit of work that reached a model:
//!
//! - one run of a per-run task (librarian, knowledge elicitation, knowledge
//!   extraction with an external API) during which at least one model request
//!   was metered, or
//! - one reactor occurrence whose model turn (`LlmTurn`, `DjehutyReply`) got
//!   a response, even if the backing does not meter its requests. The
//!   reactor admits the turn lazily, right before the model call, so a
//!   cached result or a filtered board message never waits for the budget.
//!
//! A task that is admitted but finds nothing to do (no targets, backend
//! unavailable) costs nothing. Tokens are whatever the thread-local
//! [`UsageScope`](super::llm_metrics::UsageScope) metered while the task ran.
//! Admission checks the ledger *before* the work. A run that makes many
//! model calls (the librarian) also checks
//! [`AutonomyGovernor::check_in_flight`] before each one, so it overshoots
//! the token limit by at most one call. The next admission is then refused
//! until the UTC day rolls over.
//!
//! Only a request that got a response counts
//! ([`RequestOutcome`](super::llm_metrics::RequestOutcome)): transport
//! errors and an unavailable backend are free.

use std::fmt;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::llm_metrics::LlmUsage;
use crate::akh_config::ExecBackendConfig;
use crate::engine::Engine;

/// Durable-metadata key of the per-akh budget ledger. The `v1` suffix
/// versions the JSON shape of [`BudgetLedger`].
pub const BUDGET_LEDGER_KEY: &[u8] = b"autonomy:budget:v1";

/// Upper bound on harvest batches per sleep tick (ADR 113 decision 5).
/// Each batch is at most `harvest::HARVEST_BATCH_LIMIT` turns, so one tick
/// drains up to 1 024 captured turns.
pub const HARVEST_MAX_BATCHES_PER_TICK: usize = 4;

const SECONDS_PER_DAY: u64 = 86_400;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the autonomy budget.
#[derive(Debug, Error, Diagnostic)]
pub enum AutonomyError {
    /// The persisted ledger could not be decoded.
    #[error("the autonomy budget ledger at `{key}` is corrupt: {source}")]
    #[diagnostic(
        code(akh::autonomy::corrupt_ledger),
        help(
            "The daemon resets the ledger to zero spend for today and overwrites it. \
             To inspect it first, stop the akh and read the durable meta key `{key}`. \
             Resetting grants at most one extra day of budget."
        )
    )]
    CorruptLedger {
        /// The metadata key that failed to decode.
        key: String,
        /// The decode error.
        #[source]
        source: serde_json::Error,
    },

    /// Reading the ledger from the durable store failed.
    #[error("failed to read the autonomy budget ledger at `{key}`")]
    #[diagnostic(
        code(akh::autonomy::ledger_read),
        help(
            "The engine's durable store did not answer. Check the akh's data directory \
             (disk full, permissions, or a redb lock held by another process). The daemon \
             starts with zero spend for today when this happens."
        )
    )]
    LedgerRead {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// Writing the ledger to the durable store failed.
    #[error("failed to persist the autonomy budget ledger at `{key}`")]
    #[diagnostic(
        code(akh::autonomy::ledger_write),
        help(
            "The in-process ledger is still enforced, so today's budget holds until the \
             daemon restarts. A restart before the write succeeds loses today's spend \
             record. Check disk space and permissions on the akh's data directory."
        )
    )]
    LedgerWrite {
        /// The metadata key.
        key: String,
        /// The store error.
        #[source]
        source: crate::error::StoreError,
    },

    /// Encoding the ledger failed (should not happen for this plain struct).
    #[error("failed to encode the autonomy budget ledger: {source}")]
    #[diagnostic(
        code(akh::autonomy::ledger_encode),
        help("This is a bug: BudgetLedger is plain data. Report it with the daemon log.")
    )]
    LedgerEncode {
        /// The encode error.
        #[source]
        source: serde_json::Error,
    },
}

/// Result alias for the autonomy budget.
pub type AutonomyResult<T> = std::result::Result<T, AutonomyError>;

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------

/// Every background task the [`AgentDaemon`](super::AgentDaemon) schedules.
///
/// The string names ([`DaemonTask::as_str`]) are the keys of the
/// `[autonomy] tasks` table and the `task` metric label. They are stable:
/// renaming one breaks existing `akh.toml` files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DaemonTask {
    /// Equivalence learning over new triples.
    Equivalence,
    /// Reflection over recent cycles.
    Reflection,
    /// Working-memory consolidation.
    Consolidation,
    /// Schema discovery.
    SchemaDiscovery,
    /// Forward-chaining rule inference.
    RuleInference,
    /// Knowledge-gap analysis around active goals.
    GapAnalysis,
    /// Idle OODA cycle (tool use). On by default only when the executor is
    /// contained (ADR 113 decision 4).
    IdleCycle,
    /// Continuous learning (network fetch and ingestion).
    ContinuousLearning,
    /// Drive-based goal generation.
    GoalGeneration,
    /// Sleep cycle: dedup, orphan pruning, dream connections.
    SleepCycle,
    /// Transcript harvest of captured turns into episode indexes (ADR 091).
    /// Runs on the sleep tick, before and independent of `SleepCycle`.
    Harvest,
    /// Triple extraction from fetched content. It spends LLM tokens only
    /// when `[extraction.external_api]` is configured.
    KnowledgeExtraction,
    /// Neural bridge training.
    BridgeTraining,
    /// OpenClaw journal sync.
    OpenclawSync,
    /// LLM knowledge elicitation for sparse domains (Phase 32).
    KnowledgeElicitation,
    /// KG scaffolding compaction.
    KgCompaction,
    /// Delegation queue processing.
    Delegation,
    /// Relation enrichment.
    RelationEnrichment,
    /// Daily Librarian pass (ADR 081). It is also gated on `AKH_LIBRARIAN=1`
    /// until WI-298 lands.
    Librarian,
    /// Seshat's Archive curation.
    SeshatCuration,
    /// Heka store curation (surveillance only).
    HekaCuration,
    /// Session persistence. Always on (infrastructure).
    Persist,
    /// User-authored trigger evaluation. Always on: triggers are opt-in by
    /// existing.
    TriggerEvaluation,
    /// Durable reactor schedules. Always on: schedules are opt-in by existing.
    /// Its model turns are budgeted per occurrence.
    Reactor,
}

impl DaemonTask {
    /// Every task, in daemon select-arm order.
    pub const ALL: [DaemonTask; 24] = [
        DaemonTask::Equivalence,
        DaemonTask::Reflection,
        DaemonTask::Consolidation,
        DaemonTask::SchemaDiscovery,
        DaemonTask::RuleInference,
        DaemonTask::GapAnalysis,
        DaemonTask::Persist,
        DaemonTask::IdleCycle,
        DaemonTask::ContinuousLearning,
        DaemonTask::GoalGeneration,
        DaemonTask::Harvest,
        DaemonTask::SleepCycle,
        DaemonTask::TriggerEvaluation,
        DaemonTask::Reactor,
        DaemonTask::KnowledgeExtraction,
        DaemonTask::BridgeTraining,
        DaemonTask::OpenclawSync,
        DaemonTask::KnowledgeElicitation,
        DaemonTask::KgCompaction,
        DaemonTask::Delegation,
        DaemonTask::RelationEnrichment,
        DaemonTask::Librarian,
        DaemonTask::SeshatCuration,
        DaemonTask::HekaCuration,
    ];

    /// The stable config key and metric label.
    pub const fn as_str(self) -> &'static str {
        match self {
            DaemonTask::Equivalence => "equivalence",
            DaemonTask::Reflection => "reflection",
            DaemonTask::Consolidation => "consolidation",
            DaemonTask::SchemaDiscovery => "schema_discovery",
            DaemonTask::RuleInference => "rule_inference",
            DaemonTask::GapAnalysis => "gap_analysis",
            DaemonTask::IdleCycle => "idle_cycle",
            DaemonTask::ContinuousLearning => "continuous_learning",
            DaemonTask::GoalGeneration => "goal_generation",
            DaemonTask::SleepCycle => "sleep_cycle",
            DaemonTask::Harvest => "harvest",
            DaemonTask::KnowledgeExtraction => "knowledge_extraction",
            DaemonTask::BridgeTraining => "bridge_training",
            DaemonTask::OpenclawSync => "openclaw_sync",
            DaemonTask::KnowledgeElicitation => "knowledge_elicitation",
            DaemonTask::KgCompaction => "kg_compaction",
            DaemonTask::Delegation => "delegation",
            DaemonTask::RelationEnrichment => "relation_enrichment",
            DaemonTask::Librarian => "librarian",
            DaemonTask::SeshatCuration => "seshat_curation",
            DaemonTask::HekaCuration => "heka_curation",
            DaemonTask::Persist => "persist",
            DaemonTask::TriggerEvaluation => "trigger_evaluation",
            DaemonTask::Reactor => "reactor",
        }
    }

    /// What running this task can cost, at most.
    ///
    /// `KnowledgeExtraction` is classed `Llm` because it can call an external
    /// API. The daemon meters it only when one is configured.
    pub const fn cost(self) -> TaskCost {
        match self {
            DaemonTask::Librarian
            | DaemonTask::KnowledgeExtraction
            | DaemonTask::KnowledgeElicitation
            | DaemonTask::Reactor => TaskCost::Llm,
            DaemonTask::ContinuousLearning
            | DaemonTask::OpenclawSync
            | DaemonTask::SeshatCuration
            | DaemonTask::HekaCuration
            | DaemonTask::Delegation
            | DaemonTask::IdleCycle => TaskCost::Network,
            DaemonTask::Equivalence
            | DaemonTask::Reflection
            | DaemonTask::Consolidation
            | DaemonTask::SchemaDiscovery
            | DaemonTask::RuleInference
            | DaemonTask::GapAnalysis
            | DaemonTask::GoalGeneration
            | DaemonTask::SleepCycle
            | DaemonTask::Harvest
            | DaemonTask::BridgeTraining
            | DaemonTask::KgCompaction
            | DaemonTask::RelationEnrichment
            | DaemonTask::Persist
            | DaemonTask::TriggerEvaluation => TaskCost::Free,
        }
    }

    /// Whether `[autonomy]` can switch this task off. Infrastructure tasks
    /// always run.
    ///
    /// Backward compatibility with pre-53b TOMLs (`[autonomy] enabled =
    /// false`): harvest used to run inside the sleep cycle, so without its
    /// own override it follows the sleep cycle's gate; the librarian and
    /// OpenClaw sync were only ever env-gated, so the master switch does not
    /// turn them off.
    pub const fn gating(self) -> TaskGating {
        match self {
            DaemonTask::Persist | DaemonTask::TriggerEvaluation | DaemonTask::Reactor => {
                TaskGating::AlwaysOn
            }
            DaemonTask::Harvest => TaskGating::FollowsTask(DaemonTask::SleepCycle),
            DaemonTask::Librarian | DaemonTask::OpenclawSync => TaskGating::IgnoresMasterSwitch,
            _ => TaskGating::Autonomy,
        }
    }

    /// When the daemon first runs this task after a start or wake.
    ///
    /// Cheap infrastructure (reactor, triggers, persistence) runs at once.
    /// The sleep tick (harvest, then the sleep cycle) runs after a short
    /// [`STARTUP_DELAY`] so a freshly started akh (the Roman cutover) harvests
    /// soon without competing with its own wake. Everything else, including
    /// the heavy tasks (continuous learning holds the agent write lock for
    /// 30–120 s, knowledge elicitation loads a GGUF, the librarian calls a
    /// model per source), waits one full period.
    pub const fn first_tick(self) -> FirstTick {
        match self {
            DaemonTask::Reactor | DaemonTask::TriggerEvaluation | DaemonTask::Persist => {
                FirstTick::Immediate
            }
            DaemonTask::Harvest | DaemonTask::SleepCycle => FirstTick::AfterStartupDelay,
            _ => FirstTick::AfterOnePeriod,
        }
    }

    /// How the daily budget meters this task at most. The daemon may lower
    /// it to [`BudgetMetering::Unmetered`] (knowledge extraction without an
    /// external API).
    pub const fn metering(self) -> BudgetMetering {
        match self {
            DaemonTask::Reactor => BudgetMetering::PerOccurrence,
            DaemonTask::Librarian
            | DaemonTask::KnowledgeExtraction
            | DaemonTask::KnowledgeElicitation => BudgetMetering::PerRun,
            _ => BudgetMetering::Unmetered,
        }
    }
}

impl fmt::Display for DaemonTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A task name that is not a [`DaemonTask`].
#[derive(Debug, Clone, PartialEq, Eq, Error, Diagnostic)]
#[error("unknown daemon task `{name}`")]
#[diagnostic(
    code(akh::autonomy::unknown_task),
    help(
        "Valid `[autonomy] tasks` keys are the DaemonTask names, for example `reflection`, `harvest` or `librarian`."
    )
)]
pub struct UnknownDaemonTask {
    /// The unrecognised name.
    pub name: String,
}

impl FromStr for DaemonTask {
    type Err = UnknownDaemonTask;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DaemonTask::ALL
            .into_iter()
            .find(|task| task.as_str() == s)
            .ok_or_else(|| UnknownDaemonTask { name: s.to_owned() })
    }
}

/// Delay before the sleep tick's first run after a daemon start (see
/// [`DaemonTask::first_tick`]).
pub const STARTUP_DELAY: Duration = Duration::from_secs(60);

/// When a task's ticker first fires after a daemon start or wake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstTick {
    /// Right away.
    Immediate,
    /// After [`STARTUP_DELAY`] (or one period, if that is shorter).
    AfterStartupDelay,
    /// After one full period.
    AfterOnePeriod,
}

impl FirstTick {
    /// The delay before the first tick for a task that repeats every
    /// `period`.
    pub fn delay(self, period: Duration) -> Duration {
        match self {
            FirstTick::Immediate => Duration::ZERO,
            FirstTick::AfterStartupDelay => STARTUP_DELAY.min(period),
            FirstTick::AfterOnePeriod => period,
        }
    }
}

/// What running a task can cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskCost {
    /// Local CPU and store work only.
    Free,
    /// Network I/O, but no LLM tokens.
    Network,
    /// Can spend LLM tokens and therefore counts against the daily budget.
    Llm,
}

/// Whether a task consults `[autonomy]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskGating {
    /// Infrastructure. It always runs.
    AlwaysOn,
    /// Follows `[autonomy]` (master switch, per-task override).
    Autonomy,
    /// Its own `tasks` override wins; without one it follows the gate of
    /// the named task (harvest follows the sleep cycle).
    FollowsTask(DaemonTask),
    /// Its own `tasks` override wins; the master switch does not turn it
    /// off. Its environment gate still applies (librarian, OpenClaw sync).
    IgnoresMasterSwitch,
}

/// How the daily budget meters a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetMetering {
    /// Not metered.
    Unmetered,
    /// One admission per run, taken by the daemon's task wrapper.
    PerRun,
    /// The task admits each model occurrence itself (the reactor, per claim).
    PerOccurrence,
}

// ---------------------------------------------------------------------------
// Gating
// ---------------------------------------------------------------------------

/// Where the akh's executor-facing tools run, reduced to what autonomy
/// gating needs. Derived from the effective `[exec]`
/// ([`AkhConfig::effective_exec`](crate::akh_config::AkhConfig::effective_exec)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecPlacement {
    /// Commands run in the akhomed process's own address space.
    InProcess,
    /// Commands run somewhere contained: a privilege-dropped local user, a
    /// remote host, or an akhsbxd/akhexecd sandbox.
    Contained,
    /// No executor tools are registered (`exec kind = "none"`).
    Disabled,
}

impl From<&ExecBackendConfig> for ExecPlacement {
    fn from(exec: &ExecBackendConfig) -> Self {
        match exec {
            ExecBackendConfig::InProcess => ExecPlacement::InProcess,
            ExecBackendConfig::LocalDrop { .. }
            | ExecBackendConfig::Remote { .. }
            | ExecBackendConfig::Execd { .. } => ExecPlacement::Contained,
            ExecBackendConfig::None => ExecPlacement::Disabled,
        }
    }
}

/// Which tools the idle OODA cycle may pick on its own (`[autonomy]
/// idle_cycle_tools`).
///
/// The idle cycle chooses tools with no human in the loop. By default it
/// may only choose tools whose manifest is [`DangerLevel::Safe`] and that
/// are not proxied from an MCP server, so an unattended cycle cannot spawn
/// agents, message peers, edit reactor schedules or call external MCP
/// tools. `all` lifts that only together with an explicit
/// `tasks.idle_cycle = true` (see [`IdleCycleTools::candidate_policy`]).
///
/// [`DangerLevel::Safe`]: crate::agent::tool_manifest::DangerLevel::Safe
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdleCycleTools {
    /// Safe, non-MCP tools only (the default).
    #[default]
    Safe,
    /// Every registered tool. Needs `tasks.idle_cycle = true` as well.
    All,
}

impl IdleCycleTools {
    /// The candidate policy the idle cycle runs with. `All` applies only
    /// when the operator also set `tasks.idle_cycle = true` explicitly;
    /// otherwise the cycle stays safe-only.
    pub const fn candidate_policy(
        self,
        idle_cycle_explicitly_enabled: bool,
    ) -> super::ooda::ToolCandidatePolicy {
        match (self, idle_cycle_explicitly_enabled) {
            (IdleCycleTools::All, true) => super::ooda::ToolCandidatePolicy::Unrestricted,
            _ => super::ooda::ToolCandidatePolicy::SafeNonMcp,
        }
    }
}

/// Why a task did not run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisabledReason {
    /// `[autonomy] enabled = false` and no per-task override.
    MasterSwitchOff,
    /// `[autonomy] tasks.<name> = false`.
    ExplicitlyDisabled,
    /// The idle OODA cycle defaults off while tools execute in-process
    /// (ADR 113 decision 4). Set `tasks.idle_cycle = true` to override.
    InProcessExecutor,
    /// An environment gate is not set, for example `AKH_LIBRARIAN=1`.
    EnvGate {
        /// The variable that must be set.
        var: &'static str,
    },
    /// A prerequisite is not configured for this akh.
    NotConfigured {
        /// What is missing.
        what: &'static str,
    },
    /// The task has no override of its own and follows another task's gate,
    /// which is off.
    FollowsTask {
        /// The task whose gate applies.
        task: DaemonTask,
        /// Why that task is off.
        reason: Box<DisabledReason>,
    },
}

impl fmt::Display for DisabledReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DisabledReason::MasterSwitchOff => f.write_str("[autonomy] enabled = false"),
            DisabledReason::ExplicitlyDisabled => f.write_str("disabled in [autonomy] tasks"),
            DisabledReason::InProcessExecutor => f.write_str(
                "executor tools run in-process; idle OODA tool use stays off until exec is contained \
                 (set [autonomy] tasks.idle_cycle = true to override)",
            ),
            DisabledReason::EnvGate { var } => write!(f, "environment gate {var} is not set"),
            DisabledReason::NotConfigured { what } => write!(f, "{what} is not configured"),
            DisabledReason::FollowsTask { task, reason } => write!(
                f,
                "follows `{task}`, which is off ({reason}); set [autonomy] tasks.<name> to override"
            ),
        }
    }
}

/// Whether a task may run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskGate {
    /// The task runs.
    Enabled,
    /// The task is skipped.
    Disabled(DisabledReason),
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// Per-akh daily limits for LLM-cost background work (`[autonomy.budget]`).
///
/// ```toml
/// [autonomy.budget]
/// max_tokens_per_day = 250000
/// max_turns_per_day = 40
/// ```
///
/// A limit of 0 turns every LLM-cost task off while leaving the free tasks
/// running.
///
/// Unknown keys are rejected when the TOML is loaded (a typo such as
/// `max_token_per_day` would otherwise silently leave the default in force).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AutonomyBudget {
    /// Prompt plus completion tokens per UTC day.
    pub max_tokens_per_day: u64,
    /// Turns per UTC day (see the module docs for what a turn is).
    pub max_turns_per_day: u32,
}

impl AutonomyBudget {
    /// Default token limit per UTC day.
    pub const DEFAULT_MAX_TOKENS_PER_DAY: u64 = 250_000;
    /// Default turn limit per UTC day.
    pub const DEFAULT_MAX_TURNS_PER_DAY: u32 = 40;
}

impl Default for AutonomyBudget {
    fn default() -> Self {
        Self {
            max_tokens_per_day: Self::DEFAULT_MAX_TOKENS_PER_DAY,
            max_turns_per_day: Self::DEFAULT_MAX_TURNS_PER_DAY,
        }
    }
}

/// A UTC calendar day, as whole days since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UtcDay(u64);

impl UtcDay {
    /// The day containing `unix_secs`.
    pub const fn from_unix_secs(unix_secs: u64) -> Self {
        Self(unix_secs / SECONDS_PER_DAY)
    }

    /// Today, by the system clock.
    pub fn today() -> Self {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self::from_unix_secs(secs)
    }

    /// Days since the Unix epoch.
    pub const fn days_since_epoch(self) -> u64 {
        self.0
    }

    /// The next day.
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    /// Unix seconds at 00:00:00 UTC of this day.
    pub const fn start_unix_secs(self) -> u64 {
        self.0.saturating_mul(SECONDS_PER_DAY)
    }
}

impl fmt::Display for UtcDay {
    /// ISO 8601 date (`2026-09-24`), via Howard Hinnant's civil-from-days.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let z = self.0 as i64 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + i64::from(month <= 2);
        write!(f, "{year:04}-{month:02}-{day:02}")
    }
}

/// Spend recorded for one UTC day.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetLedger {
    /// The day this spend belongs to.
    pub day: UtcDay,
    /// Tokens charged today.
    pub tokens_used: u64,
    /// Turns charged today.
    pub turns_used: u32,
}

impl BudgetLedger {
    /// An empty ledger for `day`.
    pub const fn empty(day: UtcDay) -> Self {
        Self {
            day,
            tokens_used: 0,
            turns_used: 0,
        }
    }

    /// This ledger as seen on `day`: unchanged on the same day, empty on a
    /// later one. An earlier `day` (clock stepped back) keeps the ledger, so
    /// a clock skew can never refund spend.
    pub fn rolled_to(self, day: UtcDay) -> Self {
        if day > self.day {
            Self::empty(day)
        } else {
            self
        }
    }

    /// The first limit this ledger has reached, if any.
    pub fn exhausted_dimension(
        &self,
        budget: &AutonomyBudget,
    ) -> Option<(BudgetDimension, u64, u64)> {
        if self.turns_used >= budget.max_turns_per_day {
            Some((
                BudgetDimension::Turns,
                u64::from(self.turns_used),
                u64::from(budget.max_turns_per_day),
            ))
        } else if self.tokens_used >= budget.max_tokens_per_day {
            Some((
                BudgetDimension::Tokens,
                self.tokens_used,
                budget.max_tokens_per_day,
            ))
        } else {
            None
        }
    }
}

/// Which daily limit was hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDimension {
    /// `max_tokens_per_day`.
    Tokens,
    /// `max_turns_per_day`.
    Turns,
}

impl BudgetDimension {
    /// Metric/log label.
    pub const fn as_str(self) -> &'static str {
        match self {
            BudgetDimension::Tokens => "tokens",
            BudgetDimension::Turns => "turns",
        }
    }
}

/// The daily budget refused an LLM-cost task.
#[derive(Debug, Clone, PartialEq, Eq, Error, Diagnostic)]
#[error(
    "autonomy budget exhausted for `{task}` on {day}: {used}/{limit} {} used",
    dimension.as_str()
)]
#[diagnostic(
    code(akh::autonomy::budget_exhausted),
    help(
        "LLM-cost background work resumes when the UTC day rolls over (unix {resets_at}). \
         To allow more, raise [autonomy.budget] max_turns_per_day / max_tokens_per_day in \
         the akh's TOML and restart its daemon. Free tasks keep running."
    )
)]
pub struct BudgetExhausted {
    /// The refused task.
    pub task: DaemonTask,
    /// The day whose budget is spent.
    pub day: UtcDay,
    /// Which limit was hit.
    pub dimension: BudgetDimension,
    /// Spend on that dimension.
    pub used: u64,
    /// The limit on that dimension.
    pub limit: u64,
    /// Unix seconds at which the budget resets (next UTC midnight).
    pub resets_at: u64,
}

impl BudgetExhausted {
    /// Seconds from `now` until the budget resets (at least 1).
    pub fn seconds_until_reset(&self, now_unix_secs: u64) -> u64 {
        self.resets_at.saturating_sub(now_unix_secs).max(1)
    }
}

/// Permission to run one LLM-cost unit of work. Hand it back to
/// [`AutonomyGovernor::settle`] with the metered usage.
#[derive(Debug)]
#[must_use = "an admitted ticket must be settled so its spend is charged"]
pub struct BudgetTicket {
    task: DaemonTask,
    day: UtcDay,
}

impl BudgetTicket {
    /// The admitted task.
    pub fn task(&self) -> DaemonTask {
        self.task
    }

    /// The day the ticket was admitted on.
    pub fn day(&self) -> UtcDay {
        self.day
    }
}

/// The outcome of [`AutonomyGovernor::admit`].
#[derive(Debug)]
#[must_use]
pub enum Admission {
    /// Run the work, then settle the ticket.
    Admitted(BudgetTicket),
    /// Do not run the work.
    Exhausted(BudgetExhausted),
}

/// Where the governor keeps its ledger.
enum LedgerStore {
    /// The engine's durable metadata tier.
    Durable(Arc<Engine>),
    /// Process memory only. The engine has no durable tier.
    Memory,
}

/// Enforces one akh's daily autonomy budget.
///
/// The daemon is its only writer. The ledger sits behind a mutex anyway, so
/// the governor can be shared read-only (status surfaces) without new
/// locking rules.
pub struct AutonomyGovernor {
    budget: AutonomyBudget,
    ledger: Mutex<BudgetLedger>,
    store: LedgerStore,
}

impl fmt::Debug for AutonomyGovernor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutonomyGovernor")
            .field("budget", &self.budget)
            .field("ledger", &self.ledger())
            .field("durable", &matches!(self.store, LedgerStore::Durable(_)))
            .finish()
    }
}

impl AutonomyGovernor {
    /// A governor whose ledger lives only in memory.
    pub fn in_memory(budget: AutonomyBudget) -> Self {
        Self {
            budget,
            ledger: Mutex::new(BudgetLedger::empty(UtcDay::today())),
            store: LedgerStore::Memory,
        }
    }

    /// Open the governor for `engine`, loading the persisted ledger when the
    /// engine has a durable tier. Falls back to memory without one.
    ///
    /// Callers choose how to handle a bad ledger: propagate it, or use
    /// [`AutonomyGovernor::open_or_reset`] to start today's ledger empty.
    pub fn open(engine: &Arc<Engine>, budget: AutonomyBudget) -> AutonomyResult<Self> {
        if engine.store().durable.is_none() {
            return Ok(Self::in_memory(budget));
        }
        let ledger = load_ledger(engine)?.unwrap_or_else(|| BudgetLedger::empty(UtcDay::today()));
        Ok(Self {
            budget,
            ledger: Mutex::new(ledger),
            store: LedgerStore::Durable(Arc::clone(engine)),
        })
    }

    /// Like [`AutonomyGovernor::open`], but an unreadable or corrupt ledger
    /// is logged and replaced by an empty one for today. That grants at most
    /// one extra day of budget; failing closed would stall every LLM-cost
    /// task until an operator repaired the store.
    pub fn open_or_reset(engine: &Arc<Engine>, budget: AutonomyBudget) -> Self {
        match Self::open(engine, budget) {
            Ok(governor) => governor,
            Err(error) => {
                tracing::warn!(
                    target: "akh::autonomy",
                    error = %error,
                    "autonomy budget ledger unreadable; starting today's ledger empty"
                );
                Self {
                    budget,
                    ledger: Mutex::new(BudgetLedger::empty(UtcDay::today())),
                    store: if engine.store().durable.is_some() {
                        LedgerStore::Durable(Arc::clone(engine))
                    } else {
                        LedgerStore::Memory
                    },
                }
            }
        }
    }

    /// The configured limits.
    pub fn budget(&self) -> AutonomyBudget {
        self.budget
    }

    /// Replace the limits. Spend already recorded is kept.
    pub fn set_budget(&mut self, budget: AutonomyBudget) {
        self.budget = budget;
    }

    /// The current ledger (not rolled over).
    pub fn ledger(&self) -> BudgetLedger {
        *self.ledger.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the ledger survives a restart.
    pub fn is_durable(&self) -> bool {
        matches!(self.store, LedgerStore::Durable(_))
    }

    /// Ask to run one LLM-cost unit of `task` today.
    pub fn admit(&self, task: DaemonTask) -> Admission {
        self.admit_on(task, UtcDay::today())
    }

    /// [`AutonomyGovernor::admit`] on an explicit day (deterministic tests,
    /// replay).
    pub fn admit_on(&self, task: DaemonTask, day: UtcDay) -> Admission {
        let ledger = self.ledger().rolled_to(day);
        match ledger.exhausted_dimension(&self.budget) {
            Some((dimension, used, limit)) => Admission::Exhausted(BudgetExhausted {
                task,
                day: ledger.day,
                dimension,
                used,
                limit,
                resets_at: ledger.day.next().start_unix_secs(),
            }),
            None => Admission::Admitted(BudgetTicket {
                task,
                day: ledger.day,
            }),
        }
    }

    /// Whether an admitted per-run task may make another model call, given
    /// the `in_flight` usage it has metered so far but not yet settled.
    ///
    /// Only the token limit is checked: the run as a whole is one turn and
    /// was admitted on the turn limit already. A long run (the librarian
    /// calls the model once per source) calls this before each call and
    /// stops cleanly on `Err`, so one run cannot exceed the day's token
    /// budget by more than one call.
    pub fn check_in_flight(
        &self,
        task: DaemonTask,
        in_flight: LlmUsage,
    ) -> Result<(), BudgetExhausted> {
        self.check_in_flight_on(task, in_flight, UtcDay::today())
    }

    /// [`AutonomyGovernor::check_in_flight`] on an explicit day.
    pub fn check_in_flight_on(
        &self,
        task: DaemonTask,
        in_flight: LlmUsage,
        day: UtcDay,
    ) -> Result<(), BudgetExhausted> {
        let ledger = self.ledger().rolled_to(day);
        let used = ledger.tokens_used.saturating_add(in_flight.total_tokens());
        if used >= self.budget.max_tokens_per_day {
            Err(BudgetExhausted {
                task,
                day: ledger.day,
                dimension: BudgetDimension::Tokens,
                used,
                limit: self.budget.max_tokens_per_day,
                resets_at: ledger.day.next().start_unix_secs(),
            })
        } else {
            Ok(())
        }
    }

    /// Charge the metered `usage` of an admitted ticket and return the
    /// updated ledger. A ticket whose work never reached a model is free.
    ///
    /// The in-memory ledger is updated before the write, so a failed write
    /// still enforces the budget until the daemon restarts.
    pub fn settle(&self, ticket: BudgetTicket, usage: LlmUsage) -> AutonomyResult<BudgetLedger> {
        self.settle_on(ticket, usage, UtcDay::today())
    }

    /// [`AutonomyGovernor::settle`] on an explicit day. Spend lands on
    /// `day` (the day the tokens were actually spent), even if the ticket
    /// was admitted the day before.
    pub fn settle_on(
        &self,
        ticket: BudgetTicket,
        usage: LlmUsage,
        day: UtcDay,
    ) -> AutonomyResult<BudgetLedger> {
        let BudgetTicket { .. } = ticket;
        let updated = {
            let mut guard = self.ledger.lock().unwrap_or_else(|e| e.into_inner());
            let before = *guard;
            let mut ledger = before.rolled_to(day);
            if usage.reached_model() {
                ledger.turns_used = ledger.turns_used.saturating_add(1);
                ledger.tokens_used = ledger.tokens_used.saturating_add(usage.total_tokens());
            }
            *guard = ledger;
            (ledger != before).then_some(ledger)
        };
        match updated {
            Some(ledger) => {
                self.persist(&ledger)?;
                Ok(ledger)
            }
            None => Ok(self.ledger()),
        }
    }

    fn persist(&self, ledger: &BudgetLedger) -> AutonomyResult<()> {
        let LedgerStore::Durable(engine) = &self.store else {
            return Ok(());
        };
        let bytes =
            serde_json::to_vec(ledger).map_err(|source| AutonomyError::LedgerEncode { source })?;
        engine
            .store()
            .put_meta(BUDGET_LEDGER_KEY, &bytes)
            .map_err(|source| AutonomyError::LedgerWrite {
                key: ledger_key_display(),
                source,
            })
    }
}

fn ledger_key_display() -> String {
    String::from_utf8_lossy(BUDGET_LEDGER_KEY).into_owned()
}

fn load_ledger(engine: &Engine) -> AutonomyResult<Option<BudgetLedger>> {
    let bytes = engine
        .store()
        .get_meta(BUDGET_LEDGER_KEY)
        .map_err(|source| AutonomyError::LedgerRead {
            key: ledger_key_display(),
            source,
        })?;
    bytes
        .map(|bytes| {
            serde_json::from_slice(&bytes).map_err(|source| AutonomyError::CorruptLedger {
                key: ledger_key_display(),
                source,
            })
        })
        .transpose()
}

// ---------------------------------------------------------------------------
// Task reports
// ---------------------------------------------------------------------------

/// Fact changes observed while a task ran (see
/// [`FactCounters`](crate::mutation::FactCounters) for the attribution
/// caveat).
pub type FactDelta = crate::mutation::FactTotals;

/// What a task body reports back to the daemon's task wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum TaskRun {
    /// The task ran to completion. It may have found nothing to do.
    Ran,
    /// A live turn held the agent lock, so the task yielded.
    Contended,
    /// The task failed. The string is the rendered error.
    Failed(String),
}

impl TaskRun {
    /// `Ran` on `Ok`, `Failed` with the rendered error on `Err`.
    pub fn from_result<T, E: fmt::Display>(result: Result<T, E>) -> Self {
        match result {
            Ok(_) => TaskRun::Ran,
            Err(error) => TaskRun::Failed(error.to_string()),
        }
    }
}

/// The outcome of one scheduled task tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskOutcome {
    /// The task ran.
    Ran,
    /// Gating kept the task from running.
    SkippedDisabled(DisabledReason),
    /// The daily budget kept the task from running.
    SkippedBudget(BudgetExhausted),
    /// A live turn held the agent lock.
    SkippedContended,
    /// The task failed.
    Failed(String),
}

impl TaskOutcome {
    /// The bounded `outcome` metric label.
    pub const fn label(&self) -> &'static str {
        match self {
            TaskOutcome::Ran => "ran",
            TaskOutcome::SkippedDisabled(_) => "skipped_disabled",
            TaskOutcome::SkippedBudget(_) => "skipped_budget",
            TaskOutcome::SkippedContended => "skipped_contended",
            TaskOutcome::Failed(_) => "failed",
        }
    }
}

impl From<TaskRun> for TaskOutcome {
    fn from(run: TaskRun) -> Self {
        match run {
            TaskRun::Ran => TaskOutcome::Ran,
            TaskRun::Contended => TaskOutcome::SkippedContended,
            TaskRun::Failed(error) => TaskOutcome::Failed(error),
        }
    }
}

/// Telemetry for one scheduled task tick.
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonTaskReport {
    /// The task.
    pub task: DaemonTask,
    /// What happened.
    pub outcome: TaskOutcome,
    /// Wall time spent in the task body (zero when skipped).
    pub duration: Duration,
    /// LLM usage metered on the daemon thread while the body ran.
    pub usage: LlmUsage,
    /// Engine-wide fact changes while the body ran.
    pub facts: FactDelta,
    /// The budget ledger after settling, for budget-metered runs.
    pub ledger: Option<BudgetLedger>,
}

impl DaemonTaskReport {
    /// A report for a task that did not run.
    pub fn skipped(task: DaemonTask, outcome: TaskOutcome) -> Self {
        Self {
            task,
            outcome,
            duration: Duration::ZERO,
            usage: LlmUsage::default(),
            facts: FactDelta::default(),
            ledger: None,
        }
    }

    /// Whether the run changed anything worth an `info` line: facts, model
    /// usage, a failure or a budget refusal.
    pub fn is_noteworthy(&self) -> bool {
        match &self.outcome {
            TaskOutcome::Ran => !self.facts.is_empty() || self.usage.reached_model(),
            TaskOutcome::SkippedBudget(_) | TaskOutcome::Failed(_) => true,
            TaskOutcome::SkippedDisabled(_) | TaskOutcome::SkippedContended => false,
        }
    }

    /// Log the report under target `akh::autonomy` and record the
    /// `akh_daemon_task_*` metrics (feature `tracing-otel`).
    ///
    /// Failures log at `warn`; budget refusals and runs that changed facts or
    /// spent tokens log at `info`; quiet runs and gated skips log at `debug`,
    /// because the reactor and trigger ticks fire every few seconds.
    pub fn emit(&self, workspace: &str) {
        let task = self.task.as_str();
        let outcome = self.outcome.label();
        let duration_ms = self.duration.as_millis() as u64;
        let (turns_used, tokens_used) = self
            .ledger
            .map(|l| (u64::from(l.turns_used), l.tokens_used))
            .unwrap_or_default();
        match &self.outcome {
            TaskOutcome::Failed(error) => tracing::warn!(
                target: "akh::autonomy",
                workspace,
                task,
                outcome,
                duration_ms,
                error = %error,
                llm_requests = self.usage.requests,
                prompt_tokens = self.usage.prompt_tokens,
                completion_tokens = self.usage.completion_tokens,
                facts_asserted = self.facts.asserted,
                facts_retracted = self.facts.retracted,
                "daemon task failed"
            ),
            TaskOutcome::SkippedBudget(exhausted) => tracing::info!(
                target: "akh::autonomy",
                workspace,
                task,
                outcome,
                dimension = exhausted.dimension.as_str(),
                used = exhausted.used,
                limit = exhausted.limit,
                day = %exhausted.day,
                resets_at = exhausted.resets_at,
                "daemon task skipped: autonomy budget exhausted"
            ),
            _ if self.is_noteworthy() => tracing::info!(
                target: "akh::autonomy",
                workspace,
                task,
                outcome,
                duration_ms,
                llm_requests = self.usage.requests,
                prompt_tokens = self.usage.prompt_tokens,
                completion_tokens = self.usage.completion_tokens,
                facts_asserted = self.facts.asserted,
                facts_retracted = self.facts.retracted,
                budget_turns_used = turns_used,
                budget_tokens_used = tokens_used,
                "daemon task ran"
            ),
            TaskOutcome::SkippedDisabled(reason) => tracing::debug!(
                target: "akh::autonomy",
                workspace,
                task,
                outcome,
                reason = %reason,
                "daemon task skipped"
            ),
            _ => tracing::debug!(
                target: "akh::autonomy",
                workspace,
                task,
                outcome,
                duration_ms,
                "daemon task tick"
            ),
        }
        #[cfg(feature = "tracing-otel")]
        self.record_metrics(workspace);
    }

    #[cfg(feature = "tracing-otel")]
    fn record_metrics(&self, workspace: &str) {
        use opentelemetry::KeyValue;

        let Some(metrics) = crate::metrics::global() else {
            return;
        };
        let task = self.task.as_str();
        let base = [
            KeyValue::new("workspace", workspace.to_owned()),
            KeyValue::new("task", task),
        ];
        let with = |key: &'static str, value: &'static str| {
            let mut labels = base.to_vec();
            labels.push(KeyValue::new(key, value));
            labels
        };
        let outcome_labels = with("outcome", self.outcome.label());
        metrics.daemon_task_runs_total.add(1, &outcome_labels);
        if matches!(self.outcome, TaskOutcome::Ran | TaskOutcome::Failed(_)) {
            metrics
                .daemon_task_duration_seconds
                .record(self.duration.as_secs_f64(), &outcome_labels);
        }
        if self.usage.prompt_tokens > 0 {
            metrics
                .daemon_task_llm_tokens_total
                .add(self.usage.prompt_tokens, &with("direction", "prompt"));
        }
        if self.usage.completion_tokens > 0 {
            metrics.daemon_task_llm_tokens_total.add(
                self.usage.completion_tokens,
                &with("direction", "completion"),
            );
        }
        if self.facts.asserted > 0 {
            metrics
                .daemon_task_facts_total
                .add(self.facts.asserted, &with("change", "asserted"));
        }
        if self.facts.retracted > 0 {
            metrics
                .daemon_task_facts_total
                .add(self.facts.retracted, &with("change", "retracted"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn usage(tokens: u64) -> LlmUsage {
        LlmUsage {
            requests: 1,
            prompt_tokens: tokens,
            completion_tokens: 0,
        }
    }

    fn durable_engine(dir: &std::path::Path) -> Arc<Engine> {
        Arc::new(
            Engine::new(EngineConfig {
                dimension: Dimension::TEST,
                data_dir: Some(dir.to_path_buf()),
                ..Default::default()
            })
            .expect("engine"),
        )
    }

    fn admitted(admission: Admission) -> BudgetTicket {
        match admission {
            Admission::Admitted(ticket) => ticket,
            Admission::Exhausted(e) => panic!("expected admission, got {e}"),
        }
    }

    fn exhausted(admission: Admission) -> BudgetExhausted {
        match admission {
            Admission::Admitted(ticket) => panic!("expected exhaustion, got {ticket:?}"),
            Admission::Exhausted(e) => e,
        }
    }

    #[test]
    fn task_names_round_trip_and_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for task in DaemonTask::ALL {
            assert!(seen.insert(task.as_str()), "duplicate name {task}");
            assert_eq!(task.as_str().parse::<DaemonTask>().unwrap(), task);
        }
        assert!("seshat_curaton".parse::<DaemonTask>().is_err());
        // The pre-53b config keys stay valid.
        for legacy in [
            "equivalence",
            "reflection",
            "consolidation",
            "schema_discovery",
            "rule_inference",
            "gap_analysis",
            "idle_cycle",
            "continuous_learning",
            "goal_generation",
            "sleep_cycle",
            "knowledge_extraction",
            "bridge_training",
            "openclaw_sync",
            "knowledge_elicitation",
            "kg_compaction",
            "delegation",
            "relation_enrichment",
            "seshat_curation",
            "heka_curation",
        ] {
            assert!(legacy.parse::<DaemonTask>().is_ok(), "{legacy} must parse");
        }
    }

    #[test]
    fn llm_cost_tasks_are_the_decision_4_set() {
        let llm: Vec<_> = DaemonTask::ALL
            .into_iter()
            .filter(|t| t.cost() == TaskCost::Llm)
            .collect();
        assert_eq!(
            llm,
            vec![
                DaemonTask::Reactor,
                DaemonTask::KnowledgeExtraction,
                DaemonTask::KnowledgeElicitation,
                DaemonTask::Librarian,
            ]
        );
        // Reflection and goal generation make no LLM calls (decision 4).
        assert_eq!(DaemonTask::Reflection.cost(), TaskCost::Free);
        assert_eq!(DaemonTask::GoalGeneration.cost(), TaskCost::Free);
        assert_eq!(DaemonTask::Harvest.cost(), TaskCost::Free);
        assert_eq!(
            DaemonTask::Reactor.metering(),
            BudgetMetering::PerOccurrence
        );
    }

    #[test]
    fn utc_day_formats_as_iso_date() {
        assert_eq!(UtcDay::from_unix_secs(0).to_string(), "1970-01-01");
        // 2026-09-24T12:00:00Z
        assert_eq!(
            UtcDay::from_unix_secs(1_790_251_200).to_string(),
            "2026-09-24"
        );
        assert_eq!(
            UtcDay::from_unix_secs(951_782_400).to_string(),
            "2000-02-29"
        );
    }

    #[test]
    fn governor_exhausts_on_turns_then_rolls_over() {
        let budget = AutonomyBudget {
            max_tokens_per_day: 1_000_000,
            max_turns_per_day: 2,
        };
        let governor = AutonomyGovernor::in_memory(budget);
        let day = UtcDay::from_unix_secs(1_790_251_200);

        for _ in 0..2 {
            let ticket = admitted(governor.admit_on(DaemonTask::Librarian, day));
            governor.settle_on(ticket, usage(10), day).unwrap();
        }
        let refused = exhausted(governor.admit_on(DaemonTask::Librarian, day));
        assert_eq!(refused.dimension, BudgetDimension::Turns);
        assert_eq!((refused.used, refused.limit), (2, 2));
        assert_eq!(refused.resets_at, day.next().start_unix_secs());
        assert_eq!(refused.seconds_until_reset(1_790_251_200), 43_200);

        // The next UTC day starts with an empty ledger.
        let tomorrow = day.next();
        let ticket = admitted(governor.admit_on(DaemonTask::Librarian, tomorrow));
        let ledger = governor.settle_on(ticket, usage(5), tomorrow).unwrap();
        assert_eq!(
            ledger,
            BudgetLedger {
                day: tomorrow,
                tokens_used: 5,
                turns_used: 1
            }
        );
        // A clock stepped back never refunds spend.
        assert_eq!(governor.ledger().rolled_to(day), governor.ledger());
    }

    #[test]
    fn governor_exhausts_on_tokens_after_an_overshooting_run() {
        let budget = AutonomyBudget {
            max_tokens_per_day: 100,
            max_turns_per_day: 40,
        };
        let governor = AutonomyGovernor::in_memory(budget);
        let day = UtcDay::from_unix_secs(0);
        let ticket = admitted(governor.admit_on(DaemonTask::KnowledgeElicitation, day));
        // One run may overshoot; admission is checked before the work.
        governor.settle_on(ticket, usage(250), day).unwrap();
        let refused = exhausted(governor.admit_on(DaemonTask::KnowledgeElicitation, day));
        assert_eq!(refused.dimension, BudgetDimension::Tokens);
        assert_eq!((refused.used, refused.limit), (250, 100));
    }

    #[test]
    fn work_that_never_reached_a_model_is_free() {
        let governor = AutonomyGovernor::in_memory(AutonomyBudget {
            max_tokens_per_day: 10,
            max_turns_per_day: 1,
        });
        let day = UtcDay::from_unix_secs(0);
        for _ in 0..5 {
            let ticket = admitted(governor.admit_on(DaemonTask::KnowledgeElicitation, day));
            governor
                .settle_on(ticket, LlmUsage::default(), day)
                .unwrap();
        }
        assert_eq!(governor.ledger().turns_used, 0);
    }

    #[test]
    fn zero_turn_budget_refuses_every_llm_task() {
        let governor = AutonomyGovernor::in_memory(AutonomyBudget {
            max_tokens_per_day: 1_000,
            max_turns_per_day: 0,
        });
        let refused = exhausted(governor.admit(DaemonTask::Librarian));
        assert_eq!(refused.dimension, BudgetDimension::Turns);
    }

    #[test]
    fn ledger_persists_across_engine_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let day = UtcDay::today();
        let budget = AutonomyBudget {
            max_tokens_per_day: 1_000,
            max_turns_per_day: 3,
        };
        {
            let engine = durable_engine(dir.path());
            let governor = AutonomyGovernor::open(&engine, budget).unwrap();
            assert!(governor.is_durable());
            for tokens in [100, 200] {
                let ticket = admitted(governor.admit_on(DaemonTask::Librarian, day));
                governor.settle_on(ticket, usage(tokens), day).unwrap();
            }
            engine.persist().unwrap();
        }
        let engine = durable_engine(dir.path());
        let governor = AutonomyGovernor::open(&engine, budget).unwrap();
        assert_eq!(
            governor.ledger(),
            BudgetLedger {
                day,
                tokens_used: 300,
                turns_used: 2
            }
        );
        let ticket = admitted(governor.admit_on(DaemonTask::Librarian, day));
        governor.settle_on(ticket, usage(1), day).unwrap();
        exhausted(governor.admit_on(DaemonTask::Librarian, day));
    }

    #[test]
    fn corrupt_ledger_is_a_diagnostic_and_open_or_reset_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let engine = durable_engine(dir.path());
        engine
            .store()
            .put_meta(BUDGET_LEDGER_KEY, b"not json")
            .unwrap();
        let error = AutonomyGovernor::open(&engine, AutonomyBudget::default()).unwrap_err();
        assert!(matches!(error, AutonomyError::CorruptLedger { .. }));
        let governor = AutonomyGovernor::open_or_reset(&engine, AutonomyBudget::default());
        assert!(governor.is_durable());
        assert_eq!(governor.ledger().turns_used, 0);
    }

    #[test]
    fn in_memory_engine_falls_back_to_memory_ledger() {
        let engine = Arc::new(
            Engine::new(EngineConfig {
                dimension: Dimension::TEST,
                ..Default::default()
            })
            .unwrap(),
        );
        let governor = AutonomyGovernor::open(&engine, AutonomyBudget::default()).unwrap();
        assert!(!governor.is_durable());
        let ticket = admitted(governor.admit(DaemonTask::Librarian));
        let ledger = governor.settle(ticket, usage(7)).unwrap();
        assert_eq!(ledger.tokens_used, 7);
    }

    #[test]
    fn exec_placement_follows_effective_exec() {
        assert_eq!(
            ExecPlacement::from(&ExecBackendConfig::InProcess),
            ExecPlacement::InProcess
        );
        assert_eq!(
            ExecPlacement::from(&ExecBackendConfig::None),
            ExecPlacement::Disabled
        );
        assert_eq!(
            ExecPlacement::from(&ExecBackendConfig::LocalDrop {
                user: "akh".into(),
                workdir: None,
                shell: None,
                path: None,
            }),
            ExecPlacement::Contained
        );
    }

    #[test]
    fn report_noteworthiness_and_labels() {
        let quiet = DaemonTaskReport::skipped(DaemonTask::Reactor, TaskOutcome::Ran);
        assert!(!quiet.is_noteworthy());
        let mut busy = quiet.clone();
        busy.facts.asserted = 3;
        assert!(busy.is_noteworthy());
        let failed =
            DaemonTaskReport::skipped(DaemonTask::Harvest, TaskOutcome::Failed("x".into()));
        assert!(failed.is_noteworthy());
        assert_eq!(failed.outcome.label(), "failed");
        assert_eq!(
            TaskOutcome::from(TaskRun::Contended).label(),
            "skipped_contended"
        );
    }
}
