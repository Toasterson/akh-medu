//! Background daemon for autonomous learning.
//!
//! `AgentDaemon` runs a tokio event loop that periodically fires background
//! tasks: equivalence learning, reflection, consolidation, schema discovery,
//! rule inference, gap analysis, goal generation, sleep cycles, and idle OODA
//! cycles. The agent itself stays synchronous — tokio only drives scheduling
//! and signal handling.
//!
//! Both the standalone `akh-medu daemon` and the `akhomed` server reuse this
//! same `AgentDaemon`, so every background task added here automatically works
//! in both deployment modes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use super::Agent;
use super::error::AgentResult;
use crate::akh_config::AutonomyConfig;
use crate::client::DaemonStatus;
use crate::config::UpstreamEndpointConfig;
use crate::message::AkhMessage;

/// Current unix timestamp in seconds.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Assert a "singleton" fact `(subject, predicate, value)` into the KG,
/// retracting any previous value for the same `(subject, predicate)` first
/// so re-assertion **updates** the fact rather than accumulating a trail of
/// stale ones.
///
/// [`crate::graph::index::KnowledgeGraph::insert_triple`] only dedupes when
/// the *entire* triple (subject, predicate, *and* object) is identical — an
/// object literal that changes tick to tick (a chunk count, a timestamp)
/// gets a fresh object symbol every time, so without the explicit retract
/// here a curation pass would leave behind
/// `(seshat-archive, has-pending-embeddings, "3")`,
/// `(seshat-archive, has-pending-embeddings, "5")`, ... forever.
///
/// Best-effort: engine errors are logged at WARN and swallowed — this is
/// background bookkeeping, not worth failing an otherwise-successful
/// curation pass over.
fn assert_singleton_fact(engine: &crate::engine::Engine, subject: &str, predicate: &str, value: &str) {
    let s = match engine.resolve_or_create_entity(subject) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(error = %e, subject, "daemon: failed to resolve curation subject symbol");
            return;
        }
    };
    let p = match engine.resolve_or_create_relation(predicate) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(error = %e, predicate, "daemon: failed to resolve curation predicate symbol");
            return;
        }
    };
    let o = match engine.resolve_or_create_entity(value) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!(error = %e, value, "daemon: failed to resolve curation value symbol");
            return;
        }
    };

    // Retract any prior value(s) for this (subject, predicate) that differ
    // from the new one — supersede rather than accumulate. A no-op when the
    // value hasn't changed since the last pass (the matching triple is left
    // alone, and `add_triple` below dedupes it).
    for existing in engine.triples_from(s) {
        if existing.predicate == p
            && existing.object != o
            && let Err(e) = engine.remove_triple(s, p, existing.object)
        {
            tracing::warn!(error = %e, subject, predicate, "daemon: failed to retract stale curation fact");
        }
    }

    if let Err(e) = engine.add_triple(&crate::graph::Triple::new(s, p, o)) {
        tracing::warn!(error = %e, subject, predicate, value, "daemon: failed to assert curation fact");
    }
}

/// Every daemon task name consulted via [`AgentDaemon::task_enabled`].
///
/// Task names are free-form strings — [`AutonomyConfig::task_enabled`]
/// silently falls back to the akh's master `[autonomy] enabled` switch on an
/// unrecognized one rather than erroring, so a typo in a new
/// `tokio::select!` arm (`self.task_enabled("seshat_curaton")`, say) would
/// otherwise fail quietly. **Extend this list whenever a new arm in
/// [`AgentDaemon::run`] gates itself with `task_enabled(..)`.** Purely
/// documentation/test fixture — has no runtime effect on gating.
///
/// Does *not* include `"librarian"`: that arm is gated by the
/// `AKH_LIBRARIAN` env var instead, so it never consults `[autonomy]`.
// Only consumed by `tests::known_task_names_includes_curation_tasks` today —
// a documentation/test fixture, not runtime-referenced, hence `dead_code`
// is expected outside `#[cfg(test)]` builds.
#[allow(dead_code)]
const KNOWN_TASK_NAMES: &[&str] = &[
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
];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Intervals for each daemon background task.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Equivalence learning interval (default: 5 min).
    pub equivalence_interval: Duration,
    /// Reflection interval (default: 3 min).
    pub reflection_interval: Duration,
    /// Consolidation check interval (default: 1 min).
    pub consolidation_interval: Duration,
    /// Schema discovery interval (default: 30 min).
    pub schema_discovery_interval: Duration,
    /// Rule inference interval (default: 10 min).
    pub rule_inference_interval: Duration,
    /// Gap analysis interval (default: 15 min).
    pub gap_analysis_interval: Duration,
    /// Session persistence interval (default: 60s).
    pub persist_interval: Duration,
    /// Idle OODA cycle interval (default: 30s).
    pub idle_cycle_interval: Duration,
    /// Continuous learning interval (default: 2 hours).
    pub continuous_learning_interval: Duration,
    /// Goal generation interval (default: 5 min).
    ///
    /// Runs independently of the OODA cycle so that drives can propose goals
    /// even when no active goals exist — breaking the autonomy deadlock.
    pub goal_generation_interval: Duration,
    /// Sleep/consolidation cycle interval (default: 60 min).
    ///
    /// Runs KG reorganization (duplicate merge, orphan prune, dream
    /// connections) as a standalone timer so consolidation happens even when
    /// the OODA cycle is idle.
    pub sleep_cycle_interval: Duration,
    /// Trigger evaluation interval (default: 15s).
    ///
    /// Evaluates user-defined triggers against the current agent state and
    /// fires any whose conditions are met.
    pub trigger_evaluation_interval: Duration,
    /// Knowledge extraction interval (default: 45 min).
    ///
    /// Runs LLM-backed triple extraction on recently-fetched content that
    /// hasn't been processed yet. Uses the three-tier fallback: local LLM,
    /// external API, regex.
    pub knowledge_extraction_interval: Duration,
    /// Neural bridge training interval (default: 2 hours).
    ///
    /// Runs bridge encoder alignment training during idle periods.
    /// Only starts after sufficient training data has accumulated.
    pub bridge_training_interval: Duration,
    /// OpenClaw journal sync interval (default: 24 hours).
    ///
    /// Fetches new daily memory journals from OpenClaw and persists them
    /// as episodic memories. Only runs when `OPENCLAW_API_KEY` is set.
    pub openclaw_sync_interval: Duration,
    /// LLM knowledge elicitation interval (default: 30 min).
    ///
    /// Phase 32: proactively generates structured triples about domains with
    /// detected knowledge gaps using the T5 Knowledge model (Phase 35).
    /// Distinct from `knowledge_extraction_interval` which extracts triples
    /// from already-fetched text.  Falls back gracefully when the model file
    /// is absent.
    pub knowledge_elicitation_interval: Duration,
    /// KG compaction interval (default: 30 min).
    ///
    /// Removes episode/agent scaffolding triples that accumulate over time and
    /// degrade KG query quality. Runs the [`compaction::run_kg_compaction`]
    /// heuristic on a periodic timer so the graph stays clean without manual
    /// intervention.
    pub kg_compaction_interval: Duration,
    /// Delegation result processing interval (default: 10s).
    ///
    /// Drains the pending task queue, dispatches new delegations, processes
    /// completed results, asserts result triples into the KG, and stores
    /// delegation provenance.
    pub delegation_interval: Duration,
    /// Relation-enrichment interval (default: 60 min).
    ///
    /// Runs the [`relation_enrichment::run_relation_enrichment`] pass which
    /// injects missing predicate symbols and links orphaned entities to drive
    /// the entity:relation ratio toward the healthy 2–3:1 target.
    pub relation_enrichment_interval: Duration,
    /// Daily-Librarian pass interval (default: 24 h).
    ///
    /// ADR 081 slice 4 (`agent::librarian`): LLM extraction of typed triples
    /// from captured transcript turns against the in-force schema. Distinct
    /// from `knowledge_elicitation_interval` (Phase 32, which *generates*
    /// speculative concept knowledge) — this pass *extracts* grounded facts
    /// from the akh's own captured material. The ticker fires on this
    /// interval regardless, but the task itself is off unless
    /// `AKH_LIBRARIAN=1` is set (real per-tick LLM cost, unlike the
    /// always-on harvest phase's pure KG bookkeeping) and a Postgres store is
    /// configured (turn material lives only in the WAL).
    pub librarian_interval: Duration,
    /// Seshat's Archive curation-pass interval (default: 6 h).
    ///
    /// Surveys the configured `[upstreams.seshat]` endpoint's corpus status,
    /// forces an embedding pass when chunks are pending, and asserts the
    /// resulting counts into the KG. Gated by `task_enabled("seshat_curation")`
    /// (P5 autonomy), unlike `librarian_interval` which is env-gated.
    pub seshat_curation_interval: Duration,
    /// Heka long-term-memory store curation-pass interval (default: 24 h).
    ///
    /// v1 is surveillance-only: reports unembedded-chunk and untagged-drawer
    /// counts from the configured `[heka]` Postgres database into the KG.
    /// Real curation (backfill embeddings, tag drawers, prune) needs the
    /// Phase 42 service layer, which doesn't exist yet. No-op without the
    /// `postgres` feature or a `[heka]` section.
    pub heka_curation_interval: Duration,
    /// Maximum OODA cycles (0 = unlimited).
    pub max_cycles: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            equivalence_interval: Duration::from_secs(300),
            reflection_interval: Duration::from_secs(180),
            consolidation_interval: Duration::from_secs(60),
            schema_discovery_interval: Duration::from_secs(1800),
            rule_inference_interval: Duration::from_secs(600),
            gap_analysis_interval: Duration::from_secs(900),
            persist_interval: Duration::from_secs(60),
            idle_cycle_interval: Duration::from_secs(30),
            continuous_learning_interval: Duration::from_secs(7200),
            goal_generation_interval: Duration::from_secs(300),
            sleep_cycle_interval: Duration::from_secs(3600),
            trigger_evaluation_interval: Duration::from_secs(15),
            knowledge_extraction_interval: Duration::from_secs(2700), // 45 min
            bridge_training_interval: Duration::from_secs(7200), // 2 hours
            openclaw_sync_interval: Duration::from_secs(86400), // 24 hours
            knowledge_elicitation_interval: Duration::from_secs(1800), // 30 min
            kg_compaction_interval: Duration::from_secs(1800), // 30 min
            delegation_interval: Duration::from_secs(10),
            relation_enrichment_interval: Duration::from_secs(3600), // 60 min
            librarian_interval: Duration::from_secs(86400), // 24 hours
            seshat_curation_interval: Duration::from_secs(21600), // 6 hours
            heka_curation_interval: Duration::from_secs(86400), // 24 hours
            max_cycles: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------------

/// Long-running daemon that schedules background learning tasks.
///
/// The agent is shared via `Arc<RwLock<Agent>>` so that HTTP handlers,
/// WebSocket sessions, and the daemon all operate on the same agent
/// instance. Each daemon method locks briefly, does synchronous work,
/// and releases — near-zero contention since ticks are 15s–2h apart.
pub struct AgentDaemon {
    agent: Arc<RwLock<Agent>>,
    config: DaemonConfig,
    total_cycles: usize,
    /// When true, cycle/learning methods skip their work without acquiring
    /// the agent lock — allowing external writers to acquire it uncontended.
    paused: Arc<AtomicBool>,
    /// Triple-count watermark for equivalence learning. The task only runs
    /// when the KG triple count has advanced since the last pass, preventing
    /// the "learn-on-new-triples" trigger from firing uselessly on an idle
    /// akh and drowning the logs in zero-result INFO lines.
    last_equivalence_triple_count: std::sync::atomic::AtomicUsize,
    /// Notified after each cycle completes. The suspend handler waits on
    /// this to ensure the daemon has released the agent lock.
    cycle_done: Arc<tokio::sync::Notify>,
    /// Optional external shutdown signal (replaces Ctrl+C when set).
    ///
    /// Used by `akhomed` to integrate daemon shutdown with the HTTP server's
    /// watch channel instead of relying on process-level Ctrl+C.
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
    /// Optional shared status for real-time monitoring from HTTP handlers.
    status: Option<Arc<tokio::sync::Mutex<DaemonStatus>>>,
    /// Per-akh automation gating (P5). Defaults to master-off, so a daemon
    /// constructed without explicit autonomy runs only always-on infra tasks
    /// (session persistence + user-trigger evaluation).
    autonomy: AutonomyConfig,
    /// Postgres backing store + this akh's stable id, for the ADR 091
    /// sleep-cycle transcript-harvest phase. `None` in local-only mode (no
    /// `[postgres]` section configured) — harvest is then a no-op, matching
    /// every other store-backed background task's local-only degradation.
    #[cfg(feature = "postgres")]
    db: Option<(akhomed_store::DatabaseConnection, akhomed_store::uuid::Uuid)>,
    /// Seshat's Archive upstream endpoint, for `run_seshat_curation`. `None`
    /// when no `[upstreams.seshat]` is configured — the task then no-ops.
    /// Unconditional (unlike `db`/`heka_url`): `upstream_client` has no
    /// Postgres dependency.
    seshat_upstream: Option<UpstreamEndpointConfig>,
    /// Heka Postgres URL, for `run_heka_curation`. `None` when no `[heka]`
    /// section is configured — the task then no-ops.
    #[cfg(feature = "postgres")]
    heka_url: Option<String>,
    /// Whether `run_heka_curation` has already WARNed about a nonzero
    /// unembedded-chunk backlog. Reset to `false` once the backlog clears —
    /// tracked so the WARN fires once per backlog episode instead of every
    /// tick until the Phase 42 service layer can act on it.
    #[cfg(feature = "postgres")]
    heka_warned_unembedded: AtomicBool,
}

impl AgentDaemon {
    /// Create a new daemon wrapping the given shared agent.
    pub fn new(agent: Arc<RwLock<Agent>>, config: DaemonConfig) -> Self {
        Self {
            agent,
            config,
            total_cycles: 0,
            paused: Arc::new(AtomicBool::new(false)),
            cycle_done: Arc::new(tokio::sync::Notify::new()),
            shutdown: None,
            status: None,
            autonomy: AutonomyConfig::default(),
            last_equivalence_triple_count: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(feature = "postgres")]
            db: None,
            seshat_upstream: None,
            #[cfg(feature = "postgres")]
            heka_url: None,
            #[cfg(feature = "postgres")]
            heka_warned_unembedded: AtomicBool::new(false),
        }
    }

    /// Set the per-akh automation gating (from `akh.toml` `[autonomy]`).
    pub fn with_autonomy(mut self, autonomy: AutonomyConfig) -> Self {
        self.autonomy = autonomy;
        self
    }

    /// Whether a gated background task is enabled for this akh. Always-on infra
    /// tasks (persist, trigger evaluation) do not consult this.
    ///
    /// Task names are free-form strings ([`AutonomyConfig::task_enabled`]
    /// falls back to the master switch on an unrecognized one, silently) —
    /// see [`KNOWN_TASK_NAMES`] for the full registry consulted by this
    /// method across every gated `tokio::select!` arm in [`Self::run`].
    fn task_enabled(&self, task: &str) -> bool {
        self.autonomy.task_enabled(task)
    }

    /// Pause the daemon — prevents new cycle/learning work from acquiring
    /// the agent lock. Returns handles the caller uses to resume.
    pub fn pause_handle(&self) -> (Arc<AtomicBool>, Arc<tokio::sync::Notify>) {
        (Arc::clone(&self.paused), Arc::clone(&self.cycle_done))
    }

    /// Replace the internal pause handles with externally-managed ones.
    /// Caller creates the handles, registers them in McpState, and passes
    /// clones here so the daemon respects external pause requests.
    pub fn with_pause_handles(
        mut self,
        paused: Arc<AtomicBool>,
        cycle_done: Arc<tokio::sync::Notify>,
    ) -> Self {
        self.paused = paused;
        self.cycle_done = cycle_done;
        self
    }

    /// Attach an external shutdown receiver (replaces Ctrl+C handling).
    pub fn with_shutdown(mut self, rx: tokio::sync::watch::Receiver<bool>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    /// Attach a shared status for real-time monitoring.
    pub fn with_status(mut self, status: Arc<tokio::sync::Mutex<DaemonStatus>>) -> Self {
        self.status = Some(status);
        self
    }

    /// Attach the Postgres backing store + this akh's stable id, enabling the
    /// ADR 091 sleep-cycle transcript-harvest phase. Omit (local-only mode) to
    /// leave harvest a no-op.
    #[cfg(feature = "postgres")]
    pub fn with_db(
        mut self,
        db: akhomed_store::DatabaseConnection,
        akh_id: akhomed_store::uuid::Uuid,
    ) -> Self {
        self.db = Some((db, akh_id));
        self
    }

    /// Attach the Seshat's Archive upstream endpoint, enabling
    /// `run_seshat_curation`. Omit to leave that task a no-op. Unconditional
    /// — no Postgres dependency, unlike `with_db`/`with_heka`.
    pub fn with_seshat_upstream(mut self, upstream: UpstreamEndpointConfig) -> Self {
        self.seshat_upstream = Some(upstream);
        self
    }

    /// Attach the Heka Postgres connection URL, enabling the v1
    /// surveillance-only `run_heka_curation` pass. Omit (no `[heka]`
    /// section) to leave it a no-op.
    #[cfg(feature = "postgres")]
    pub fn with_heka(mut self, url: String) -> Self {
        self.heka_url = Some(url);
        self
    }

    /// Acquire the agent lock, recovering from poison rather than panicking.
    ///
    /// A poisoned mutex means a previous task panicked while holding the guard.
    /// Recovering via `into_inner()` clears the poison flag so all subsequent
    /// callers succeed normally. The agent data may be slightly inconsistent if
    /// the prior panic left a write half-applied, but that is still preferable
    /// to permanently blocking every future operation on this workspace.
    ///
    /// Locking discipline: each per-workspace `Arc<RwLock<Agent>>` is independent.
    /// A panic (and subsequent poison) in workspace A cannot affect workspace B.
    fn lock_agent(&self) -> std::sync::RwLockWriteGuard<'_, Agent> {
        self.agent.write().unwrap_or_else(|e| {
            tracing::warn!("agent mutex was poisoned; recovering (a previous task may have panicked while holding the lock)");
            e.into_inner()
        })
    }

    /// Try to acquire a write lock on the agent without blocking.
    /// Returns `None` if the lock is currently held (by a live ACP turn
    /// or another background task). The daemon uses this to yield to
    /// session work: if a turn is in progress, the daemon skips its
    /// cycle rather than blocking the turn.
    fn try_write_agent(&self) -> Option<std::sync::RwLockWriteGuard<'_, Agent>> {
        self.agent.try_write().ok()
    }

    /// Run the main daemon loop.
    ///
    /// Schedules all background tasks as tokio intervals and selects on them.
    /// Shuts down cleanly on Ctrl+C (or external shutdown signal), persisting
    /// the session before exit.
    ///
    /// ## Concurrency note
    ///
    /// Every background method (`run_*`) is synchronous — it acquires
    /// `std::sync::Mutex` guards and calls redb write transactions.  Calling
    /// blocking code directly on a Tokio async thread starves the runtime,
    /// preventing other async tasks (MCP handlers, HTTP requests) from making
    /// progress.  We therefore wrap every such call in
    /// `tokio::task::block_in_place`, which signals Tokio to migrate waiting
    /// async tasks to other threads before this thread blocks.
    pub async fn run(&mut self) -> AgentResult<()> {
        use tokio::time::interval;

        let mut equivalence_tick = interval(self.config.equivalence_interval);
        let mut reflection_tick = interval(self.config.reflection_interval);
        let mut consolidation_tick = interval(self.config.consolidation_interval);
        let mut schema_tick = interval(self.config.schema_discovery_interval);
        let mut rules_tick = interval(self.config.rule_inference_interval);
        let mut gaps_tick = interval(self.config.gap_analysis_interval);
        let mut persist_tick = interval(self.config.persist_interval);
        let mut idle_cycle_tick = interval(self.config.idle_cycle_interval);
        let mut continuous_learning_tick = interval(self.config.continuous_learning_interval);
        let mut goal_gen_tick = interval(self.config.goal_generation_interval);
        let mut sleep_tick = interval(self.config.sleep_cycle_interval);
        let mut trigger_tick = interval(self.config.trigger_evaluation_interval);
        let mut extraction_tick = interval(self.config.knowledge_extraction_interval);
        let mut bridge_training_tick = interval(self.config.bridge_training_interval);
        let mut openclaw_sync_tick = interval(self.config.openclaw_sync_interval);
        let mut elicitation_tick = interval(self.config.knowledge_elicitation_interval);
        let mut kg_compaction_tick = interval(self.config.kg_compaction_interval);
        let mut delegation_tick = interval(self.config.delegation_interval);
        let mut relation_enrichment_tick = interval(self.config.relation_enrichment_interval);
        let mut librarian_tick = interval(self.config.librarian_interval);
        let mut seshat_curation_tick = interval(self.config.seshat_curation_interval);
        let mut heka_curation_tick = interval(self.config.heka_curation_interval);

        tokio::task::block_in_place(|| {
            self.lock_agent().sink().emit(&AkhMessage::system(
                "daemon started — background learning active",
            ));
        });

        // Startup burst (P5-gated): seed the causal world model + initial goals
        // so an autonomy-enabled akh can act immediately. A fresh akh with
        // automation off seeds nothing and stays idle until prompted.
        // Continuous learning is deliberately NOT run at startup — it holds the
        // agent mutex for 30-120s (network I/O + ingestion), blocking handlers.
        if self.task_enabled("goal_generation") {
            self.bootstrap_causal_schemas();
            tokio::task::block_in_place(|| self.run_goal_generation());
        }

        loop {
            // When paused, spin-sleep — prevents daemon from acquiring
            // the agent lock while batch writes are in progress.
            if self.paused.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }

            // P5: each cognitive task arm is gated by `task_enabled(..)`; the
            // ticker still fires but does nothing when disabled. `persist` and
            // `trigger` are always-on infra (session durability + user-authored
            // triggers, which are opt-in by being added).
            tokio::select! {
                _ = equivalence_tick.tick() => {
                    if self.task_enabled("equivalence") {
                        tokio::task::block_in_place(|| self.run_equivalence_learning());
                    }
                }
                _ = reflection_tick.tick() => {
                    if self.task_enabled("reflection") {
                        tokio::task::block_in_place(|| self.run_reflection());
                    }
                }
                _ = consolidation_tick.tick() => {
                    if self.task_enabled("consolidation") {
                        tokio::task::block_in_place(|| self.run_consolidation_check());
                    }
                }
                _ = schema_tick.tick() => {
                    if self.task_enabled("schema_discovery") {
                        tokio::task::block_in_place(|| self.run_schema_discovery());
                    }
                }
                _ = rules_tick.tick() => {
                    if self.task_enabled("rule_inference") {
                        tokio::task::block_in_place(|| self.run_rule_inference());
                    }
                }
                _ = gaps_tick.tick() => {
                    if self.task_enabled("gap_analysis") {
                        tokio::task::block_in_place(|| self.run_gap_analysis());
                    }
                }
                _ = persist_tick.tick() => {
                    // Always on: session persistence is infra, not autonomy.
                    tokio::task::block_in_place(|| self.persist());
                }
                _ = idle_cycle_tick.tick() => {
                    if self.task_enabled("idle_cycle") {
                        tokio::task::block_in_place(|| self.run_idle_cycle());
                        if self.config.max_cycles > 0 && self.total_cycles >= self.config.max_cycles {
                            tokio::task::block_in_place(|| {
                                self.lock_agent().sink().emit(&AkhMessage::system(
                                    format!(
                                        "daemon: max cycles ({}) reached, shutting down",
                                        self.config.max_cycles,
                                    ),
                                ));
                            });
                            break;
                        }
                    }
                }
                _ = continuous_learning_tick.tick() => {
                    if self.task_enabled("continuous_learning") {
                        tokio::task::block_in_place(|| self.run_continuous_learning());
                    }
                }
                _ = goal_gen_tick.tick() => {
                    if self.task_enabled("goal_generation") {
                        tokio::task::block_in_place(|| self.run_goal_generation());
                    }
                }
                _ = sleep_tick.tick() => {
                    if self.task_enabled("sleep_cycle") {
                        tokio::task::block_in_place(|| self.run_sleep_cycle());
                    }
                }
                _ = trigger_tick.tick() => {
                    // Always on: user-authored triggers are opt-in by existing.
                    tokio::task::block_in_place(|| self.run_trigger_evaluation());
                }
                _ = extraction_tick.tick() => {
                    if self.task_enabled("knowledge_extraction") {
                        tokio::task::block_in_place(|| self.run_knowledge_extraction());
                    }
                }
                _ = bridge_training_tick.tick() => {
                    if self.task_enabled("bridge_training") {
                        // run_bridge_training dispatches heavy work to spawn_blocking
                        // internally; block_in_place covers only the brief mutex
                        // acquisition at the start.
                        tokio::task::block_in_place(|| self.run_bridge_training());
                    }
                }
                _ = openclaw_sync_tick.tick() => {
                    if self.task_enabled("openclaw_sync") {
                        self.run_openclaw_sync();
                    }
                }
                _ = elicitation_tick.tick() => {
                    if self.task_enabled("knowledge_elicitation") {
                        self.run_knowledge_elicitation_task();
                    }
                }
                _ = kg_compaction_tick.tick() => {
                    if self.task_enabled("kg_compaction") {
                        self.run_kg_compaction();
                    }
                }
                _ = delegation_tick.tick() => {
                    if self.task_enabled("delegation") {
                        self.run_delegation_tick().await;
                    }
                }
                _ = relation_enrichment_tick.tick() => {
                    if self.task_enabled("relation_enrichment") {
                        tokio::task::block_in_place(|| self.run_relation_enrichment());
                    }
                }
                _ = librarian_tick.tick() => {
                    // ADR 081 slice 4: gated on AKH_LIBRARIAN=1 *inside*
                    // run_librarian_task (env-gated, like `openclaw_sync`'s
                    // OPENCLAW_API_KEY check) rather than `task_enabled(..)`,
                    // since this is an opt-in LLM-cost pass, not a
                    // per-workspace autonomy toggle. No-op without the
                    // `postgres` feature (turn material lives only in the WAL).
                    #[cfg(feature = "postgres")]
                    tokio::task::block_in_place(|| self.run_librarian_task());
                }
                _ = seshat_curation_tick.tick() => {
                    if self.task_enabled("seshat_curation") {
                        tokio::task::block_in_place(|| self.run_seshat_curation());
                    }
                }
                _ = heka_curation_tick.tick() => {
                    if self.task_enabled("heka_curation") {
                        self.run_heka_curation().await;
                    }
                }
                _ = self.wait_for_shutdown() => {
                    tokio::task::block_in_place(|| {
                        self.lock_agent().sink().emit(&AkhMessage::system(
                            "daemon: received shutdown signal, persisting session...",
                        ));
                    });
                    break;
                }
            }
        }

        // Final persist on shutdown.
        tokio::task::block_in_place(|| self.persist());
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.running = false;
        }
        tokio::task::block_in_place(|| {
            self.agent
                .read()
                .unwrap()
                .sink()
                .emit(&AkhMessage::system("daemon stopped"));
        });
        Ok(())
    }

    /// Wait for either the external shutdown receiver, Ctrl+C, or SIGTERM.
    async fn wait_for_shutdown(&mut self) {
        if let Some(ref mut rx) = self.shutdown {
            let _ = rx.changed().await;
        } else {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to register SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c.await.ok();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Background task methods
    // -----------------------------------------------------------------------

    fn run_equivalence_learning(&mut self) {
        let engine = self.lock_agent().engine.clone();
        let current = engine.triple_count();
        let last = self
            .last_equivalence_triple_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if current == last {
            tracing::debug!(
                current,
                last,
                "daemon: equivalence learning skipped, no new triples"
            );
            return;
        }

        match engine.learn_equivalences() {
            Ok(count) => {
                if count > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:equivalence] {count} new equivalences discovered",
                        )));
                    }
                    tracing::info!(count, "daemon: equivalence learning complete");
                } else {
                    tracing::debug!("daemon: equivalence learning complete: 0 equivalences");
                }
                self.last_equivalence_triple_count
                    .store(current, std::sync::atomic::Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: equivalence learning failed");
            }
        }
    }

    fn run_reflection(&mut self) {
        let mut agent = self.lock_agent();
        match agent.reflect() {
            Ok(result) => {
                let adj_count = result.adjustments.len();
                // Auto-apply safe priority adjustments.
                let safe: Vec<_> = result
                    .adjustments
                    .iter()
                    .filter(|a| {
                        matches!(
                            a,
                            super::reflect::Adjustment::IncreasePriority { .. }
                                | super::reflect::Adjustment::DecreasePriority { .. }
                        )
                    })
                    .cloned()
                    .collect();
                let applied = agent.apply_adjustments(&safe).unwrap_or(0);
                if adj_count > 0 {
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:reflection] {adj_count} insights, {applied} adjustments applied",
                    )));
                }
                tracing::info!(adj_count, applied, "daemon: reflection complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: reflection failed");
            }
        }
    }

    fn run_consolidation_check(&mut self) {
        let mut agent = self.lock_agent();
        let wm_len = agent.working_memory().len();
        let threshold = agent.config.consolidation.auto_consolidate_at;
        if wm_len < threshold {
            return;
        }

        match agent.consolidate() {
            Ok(result) => {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:consolidation] {} entries evicted, {} episodes stored",
                    result.entries_evicted,
                    result.episodes_created.len(),
                )));
                tracing::info!(
                    evicted = result.entries_evicted,
                    episodes = result.episodes_created.len(),
                    "daemon: consolidation complete",
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: consolidation failed");
            }
        }
    }

    fn run_schema_discovery(&self) {
        let engine = self.lock_agent().engine.clone();
        let config = crate::autonomous::schema::SchemaDiscoveryConfig::default();
        match engine.discover_schema(config) {
            Ok(result) => {
                let types = result.types.len();
                let hierarchies = result.relation_hierarchies.len();
                if types > 0 || hierarchies > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:schema] discovered {types} types, {hierarchies} relation hierarchies",
                        )));
                    }
                }
                tracing::info!(types, hierarchies, "daemon: schema discovery complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: schema discovery failed");
            }
        }
    }

    fn run_rule_inference(&self) {
        let engine = self.lock_agent().engine.clone();
        let config = crate::autonomous::rule_engine::RuleEngineConfig::default();
        match engine.run_rules(config) {
            Ok(result) => {
                let derived_count = result.derived.len();
                if derived_count > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:rules] {} new triples from {} iterations",
                            derived_count, result.iterations,
                        )));
                    }
                }
                tracing::info!(
                    triples = derived_count,
                    iterations = result.iterations,
                    "daemon: rule inference complete",
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: rule inference failed");
            }
        }
    }

    fn run_gap_analysis(&self) {
        use super::goal;

        let (engine, goal_ids) = {
            let agent = self.lock_agent();
            let active = goal::active_goals(agent.goals());
            if active.is_empty() {
                return;
            }
            let ids: Vec<_> = active.iter().map(|g| g.symbol_id).collect();
            (agent.engine.clone(), ids)
        };

        let config = crate::autonomous::gap::GapAnalysisConfig::default();
        match engine.analyze_gaps(&goal_ids, config) {
            Ok(result) => {
                let gaps = result.gaps.len();
                if gaps > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:gaps] {gaps} knowledge gaps identified around active goals",
                        )));
                    }
                }
                tracing::info!(gaps, "daemon: gap analysis complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: gap analysis failed");
            }
        }
    }

    fn run_continuous_learning(&self) {
        // Extract what we need from the agent, then DROP the lock before the
        // expensive learning pipeline runs.  `run_continuous_learning` only
        // needs an `&Arc<Engine>` and a curiosity config — it does not mutate
        // the Agent, so holding the mutex during network I/O + ingestion would
        // needlessly block every HTTP handler for 30-120s.
        let (engine, curiosity_cfg) = {
            let agent = self.lock_agent();
            (agent.engine.clone(), agent.curiosity_config.clone())
        };
        // Lock is released here — HTTP handlers can proceed.

        let config = super::continuous_learning::ContinuousLearningConfig::default();
        match super::continuous_learning::run_continuous_learning(&engine, &curiosity_cfg, &config)
        {
            Ok(result) => {
                // Re-acquire lock briefly just for the status message.
                if let Ok(agent) = self.agent.read() {
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:learning] {} targets, {} resources, {} ingested, Dreyfus: {}",
                        result.targets_found,
                        result.resources_discovered,
                        result.concepts_ingested,
                        result.dreyfus_level,
                    )));
                }
                tracing::info!(
                    targets = result.targets_found,
                    resources = result.resources_discovered,
                    ingested = result.concepts_ingested,
                    dreyfus = %result.dreyfus_level,
                    "daemon: continuous learning complete",
                );
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.last_learning_at = Some(now_secs());
                }
            }
            Err(e) => {
                tracing::info!(error = %e, "daemon: continuous learning skipped");
            }
        }
    }

    fn run_goal_generation(&mut self) {
        // Phase 1: Brief lock — snapshot inputs + update drives (needs &WorkingMemory).
        let (
            engine,
            goals_snapshot,
            drives_snapshot,
            gen_config,
            predicates,
            cycle_count,
            impasse,
            reflection,
            watch_firings,
            curiosity_config,
        ) = {
            let mut guard = self.lock_agent();
            // Explicit reborrow so the borrow checker can split field access.
            let agent: &mut Agent = &mut *guard;

            // Update drives while we hold the lock (needs &WorkingMemory which is !Clone).
            let goal_symbols: Vec<crate::symbol::SymbolId> =
                super::goal::active_goals(&agent.goals)
                    .iter()
                    .map(|g| g.symbol_id)
                    .collect();
            agent.drives.update_with_curiosity(
                &agent.engine,
                &agent.working_memory,
                &goal_symbols,
                agent.cycle_count,
                Some(&agent.curiosity_config),
            );

            (
                agent.engine.clone(),
                agent.goals.clone(),
                agent.drives.clone(),
                agent.config.goal_generation.clone(),
                agent.predicates.clone(),
                agent.cycle_count,
                agent.last_impasse.clone(),
                agent.last_reflection.clone(),
                agent.last_watch_firings.clone(),
                agent.curiosity_config.clone(),
            )
        };
        // Lock released — HTTP handlers can proceed.

        // Phase 2: Expensive computation (no lock held).
        let goal_symbols_snapshot: Vec<crate::symbol::SymbolId> = super::goal::active_goals(&goals_snapshot)
            .iter()
            .map(|g| g.symbol_id)
            .collect();
        let curiosity_proposals = super::curiosity::compute_directed_curiosity(
            &engine,
            &curiosity_config,
            &goal_symbols_snapshot,
        )
        .map(|report| report.proposals)
        .unwrap_or_default();

        let gen_result = match super::goal_generation::generate_goals(
            &engine,
            &goals_snapshot,
            &drives_snapshot,
            &gen_config,
            &predicates,
            cycle_count,
            impasse.as_ref(),
            reflection.as_ref(),
            &watch_firings,
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "daemon: goal generation skipped");
                return;
            }
        };

        // Merge curiosity proposals.
        let mut gen_result = gen_result;
        for cp in curiosity_proposals {
            if gen_result.activated.len() < gen_config.max_activations {
                gen_result.activated.push(cp);
            } else if gen_result.dormant.len() < gen_config.max_dormant {
                gen_result.dormant.push(cp);
            }
        }

        // Phase 3: Brief lock — apply results (push goals, store provenance).
        let mut agent = self.lock_agent();

        for proposal in &gen_result.activated {
            match super::goal::create_goal(
                &agent.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &agent.predicates,
            ) {
                Ok(mut g) => {
                    g.source = Some(proposal.source.clone());
                    let (drive_name, drive_strength) =
                        super::goal_generation::provenance_from_source(&proposal.source);
                    let mut prov = crate::provenance::ProvenanceRecord::new(
                        g.symbol_id,
                        crate::provenance::DerivationKind::AutonomousGoalGeneration {
                            drive: drive_name,
                            strength: drive_strength,
                        },
                    )
                    .with_confidence(proposal.feasibility);
                    if let Err(e) = agent.engine.store_provenance(&mut prov) {
                        tracing::debug!("failed to store goal provenance: {e}");
                    }
                    agent.goals.push(g);
                }
                Err(e) => tracing::debug!("failed to create activated goal: {e}"),
            }
        }

        for proposal in &gen_result.dormant {
            match super::goal::create_goal(
                &agent.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &agent.predicates,
            ) {
                Ok(mut g) => {
                    g.source = Some(proposal.source.clone());
                    let _ = super::goal::update_goal_status(
                        &agent.engine,
                        &mut g,
                        super::goal::GoalStatus::Dormant,
                        &agent.predicates,
                    );
                    agent.goals.push(g);
                }
                Err(e) => tracing::debug!("failed to create dormant goal: {e}"),
            }
        }

        // Clear impasse after it's been consumed.
        agent.last_impasse = None;

        let activated = gen_result.activated.len();
        let dormant = gen_result.dormant.len();
        if activated > 0 || dormant > 0 {
            agent.sink().emit(&AkhMessage::system(format!(
                "[daemon:goals] {activated} activated, {dormant} dormant",
            )));
        }
        tracing::info!(activated, dormant, "daemon: goal generation complete");

        let goal_count = super::goal::active_goals(&agent.goals).len();
        drop(agent); // Release agent lock before status lock.
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.last_goal_gen_at = Some(now_secs());
            st.active_goals = goal_count;
        }
    }

    fn run_sleep_cycle(&mut self) {
        let mut guard = self.lock_agent();
        // Explicit reborrow so the borrow checker can see partial field access.
        let agent: &mut Agent = &mut *guard;
        match super::sleep::run_sleep_cycle(
            &agent.engine,
            &agent.working_memory,
            &mut agent.sleep_cycle,
            agent.cycle_count,
        ) {
            Ok(metrics) => {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:sleep] replayed {}, merged {}, pruned {}, dream {}",
                    metrics.episodes_replayed,
                    metrics.duplicates_merged,
                    metrics.orphans_pruned,
                    metrics.dream_connections_found,
                )));
                tracing::info!(?metrics, "daemon: sleep cycle complete");

                // ADR 091: transcript-harvest phase, still inside the agent
                // lock (this whole method already runs inside the caller's
                // `block_in_place`) so the store calls' internal `block_on`
                // never blocks a live async task on this thread.
                #[cfg(feature = "postgres")]
                if let Some((db, akh_id)) = self.db.as_ref() {
                    match super::harvest::harvest_daily_log(
                        db,
                        &agent.engine,
                        agent.predicates(),
                        *akh_id,
                    ) {
                        Ok(hm) => tracing::info!(
                            turns_harvested = hm.turns_harvested,
                            episodes_minted = hm.episodes_minted,
                            last_seq = hm.last_seq,
                            "daemon: transcript harvest complete"
                        ),
                        Err(e) => tracing::warn!(error = %e, "daemon: transcript harvest failed"),
                    }
                }

                drop(guard); // Release agent lock before status lock.
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.last_sleep_at = Some(now_secs());
                }
            }
            Err(e) => tracing::debug!(error = %e, "daemon: sleep cycle skipped"),
        }
    }

    /// ADR 081 slice 4 — daily Librarian pass: LLM extraction of typed
    /// triples from captured transcript turns against the in-force schema.
    ///
    /// Off by default (`AKH_LIBRARIAN=1` opts in) — unlike the always-on
    /// harvest phase's pure KG bookkeeping, this is a real per-tick LLM cost
    /// (ADR 081 decision 10: gpt-oss 120B). No-op without a configured
    /// Postgres store, matching every other store-backed background task's
    /// local-only degradation (turn material lives only in the WAL).
    #[cfg(feature = "postgres")]
    fn run_librarian_task(&self) {
        if std::env::var("AKH_LIBRARIAN").as_deref() != Ok("1") {
            return;
        }
        let Some((db, akh_id)) = self.db.as_ref() else {
            tracing::debug!("daemon: librarian pass skipped — no Postgres store configured");
            return;
        };

        let engine = {
            let agent = self.lock_agent();
            agent.engine.clone()
        };

        let model = super::librarian::OllamaLibrarianModel::from_env();
        let config = super::librarian::LibrarianConfig::default();

        // No notes supplied yet — the Note carrier (ADR 081 P1) has no
        // durable "all notes since X" catalog query today, so this daemon
        // wiring covers the turn-WAL half of slice 4 only; a caller with a
        // note catalog can call `run_librarian_pass_over_wal` directly with
        // its own `notes` slice.
        match super::librarian::run_librarian_pass_over_wal(db, &engine, &model, *akh_id, &[], &config) {
            Ok(report) => {
                if report.typed_triples_asserted > 0 || report.frontier_triples_asserted > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:librarian] {} source(s): +{} typed, +{} frontier triple(s)",
                            report.sources_processed,
                            report.typed_triples_asserted,
                            report.frontier_triples_asserted,
                        )));
                    }
                }
                tracing::info!(
                    sources_processed = report.sources_processed,
                    sources_skipped_empty = report.sources_skipped_empty,
                    typed = report.typed_triples_asserted,
                    frontier = report.frontier_triples_asserted,
                    last_seq = report.last_seq,
                    "daemon: librarian pass complete",
                );
            }
            Err(e) => tracing::warn!(error = %e, "daemon: librarian pass failed"),
        }
    }

    /// Periodic Seshat's Archive curation pass.
    ///
    /// Surveys the configured `[upstreams.seshat]` endpoint's corpus status,
    /// forces an embedding pass when chunks are pending, and asserts the
    /// resulting counts into the KG so the akh can reason about its own
    /// library. Gated by `task_enabled("seshat_curation")` — see
    /// [`KNOWN_TASK_NAMES`].
    ///
    /// Deliberately **not** in v1: pushing KG summaries back into Seshat via
    /// `seshat_ingest`. That tool appends to an ever-growing corpus with no
    /// delete/dedupe support — doing this safely needs slug-idempotency
    /// verified first (re-ingesting the same summary on every tick would
    /// otherwise flood the archive).
    fn run_seshat_curation(&self) {
        let Some(cfg) = self.seshat_upstream.clone() else {
            tracing::debug!("daemon: seshat curation skipped — no [upstreams.seshat] configured");
            return;
        };

        use crate::upstream_client::UpstreamMcpClient;

        // `UpstreamEndpointConfig::url` is a base; the `/mcp` transport path
        // is appended here the same way `UpstreamMcpClient::new` does it, but
        // with a longer timeout — `seshat_embed_now` over a real backlog can
        // run well past the client's 30s default.
        let base = cfg.url.trim_end_matches('/');
        let client = UpstreamMcpClient::from_endpoint_with_timeout(
            format!("{base}/mcp"),
            cfg.resolved_auth(),
            Duration::from_secs(120),
        );

        let status_text = match client.call_tool_text("seshat_status", serde_json::json!({})) {
            Ok(text) => text,
            Err(e) => {
                tracing::debug!(error = %e, "daemon: seshat_status call failed, skipping curation pass");
                return;
            }
        };
        let status: serde_json::Value = match serde_json::from_str(&status_text) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(error = %e, "daemon: seshat_status response was not valid JSON");
                return;
            }
        };

        let total_documents = status["total_documents"].as_u64().unwrap_or(0);
        let total_chunks = status["total_chunks"].as_u64().unwrap_or(0);
        let embedded_chunks = status["embedded_chunks"].as_u64().unwrap_or(0);
        let pending_embedding = status["pending_embedding"].as_u64().unwrap_or(0);
        let cached_microtheories = status["cached_microtheories"].as_u64().unwrap_or(0);

        let mut chunks_processed = 0u64;
        let mut embeddings_created = 0u64;
        if pending_embedding > 0 {
            match client.call_tool_text("seshat_embed_now", serde_json::json!({})) {
                Ok(text) => {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                        chunks_processed = v["chunks_processed"].as_u64().unwrap_or(0);
                        embeddings_created = v["embeddings_created"].as_u64().unwrap_or(0);
                    }
                }
                Err(e) => {
                    // A seshd built without its `embedding` feature hard-errors
                    // on every `seshat_embed_now` call — DEBUG (not WARN) so an
                    // akh pointed at such a build doesn't spam its logs on every
                    // curation tick.
                    tracing::debug!(
                        error = %e,
                        "daemon: seshat_embed_now failed (upstream may be built without the `embedding` feature)"
                    );
                }
            }
        }

        // Assert corpus stats into the KG so the akh can reason about its
        // library. The engine Arc is cloned under a brief lock *after* all
        // upstream I/O above has completed — never held across a network call.
        let engine = { self.lock_agent().engine.clone() };
        let now = now_secs().to_string();
        assert_singleton_fact(
            &engine,
            "seshat-archive",
            "has-document-count",
            &total_documents.to_string(),
        );
        assert_singleton_fact(
            &engine,
            "seshat-archive",
            "has-pending-embeddings",
            &pending_embedding.to_string(),
        );
        assert_singleton_fact(&engine, "seshat-archive", "last-curated-at", &now);

        if let Ok(agent) = self.agent.read() {
            agent.sink().emit(&AkhMessage::system(format!(
                "[daemon:seshat_curation] {total_documents} doc(s), {embedded_chunks}/{total_chunks} \
                 chunk(s) embedded, {pending_embedding} pending, {cached_microtheories} cached \
                 microtheor{plural}; embed_now: +{chunks_processed} chunk(s), +{embeddings_created} \
                 embedding(s)",
                plural = if cached_microtheories == 1 { "y" } else { "ies" },
            )));
        }
        tracing::info!(
            total_documents,
            total_chunks,
            embedded_chunks,
            pending_embedding,
            cached_microtheories,
            chunks_processed,
            embeddings_created,
            "daemon: seshat curation pass complete",
        );
    }

    /// Periodic Heka long-term-memory store curation pass.
    ///
    /// **v1 is surveillance-only.** It reports unembedded-chunk and
    /// untagged-drawer counts from the configured `[heka]` Postgres database
    /// into the KG; it does not (yet) backfill embeddings, tag drawers, or
    /// prune anything. Heka has no ingest/query/embed service today (Phase 42
    /// hasn't landed) — there is nothing this pass could call to *act* on a
    /// backlog even if it wanted to, so it only measures one.
    ///
    /// A thin dispatcher (mirroring `run_delegation_tick`): the real body
    /// lives in `run_heka_curation_inner`, compiled only under the
    /// `postgres` feature (raw SQL against `akhomed_store`'s sea_orm
    /// connection type — see the branch's commit message for why this reuses
    /// that instead of depending on the `heka` crate directly).
    async fn run_heka_curation(&self) {
        #[cfg(feature = "postgres")]
        self.run_heka_curation_inner().await;
    }

    #[cfg(feature = "postgres")]
    async fn run_heka_curation_inner(&self) {
        use akhomed_store::sea_orm::{ConnectionTrait, DbBackend, Statement};

        let Some(url) = self.heka_url.clone() else {
            tracing::debug!("daemon: heka curation skipped — no [heka] configured");
            return;
        };

        let opts = akhomed_store::PoolOptions::new(url);
        let db = match akhomed_store::connect(&opts).await {
            Ok(db) => db,
            Err(e) => {
                tracing::warn!(error = %e, "daemon: failed to connect to heka Postgres");
                return;
            }
        };

        // Matches the partial index `idx_heka_chunks_unembedded` predicate
        // exactly (crates/heka/migrations/20260427_000001_init.sql) so the
        // planner can use it.
        let unembedded = match db
            .query_one(Statement::from_string(
                DbBackend::Postgres,
                "SELECT count(*) AS n FROM heka_drawer_chunks WHERE embedded_at IS NULL".to_owned(),
            ))
            .await
        {
            Ok(Some(row)) => row.try_get::<i64>("", "n").unwrap_or(0),
            Ok(None) => 0,
            Err(e) => {
                tracing::warn!(error = %e, "daemon: heka unembedded-chunk count query failed");
                return;
            }
        };

        // Matches the partial index `idx_heka_drawers_untagged` predicate
        // exactly (same migration).
        let untagged = match db
            .query_one(Statement::from_string(
                DbBackend::Postgres,
                "SELECT count(*) AS n FROM heka_drawers WHERE array_length(tags::text[], 1) IS NULL"
                    .to_owned(),
            ))
            .await
        {
            Ok(Some(row)) => row.try_get::<i64>("", "n").unwrap_or(0),
            Ok(None) => 0,
            Err(e) => {
                tracing::warn!(error = %e, "daemon: heka untagged-drawer count query failed");
                return;
            }
        };

        let engine = { self.lock_agent().engine.clone() };
        let now = now_secs().to_string();
        assert_singleton_fact(
            &engine,
            "heka-store",
            "has-unembedded-chunks",
            &unembedded.to_string(),
        );
        assert_singleton_fact(
            &engine,
            "heka-store",
            "has-untagged-drawers",
            &untagged.to_string(),
        );
        assert_singleton_fact(&engine, "heka-store", "last-curated-at", &now);

        if unembedded > 0 {
            // WARN once per backlog episode, not every tick — nothing can
            // embed these until the Phase 42 service layer lands.
            if !self
                .heka_warned_unembedded
                .swap(true, Ordering::Relaxed)
            {
                tracing::warn!(
                    unembedded_chunks = unembedded,
                    "daemon: heka has unembedded chunks, but no Phase 42 service exists yet \
                     to embed them — this is a standing backlog until that lands",
                );
            }
        } else {
            self.heka_warned_unembedded
                .store(false, Ordering::Relaxed);
        }

        if let Ok(agent) = self.agent.read() {
            agent.sink().emit(&AkhMessage::system(format!(
                "[daemon:heka_curation] {unembedded} unembedded chunk(s), {untagged} untagged drawer(s) \
                 (surveillance only — v1)",
            )));
        }
        tracing::info!(
            unembedded,
            untagged,
            "daemon: heka curation pass complete",
        );
    }

    fn run_trigger_evaluation(&mut self) {
        use super::trigger::{self as trigger_mod, TriggerStore};

        let mut guard = self.lock_agent();
        // Explicit reborrow for partial field access.
        let agent: &mut Agent = &mut *guard;

        let triggers = TriggerStore::new(&agent.engine).list();
        if triggers.is_empty() {
            return;
        }

        let trigger_count = triggers.len();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Collect which triggers should fire (immutable access).
        let to_fire: Vec<usize> = triggers
            .iter()
            .enumerate()
            .filter(|(_, t)| trigger_mod::should_fire(t, agent, now))
            .map(|(i, _)| i)
            .collect();

        // Execute and update last-fired (mutable access).
        for idx in to_fire {
            let trigger = &triggers[idx];
            match trigger_mod::execute_trigger(trigger, agent) {
                Ok(msg) => tracing::info!("{msg}"),
                Err(e) => tracing::warn!(error = %e, "trigger execution failed"),
            }
            TriggerStore::new(&agent.engine).update_last_fired(&trigger.id, now);
        }

        // Sync trigger count in shared status.
        drop(guard); // Release agent lock before status lock.
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.trigger_count = trigger_count;
        }
    }

    fn run_idle_cycle(&mut self) {
        use super::goal;

        // Respect pause — don't even try to acquire the agent lock.
        if self.paused.load(Ordering::SeqCst) {
            self.cycle_done.notify_one();
            return;
        }

        // Borrow only the `agent` field (not all of `&self`) so the
        // `self.total_cycles += 1` below does not conflict with the live guard.
        let mut agent = self.agent.write().unwrap_or_else(|e| e.into_inner());
        if goal::active_goals(agent.goals()).is_empty() {
            drop(agent);
            self.cycle_done.notify_one();
            return;
        }

        match agent.run_cycle() {
            Ok(result) => {
                self.total_cycles += 1;
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:cycle] tool={}, success={}",
                    result.decision.chosen_tool, result.action_result.tool_output.success,
                )));
                tracing::info!(
                    tool = %result.decision.chosen_tool,
                    cycle = self.total_cycles,
                    "daemon: OODA cycle complete",
                );
                drop(agent); // Release agent lock before status lock.
                // Sync status for real-time HTTP monitoring.
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.total_cycles = self.total_cycles;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: OODA cycle failed");
            }
        }
        // Notify any waiter that we've released the agent lock.
        self.cycle_done.notify_one();
    }

    /// Run autonomous knowledge extraction on recently-fetched content.
    ///
    /// The akh decides what to extract based on its own curiosity targets:
    /// 1. Query gap analysis for ZPD-proximal concepts with knowledge gaps
    /// 2. For each target, gather any text associated with it (doc:mentions triples)
    /// 3. Run the TripleExtractor (local LLM → external API → regex fallback)
    /// 4. Store extracted triples with LlmTripleExtraction provenance
    ///
    /// Drops the agent lock during extraction (expensive I/O) to avoid blocking
    /// HTTP handlers.
    fn run_knowledge_extraction(&self) {
        use crate::extraction::{ExtractionConfig, TripleExtractor};

        // Extract engine ref and drop agent lock.
        let engine = {
            let agent = self.lock_agent();
            agent.engine.clone()
        };

        // Load extraction config (use defaults for now — config.toml integration
        // reads from the file at daemon startup and passes through DaemonConfig).
        let config = ExtractionConfig::default();
        if !config.enabled {
            return;
        }

        let extractor = TripleExtractor::new(config);

        // Find recently-added content paragraphs that reference gap entities.
        // Strategy: query for `doc:mentions` triples pointing to entities that
        // have few outgoing relations (knowledge gaps).
        let doc_mentions = engine.resolve_symbol("doc:mentions").ok();
        let Some(mentions_pred) = doc_mentions else {
            return; // No content has been ingested yet.
        };

        let mention_triples = engine
            .knowledge_graph()
            .triples_for_predicate(mentions_pred);

        if mention_triples.is_empty() {
            return;
        }

        // Collect paragraph entities that mention under-connected concepts.
        let mut extraction_targets: Vec<(crate::symbol::SymbolId, String)> = Vec::new();
        for (para_id, entity_id) in &mention_triples {
            // Check if the mentioned entity has few outgoing relations (gap signal).
            let outgoing = engine.triples_from(*entity_id);
            if outgoing.len() < 3 {
                let para_label = engine.resolve_label(*para_id);
                // Only process paragraph-like entities (from content ingest).
                if para_label.starts_with("para:") {
                    extraction_targets.push((*para_id, para_label));
                }
            }
        }

        // Limit to avoid excessive extraction in a single cycle.
        extraction_targets.truncate(10);

        if extraction_targets.is_empty() {
            return;
        }

        let mut total_extracted = 0usize;

        for (para_id, para_label) in &extraction_targets {
            // Get the paragraph text from the KG (stored as outgoing triples).
            let para_triples = engine.triples_from(*para_id);
            // Build text from the paragraph's doc:content triple if it exists,
            // or reconstruct from mentioned entity labels.
            let text: String = para_triples
                .iter()
                .filter(|t| {
                    let pred = engine.resolve_label(t.predicate);
                    pred == "doc:mentions"
                })
                .map(|t| engine.resolve_label(t.object))
                .collect::<Vec<_>>()
                .join(", ");

            if text.len() < 20 {
                continue; // Too short to extract from.
            }

            match extractor.extract_and_store(&text, &engine) {
                Ok(count) => total_extracted += count,
                Err(e) => {
                    tracing::debug!(para = %para_label, error = %e, "extraction failed");
                }
            }
        }

        if total_extracted > 0 {
            let _ = engine.persist();
            if let Ok(agent) = self.agent.read() {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:extraction] {total_extracted} triples extracted from {} paragraphs",
                    extraction_targets.len(),
                )));
            }
            tracing::info!(
                triples = total_extracted,
                paragraphs = extraction_targets.len(),
                "daemon: knowledge extraction complete"
            );
        }
    }

    fn run_bridge_training(&self) {
        use crate::training::{BridgeTrainingSession, TrainerConfig};
        use crate::vsa::Dimension;
        use crate::vsa::neural_bridge::NeuralVsaBridge;

        let engine = {
            let agent = self.lock_agent();
            agent.engine.clone()
        };

        // Check if we have enough grounded symbols to train.
        let symbol_count = engine.item_memory().all_symbols().len();
        if symbol_count < 50 {
            tracing::debug!(
                symbols = symbol_count,
                "daemon: bridge training skipped — insufficient grounded symbols (need ≥50)"
            );
            return;
        }

        // Training is CPU-bound — run in spawn_blocking to avoid starving
        // the tokio event loop (which handles HTTP/WS/MCP requests).
        let agent_ref = self.agent.clone();
        tokio::task::spawn_blocking(move || {
            let hidden_dim = 128;
            let mut bridge = NeuralVsaBridge::new(hidden_dim, Dimension::DEFAULT);

            let config = TrainerConfig {
                max_steps: 100,
                ..Default::default()
            };
            let session = BridgeTrainingSession::new(config);
            let result = session.train_from_grounded_symbols(&mut bridge, &engine);

            if result.improved {
                if let Ok(agent) = agent_ref.read() {
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:bridge-training] {} steps, match rate {:.1}% → {:.1}% ({} symbols)",
                        result.steps,
                        result.initial_match_rate * 100.0,
                        result.final_match_rate * 100.0,
                        result.pairs_used,
                    )));
                }
            }
            tracing::info!(
                steps = result.steps,
                pairs = result.pairs_used,
                initial = result.initial_match_rate,
                final_rate = result.final_match_rate,
                improved = result.improved,
                "daemon: bridge training complete"
            );
        });
    }

    fn run_openclaw_sync(&self) {
        use super::memory_sync::{MemorySyncConfig, sync_openclaw_journals};
        use super::openclaw::OpenClawConfig;

        // Load config from env — if the key is absent, skip silently.
        let Some(openclaw_cfg) = OpenClawConfig::from_env() else {
            tracing::debug!("daemon: OPENCLAW_API_KEY not set, skipping journal sync");
            return;
        };

        let (engine, predicates, cycle_count) = {
            let agent = self.agent.read().unwrap();
            (
                agent.engine.clone(),
                agent.predicates.clone(),
                agent.cycle_count,
            )
        };
        // Agent lock released — HTTP I/O runs without blocking other handlers.

        let sync_cfg = MemorySyncConfig::default();
        match sync_openclaw_journals(
            &engine,
            &openclaw_cfg,
            &sync_cfg,
            &predicates,
            cycle_count as u64,
        ) {
            Ok(result) => {
                if result.journals_fetched > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:openclaw] fetched {}, ingested {}, skipped {}",
                            result.journals_fetched, result.ingested, result.skipped,
                        )));
                    }
                }
                tracing::info!(
                    fetched = result.journals_fetched,
                    ingested = result.ingested,
                    skipped = result.skipped,
                    "daemon: OpenClaw journal sync complete",
                );
                // Persist KG changes after sync.
                if result.ingested > 0 {
                    let _ = engine.persist();
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: OpenClaw journal sync failed");
            }
        }
    }

    /// Run one KG compaction pass: remove episode/agent scaffolding triples.
    ///
    /// Releases the agent lock before the scan (CPU-bound) to avoid blocking
    /// HTTP handlers during the triple classification loop.
    fn run_kg_compaction(&self) {
        use super::compaction::{CompactionConfig, run_kg_compaction};

        // Brief lock: clone the engine Arc, then release.
        let engine = {
            let agent = self.agent.read().unwrap();
            agent.engine.clone()
        };

        let config = CompactionConfig::default();
        match run_kg_compaction(&engine, &config) {
            Ok(metrics) => {
                if metrics.scaffolding_removed > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:compaction] removed {}/{} scaffolding triples, density {:.1}%",
                            metrics.scaffolding_removed,
                            metrics.scaffolding_found,
                            metrics.knowledge_density * 100.0,
                        )));
                    }
                }
                tracing::info!(
                    total_before = metrics.total_before,
                    scaffolding_found = metrics.scaffolding_found,
                    scaffolding_removed = metrics.scaffolding_removed,
                    knowledge_density = metrics.knowledge_density,
                    "daemon: KG compaction complete",
                );
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.knowledge_density = Some(metrics.knowledge_density);
                    st.last_compaction_at = Some(now_secs());
                }
            }
            Err(e) => {
                tracing::debug!(error = %e, "daemon: KG compaction skipped");
            }
        }
    }

    /// Run one relation-enrichment pass.
    ///
    /// Injects missing predicate symbols and links orphaned entities to drive
    /// the entity:relation ratio toward the healthy 2–3:1 target range.
    /// Releases the agent lock before the pass runs to avoid blocking HTTP
    /// handlers during the symbol-scan loop.
    fn run_relation_enrichment(&self) {
        use super::relation_enrichment::{RelationEnrichmentConfig, run_relation_enrichment};

        // Brief lock: clone the engine Arc, then release.
        let engine = {
            let agent = self.agent.read().unwrap();
            agent.engine.clone()
        };

        let config = RelationEnrichmentConfig::default();
        match run_relation_enrichment(&engine, &config) {
            Ok(metrics) => {
                if metrics.relations_added > 0 || metrics.triples_added > 0 {
                    if let Ok(agent) = self.agent.read() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:relation-enrichment] +{} predicates, +{} links, ratio {:.2}→{:.2} ({}x under-represented)",
                            metrics.relations_added,
                            metrics.triples_added,
                            metrics.ratio_before,
                            metrics.ratio_after,
                            metrics.under_represented_count,
                        )));
                    }
                }
                tracing::info!(
                    relations_added = metrics.relations_added,
                    triples_added = metrics.triples_added,
                    ratio_before = metrics.ratio_before,
                    ratio_after = metrics.ratio_after,
                    under_represented = metrics.under_represented_count,
                    "daemon: relation enrichment complete",
                );
            }
            Err(e) => {
                tracing::debug!(error = %e, "daemon: relation enrichment skipped");
            }
        }
    }

    /// Phase 32 — proactive LLM knowledge elicitation for gap domains.
    ///
    /// Identifies domains with low triple density from recent gap analysis,
    /// then runs the KnowledgeElicitor (taxonomy + relationships pass) on each.
    /// Falls back silently when the `candle-backend` feature is disabled or the
    /// model file is absent — bootstrap continues via web-API-only mode.
    fn run_knowledge_elicitation_task(&self) {
        use crate::bootstrap::knowledge_elicit::{
            ElicitConfig, ElicitPass, ElicitValidator, KnowledgeElicitor, ingest_validated,
        };

        // Extract engine + data_dir without holding the agent lock.
        let (engine, data_dir) = {
            let agent = self.agent.read().unwrap();
            let dir = agent.engine.config().data_dir.clone();
            (agent.engine.clone(), dir)
        };

        // Locate domains with few outgoing concept triples as elicitation targets.
        // Strategy: look for symbols whose label contains no `:` prefix (plain
        // concept labels) and that have fewer than 5 outgoing relations.
        let expand_pred = engine.resolve_symbol("expand:subclass_of").ok();
        let is_a_pred = engine.resolve_symbol("is-a").ok();

        if expand_pred.is_none() && is_a_pred.is_none() {
            // No domain ontology present yet — nothing to elicit for.
            return;
        }

        // Collect candidate domain labels by sampling concept symbols with low
        // outgoing degree.  Limit to 3 domains per cycle to avoid rate pressure.
        let mut domain_concepts: Vec<(String, Vec<String>)> = Vec::new();

        let all_symbols = engine.item_memory().all_symbols();
        let mut candidates: Vec<(String, usize)> = all_symbols
            .iter()
            .filter_map(|sym| {
                let label = engine.resolve_label(*sym);
                // Skip relation / namespace symbols.
                if label.contains(':') || label.len() < 3 {
                    return None;
                }
                let degree = engine.triples_from(*sym).len();
                // Elicitation targets: concepts with 1–8 outgoing relations
                // (seeded but sparse).
                if degree >= 1 && degree <= 8 {
                    Some((label, degree))
                } else {
                    None
                }
            })
            .collect();

        // Sort by ascending degree (sparsest first).
        candidates.sort_by_key(|(_, d)| *d);
        candidates.truncate(30);

        // Group into domain buckets by shared prefix heuristic or just take the
        // top-N concepts as seeds for a single "general" elicitation.
        if !candidates.is_empty() {
            let seeds: Vec<String> = candidates
                .iter()
                .take(10)
                .map(|(label, _)| label.clone())
                .collect();
            // Use the most common 2-grams as a domain hint, or "general knowledge"
            // as fallback.
            let domain = guess_domain(&seeds).unwrap_or_else(|| "general knowledge".to_string());
            domain_concepts.push((domain, seeds));
        }

        if domain_concepts.is_empty() {
            return;
        }

        let config = ElicitConfig::default();
        let mut elicitor = KnowledgeElicitor::new(config, data_dir);

        if !elicitor.is_available() {
            tracing::debug!("daemon:elicitation: backend unavailable, skipping");
            return;
        }

        let mut total_inserted = 0usize;
        let validator = ElicitValidator::new(None);

        for (domain, seeds) in &domain_concepts {
            let batch = elicitor.elicit(
                domain,
                seeds,
                &[ElicitPass::Taxonomy, ElicitPass::Relationships],
            );

            if batch.triples.is_empty() {
                continue;
            }

            let validated = validator.validate(&batch.triples, batch.total_rounds, &engine);
            let inserted = ingest_validated(&validated, &engine, &batch, ElicitPass::Taxonomy);
            total_inserted += inserted;

            tracing::info!(
                domain,
                elicited = batch.triples.len(),
                accepted = validated.accepted.len(),
                rejected = validated.rejected.len(),
                inserted,
                "daemon:elicitation: domain pass complete"
            );
        }

        if total_inserted > 0 {
            let _ = engine.persist();
            if let Ok(agent) = self.agent.read() {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:elicitation] {} triples generated across {} domain(s)",
                    total_inserted,
                    domain_concepts.len(),
                )));
            }
        }
    }

    fn persist(&self) {
        let agent = self.lock_agent();
        if let Err(e) = agent.persist_session() {
            tracing::warn!(error = %e, "daemon: session persist failed");
        } else {
            tracing::debug!("daemon: session persisted");
            drop(agent); // Release agent lock before status lock.
            if let Some(ref status) = self.status
                && let Ok(mut st) = status.try_lock()
            {
                st.last_persist_at = Some(now_secs());
            }
        }
    }

    /// Ensure the causal world model has schemas for all registered tools.
    ///
    /// Called once at daemon startup so that any tools added after the agent
    /// was last persisted (e.g. new skills installed between runs) are
    /// immediately reflected in the world model.  Existing schemas are not
    /// overwritten, so learned success rates are preserved.
    fn bootstrap_causal_schemas(&mut self) {
        let mut agent = self.agent.write().unwrap();
        let signatures = agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).list();
        if signatures.is_empty() {
            return;
        }
        // Clone the `Arc<Engine>` so the read borrow of `engine` does not
        // conflict with the `&mut` borrow of `causal_manager` below.
        let engine = agent.engine.clone();
        match agent
            .causal_manager
            .bootstrap_from_signatures(&signatures, &engine)
        {
            Ok(0) => {
                tracing::debug!(
                    total = agent.causal_manager.schemas.len(),
                    "daemon: causal world model already up-to-date",
                );
            }
            Ok(n) => {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:causal] bootstrapped {n} new action schema(s) from tool registry \
                     ({} total)",
                    agent.causal_manager.schemas.len(),
                )));
                tracing::info!(
                    new_schemas = n,
                    total = agent.causal_manager.schemas.len(),
                    "daemon: causal world model bootstrapped",
                );
                if let Err(e) = agent.causal_manager.persist(&engine) {
                    tracing::warn!("daemon: failed to persist causal schemas: {e}");
                }
            }
            Err(e) => {
                tracing::warn!("daemon: failed to bootstrap causal schemas: {e}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase 36: Delegation tick
    // -----------------------------------------------------------------------

    /// Process the delegation subsystem on each tick (every 10s).
    ///
    /// When the `delegation` feature is disabled this is a no-op.
    async fn run_delegation_tick(&self) {
        #[cfg(feature = "delegation")]
        self.run_delegation_tick_inner().await;
    }

    /// Inner implementation (only compiled with the `delegation` feature).
    #[cfg(feature = "delegation")]
    async fn run_delegation_tick_inner(&self) {
        use super::delegation::DelegationOutcome;
        use crate::provenance::{DerivationKind, ProvenanceRecord};

        // --- Phase 1: Drain pending queue and dispatch ---
        let pending = {
            let agent = self.agent.read().unwrap();
            agent.delegation_manager().drain_pending()
        };

        for task in pending {
            let hint = task.target_hint.clone();
            let agent = self.agent.read().unwrap();
            match agent.delegation_manager().dispatch(task, hint.as_deref()) {
                Ok(id) => {
                    tracing::info!(delegation_id = %id, "delegation dispatched");
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:delegation] dispatched delegation {id}",
                    )));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "daemon: failed to dispatch delegation");
                }
            }
        }

        // --- Phase 2: Process completed results ---
        let completed: Vec<DelegationOutcome> = {
            let agent = self.agent.read().unwrap();
            let mut results = Vec::new();
            while let Some(outcome) = agent.delegation_manager().try_recv() {
                results.push(outcome);
            }
            results
        };

        for outcome in completed {
            let target_label = outcome
                .target
                .as_ref()
                .map(|t| t.to_string())
                .unwrap_or_else(|| "unknown".to_string());

            if let Some(ref err) = outcome.error {
                tracing::warn!(
                    delegation_id = outcome.id.0,
                    error = %err,
                    "delegation failed"
                );
            } else {
                tracing::info!(
                    delegation_id = outcome.id.0,
                    target = %target_label,
                    wall_time_ms = outcome.wall_time.as_millis(),
                    "delegation completed"
                );
            }

            // Assert result triples into KG.
            if let Some(ref structured) = outcome.structured {
                let engine = self.agent.read().unwrap().engine.clone();
                let tuples: Vec<(String, String, String, f32)> = structured
                    .triples
                    .iter()
                    .map(|t| (t.subject.clone(), t.predicate.clone(), t.object.clone(), t.confidence))
                    .collect();
                if let Err(e) = engine.ingest_label_triples(&tuples) {
                    tracing::debug!(error = %e, "delegation: failed to assert triples");
                }
                // Store delegation provenance.
                // Use the delegation ID as a synthetic SymbolId (must be non-zero).
                // If delegation ID happens to be 0, resolve a placeholder entity.
                let sym_id = crate::symbol::SymbolId::new(outcome.id.0.max(1))
                    .unwrap_or_else(|| {
                        engine.resolve_or_create_entity("delegation:placeholder")
                            .unwrap_or_else(|_| {
                                // Last resort: use id=1 which is always valid.
                                crate::symbol::SymbolId::new(1).unwrap()
                            })
                    });
                let mut prov = ProvenanceRecord::new(
                    sym_id,
                    DerivationKind::Delegation {
                        target: target_label.clone(),
                        task_description: outcome.stdout.chars().take(200).collect(),
                        model_used: None,
                        tokens_consumed: outcome.tokens_consumed,
                        cost_usd: outcome.cost_usd,
                        wall_time_ms: outcome.wall_time.as_millis() as u64,
                    },
                );
                if let Ok(agent) = self.agent.read() {
                    if let Err(e) = agent.engine.store_provenance(&mut prov) {
                        tracing::debug!(error = %e, "delegation: failed to store provenance");
                    }
                    // Add summary to working memory if non-empty.
                    if !structured.summary.is_empty() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:delegation] {target_label}: {}", structured.summary
                        )));
                    }
                }
            }

            // Remove from active tracking.
            if let Ok(agent) = self.agent.read() {
                agent.delegation_manager().complete(outcome.id);
            }
        }

        // --- Phase 3: Check timeouts ---
        let timed_out = {
            let agent = self.agent.read().unwrap();
            agent.delegation_manager().check_timeouts()
        };
        for id in timed_out {
            tracing::warn!(delegation_id = %id, "delegation timed out");
            if let Ok(agent) = self.agent.read() {
                agent.delegation_manager().complete(id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Heuristically guess a domain label from a set of seed concept labels.
///
/// Uses the longest common word among all seeds, falling back to `None`.
fn guess_domain(seeds: &[String]) -> Option<String> {
    if seeds.is_empty() {
        return None;
    }
    // Collect all words from all seed labels.
    let all_words: Vec<&str> = seeds
        .iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    // Find the word that appears in the most seeds.
    let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in &all_words {
        if w.len() >= 4 {
            *freq.entry(w).or_insert(0) += 1;
        }
    }
    freq.into_iter()
        .max_by_key(|(_, count)| *count)
        .filter(|(_, count)| *count >= 2)
        .map(|(word, _)| word.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_reasonable_intervals() {
        let config = DaemonConfig::default();
        assert_eq!(config.equivalence_interval, Duration::from_secs(300));
        assert_eq!(config.reflection_interval, Duration::from_secs(180));
        assert_eq!(config.consolidation_interval, Duration::from_secs(60));
        assert_eq!(config.schema_discovery_interval, Duration::from_secs(1800));
        assert_eq!(config.rule_inference_interval, Duration::from_secs(600));
        assert_eq!(config.gap_analysis_interval, Duration::from_secs(900));
        assert_eq!(config.persist_interval, Duration::from_secs(60));
        assert_eq!(config.idle_cycle_interval, Duration::from_secs(30));
        assert_eq!(
            config.continuous_learning_interval,
            Duration::from_secs(7200)
        );
        assert_eq!(config.goal_generation_interval, Duration::from_secs(300));
        assert_eq!(config.sleep_cycle_interval, Duration::from_secs(3600));
        assert_eq!(config.trigger_evaluation_interval, Duration::from_secs(15));
        assert_eq!(
            config.knowledge_extraction_interval,
            Duration::from_secs(2700)
        );
        assert_eq!(
            config.openclaw_sync_interval,
            Duration::from_secs(86400)
        );
        assert_eq!(
            config.knowledge_elicitation_interval,
            Duration::from_secs(1800)
        );
        assert_eq!(
            config.relation_enrichment_interval,
            Duration::from_secs(3600)
        );
        assert_eq!(config.librarian_interval, Duration::from_secs(86400));
        assert_eq!(
            config.seshat_curation_interval,
            Duration::from_secs(21600)
        );
        assert_eq!(config.heka_curation_interval, Duration::from_secs(86400));
        assert_eq!(config.max_cycles, 0);
    }

    #[test]
    fn known_task_names_includes_curation_tasks() {
        // Regression guard for the typo-falls-back-to-master-switch trap
        // documented on `KNOWN_TASK_NAMES`: the two curation tasks added
        // alongside this test must be registered here.
        assert!(KNOWN_TASK_NAMES.contains(&"seshat_curation"));
        assert!(KNOWN_TASK_NAMES.contains(&"heka_curation"));
    }
}
