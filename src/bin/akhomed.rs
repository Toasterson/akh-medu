//! akhomed — the akh-medu daemon.
//!
//! Single authority over engine instances; ACP clients (the `akh` CLI/TUI,
//! Anima) connect here. Hosts N engine workspaces behind a single inbound
//! transport — the Agent Client Protocol (Unix socket + optional TCP). The
//! legacy surfaces (REST, WebSocket, web UI, inbound MCP) were retired in
//! ADR-061 (P8.4 deleted REST/WS/web-UI; P8.5 the inbound MCP server).
//!
//! Build and run: `cargo run --features server,acp --bin akhomed`
//!
//! See `docs/ai/decisions/061-acp-only-retire-legacy-surfaces.md`.


use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use miette::Diagnostic;
use thiserror::Error;
use tokio::sync::RwLock;

use akh_medu::agent::trigger::TriggerStore;
use akh_medu::agent::model_manager::AgentModelManager;
use akh_medu::agent::ollama::OllamaClient;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::akh_config::AkhConfig;
use akh_medu::client::DaemonStatus;
use akh_medu::config::AkhomedConfig;
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::paths::AkhPaths;
use akh_medu::schema::SchemaRegistry as _;
use akh_medu::seeds::SeedRegistry;
use akh_medu::vsa::Dimension;
use akh_medu::workspace::WorkspaceManager;

// ── Server state ──────────────────────────────────────────────────────────

struct ServerState {
    paths: AkhPaths,
    config: RwLock<AkhomedConfig>,
    workspaces: Arc<RwLock<HashMap<String, Arc<Engine>>>>,
    daemons: RwLock<HashMap<String, WorkspaceDaemon>>,
    /// One shared Agent per workspace — daemon, HTTP handlers, and WS sessions
    /// all operate on the same instance. See ADR-027.
    agents: Arc<RwLock<HashMap<String, Arc<Mutex<Agent>>>>>,
    /// Non-blocking psyche snapshots.  Populated at agent creation time so
    /// monitoring reads never block on the agent mutex.
    psyche_snapshots: PsycheSnapshotMap,
    /// Cached NLU pipeline per workspace — avoids reloading 1GB+ LLM model on
    /// every WS connection or MCP chat call.
    nlu_pipelines: RwLock<HashMap<String, Arc<Mutex<akh_medu::nlu::NluPipeline>>>>,
    /// Cached per-akh config (`akh.toml`) per workspace — model pin, explicit
    /// system prompt, related workspaces, autonomy, etc. (P1 / ADR-053).
    akh_configs: Arc<RwLock<HashMap<String, Arc<AkhConfig>>>>,
    /// Broadcast channel for audit entries — WS clients subscribe to this.
    audit_broadcast: tokio::sync::broadcast::Sender<akh_medu::audit::AuditEntry>,
    /// handles here; the daemon control-plane reads them for suspend/resume.
    daemon_pauses: RwLock<HashMap<String, DaemonPauseHandle>>,
    /// Phase 38: optional Postgres backing-store pool. `None` when no
    /// `[postgres]` section is configured (legacy local-only mode) **or** when
    /// the connection has not been established yet (boot-time race / PG down).
    /// Wrapped in a `RwLock` so the background reconnector (ADR 077) can swap a
    /// live pool in at runtime without restarting the daemon — recovering from
    /// a startup that fell back to local-only. Consumed by the mutation WAL
    /// flusher and snapshot writer (sub-phases 38b/38c).
    pg_pool: std::sync::RwLock<Option<Arc<akhomed_store::DatabaseConnection>>>,
    /// Phase 38b: per-workspace WAL flusher tasks, keyed by workspace name.
    /// Drained on graceful shutdown so the final batch reaches Postgres.
    wal_flushers: RwLock<HashMap<String, WalFlusher>>,
    /// Phase 38c: per-workspace periodic snapshot tasks, keyed by workspace name.
    /// Stopped on graceful shutdown, followed by a final synchronous snapshot.
    snapshot_tickers: RwLock<HashMap<String, SnapshotTicker>>,
    /// Phase 38c: in-flight `_akh/akh/wake` workspace names — single-flights wake
    /// so two concurrent wakes of the same cold workspace can't race on the redb.
    waking: Arc<Mutex<std::collections::HashSet<String>>>,
    /// Per-workspace load locks serialising the slow (first-load) path of
    /// `get_engine`: concurrent first-loads must not each build a separate Engine
    /// (each would wire its own WAL flusher + snapshot ticker and all but the
    /// last would leak as orphaned background tasks). Keyed by workspace name.
    engine_load_locks: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Likewise for `get_agent` — avoids a wasted concurrent heavy `Agent::resume`.
    agent_load_locks: RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// Fetch (creating if needed) the per-workspace load lock from `locks`.
///
/// Used to serialise `get_engine` / `get_agent` slow paths per workspace name so
/// each workspace is constructed at most once even under concurrent first-loads.
async fn fetch_load_lock(
    locks: &RwLock<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    name: &str,
) -> Arc<tokio::sync::Mutex<()>> {
    if let Some(lock) = locks.read().await.get(name) {
        return Arc::clone(lock);
    }
    Arc::clone(
        locks
            .write()
            .await
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
    )
}

/// RAII token marking a workspace as mid-wake; clears the name on drop so an
/// early `?`-return still releases the single-flight slot.
struct WakeGuard {
    set: Arc<Mutex<std::collections::HashSet<String>>>,
    name: String,
}

impl Drop for WakeGuard {
    fn drop(&mut self) {
        self.set
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.name);
    }
}

/// Handle to a workspace's background WAL flusher task (Phase 38b).
struct WalFlusher {
    shutdown: tokio::sync::watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
    /// The log itself, so sleep/relocate can force a synchronous flush of the
    /// buffer before evicting the engine (Phase 38e/g) instead of waiting for
    /// the next interval tick.
    log: Arc<akh_medu::wal::MutationLog>,
}

/// Handle to a workspace's background snapshot ticker task (Phase 38c).
struct SnapshotTicker {
    shutdown: tokio::sync::watch::Sender<bool>,
    handle: tokio::task::JoinHandle<()>,
}

/// Capture + persist one snapshot of `name`'s agent (Phase 38c).
///
/// The agent lives behind a `std::sync::Mutex`, so capture (sync) runs under the
/// lock and the guard is dropped before the async DB persist. A not-yet-loaded
/// agent is a quiet no-op (the ticker fires before the first interaction).
async fn snapshot_once(
    pool: &akhomed_store::DatabaseConnection,
    agents: &RwLock<HashMap<String, Arc<Mutex<Agent>>>>,
    name: &str,
    akh_id: akhomed_store::uuid::Uuid,
    workspace_id: akhomed_store::uuid::Uuid,
) -> Result<(), String> {
    let agent = {
        let map = agents.read().await;
        match map.get(name) {
            Some(a) => Arc::clone(a),
            None => return Ok(()),
        }
    };
    // The akh row is the identity these rows hang off; ensure it exists.
    akhomed_store::ensure_akh(pool, akh_id, name, None)
        .await
        .map_err(|e| e.to_string())?;
    // Capture under the lock (sync), release, then persist lock-free.
    let data = {
        let guard = agent.lock().unwrap_or_else(|e| e.into_inner());
        akh_medu::snapshot::capture_snapshot(&guard, akh_id, workspace_id)
            .map_err(|e| e.to_string())?
    };
    akh_medu::snapshot::persist_snapshot(data, pool)
        .await
        .map_err(|e| e.to_string())
}

/// Background daemon state for a workspace.
struct WorkspaceDaemon {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
}

/// Per-workspace psyche snapshot map.
///
/// Each value is an `Arc<RwLock<Option<Psyche>>>` extracted from the agent at
/// creation time so monitoring code can read psyche state without holding the
/// agent mutex. (Formerly shared with the inbound MCP server's `McpState`;
/// after ADR-061 P8.5 only `ServerState` uses it.)
type PsycheSnapshotMap = Arc<
    RwLock<
        HashMap<String, Arc<std::sync::RwLock<Option<akh_medu::compartment::psyche::Psyche>>>>,
    >,
>;

/// Pause control for a single workspace daemon. The daemon clones `paused` /
/// `cycle_done` into its `AgentDaemon`; the control-plane toggles `paused` and
/// waits on `cycle_done` to suspend/resume cleanly between OODA cycles.
#[derive(Clone)]
struct DaemonPauseHandle {
    paused: Arc<std::sync::atomic::AtomicBool>,
    cycle_done: Arc<tokio::sync::Notify>,
}

impl DaemonPauseHandle {
    fn new() -> Self {
        Self {
            paused: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            cycle_done: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

/// ADR 091 decision 5: pages a captured turn's raw body back out of the WAL
/// at recall time. `Engine::turn_body` is a synchronous seam (the recall path
/// stays sync/testable — see `src/mutation.rs::TurnBodyReader`); this is the
/// one real, Postgres-backed implementation, installed alongside the
/// `MutationSink` in `ServerState::wire_wal`.
///
/// Bridges the async DB hop with the runtime `Handle` captured at
/// construction time (`wire_wal` always runs inside the daemon's tokio
/// runtime, so `Handle::current()` there is infallible). `turn_body` may be
/// called either from a `block_in_place` context (the agentic turn's recall
/// path already runs under one — see `agentic_chat.rs`) or, conceivably, from
/// a plain sync context with no runtime entered at all. `Handle::try_current()`
/// distinguishes the two: if it succeeds we are on a thread that already has
/// a runtime context entered, so a bare `handle.block_on` would panic — wrap
/// it in `tokio::task::block_in_place` (safe to nest). If it fails, no
/// runtime is entered on this thread and `handle.block_on` is used directly.
struct PgTurnBodyReader {
    pool: Arc<akhomed_store::DatabaseConnection>,
    akh_id: akhomed_store::uuid::Uuid,
    handle: tokio::runtime::Handle,
}

impl akh_medu::mutation::TurnBodyReader for PgTurnBodyReader {
    fn turn_body(&self, seq: i64) -> Option<String> {
        let fut = akhomed_store::load_turn_by_seq(&self.pool, self.akh_id, seq);
        let result = if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| self.handle.block_on(fut))
        } else {
            self.handle.block_on(fut)
        };
        match result {
            Ok(Some(payload)) => {
                let user_prompt = payload
                    .get("user_prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let agent_answer = payload
                    .get("agent_answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Some(format!("User: {user_prompt}\nAkh: {agent_answer}"))
            }
            Ok(None) => None,
            Err(e) => {
                tracing::warn!(seq, error = %e, "turn body lookup failed; recall falls back to summary");
                None
            }
        }
    }
}

impl ServerState {
    fn new(
        paths: AkhPaths,
        config: AkhomedConfig,
        pg_pool: Option<Arc<akhomed_store::DatabaseConnection>>,
    ) -> Self {
        let (audit_tx, _) = tokio::sync::broadcast::channel(256);
        Self {
            paths,
            config: RwLock::new(config),
            workspaces: Arc::new(RwLock::new(HashMap::new())),
            daemons: RwLock::new(HashMap::new()),
            agents: Arc::new(RwLock::new(HashMap::new())),
            psyche_snapshots: Arc::new(RwLock::new(HashMap::new())),
            nlu_pipelines: RwLock::new(HashMap::new()),
            akh_configs: Arc::new(RwLock::new(HashMap::new())),
            audit_broadcast: audit_tx,
            daemon_pauses: RwLock::new(HashMap::new()),
            pg_pool: std::sync::RwLock::new(pg_pool),
            wal_flushers: RwLock::new(HashMap::new()),
            snapshot_tickers: RwLock::new(HashMap::new()),
            waking: Arc::new(Mutex::new(std::collections::HashSet::new())),
            engine_load_locks: RwLock::new(HashMap::new()),
            agent_load_locks: RwLock::new(HashMap::new()),
        }
    }

    /// Get or lazily load the [`AkhConfig`] (`akh.toml`) for a workspace.
    ///
    /// The akh config is the existing per-workspace config file parsed as an
    /// [`AkhConfig`]; a legacy workspace-only TOML loads with all akh-level
    /// fields defaulted, and a missing file yields a name-only default — so this
    /// never fails for a workspace that opens. A *malformed* config is **not**
    /// fatal: it logs a `tracing::warn!` (naming the file and parse error) and
    /// falls back to name-only defaults, mirroring [`Self::get_engine`]'s
    /// tolerance of a broken workspace TOML so a bad `akh.toml` can't keep the
    /// whole ACP server from starting.
    async fn get_akh_config(&self, name: &str) -> Result<Arc<AkhConfig>, String> {
        {
            let map = self.akh_configs.read().await;
            if let Some(cfg) = map.get(name) {
                return Ok(Arc::clone(cfg));
            }
        }

        let config_path = self.paths.workspace_config_file(name);
        let cfg = if config_path.exists() {
            match AkhConfig::load(&config_path) {
                Ok(cfg) => cfg,
                // A malformed akh.toml (e.g. a hand-written file missing the
                // required `name`) used to abort ACP startup. Match get_engine's
                // forgiveness: warn loudly so a real misconfiguration is still
                // visible, then fall back to name-only defaults. Note this
                // silently drops any akh-level settings the file *did* parse
                // (model pin, system prompt, endpoint) — fix the file to apply them.
                Err(e) => {
                    tracing::warn!(
                        workspace = %name,
                        config = %config_path.display(),
                        error = %e,
                        "akh.toml failed to parse; falling back to name-only akh \
                         defaults (model pin, system prompt, and endpoint settings \
                         are ignored until the file is fixed)"
                    );
                    AkhConfig::with_name(name)
                }
            }
        } else {
            tracing::debug!(
                workspace = %name,
                "no akh.toml found; using name-only akh defaults"
            );
            AkhConfig::with_name(name)
        };

        let cfg = Arc::new(cfg);
        let mut map = self.akh_configs.write().await;
        let entry = map.entry(name.to_string()).or_insert(Arc::clone(&cfg));
        Ok(Arc::clone(entry))
    }

    /// The Postgres backing-store pool, if connected. `None` in local-only mode
    /// (not configured, or the reconnector hasn't established a pool yet). Reads
    /// the swappable slot; a poisoned lock degrades to the inner value rather
    /// than panicking the data-plane.
    fn pg_pool(&self) -> Option<Arc<akhomed_store::DatabaseConnection>> {
        self.pg_pool
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// ADR 077: spawn the background Postgres reconnector. When startup fell
    /// back to local-only (boot-time network race, or PG transiently down),
    /// this task keeps retrying the connection with capped backoff. On success
    /// it swaps the live pool into `pg_pool` and re-enables durability for every
    /// already-loaded akh — snapshotting each one's in-memory state (everything
    /// accumulated while local-only) before wiring the WAL, so nothing is lost.
    /// No-op (returns immediately) when no `[postgres]` section is configured.
    /// Self-terminates once connected.
    fn spawn_pg_reconnector(self: &Arc<Self>) {
        let me = Arc::clone(self);
        tokio::spawn(async move {
            let mut backoff = Duration::from_secs(2);
            loop {
                // Already connected (startup or a concurrent path won)? Done.
                if me.pg_pool().is_some() {
                    return;
                }
                // Re-read config each attempt so a `config set` can correct a
                // bad URL without a restart. No `[postgres]` → nothing to do.
                let Some((opts, migrate)) = pg_pool_options(&*me.config.read().await) else {
                    return;
                };
                tokio::time::sleep(backoff).await;
                match try_connect_pg(&opts, migrate).await {
                    Ok(pool) => {
                        *me.pg_pool.write().unwrap_or_else(|e| e.into_inner()) = Some(pool);
                        tracing::info!(
                            "reconnected to PostgreSQL backing store — re-enabling durability"
                        );
                        me.rewire_after_reconnect().await;
                        return;
                    }
                    Err(e) => {
                        tracing::warn!(
                            backoff_secs = backoff.as_secs(),
                            error = %e,
                            "PostgreSQL reconnect attempt failed — will retry"
                        );
                        backoff = (backoff * 2).min(Duration::from_secs(60));
                    }
                }
            }
        });
    }

    /// After a runtime reconnect, re-enable the WAL + periodic snapshot for
    /// every already-loaded workspace and persist a full snapshot of each, so
    /// the state accumulated during local-only mode reaches Postgres. New
    /// workspaces loaded after this point wire normally via `get_engine`.
    async fn rewire_after_reconnect(&self) {
        // Capture names + engine handles under a short read lock, then release
        // it before doing per-workspace work (wire_wal / snapshot take locks).
        let loaded: Vec<(String, Arc<Engine>)> = {
            let map = self.workspaces.read().await;
            map.iter()
                .map(|(n, e)| (n.clone(), Arc::clone(e)))
                .collect()
        };
        if loaded.is_empty() {
            return;
        }
        for (name, engine) in loaded {
            // Wire the WAL first so mutations from here on are captured, then
            // snapshot the full current state (idempotent; materialises into the
            // relational tables what was accumulated while local-only).
            self.wire_wal(&name, &engine).await;
            match self.snapshot_workspace(&name).await {
                Ok(()) => tracing::info!(
                    workspace = %name,
                    "re-enabled durability + snapshotted after reconnect"
                ),
                Err(e) => tracing::warn!(
                    workspace = %name,
                    error = %e,
                    "post-reconnect snapshot failed; live mutations still flush via WAL"
                ),
            }
        }
    }

    /// Phase 38b: install the mutation WAL on `engine` for workspace `name`,
    /// when a Postgres backing store is configured. Best-effort: any failure
    /// logs and leaves the workspace in local-only mode (no WAL).
    async fn wire_wal(&self, name: &str, engine: &Engine) {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return;
        };
        let akh_id = akh_medu::wal::akh_id_for(name);
        let workspace_id = akh_medu::wal::workspace_id_for(name);

        // The akh row is the FK target for mutation_log; ensure it exists first.
        if let Err(e) = akhomed_store::ensure_akh(pool, akh_id, name, None).await {
            tracing::warn!(workspace = %name, error = %e, "could not ensure akh row; WAL disabled");
            return;
        }

        let (wal_cfg, snapshot_interval_secs) = {
            let cfg = self.config.read().await;
            match cfg.postgres.as_ref() {
                Some(pg) => (
                    akh_medu::wal::WalConfig::new(
                        Duration::from_secs(pg.wal_flush_interval_secs),
                        pg.wal_batch_size,
                        pg.wal_max_buffer_mb.saturating_mul(1024 * 1024),
                    ),
                    pg.snapshot_interval_secs,
                ),
                None => (akh_medu::wal::WalConfig::default(), 0),
            }
        };

        let log = Arc::new(akh_medu::wal::MutationLog::new(
            (**pool).clone(),
            akh_id,
            workspace_id,
            wal_cfg,
        ));
        if engine
            .set_mutation_sink(Arc::clone(&log) as Arc<dyn akh_medu::mutation::MutationSink>)
            .is_err()
        {
            tracing::warn!(workspace = %name, "mutation sink already installed; skipping WAL wiring");
            return;
        }

        // ADR 091 decision 5: install the WAL body-paging reader alongside the
        // sink so `recall_episodes` can page a harvested turn's raw text back
        // out of Postgres. Best-effort/independent of the sink install above —
        // a reader failing to install just means recall keeps using the
        // has_summary/label fallback (see `set_turn_body_reader`'s doc comment).
        let turn_body_reader = Arc::new(PgTurnBodyReader {
            pool: Arc::clone(pool),
            akh_id,
            handle: tokio::runtime::Handle::current(),
        });
        if engine
            .set_turn_body_reader(turn_body_reader as Arc<dyn akh_medu::mutation::TurnBodyReader>)
            .is_err()
        {
            tracing::warn!(workspace = %name, "turn body reader already installed; skipping");
        }

        let (shutdown, shutdown_rx) = tokio::sync::watch::channel(false);
        let handle = Arc::clone(&log).spawn_flusher(shutdown_rx);
        self.wal_flushers.write().await.insert(
            name.to_string(),
            WalFlusher {
                shutdown,
                handle,
                log: Arc::clone(&log),
            },
        );
        tracing::info!(workspace = %name, %akh_id, "mutation WAL enabled (Postgres backing store)");

        // Phase 38c: periodic snapshot of the akh's relational state. `0` means
        // no timer (a final snapshot on shutdown still runs).
        if snapshot_interval_secs > 0 {
            let ticker = self.spawn_snapshot_ticker(
                name.to_string(),
                Arc::clone(pool),
                Duration::from_secs(snapshot_interval_secs),
            );
            self.snapshot_tickers
                .write()
                .await
                .insert(name.to_string(), ticker);
            tracing::info!(
                workspace = %name,
                interval_secs = snapshot_interval_secs,
                "periodic snapshot enabled"
            );
        }
    }

    /// Spawn the background snapshot ticker for `name`. The task captures the
    /// agent state under its (sync) lock, then persists it lock-free; it skips
    /// quietly if the agent is not loaded yet.
    fn spawn_snapshot_ticker(
        &self,
        name: String,
        pool: Arc<akhomed_store::DatabaseConnection>,
        interval: Duration,
    ) -> SnapshotTicker {
        let agents = Arc::clone(&self.agents);
        let (shutdown, mut shutdown_rx) = tokio::sync::watch::channel(false);
        let handle = tokio::spawn(async move {
            let akh_id = akh_medu::wal::akh_id_for(&name);
            let workspace_id = akh_medu::wal::workspace_id_for(&name);
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            tick.tick().await; // consume the immediate first tick
            loop {
                tokio::select! {
                    _ = tick.tick() => {
                        if let Err(e) =
                            snapshot_once(&pool, &agents, &name, akh_id, workspace_id).await
                        {
                            tracing::warn!(workspace = %name, error = %e, "periodic snapshot failed");
                        }
                    }
                    _ = shutdown_rx.changed() => break,
                }
            }
        });
        SnapshotTicker { shutdown, handle }
    }

    /// Snapshot one workspace on demand (control-plane `_akh/akh/snapshot`).
    /// Loads the agent if needed, then captures + persists its state.
    async fn snapshot_workspace(&self, name: &str) -> Result<(), String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        // Ensure the agent is live so there is state to capture.
        self.get_agent(name).await?;
        let akh_id = akh_medu::wal::akh_id_for(name);
        let workspace_id = akh_medu::wal::workspace_id_for(name);
        snapshot_once(pool, &self.agents, name, akh_id, workspace_id).await
    }

    /// Wake a cold workspace from its Postgres snapshot (control-plane
    /// `_akh/akh/wake`). Populates the local redb, then routes through the normal
    /// lazy `get_agent` path so resume + model-manager + WAL wiring all apply.
    async fn wake_workspace(&self, name: &str) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        // Single-flight: refuse a second concurrent wake of the same workspace
        // (otherwise both would open the same redb and one fails confusingly).
        // The guard clears the slot on drop, including the `?` early returns below.
        let _wake_guard = {
            let mut waking = self.waking.lock().unwrap_or_else(|e| e.into_inner());
            if !waking.insert(name.to_string()) {
                return Err(format!("wake of \"{name}\" is already in progress"));
            }
            WakeGuard {
                set: Arc::clone(&self.waking),
                name: name.to_string(),
            }
        };

        // Wake operates on a *cold* workspace; a loaded one is already awake.
        if self.workspaces.read().await.contains_key(name)
            || self.agents.read().await.contains_key(name)
        {
            return Err(format!(
                "workspace \"{name}\" is already loaded; wake operates on a cold workspace"
            ));
        }

        let akh_id = akh_medu::wal::akh_id_for(name);
        let workspace_id = akh_medu::wal::workspace_id_for(name);

        // 38g split-brain guard: a cold akh may live on another host. Acquire its
        // Postgres advisory lock before touching state so two hosts can't wake the
        // same akh concurrently; refuse fast (non-zero CLI exit) if it's held.
        self.acquire_akh_lock_or_refuse(name, akh_id, "wake").await?;

        let ws_paths = self.paths.workspace(name);
        std::fs::create_dir_all(&ws_paths.kg_dir)
            .map_err(|e| format!("cannot create workspace kg dir: {e}"))?;

        // Reconstruct the local redb from Postgres. `wake` persists the KG
        // snapshot to redb before returning; dropping the engine releases the
        // redb lock so the lazy path can reopen it.
        let engine = akh_medu::snapshot::wake(&ws_paths.kg_dir, akh_id, workspace_id, pool)
            .await
            .map_err(|e| format!("wake failed: {e}"))?;
        let triples = engine.all_triples().len();
        let symbols = engine.all_symbols().len();
        drop(engine);

        // Bring the akh fully online via the standard path (resume reads the redb
        // wake just populated; this also wires the WAL + snapshot ticker).
        self.get_agent(name).await?;

        // Heartbeat: mark this host as the akh's live owner (best-effort).
        if let Err(e) =
            akhomed_store::touch_last_seen(pool, akh_id, akhomed_store::chrono::Utc::now()).await
        {
            tracing::warn!(workspace = %name, error = %e, "could not update last_seen after wake");
        }

        tracing::info!(workspace = %name, %akh_id, triples, symbols, "woke akh from Postgres");
        Ok(serde_json::json!({
            "woke": name,
            "triples": triples,
            "symbols": symbols,
        }))
    }

    /// 38g split-brain guard: try to take the Akh's Postgres advisory lock,
    /// erroring with an operator-facing message if another host holds it.
    ///
    /// `op` names the action (`"wake"` / `"relocate"`) for the error text. A
    /// transient DB error acquiring the lock is treated as a *refusal* (fail
    /// closed) rather than silently proceeding — split-brain is worse than a
    /// retry. No-op when there is no Postgres pool (local-only mode has a single
    /// host by construction, so there is nothing to guard against).
    async fn acquire_akh_lock_or_refuse(
        &self,
        name: &str,
        akh_id: akhomed_store::uuid::Uuid,
        op: &str,
    ) -> Result<(), String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Ok(());
        };
        match akhomed_store::try_acquire_akh_lock(pool, akh_id).await {
            Ok(true) => Ok(()),
            Ok(false) => Err(format!(
                "refusing to {op} \"{name}\": another host already holds the akh's \
                 advisory lock (split-brain guard). Confirm no other akhomed is \
                 serving this akh, then retry."
            )),
            Err(e) => Err(format!(
                "could not acquire the akh advisory lock for \"{name}\" ({e}); \
                 refusing to {op} to avoid split-brain. Check the Postgres backing store."
            )),
        }
    }

    /// Release the Akh's advisory lock (best-effort) — paired with
    /// [`Self::acquire_akh_lock_or_refuse`] so a later wake on this host can
    /// re-acquire. No-op without a pool. Logs (does not error) on failure.
    async fn release_akh_lock(&self, name: &str, akh_id: akhomed_store::uuid::Uuid) {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return;
        };
        if let Err(e) = akhomed_store::release_akh_lock(pool, akh_id).await {
            tracing::warn!(workspace = %name, error = %e, "could not release akh advisory lock");
        }
    }

    /// Evict a workspace's cached engine + agent and stop its background tasks
    /// (daemon, WAL flusher, snapshot ticker), freeing the akh's memory. Shared
    /// by `sleep_workspace`; the akh's durable state must already be persisted
    /// (snapshot + WAL flush) before calling this, or unflushed mutations are lost.
    async fn evict_workspace(&self, name: &str) {
        // Stop the OODA daemon first so no further mutations are produced.
        self.stop_workspace_daemon(name).await;

        // Stop + drain the snapshot ticker.
        if let Some(ticker) = self.snapshot_tickers.write().await.remove(name) {
            let _ = ticker.shutdown.send(true);
            if tokio::time::timeout(Duration::from_secs(5), ticker.handle)
                .await
                .is_err()
            {
                tracing::warn!(workspace = %name, "snapshot ticker did not stop within 5s on evict");
            }
        }

        // Stop + drain the WAL flusher (the final batch was already flushed by
        // the caller's snapshot path; this drains anything buffered since).
        if let Some(mut flusher) = self.wal_flushers.write().await.remove(name) {
            let _ = flusher.shutdown.send(true);
            match tokio::time::timeout(Duration::from_secs(5), &mut flusher.handle).await {
                Ok(_) => tracing::info!(workspace = %name, "WAL flusher drained on evict"),
                Err(_) => {
                    tracing::warn!(workspace = %name, "WAL flusher did not drain within 5s on evict, aborting");
                    flusher.handle.abort();
                }
            }
        }

        // Drop the cached agent + engine + ancillary caches so memory is freed.
        // Dropping the engine releases the redb lock so a later wake can reopen it.
        self.agents.write().await.remove(name);
        self.workspaces.write().await.remove(name);
        self.psyche_snapshots.write().await.remove(name);
        self.nlu_pipelines.write().await.remove(name);
        self.daemon_pauses.write().await.remove(name);

        // Release the advisory lock so this host (or another) can re-acquire on
        // the next wake (no-op in local-only mode).
        self.release_akh_lock(name, akh_medu::wal::akh_id_for(name))
            .await;
    }

    /// Deep-sleep a workspace (control-plane `_akh/akh/sleep`).
    ///
    /// Force a WAL flush + write a snapshot (so the akh is recoverable), then
    /// evict its cached engine + agent and stop its background tasks to free
    /// memory. Without a Postgres pool this still evicts (freeing RAM) but warns
    /// that durability requires Postgres — there is no relational copy to wake from.
    async fn sleep_workspace(&self, name: &str) -> Result<serde_json::Value, String> {
        if self.pg_pool().is_some() {
            // Snapshot first so the relational copy is current; snapshot_workspace
            // loads the agent if needed (so there is state to capture) and ensures
            // the akh row exists.
            self.snapshot_workspace(name).await?;
            // Flush the WAL synchronously so any mutation since the snapshot
            // reaches Postgres before we drop the in-memory engine.
            self.flush_wal(name).await;
            self.evict_workspace(name).await;
            tracing::info!(workspace = %name, "akh deep-slept (snapshot + evict)");
            Ok(serde_json::json!({
                "workspace": name,
                "slept": true,
                "level": "deep",
            }))
        } else {
            // Local-only: nothing to snapshot to, but freeing memory is still
            // useful. Persist the local redb best-effort, then evict.
            if let Ok(engine) = self.get_engine(name).await
                && let Err(e) = engine.persist()
            {
                tracing::warn!(workspace = %name, error = %e, "local persist before sleep failed");
            }
            self.evict_workspace(name).await;
            tracing::warn!(workspace = %name, "akh slept (local-only: evicted, no Postgres durability)");
            Ok(serde_json::json!({
                "workspace": name,
                "slept": true,
                "level": "deep",
                "note": "evicted from memory, but durable relocation needs a [postgres] backing store",
            }))
        }
    }

    /// Force a synchronous WAL flush for `name`, if a flusher is wired (Phase
    /// 38e/g). Used by sleep/relocate so any mutation since the snapshot reaches
    /// Postgres before the in-memory engine is dropped — without waiting for the
    /// next interval tick. A no-op when no flusher is wired (local-only mode).
    async fn flush_wal(&self, name: &str) {
        let log = self
            .wal_flushers
            .read()
            .await
            .get(name)
            .map(|f| Arc::clone(&f.log));
        if let Some(log) = log {
            // flush_once drains at most one batch; loop until the buffer is empty
            // so the relational copy is fully current before eviction/hand-off.
            let mut total = 0usize;
            loop {
                let flushed = log.flush_once().await;
                total += flushed;
                if flushed == 0 || log.pending() == 0 {
                    break;
                }
            }
            tracing::debug!(workspace = %name, rows = total, "forced WAL flush before sleep/relocate");
        }
    }

    /// Relocate a workspace (control-plane `_akh/akh/relocate`).
    ///
    /// Force a WAL flush + snapshot so another host can `akh akh wake` this akh,
    /// then update `last_seen`. Requires Postgres — relocation is meaningless
    /// without the shared backing store, so without a pool this returns a clear
    /// error telling the operator to configure `[postgres]`.
    async fn relocate_workspace(&self, name: &str) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err(
                "relocation requires a Postgres backing store (no [postgres] section configured). \
                 Add a [postgres] section to akhomed's config.toml (or set DATABASE_URL) so the \
                 akh's state can be woken on another host."
                    .into(),
            );
        };
        let akh_id = akh_medu::wal::akh_id_for(name);

        // Guard against another host already owning this akh.
        self.acquire_akh_lock_or_refuse(name, akh_id, "relocate")
            .await?;

        // Snapshot (loads the agent if needed) + flush so the relational copy is
        // current and complete before we hand off.
        self.snapshot_workspace(name).await?;
        self.flush_wal(name).await;

        // Mark the heartbeat so the recovery host sees a fresh `last_seen`.
        if let Err(e) =
            akhomed_store::touch_last_seen(pool, akh_id, akhomed_store::chrono::Utc::now()).await
        {
            tracing::warn!(workspace = %name, error = %e, "could not update last_seen on relocate");
        }

        // Release the lock so the recovery host can acquire it on wake.
        self.release_akh_lock(name, akh_id).await;

        tracing::info!(workspace = %name, %akh_id, "akh prepared for relocation");
        Ok(serde_json::json!({
            "workspace": name,
            "relocatable": true,
            "hint": "run `akh akh wake` on the recovery host",
        }))
    }

    /// Join: upsert a membership row for this workspace's akh (38f). The
    /// workspace name resolves to the stable workspace/akh ids; the akh row is
    /// ensured first so the membership FK target exists. Requires Postgres.
    async fn workspace_join(&self, name: &str) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err(membership_needs_pg("join"));
        };
        let akh_id = akh_medu::wal::akh_id_for(name);
        let workspace_id = akh_medu::wal::workspace_id_for(name);
        akhomed_store::ensure_akh(pool, akh_id, name, None)
            .await
            .map_err(|e| e.to_string())?;
        akhomed_store::join_workspace(
            pool,
            workspace_id,
            akh_id,
            "member",
            akhomed_store::chrono::Utc::now(),
        )
        .await
        .map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "workspace": name,
            "akh_id": akh_id.to_string(),
            "role": "member",
            "joined": true,
        }))
    }

    /// Leave: delete this workspace's akh membership row (38f). Requires Postgres.
    async fn workspace_leave(&self, name: &str) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err(membership_needs_pg("leave"));
        };
        let akh_id = akh_medu::wal::akh_id_for(name);
        let workspace_id = akh_medu::wal::workspace_id_for(name);
        let removed = akhomed_store::leave_workspace(pool, workspace_id, akh_id)
            .await
            .map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "workspace": name,
            "akh_id": akh_id.to_string(),
            "left": removed,
        }))
    }

    /// List members: all membership rows for a workspace as a JSON array of
    /// `{akh_id, role, joined_at}` (38f). Requires Postgres.
    async fn workspace_list_members(&self, name: &str) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err(membership_needs_pg("list-members"));
        };
        let workspace_id = akh_medu::wal::workspace_id_for(name);
        let rows = akhomed_store::list_members(pool, workspace_id)
            .await
            .map_err(|e| e.to_string())?;
        let members: Vec<serde_json::Value> = rows
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "akh_id": m.akh_id.to_string(),
                    "role": m.role,
                    "joined_at": m.joined_at.to_rfc3339(),
                })
            })
            .collect();
        Ok(serde_json::json!({
            "workspace": name,
            "members": members,
        }))
    }

    /// Run the akhomed-store migrations on demand (control-plane `_akh/migrate`).
    /// Exposes the same runner the daemon calls on start when `migrate_on_start`
    /// is set; requires Postgres.
    async fn run_migrations_op(&self) -> Result<serde_json::Value, String> {
        if self.pg_pool().is_none() {
            return Err(
                "migrations require a Postgres backing store (no [postgres] section configured). \
                 Add a [postgres] section to akhomed's config.toml (or set DATABASE_URL) and retry."
                    .into(),
            );
        }
        // Build a fresh PoolOptions from config so the migration runs under its
        // own advisory-locked single connection (safe even if other hosts are
        // migrating the same cluster concurrently), rather than on the shared
        // pool. Drop the config guard before the await.
        let opts = {
            let cfg = self.config.read().await;
            let pg = cfg.postgres.as_ref().ok_or_else(|| {
                "migrations require a Postgres backing store (no [postgres] section configured)."
                    .to_string()
            })?;
            let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| pg.url.clone());
            akhomed_store::PoolOptions {
                url,
                max_connections: pg.pool_size,
                connect_timeout: std::time::Duration::from_secs(pg.connect_timeout_secs),
            }
        };
        akhomed_store::run_migrations_locked(&opts)
            .await
            .map_err(|e| e.to_string())?;
        tracing::info!("akhomed-store migrations run via ACP control-plane");
        Ok(serde_json::json!({ "migrated": true }))
    }

    /// Admit a compiled schema pack to the Postgres registry (`_akh/schema/admit`,
    /// ADR 081 decision 9). The CLI already ran the human-confirm gate
    /// (`--yes`) and decoded the blob locally to build these params; this
    /// just persists the row via [`akh_medu::schema::registry_pg::PgSchemaRegistry`]
    /// (which re-checks duplicate-admission + the cross-pack type-collision
    /// guard against the live registry — never trust the client alone).
    async fn schema_admit(&self, params: &serde_json::Value) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        let get_str = |key: &str| -> Result<String, String> {
            params
                .get(key)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| format!("missing required param: {key}"))
        };
        let name = get_str("name")?;
        let version = get_str("version")?;
        let checksum = get_str("checksum")?;
        let confirmed_by = get_str("confirmed_by")?;
        let blob_hex = get_str("blob_hex")?;
        let scope = params
            .get("scope")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("shared");
        let source_url = params
            .get("source_url")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);

        let blob = hex_decode(&blob_hex)?;
        let row = akh_medu::schema::RegistryRow {
            name: name.clone(),
            version: version.clone(),
            scope: akh_medu::schema::registry_pg::scope_from_string(scope),
            source_url,
            checksum,
            confirmed_by,
            blob,
        };
        let registry = akh_medu::schema::registry_pg::PgSchemaRegistry::new(pool);
        registry
            .admit(row)
            .map_err(|e| format!("admit failed: {e}"))?;
        tracing::info!(pack = %format!("{name}@{version}"), "schema pack admitted to the registry");
        Ok(serde_json::json!({ "admitted": true, "name": name, "version": version }))
    }

    /// List schema packs admitted to the Postgres registry (`_akh/schema/list`).
    async fn schema_list(&self) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        let registry = akh_medu::schema::registry_pg::PgSchemaRegistry::new(pool);
        let packs = registry.list().map_err(|e| format!("list failed: {e}"))?;
        Ok(serde_json::json!({
            "packs": packs
                .into_iter()
                .map(|(name, version)| serde_json::json!({ "name": name, "version": version }))
                .collect::<Vec<_>>(),
        }))
    }

    /// Fetch a registry pack (by `name`, or `name`+`version`) and install +
    /// activate it into a workspace's effective schema (`_akh/schema/install`).
    /// Records the activation so a later wake converges it automatically
    /// ([`akh_medu::snapshot::wake`]).
    async fn schema_install(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        let ws = params
            .get("workspace")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing required param: workspace".to_string())?;
        let name = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing required param: name".to_string())?;
        let version = params.get("version").and_then(serde_json::Value::as_str);

        let row = match version {
            Some(v) => akhomed_store::get_pack(pool, name, v)
                .await
                .map_err(|e| e.to_string())?,
            None => akhomed_store::latest_pack(pool, name)
                .await
                .map_err(|e| e.to_string())?,
        };
        let Some(row) = row else {
            return Err(format!(
                "no admitted schema pack found for \"{name}{}\"",
                version.map(|v| format!("@{v}")).unwrap_or_default()
            ));
        };
        let registry_row = akh_medu::schema::RegistryRow {
            name: row.name,
            version: row.version,
            scope: akh_medu::schema::registry_pg::scope_from_string(&row.scope),
            source_url: row.source_url,
            checksum: row.checksum,
            confirmed_by: row.confirmed_by,
            blob: row.blob,
        };
        let pack = registry_row
            .decode()
            .map_err(|e| format!("decode failed: {e}"))?;

        let engine = self.get_engine(ws).await?;
        let report = akh_medu::schema::install_pack(&engine, &pack)
            .map_err(|e| format!("install failed: {e}"))?;

        let importer_label = akh_medu::schema::install::effective_schema_label(ws);
        let importer = engine
            .resolve_or_create_entity(&importer_label)
            .map_err(|e| format!("importer entity failed: {e}"))?;
        akh_medu::schema::activate_pack(&engine, importer, &pack)
            .map_err(|e| format!("activate failed: {e}"))?;

        let workspace_id = akh_medu::wal::workspace_id_for(ws);
        let akh_id = akh_medu::wal::akh_id_for(ws);
        akhomed_store::record_pack_activation(
            pool,
            workspace_id,
            &pack.microtheory_label(),
            akh_id,
            akhomed_store::chrono::Utc::now(),
        )
        .await
        .map_err(|e| format!("recording activation failed: {e}"))?;

        tracing::info!(
            workspace = %ws,
            pack = %format!("{}@{}", pack.name, pack.version),
            triples_applied = report.triples_applied,
            already_installed = report.already_installed,
            "schema pack installed and activated"
        );
        Ok(serde_json::json!({
            "installed": pack.name,
            "version": pack.version,
            "workspace": ws,
            "already_installed": report.already_installed,
            "triples_applied": report.triples_applied,
            "microtheory": pack.microtheory_label(),
        }))
    }

    /// Report the in-force schema for a workspace's engine — every declared
    /// type with its effective (inherited-included) property shape, and
    /// every declared relation (`_akh/schema/digest`, ADR 081 Slice 3
    /// work-item 4: the `get_graph_schema`-style digest). Does not require
    /// Postgres (unlike `schema/admit|list|install`) — it introspects
    /// whatever the workspace's already-loaded engine has installed.
    async fn schema_digest(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let ws = params
            .get("workspace")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing required param: workspace".to_string())?;
        let engine = self.get_engine(ws).await?;
        let digest = akh_medu::schema::digest(&engine).map_err(|e| format!("digest failed: {e}"))?;
        serde_json::to_value(&digest).map_err(|e| format!("digest serialization failed: {e}"))
    }

    /// Promote a pending frontier→core promotion proposal (`_akh/schema/promote`,
    /// ADR 081 Slice 5). The CLI already ran the human-confirm gate
    /// (`--yes`); this compiles+admits+installs+activates the promotion
    /// pack (or records an alias triple for an `alias_of` proposal) — see
    /// `akh_medu::agent::promotion::promote_proposal` for the full decision
    /// (new-subtype-vs-standalone-type fallback, one-pack-per-proposal
    /// naming). Needs both the workspace's engine (to read/apply the
    /// proposal) and Postgres (to admit the new pack), so — unlike
    /// `schema/proposals`/`schema/reject` — this is control-plane.
    async fn schema_promote(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let pool = self.pg_pool();
        let Some(pool) = pool.as_ref() else {
            return Err("no Postgres backing store configured (no [postgres] section)".into());
        };
        let ws = params
            .get("workspace")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing required param: workspace".to_string())?;
        let id = params
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "missing required param: id".to_string())?;
        let confirmed_by = params
            .get("confirmed_by")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");

        let engine = self.get_engine(ws).await?;
        let registry = akh_medu::schema::registry_pg::PgSchemaRegistry::new(pool);

        let importer_label = akh_medu::schema::install::effective_schema_label(ws);
        let importer = engine
            .resolve_or_create_entity(&importer_label)
            .map_err(|e| format!("importer entity failed: {e}"))?;

        let outcome = akh_medu::agent::promotion::promote_proposal(
            &engine,
            &registry,
            id,
            confirmed_by,
            importer,
        )
        .map_err(|e| format!("promotion failed: {e}"))?;

        // A `Promoted` outcome installed a new pack — record its activation
        // so a later wake converges it automatically (`schema_install`'s
        // exact precedent). An `Aliased` outcome installed no pack, so
        // there is nothing to record.
        if let akh_medu::agent::promotion::PromotionOutcome::Promoted {
            microtheory_label, ..
        } = &outcome
        {
            let workspace_id = akh_medu::wal::workspace_id_for(ws);
            let akh_id = akh_medu::wal::akh_id_for(ws);
            akhomed_store::record_pack_activation(
                pool,
                workspace_id,
                microtheory_label,
                akh_id,
                akhomed_store::chrono::Utc::now(),
            )
            .await
            .map_err(|e| format!("recording activation failed: {e}"))?;
        }

        tracing::info!(
            workspace = %ws,
            proposal = %id,
            "promotion proposal confirmed"
        );
        serde_json::to_value(&outcome).map_err(|e| format!("outcome serialization failed: {e}"))
    }

    /// Phase 38b/38c: stop background Postgres tasks on graceful shutdown.
    ///
    /// Order: stop snapshot tickers, take one final snapshot per loaded
    /// workspace (so the relational copy is current), then drain the WAL
    /// flushers (any mutations buffered after the final snapshot land newer than
    /// its watermark and are replayed idempotently on the next wake).
    async fn shutdown_wals(&self) {
        // Stop periodic snapshot tickers.
        let tickers: Vec<(String, SnapshotTicker)> =
            self.snapshot_tickers.write().await.drain().collect();
        for (name, ticker) in tickers {
            let _ = ticker.shutdown.send(true);
            if tokio::time::timeout(Duration::from_secs(5), ticker.handle)
                .await
                .is_err()
            {
                tracing::warn!(workspace = %name, "snapshot ticker did not stop within 5s");
            }
        }

        // Final snapshot of every loaded workspace (best-effort).
        if self.pg_pool().is_some() {
            let names: Vec<String> = self.agents.read().await.keys().cloned().collect();
            for name in names {
                match self.snapshot_workspace(&name).await {
                    Ok(()) => tracing::info!(workspace = %name, "final snapshot written on shutdown"),
                    Err(e) => {
                        tracing::warn!(workspace = %name, error = %e, "final snapshot on shutdown failed")
                    }
                }
            }
        }

        let flushers: Vec<(String, WalFlusher)> =
            self.wal_flushers.write().await.drain().collect();
        for (name, mut flusher) in flushers {
            let _ = flusher.shutdown.send(true);
            match tokio::time::timeout(Duration::from_secs(5), &mut flusher.handle).await {
                Ok(_) => tracing::info!(workspace = %name, "WAL flusher drained"),
                Err(_) => {
                    tracing::warn!(workspace = %name, "WAL flusher did not drain within 5s, aborting");
                    flusher.handle.abort();
                }
            }
        }
    }

    /// Get or lazily open an engine for the given workspace.
    async fn get_engine(&self, name: &str) -> Result<Arc<Engine>, String> {
        // Check if already loaded.
        {
            let map = self.workspaces.read().await;
            if let Some(engine) = map.get(name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Serialize the slow path per workspace: two concurrent first-loads must
        // not each build a separate Engine (each would wire its own WAL flusher +
        // snapshot ticker, and all but the winner would leak as orphaned tasks).
        let load_lock = fetch_load_lock(&self.engine_load_locks, name).await;
        let _load_guard = load_lock.lock().await;

        // Double-check: a racing winner may have populated the cache while we
        // waited for the load lock.
        {
            let map = self.workspaces.read().await;
            if let Some(engine) = map.get(name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Try to load it. A workspace with a config file but no data dir yet is
        // lazily created from that config (preserving akh-level fields), so any
        // *configured* akh loads on first use — control-plane (snapshot/sleep),
        // data-plane, and chat alike — not only the one bound at startup. Only a
        // workspace with neither data dir nor config is genuinely "not found".
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.root.exists() {
            let config_path = self.paths.workspace_config_file(name);
            if !config_path.exists() {
                return Err(format!("workspace \"{name}\" not found"));
            }
            let wsm = WorkspaceManager::new(self.paths.clone());
            let akh_cfg = AkhConfig::load(&config_path).ok();
            let ws_cfg = akh_cfg
                .as_ref()
                .map(|c| c.workspace.clone())
                .unwrap_or_else(|| akh_medu::workspace::WorkspaceConfig::with_name(name));
            wsm.create(ws_cfg)
                .map_err(|e| format!("failed to create workspace \"{name}\" from config: {e}"))?;
            // `create` persists only the WorkspaceConfig view; restore the full
            // AkhConfig so akh-level fields (backend/model/endpoint) survive.
            if let Some(c) = akh_cfg
                && let Err(e) = c.save(&config_path)
            {
                tracing::warn!(workspace = %name, error = %e, "failed to restore akh.toml after lazy create");
            }
        }

        // Honour the akh's read-root unlock (ADR 076 §4): when `openclaw_import`
        // is on, `effective_read_roots()` adds `~/.openclaw` so the agent's
        // `LocalFs` can read the akh's own OpenClaw past life (read-only; writes
        // still stay confined to `scratch/`). No config / legacy TOML → empty.
        let extra_read_roots = AkhConfig::load(&self.paths.workspace_config_file(name))
            .map(|c| c.effective_read_roots())
            .unwrap_or_default();

        let config = EngineConfig {
            dimension: Dimension::DEFAULT,
            data_dir: Some(ws_paths.kg_dir.clone()),
            compartments_dir: Some(ws_paths.compartments_dir.clone()),
            extra_read_roots,
            ..Default::default()
        };

        let engine = Engine::new(config)
            .map_err(|e| format!("failed to open workspace \"{name}\": {e}"))?;

        // Wire audit broadcast sender so WS clients receive live entries.
        if let Some(ledger) = engine.audit_ledger() {
            ledger.set_broadcast(self.audit_broadcast.clone());
        }

        // Apply seed packs listed in the workspace config (idempotent — seeds
        // track themselves via akh:seed-applied triples so re-opening a fully
        // seeded workspace is a no-op).
        {
            let config_path = self.paths.workspace_config_file(name);
            let ws_config = if config_path.exists() {
                akh_medu::workspace::WorkspaceConfig::load(&config_path)
                    .unwrap_or_else(|_| akh_medu::workspace::WorkspaceConfig::with_name(name))
            } else {
                akh_medu::workspace::WorkspaceConfig::with_name(name)
            };
            if !ws_config.seed_packs.is_empty() {
                let seeds_dir = self.paths.seeds_dir();
                let registry = SeedRegistry::discover(&seeds_dir);
                match registry.apply_all(&ws_config.seed_packs, &engine) {
                    Ok(reports) => {
                        let applied: usize = reports.iter().map(|r| r.triples_applied).sum();
                        if applied > 0 {
                            tracing::info!(
                                workspace = %name,
                                triples = applied,
                                "applied seed packs on first open"
                            );
                            if let Err(e) = engine.persist() {
                                tracing::warn!(workspace = %name, "persist after seeding failed: {e}");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(workspace = %name, "seed application failed: {e}");
                    }
                }
            }
        }

        // Phase 38b: install the mutation WAL (no-op in local-only mode).
        self.wire_wal(name, &engine).await;

        let engine = Arc::new(engine);
        let mut map = self.workspaces.write().await;
        map.insert(name.to_string(), Arc::clone(&engine));
        Ok(engine)
    }

    /// Get or lazily create a shared Agent for the given workspace.
    ///
    /// One Agent per workspace — daemon, HTTP handlers, and WS sessions all
    /// share the same instance. Construction is heavy (Agent::resume
    /// deserializes + rebuilds KG state), so we cache in `self.agents`.
    async fn get_agent(&self, name: &str) -> Result<Arc<Mutex<Agent>>, String> {
        // Fast path: already cached.
        {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(name) {
                return Ok(Arc::clone(agent));
            }
        }

        // Serialize the slow path per workspace so concurrent first-callers don't
        // each run the heavy `Agent::resume`; the loser reuses the winner's agent.
        // A distinct lock map from `get_engine`'s, since `get_engine` is called
        // below while this guard is held (same map would self-deadlock).
        let load_lock = fetch_load_lock(&self.agent_load_locks, name).await;
        let _load_guard = load_lock.lock().await;

        // Double-check after acquiring the load lock.
        {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(name) {
                return Ok(Arc::clone(agent));
            }
        }

        // Slow path: create agent (heavy sync work inside spawn_blocking).
        let engine = self.get_engine(name).await?;
        let mut agent = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            move || {
                let config = AgentConfig::default();
                if Agent::has_persisted_session(&engine) {
                    Agent::resume(engine, config)
                } else {
                    Agent::new(engine, config)
                }
            }
        })
        .await
        .map_err(|e| format!("agent task panicked: {e}"))?
        .map_err(|e| format!("failed to create agent: {e}"))?;

        tracing::info!(workspace = %name, "shared agent created");

        // Wire the per-akh model manager: pinned model + Ollama endpoint +
        // authoritative system prompt (P1 / ADR-053). Construction is cheap
        // (config clone + empty registry), and attaching it before the agent is
        // shared means the first prompt turn already routes to the akh's pinned
        // model with its own prompt. A missing/legacy akh.toml yields defaults,
        // reproducing today's behaviour.
        let akh_config = self.get_akh_config(name).await?;
        let manager = Arc::new(AgentModelManager::new(akh_config.to_model_manager_config()));
        agent.set_model_manager(manager);
        agent.register_with_model_manager(name, name);

        // ADR 076: inject the OpenClaw self-import surface for akhs that opted in
        // (`openclaw_import = true`). The read-root unlock is already applied via
        // `EngineConfig.extra_read_roots` in `get_engine`; here we add the
        // discovery tool, scoped to this akh's id (= the workspace name). Off by
        // default, so normal akhs never see it.
        if akh_config.openclaw_import {
            agent.register_tool(Box::new(
                akh_medu::agent::tools::OpenClawDiscoverTool::new(name),
            ));
            tracing::info!(workspace = %name, "OpenClaw self-import surface enabled");
        }

        // ADR 070: if this akh is Candle-backed, load a local GGUF model and
        // attach it as the chat/agentic backing (full, grammar-constrained tool
        // calling). When the model is absent we log and leave the Ollama path —
        // run_agentic_turn falls back to it, then to the symbolic pipeline.
        #[cfg(feature = "candle-backend")]
        if akh_config.backend == akh_medu::akh_config::ChatBacking::Candle {
            let model_dir = self.paths.models_dir();
            match akh_medu::nlu::candle_backend::CandleBackend::try_load(&model_dir) {
                Some(mut cb) => {
                    // Phase 47: share this akh's live inference options (Arc) with the backend and
                    // seed speculative decode from the persisted config. The `inference_config`
                    // tool flips the same Arc at runtime; the backend reads it each turn.
                    let opts = agent.engine().inference_options().clone();
                    opts.set_spec_decode(akh_config.spec_decode);
                    cb.set_inference_options(opts);
                    let system_prompt = akh_config.system_prompt.clone().unwrap_or_else(|| {
                        "You are an akh: a precise, curious neuro-symbolic assistant. \
                         Answer grounded in your knowledge graph; call tools when useful."
                            .to_string()
                    });
                    let backend = std::sync::Arc::new(akh_medu::agent::chat_backend::CandleChatBackend::new(
                        std::sync::Arc::new(std::sync::Mutex::new(cb)),
                        system_prompt,
                    ));
                    agent.set_chat_backend(backend);
                    tracing::info!(
                        workspace = %name, backend = "candle", spec_decode = akh_config.spec_decode,
                        "Candle chat backing attached"
                    );
                }
                None => {
                    tracing::warn!(
                        workspace = %name, model_dir = %model_dir.display(),
                        "akh.toml selects backend=candle but no GGUF model was found; \
                         falling back to Ollama/symbolic. Run `akh setup models`."
                    );
                }
            }
        }

        // Phase 44b/c/h/i (extended ADR 081 slice 7): assemble each turn through
        // a curating context manager, guarded by `SafetyEnforced` so eviction can
        // never drop a safety-pinned span. `AKH_CONTEXT_STRATEGY` selects the
        // strategy (default `compacting`):
        //   compacting — history + KG + recall + schema digest + active-note
        //                frontmatter within the model's real budget, with
        //                budget-driven section + tool degradation (44b/c/d/e/g);
        //   passive    — lean context; the longer thread is carried by background
        //                consolidation + recall, not verbatim history (44h);
        //   virtual    — MemGPT tiers (core/recall/archival) + the self-edit tools
        //                core_memory_append/replace, archival_search, query_graph
        //                (44i; the latter two extended in ADR 081 slice 7).
        // The lib default stays pass-through (tests / embedded).
        //
        // This `virtual`-strategy branch is also today's **tier-gating
        // mechanism** (ADR 081 slice 7 decision 4): small/append-only models run
        // `compacting`/`passive` and get no self-edit tools at all; only capable
        // models selected onto `virtual` get the self-edit + archival/query tools.
        {
            use akh_medu::agent::context::{
                CompactingManager, ContextManager, PassThroughManager, PassiveManager,
                SafetyEnforced, VirtualContextManager,
            };
            let strategy =
                std::env::var("AKH_CONTEXT_STRATEGY").unwrap_or_else(|_| "compacting".to_string());
            let manager: Box<dyn ContextManager> = match strategy.as_str() {
                // Byte-identical no-op (ADR 065): the "harness off" control for
                // A/B benchmarking — no history/KG/recall pre-loaded into the
                // prompt (the raw single-turn agentic loop). Not a production
                // default; explicit opt-in via AKH_CONTEXT_STRATEGY=passthrough.
                "passthrough" | "none" | "off" => Box::new(PassThroughManager),
                "passive" => Box::new(SafetyEnforced::new(PassiveManager::default())),
                "virtual" => {
                    // Register the self-edit memory tools with this akh's id baked
                    // in, so each tool always edits its own akh's core memory.
                    agent.register_tool(Box::new(
                        akh_medu::agent::tools::CoreMemoryAppendTool::new(name),
                    ));
                    agent.register_tool(Box::new(
                        akh_medu::agent::tools::CoreMemoryReplaceTool::new(name),
                    ));
                    // No persistent note store is wired to the daemon yet (`Note`
                    // shipped as a carrier type only — ADR 081 work-item P1), so
                    // archival_search stays KG-only here; `ArchivalSearchTool::with_notes`
                    // is ready for whichever slice adds a durable note catalog.
                    agent.register_tool(Box::new(akh_medu::agent::tools::ArchivalSearchTool::new()));
                    agent.register_tool(Box::new(akh_medu::agent::tools::QueryGraphTool));
                    Box::new(SafetyEnforced::new(VirtualContextManager::default()))
                }
                _ => Box::new(SafetyEnforced::new(CompactingManager)),
            };
            tracing::info!(workspace = %name, %strategy, "context manager attached");
            agent.set_context_manager(manager);
        }

        // ADR 091: per-akh transcript-capture opt-out (default on). The
        // `AKH_TRANSCRIPT_CAPTURE` env override is applied at the emit site
        // (`agentic_chat::Agent::transcript_capture_enabled`), not here — this
        // only wires the config-file default.
        agent.set_capture_transcripts(akh_config.capture_transcripts);

        // Extract the snapshot Arc before moving agent into Mutex so it can be
        // registered without ever re-acquiring the agent lock.
        let snapshot = agent.psyche_snapshot();
        let shared = Arc::new(Mutex::new(agent));
        let mut agents = self.agents.write().await;
        // Another request may have raced us — keep the first one.
        let entry = agents.entry(name.to_string()).or_insert(Arc::clone(&shared));
        let result = Arc::clone(entry);
        drop(agents);

        // Register snapshot (idempotent — or_insert keeps the race-winner's snapshot).
        self.psyche_snapshots
            .write()
            .await
            .entry(name.to_string())
            .or_insert(snapshot);

        Ok(result)
    }

    /// Get or lazily create a shared NLU pipeline for the given workspace.
    /// This avoids reloading the 1GB+ LLM model on every WS/MCP session.
    async fn get_nlu_pipeline(
        &self,
        name: &str,
        engine: &Engine,
    ) -> Arc<Mutex<akh_medu::nlu::NluPipeline>> {
        // Fast path: already cached.
        {
            let pipelines = self.nlu_pipelines.read().await;
            if let Some(pipeline) = pipelines.get(name) {
                return Arc::clone(pipeline);
            }
        }

        // Slow path: create pipeline (loads models from disk).
        let data_dir = engine.config().data_dir.clone();
        let ranker_bytes = engine
            .store()
            .get_meta(b"nlu_ranker_state")
            .ok()
            .flatten();

        let pipeline = tokio::task::spawn_blocking(move || {
            let data_dir_ref = data_dir.as_deref();
            ranker_bytes
                .and_then(|bytes| akh_medu::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
                .map(|ranker| {
                    akh_medu::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir_ref)
                })
                .unwrap_or_else(|| akh_medu::nlu::NluPipeline::new_with_models(data_dir_ref))
        })
        .await
        .unwrap_or_else(|_| akh_medu::nlu::NluPipeline::new());

        tracing::info!(workspace = %name, "shared NLU pipeline created");
        let shared = Arc::new(Mutex::new(pipeline));
        let mut pipelines = self.nlu_pipelines.write().await;
        let entry = pipelines
            .entry(name.to_string())
            .or_insert(Arc::clone(&shared));
        Arc::clone(entry)
    }

    // ── daemon lifecycle (shared by the REST handlers and the ACP control-plane)
    //
    // These hold the actual logic; the REST handlers and `ServerControl` impl
    // are thin adapters. When REST is deleted (ADR-061 increment 4) only the
    // adapters vanish — this stays.

    /// Start (or return the already-running) background daemon for a workspace.
    ///
    /// `max_cycles == 0` means run indefinitely. Background tasks are gated by
    /// the akh's `[autonomy]` config (P5): a fresh akh runs only always-on infra
    /// unless it opts in. Idempotent — a second call while one is live returns
    /// the existing status rather than spawning a duplicate.
    async fn spawn_workspace_daemon(
        &self,
        ws_name: &str,
        max_cycles: usize,
    ) -> Result<DaemonStatus, String> {
        // Already running? Return its current status.
        {
            let daemons = self.daemons.read().await;
            if let Some(d) = daemons.get(ws_name)
                && !d.handle.is_finished()
            {
                return Ok(d.status.lock().await.clone());
            }
        }

        let engine = self.get_engine(ws_name).await?;
        let shared_agent = self.get_agent(ws_name).await?;

        // Build DaemonConfig from global config with request-level override.
        let mut daemon_config = self.config.read().await.daemon.to_daemon_config();
        if max_cycles > 0 {
            daemon_config.max_cycles = max_cycles;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let trigger_count = TriggerStore::new(&engine).list().len();

        let status = Arc::new(tokio::sync::Mutex::new(DaemonStatus {
            running: true,
            total_cycles: 0,
            started_at: now,
            trigger_count,
            last_persist_at: None,
            last_learning_at: None,
            last_sleep_at: None,
            last_goal_gen_at: None,
            active_goals: 0,
            kg_symbols: 0,
            kg_triples: 0,
            knowledge_density: None,
            last_compaction_at: None,
        }));

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let daemon_status = Arc::clone(&status);
        let daemon_ws_name = ws_name.to_string();

        // Register pause handle so the control-plane can suspend/resume this daemon.
        let pause_handle = DaemonPauseHandle::new();
        self.daemon_pauses
            .write()
            .await
            .insert(ws_name.to_string(), pause_handle.clone());
        let task_handle = pause_handle.clone();

        // P5: gate background tasks by the akh's [autonomy] config.
        let autonomy = self
            .get_akh_config(ws_name)
            .await
            .map(|c| c.autonomy.clone())
            .unwrap_or_default();

        let pool = self.pg_pool();
        let handle = tokio::task::spawn(async move {
            run_daemon_task(
                shared_agent,
                daemon_status,
                shutdown_rx,
                daemon_config,
                autonomy,
                daemon_ws_name,
                task_handle,
                pool,
            )
            .await;
        });

        let daemon = WorkspaceDaemon {
            handle,
            shutdown_tx,
            status: Arc::clone(&status),
        };

        let st = status.lock().await.clone();
        self.daemons.write().await.insert(ws_name.to_string(), daemon);
        Ok(st)
    }

    /// Stop a workspace's background daemon. Returns `true` if one was running,
    /// `false` if there was nothing to stop.
    async fn stop_workspace_daemon(&self, ws_name: &str) -> bool {
        let mut daemons = self.daemons.write().await;
        if let Some(d) = daemons.remove(ws_name) {
            let _ = d.shutdown_tx.send(true);
            // Give the task a moment to finish.
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), d.handle).await;
            true
        } else {
            false
        }
    }

    /// Snapshot a workspace's daemon status, enriched with live KG stats. When
    /// no daemon is running, returns a `running: false` zero-status.
    async fn daemon_status_snapshot(&self, ws_name: &str) -> DaemonStatus {
        let daemons = self.daemons.read().await;
        if let Some(d) = daemons.get(ws_name) {
            let mut st = d.status.lock().await.clone();
            if let Ok(engine) = self.get_engine(ws_name).await {
                st.kg_symbols = engine.all_symbols().len();
                st.kg_triples = engine.all_triples().len();
            }
            st
        } else {
            DaemonStatus {
                running: false,
                total_cycles: 0,
                started_at: 0,
                trigger_count: 0,
                last_persist_at: None,
                last_learning_at: None,
                last_sleep_at: None,
                last_goal_gen_at: None,
                active_goals: 0,
                kg_symbols: 0,
                kg_triples: 0,
                knowledge_density: None,
                last_compaction_at: None,
            }
        }
    }

    /// The akhs this daemon currently owns + has awake, as JSON
    /// `[{ name, awake, owned_here, model }]`.
    ///
    /// The live in-memory agent set IS this host's owned+awake set: `wake` loads
    /// an akh's agent here and takes its Postgres advisory lock; `sleep`/
    /// `relocate` evict the agent and release the lock (so `last_seen` +
    /// `pg_try_advisory_lock` keep ownership single-writer, 38g). A dynamic akh
    /// gateway mirrors exactly this set to advertise routing — it must reflect
    /// what is *awake here*, not what is merely configured on disk. Side-effect
    /// free: reads caches only, never lazily loads a workspace. The pinned model
    /// comes from the cached `akh.toml` when known (`null` otherwise).
    async fn owned_awake_akhs(&self) -> Vec<serde_json::Value> {
        let names: Vec<String> = self.agents.read().await.keys().cloned().collect();
        let configs = self.akh_configs.read().await;
        let mut out: Vec<serde_json::Value> = names
            .into_iter()
            .map(|name| {
                let model = configs.get(&name).and_then(|c| c.model.clone());
                serde_json::json!({
                    "name": name,
                    "awake": true,
                    "owned_here": true,
                    "model": model,
                })
            })
            .collect();
        out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        out
    }
}

/// Operator-facing error for a membership op attempted without Postgres (38f).
/// `op` is the verb (`join` / `leave` / `list-members`).
fn membership_needs_pg(op: &str) -> String {
    format!(
        "workspace {op} requires a Postgres backing store (no [postgres] section configured). \
         Membership rows live in the shared store; add a [postgres] section to akhomed's \
         config.toml (or set DATABASE_URL) so multiple akhs can join the workspace."
    )
}

/// Decode a lowercase-hex string back into bytes — the counterpart of the
/// `akh` CLI's `hex_encode`, used to carry a schema-pack blob over the ACP
/// JSON wire (`_akh/schema/admit`'s `blob_hex` param; see ADR 081 decision 9).
fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd-length hex string ({} chars)", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| format!("invalid hex byte at offset {i}: {e}"))
        })
        .collect()
}

/// Background daemon task: delegates to [`AgentDaemon`] for the full set of
/// background learning tasks (goal generation, OODA cycles, equivalence
/// learning, reflection, consolidation, schema discovery, rule inference,
/// gap analysis, continuous learning, sleep cycles, and trigger evaluation).
async fn run_daemon_task(
    shared_agent: Arc<Mutex<Agent>>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    daemon_config: akh_medu::agent::DaemonConfig,
    autonomy: akh_medu::akh_config::AutonomyConfig,
    ws_name: String,
    pause_handle: DaemonPauseHandle,
    // ADR 091: Postgres pool for the sleep-cycle transcript-harvest phase.
    // `None` in local-only mode — harvest is then a no-op.
    pool: Option<Arc<akhomed_store::DatabaseConnection>>,
) {
    use akh_medu::agent::AgentDaemon;

    let mut daemon = AgentDaemon::new(shared_agent, daemon_config)
        .with_autonomy(autonomy)
        .with_shutdown(shutdown_rx)
        .with_status(Arc::clone(&status))
        .with_pause_handles(pause_handle.paused, pause_handle.cycle_done);
    if let Some(pool) = pool {
        let akh_id = akh_medu::wal::akh_id_for(&ws_name);
        daemon = daemon.with_db((*pool).clone(), akh_id);
    }

    if let Err(e) = daemon.run().await {
        tracing::error!(error = %e, ws = %ws_name, "daemon task failed");
    }

    // AgentDaemon::run() already sets status.running = false and persists,
    // but belt-and-suspenders in case of early return.
    status.lock().await.running = false;
    tracing::info!(ws = %ws_name, "daemon stopped");
}

// ── Trigger handlers ─────────────────────────────────────────────────────

/// akhomed's `ServerState` as the ACP **control-plane** backend (ADR-061
/// increment 1). Serves the `_akh/{workspace,daemon,config}/*` methods that
/// span the whole daemon — the ones with no home in a single-workspace
/// `Engine`. Wired into the ACP agent via `AkhAcpAgent::with_control`; the
/// per-workspace data-plane (`_akh/health`, `symbols/*`, …) is handled inside
/// the `acp` module instead.
///
/// Daemon ops take an explicit `workspace` param (the control-plane is flat,
/// not bound to one akh); `config/{get,set}` operate on the global
/// `AkhomedConfig`. Errors collapse to a string surfaced as a JSON-RPC error.
#[cfg(feature = "acp")]
impl akh_medu::acp::ServerControl for ServerState {
    /// Resolve a workspace's engine + agent + a fresh chat processor (built from
    /// the cached, Arc-shared NLU pipeline — no second model load). Lazily loads
    /// the workspace, so the ACP server can serve any configured akh per request.
    fn resolve<'a>(&'a self, workspace: &'a str) -> akh_medu::acp::ResolveFuture<'a> {
        Box::pin(async move {
            let engine = self.get_engine(workspace).await?;
            let agent = self.get_agent(workspace).await?;
            let nlu = self.get_nlu_pipeline(workspace, &engine).await;
            let processor = {
                let p = nlu.lock().unwrap_or_else(|e| e.into_inner());
                akh_medu::chat::ChatProcessor::new(&engine, p.clone())
            };
            Ok(akh_medu::acp::WorkspaceHandles {
                engine,
                agent,
                processor: std::sync::Arc::new(std::sync::Mutex::new(processor)),
            })
        })
    }

    fn dispatch<'a>(
        &'a self,
        method: &'a str,
        params: &'a serde_json::Value,
    ) -> akh_medu::acp::ControlFuture<'a> {
        Box::pin(async move {
            let op = method.strip_prefix("_akh/").unwrap_or(method);
            match op {
                // ── workspace group ──────────────────────────────────────
                "workspace/list" => Ok(serde_json::json!({
                    "workspaces": self.paths.list_workspaces(),
                })),
                "workspace/create" => {
                    let name = params
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: name".to_string())?;
                    let manager = WorkspaceManager::new(self.paths.clone());
                    let config = akh_medu::workspace::WorkspaceConfig {
                        name: name.to_string(),
                        ..Default::default()
                    };
                    manager
                        .create(config)
                        .map_err(|e| format!("create failed: {e}"))?;
                    // Optional role assignment (mirrors the REST create body).
                    if let Some(role) = params.get("role").and_then(serde_json::Value::as_str) {
                        let engine = self.get_engine(name).await?;
                        engine
                            .assign_role(role)
                            .map_err(|e| format!("assign_role failed: {e}"))?;
                        engine.persist().map_err(|e| format!("persist failed: {e}"))?;
                    }
                    Ok(serde_json::json!({ "name": name, "created": true }))
                }
                "workspace/delete" => {
                    let name = params
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: name".to_string())?;
                    // Drop the cached engine + agent before removing the data.
                    self.workspaces.write().await.remove(name);
                    self.agents.write().await.remove(name);
                    let manager = WorkspaceManager::new(self.paths.clone());
                    manager
                        .delete(name)
                        .map_err(|e| format!("delete failed: {e}"))?;
                    Ok(serde_json::json!({ "deleted": name }))
                }
                "workspace/info" => {
                    let name = params
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: name".to_string())?;
                    // Opens (lazily loads) the engine and reports live KG stats.
                    let engine = self.get_engine(name).await?;
                    Ok(serde_json::json!({
                        "name": name,
                        "kg_symbols": engine.all_symbols().len(),
                        "kg_triples": engine.all_triples().len(),
                        "role": engine.assigned_role(),
                        "data_dir": self.paths.workspace(name).root.display().to_string(),
                    }))
                }

                // ── daemon group (explicit `workspace` param) ────────────
                "daemon/start" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    let max_cycles = params
                        .get("max_cycles")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as usize;
                    let st = self
                        .spawn_workspace_daemon(ws, max_cycles)
                        .await?;
                    serde_json::to_value(st).map_err(|e| e.to_string())
                }
                "daemon/stop" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    let stopped = self.stop_workspace_daemon(ws).await;
                    Ok(serde_json::json!({ "stopped": stopped }))
                }
                "daemon/status" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    let st = self.daemon_status_snapshot(ws).await;
                    serde_json::to_value(st).map_err(|e| e.to_string())
                }
                // The akhs this daemon currently owns + has awake — the live set
                // an akh gateway mirrors to advertise routing (ADR-061: akhomed is
                // the only inbound surface; the gateway never reads akh-pg).
                "daemon/akhs" => Ok(serde_json::json!({
                    "akhs": self.owned_awake_akhs().await,
                })),

                // ── config group (global AkhomedConfig) ──────────────────
                "config/get" => {
                    let config = self.config.read().await;
                    let value = serde_json::to_value(&*config).map_err(|e| e.to_string())?;
                    Ok(serde_json::json!({ "config": value }))
                }
                // `config/set` merges by default: each top-level table present in
                // `config` (daemon/extraction/upstreams/acp/postgres) overwrites
                // that table in the currently persisted config; tables the caller
                // omits are left untouched. This is what stops a partial payload
                // (e.g. just `{"acp": {...}}`) from silently resetting unrelated
                // sections — postgres in particular — to their type defaults on
                // next start. Pass `"replace": true` to opt into the old
                // wholesale-replace behavior (the payload becomes the entire
                // config verbatim, missing tables reset to defaults).
                "config/set" => {
                    let cfg_val = params.get("config").ok_or_else(|| {
                        "missing required param: config (a full or partial AkhomedConfig \
                         object — top-level tables only, merged by default; pass \
                         \"replace\": true for wholesale replace)"
                            .to_string()
                    })?;
                    let replace = params
                        .get("replace")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false);

                    let new_config: AkhomedConfig = if replace {
                        serde_json::from_value(cfg_val.clone())
                            .map_err(|e| format!("invalid config payload: {e}"))?
                    } else {
                        let current = self.config.read().await.clone();
                        let mut merged = serde_json::to_value(&current)
                            .map_err(|e| format!("failed to snapshot current config: {e}"))?;
                        match (&mut merged, cfg_val) {
                            (
                                serde_json::Value::Object(merged_map),
                                serde_json::Value::Object(incoming_map),
                            ) => {
                                for (key, value) in incoming_map {
                                    merged_map.insert(key.clone(), value.clone());
                                }
                            }
                            _ => {
                                return Err(
                                    "config payload must be a JSON/TOML object of top-level \
                                     sections (daemon, extraction, upstreams, acp, postgres)"
                                        .to_string(),
                                );
                            }
                        }
                        serde_json::from_value(merged)
                            .map_err(|e| format!("invalid config payload after merge: {e}"))?
                    };

                    let config_path = self.paths.global_config_file();
                    new_config
                        .save(&config_path)
                        .map_err(|e| format!("save failed: {e}"))?;
                    *self.config.write().await = new_config.clone();
                    tracing::info!(
                        path = %config_path.display(),
                        replace,
                        "config updated via ACP control-plane"
                    );
                    let value = serde_json::to_value(&new_config).map_err(|e| e.to_string())?;
                    Ok(serde_json::json!({ "config": value, "replace": replace }))
                }

                // ── model group (per-akh model + reasoning effort) ────────
                //
                // `model/list` builds a transient OllamaClient from the akh's
                // config and queries `/api/tags`; a cloud endpoint that lists
                // nothing (or errors) still returns the current model/effort so
                // the TUI picker can open. `model/set` persists akh.toml AND
                // rebuilds the running model manager so the next chat turn uses
                // the new model/effort.
                "model/list" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("default");
                    let cfg = self.get_akh_config(ws).await?;
                    let mm = cfg.to_model_manager_config();
                    let current = mm.ollama.model.clone();
                    let effort = mm.ollama.reasoning_effort.clone();
                    let client = OllamaClient::new(mm.ollama);
                    match client.list_models() {
                        Ok(models) => Ok(serde_json::json!({
                            "current": current,
                            "effort": effort,
                            "models": models,
                        })),
                        // Cloud endpoints frequently 404 `/api/tags` or are
                        // overloaded — don't fail the whole call, just report no
                        // models + the error so the picker still opens.
                        Err(e) => Ok(serde_json::json!({
                            "current": current,
                            "effort": effort,
                            "models": Vec::<String>::new(),
                            "error": e.to_string(),
                        })),
                    }
                }
                "model/set" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("default")
                        .to_string();
                    let model = params
                        .get("model")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                    let effort_raw = params
                        .get("effort")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                    if model.is_none() && effort_raw.is_none() {
                        return Err(
                            "model/set requires at least one of: model, effort".to_string()
                        );
                    }

                    // (a) Load the akh config (fall back to name-only defaults
                    //     so a missing/broken akh.toml still applies the change).
                    let config_path = self.paths.workspace_config_file(&ws);
                    let mut cfg = AkhConfig::load(&config_path)
                        .unwrap_or_else(|_| AkhConfig::with_name(&ws));

                    // (b) Apply the requested fields.
                    if let Some(m) = &model {
                        cfg.model = Some(m.clone());
                    }
                    if let Some(level) = &effort_raw {
                        match level.trim().to_ascii_lowercase().as_str() {
                            "low" | "medium" | "high" => {
                                cfg.reasoning_effort = Some(level.trim().to_ascii_lowercase())
                            }
                            "off" | "none" => cfg.reasoning_effort = None,
                            other => {
                                return Err(format!(
                                    "unknown reasoning effort \"{other}\" \
                                     (expected low|medium|high|off)"
                                ));
                            }
                        }
                    }

                    // (c) Persist akh.toml.
                    cfg.save(&config_path)
                        .map_err(|e| format!("save akh.toml failed: {e}"))?;

                    let new_model = cfg.model.clone();
                    let new_effort = cfg.reasoning_effort.clone();
                    let mm_config = cfg.to_model_manager_config();

                    // (d) Rebuild the running model manager so the next chat turn
                    //     uses the new model/effort. ADR-027: the agent
                    //     std::sync::Mutex is locked only synchronously (no await
                    //     held across the lock).
                    let agent = self.get_agent(&ws).await?;
                    {
                        let mut a = agent.lock().unwrap_or_else(|e| e.into_inner());
                        a.set_model_manager(Arc::new(AgentModelManager::new(mm_config)));
                        a.register_with_model_manager(&ws, &ws);
                    }

                    // Refresh the cached akh.toml so subsequent reads (and the
                    // next get_agent rebuild) see the persisted change.
                    self.akh_configs
                        .write()
                        .await
                        .insert(ws.clone(), Arc::new(cfg));

                    tracing::info!(
                        workspace = %ws,
                        model = ?new_model,
                        effort = ?new_effort,
                        "model/effort updated via ACP control-plane"
                    );
                    Ok(serde_json::json!({
                        "model": new_model,
                        "effort": new_effort,
                    }))
                }

                // ── akh group (relocation: snapshot/wake an akh) ─────────
                "akh/snapshot" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    self.snapshot_workspace(ws).await?;
                    Ok(serde_json::json!({ "snapshot": ws, "ok": true }))
                }
                "akh/wake" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    self.wake_workspace(ws).await
                }
                "akh/sleep" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    self.sleep_workspace(ws).await
                }
                "akh/relocate" => {
                    let ws = params
                        .get("workspace")
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: workspace".to_string())?;
                    self.relocate_workspace(ws).await
                }

                // ── workspace membership group (38f, Postgres-backed) ─────
                "workspace/join" => {
                    let ws = params
                        .get("id")
                        .or_else(|| params.get("workspace"))
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: id".to_string())?;
                    self.workspace_join(ws).await
                }
                "workspace/leave" => {
                    let ws = params
                        .get("id")
                        .or_else(|| params.get("workspace"))
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: id".to_string())?;
                    self.workspace_leave(ws).await
                }
                "workspace/list-members" => {
                    let ws = params
                        .get("id")
                        .or_else(|| params.get("workspace"))
                        .and_then(serde_json::Value::as_str)
                        .ok_or_else(|| "missing required param: id".to_string())?;
                    self.workspace_list_members(ws).await
                }

                // ── projection group (38d AGE read-model: deferred stub) ──
                "projection/refresh" => Ok(serde_json::json!({
                    "refreshed": false,
                    "note": "AGE projection deferred to Phase 38d",
                })),

                // ── migrate (run akhomed-store migrations on demand) ──────
                "migrate" => self.run_migrations_op().await,

                // ── schema group (ADR 081 decision 9: Postgres registry) ──
                "schema/admit" => self.schema_admit(params).await,
                "schema/list" => self.schema_list().await,
                "schema/install" => self.schema_install(params).await,
                "schema/digest" => self.schema_digest(params).await,
                "schema/promote" => self.schema_promote(params).await,

                other => Err(format!("unknown control-plane method: _akh/{other}")),
            }
        })
    }
}

/// Fatal errors that abort daemon startup before the async runtime settles
/// into normal operation. akhomed has no supervisor above it besides
/// launchd/systemd restart policies, so these are rendered as full miette
/// diagnostics to stderr (captured in the service log) and exit the process —
/// there is nothing left to fall back to.
#[derive(Debug, Error, Diagnostic)]
enum StartupError {
    /// `AKH_SERVER_PORT` didn't parse. This used to `.expect()`-panic, which
    /// under launchd's `KeepAlive` reads as a silent crash-loop with the real
    /// cause buried in a raw Rust panic message in the stderr log.
    #[error("AKH_SERVER_PORT={value:?} is not a valid TCP port")]
    #[diagnostic(
        code(akh::daemon::invalid_port),
        help(
            "AKH_SERVER_PORT must be an integer in 0..=65535 (e.g. `AKH_SERVER_PORT=8200`). \
             Unset the variable to use the default port 8200."
        )
    )]
    InvalidPort {
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    /// Single-instance guard: another akhomed is already accepting
    /// connections on the ACP Unix socket this instance would bind.
    #[error("another akhomed is already serving ACP on {socket}")]
    #[diagnostic(
        code(akh::daemon::already_running),
        help(
            "A live daemon ({owner}) is still accepting connections on this socket, so \
             it was not reclaimed. Stop it first (`akh service stop`, or `kill <pid>`), \
             or run this instance against a different socket via `acp.socket_path` in \
             config.toml / `AKH_ACP_SOCKET`."
        )
    )]
    AlreadyRunning { socket: String, owner: String },
}

/// Best-effort description of who owns a live socket, for the
/// [`StartupError::AlreadyRunning`] diagnostic. Reads the discovery PID file
/// (written by [`akh_medu::client::write_pid_file`] at startup) and reports
/// the PID it recorded; the socket connect already proved *something* is
/// listening; this is purely to make the message actionable.
fn describe_socket_owner(paths: &AkhPaths) -> String {
    std::fs::read_to_string(paths.pid_file())
        .ok()
        .and_then(|contents| serde_json::from_str::<akh_medu::client::ServerInfo>(&contents).ok())
        .map(|info| format!("pid {}", info.pid))
        .unwrap_or_else(|| "pid unknown — no readable PID file".to_string())
}

/// True if the discovery PID file currently names *this* process as the
/// owner. Consulted at shutdown before unlinking the shared socket/PID
/// artifacts: if a second `akhomed` raced in (started while this one was
/// still up, or during our own drain), it will have overwritten the PID file
/// with its own `std::process::id()` on startup, and this instance must not
/// then rip the artifacts out from under it on the way down. A missing or
/// unparseable PID file is treated as "not ours" — nothing to reclaim.
fn owns_pid_file(paths: &AkhPaths) -> bool {
    std::fs::read_to_string(paths.pid_file())
        .ok()
        .and_then(|contents| serde_json::from_str::<akh_medu::client::ServerInfo>(&contents).ok())
        .is_some_and(|info| info.pid == std::process::id())
}

/// Bring up the ACP transport (Unix socket + optional TCP) for one workspace,
/// serving alongside the REST/WS server during the migration to ACP-only.
///
/// Returns `(unix_socket_path, tcp_addr, accept_loop_handles)` — the path/addr
/// strings (present only for transports that bound successfully) go into the
/// discovery record; the handles are joined on shutdown. A failure to open the
/// workspace or bind a listener disables that transport and is logged, never
/// fatal — the daemon still serves REST.
///
/// Exception: if the Unix socket already has a live listener (a sibling
/// akhomed still running), this refuses to steal it — see the single-instance
/// guard note inline — and exits the whole process rather than limping along
/// with a half-started daemon that shares no transport with `akh` clients.
#[cfg(feature = "acp")]
async fn start_acp_server(
    state: Arc<ServerState>,
    acp_cfg: akh_medu::config::AcpSection,
    paths: &AkhPaths,
    fallback_ws: &str,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> (Option<String>, Option<String>, Vec<tokio::task::JoinHandle<()>>) {
    use akh_medu::acp::{serve_tcp_loop, serve_unix_loop, AkhAcpAgent};
    use std::path::PathBuf;

    let ws = acp_cfg
        .workspace
        .clone()
        .unwrap_or_else(|| fallback_ws.to_string());

    // Resolve the akh's shared engine + agent (ADR-027) and build a chat
    // processor from a cheap clone of the cached NLU pipeline (Arc-shared
    // models — no second model load).
    // Warm + validate the bound workspace so a misconfigured akh fails fast and
    // first-chat latency is low. We don't *hold* the engine here: the ACP agent
    // runs in provider mode and resolves each request's workspace on demand from
    // `ServerState`, so it pins no engine (a permanent hold would block a
    // same-host `sleep`/`wake`) and serves any configured akh per request (`-w`).
    if let Err(msg) = state.get_engine(&ws).await {
        tracing::warn!(workspace = %ws, error = %msg, "ACP disabled: cannot open workspace");
        return (None, None, Vec::new());
    }
    // Wire the server-level control-plane + per-request workspace resolution:
    // `ServerState` answers the `_akh/{workspace,daemon,config}/*` methods and
    // resolves engine/agent/processor for any workspace by name.
    let control: Arc<dyn akh_medu::acp::ServerControl> = state.clone();
    let acp_agent = AkhAcpAgent::with_provider(control, &ws);

    let mut handles = Vec::new();
    let mut socket_str = None;
    let mut tcp_str = None;

    if acp_cfg.unix {
        let sock = acp_cfg
            .socket_path
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| paths.runtime_dir.join("akhomed-acp.sock"));

        // Single-instance guard: a socket file on disk proves nothing (a
        // prior crash or unclean shutdown leaves one behind), so probe for a
        // live listener before reclaiming it. Only bind-stealing an *actually
        // serving* sibling daemon is fatal — that would silently split the
        // workspace across two processes each thinking they own its
        // redb/PG locks.
        if akh_medu::client::unix_socket_has_listener(&sock) {
            let err = StartupError::AlreadyRunning {
                socket: sock.display().to_string(),
                owner: describe_socket_owner(paths),
            };
            eprintln!("{:?}", miette::Report::new(err));
            std::process::exit(1);
        }
        // Clear a stale (dead-listener, or never-bound) socket so bind()
        // doesn't fail with EADDRINUSE.
        let _ = std::fs::remove_file(&sock);
        match tokio::net::UnixListener::bind(&sock) {
            Ok(listener) => {
                tracing::info!(socket = %sock.display(), workspace = %ws, "ACP serving on Unix socket");
                socket_str = Some(sock.display().to_string());
                let a = acp_agent.clone();
                let rx = shutdown_rx.clone();
                handles.push(tokio::spawn(async move {
                    if let Err(e) = serve_unix_loop(a, listener, rx).await {
                        tracing::error!("ACP unix accept loop ended: {e}");
                    }
                }));
            }
            Err(e) => {
                tracing::error!(path = %sock.display(), error = %e, "ACP unix bind failed");
            }
        }
    }

    if let Some(addr) = acp_cfg.tcp_addr() {
        match tokio::net::TcpListener::bind(&addr).await {
            Ok(listener) => {
                tracing::info!(addr = %addr, workspace = %ws, "ACP serving on TCP");
                tcp_str = Some(addr);
                let a = acp_agent.clone();
                let rx = shutdown_rx.clone();
                handles.push(tokio::spawn(async move {
                    if let Err(e) = serve_tcp_loop(a, listener, rx).await {
                        tracing::error!("ACP tcp accept loop ended: {e}");
                    }
                }));
            }
            Err(e) => {
                tracing::error!(addr = %addr, error = %e, "ACP tcp bind failed");
            }
        }
    }

    (socket_str, tcp_str, handles)
}

/// Phase 38: open the Postgres backing-store pool if `[postgres]` is configured.
///
/// Returns `None` (and logs a clear warning) when no `[postgres]` section is
/// present, so the daemon degrades to legacy local-only mode. A connection or
/// migration failure is also non-fatal: it logs an error and falls back to
/// local-only rather than refusing to boot.
/// Build the connection `PoolOptions` from the `[postgres]` config, honouring
/// the `DATABASE_URL` override (K8s secret injection). Returns `None` when no
/// `[postgres]` section is present (legacy local-only mode).
fn pg_pool_options(config: &AkhomedConfig) -> Option<(akhomed_store::PoolOptions, bool)> {
    let pg = config.postgres.as_ref()?;
    let url = std::env::var("DATABASE_URL").unwrap_or_else(|_| pg.url.clone());
    Some((
        akhomed_store::PoolOptions {
            url,
            max_connections: pg.pool_size,
            connect_timeout: std::time::Duration::from_secs(pg.connect_timeout_secs),
        },
        pg.migrate_on_start,
    ))
}

/// One connect attempt + optional advisory-locked migration. Returns the live
/// pool on success, or a human-readable error describing what failed.
async fn try_connect_pg(
    opts: &akhomed_store::PoolOptions,
    migrate: bool,
) -> Result<Arc<akhomed_store::DatabaseConnection>, String> {
    let db = akhomed_store::connect(opts)
        .await
        .map_err(|e| format!("connect: {e}"))?;
    if migrate {
        // Advisory-locked so several hosts starting against one fresh cluster
        // database don't race on creating `seaql_migrations`.
        akhomed_store::run_migrations_locked(opts)
            .await
            .map_err(|e| format!("migrations: {e}"))?;
    }
    Ok(Arc::new(db))
}

/// Number of quick startup connect attempts before falling back to local-only
/// (the background reconnector then keeps trying). Sized to ride out the common
/// boot-time race where launchd starts akhomed before the LAN route is up.
const PG_STARTUP_ATTEMPTS: u32 = 5;

/// Open the Postgres backing store at startup, retrying briefly to absorb a
/// boot-time network race. On exhaustion returns `None` (local-only); the
/// caller is expected to spawn [`ServerState::spawn_pg_reconnector`] so the
/// daemon recovers without a restart (ADR 077).
async fn init_postgres(
    config: &AkhomedConfig,
) -> Option<Arc<akhomed_store::DatabaseConnection>> {
    let Some((opts, migrate)) = pg_pool_options(config) else {
        tracing::warn!(
            "[postgres] not configured — running in legacy local-only mode \
             (no WAL, no cross-host relocation; Phase 38)"
        );
        return None;
    };

    let mut backoff = Duration::from_secs(1);
    for attempt in 1..=PG_STARTUP_ATTEMPTS {
        match try_connect_pg(&opts, migrate).await {
            Ok(pool) => {
                tracing::info!("connected to PostgreSQL backing store");
                return Some(pool);
            }
            Err(e) => {
                if attempt < PG_STARTUP_ATTEMPTS {
                    tracing::warn!(
                        attempt,
                        max = PG_STARTUP_ATTEMPTS,
                        backoff_secs = backoff.as_secs(),
                        error = %e,
                        "PostgreSQL not reachable yet — retrying"
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(8));
                } else {
                    tracing::error!(
                        error = %e,
                        "PostgreSQL unavailable after {PG_STARTUP_ATTEMPTS} attempts — \
                         falling back to local-only mode; background reconnector will keep trying"
                    );
                }
            }
        }
    }
    None
}

/// Raise the soft `RLIMIT_NOFILE` to the hard limit. Container runtimes often
/// leave the soft limit at 1024 (docker's default) while the hard limit is in
/// the hundreds of thousands; a long-lived daemon accepting many short TCP
/// connections (the gateway's ACP discovery polls) exhausts 1024 in days.
#[cfg(unix)]
fn raise_nofile_limit() {
    // SAFETY: plain libc rlimit syscalls on a zeroed struct; no aliasing.
    unsafe {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) != 0 {
            return;
        }
        if lim.rlim_cur >= lim.rlim_max {
            return;
        }
        let soft_before = lim.rlim_cur;
        lim.rlim_cur = lim.rlim_max;
        // macOS refuses a soft limit above OPEN_MAX (10240) even when the hard
        // limit reports RLIM_INFINITY (armini runs akhomed natively). libc
        // doesn't export OPEN_MAX, so use its (ABI-stable) value.
        #[cfg(target_os = "macos")]
        {
            lim.rlim_cur = lim.rlim_cur.min(10240);
        }
        if lim.rlim_cur <= soft_before {
            return;
        }
        if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) == 0 {
            tracing::info!(from = soft_before, to = lim.rlim_cur, "raised RLIMIT_NOFILE soft limit");
        } else {
            tracing::warn!(
                soft = soft_before,
                hard = lim.rlim_max,
                "could not raise RLIMIT_NOFILE soft limit"
            );
        }
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();

    #[cfg(unix)]
    raise_nofile_limit();

    // `bind`/`port` no longer bind an HTTP listener (ADR-061 P8.5) — they are
    // retained only for the discovery PID record consumed by `akh` clients.
    let bind = std::env::var("AKH_SERVER_BIND").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("AKH_SERVER_PORT").unwrap_or_else(|_| "8200".to_string());

    let paths = AkhPaths::resolve().unwrap_or_else(|e| {
        tracing::error!("failed to resolve XDG paths: {e}");
        std::process::exit(1);
    });
    if let Err(e) = paths.ensure_dirs() {
        tracing::error!("failed to create XDG directories: {e}");
        std::process::exit(1);
    }

    let port_num: u16 = match port.parse() {
        Ok(p) => p,
        Err(source) => {
            let err = StartupError::InvalidPort {
                value: port.clone(),
                source,
            };
            eprintln!("{:?}", miette::Report::new(err));
            std::process::exit(1);
        }
    };

    // Load global config from $XDG_CONFIG_HOME/akh-medu/config.toml.
    // Creates a default config file on first boot.
    let config_path = paths.global_config_file();
    let mut config = match AkhomedConfig::load_or_create(&config_path) {
        Ok(c) => {
            tracing::info!(path = %config_path.display(), "loaded config");
            c
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load config, using defaults");
            AkhomedConfig::default()
        }
    };

    // ACP TCP env overrides (mirrors AKH_SERVER_* / AKH_AUTO_START). Lets the
    // Helm chart enable + expose the ACP TCP listener — the daemon's only inbound
    // surface (ADR-061) — without mounting a config.toml. `AKH_ACP_TCP_PORT`
    // turns TCP on; `AKH_ACP_TCP_BIND` sets the host (config defaults to the
    // loopback `127.0.0.1`, so set `0.0.0.0` to reach a kubelet probe/Service).
    if let Ok(p) = std::env::var("AKH_ACP_TCP_PORT") {
        match p.trim().parse::<u16>() {
            Ok(port) => config.acp.tcp_port = Some(port),
            Err(e) => tracing::warn!("ignoring invalid AKH_ACP_TCP_PORT={p:?}: {e}"),
        }
    }
    if let Ok(host) = std::env::var("AKH_ACP_TCP_BIND") {
        let host = host.trim();
        if !host.is_empty() {
            config.acp.tcp_bind = Some(host.to_string());
        }
    }

    // AKH_AUTO_START env var overrides config if set.
    // "default" is treated as "no auto-start" (daemons are paused by default).
    let env_auto_start: Option<Vec<String>> = match std::env::var("AKH_AUTO_START") {
        Ok(val) => {
            let trimmed = val.trim();
            if trimmed.is_empty() || trimmed == "default" {
                Some(Vec::new())
            } else {
                Some(
                    trimmed
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty() && s != "default")
                        .collect(),
                )
            }
        }
        Err(_) => None,
    };

    // CLI --auto-start <workspace> adds to the auto-start list.
    // This is opt-in: workspaces are NOT auto-started unless explicitly listed.
    let mut cli_auto_start = Vec::new();
    {
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--auto-start" {
                if let Some(ws) = args.next() {
                    if !ws.is_empty() && ws != "default" {
                        cli_auto_start.push(ws);
                    }
                }
            }
        }
    }

    // Merge: env var takes precedence if set; otherwise use config + CLI.
    let auto_start: Vec<String> = if let Some(env_list) = env_auto_start {
        // Env var overrides everything. Merge with CLI for dedup.
        let mut merged = env_list;
        for w in cli_auto_start {
            if !merged.contains(&w) {
                merged.push(w);
            }
        }
        merged
    } else {
        // Config provides defaults (filtered to remove "default" sentinel),
        // CLI --auto-start appends.
        let mut merged: Vec<String> = config
            .daemon
            .auto_start
            .iter()
            .filter(|s| s.as_str() != "default")
            .cloned()
            .collect();
        for w in cli_auto_start {
            if !merged.contains(&w) {
                merged.push(w);
            }
        }
        merged
    };

    // Phase 38: open the Postgres backing store (or fall back to local-only).
    let pg_pool = init_postgres(&config).await;

    let state = Arc::new(ServerState::new(paths.clone(), config, pg_pool));

    tracing::info!(
        postgres = state.pg_pool().is_some(),
        "akhomed initialized"
    );

    // ADR 077: if we started local-only but a [postgres] section is configured,
    // keep trying to connect in the background and re-enable durability without
    // a restart (recovers from the launchd boot-time network race). No-op when
    // already connected or when no backing store is configured.
    if state.pg_pool().is_none() {
        state.spawn_pg_reconnector();
    }

    // Write PID file so `akh` CLI can discover this server. ACP transport
    // details (socket/tcp) are filled in later, once those listeners bind.
    if let Err(e) = akh_medu::client::write_pid_file(&paths, port_num, &bind, None, None) {
        tracing::warn!("failed to write PID file: {e}");
    }

    // Auto-start workspace daemons from config (or AKH_AUTO_START override).
    if !auto_start.is_empty() {
        let daemon_config = state.config.read().await.daemon.to_daemon_config();
        for ws_name in &auto_start {
            tracing::info!(workspace = %ws_name, "auto-starting daemon");
            match state.get_agent(ws_name).await {
                Ok(shared_agent) => {
                    let engine = state.get_engine(ws_name).await.unwrap();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let trigger_count = TriggerStore::new(&engine).list().len();
                    let status = Arc::new(tokio::sync::Mutex::new(DaemonStatus {
                        running: true,
                        total_cycles: 0,
                        started_at: now,
                        trigger_count,
                        last_persist_at: None,
                        last_learning_at: None,
                        last_sleep_at: None,
                        last_goal_gen_at: None,
                        active_goals: 0,
                        kg_symbols: 0,
                        kg_triples: 0,
                        knowledge_density: None,
                        last_compaction_at: None,
                    }));
                    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                    let daemon_status = Arc::clone(&status);
                    let daemon_ws = ws_name.clone();
                    let dc = daemon_config.clone();
                    // P5: per-akh automation gating from [autonomy] (off by default).
                    let autonomy = state
                        .get_akh_config(ws_name)
                        .await
                        .map(|c| c.autonomy.clone())
                        .unwrap_or_default();
                    let pause_handle = DaemonPauseHandle::new();
                    state.daemon_pauses.write().await.insert(ws_name.clone(), pause_handle.clone());
                    let task_handle = pause_handle.clone();
                    let pool = state.pg_pool();
                    let handle = tokio::task::spawn(async move {
                        run_daemon_task(
                            shared_agent,
                            daemon_status,
                            shutdown_rx,
                            dc,
                            autonomy,
                            daemon_ws,
                            task_handle,
                            pool,
                        )
                        .await;
                    });
                    state.daemons.write().await.insert(
                        ws_name.to_string(),
                        WorkspaceDaemon {
                            handle,
                            shutdown_tx,
                            status,
                        },
                    );
                    tracing::info!(workspace = %ws_name, "daemon auto-started");
                }
                Err(msg) => {
                    tracing::warn!(
                        workspace = %ws_name,
                        error = %msg,
                        "failed to auto-start daemon (workspace may not exist yet)",
                    );
                }
            }
        }
    }

    // Self-bootstrap the ACP workspace. A pure-ACP `akh` client can't create the
    // first workspace (the daemon must already serve ACP to accept
    // `_akh/workspace/create`), and the ACP server can only bind an *existing*
    // workspace — so ensure the one it will bind exists before starting it. The
    // resolution here must match `start_acp_server`'s (acp.workspace → first
    // auto_start → "default").
    #[cfg(feature = "acp")]
    {
        let acp_ws = state
            .config
            .read()
            .await
            .acp
            .workspace
            .clone()
            .or_else(|| auto_start.first().cloned())
            .unwrap_or_else(|| "default".to_string());
        // Ensure the workspace the ACP server will bind *exists* (data dir), but
        // never clobber an existing `akh.toml`. `WorkspaceManager::create` persists
        // only the `WorkspaceConfig` view, so calling it with a default would drop
        // akh-level fields (`backend`, `model`, `[ollama_endpoint]`, ADR 070). When
        // an `akh.toml` already exists, create from its embedded WorkspaceConfig and
        // then restore the full AkhConfig so the chat-backing selection survives.
        let cfg_file = paths.workspace_config_file(&acp_ws);
        if !paths.workspace(&acp_ws).exists() {
            let wsm = WorkspaceManager::new(paths.clone());
            let existing = cfg_file.exists().then(|| AkhConfig::load(&cfg_file).ok()).flatten();
            let ws_cfg = existing
                .as_ref()
                .map(|c| c.workspace.clone())
                .unwrap_or_else(|| akh_medu::workspace::WorkspaceConfig::with_name(&acp_ws));
            match wsm.create(ws_cfg) {
                Ok(_) => {
                    // Restore akh-level fields that `create` doesn't persist.
                    if let Some(akh_cfg) = existing {
                        if let Err(e) = akh_cfg.save(&cfg_file) {
                            tracing::warn!(workspace = %acp_ws, error = %e, "failed to restore akh.toml after create");
                        }
                    }
                    tracing::info!(workspace = %acp_ws, "ensured ACP workspace exists (self-bootstrap)")
                }
                Err(e) => {
                    tracing::warn!(workspace = %acp_ws, error = %e, "failed to auto-create ACP workspace")
                }
            }
        }
    }

    // ── ACP transport (Unix socket + optional TCP) ── the daemon's sole
    // inbound surface after ADR-061 P8.5 (REST/WS/web-UI/inbound-MCP retired).
    // Behind the `acp` feature; with it off the daemon has no inbound surface
    // and simply runs its workspace daemons until signalled.
    #[cfg(feature = "acp")]
    let (acp_shutdown_tx, acp_handles, acp_socket_path) = {
        let acp_cfg = state.config.read().await.acp.clone();
        let fallback_ws = auto_start
            .first()
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        let (acp_tx, acp_rx) = tokio::sync::watch::channel(false);
        let (acp_socket, acp_tcp, handles) =
            start_acp_server(Arc::clone(&state), acp_cfg, &paths, &fallback_ws, acp_rx).await;
        // Republish the discovery record now that ACP endpoints are known.
        if let Err(e) = akh_medu::client::write_pid_file(
            &paths,
            port_num,
            &bind,
            acp_socket.as_deref(),
            acp_tcp.as_deref(),
        ) {
            tracing::warn!("failed to update PID file with ACP endpoints: {e}");
        }
        (acp_tx, handles, acp_socket)
    };

    // No HTTP/WS/MCP surface remains (ADR-061 P8.5) — ACP is the only inbound
    // transport. Hold the process open until SIGTERM/SIGINT, then drain the ACP
    // accept loops and workspace daemons. The ACP listeners and workspace
    // daemons run as their own spawned tasks; main just waits and tears down.
    let paths_for_shutdown = paths.clone();
    let state_for_shutdown = state;

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
    tracing::info!("akhomed shutting down — draining workspace daemons");

    // Stop ACP accept loops first so no new sessions start mid-drain.
    #[cfg(feature = "acp")]
    {
        tracing::info!("stopping ACP listeners");
        let _ = acp_shutdown_tx.send(true);
        for h in acp_handles {
            let _ = tokio::time::timeout(Duration::from_secs(3), h).await;
        }
        if let Some(path) = acp_socket_path {
            // Only unlink the socket if the PID file still names us as
            // owner — a colliding second instance may have claimed both
            // artifacts already (the single-instance guard in
            // `start_acp_server` prevents the common case, but this covers
            // the race where it started between our probe and our bind).
            if owns_pid_file(&paths_for_shutdown) {
                let _ = std::fs::remove_file(path);
            } else {
                tracing::warn!(
                    socket = %path,
                    "not removing ACP socket on shutdown — PID file now names a different owner"
                );
            }
        }
    }

    // Signal all workspace daemons to stop, then join with timeout.
    let mut daemons = state_for_shutdown.daemons.write().await;
    let mut aborted = false;
    for (name, mut daemon) in daemons.drain() {
        tracing::info!(workspace = %name, "stopping workspace daemon");
        let _ = daemon.shutdown_tx.send(true);
        match tokio::time::timeout(Duration::from_secs(5), &mut daemon.handle).await {
            Ok(Ok(())) => tracing::info!(workspace = %name, "daemon stopped cleanly"),
            Ok(Err(e)) => tracing::warn!(workspace = %name, error = %e, "daemon task panicked"),
            Err(_) => {
                tracing::warn!(workspace = %name, "daemon did not stop within 5s, aborting");
                daemon.handle.abort();
                aborted = true;
            }
        }
    }
    drop(daemons);

    // Phase 38b: drain WAL flushers (after the daemons stop generating
    // mutations) so the final batch reaches Postgres.
    tracing::info!("draining WAL flushers");
    state_for_shutdown.shutdown_wals().await;

    if owns_pid_file(&paths_for_shutdown) {
        akh_medu::client::remove_pid_file(&paths_for_shutdown);
    } else {
        tracing::warn!("not removing PID file on shutdown — owned by a different process");
    }

    // abort() only cancels at .await points. Daemon tasks run sync CPU work
    // that never yields, so abort alone won't stop them. Force exit after a
    // short grace period to avoid leaving the process hanging.
    if aborted {
        tracing::info!("force-exiting after abort (daemon tasks run sync CPU work)");
        std::process::exit(0);
    }

    // Belt-and-suspenders: clean up PID file on normal exit too (already
    // removed above in the common case; this only fires if the first call's
    // ownership check raced against a startup collision resolving in our
    // favour in the interim — cheap enough to just check again).
    if owns_pid_file(&paths) {
        akh_medu::client::remove_pid_file(&paths);
    }
}

// ── ACP control-plane (`ServerControl for ServerState`) ─────────────────────
//
// Exercises the real backend directly (no socket): the lib's routing tests
// (`src/acp/mod.rs`) use a mock; here we drive `ServerState::dispatch` against
// a tempdir-backed state to prove the workspace + config ops actually mutate
// disk. Daemon ops are covered by their (unchanged) REST behaviour and only the
// cheap, agent-free `daemon/status` path is checked here.
#[cfg(all(test, feature = "acp"))]
mod control_plane_tests {
    use super::*;
    use akh_medu::acp::ServerControl;
    use serde_json::json;

    /// Build a `ServerState` rooted under a fresh tempdir.
    fn temp_state() -> (tempfile::TempDir, ServerState) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path();
        let paths = AkhPaths {
            config_dir: base.join("config"),
            data_dir: base.join("data"),
            state_dir: base.join("state"),
            runtime_dir: base.join("run"),
            cache_dir: base.join("cache"),
        };
        paths.ensure_dirs().expect("ensure dirs");
        let state = ServerState::new(paths, AkhomedConfig::default(), None);
        (tmp, state)
    }

    #[tokio::test]
    async fn control_workspace_lifecycle() {
        let (_tmp, state) = temp_state();

        // list → empty to start.
        let list = state
            .dispatch("_akh/workspace/list", &json!({}))
            .await
            .expect("list ok");
        assert_eq!(list["workspaces"].as_array().unwrap().len(), 0);

        // create "alpha".
        let created = state
            .dispatch("_akh/workspace/create", &json!({ "name": "alpha" }))
            .await
            .expect("create ok");
        assert_eq!(created["name"], "alpha");
        assert_eq!(created["created"], true);

        // list now contains it.
        let list = state
            .dispatch("_akh/workspace/list", &json!({}))
            .await
            .expect("list ok");
        let names: Vec<&str> = list["workspaces"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(names.contains(&"alpha"), "created workspace missing: {list}");

        // delete it.
        let deleted = state
            .dispatch("_akh/workspace/delete", &json!({ "name": "alpha" }))
            .await
            .expect("delete ok");
        assert_eq!(deleted["deleted"], "alpha");

        let list = state
            .dispatch("_akh/workspace/list", &json!({}))
            .await
            .expect("list ok");
        assert_eq!(list["workspaces"].as_array().unwrap().len(), 0);

        // missing required param → error.
        let bad = state.dispatch("_akh/workspace/create", &json!({})).await;
        assert!(bad.is_err(), "create without name must error");
    }

    #[tokio::test]
    async fn control_config_round_trips_to_disk() {
        let (_tmp, state) = temp_state();

        // get → wraps the AkhomedConfig.
        let got = state
            .dispatch("_akh/config/get", &json!({}))
            .await
            .expect("get ok");
        assert!(got["config"].is_object(), "config should be an object: {got}");

        // set with a mutated config → persisted + reflected on get.
        let cfg = AkhomedConfig {
            acp: akh_medu::config::AcpSection {
                tcp_port: Some(8299),
                ..Default::default()
            },
            ..Default::default()
        };
        let set = state
            .dispatch("_akh/config/set", &json!({ "config": cfg }))
            .await
            .expect("set ok");
        assert_eq!(set["config"]["acp"]["tcp_port"], 8299);

        // it landed on disk (config.toml under config_dir).
        assert!(
            state.paths.global_config_file().exists(),
            "config/set should have written config.toml"
        );

        // get reflects the change.
        let got = state
            .dispatch("_akh/config/get", &json!({}))
            .await
            .expect("get ok");
        assert_eq!(got["config"]["acp"]["tcp_port"], 8299);

        // set without a `config` param → error (never silently resets).
        let bad = state.dispatch("_akh/config/set", &json!({})).await;
        assert!(bad.is_err(), "config/set without config payload must error");
    }

    #[tokio::test]
    async fn control_config_set_merges_by_default() {
        let (_tmp, state) = temp_state();

        // Establish a baseline with a postgres section set.
        let baseline = AkhomedConfig {
            postgres: Some(akh_medu::config::PostgresConfig {
                url: "postgres://baseline/db".into(),
                pool_size: 10,
                connect_timeout_secs: 10,
                migrate_on_start: false,
                wal_flush_interval_secs: 2,
                wal_batch_size: 64,
                wal_max_buffer_mb: 16,
                snapshot_interval_secs: 300,
            }),
            ..Default::default()
        };
        state
            .dispatch("_akh/config/set", &json!({ "config": baseline }))
            .await
            .expect("baseline set ok");

        // A partial payload touching only `acp` must not wipe `postgres`.
        let partial = state
            .dispatch(
                "_akh/config/set",
                &json!({ "config": { "acp": { "tcp_port": 9100 } } }),
            )
            .await
            .expect("partial merge set ok");
        assert_eq!(partial["config"]["acp"]["tcp_port"], 9100);
        assert_eq!(
            partial["config"]["postgres"]["url"], "postgres://baseline/db",
            "partial config/set must merge at the table level, not reset \
             unrelated sections: {partial}"
        );

        // `replace: true` opts into the old wholesale-replace behavior: a
        // payload that omits `postgres` resets it to the type default (None).
        let replaced = state
            .dispatch(
                "_akh/config/set",
                &json!({ "config": { "acp": { "tcp_port": 9200 } }, "replace": true }),
            )
            .await
            .expect("replace set ok");
        assert_eq!(replaced["config"]["acp"]["tcp_port"], 9200);
        assert!(
            replaced["config"]["postgres"].is_null(),
            "replace: true must reset omitted sections to defaults: {replaced}"
        );
    }

    #[tokio::test]
    async fn get_akh_config_tolerates_malformed_toml() {
        let (_tmp, state) = temp_state();

        // A hand-written akh.toml that omits the flattened, required `name`
        // field. Everything else is valid — the sole defect is the missing
        // name, which `AkhConfig::load` rejects with a `missing field name`
        // parse error. Before the fix this aborted ACP startup.
        let config_path = state.paths.workspace_config_file("default");
        std::fs::write(
            &config_path,
            "model = \"x\"\n\n[ollama_endpoint]\nkind = \"local\"\n",
        )
        .expect("write malformed default.toml");

        // It must load (not error), defaulting the name to the workspace.
        let cfg = state
            .get_akh_config("default")
            .await
            .expect("malformed akh.toml must fall back, not error");
        assert_eq!(cfg.name(), "default", "name defaults to the workspace");
        // The fallback is name-only defaults: the unparseable file's akh-level
        // settings are intentionally dropped (the warning tells the operator).
        assert!(cfg.model.is_none(), "broken config falls back to defaults");

        // A second call returns the cached Arc.
        let again = state.get_akh_config("default").await.expect("cached load");
        assert!(Arc::ptr_eq(&cfg, &again), "config should be cached");
    }

    #[tokio::test]
    async fn control_daemon_status_when_idle() {
        let (_tmp, state) = temp_state();
        // No daemon running → cheap zero-status (no engine/agent construction).
        let st = state
            .dispatch("_akh/daemon/status", &json!({ "workspace": "alpha" }))
            .await
            .expect("status ok");
        assert_eq!(st["running"], false);

        // daemon ops require an explicit workspace.
        let bad = state.dispatch("_akh/daemon/status", &json!({})).await;
        assert!(bad.is_err(), "daemon/status without workspace must error");
    }

    #[tokio::test]
    async fn control_daemon_akhs_reports_live_set() {
        let (_tmp, state) = temp_state();
        // It is a control method (routed to ServerControl, not the data-plane).
        assert!(akh_medu::acp::is_control_method("_akh/daemon/akhs"));

        // No agent loaded → empty owned+awake set (no lazy load, no workspace arg).
        let res = state
            .dispatch("_akh/daemon/akhs", &json!({}))
            .await
            .expect("daemon/akhs ok");
        assert_eq!(
            res["akhs"].as_array().expect("akhs is an array").len(),
            0,
            "a daemon with no awake agents reports an empty set: {res}"
        );
    }

    #[tokio::test]
    async fn control_unknown_method_errors() {
        let (_tmp, state) = temp_state();
        let bad = state.dispatch("_akh/workspace/teleport", &json!({})).await;
        assert!(bad.is_err(), "unknown control method must error");
    }

    #[tokio::test]
    async fn akh_snapshot_wake_are_registered_control_methods() {
        // Both must be routed to the control-plane, not mistaken for data-plane.
        assert!(akh_medu::acp::is_control_method("_akh/akh/snapshot"));
        assert!(akh_medu::acp::is_control_method("_akh/akh/wake"));
    }

    #[tokio::test]
    async fn akh_snapshot_wake_need_workspace_param() {
        let (_tmp, state) = temp_state();
        // Missing the required `workspace` param is a clean error (recognised op).
        let r = state.dispatch("_akh/akh/snapshot", &json!({})).await;
        assert!(
            r.as_ref().is_err_and(|e| e.contains("workspace")),
            "expected missing-param error, got {r:?}",
        );
        let r = state.dispatch("_akh/akh/wake", &json!({})).await;
        assert!(
            r.as_ref().is_err_and(|e| e.contains("workspace")),
            "expected missing-param error, got {r:?}",
        );
    }

    #[tokio::test]
    async fn akh_snapshot_without_postgres_errors_cleanly() {
        // temp_state has no pg_pool → snapshot reports the missing backing store
        // rather than panicking or claiming success.
        let (_tmp, state) = temp_state();
        let r = state
            .dispatch("_akh/akh/snapshot", &json!({ "workspace": "alpha" }))
            .await;
        assert!(
            r.as_ref().is_err_and(|e| e.contains("Postgres")),
            "expected no-Postgres error, got {r:?}",
        );
    }

    #[tokio::test]
    async fn phase38_new_methods_are_registered_control_methods() {
        for m in [
            "_akh/akh/sleep",
            "_akh/akh/relocate",
            "_akh/workspace/join",
            "_akh/workspace/leave",
            "_akh/workspace/list-members",
            "_akh/projection/refresh",
            "_akh/migrate",
        ] {
            assert!(
                akh_medu::acp::is_control_method(m),
                "{m} must route to the control-plane",
            );
        }
    }

    #[tokio::test]
    async fn relocate_and_membership_and_migrate_need_postgres() {
        // No pg_pool → relocation/membership/migrate report the missing backing
        // store cleanly (recognised ops, not unknown-method or panic).
        let (_tmp, state) = temp_state();
        for (method, params) in [
            ("_akh/akh/relocate", json!({ "workspace": "alpha" })),
            ("_akh/workspace/join", json!({ "id": "alpha" })),
            ("_akh/workspace/leave", json!({ "id": "alpha" })),
            ("_akh/workspace/list-members", json!({ "id": "alpha" })),
            ("_akh/migrate", json!({})),
        ] {
            let r = state.dispatch(method, &params).await;
            assert!(
                r.as_ref().is_err_and(|e| e.contains("Postgres")),
                "{method} expected no-Postgres error, got {r:?}",
            );
        }
    }

    #[tokio::test]
    async fn membership_ops_need_id_param() {
        let (_tmp, state) = temp_state();
        for method in [
            "_akh/workspace/join",
            "_akh/workspace/leave",
            "_akh/workspace/list-members",
        ] {
            let r = state.dispatch(method, &json!({})).await;
            assert!(
                r.as_ref().is_err_and(|e| e.contains("id")),
                "{method} without id must error, got {r:?}",
            );
        }
    }

    #[tokio::test]
    async fn projection_refresh_is_a_stub() {
        // The AGE projection is deferred (Phase 38d); refresh returns an explicit
        // stub result rather than erroring, and needs no Postgres.
        let (_tmp, state) = temp_state();
        let r = state
            .dispatch("_akh/projection/refresh", &json!({ "workspace": "alpha" }))
            .await
            .expect("projection/refresh must succeed (stub)");
        assert_eq!(r["refreshed"], false);
        assert!(
            r["note"].as_str().unwrap_or_default().contains("deferred"),
            "stub should note deferral, got {r}",
        );
    }

    #[tokio::test]
    async fn sleep_without_postgres_evicts_and_notes_durability() {
        // temp_state has no pg_pool → sleep still evicts (frees memory) but the
        // result flags that durable relocation needs Postgres.
        let (_tmp, state) = temp_state();
        state
            .dispatch("_akh/workspace/create", &json!({ "name": "alpha" }))
            .await
            .expect("create");
        let r = state
            .dispatch("_akh/akh/sleep", &json!({ "workspace": "alpha" }))
            .await
            .expect("sleep must succeed (local-only evict)");
        assert_eq!(r["slept"], true);
        assert_eq!(r["level"], "deep");
        assert!(
            r["note"]
                .as_str()
                .unwrap_or_default()
                .to_ascii_lowercase()
                .contains("postgres"),
            "local-only sleep should note Postgres durability gap, got {r}",
        );
        // The engine/agent caches are emptied after eviction.
        assert!(
            state.workspaces.read().await.is_empty() && state.agents.read().await.is_empty(),
            "sleep must evict cached engine + agent",
        );
    }

    #[tokio::test]
    async fn get_engine_slow_path_is_single_flight() {
        let (_tmp, state) = temp_state();
        let state = Arc::new(state);

        // Materialise the workspace on disk (create without a role does not open it).
        state
            .dispatch("_akh/workspace/create", &json!({ "name": "alpha" }))
            .await
            .expect("create");
        assert!(
            state.workspaces.read().await.is_empty(),
            "create must not pre-open the engine",
        );

        // Fire N concurrent first-loads; they race on the per-workspace load lock.
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = Arc::clone(&state);
            handles.push(tokio::spawn(async move { s.get_engine("alpha").await }));
        }
        let mut engines = Vec::new();
        for h in handles {
            engines.push(h.await.expect("join").expect("get_engine"));
        }

        // All callers must share exactly one Engine instance (no duplicate build,
        // hence no leaked WAL flusher / snapshot ticker).
        for e in &engines[1..] {
            assert!(
                Arc::ptr_eq(&engines[0], e),
                "concurrent first-loads must share one engine",
            );
        }
        assert_eq!(state.workspaces.read().await.len(), 1, "exactly one engine cached");
    }

    #[tokio::test]
    async fn get_agent_slow_path_is_single_flight() {
        let (_tmp, state) = temp_state();
        let state = Arc::new(state);
        state
            .dispatch("_akh/workspace/create", &json!({ "name": "beta" }))
            .await
            .expect("create");

        let mut handles = Vec::new();
        for _ in 0..6 {
            let s = Arc::clone(&state);
            handles.push(tokio::spawn(async move { s.get_agent("beta").await }));
        }
        let mut agents = Vec::new();
        for h in handles {
            agents.push(h.await.expect("join").expect("get_agent"));
        }

        for a in &agents[1..] {
            assert!(
                Arc::ptr_eq(&agents[0], a),
                "concurrent first-loads must share one agent",
            );
        }
        assert_eq!(state.agents.read().await.len(), 1, "exactly one agent cached");
        assert_eq!(state.workspaces.read().await.len(), 1, "and exactly one engine");
    }
}
