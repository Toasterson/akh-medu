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

use tokio::sync::RwLock;

use akh_medu::agent::trigger::TriggerStore;
use akh_medu::agent::model_manager::AgentModelManager;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::akh_config::AkhConfig;
use akh_medu::client::DaemonStatus;
use akh_medu::config::AkhomedConfig;
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::paths::AkhPaths;
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
    /// Shared registry of per-workspace pause handles. Daemons write their
    /// handles here; the daemon control-plane reads them for suspend/resume.
    daemon_pauses: RwLock<HashMap<String, DaemonPauseHandle>>,
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

impl ServerState {
    fn new(paths: AkhPaths, config: AkhomedConfig) -> Self {
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

    /// Get or lazily open an engine for the given workspace.
    async fn get_engine(&self, name: &str) -> Result<Arc<Engine>, String> {
        // Check if already loaded.
        {
            let map = self.workspaces.read().await;
            if let Some(engine) = map.get(name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Try to load it.
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.root.exists() {
            return Err(format!("workspace \"{name}\" not found"));
        }

        let config = EngineConfig {
            dimension: Dimension::DEFAULT,
            data_dir: Some(ws_paths.kg_dir.clone()),
            compartments_dir: Some(ws_paths.compartments_dir.clone()),
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

        let handle = tokio::task::spawn(async move {
            run_daemon_task(
                shared_agent,
                daemon_status,
                shutdown_rx,
                daemon_config,
                autonomy,
                daemon_ws_name,
                task_handle,
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
) {
    use akh_medu::agent::AgentDaemon;

    let mut daemon = AgentDaemon::new(shared_agent, daemon_config)
        .with_autonomy(autonomy)
        .with_shutdown(shutdown_rx)
        .with_status(Arc::clone(&status))
        .with_pause_handles(pause_handle.paused, pause_handle.cycle_done);

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

                // ── config group (global AkhomedConfig) ──────────────────
                "config/get" => {
                    let config = self.config.read().await;
                    let value = serde_json::to_value(&*config).map_err(|e| e.to_string())?;
                    Ok(serde_json::json!({ "config": value }))
                }
                "config/set" => {
                    let cfg_val = params.get("config").ok_or_else(|| {
                        "missing required param: config (the full AkhomedConfig object)".to_string()
                    })?;
                    let new_config: AkhomedConfig = serde_json::from_value(cfg_val.clone())
                        .map_err(|e| format!("invalid config payload: {e}"))?;
                    let config_path = self.paths.global_config_file();
                    new_config
                        .save(&config_path)
                        .map_err(|e| format!("save failed: {e}"))?;
                    *self.config.write().await = new_config.clone();
                    tracing::info!(
                        path = %config_path.display(),
                        "config updated via ACP control-plane"
                    );
                    let value = serde_json::to_value(&new_config).map_err(|e| e.to_string())?;
                    Ok(serde_json::json!({ "config": value }))
                }

                other => Err(format!("unknown control-plane method: _akh/{other}")),
            }
        })
    }
}

/// Bring up the ACP transport (Unix socket + optional TCP) for one workspace,
/// serving alongside the REST/WS server during the migration to ACP-only.
///
/// Returns `(unix_socket_path, tcp_addr, accept_loop_handles)` — the path/addr
/// strings (present only for transports that bound successfully) go into the
/// discovery record; the handles are joined on shutdown. A failure to open the
/// workspace or bind a listener disables that transport and is logged, never
/// fatal — the daemon still serves REST.
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
    let engine = match state.get_engine(&ws).await {
        Ok(e) => e,
        Err(msg) => {
            tracing::warn!(workspace = %ws, error = %msg, "ACP disabled: cannot open workspace");
            return (None, None, Vec::new());
        }
    };
    let agent = match state.get_agent(&ws).await {
        Ok(a) => a,
        Err(msg) => {
            tracing::warn!(workspace = %ws, error = %msg, "ACP disabled: cannot load agent");
            return (None, None, Vec::new());
        }
    };
    let nlu = state.get_nlu_pipeline(&ws, &engine).await;
    let processor = {
        let p = nlu.lock().unwrap_or_else(|e| e.into_inner());
        akh_medu::chat::ChatProcessor::new(&engine, p.clone())
    };
    // Wire the server-level control-plane: `ServerState` answers the
    // `_akh/{workspace,daemon,config}/*` methods that span the whole daemon.
    let control: Arc<dyn akh_medu::acp::ServerControl> = state.clone();
    let acp_agent = AkhAcpAgent::new(engine, agent, processor)
        .with_name(&ws)
        .with_control(control);

    let mut handles = Vec::new();
    let mut socket_str = None;
    let mut tcp_str = None;

    if acp_cfg.unix {
        let sock = acp_cfg
            .socket_path
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| paths.runtime_dir.join("akhomed-acp.sock"));
        // Clear a stale socket so bind() doesn't fail with EADDRINUSE.
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();

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

    let port_num: u16 = port.parse().expect("AKH_SERVER_PORT must be a valid u16");

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

    let state = Arc::new(ServerState::new(paths.clone(), config));

    tracing::info!("akhomed initialized");

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
                    let handle = tokio::task::spawn(async move {
                        run_daemon_task(
                            shared_agent,
                            daemon_status,
                            shutdown_rx,
                            dc,
                            autonomy,
                            daemon_ws,
                            task_handle,
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
            let _ = std::fs::remove_file(path);
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

    akh_medu::client::remove_pid_file(&paths_for_shutdown);

    // abort() only cancels at .await points. Daemon tasks run sync CPU work
    // that never yields, so abort alone won't stop them. Force exit after a
    // short grace period to avoid leaving the process hanging.
    if aborted {
        tracing::info!("force-exiting after abort (daemon tasks run sync CPU work)");
        std::process::exit(0);
    }

    // Belt-and-suspenders: clean up PID file on normal exit too.
    akh_medu::client::remove_pid_file(&paths);
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
        let state = ServerState::new(paths, AkhomedConfig::default());
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
    async fn control_unknown_method_errors() {
        let (_tmp, state) = temp_state();
        let bad = state.dispatch("_akh/workspace/teleport", &json!({})).await;
        assert!(bad.is_err(), "unknown control method must error");
    }
}
