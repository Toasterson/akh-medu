//! Delegation subsystem: dispatch tasks to external workers (native-runtime
//! sessions via akhsessd, peers, remote triggers).
//!
//! The [`DelegationManager`] is the central coordinator that selects a target,
//! enforces cost budgets via [`CostGuard`], and tracks active delegations.
//! Results flow back through an mpsc channel and are processed by the daemon
//! event loop.
//!
//! ### Synchronous enqueue / asynchronous dispatch
//!
//! The OODA loop's `DelegateTool` calls [`DelegationManager::enqueue`], which
//! is synchronous and simply pushes the task into a pending queue.  The daemon
//! calls [`DelegationManager::drain_pending`] on each delegation tick, which
//! pulls tasks from the queue and calls the async [`DelegationManager::dispatch`]
//! for each one.  This decouples the synchronous tool context from the async
//! subprocess machinery.

pub mod native_session;
pub mod peer;
pub mod remote_trigger;

mod error;
mod types;

pub use error::{DelegationError, DelegationResult};
pub use native_session::{DelegationGrantIssuer, GrantRequest, IssuedGrant, NativeSessionError};
pub use types::*;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dashmap::DashMap;

// ---------------------------------------------------------------------------
// CostGuard — hard budget enforcement
// ---------------------------------------------------------------------------

/// Tracks daily and per-task cost limits to prevent runaway spending.
///
/// All monetary values are stored in **cents** (u64) to avoid floating-point
/// drift. The daily budget resets at midnight UTC.
pub struct CostGuard {
    daily_budget_cents: u64,
    spent_today_cents: AtomicU64,
    per_task_max_cents: u64,
    /// Unix timestamp of the next daily reset (midnight UTC).
    budget_reset_at: AtomicU64,
    /// Model name → estimated cents per 1M input tokens.
    model_costs: std::collections::HashMap<String, u64>,
}

impl CostGuard {
    /// Create a new cost guard from config values.
    pub fn new(daily_budget_cents: u64, per_task_max_cents: u64) -> Self {
        let mut model_costs = std::collections::HashMap::new();
        // Approximate costs (cents per 1M input tokens) as of 2026-Q1.
        model_costs.insert("haiku".into(), 25); // $0.25/M
        model_costs.insert("sonnet".into(), 300); // $3/M
        model_costs.insert("opus".into(), 1500); // $15/M

        Self {
            daily_budget_cents,
            spent_today_cents: AtomicU64::new(0),
            per_task_max_cents,
            budget_reset_at: AtomicU64::new(Self::next_midnight_utc()),
            model_costs,
        }
    }

    /// Check whether we can afford an estimated token count on the given model.
    pub fn can_afford(&self, estimated_tokens: u64, model: &str) -> bool {
        self.maybe_reset_daily();
        // If daily budget is 0, only local (free) targets are allowed.
        if self.daily_budget_cents == 0 {
            return false;
        }
        let cost = self.estimate_cost(estimated_tokens, model);
        let spent = self.spent_today_cents.load(Ordering::Relaxed);
        let remaining = self.daily_budget_cents.saturating_sub(spent);
        cost <= remaining && cost <= self.per_task_max_cents
    }

    /// Whether local-only mode is in effect (daily_budget_cents == 0).
    pub fn is_local_only(&self) -> bool {
        self.daily_budget_cents == 0
    }

    /// Record actual token spend after a delegation completes.
    pub fn record_spend(&self, actual_tokens: u64, model: &str) {
        let cost = self.estimate_cost(actual_tokens, model);
        self.spent_today_cents.fetch_add(cost, Ordering::Relaxed);
    }

    /// Record spend in cents directly (when we know the exact cost).
    pub fn record_spend_cents(&self, cents: u64) {
        self.spent_today_cents.fetch_add(cents, Ordering::Relaxed);
    }

    /// Current daily spend in cents.
    pub fn spent_today_cents(&self) -> u64 {
        self.maybe_reset_daily();
        self.spent_today_cents.load(Ordering::Relaxed)
    }

    /// Remaining daily budget in cents.
    pub fn remaining_cents(&self) -> u64 {
        self.maybe_reset_daily();
        self.daily_budget_cents
            .saturating_sub(self.spent_today_cents.load(Ordering::Relaxed))
    }

    /// Daily budget in cents.
    pub fn daily_budget_cents(&self) -> u64 {
        self.daily_budget_cents
    }

    fn estimate_cost(&self, tokens: u64, model: &str) -> u64 {
        let rate = self.model_costs.get(model).copied().unwrap_or(300); // default to sonnet
        // cost = tokens * rate / 1_000_000
        tokens.saturating_mul(rate) / 1_000_000
    }

    fn maybe_reset_daily(&self) {
        let now = now_secs();
        let reset_at = self.budget_reset_at.load(Ordering::Relaxed);
        if now >= reset_at {
            self.spent_today_cents.store(0, Ordering::Relaxed);
            self.budget_reset_at
                .store(Self::next_midnight_utc(), Ordering::Relaxed);
        }
    }

    fn next_midnight_utc() -> u64 {
        let now = now_secs();
        let secs_since_midnight = now % 86_400;
        now + (86_400 - secs_since_midnight)
    }
}

impl std::fmt::Debug for CostGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CostGuard")
            .field("daily_budget_cents", &self.daily_budget_cents)
            .field(
                "spent_today_cents",
                &self.spent_today_cents.load(Ordering::Relaxed),
            )
            .field("per_task_max_cents", &self.per_task_max_cents)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// DelegationManager
// ---------------------------------------------------------------------------

/// Central coordinator for delegating tasks to external workers.
///
/// The manager selects a target based on capabilities and load, enforces cost
/// budgets, and tracks active delegations. Results arrive via an mpsc channel
/// consumed by the daemon event loop.
pub struct DelegationManager {
    config: DelegationConfig,
    active: DashMap<u64, ActiveDelegation>,
    cost_guard: CostGuard,
    next_id: AtomicU64,
    /// Tasks enqueued by synchronous tool calls, pending async dispatch.
    pending_tasks: Arc<Mutex<Vec<DelegationTask>>>,
    /// Completed results waiting to be processed by the daemon loop.
    #[cfg(feature = "daemon")]
    result_tx: tokio::sync::mpsc::Sender<DelegationResult<DelegationOutcome>>,
    #[cfg(feature = "daemon")]
    result_rx: Mutex<tokio::sync::mpsc::Receiver<DelegationResult<DelegationOutcome>>>,
    /// Known peer registry.
    peer_registry: peer::PeerRegistry,
    /// Mints per-delegation grants for native sessions (Phase 53e). `None`
    /// when the `/v1` core API is not wired, which disables that target.
    grant_issuer: Option<Arc<dyn DelegationGrantIssuer>>,
}

impl DelegationManager {
    /// Create a new delegation manager from config.
    #[cfg(feature = "daemon")]
    pub fn new(config: DelegationConfig) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cost_guard = CostGuard::new(config.daily_cost_budget_cents, config.per_task_max_cents);
        let peer_registry = peer::PeerRegistry::new(&config.peers);
        Self {
            config,
            active: DashMap::new(),
            cost_guard,
            next_id: AtomicU64::new(1),
            pending_tasks: Arc::new(Mutex::new(Vec::new())),
            result_tx: tx,
            result_rx: Mutex::new(rx),
            peer_registry,
            grant_issuer: None,
        }
    }

    /// Create a manager without async channels (for non-daemon builds).
    #[cfg(not(feature = "daemon"))]
    pub fn new(config: DelegationConfig) -> Self {
        let cost_guard = CostGuard::new(config.daily_cost_budget_cents, config.per_task_max_cents);
        let peer_registry = peer::PeerRegistry::new(&config.peers);
        Self {
            config,
            active: DashMap::new(),
            cost_guard,
            next_id: AtomicU64::new(1),
            pending_tasks: Arc::new(Mutex::new(Vec::new())),
            peer_registry,
            grant_issuer: None,
        }
    }

    /// Attach the grant issuer native-session delegation mints through.
    pub fn with_grant_issuer(mut self, issuer: Arc<dyn DelegationGrantIssuer>) -> Self {
        self.grant_issuer = Some(issuer);
        self
    }

    /// Whether native-session delegation can run: enabled and able to mint.
    pub fn native_available(&self) -> bool {
        self.config.native.enabled && self.grant_issuer.is_some()
    }

    fn native_target(&self) -> DelegationTargetKind {
        DelegationTargetKind::NativeSession {
            service: self.config.native.service.clone(),
            workspace_profile: self.config.native.workspace_profile.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Pending task queue (synchronous enqueue from OODA tool context)
    // -----------------------------------------------------------------------

    /// Enqueue a task for later async dispatch (called from synchronous tool context).
    ///
    /// Returns an error if delegation is disabled or at capacity.  The actual
    /// subprocess spawning happens on the next daemon delegation tick.
    pub fn enqueue(&self, task: DelegationTask) -> DelegationResult<()> {
        if !self.config.enabled {
            return Err(DelegationError::Disabled);
        }
        let pending = self.active.len() + self.pending_tasks.lock().map(|q| q.len()).unwrap_or(0);
        if pending >= self.config.max_concurrent {
            return Err(DelegationError::AtCapacity {
                max: self.config.max_concurrent,
                active: pending,
            });
        }
        if let Ok(mut q) = self.pending_tasks.lock() {
            q.push(task);
        }
        Ok(())
    }

    /// Clone the pending task queue Arc so the `DelegateTool` can push into it.
    pub fn pending_queue(&self) -> Arc<Mutex<Vec<DelegationTask>>> {
        Arc::clone(&self.pending_tasks)
    }

    /// Drain the pending queue and return tasks for async dispatch.
    ///
    /// Called by the daemon delegation tick under `#[cfg(feature = "daemon")]`.
    pub fn drain_pending(&self) -> Vec<DelegationTask> {
        self.pending_tasks
            .lock()
            .map(|mut q| std::mem::take(&mut *q))
            .unwrap_or_default()
    }

    // -----------------------------------------------------------------------
    // Core methods
    // -----------------------------------------------------------------------

    /// Allocate the next delegation ID.
    pub fn next_id(&self) -> DelegationId {
        DelegationId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Access the cost guard.
    pub fn cost_guard(&self) -> &CostGuard {
        &self.cost_guard
    }

    /// Access the config.
    pub fn config(&self) -> &DelegationConfig {
        &self.config
    }

    /// Whether delegation is enabled at all.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Register an active delegation for tracking.
    pub fn track(&self, delegation: ActiveDelegation) {
        self.active.insert(delegation.id.0, delegation);
    }

    /// Mark a delegation as completed and remove from active tracking.
    pub fn complete(&self, id: DelegationId) -> Option<ActiveDelegation> {
        self.active.remove(&id.0).map(|(_, v)| v)
    }

    /// Get a sender handle for posting results (daemon builds only).
    #[cfg(feature = "daemon")]
    pub fn result_sender(&self) -> tokio::sync::mpsc::Sender<DelegationResult<DelegationOutcome>> {
        self.result_tx.clone()
    }

    /// Try to receive a completed result (non-blocking).
    #[cfg(feature = "daemon")]
    pub fn try_recv(&self) -> Option<DelegationOutcome> {
        self.result_rx
            .lock()
            .ok()
            .and_then(|mut rx| rx.try_recv().ok())
            .and_then(|r| r.ok())
    }

    /// List active delegations.
    pub fn active_delegations(&self) -> Vec<ActiveDelegation> {
        self.active.iter().map(|r| r.value().clone()).collect()
    }

    /// Number of currently active delegations.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Number of tasks queued but not yet dispatched.
    pub fn pending_count(&self) -> usize {
        self.pending_tasks.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Whether we can accept another delegation (below max_concurrent).
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.config.max_concurrent
    }

    /// Check for timed-out delegations and mark them accordingly.
    pub fn check_timeouts(&self) -> Vec<DelegationId> {
        let now = now_secs();
        let mut timed_out = Vec::new();
        for mut entry in self.active.iter_mut() {
            let del = entry.value_mut();
            let elapsed = now.saturating_sub(del.started_at);
            if elapsed > del.timeout_secs && !matches!(del.status, DelegationStatus::TimedOut) {
                del.status = DelegationStatus::TimedOut;
                timed_out.push(DelegationId(del.id.0));
            }
        }
        timed_out
    }

    /// Access the peer registry.
    pub fn peer_registry(&self) -> &peer::PeerRegistry {
        &self.peer_registry
    }

    /// Mutable access to the peer registry.
    pub fn peer_registry_mut(&mut self) -> &mut peer::PeerRegistry {
        &mut self.peer_registry
    }

    // -----------------------------------------------------------------------
    // Target selection
    // -----------------------------------------------------------------------

    /// Select the best target for a task based on config, capabilities, and
    /// the sovereignty gradient (native session → peer → remote).
    ///
    /// `native` (alias `local`, kept for prompts written before Phase 53e)
    /// requires `[delegation.native] enabled` and a wired grant issuer.
    pub fn select_target(
        &self,
        required_capabilities: &[String],
        target_hint: Option<&str>,
    ) -> DelegationResult<DelegationTargetKind> {
        // Honour explicit hint first.
        if let Some(hint) = target_hint {
            return match hint {
                "native" | "local" => {
                    if self.native_available() {
                        Ok(self.native_target())
                    } else {
                        Err(DelegationError::NoTarget {
                            required: vec!["native".to_string()],
                        })
                    }
                }
                "remote" => {
                    if self.config.remote.enabled {
                        Ok(DelegationTargetKind::RemoteTrigger {
                            akhomed_public_url: self.config.remote.akhomed_public_url.clone(),
                            model: self.config.remote.model.clone(),
                        })
                    } else {
                        Err(DelegationError::NoTarget {
                            required: vec!["remote".to_string()],
                        })
                    }
                }
                "peer" => {
                    if let Some(endpoint) = self.peer_registry.best_peer(required_capabilities) {
                        Ok(DelegationTargetKind::AkhMeduPeer { endpoint })
                    } else {
                        Err(DelegationError::NoTarget {
                            required: vec!["peer".to_string()],
                        })
                    }
                }
                _ => Err(DelegationError::NoTarget {
                    required: vec![hint.to_string()],
                }),
            };
        }

        // Sovereignty gradient: native session → peer → remote.
        if self.native_available() {
            return Ok(self.native_target());
        }

        if self.config.peers.enabled
            && let Some(endpoint) = self.peer_registry.best_peer(required_capabilities)
        {
            return Ok(DelegationTargetKind::AkhMeduPeer { endpoint });
        }

        if self.config.remote.enabled && !self.cost_guard.is_local_only() {
            return Ok(DelegationTargetKind::RemoteTrigger {
                akhomed_public_url: self.config.remote.akhomed_public_url.clone(),
                model: self.config.remote.model.clone(),
            });
        }

        Err(DelegationError::NoTarget {
            required: vec!["any".to_string()],
        })
    }

    // -----------------------------------------------------------------------
    // Dispatch
    // -----------------------------------------------------------------------

    /// Dispatch a task to an appropriate target.
    ///
    /// Checks capacity, selects a target, verifies the budget, creates the
    /// active delegation record, and spawns the worker.
    #[cfg(feature = "daemon")]
    pub fn dispatch(
        &self,
        task: DelegationTask,
        target_hint: Option<&str>,
    ) -> DelegationResult<DelegationId> {
        if !self.config.enabled {
            return Err(DelegationError::Disabled);
        }
        if !self.has_capacity() {
            return Err(DelegationError::AtCapacity {
                max: self.config.max_concurrent,
                active: self.active_count(),
            });
        }

        let hint = target_hint.or(task.target_hint.as_deref());
        let target = self.select_target(&task.required_capabilities, hint)?;

        let model = match &target {
            DelegationTargetKind::RemoteTrigger { model, .. } => model.clone(),
            // Native sessions bill against their runtime profile, peers are free.
            DelegationTargetKind::NativeSession { .. }
            | DelegationTargetKind::AkhMeduPeer { .. } => String::new(),
        };

        // Native sessions: resolve everything that can fail before tracking.
        let native = match &target {
            DelegationTargetKind::NativeSession { .. } => {
                let issuer =
                    self.grant_issuer
                        .clone()
                        .ok_or_else(|| DelegationError::GrantUnavailable {
                            reason: "no grant issuer is wired for this akh".into(),
                        })?;
                let client = native_session::NativeSessionClient::from_config(&self.config.native)?;
                Some((issuer, client))
            }
            _ => None,
        };

        // Budget check — native sessions and peers are free here; only remote
        // triggers incur API cost.
        let is_paid = matches!(target, DelegationTargetKind::RemoteTrigger { .. });
        if is_paid {
            let estimated_tokens = 4_000u64;
            if !self.cost_guard.can_afford(estimated_tokens, &model) {
                return Err(DelegationError::BudgetExceeded {
                    spent_cents: self.cost_guard.spent_today_cents(),
                    budget_cents: self.config.daily_cost_budget_cents,
                });
            }
        }

        let id = self.next_id();
        let timeout_secs = match &target {
            DelegationTargetKind::NativeSession { .. } => self.config.native.timeout_seconds,
            DelegationTargetKind::RemoteTrigger { .. } => 300, // 5 min for remote
            DelegationTargetKind::AkhMeduPeer { .. } => 60,
        };

        let delegation = ActiveDelegation {
            id,
            goal_id: task.goal_id,
            target: target.clone(),
            task_description: task.description.clone(),
            started_at: now_secs(),
            timeout_secs,
            status: DelegationStatus::Running,
        };
        self.track(delegation);

        match target.clone() {
            DelegationTargetKind::NativeSession { .. } => {
                let (issuer, client) = native.expect("resolved above for a native target");
                let config = self.config.native.clone();
                let tx = self.result_sender();
                let task_clone = task;
                let id_clone = id;
                let target_clone = target;
                let handle = tokio::runtime::Handle::current();
                // ureq is blocking: run the whole session on a blocking thread
                // and hand the outcome back through the async channel.
                tokio::task::spawn_blocking(move || {
                    let result = native_session::run_native_delegation(
                        &client,
                        issuer.as_ref(),
                        &config,
                        &task_clone,
                        id_clone,
                    );
                    let outcome = match result {
                        Ok(mut outcome) => {
                            outcome.target = Some(target_clone);
                            outcome
                        }
                        Err(e) => DelegationOutcome {
                            id: id_clone,
                            stdout: String::new(),
                            structured: None,
                            wall_time: Duration::ZERO,
                            tokens_consumed: None,
                            cost_usd: None,
                            exit_code: None,
                            goal_id: task_clone.goal_id,
                            error: Some(e.to_string()),
                            target: Some(target_clone),
                        },
                    };
                    handle.block_on(async move {
                        let _ = tx.send(Ok(outcome)).await;
                    });
                });
            }
            DelegationTargetKind::RemoteTrigger {
                akhomed_public_url,
                model: m,
            } => {
                let worker = remote_trigger::RemoteTriggerWorker::new(
                    akhomed_public_url,
                    m.clone(),
                    self.config.remote.max_cost_per_task_cents,
                );
                let tx = self.result_sender();
                let task_clone = task;
                let id_clone = id;
                let target_clone = target;
                tokio::spawn(async move {
                    let result = worker.delegate_once(&task_clone, id_clone).await;
                    let outcome = match result {
                        Ok(mut outcome) => {
                            outcome.target = Some(target_clone);
                            outcome
                        }
                        Err(e) => DelegationOutcome {
                            id: id_clone,
                            stdout: String::new(),
                            structured: None,
                            wall_time: Duration::ZERO,
                            tokens_consumed: None,
                            cost_usd: None,
                            exit_code: None,
                            goal_id: task_clone.goal_id,
                            error: Some(e.to_string()),
                            target: Some(target_clone),
                        },
                    };
                    let _ = tx.send(Ok(outcome)).await;
                });
            }
            DelegationTargetKind::AkhMeduPeer { endpoint } => {
                let tx = self.result_sender();
                let task_clone = task;
                let id_clone = id;
                let target_clone = target;
                // Phase 53a: `delegation` joined the `server` profile and this
                // arm failed to compile — the ACP client's async-closure future
                // (`chat_over_unix`/`chat_over_tcp`) is not provably `Send` for
                // all lifetimes (higher-ranked inference limitation), so it
                // cannot be a `tokio::spawn` task. Drive it on a blocking thread
                // through the runtime handle instead; `Handle::block_on` has no
                // `Send` bound, and the captured values are all `Send`.
                let handle = tokio::runtime::Handle::current();
                tokio::task::spawn_blocking(move || {
                    handle.block_on(async move {
                        let result = peer::delegate_to_peer(&endpoint, &task_clone, id_clone).await;
                        let outcome = match result {
                            Ok(mut outcome) => {
                                outcome.target = Some(target_clone);
                                outcome
                            }
                            Err(e) => DelegationOutcome {
                                id: id_clone,
                                stdout: String::new(),
                                structured: None,
                                wall_time: Duration::ZERO,
                                tokens_consumed: None,
                                cost_usd: None,
                                exit_code: None,
                                goal_id: task_clone.goal_id,
                                error: Some(e.to_string()),
                                target: Some(target_clone),
                            },
                        };
                        let _ = tx.send(Ok(outcome)).await;
                    })
                });
            }
        }

        Ok(id)
    }
}

impl std::fmt::Debug for DelegationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelegationManager")
            .field("config", &self.config)
            .field("active_count", &self.active.len())
            .field("pending_count", &self.pending_count())
            .field("cost_guard", &self.cost_guard)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopIssuer;
    impl DelegationGrantIssuer for NoopIssuer {
        fn issue(&self, _: &GrantRequest) -> DelegationResult<IssuedGrant> {
            Err(DelegationError::GrantUnavailable {
                reason: "test".into(),
            })
        }
        fn bind_session(&self, _: &str, _: &str) -> DelegationResult<()> {
            Ok(())
        }
        fn revoke(&self, _: &str) -> DelegationResult<()> {
            Ok(())
        }
    }

    fn native_config() -> DelegationConfig {
        let mut config = DelegationConfig {
            enabled: true,
            ..Default::default()
        };
        config.native.enabled = true;
        config.native.workspace_profile = "code".into();
        config
    }

    #[test]
    fn native_sessions_need_a_grant_issuer() {
        let manager = DelegationManager::new(native_config());
        assert!(matches!(
            manager.select_target(&[], Some("native")),
            Err(DelegationError::NoTarget { .. })
        ));
        assert!(manager.select_target(&[], None).is_err());

        let manager =
            DelegationManager::new(native_config()).with_grant_issuer(Arc::new(NoopIssuer));
        let expected = DelegationTargetKind::NativeSession {
            service: "akhsessd".into(),
            workspace_profile: "code".into(),
        };
        assert_eq!(manager.select_target(&[], None).unwrap(), expected);
        // `local` is the pre-53e hint; it now means a native session.
        assert_eq!(manager.select_target(&[], Some("local")).unwrap(), expected);
        assert_eq!(expected.to_string(), "native-session:akhsessd/code");
    }

    #[test]
    fn a_partial_delegation_table_parses_with_defaults() {
        let config: DelegationConfig = toml::from_str(
            "enabled = true\n[native]\nenabled = true\nurl = \"http://127.0.0.1:9\"\n\
             grant_tools = { kind = \"explicit\", tools = [\"kg_query\"] }\n",
        )
        .unwrap();
        assert!(config.enabled && config.native.enabled);
        assert_eq!(config.native.core_principal, "sessions");
        assert_eq!(
            config.native.grant_tools,
            GrantTools::Explicit {
                tools: ["kg_query".to_string()].into()
            }
        );
        assert_eq!(config.max_concurrent, 3);
    }
}
