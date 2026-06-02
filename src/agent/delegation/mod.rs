//! Delegation subsystem: dispatch tasks to external workers (Claude Code, peers).
//!
//! The [`DelegationManager`] is the central coordinator that selects a target,
//! enforces cost budgets via [`CostGuard`], and tracks active delegations.
//! Results flow back through an mpsc channel and are processed by the daemon
//! event loop.

#[cfg(feature = "daemon")]
pub mod local_claude;

mod error;
mod types;

pub use error::{DelegationError, DelegationResult};
pub use types::*;

use std::sync::atomic::{AtomicU64, Ordering};
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
        let cost = self.estimate_cost(estimated_tokens, model);
        let spent = self.spent_today_cents.load(Ordering::Relaxed);
        let remaining = self.daily_budget_cents.saturating_sub(spent);
        cost <= remaining && cost <= self.per_task_max_cents
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
    /// Completed results waiting to be processed by the daemon loop.
    #[cfg(feature = "daemon")]
    result_tx: tokio::sync::mpsc::Sender<DelegationResult<DelegationOutcome>>,
    #[cfg(feature = "daemon")]
    result_rx: std::sync::Mutex<tokio::sync::mpsc::Receiver<DelegationResult<DelegationOutcome>>>,
}

impl DelegationManager {
    /// Create a new delegation manager from config.
    #[cfg(feature = "daemon")]
    pub fn new(config: DelegationConfig) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cost_guard =
            CostGuard::new(config.daily_cost_budget_cents, config.per_task_max_cents);
        Self {
            config,
            active: DashMap::new(),
            cost_guard,
            next_id: AtomicU64::new(1),
            result_tx: tx,
            result_rx: std::sync::Mutex::new(rx),
        }
    }

    /// Create a manager without async channels (for non-daemon builds).
    #[cfg(not(feature = "daemon"))]
    pub fn new(config: DelegationConfig) -> Self {
        let cost_guard =
            CostGuard::new(config.daily_cost_budget_cents, config.per_task_max_cents);
        Self {
            config,
            active: DashMap::new(),
            cost_guard,
            next_id: AtomicU64::new(1),
        }
    }

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

    /// Number of currently active (non-completed) delegations.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Whether we can accept another delegation (below max_concurrent).
    pub fn has_capacity(&self) -> bool {
        self.active.len() < self.config.max_concurrent
    }

    /// Select the best target for a task based on config and capabilities.
    pub fn select_target(
        &self,
        _required_capabilities: &[String],
        target_hint: Option<&str>,
    ) -> DelegationResult<DelegationTargetKind> {
        // If the caller gave an explicit hint, honour it.
        if let Some(hint) = target_hint {
            return match hint {
                "local" if self.config.local.enabled => {
                    Ok(DelegationTargetKind::LocalClaude {
                        model: self.config.local.model.clone(),
                        allowed_tools: self.config.local.allowed_tools.clone(),
                        working_dir: self.config.local.working_dir.clone(),
                    })
                }
                _ => Err(DelegationError::NoTarget {
                    required: vec![hint.to_string()],
                }),
            };
        }

        // Default selection priority: local → (future: remote → peer)
        if self.config.local.enabled {
            return Ok(DelegationTargetKind::LocalClaude {
                model: self.config.local.model.clone(),
                allowed_tools: self.config.local.allowed_tools.clone(),
                working_dir: self.config.local.working_dir.clone(),
            });
        }

        Err(DelegationError::NoTarget {
            required: vec!["any".to_string()],
        })
    }

    /// Dispatch a task to an appropriate target.
    ///
    /// This is the primary entry point used by the OODA loop's delegate tool.
    /// It checks capacity, selects a target, verifies the budget, creates the
    /// active delegation record, and (for local targets) spawns the subprocess.
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

        let target = self.select_target(&task.required_capabilities, target_hint)?;

        // Budget check — estimate ~4K tokens for a typical task
        let estimated_tokens = 4_000u64;
        let model = match &target {
            DelegationTargetKind::LocalClaude { model, .. } => model.as_str(),
        };
        if !self.cost_guard.can_afford(estimated_tokens, model) {
            return Err(DelegationError::BudgetExceeded {
                spent_cents: self.cost_guard.spent_today_cents(),
                budget_cents: self.config.daily_cost_budget_cents,
            });
        }

        let id = self.next_id();
        let timeout_secs = match &target {
            DelegationTargetKind::LocalClaude { .. } => {
                self.config.local.timeout_seconds
            }
        };

        let delegation = ActiveDelegation {
            id: id.clone(),
            goal_id: task.goal_id,
            target: target.clone(),
            task_description: task.description.clone(),
            started_at: now_secs(),
            timeout_secs,
            status: DelegationStatus::Pending,
        };
        self.track(delegation);

        // Spawn the worker based on target kind.
        match target {
            DelegationTargetKind::LocalClaude {
                model,
                allowed_tools,
                working_dir,
            } => {
                let worker = local_claude::LocalClaudeWorker::new(
                    &self.config.local,
                    self.config.akhomed_url.clone(),
                );
                let tx = self.result_sender();
                let task_clone = task;
                let id_clone = id.clone();
                tokio::spawn(async move {
                    let result =
                        worker.execute(&task_clone, &model, &allowed_tools, working_dir.as_deref()).await;
                    let outcome = match result {
                        Ok(outcome) => outcome,
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
                        },
                    };
                    let _ = tx.send(Ok(outcome)).await;
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
