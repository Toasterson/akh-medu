use super::*;
use crate::{broker::Broker, session::Sessions, store::key};
use crate::{
    config::Principal,
    http::{Api, Client},
    types::*,
};
use axum::extract::Query;
use axum::{
    Extension,
    extract::{Path, State},
};
use serde::Deserialize;
use serde_json::Value;
use std::time::Instant;

#[cfg(all(test, feature = "core-integration-tests", feature = "candle-runtime"))]
#[path = "sessions_native_gate.rs"]
mod native_gate;

/// Deterministic core of the dispatch retry backoff: delay doubles with each
/// consecutive failure and is capped, so a permanently broken downstream
/// (an unreachable cross-cluster runtime, say) cannot turn into a fixed
/// sub-second retry storm that starves the runtime and the readiness probe
/// along with it. Jitter is applied separately by `jittered_backoff` so this
/// half stays a pure, easily pinned unit test.
fn backoff_after(consecutive_failures: u32) -> Duration {
    const BASE: Duration = Duration::from_millis(500);
    const CAP: Duration = Duration::from_secs(60);
    let exponent = consecutive_failures.min(7); // 500ms << 7 == 64s, already past the cap
    BASE.saturating_mul(1u32 << exponent).min(CAP)
}

/// Spread out retries for sessions that fail in lockstep (e.g. every session
/// dispatching through the same unreachable runtime) so they don't all wake
/// up on the same tick forever. The jitter source is derived from the
/// session id, not real randomness, so behavior stays reproducible in tests
/// and logs while still decorrelating sessions from each other.
fn jittered_backoff(consecutive_failures: u32, session_id: &str) -> Duration {
    let base = backoff_after(consecutive_failures);
    let millis = base.as_millis() as u64;
    let span = millis / 5; // +/- 20%
    if span == 0 {
        return base;
    }
    let entropy = session_id.bytes().fold(0xcbf29ce484222325u64, |hash, b| {
        (hash ^ b as u64).wrapping_mul(0x100000001b3)
    });
    let offset = (entropy % (2 * span + 1)) as i64 - span as i64;
    Duration::from_millis((millis as i64 + offset).max(0) as u64)
}

#[cfg(test)]
mod backoff_tests {
    use super::*;

    #[test]
    fn backoff_grows_and_saturates() {
        assert_eq!(backoff_after(0), Duration::from_millis(500));
        assert_eq!(backoff_after(1), Duration::from_secs(1));
        assert_eq!(backoff_after(2), Duration::from_secs(2));
        assert_eq!(backoff_after(20), Duration::from_secs(60));
    }

    #[test]
    fn jitter_stays_within_twenty_percent_and_never_negative() {
        for failures in 0..10 {
            for id in ["a", "session-b", "01a09a24-e09d-78a2-acfc-6731170068b2"] {
                let base = backoff_after(failures).as_millis() as i64;
                let jittered = jittered_backoff(failures, id).as_millis() as i64;
                let span = base / 5;
                assert!(jittered >= 0);
                assert!((jittered - base).abs() <= span.max(1));
            }
        }
    }

    #[test]
    fn jitter_decorrelates_sessions_sharing_a_failure_count() {
        let a = jittered_backoff(3, "session-a");
        let b = jittered_backoff(3, "session-b");
        assert_ne!(a, b, "distinct sessions should not retry in lockstep");
    }
}

#[derive(Clone)]
pub struct SessionApp {
    pub sessions: Sessions,
    pub broker: Broker,
    pub client: Client,
    pub admission_lock: Arc<std::sync::Mutex<()>>,
    /// Per-core `neural_spans_v1` capability, cached briefly so every turn
    /// does not re-ask. Transient failures are not cached.
    core_capabilities: Arc<std::sync::Mutex<std::collections::HashMap<String, (Instant, bool)>>>,
}

/// Retained runtime captures for a turn, keyed by session and turn.
fn span_captures_key(session_id: &str, turn_id: &str) -> String {
    key("span_captures", &[session_id, turn_id])
}

impl SessionApp {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        let sessions = Sessions::new(store, config.clone());
        Ok(Self {
            broker: Broker::new(sessions.clone())?,
            admission_lock: Default::default(),
            sessions,
            client: Client::new(config)?,
            core_capabilities: Default::default(),
        })
    }

    /// Whether `core` advertises the neural span annex. An older core without
    /// `/v1/capabilities` is remembered as unsupported; unreachable cores are
    /// treated as unsupported for this call only.
    pub async fn core_supports_neural_spans(&self, core: &str) -> bool {
        const TTL: Duration = Duration::from_secs(60);
        if let Some((at, supported)) = self
            .core_capabilities
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(core)
        {
            if at.elapsed() < TTL {
                return *supported;
            }
        }
        let supported = match self.client.get::<Value>(core, "/v1/capabilities").await {
            Ok(capabilities) => capabilities["neural_spans_v1"] == true,
            Err(error) if error.code == "not_found" => false,
            Err(error) => {
                tracing::debug!(%core, error = %error, "core capabilities unavailable");
                return false;
            }
        };
        self.core_capabilities
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(core.to_string(), (Instant::now(), supported));
        supported
    }

    /// Same-space recall on behalf of the runtime, under this service's core
    /// principal and the session's akh/session scope. Never invents hits.
    pub async fn neural_recall(
        &self,
        session: &SessionRecord,
        request: neural_spans::RecallRequest,
    ) -> Result<neural_spans::RecallReply> {
        request.validate()?;
        let Some(core) = &session.core_service else {
            return Ok(neural_spans::RecallReply::default());
        };
        if !self.core_supports_neural_spans(core).await {
            return Ok(neural_spans::RecallReply::default());
        }
        let query = neural_spans::SpanQuery {
            akh: session.akh.clone(),
            session_id: session.id.clone(),
            space: request.space,
            vector: request.vector,
            limit: request.limit,
        };
        match self
            .client
            .post::<Vec<neural_spans::SpanHit>>(core, "/v1/spans/query", &query)
            .await
        {
            Ok(hits) => Ok(neural_spans::RecallReply {
                supported: true,
                hits,
            }),
            // Capture policy disabled for this akh: recall is simply unavailable.
            Err(error) if error.code == "forbidden" => Ok(neural_spans::RecallReply::default()),
            Err(error) => Err(error),
        }
    }

    /// Retain runtime captures of the active turn's input until the turn is
    /// published to the core. Validated now against the accepted input text so
    /// publication cannot fail later on a malformed capture.
    pub fn retain_span_captures(
        &self,
        session: &SessionRecord,
        delivery: neural_spans::SpanDelivery,
    ) -> Result<neural_spans::SpanDeliveryReply> {
        let turn = session
            .turns
            .iter()
            .find(|t| t.id == delivery.call.turn_id)
            .ok_or_else(|| Fault::missing("turn not found"))?;
        delivery.validate(&turn.input.text)?;
        if session.core_service.is_none() || delivery.captures.is_empty() {
            return Ok(neural_spans::SpanDeliveryReply { accepted: false });
        }
        let k = span_captures_key(&session.id, &turn.id);
        self.sessions
            .store
            .transaction(|tx| tx.put(&k, &delivery.captures))?;
        Ok(neural_spans::SpanDeliveryReply { accepted: true })
    }
    /// Each session progresses independently. Dropping this task aborts its owned
    /// JoinSet, so shutdown cannot leave detached allocation/dispatch workers.
    pub async fn run_dispatcher(self) {
        use std::collections::{HashMap, HashSet};
        let mut workers: tokio::task::JoinSet<Result<()>> = tokio::task::JoinSet::new();
        let mut running: HashMap<tokio::task::Id, String> = HashMap::new();
        let mut last = HashMap::new();
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            tokio::select! {
                result = workers.join_next_with_id(), if !workers.is_empty() => {
                    match result {
                        Some(Ok((id, result))) => {
                            let session = running.remove(&id).unwrap_or_default();
                            if let Err(error) = result { tracing::error!(%session,code=%error.code,"session reconciliation failed"); }
                        }
                        Some(Err(error)) => {
                            let session = running.remove(&error.id()).unwrap_or_default();
                            tracing::error!(%session,"session reconciliation task interrupted");
                        }
                        None => {}
                    }
                }
                _ = interval.tick() => {
                    let mut sessions = match self.sessions.store.scan::<SessionRecord>("session/") {
                        Ok(sessions) => sessions,
                        Err(error) => { tracing::error!(code=%error.code,"session scan failed"); continue; }
                    };
                    // Oldest scheduled first prevents inactive sessions at the start
                    // of the ledger from starving later sessions when capacity is full.
                    sessions.sort_by_key(|session| last.get(&session.id).copied());
                    let active: HashSet<_> = running.values().cloned().collect();
                    for session in sessions {
                        if workers.len() >= 64 { break; }
                        if active.contains(&session.id) { continue; }
                        let retry_after = if session.closed_at.is_some() {
                            Duration::from_secs(5)
                        } else {
                            jittered_backoff(session.dispatch_failures, &session.id)
                        };
                        if last.get(&session.id).is_some_and(|at: &tokio::time::Instant| at.elapsed() < retry_after) { continue; }
                        let id = session.id.clone();
                        let worker = self.clone();
                        last.insert(id.clone(), tokio::time::Instant::now());
                        let task = workers.spawn(async move { worker.tick_session(session).await });
                        running.insert(task.id(), id);
                    }
                }
            }
        }
    }
    async fn tick_session(&self, session: SessionRecord) -> Result<()> {
        self.reconcile_publications(&session.id).await?;
        if session.closed_at.is_some() {
            self.sessions.remove_archived_mcp_credentials(&session.id)?;
            if self
                .sessions
                .store
                .get::<bool>(&key("workspace_released", &[&session.id]))?
                != Some(true)
            {
                match self.release_archived(&session).await {
                    Ok(()) => self.sessions.store.transaction(|tx| {
                        tx.put(&key("workspace_released", &[&session.id]), &true)
                    })?,
                    Err(error) => {
                        tracing::warn!(session=%session.id,code=%error.code,"archived workspace release deferred")
                    }
                }
            }
        } else if session.workspace.is_some() {
            let k = key("lease_renewed", &[&session.id]);
            if self
                .sessions
                .store
                .get::<u64>(&k)?
                .is_none_or(|at| now_ms().saturating_sub(at) >= 60_000)
            {
                match self
                    .client
                    .post::<Allocation>(
                        "sandboxes",
                        &format!("/v1/allocations/{}/renew", http::segment(&session.id)),
                        &json!({}),
                    )
                    .await
                {
                    Ok(_) => {}
                    Err(error) => {
                        tracing::warn!(session=%session.id,code=%error.code,"workspace lease renewal deferred");
                        // The manager already told us there is no implicit
                        // reacquisition (crates/akh-session-services/src/executor.rs
                        // renew_lease); waiting longer cannot fix it. Escalate
                        // now instead of warning forever with the sandbox
                        // still held.
                        if error.code == "expired"
                            && let Err(force_error) = self.sessions.force_retire(
                                &session.id,
                                "workspace lease expired; the executor will not implicitly reacquire it",
                            )
                        {
                            tracing::error!(session=%session.id,code=%force_error.code,"forced retirement after lease expiry failed");
                        }
                    }
                }
                self.sessions
                    .store
                    .transaction(|tx| tx.put(&k, &now_ms()))?;
            }
        }
        if let Err(error) = self.publish_completions(&session).await {
            tracing::warn!(session=%session.id,code=%error.code,"cognitive completion delivery deferred");
        }
        if let Err(error) = self.reconcile_progress(&session) {
            tracing::warn!(session=%session.id,code=%error.code,"session progress reconciliation deferred");
        }
        if !session.turns.iter().any(|t| !t.status.terminal()) {
            return Ok(());
        }
        if let Err(error) = self.dispatch(&session).await {
            tracing::warn!(session=%session.id,code=%error.code,"session reconciliation deferred");
            self.sessions.store.transaction(|tx| {
                let k = key("dispatch_error", &[&session.id]);
                let mut current: SessionRecord = tx
                    .get(&key("session", &[&session.id]))?
                    .ok_or_else(|| Fault::missing("session disappeared"))?;
                // Tracked on every failure, not only the first with a new
                // message, so the dispatcher's retry backoff (jittered_backoff)
                // keeps climbing for as long as the same problem persists
                // instead of resetting to the fast retry every tick.
                current.dispatch_failures = current.dispatch_failures.saturating_add(1);
                if tx.get::<Fault>(&k)?.as_ref() != Some(&error) {
                    tx.put(&k, &error)?;
                    crate::session::append_event(
                        tx,
                        &mut current,
                        None,
                        "dispatch_deferred",
                        json!({"error":error}),
                    )?;
                } else {
                    tx.put(&key("session", &[&session.id]), &current)?;
                }
                Ok(())
            })?;
        } else if session.dispatch_failures != 0 {
            self.sessions.store.transaction(|tx| {
                let mut current: SessionRecord = tx
                    .get(&key("session", &[&session.id]))?
                    .ok_or_else(|| Fault::missing("session disappeared"))?;
                current.dispatch_failures = 0;
                tx.put(&key("session", &[&session.id]), &current)
            })?;
        }
        Ok(())
    }
    /// Guarantee forward progress independent of any external caller. Runs on
    /// every tick but is throttled to `reconciliation_interval_ms` per
    /// session so it costs one durable read per tick, not one transaction:
    /// - A session that never made progress (no turn ever dispatched, an
    ///   allocated sandbox nobody is driving) is force-retired once it has
    ///   been quiet longer than `session_stall_threshold_ms`, so its caller
    ///   gets a terminal status instead of "provisioning" forever.
    /// - A session already retiring is finalized (closed + eligible for
    ///   sandbox release) as soon as its turns are terminal, and any turn
    ///   still waiting on a runtime that never confirmed cancellation is
    ///   forced terminal after `retirement_cancel_grace_ms` so retirement
    ///   cannot stay "pending" solely because a remote runtime is
    ///   unreachable.
    fn reconcile_progress(&self, session: &SessionRecord) -> Result<()> {
        if session.closed_at.is_some() {
            return Ok(());
        }
        let config = &self.sessions.config;
        let k = key("reconciliation_checked", &[&session.id]);
        if self
            .sessions
            .store
            .get::<u64>(&k)?
            .is_some_and(|at| now_ms().saturating_sub(at) < config.reconciliation_interval_ms)
        {
            return Ok(());
        }
        self.sessions
            .store
            .transaction(|tx| tx.put(&k, &now_ms()))?;
        let mut retiring = session.retirement.is_some();
        if !retiring
            && crate::session::is_stalled(
                session.last_progress_at,
                now_ms(),
                config.session_stall_threshold_ms,
            )
        {
            tracing::warn!(session=%session.id,"session made no progress within the stall threshold; forcing retirement");
            self.sessions.force_retire(
                &session.id,
                "no progress observed within the reconciliation stall threshold",
            )?;
            retiring = true;
        }
        if !retiring {
            return Ok(());
        }
        // Re-read: `force_retire` just above may be what set retirement, and
        // a session with no turns at all can go straight from admitted to
        // closed in this same pass instead of waiting for another sweep.
        let current: SessionRecord = self
            .sessions
            .store
            .get(&key("session", &[&session.id]))?
            .ok_or_else(|| Fault::missing("session disappeared"))?;
        if let Some(retirement) = &current.retirement
            && crate::session::should_force_cancel(
                retirement.at,
                now_ms(),
                config.retirement_cancel_grace_ms,
            )
        {
            self.sessions
                .force_cancel_unreachable(&session.id, config.retirement_cancel_grace_ms)?;
        }
        if self.sessions.finalize_retirement_if_ready(&session.id)? {
            tracing::info!(session=%session.id,"retirement finalized by reconciliation sweep");
        }
        Ok(())
    }
    async fn release_archived(&self, session: &SessionRecord) -> Result<()> {
        let path = format!("/v1/allocations/{}", http::segment(&session.id));
        if session.workspace.is_none() {
            match self.client.get::<Allocation>("sandboxes", &path).await {
                Err(error) if error.code == "not_found" => return Ok(()),
                Err(error) => return Err(error),
                Ok(_) => {}
            }
        }
        // A 404 from a downstream executor is NOT proof that the allocation is
        // absent. Only the preceding explicit manager lookup can establish that.
        let allocation: Allocation = self
            .client
            .post("sandboxes", &format!("{path}/release"), &json!({}))
            .await?;
        if allocation.request.session_id != session.id
            || !matches!(allocation.state.as_str(), "released" | "retained")
        {
            return Err(Fault::conflict(
                "sandbox manager did not confirm this session's release",
            ));
        }
        Ok(())
    }
    async fn publish_completions(&self, session: &SessionRecord) -> Result<()> {
        let Some(core) = &session.core_service else {
            return Ok(());
        };
        for turn in session
            .turns
            .iter()
            .filter(|turn| turn.status == TurnStatus::Completed)
        {
            let k = key("core_accepted", &[&turn.id]);
            if self.sessions.store.get::<ToolResult>(&k)?.is_some() {
                continue;
            }
            let request_key = key("core_accepted_request", &[&session.id, &turn.id]);
            let request =
                if let Some(request) = self.sessions.store.get::<AcceptedTurn>(&request_key)? {
                    request
                } else {
                    let tools: Vec<ToolEvidence> = self
                        .sessions
                        .store
                        .scan::<crate::broker::ToolRecord>(&key("tool", &[&session.id, &turn.id]))?
                        .into_iter()
                        .filter_map(|record| {
                            record.result.map(|result| ToolEvidence {
                                tool: record.call.name,
                                ok: result.success,
                                receipt: result.receipt,
                            })
                        })
                        .collect();
                    let receipts = tools
                        .iter()
                        .filter_map(|tool| tool.receipt.clone())
                        .collect();
                    // Only a core that advertises the annex receives the new field;
                    // an empty list preserves the original wire payload and fingerprint.
                    let span_captures =
                        match self.sessions.store.get::<Vec<neural_spans::SpanCapture>>(
                            &span_captures_key(&session.id, &turn.id),
                        )? {
                            Some(captures)
                                if !captures.is_empty()
                                    && self.core_supports_neural_spans(core).await =>
                            {
                                captures
                            }
                            _ => vec![],
                        };
                    let candidate = AcceptedTurn {
                        span_captures,
                        request_id: format!("turn-{}", turn.id),
                        akh: session.akh.clone(),
                        session_id: session.id.clone(),
                        turn_id: turn.id.clone(),
                        input: turn.input.clone(),
                        output: turn.output.clone().unwrap_or_default(),
                        model: session.runtime.model.clone(),
                        tool_receipts: receipts,
                        tools,
                    };
                    // Persist before any send: a lost reply must never cause this
                    // operation ID to acquire a different payload after restart,
                    // capability changes, or late runtime capture delivery.
                    let admitted = self.sessions.store.transaction(|tx| {
                        if tx.get::<ToolResult>(&k)?.is_some() {
                            return Ok(None);
                        }
                        if let Some(existing) = tx.get::<AcceptedTurn>(&request_key)? {
                            return Ok(Some(existing));
                        }
                        tx.put(&request_key, &candidate)?;
                        Ok(Some(candidate))
                    })?;
                    let Some(admitted) = admitted else { continue };
                    admitted
                };
            let result: ToolResult = self
                .client
                .post(core, "/v1/turns/accepted", &request)
                .await?;
            if !result.success {
                return Err(Fault::new("core_error", result.text));
            }
            self.sessions.store.transaction(|tx| {
                if tx.get::<ToolResult>(&k)?.is_some() {
                    return Ok(());
                }
                tx.put(&k, &result)?;
                tx.delete(&request_key)?;
                let mut current: SessionRecord = tx
                    .get(&key("session", &[&session.id]))?
                    .ok_or_else(|| Fault::missing("session disappeared"))?;
                crate::session::append_event(
                    tx,
                    &mut current,
                    Some(&turn.id),
                    "cognitive_turn_accepted",
                    json!({"receipt":result.receipt,"data":result.data}),
                )?;
                Ok(())
            })?;
        }
        Ok(())
    }
    async fn dispatch(&self, session: &SessionRecord) -> Result<()> {
        if session
            .turns
            .iter()
            .find(|t| !t.status.terminal())
            .is_some_and(|t| t.status == TurnStatus::Queued)
        {
            self.sessions
                .store
                .transaction(|tx| self.sessions.check_mcp_credentials(tx, session))?;
        }
        if session.workspace.is_none() {
            let allocation: Allocation = self
                .client
                .post(
                    "sandboxes",
                    "/v1/allocations",
                    &Allocate {
                        bootstrap: self.sessions.check_repository_bootstrap(session, false)?,
                        session_id: session.id.clone(),
                        principal: session.principal.clone(),
                        profile: session.workspace_profile.clone(),
                    },
                )
                .await?;
            if allocation.state != "ready" {
                return Ok(());
            }
            self.sessions.bind_prepared_workspace(
                &session.id,
                allocation
                    .binding
                    .ok_or_else(|| Fault::conflict("ready allocation has no workspace"))?,
                allocation.bootstrap_result,
            )?;
        }
        let current: SessionRecord = self
            .sessions
            .store
            .get(&key("session", &[&session.id]))?
            .ok_or_else(|| Fault::missing("session disappeared"))?;
        let Some(turn) = current.turns.iter().find(|t| !t.status.terminal()) else {
            return Ok(());
        };
        if turn.status == TurnStatus::Queued {
            if let Some(core) = &current.core_service {
                let k = key("turn_context", &[&turn.id]);
                if self.sessions.store.get::<ContextResponse>(&k)?.is_none() {
                    let context: ContextResponse = self
                        .client
                        .post(
                            core,
                            "/v1/context",
                            &ContextRequest {
                                request_id: turn.id.clone(),
                                akh: current.akh.clone(),
                                session_id: current.id.clone(),
                                turn_id: turn.id.clone(),
                                input: turn.input.clone(),
                                tools: current.tools.clone(),
                            },
                        )
                        .await?;
                    if context.schema_version != SCHEMA_VERSION {
                        return Err(Fault::conflict("unsupported cognitive context schema"));
                    }
                    if context
                        .model
                        .as_ref()
                        .is_some_and(|model| current.runtime.model.as_ref() != Some(model))
                    {
                        return Err(Fault::conflict(
                            "session runtime model differs from the authoritative Akh model; reconcile the profile before dispatch",
                        ));
                    }
                    self.sessions.store.transaction(|tx| {
                        tx.put(&k, &context)?;
                        let mut session: SessionRecord =
                            tx.get(&key("session", &[&current.id]))?
                                .ok_or_else(|| Fault::missing("session disappeared"))?;
                        crate::session::append_event(
                            tx,
                            &mut session,
                            Some(&turn.id),
                            "context_prepared",
                            json!({"service":core,"evidence":context.evidence}),
                        )?;
                        Ok(())
                    })?;
                }
            }
        }
        let request = match self.sessions.next_job(&session.id)? {
            Some(request) => request,
            None if turn.status == TurnStatus::Unknown => {
                match self
                    .sessions
                    .store
                    .get::<RuntimeJobRequest>(&key("dispatch", &[&turn.runtime_job_id]))?
                {
                    Some(request) => request,
                    None => return Ok(()),
                }
            }
            None => return Ok(()),
        };
        let path = format!("/v1/jobs/{}", http::segment(&request.id));
        let runtime = &request.binding.runtime;
        let mut job: RuntimeJob = match self.client.get(runtime, &path).await {
            Ok(job) => job,
            Err(error)
                if error.code == "not_found"
                    && matches!(turn.status, TurnStatus::Queued | TurnStatus::Dispatching) =>
            {
                // Repeating this request is safe only because the runtime persists exact request identity.
                self.client.post(runtime, "/v1/jobs", &request).await?
            }
            Err(error) if error.code == "not_found" => RuntimeJob {
                request: request.clone(),
                status: TurnStatus::Unknown,
                provider_session: current.provider_session.clone(),
                output: None,
                error: Some(Fault::new(
                    "unknown_outcome",
                    "runtime lost a previously admitted job; no redispatch occurred",
                )),
                events: vec![],
            },
            Err(error) => return Err(error),
        };
        if job.request != request {
            return Err(Fault::conflict(
                "runtime job does not match the durable dispatch request",
            ));
        }
        if turn.cancel_requested && !job.status.terminal() {
            job = self
                .client
                .post(runtime, &format!("{path}/cancel"), &json!({}))
                .await?;
        }
        let job = self.broker.settle(job).await?;
        self.sessions.apply_job(&job)
    }
}

pub fn session_router(app: SessionApp) -> Router {
    let config = app.sessions.config.clone();
    protect(
        Router::new()
            .route("/acp", get(crate::acp::upgrade))
            .route("/a2a/{akh}", post(crate::a2a::endpoint))
            .route(
                "/a2a/{akh}/.well-known/agent-card.json",
                get(crate::a2a::card),
            )
            .route("/v1/sessions", get(list_sessions).post(create_session))
            .route("/v1/sessions/{id}", get(get_session))
            .route("/v1/sessions/{id}/access", get(session_access))
            .route("/v1/sessions/{id}/lifecycle", get(session_lifecycle))
            .route(
                "/v1/sessions/{id}/services/{helper}/actions/{action}",
                post(session_service_action),
            )
            .route("/v1/sessions/{id}/inputs", post(submit_input))
            .route("/v1/sessions/{id}/events", get(session_events))
            .route("/v1/sessions/{id}/tools", get(session_tools))
            .route("/v1/sessions/{id}/pending", get(pending))
            .route("/v1/sessions/{id}/repository", post(prepare_repository))
            .route(
                "/v1/sessions/{id}/repository/snapshots",
                post(snapshot_repository),
            )
            .route(
                "/v1/sessions/{id}/repository/publications",
                post(publish_repository),
            )
            .route(
                "/v1/sessions/{id}/repository/publications/{request}",
                get(publication),
            )
            .route(
                "/v1/sessions/{id}/repository/publications/{request}/reconcile",
                post(reconcile_publication),
            )
            .route("/v1/sessions/{id}/input-decisions", post(decide_input))
            .route(
                "/v1/sessions/{id}/mcp-credentials/{service}",
                post(provision_mcp_credential),
            )
            .route("/v1/sessions/{id}/attach", post(attach))
            .route("/v1/sessions/{id}/detach", post(detach))
            .route("/v1/sessions/{id}/archive", post(archive))
            .route("/v1/sessions/{id}/archive-selected", post(archive_selected))
            .route("/v1/sessions/{id}/retire", post(retire))
            .route("/v1/sessions/{id}/turns/{turn}/cancel", post(cancel_turn))
            .route(
                "/v1/sessions/{id}/turns/{turn}/responses/{input}",
                post(respond),
            )
            .route("/v1/automations/{id}/invoke", post(invoke))
            .route("/v1/automations/{id}/revoke", post(revoke_automation))
            .route("/v1/runtime/tools", post(runtime_tool))
            .route("/v1/runtime/tools/reconcile", post(reconcile_tool))
            .route("/v1/runtime/input", post(runtime_input))
            .route("/v1/runtime/recall", post(runtime_recall))
            .route("/v1/runtime/spans", post(runtime_spans))
            .with_state(app),
        config,
    )
}
async fn revoke_automation(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<RevokeAutomation>,
) -> Api<AutomationRevocation> {
    Ok(Json(app.sessions.revoke_unadmitted(&p, &id, &request)?))
}
async fn session_lifecycle(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Value> {
    http::role(&p, "client")?;
    let session = app.sessions.get(&p, &id)?;
    let released = session.closed_at.is_some()
        && app
            .sessions
            .store
            .get::<bool>(&key("workspace_released", &[&id]))?
            == Some(true);
    Ok(Json(
        json!({"schema_version":SCHEMA_VERSION,"session_id":id,"closed_at":session.closed_at,"workspace_released":released}),
    ))
}
async fn retire(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<AttachmentQuery>,
    Json(request): Json<RetireSession>,
) -> Api<SessionRetirement> {
    http::role(&p, "client")?;
    Ok(Json(app.sessions.retire(
        &p,
        &id,
        query.attachment_id.as_deref(),
        &request,
    )?))
}
async fn provision_mcp_credential(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, service)): Path<(String, String)>,
    Json(request): Json<credentials::ProvisionMcpCredential>,
) -> Api<credentials::McpCredentialReceipt> {
    Ok(Json(
        app.sessions
            .provision_mcp_credential(&p, &id, &service, &request)?,
    ))
}
async fn prepare_repository(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<bootstrap::BootstrapRepository>,
) -> Api<bootstrap::RepositoryBootstrapReceipt> {
    Ok(Json(app.sessions.prepare_repository(&p, &id, &request)?))
}
async fn snapshot_repository(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<repository::SnapshotSessionRepository>,
) -> Api<OperationReceipt> {
    Ok(Json(app.snapshot_repository(&p, &id, &request).await?))
}
async fn publish_repository(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<publication::PublishSessionRepository>,
) -> Api<publication::SessionPublication> {
    Ok(Json(app.publish_repository(&p, &id, &request).await?))
}
async fn publication(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, request)): Path<(String, String)>,
) -> Api<publication::SessionPublication> {
    Ok(Json(app.sessions.publication(&p, &id, &request)?))
}
async fn reconcile_publication(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, request)): Path<(String, String)>,
    Json(resolution): Json<Reconciliation>,
) -> Api<publication::SessionPublication> {
    Ok(Json(
        app.reconcile_publication(&p, &id, &request, &resolution)
            .await?,
    ))
}
async fn archive(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<AttachmentQuery>,
) -> Api<SessionRecord> {
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    http::role(&p, "client")?;
    Ok(Json(app.sessions.archive_controlled(
        &p,
        &id,
        query.attachment_id.as_deref(),
    )?))
}
async fn archive_selected(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<AttachmentQuery>,
    Json(selection): Json<ArchiveSelection>,
) -> Api<SessionRecord> {
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    http::role(&p, "client")?;
    Ok(Json(app.sessions.archive_selected(
        &p,
        &id,
        query.attachment_id.as_deref(),
        &selection,
    )?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolResolution {
    call: ToolCall,
    resolution: Reconciliation,
}
async fn reconcile_tool(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(request): Json<ToolResolution>,
) -> Api<ToolResult> {
    Ok(Json(
        app.broker
            .reconcile_tool(&p, &request.call, &request.resolution)
            .await?,
    ))
}
async fn create_session(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(request): Json<NewSession>,
) -> Api<SessionRecord> {
    http::role(&p, "client")?;
    Ok(Json(app.sessions.create(&p, &request)?))
}
async fn get_session(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<SessionRecord> {
    Ok(Json(app.sessions.get(&p, &id)?))
}
async fn list_sessions(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
) -> Api<Vec<SessionRecord>> {
    http::role(&p, "client")?;
    Ok(Json(app.sessions.list(&p)?))
}
pub fn controller(
    app: &SessionApp,
    p: &Principal,
    id: &str,
    attachment: Option<&str>,
) -> Result<()> {
    let session = app.sessions.get(p, id)?;
    if let Some(lease) = session.controller {
        if lease.expires_at > now_ms() && Some(lease.attachment_id.as_str()) != attachment {
            return Err(Fault::conflict(
                "another client holds this session's controller lease",
            ));
        }
    }
    Ok(())
}
#[derive(Deserialize, Default)]
struct AttachmentQuery {
    attachment_id: Option<String>,
}
async fn submit_input(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<AttachmentQuery>,
    Json(request): Json<SubmitInput>,
) -> Api<TurnRecord> {
    http::role(&p, "client")?;
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    Ok(Json(app.sessions.submit_controlled(
        &p,
        &id,
        query.attachment_id.as_deref(),
        &request,
    )?))
}
#[derive(Deserialize, Default)]
struct Cursor {
    #[serde(default)]
    after: u64,
}
async fn session_events(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<Cursor>,
) -> Api<Vec<SessionEvent>> {
    Ok(Json(app.sessions.events(&p, &id, query.after)?))
}
#[derive(Deserialize)]
struct ToolCursor {
    after: Option<String>,
    limit: Option<usize>,
}
async fn session_tools(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<ToolCursor>,
) -> Api<crate::broker::ToolPage> {
    Ok(Json(app.broker.records(
        &p,
        &id,
        query.after.as_deref(),
        query.limit.unwrap_or(100),
    )?))
}
async fn pending(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Vec<PendingInput>> {
    Ok(Json(app.sessions.pending_inputs(&p, &id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Attachment {
    attachment_id: String,
    #[serde(default)]
    controller: bool,
}
async fn attach(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Attachment>,
) -> Api<SessionRecord> {
    http::role(&p, "client")?;
    Ok(Json(app.sessions.attach(
        &p,
        &id,
        &request.attachment_id,
        request.controller,
    )?))
}
async fn detach(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Attachment>,
) -> Api<Value> {
    app.sessions.detach(&p, &id, &request.attachment_id)?;
    Ok(Json(json!({"detached":true})))
}
async fn cancel_turn(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, turn)): Path<(String, String)>,
    Query(query): Query<AttachmentQuery>,
) -> Api<TurnRecord> {
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    http::role(&p, "client")?;
    Ok(Json(app.sessions.cancel_controlled(
        &p,
        &id,
        &turn,
        query.attachment_id.as_deref(),
    )?))
}
#[derive(Deserialize, Default)]
struct ResponseQuery {
    attachment_id: Option<String>,
    generation: Option<u64>,
}
async fn decide_input(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Query(query): Query<AttachmentQuery>,
    Json(request): Json<InputDecisionRequest>,
) -> Api<InputDecision> {
    http::role(&p, "client")?;
    Ok(Json(app.sessions.decide_input(
        &p,
        &id,
        query.attachment_id.as_deref(),
        &request,
    )?))
}
async fn respond(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, turn, input)): Path<(String, String, String)>,
    Query(query): Query<ResponseQuery>,
    Json(response): Json<Value>,
) -> Api<PendingInput> {
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    http::role(&p, "client")?;
    Ok(Json(app.sessions.respond_controlled(
        &p,
        &id,
        &turn,
        &input,
        query.generation,
        query.attachment_id.as_deref(),
        response,
    )?))
}
async fn invoke(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<InvokeAutomation>,
) -> Api<Admission> {
    Ok(Json(app.sessions.invoke(&p, &id, &request)?))
}
async fn runtime_tool(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(call): Json<ToolCall>,
) -> Api<CallbackReply> {
    Ok(Json(app.broker.tool(&p, call).await?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRequest {
    call: ToolCall,
    kind: String,
    prompt: Value,
}
async fn runtime_input(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(request): Json<InputRequest>,
) -> Api<CallbackReply> {
    Ok(Json(app.broker.input(
        &p,
        &request.call,
        &request.kind,
        request.prompt,
    )?))
}
async fn runtime_recall(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(request): Json<neural_spans::RecallRequest>,
) -> Api<neural_spans::RecallReply> {
    let session = app.sessions.validate_runtime_call(&p, &request.call)?;
    Ok(Json(app.neural_recall(&session, request).await?))
}
async fn runtime_spans(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Json(delivery): Json<neural_spans::SpanDelivery>,
) -> Api<neural_spans::SpanDeliveryReply> {
    let session = app.sessions.validate_runtime_call(&p, &delivery.call)?;
    Ok(Json(app.retain_span_captures(&session, delivery)?))
}

async fn session_access(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Vec<WorkspaceAccessDescriptor>> {
    app.sessions.get(&p, &id)?;
    let allocation: Allocation = app
        .client
        .get(
            "sandboxes",
            &format!("/v1/allocations/{}", http::segment(&id)),
        )
        .await?;
    let mut access = vec![];
    if let Some(workspace) = allocation.binding {
        let client = app.client.in_sandbox(&workspace.sandbox_id)?;
        for descriptor in allocation.access {
            let status: crate::types::HelperStatus = client
                .get(
                    &workspace.executor,
                    &format!("/v1/services/{}", http::segment(&descriptor.id)),
                )
                .await?;
            if let Some(descriptor) = status.access {
                access.push(descriptor);
            }
        }
    }
    Ok(Json(access))
}
async fn session_service_action(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, helper, action)): Path<(String, String, String)>,
) -> Api<Value> {
    let session = app.sessions.get(&p, &id)?;
    let workspace = session
        .workspace
        .ok_or_else(|| Fault::conflict("workspace is not ready"))?;
    Ok(Json(
        app.client
            .in_sandbox(&workspace.sandbox_id)?
            .post(
                &workspace.executor,
                &format!(
                    "/v1/services/{}/actions/{}",
                    http::segment(&helper),
                    http::segment(&action)
                ),
                &json!({"session_id":id}),
            )
            .await?,
    ))
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use axum::{
        Router,
        http::StatusCode,
        routing::{get, post},
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[tokio::test]
    async fn archive_waits_for_allocation_cleanup_even_without_a_bound_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-lifecycle-test-credential").unwrap();
        let exists = Arc::new(AtomicBool::new(true));
        let released = Arc::new(AtomicBool::new(false));
        let posts = Arc::new(AtomicUsize::new(0));
        let exists_peer = exists.clone();
        let released_peer = released.clone();
        let posts_peer = posts.clone();
        fn allocation(id: String, state: &str) -> Allocation {
            Allocation {
                bootstrap_profile_sha256: None,
                bootstrap_result: None,
                resource_uid: None,
                request: Allocate {
                    bootstrap: None,
                    session_id: id.clone(),
                    principal: "user".into(),
                    profile: "local".into(),
                },
                id,
                profile_revision: "1".into(),
                state: state.into(),
                binding: None,
                error: None,
                lease_until: 0,
                access: vec![],
            }
        }
        let peer = Router::new()
            .route(
                "/v1/allocations/{id}",
                get(move |Path(id): Path<String>| {
                    let exists = exists_peer.clone();
                    async move {
                        if exists.load(Ordering::SeqCst) {
                            (StatusCode::OK, Json(json!(allocation(id, "allocating"))))
                        } else {
                            (
                                StatusCode::NOT_FOUND,
                                Json(json!(Fault::missing("allocation absent"))),
                            )
                        }
                    }
                }),
            )
            .route(
                "/v1/allocations/{id}/release",
                post(move |Path(id): Path<String>| {
                    let released = released_peer.clone();
                    let posts = posts_peer.clone();
                    async move {
                        posts.fetch_add(1, Ordering::SeqCst);
                        if released.load(Ordering::SeqCst) {
                            (StatusCode::OK, Json(json!(allocation(id, "released"))))
                        } else {
                            (
                                StatusCode::NOT_FOUND,
                                Json(json!(Fault::missing("downstream executor unavailable"))),
                            )
                        }
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, peer).await.unwrap() });
        let config:Config=serde_json::from_value(json!({
            "service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir.path(),
            "principals":[{"id":"user","token_file":token,"roles":["client"],"akhs":["test"]}],
            "endpoints":{"sandboxes":{"url":format!("http://{address}"),"token_file":token}},
            "akhs":{"test":{"instructions":"test","runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"workspace_profiles":["local"]}}
        })).unwrap();
        let principal = config.principals[0].clone();
        let app = SessionApp::new(
            Arc::new(config),
            Store::open(&dir.path().join("state.redb")).unwrap(),
        )
        .unwrap();
        let session = app
            .sessions
            .create(
                &principal,
                &NewSession {
                    request_id: "create".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        let archived = app.sessions.archive(&principal, &session.id).unwrap();
        let marker = key("workspace_released", &[&session.id]);
        app.tick_session(archived.clone()).await.unwrap();
        assert_eq!(posts.load(Ordering::SeqCst), 1);
        assert_ne!(
            app.sessions.store.get::<bool>(&marker).unwrap(),
            Some(true),
            "downstream 404 must keep ownership fenced"
        );
        let binding = WorkspaceBinding {
            id: "workspace".into(),
            sandbox_id: "sandbox".into(),
            executor: "executor".into(),
            generation: 1,
            site: "test".into(),
            root: "/workspaces/test".into(),
            platform: "linux".into(),
            confinement: "fixture".into(),
        };
        assert_eq!(
            app.sessions
                .bind_workspace(&session.id, binding)
                .unwrap_err()
                .code,
            "conflict"
        );
        released.store(true, Ordering::SeqCst);
        app.tick_session(archived).await.unwrap();
        assert_eq!(app.sessions.store.get::<bool>(&marker).unwrap(), Some(true));
        let never_allocated = app
            .sessions
            .create(
                &principal,
                &NewSession {
                    request_id: "empty".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        let never_allocated = app
            .sessions
            .archive(&principal, &never_allocated.id)
            .unwrap();
        exists.store(false, Ordering::SeqCst);
        app.tick_session(never_allocated.clone()).await.unwrap();
        assert_eq!(
            app.sessions
                .store
                .get::<bool>(&key("workspace_released", &[&never_allocated.id]))
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            posts.load(Ordering::SeqCst),
            2,
            "a proven absent allocation needs no release"
        );
        server.abort();
    }

    /// A private fake cognitive core: records span queries and accepted turns,
    /// answers capability discovery, and returns canned hits.
    struct FakeCore {
        address: std::net::SocketAddr,
        fail_next_accept: Arc<std::sync::atomic::AtomicBool>,
        advertise_spans: Arc<std::sync::atomic::AtomicBool>,
        queries: Arc<std::sync::Mutex<Vec<Value>>>,
        accepted: Arc<std::sync::Mutex<Vec<Value>>>,
        server: tokio::task::JoinHandle<()>,
    }
    async fn fake_core(with_capabilities: bool, hits: Vec<Value>) -> FakeCore {
        use axum::routing::{get, post};
        let queries: Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let accepted: Arc<std::sync::Mutex<Vec<Value>>> = Default::default();
        let (q, a) = (queries.clone(), accepted.clone());
        let fail_next_accept = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let advertise_spans = Arc::new(std::sync::atomic::AtomicBool::new(with_capabilities));
        let fail = fail_next_accept.clone();
        let mut router = Router::new()
            .route(
                "/v1/spans/query",
                post(move |Json(body): Json<Value>| {
                    let q = q.clone();
                    let hits = hits.clone();
                    async move {
                        q.lock().unwrap().push(body);
                        Json(Value::Array(hits))
                    }
                }),
            )
            .route(
                "/v1/turns/accepted",
                post(move |Json(body): Json<Value>| {
                    let a = a.clone();
                    let fail = fail.clone();
                    async move {
                        a.lock().unwrap().push(body.clone());
                        if fail.swap(false, std::sync::atomic::Ordering::SeqCst) {
                            return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"code":"unavailable","message":"reply lost after acceptance"})));
                        }
                        (StatusCode::OK, Json(json!({"success":true,"text":"staged","data":{"captured":true},"receipt":body["request_id"]})))
                    }
                }),
            );
        if with_capabilities {
            let advertised = advertise_spans.clone();
            router = router.route(
                "/v1/capabilities",
                get(move || {
                    let supported = advertised.load(std::sync::atomic::Ordering::SeqCst);
                    async move { Json(json!({"neural_spans_v1":supported,"vector_bits":10000})) }
                }),
            );
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        FakeCore {
            address,
            fail_next_accept,
            advertise_spans,
            queries,
            accepted,
            server,
        }
    }

    fn neural_test_config(
        dir: &std::path::Path,
        token: &std::path::Path,
        core: std::net::SocketAddr,
    ) -> Value {
        json!({
            "service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir,
            "endpoints":{"core":{"url":format!("http://{core}"),"token_file":token}},
            "principals":[
                {"id":"user","token_file":token,"roles":["client"],"akhs":["test"]},
                {"id":"runtime","token_file":token,"roles":["runtime"]}
            ],
            "akhs":{"test":{"instructions":"test","core_service":"core","runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"internal","execution_mode":"brokered"},"workspace_profiles":["local"]}}
        })
    }

    fn span_space() -> neural_spans::SpanSpace {
        neural_spans::SpanSpace {
            model_sha256: "1".repeat(64),
            tokenizer_sha256: "2".repeat(64),
            policy_sha256: "3".repeat(64),
            projector_sha256: "4".repeat(64),
        }
    }

    /// Create a session with one dispatched turn and return (app, runtime
    /// principal, session record).
    async fn dispatched_session(
        config: Value,
        dir: &std::path::Path,
    ) -> (SessionApp, Principal, SessionRecord) {
        let config: Config = serde_json::from_value(config).unwrap();
        let user = config.principals[0].clone();
        let runtime = config.principals[1].clone();
        let app = SessionApp::new(
            Arc::new(config),
            Store::open(&dir.join("state.redb")).unwrap(),
        )
        .unwrap();
        let session = app
            .sessions
            .create(
                &user,
                &NewSession {
                    request_id: "create".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        app.sessions
            .submit(
                &user,
                &session.id,
                &SubmitInput {
                    request_id: "turn".into(),
                    text: "  Remember the word cedar  ".into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.turns[0].status = TurnStatus::Running;
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let record = app.sessions.get(&user, &session.id).unwrap();
        (app, runtime, record)
    }

    fn runtime_call(record: &SessionRecord, name: &str) -> ToolCall {
        ToolCall {
            id: format!("neural-{}", record.turns[0].id),
            session_id: record.id.clone(),
            turn_id: record.turns[0].id.clone(),
            generation: record.turns[0].generation,
            name: name.into(),
            arguments: json!({}),
        }
    }

    #[tokio::test]
    async fn accepted_payload_is_frozen_across_reply_loss_and_capability_change() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-neural-test-credential").unwrap();
        let core = fake_core(true, vec![]).await;
        let (app, _, record) = dispatched_session(
            neural_test_config(dir.path(), &token, core.address),
            dir.path(),
        )
        .await;
        app.retain_span_captures(
            &record,
            neural_spans::SpanDelivery {
                call: runtime_call(&record, neural_spans::SpanDelivery::CALL_NAME),
                captures: vec![neural_spans::SpanCapture {
                    space: span_space(),
                    source: neural_spans::SpanSource::User,
                    byte_start: 2,
                    byte_end: 25,
                    vector: vec![9; neural_spans::VECTOR_BYTES],
                }],
            },
        )
        .unwrap();
        app.sessions
            .store
            .transaction(|tx| {
                let mut current: SessionRecord = tx.get(&key("session", &[&record.id]))?.unwrap();
                current.turns[0].status = TurnStatus::Completed;
                current.turns[0].output = Some("cedar".into());
                tx.put(&key("session", &[&record.id]), &current)
            })
            .unwrap();
        let user = &app.sessions.config.principals[0];
        let completed = app.sessions.get(user, &record.id).unwrap();
        core.fail_next_accept
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(app.publish_completions(&completed).await.is_err());
        core.advertise_spans
            .store(false, std::sync::atomic::Ordering::SeqCst);
        // Fresh app discards the capability cache, as a daemon restart would.
        let restarted =
            SessionApp::new(app.sessions.config.clone(), app.sessions.store.clone()).unwrap();
        restarted.publish_completions(&completed).await.unwrap();
        let accepted = core.accepted.lock().unwrap().clone();
        assert_eq!(accepted.len(), 2);
        assert!(accepted[0].get("span_captures").is_some());
        assert_eq!(
            accepted[0], accepted[1],
            "a retry must preserve the accepted operation fingerprint"
        );
        restarted.publish_completions(&completed).await.unwrap();
        assert_eq!(core.accepted.lock().unwrap().len(), 2);
        assert!(
            restarted
                .sessions
                .store
                .get::<AcceptedTurn>(&key(
                    "core_accepted_request",
                    &[&record.id, &record.turns[0].id]
                ))
                .unwrap()
                .is_none(),
            "receipt must replace the temporary outbox payload"
        );
        core.server.abort();
    }

    #[tokio::test]
    async fn runtime_recall_and_span_delivery_ride_the_session_scope_to_the_core() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-neural-test-credential").unwrap();
        let hit = json!({"turn_id":"old","source":"user","byte_start":0,"byte_end":5,"text":"cedar","hamming_distance":12});
        let core = fake_core(true, vec![hit.clone()]).await;
        let (app, runtime, record) = dispatched_session(
            neural_test_config(dir.path(), &token, core.address),
            dir.path(),
        )
        .await;

        // Recall: the daemon adds akh/session scope and forwards the vector.
        let request = neural_spans::RecallRequest {
            call: runtime_call(&record, neural_spans::RecallRequest::CALL_NAME),
            space: span_space(),
            vector: vec![9; neural_spans::VECTOR_BYTES],
            limit: 4,
        };
        let session = app
            .sessions
            .validate_runtime_call(&runtime, &request.call)
            .unwrap();
        let reply = app.neural_recall(&session, request.clone()).await.unwrap();
        assert!(reply.supported);
        assert_eq!(reply.hits.len(), 1);
        assert_eq!(reply.hits[0].text, "cedar");
        let queries = core.queries.lock().unwrap().clone();
        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0]["akh"], "test");
        assert_eq!(queries[0]["session_id"], record.id);
        assert_eq!(queries[0]["limit"], 4);
        assert_eq!(queries[0]["vector"][0], 9);
        // A client principal or a stale generation never reaches the core.
        let user = app.sessions.config.principals[0].clone();
        assert!(
            app.sessions
                .validate_runtime_call(&user, &request.call)
                .is_err()
        );
        let mut stale = request.clone();
        stale.call.generation += 1;
        assert!(
            app.sessions
                .validate_runtime_call(&runtime, &stale.call)
                .is_err()
        );

        // Delivery: validated against the turn's accepted input, retained, and
        // attached to the accepted turn on publication.
        let capture = neural_spans::SpanCapture {
            space: span_space(),
            source: neural_spans::SpanSource::User,
            byte_start: 2,
            byte_end: 25,
            vector: vec![9; neural_spans::VECTOR_BYTES],
        };
        let mut bad = capture.clone();
        bad.byte_end = 200;
        let delivery = neural_spans::SpanDelivery {
            call: runtime_call(&record, neural_spans::SpanDelivery::CALL_NAME),
            captures: vec![bad],
        };
        assert!(app.retain_span_captures(&session, delivery).is_err());
        let delivery = neural_spans::SpanDelivery {
            call: runtime_call(&record, neural_spans::SpanDelivery::CALL_NAME),
            captures: vec![capture.clone()],
        };
        assert!(
            app.retain_span_captures(&session, delivery)
                .unwrap()
                .accepted
        );

        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.turns[0].status = TurnStatus::Completed;
                record.turns[0].output = Some("cedar".into());
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let completed = app.sessions.get(&user, &session.id).unwrap();
        app.publish_completions(&completed).await.unwrap();
        let accepted = core.accepted.lock().unwrap().clone();
        assert_eq!(accepted.len(), 1);
        assert_eq!(accepted[0]["input"]["text"], "  Remember the word cedar  ");
        assert_eq!(accepted[0]["span_captures"].as_array().unwrap().len(), 1);
        assert_eq!(accepted[0]["span_captures"][0]["byte_end"], 25);
        assert_eq!(accepted[0]["span_captures"][0]["source"], "user");
        // Publication is idempotent: the receipt prevents a second post.
        app.publish_completions(&completed).await.unwrap();
        assert_eq!(core.accepted.lock().unwrap().len(), 1);
        core.server.abort();
    }

    #[cfg(feature = "core-integration-tests")]
    #[tokio::test]
    async fn accepted_capture_round_trips_through_real_core_http() {
        use akh_medu::{
            akh_config::AkhConfig,
            core_service::{CoreApp, CoreBackend, CoreConfig, CoreFuture, CorePrincipal},
            engine::Engine,
        };
        use tower::ServiceExt;

        struct Backend(Arc<Engine>);
        impl CoreBackend for Backend {
            fn engine<'a>(&'a self, _: &'a str) -> CoreFuture<'a, Arc<Engine>> {
                Box::pin(async { Ok(self.0.clone()) })
            }
            fn config<'a>(&'a self, _: &'a str) -> CoreFuture<'a, Arc<AkhConfig>> {
                Box::pin(async {
                    Ok(Arc::new(AkhConfig {
                        capture_transcripts: true,
                        ..Default::default()
                    }))
                })
            }
        }
        // Abort even on assertion failure; no detached local server survives.
        struct Server(tokio::task::JoinHandle<()>);
        impl Drop for Server {
            fn drop(&mut self) {
                self.0.abort();
            }
        }
        async fn post(
            router: &Router,
            path: &str,
            token: &str,
            body: Value,
        ) -> (StatusCode, Value) {
            let response = router
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("authorization", format!("Bearer {token}"))
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 65536)
                .await
                .unwrap();
            (status, serde_json::from_slice(&bytes).unwrap())
        }
        assert!(
            std::env::var_os("AKH_TRANSCRIPT_CAPTURE").is_none(),
            "run without a process-wide transcript policy override"
        );
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("core.token");
        std::fs::write(&token, "private-real-core-test-credential").unwrap();
        let runtime_token = dir.path().join("runtime.token");
        let runtime_secret = "private-runtime-test-credential";
        std::fs::write(&runtime_token, runtime_secret).unwrap();
        let core = CoreApp::new(
            CoreConfig {
                listen: "127.0.0.1:0".parse().unwrap(),
                data_dir: dir.path().join("core"),
                principals: vec![CorePrincipal {
                    id: "sessions".into(),
                    token_file: token.clone(),
                    akhs: vec!["test".into()],
                }],
            },
            Arc::new(Backend(Arc::new(Engine::new(Default::default()).unwrap()))),
        )
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let _server = Server(tokio::spawn(async move {
            axum::serve(listener, akh_medu::core_service::router(core))
                .await
                .unwrap();
        }));
        let mut config = neural_test_config(dir.path(), &token, address);
        config["principals"][1]["token_file"] = json!(runtime_token);
        let (app, _, record) = dispatched_session(config, dir.path()).await;
        let router = session_router(app.clone());
        let query = json!(neural_spans::RecallRequest {
            call: runtime_call(&record, neural_spans::RecallRequest::CALL_NAME),
            space: span_space(),
            vector: vec![9; neural_spans::VECTOR_BYTES],
            limit: 4,
        });
        let before = post(&router, "/v1/runtime/recall", runtime_secret, query.clone()).await;
        assert_eq!(before.0, StatusCode::OK);
        assert_eq!(before.1, json!({"supported":true,"hits":[]}));
        let delivery = json!(neural_spans::SpanDelivery {
            call: runtime_call(&record, neural_spans::SpanDelivery::CALL_NAME),
            captures: vec![neural_spans::SpanCapture {
                space: span_space(),
                source: neural_spans::SpanSource::User,
                byte_start: 2,
                byte_end: 25,
                vector: vec![9; neural_spans::VECTOR_BYTES]
            }],
        });
        assert_eq!(
            post(&router, "/v1/runtime/spans", runtime_secret, delivery).await,
            (StatusCode::OK, json!({"accepted":true}))
        );
        // Capturing a running turn must not publish unaccepted memory.
        assert_eq!(
            post(&router, "/v1/runtime/recall", runtime_secret, query.clone())
                .await
                .1,
            json!({"supported":true,"hits":[]})
        );
        // Scheduler fixture: this test supplies completion, not model inference.
        app.sessions
            .store
            .transaction(|tx| {
                let mut current: SessionRecord = tx.get(&key("session", &[&record.id]))?.unwrap();
                current.turns[0].status = TurnStatus::Completed;
                current.turns[0].output = Some("Noted.".into());
                tx.put(&key("session", &[&record.id]), &current)
            })
            .unwrap();
        let owner = &app.sessions.config.principals[0];
        let completed = app.sessions.get(owner, &record.id).unwrap();
        app.publish_completions(&completed).await.unwrap();
        app.publish_completions(&completed).await.unwrap();
        app.sessions
            .submit(
                owner,
                &record.id,
                &SubmitInput {
                    request_id: "query-turn".into(),
                    text: "What word should I remember?".into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        app.sessions
            .store
            .transaction(|tx| {
                let mut current: SessionRecord = tx.get(&key("session", &[&record.id]))?.unwrap();
                current.turns[1].status = TurnStatus::Running;
                tx.put(&key("session", &[&record.id]), &current)
            })
            .unwrap();
        let current = app.sessions.get(owner, &record.id).unwrap();
        // A completed turn cannot replay a recall callback with its original
        // (otherwise valid) generation and runtime credential.
        assert_eq!(
            post(&router, "/v1/runtime/recall", runtime_secret, query.clone())
                .await
                .0,
            StatusCode::CONFLICT
        );
        let mut recall = query;
        recall["call"]["turn_id"] = json!(current.turns[1].id);
        recall["call"]["generation"] = json!(current.turns[1].generation);
        let after = post(
            &router,
            "/v1/runtime/recall",
            runtime_secret,
            recall.clone(),
        )
        .await;
        assert_eq!(after.0, StatusCode::OK);
        assert_eq!(
            after.1["hits"],
            json!([{"turn_id":record.turns[0].id,
            "source":"user","byte_start":2,"byte_end":25,
            "text":"Remember the word cedar","hamming_distance":0}])
        );
        assert_eq!(
            post(
                &router,
                "/v1/runtime/recall",
                "incorrect-credential",
                recall.clone()
            )
            .await
            .0,
            StatusCode::UNAUTHORIZED
        );
        let mut stale = recall.clone();
        stale["call"]["generation"] = json!(current.generation + 1);
        assert_eq!(
            post(&router, "/v1/runtime/recall", runtime_secret, stale)
                .await
                .0,
            StatusCode::CONFLICT
        );
        let other = app
            .sessions
            .create(
                owner,
                &NewSession {
                    request_id: "other-session".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        let other_turn = app
            .sessions
            .submit(
                owner,
                &other.id,
                &SubmitInput {
                    request_id: "other-query".into(),
                    text: "What word should I remember?".into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        recall["call"]["session_id"] = json!(other.id);
        recall["call"]["turn_id"] = json!(other_turn.id);
        recall["call"]["generation"] = json!(other_turn.generation);
        assert_eq!(
            post(&router, "/v1/runtime/recall", runtime_secret, recall).await,
            (StatusCode::OK, json!({"supported":true,"hits":[]}))
        );
        #[cfg(feature = "candle-runtime")]
        native_gate::run(app, router, runtime_secret, dir.path()).await;
    }

    #[tokio::test]
    async fn an_older_core_without_capabilities_gets_the_legacy_payload_and_no_recall() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-neural-test-credential").unwrap();
        let core = fake_core(false, vec![]).await;
        let (app, runtime, record) = dispatched_session(
            neural_test_config(dir.path(), &token, core.address),
            dir.path(),
        )
        .await;
        let request = neural_spans::RecallRequest {
            call: runtime_call(&record, neural_spans::RecallRequest::CALL_NAME),
            space: span_space(),
            vector: vec![0; neural_spans::VECTOR_BYTES],
            limit: 2,
        };
        let session = app
            .sessions
            .validate_runtime_call(&runtime, &request.call)
            .unwrap();
        let reply = app.neural_recall(&session, request).await.unwrap();
        assert!(!reply.supported);
        assert!(
            core.queries.lock().unwrap().is_empty(),
            "no query without capability"
        );
        let delivery = neural_spans::SpanDelivery {
            call: runtime_call(&record, neural_spans::SpanDelivery::CALL_NAME),
            captures: vec![neural_spans::SpanCapture {
                space: span_space(),
                source: neural_spans::SpanSource::User,
                byte_start: 2,
                byte_end: 25,
                vector: vec![1; neural_spans::VECTOR_BYTES],
            }],
        };
        assert!(
            app.retain_span_captures(&session, delivery)
                .unwrap()
                .accepted
        );
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.turns[0].status = TurnStatus::Completed;
                record.turns[0].output = Some("cedar".into());
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let user = app.sessions.config.principals[0].clone();
        let completed = app.sessions.get(&user, &session.id).unwrap();
        app.publish_completions(&completed).await.unwrap();
        let accepted = core.accepted.lock().unwrap().clone();
        assert_eq!(accepted.len(), 1);
        assert!(
            accepted[0].get("span_captures").is_none(),
            "legacy wire payload must not carry the new field: {}",
            accepted[0]
        );
        core.server.abort();
    }

    fn lifecycle_test_config(dir: &std::path::Path, token: &std::path::Path) -> Value {
        json!({
            "service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir,
            "principals":[{"id":"user","token_file":token,"roles":["client"],"akhs":["test"]}],
            "akhs":{"test":{"instructions":"test","runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"workspace_profiles":["local"]}}
        })
    }

    /// Reproduces the incident this module exists to fix: a session is
    /// admitted (in the real outage, a sandbox was already allocated to it)
    /// and then never receives a turn — because the pod that would have
    /// dispatched one was rescheduled mid-flight. With no non-terminal turn,
    /// the old dispatcher had nothing that would ever call `dispatch()`
    /// again, so the session sat in "provisioning" with zero witnesses
    /// indefinitely. The stall sweep must notice and force it to a terminal
    /// state instead.
    #[tokio::test]
    async fn stalled_session_with_no_turns_is_force_retired_and_closed_in_one_pass() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-lifecycle-test-credential").unwrap();
        let config: Config =
            serde_json::from_value(lifecycle_test_config(dir.path(), &token)).unwrap();
        let principal = config.principals[0].clone();
        let app = SessionApp::new(
            Arc::new(config),
            Store::open(&dir.path().join("state.redb")).unwrap(),
        )
        .unwrap();
        let session = app
            .sessions
            .create(
                &principal,
                &NewSession {
                    request_id: "create".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        // Simulate the pod-reschedule gap: no event has touched this session
        // since long before the stall threshold.
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.last_progress_at = 0;
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        app.tick_session(app.sessions.get(&principal, &session.id).unwrap())
            .await
            .unwrap();
        let after = app.sessions.get(&principal, &session.id).unwrap();
        assert!(
            after.retirement.is_some(),
            "a session with no progress must be force-retired, never left provisioning forever"
        );
        assert!(
            after.closed_at.is_some(),
            "with zero turns there is nothing to wait on; it should close in the same pass"
        );
    }

    /// A session whose only turn is mid-flight on an unreachable runtime
    /// (the cross-cluster case in the incident: the relay could not reach
    /// the runtime to confirm cancellation). Retirement must still complete
    /// once the bounded grace period elapses, rather than staying
    /// "pending_central_retirement" forever.
    #[tokio::test]
    async fn retirement_completes_once_an_unreachable_runtime_stops_blocking_it() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-lifecycle-test-credential").unwrap();
        let mut value = lifecycle_test_config(dir.path(), &token);
        value["retirement_cancel_grace_ms"] = json!(0);
        let config: Config = serde_json::from_value(value).unwrap();
        let principal = config.principals[0].clone();
        let app = SessionApp::new(
            Arc::new(config),
            Store::open(&dir.path().join("state.redb")).unwrap(),
        )
        .unwrap();
        let session = app
            .sessions
            .create(
                &principal,
                &NewSession {
                    request_id: "create".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        app.sessions
            .submit(
                &principal,
                &session.id,
                &SubmitInput {
                    request_id: "turn".into(),
                    text: "do work".into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        app.sessions
            .retire(
                &principal,
                &session.id,
                None,
                &RetireSession {
                    request_id: "retire".into(),
                    reason: "operator requested shutdown".into(),
                },
            )
            .unwrap();
        // Advance the turn past Queued (which `retire` already cancels
        // directly) into a dispatched state that only the unreachable
        // runtime could otherwise confirm as cancelled.
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.turns[0].status = TurnStatus::Running;
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let stuck = app.sessions.get(&principal, &session.id).unwrap();
        assert!(
            stuck.turns[0].cancel_requested && !stuck.turns[0].status.terminal(),
            "the turn must look exactly like one waiting on a runtime that never answered"
        );
        app.tick_session(stuck).await.unwrap();
        let after = app.sessions.get(&principal, &session.id).unwrap();
        assert!(
            after.turns[0].status.terminal(),
            "an unconfirmed cancellation past the grace period must close locally"
        );
        assert_eq!(
            after.turns[0].error.as_ref().map(|e| e.code.as_str()),
            Some("unknown_outcome")
        );
        assert!(
            after.closed_at.is_some(),
            "retirement must finalize once every turn is terminal, with no relay involved"
        );
    }

    /// The other literal shape of the incident: a turn was genuinely
    /// dispatched, the runtime pod it was running on got rescheduled, and the
    /// runtime's own restart-time scan (`Runtime::new`) marked that job
    /// `Unknown` — which `dispatch()`/`apply_job` faithfully copies onto the
    /// turn. Nobody ever called `retire`; Anima was still waiting on this
    /// session as "provisioning". The stall sweep, not an explicit
    /// cancellation, must be what force-retires and closes it.
    #[tokio::test]
    async fn a_turn_left_unknown_by_a_restarted_runtime_is_force_retired_with_no_explicit_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "private-lifecycle-test-credential").unwrap();
        let mut value = lifecycle_test_config(dir.path(), &token);
        // Keep the default (non-zero) grace period so forcing retirement and
        // forcing the unconfirmed cancellation are observably two different
        // passes below; the grace clock is backdated between them instead of
        // zeroed globally (a zeroed grace collapses both into one pass, which
        // is exactly what `retirement_completes_once_an_unreachable_runtime_stops_blocking_it`
        // already covers).
        // Every reconciliation pass in this test is driven by an explicit
        // `tick_session` call rather than the dispatcher's own timer, so the
        // per-session throttle must not swallow the second pass.
        value["reconciliation_interval_ms"] = json!(0);
        let config: Config = serde_json::from_value(value).unwrap();
        let principal = config.principals[0].clone();
        let app = SessionApp::new(
            Arc::new(config),
            Store::open(&dir.path().join("state.redb")).unwrap(),
        )
        .unwrap();
        let session = app
            .sessions
            .create(
                &principal,
                &NewSession {
                    request_id: "create".into(),
                    akh: "test".into(),
                    workspace_profile: "local".into(),
                    conversation_key: None,
                },
            )
            .unwrap();
        app.sessions
            .submit(
                &principal,
                &session.id,
                &SubmitInput {
                    request_id: "turn".into(),
                    text: "do work".into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        // `dispatch()` binds a workspace before ever asking for a job; mirror
        // that so `next_job` has one to dispatch against.
        app.sessions
            .bind_workspace(
                &session.id,
                WorkspaceBinding {
                    id: "workspace".into(),
                    sandbox_id: "sandbox".into(),
                    executor: "executor".into(),
                    generation: 1,
                    site: "test".into(),
                    root: "/workspaces/test".into(),
                    platform: "linux".into(),
                    confinement: "fixture".into(),
                },
            )
            .unwrap();
        let request = app.sessions.next_job(&session.id).unwrap().unwrap();
        app.sessions
            .apply_job(&RuntimeJob {
                request,
                status: TurnStatus::Unknown,
                provider_session: None,
                output: None,
                error: Some(Fault::new(
                    "unknown_outcome",
                    "runtime restarted before a durable terminal result",
                )),
                events: vec![],
            })
            .unwrap();
        // Simulate the pod-reschedule gap: no further event since long
        // before the stall threshold, and critically, no `retire` call.
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                record.last_progress_at = 0;
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let stuck = app.sessions.get(&principal, &session.id).unwrap();
        assert_eq!(stuck.turns[0].status, TurnStatus::Unknown);
        assert!(stuck.retirement.is_none());
        // First pass: retirement did not exist at the top of this call, so it
        // can only be discovered and forced here — the still-fresh grace
        // period means the Unknown turn cannot be force-cancelled yet, so
        // finalizing has to wait for a later pass.
        app.tick_session(stuck).await.unwrap();
        let mid = app.sessions.get(&principal, &session.id).unwrap();
        assert!(
            mid.retirement.is_some(),
            "no owner ever asked to retire this session; staleness alone must force it"
        );
        assert!(
            mid.closed_at.is_none(),
            "the grace period has not elapsed yet; the unconfirmed turn cannot be closed this soon"
        );
        // Simulate the grace period elapsing (a real deployment would just
        // wait; backdating keeps this test fast and deterministic).
        app.sessions
            .store
            .transaction(|tx| {
                let mut record: SessionRecord = tx.get(&key("session", &[&session.id]))?.unwrap();
                let mut retirement = record.retirement.clone().unwrap();
                retirement.at = 0;
                record.retirement = Some(retirement);
                tx.put(&key("session", &[&session.id]), &record)
            })
            .unwrap();
        let mid = app.sessions.get(&principal, &session.id).unwrap();
        // Second pass: the grace period has now elapsed, so the still-Unknown
        // turn — never confirmed cancelled by any runtime — is force-cancelled
        // locally and the session finalized.
        app.tick_session(mid).await.unwrap();
        let after = app.sessions.get(&principal, &session.id).unwrap();
        assert!(after.turns[0].status.terminal());
        assert!(
            after.closed_at.is_some(),
            "retirement must finalize even though no runtime ever confirmed this turn"
        );
    }
}
