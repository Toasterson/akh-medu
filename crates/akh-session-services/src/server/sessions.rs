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
#[derive(Clone)]
pub struct SessionApp {
    pub sessions: Sessions,
    pub broker: Broker,
    pub client: Client,
    pub admission_lock: Arc<std::sync::Mutex<()>>,
}
impl SessionApp {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        let sessions = Sessions::new(store, config.clone());
        Ok(Self {
            broker: Broker::new(sessions.clone())?,
            admission_lock: Default::default(),
            sessions,
            client: Client::new(config)?,
        })
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
                        let retry_after = if session.closed_at.is_some() { Duration::from_secs(5) } else { Duration::from_millis(500) };
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
                        tracing::warn!(session=%session.id,code=%error.code,"workspace lease renewal deferred")
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
        if !session.turns.iter().any(|t| !t.status.terminal()) {
            return Ok(());
        }
        if let Err(error) = self.dispatch(&session).await {
            tracing::warn!(session=%session.id,code=%error.code,"session reconciliation deferred");
            self.sessions.store.transaction(|tx| {
                let k = key("dispatch_error", &[&session.id]);
                if tx.get::<Fault>(&k)?.as_ref() != Some(&error) {
                    tx.put(&k, &error)?;
                    let mut current: SessionRecord = tx
                        .get(&key("session", &[&session.id]))?
                        .ok_or_else(|| Fault::missing("session disappeared"))?;
                    crate::session::append_event(
                        tx,
                        &mut current,
                        None,
                        "dispatch_deferred",
                        json!({"error":error}),
                    )?;
                }
                Ok(())
            })?;
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
            let result: ToolResult = self
                .client
                .post(
                    core,
                    "/v1/turns/accepted",
                    &AcceptedTurn {
                        request_id: format!("turn-{}", turn.id),
                        akh: session.akh.clone(),
                        session_id: session.id.clone(),
                        turn_id: turn.id.clone(),
                        input: turn.input.clone(),
                        output: turn.output.clone().unwrap_or_default(),
                        model: session.runtime.model.clone(),
                        tool_receipts: receipts,
                        tools,
                    },
                )
                .await?;
            if !result.success {
                return Err(Fault::new("core_error", result.text));
            }
            self.sessions.store.transaction(|tx| {
                tx.put(&k, &result)?;
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
}
