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
    pub async fn tick(&self) -> Result<()> {
        for session in self.sessions.store.scan::<SessionRecord>("session/")? {
            if let Err(error) = self.publish_completions(&session).await {
                tracing::warn!(session=%session.id,code=%error.code,"cognitive completion delivery deferred");
            }
            if !session.turns.iter().any(|t| !t.status.terminal()) {
                continue;
            }
            if let Err(error) = self.dispatch(&session).await {
                tracing::warn!(session=%session.id,code=%error.code,"session reconciliation deferred");
                self.sessions.store.transaction(|tx| {
                    let k = key("dispatch_error", &[&session.id]);
                    if tx.get::<Fault>(&k)?.as_ref() != Some(&error) {
                        tx.put(&k, &error)?;
                        let mut current: SessionRecord =
                            tx.get(&key("session", &[&session.id]))?
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
        if session.workspace.is_none() {
            let allocation: Allocation = self
                .client
                .post(
                    "sandboxes",
                    "/v1/allocations",
                    &Allocate {
                        session_id: session.id.clone(),
                        principal: session.principal.clone(),
                        profile: session.workspace_profile.clone(),
                    },
                )
                .await?;
            if allocation.state != "ready" {
                return Ok(());
            }
            self.sessions.bind_workspace(
                &session.id,
                allocation
                    .binding
                    .ok_or_else(|| Fault::conflict("ready allocation has no workspace"))?,
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
            .route(
                "/v1/sessions/{id}/services/{helper}/actions/{action}",
                post(session_service_action),
            )
            .route("/v1/sessions/{id}/inputs", post(submit_input))
            .route("/v1/sessions/{id}/events", get(session_events))
            .route("/v1/sessions/{id}/pending", get(pending))
            .route("/v1/sessions/{id}/attach", post(attach))
            .route("/v1/sessions/{id}/detach", post(detach))
            .route("/v1/sessions/{id}/turns/{turn}/cancel", post(cancel_turn))
            .route(
                "/v1/sessions/{id}/turns/{turn}/responses/{input}",
                post(respond),
            )
            .route("/v1/automations/{id}/invoke", post(invoke))
            .route("/v1/runtime/tools", post(runtime_tool))
            .route("/v1/runtime/input", post(runtime_input))
            .with_state(app),
        config,
    )
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
    Ok(Json(app.sessions.submit(&p, &id, &request)?))
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
    Ok(Json(app.sessions.cancel(&p, &id, &turn)?))
}
async fn respond(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path((id, turn, input)): Path<(String, String, String)>,
    Query(query): Query<AttachmentQuery>,
    Json(response): Json<Value>,
) -> Api<PendingInput> {
    controller(&app, &p, &id, query.attachment_id.as_deref())?;
    Ok(Json(
        app.sessions.respond(&p, &id, &turn, &input, response)?,
    ))
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
