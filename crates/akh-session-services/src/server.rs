use crate::{
    broker::Broker,
    config::{Config, Principal},
    executor::{AttachWorkspace, Executor},
    http::{self, Api, ApiError, Client},
    ingress::Events,
    runtime::{CallbackReply, Runtime},
    sandbox::{Allocate, Allocation, Sandboxes},
    session::{Admission, Sessions},
    store::{Store, key},
    types::*,
};
use axum::{
    Extension, Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Duration};

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

pub fn protect(router: Router, config: Arc<Config>) -> Router {
    router.layer(DefaultBodyLimit::max(2 * 1024 * 1024)).layer(
        axum::middleware::from_fn_with_state(config, http::authorization),
    )
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

pub fn runtime_router(runtime: Runtime) -> Router {
    let config = runtime.config.clone();
    protect(
        Router::new()
            .route("/v1/jobs", post(start_job))
            .route("/v1/jobs/{id}", get(get_job))
            .route("/v1/jobs/{id}/cancel", post(cancel_job))
            .with_state(runtime),
        config,
    )
}
async fn start_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Json(request): Json<RuntimeJobRequest>,
) -> Api<RuntimeJob> {
    Ok(Json(runtime.submit(&p, request)?))
}
async fn get_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(runtime.get(&id)?))
}
async fn cancel_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(runtime.cancel(&id)?))
}

pub fn executor_router(executor: Executor) -> Router {
    let config = executor.config.clone();
    protect(
        Router::new()
            .route("/v1/services", post(launch_service))
            .route("/v1/services/{id}", get(service_status))
            .route("/v1/services/{id}/actions/{action}", post(service_action))
            .route("/v1/workspaces", post(attach_workspace))
            .route("/v1/workspaces/{id}/release", post(release_workspace))
            .route("/v1/operations", post(operation))
            .route("/v1/operations/{id}", get(receipt))
            .with_state(executor),
        config,
    )
}
async fn launch_service(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<crate::executor_helpers::LaunchService>,
) -> Api<crate::helper::Status> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.launch_service(request)?))
}
async fn service_status(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<crate::helper::Status> {
    if !p.has_role("session") && !p.has_role("sandbox") {
        http::role(&p, "admin")?;
    }
    Ok(Json(executor.service_status(&id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceAction {
    session_id: String,
}
async fn service_action(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path((id, action)): Path<(String, String)>,
    Json(request): Json<ServiceAction>,
) -> Api<Value> {
    http::role(&p, "session")?;
    let manifest = executor.service_manifest(&id)?;
    if manifest.session_id != request.session_id {
        return Err(Fault::new("forbidden", "service belongs to another session").into());
    }
    Ok(Json(crate::helper::action(&manifest, &action).await?))
}
async fn attach_workspace(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<AttachWorkspace>,
) -> Api<WorkspaceBinding> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.attach(&request)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Release {
    session_id: String,
    generation: u64,
}
async fn release_workspace(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Release>,
) -> Api<Value> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.release(
        &id,
        &request.session_id,
        request.generation,
    )?))
}
async fn operation(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<WorkspaceOperation>,
) -> Api<OperationReceipt> {
    http::role(&p, "session")?;
    Ok(Json(
        tokio::task::spawn_blocking(move || executor.operate(request))
            .await
            .map_err(|_| ApiError(Fault::new("executor", "operation worker failed")))??,
    ))
}
async fn receipt(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<OperationReceipt> {
    http::role(&p, "session")?;
    Ok(Json(executor.receipt(&id)?))
}

pub fn sandbox_router(service: Sandboxes) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/allocations", post(allocate))
            .route("/v1/allocations/{id}", get(allocation))
            .route("/v1/allocations/{id}/release", post(release_allocation))
            .with_state(service),
        config,
    )
}
async fn allocate(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Json(request): Json<Allocate>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.allocate(request).await?))
}
async fn allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.get(&id)?))
}
async fn release_allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.release(&id).await?))
}

pub fn event_router(service: Events) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/events", post(ingest))
            .route("/v1/deliveries", get(deliveries))
            .with_state(service),
        config,
    )
}
async fn ingest(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
    Json(event): Json<EventEnvelope>,
) -> Api<crate::ingress::IngressReceipt> {
    Ok(Json(service.ingest(&p, event)?))
}
async fn deliveries(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
) -> Api<Vec<crate::ingress::Delivery>> {
    http::role(&p, "admin")?;
    Ok(Json(service.store.scan("delivery/")?))
}

#[derive(clap::Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    check: bool,
}
pub async fn run(kind: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    let args = Args::parse();
    let config = Arc::new(Config::load(&args.config)?);
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    for principal in &config.principals {
        crate::config::read_token(&principal.token_file)?;
    }
    if args.check {
        println!("{kind}: configuration valid");
        return Ok(());
    }
    let store = Store::open(&config.data_dir.join(format!("{kind}.redb")))?;
    let mut tasks = vec![];
    let mut runtime_cleanup = None;
    let mut executor_cleanup = None;
    let router = match kind {
        "akhsessd" => {
            let app = SessionApp::new(config.clone(), store)?;
            let worker = app.clone();
            tasks.push(tokio::spawn(async move {
                loop {
                    if let Err(error) = worker.tick().await {
                        tracing::error!(code=%error.code,"session reconciliation failed");
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }));
            session_router(app)
        }
        "akhrtd" => {
            let runtime = Runtime::new(config.clone(), store)?;
            runtime_cleanup = Some(runtime.clone());
            runtime_router(runtime)
        }
        "akhexecd" => {
            let executor = Executor::new(config.clone(), store)?;
            executor_cleanup = Some(executor.clone());
            executor_router(executor)
        }
        "akhsbxd" => sandbox_router(Sandboxes::new(config.clone(), store)?),
        "akheventd" => {
            let service = Events::new(config.clone(), store)?;
            service.sources();
            let worker = service.clone();
            tasks.push(tokio::spawn(async move {
                loop {
                    if let Err(error) = worker.drain().await {
                        tracing::error!(code=%error.code,"event outbox drain failed");
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }));
            event_router(service)
        }
        _ => return Err("unknown service".into()),
    };
    let identity = config.service_id.clone();
    let router=router.route("/healthz",get(move||{let identity=identity.clone();async move{Json(json!({"service_id":identity,"schema_version":SCHEMA_VERSION,"status":"listening"}))}}));
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    tracing::info!(service=%config.service_id,address=%listener.local_addr()?,"service listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            shutdown().await;
        })
        .await?;
    for task in tasks {
        task.abort();
    }
    if let Some(runtime) = runtime_cleanup {
        runtime.cancel_all();
    }
    if let Some(executor) = executor_cleanup {
        executor.cancel_all();
    }
    Ok(())
}
async fn shutdown() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("install SIGTERM handler");
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
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
            let status: crate::helper::Status = client
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
