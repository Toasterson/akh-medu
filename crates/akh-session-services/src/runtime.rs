//! Durable provider jobs. A restart never repeats a possibly effective turn.
use crate::{
    config::{Config, Principal, RuntimeProfile},
    http::Client,
    store::{Store, Transaction, key},
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conformance {
    pub profile_revision: String,
    pub profile_sha256: String,
    pub provider_sha256: String,
    pub bridge_sha256: Option<String>,
    pub platform: String,
    pub provider_version: String,
    pub sdk_version: Option<String>,
    pub checks: BTreeSet<String>,
    pub evidence: Vec<String>,
    #[serde(default)]
    pub artifact_sha256: std::collections::BTreeMap<String, String>,
}

pub fn profile_digest(profile: &RuntimeProfile) -> Result<String> {
    let mut value = serde_json::to_value(profile).map_err(|e| Fault::invalid(e.to_string()))?;
    value.as_object_mut().unwrap().remove("conformance_file");
    Ok(crate::digest(
        &serde_json::to_vec(&value).map_err(|e| Fault::invalid(e.to_string()))?,
    ))
}
fn file_digest(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|_| Fault::invalid("conformance executable cannot be read"))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| Fault::invalid("cannot hash provider executable"))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub fn validate_conformance(profile: &RuntimeProfile) -> Result<()> {
    if profile.kind != RuntimeKind::Claude && !profile.native_aliases.is_empty() {
        return Err(Fault::new(
            "unsupported",
            "native aliases are implemented only by the Claude bridge",
        ));
    }
    if profile.execution_mode != ExecutionMode::Brokered || profile.kind == RuntimeKind::Internal {
        return Ok(());
    }
    let path = profile.conformance_file.as_ref().ok_or_else(|| {
        Fault::new(
            "conformance_required",
            "brokered native runtimes require installed-version conformance evidence",
        )
    })?;
    let evidence: Conformance = serde_json::from_slice(&std::fs::read(path).map_err(|_| {
        Fault::new(
            "conformance_required",
            "conformance evidence cannot be read",
        )
    })?)
    .map_err(|_| Fault::new("conformance_required", "invalid conformance document"))?;
    let required = [
        "native_write_denied_before_effect",
        "native_exec_denied_before_effect",
        "hooks_disabled",
        "supplied_tool_succeeds",
        "exact_resume",
        "cancel_teardown",
        "workspace_semantics",
    ];
    let mut artifacts = std::collections::BTreeMap::new();
    if let Some(scope) = &profile.isolation {
        for path in &scope.validation_artifacts {
            if !path.is_absolute() {
                return Err(Fault::invalid(
                    "conformance artifact paths must be absolute",
                ));
            }
            artifacts.insert(path.to_string_lossy().into_owned(), file_digest(path)?);
        }
    }
    for path in profile.node_binary.iter().chain(
        profile
            .launcher
            .first()
            .map(std::path::PathBuf::from)
            .iter(),
    ) {
        if !path.is_absolute() {
            return Err(Fault::invalid(
                "conformance requires absolute executable paths",
            ));
        }
        artifacts.insert(path.to_string_lossy().into_owned(), file_digest(path)?);
    }
    if evidence.profile_revision != profile.revision
        || evidence.artifact_sha256 != artifacts
        || evidence.profile_sha256 != profile_digest(profile)?
        || evidence.platform != std::env::consts::OS
        || evidence.provider_sha256 != file_digest(&profile.binary)?
        || evidence.bridge_sha256
            != profile
                .bridge_script
                .as_ref()
                .map(|p| file_digest(p))
                .transpose()?
        || evidence.provider_version.is_empty()
        || evidence.evidence.is_empty()
        || required.iter().any(|name| !evidence.checks.contains(*name))
    {
        return Err(Fault::new(
            "conformance_required",
            "evidence does not cover this exact runtime profile, executable, platform and required behaviors",
        ));
    }
    Ok(())
}

#[derive(Clone)]
pub struct Runtime {
    pub config: Arc<Config>,
    pub store: Store,
    client: Client,
    active: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}
fn job_key(id: &str) -> String {
    key("job", &[id])
}
fn unknown_since_key(id: &str) -> String {
    key("unknown_since", &[id])
}
/// Record the first time this job's outcome became unknown, if not already
/// recorded. Idempotent across repeated restarts so a job that is still
/// `Unknown` after several redeploys does not keep resetting its own
/// staleness clock back to "just now" (see `resolve_stale_unknown_jobs`).
fn note_unknown_since(tx: &mut Transaction<'_>, id: &str) -> Result<()> {
    if tx.get::<u64>(&unknown_since_key(id))?.is_none() {
        tx.put(&unknown_since_key(id), &now_ms())?;
    }
    Ok(())
}

impl Runtime {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        store.transaction(|tx| {
            for mut job in tx.scan::<RuntimeJob>("job/")? {
                // Already-Unknown jobs are left alone here: re-marking them
                // would be a no-op status-wise, but must never touch their
                // `unknown_since` clock, or a runtime that keeps crash-looping
                // could make its own quarantined jobs look perpetually fresh
                // and never eligible for `resolve_stale_unknown_jobs`.
                if job.request.binding.runtime == config.service_id && !job.status.terminal() && job.status != TurnStatus::Unknown {
                    job.status = TurnStatus::Unknown;
                    job.error = Some(Fault::new("unknown_outcome", "runtime restarted before a durable terminal result; inspect provider continuation and execution receipts"));
                    tx.put(&job_key(&job.request.id), &job)?;
                    note_unknown_since(tx, &job.request.id)?;
                }
            } Ok(())
        })?;
        Ok(Self {
            client: Client::new(config.clone())?,
            config,
            store,
            active: Default::default(),
        })
    }
    /// Automatic, evidence-free closure of a job whose outcome has been
    /// unknown for longer than the configured stall threshold. This never
    /// claims success — it only ever transitions `Unknown` -> `Failed` — so
    /// it cannot corrupt history the way a mistaken admin `reconcile` call
    /// claiming `Completed` could; that endpoint still exists for a human who
    /// has actual evidence. Without this sweep, one lost job (this process
    /// restarting mid-turn, or its worker panicking) permanently quarantines
    /// every future job for the same Akh: `submit` below refuses to admit any
    /// new job while any job on this runtime stays `Unknown`.
    pub fn resolve_stale_unknown_jobs(&self) -> Result<Vec<String>> {
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        let threshold = self.config.session_stall_threshold_ms;
        let now = now_ms();
        self.store.transaction(|tx| {
            let mut resolved = vec![];
            for mut job in tx.scan::<RuntimeJob>("job/")? {
                if job.request.binding.runtime != self.config.service_id || job.status != TurnStatus::Unknown || active.contains_key(&job.request.id) {
                    continue;
                }
                let Some(since) = tx.get::<u64>(&unknown_since_key(&job.request.id))? else {
                    // No recorded clock (e.g. a record from before this sweep
                    // existed): start the clock now rather than guessing.
                    note_unknown_since(tx, &job.request.id)?;
                    continue;
                };
                if now.saturating_sub(since) < threshold {
                    continue;
                }
                job.status = TurnStatus::Failed;
                job.error = Some(Fault::new(
                    "unknown_outcome",
                    "runtime automatically closed this job after its outcome stayed unknown past the reconciliation stall threshold; no completion was ever inferred",
                ));
                tx.put(&job_key(&job.request.id), &job)?;
                resolved.push(job.request.id.clone());
            }
            Ok(resolved)
        })
    }
    pub fn get(&self, id: &str) -> Result<RuntimeJob> {
        let job: RuntimeJob = self
            .store
            .get(&job_key(id))?
            .ok_or_else(|| Fault::missing("runtime job not found"))?;
        if job.request.binding.runtime != self.config.service_id {
            return Err(Fault::new(
                "forbidden",
                "runtime job belongs to another service",
            ));
        }
        Ok(job)
    }
    pub fn submit(&self, principal: &Principal, request: RuntimeJobRequest) -> Result<RuntimeJob> {
        crate::http::role(principal, "session")?;
        validate_key(&request.id)?;
        if request.binding.runtime != self.config.service_id || request.generation == 0 {
            return Err(Fault::new(
                "forbidden",
                "runtime binding is not owned by this service",
            ));
        }
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = self.store.get::<RuntimeJob>(&job_key(&request.id))? {
            if old.request != request {
                return Err(Fault::conflict("job id reused with a different dispatch"));
            }
            return Ok(old);
        }
        let profile = self
            .config
            .runtimes
            .get(&request.binding.profile)
            .ok_or_else(|| Fault::missing("runtime profile not found"))?
            .clone();
        if profile.revision != request.binding.revision
            || profile.kind != request.binding.kind
            || profile.execution_mode != request.binding.execution_mode
        {
            return Err(Fault::conflict(
                "installed runtime profile differs from session binding",
            ));
        }
        if self.store.scan::<RuntimeJob>("job/")?.iter().any(|j| {
            j.request.binding.runtime == self.config.service_id && j.status == TurnStatus::Unknown
        }) {
            return Err(Fault::new(
                "unknown_outcome",
                "runtime is quarantined until unknown jobs and possible orphan processes are reconciled",
            ));
        }
        // One provider child per daemon keeps credential files and capacity ownership unambiguous.
        // Scale runtime replicas with distinct service IDs for additional capacity.
        if !active.is_empty() {
            return Err(Fault::new("busy", "runtime capacity is occupied"));
        }
        let profile_lease = crate::runtime_profile::acquire(&profile)?;
        validate_conformance(&profile)?;
        if profile.kind == RuntimeKind::Internal {
            crate::internal::validate(&profile, &request)?;
        }
        let job = RuntimeJob {
            provider_session: request.provider_session.clone(),
            request: request.clone(),
            status: TurnStatus::Running,
            output: None,
            error: None,
            events: vec![],
        };
        self.store
            .transaction(|tx| tx.put(&job_key(&request.id), &job))?;
        let cancel = Arc::new(AtomicBool::new(false));
        active.insert(request.id.clone(), cancel.clone());
        let service = self.clone();
        let handle = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            let _profile_lease = profile_lease;
            let mut host = RemoteHost {
                service: service.clone(),
                request: request.clone(),
                handle,
                cancel: cancel.clone(),
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if profile.kind == RuntimeKind::Internal {
                    crate::internal::run_loop(&service.store, &profile, &request, &mut host, &cancel)
                } else {
                    akh_runtime::run(&profile, &request, &mut host, &cancel)
                        .map_err(|message| Fault::new("unknown_outcome", message))
                }
            }))
            .unwrap_or_else(|_| {
                Err(Fault::new("unknown_outcome", "runtime worker panicked; outcome requires reconciliation"))
            });
            let saved = service.store.transaction(|tx| {
                let mut job: RuntimeJob = tx
                    .get(&job_key(&request.id))?
                    .ok_or_else(|| Fault::missing("runtime job disappeared"))?;
                match result {
                    Ok(output) if output.len() <= 4 * 1024 * 1024 => {
                        job.status = TurnStatus::Completed;
                        job.output = Some(output);
                    }
                    Ok(_) => {
                        job.status = TurnStatus::Unknown;
                        job.error = Some(Fault::new(
                            "capacity",
                            "provider answer exceeded the durable job response budget",
                        ));
                    }
                    Err(error) if profile.kind == RuntimeKind::Internal && error.code == "model_incomplete" => {
                        // This typed error is raised before the new assistant
                        // tool plan is persisted or executed. Every preceding
                        // iteration has durable receipts. Retire continuation
                        // without claiming success or replaying earlier effects.
                        crate::internal::reconcile(tx, &job, false)?;
                        job.status = TurnStatus::Failed;
                        job.error = Some(error);
                    }
                    Err(error) => {
                        // An internal model can stop between a persisted tool plan and
                        // its receipt. Cancellation alone cannot close that history.
                        job.status = if cancel.load(Ordering::SeqCst)
                            && profile.kind != RuntimeKind::Internal
                        {
                            TurnStatus::Cancelled
                        } else {
                            TurnStatus::Unknown
                        };
                        job.error = Some(Fault::new(
                            if job.status == TurnStatus::Unknown {
                                "unknown_outcome"
                            } else {
                                "cancelled"
                            },
                            error.to_string(),
                        ));
                    }
                }
                tx.put(&job_key(&request.id), &job)?;
                if job.status == TurnStatus::Unknown {
                    note_unknown_since(tx, &request.id)?;
                }
                Ok(())
            });
            if let Err(error) = saved {
                tracing::error!(job=%request.id, code=%error.code, "failed to save runtime result");
            }
            service
                .active
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&request.id);
        });
        Ok(job)
    }
    pub fn cancel(&self, id: &str) -> Result<RuntimeJob> {
        let job = self.get(id)?;
        if let Some(flag) = self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
        {
            flag.store(true, Ordering::SeqCst);
        }
        Ok(job)
    }
    pub fn cancel_all(&self) {
        for flag in self
            .active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            flag.store(true, Ordering::SeqCst);
        }
    }
    pub fn reconcile(
        &self,
        principal: &Principal,
        id: &str,
        resolution: &Reconciliation,
    ) -> Result<RuntimeJob> {
        crate::http::role(principal, "admin")?;
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if active.contains_key(id) {
            return Err(Fault::new("busy", "runtime worker is still active"));
        }
        self.store.transaction(|tx| {
            let target=job_key(id);
            let mut job:RuntimeJob=tx.get(&target)?.ok_or_else(||Fault::missing("runtime job not found"))?;
            if job.request.binding.runtime != self.config.service_id {
                return Err(Fault::new("forbidden", "runtime job belongs to another service"));
            }
            crate::recovery::validate(principal,&job.request,resolution)?;
            if crate::recovery::already(tx,principal,&target,resolution)? { return Ok(job); }
            if job.status!=TurnStatus::Unknown { return Err(Fault::conflict("only an unknown job can be reconciled")); }
            if resolution.result.is_some() || (resolution.status==TurnStatus::Completed && resolution.output.is_none()) {
                return Err(Fault::invalid("runtime completion requires output; tool receipts must be reconciled at their owner"));
            }
            if job.request.binding.kind==RuntimeKind::Internal {
                crate::internal::reconcile(tx,&job,resolution.status==TurnStatus::Completed)?;
            }
            job.status=resolution.status;
            job.output=resolution.output.clone();
            job.error=if job.status==TurnStatus::Completed {None} else {Some(Fault::new("operator_resolved","operator resolved the unknown turn; see reconciliation evidence"))};
            tx.put(&target,&job)?;
            crate::recovery::record(tx,principal,&target,resolution)?;
            Ok(job)
        })
    }
}

pub use akh_session_types::CallbackReply;

struct RemoteHost {
    service: Runtime,
    request: RuntimeJobRequest,
    handle: tokio::runtime::Handle,
    cancel: Arc<AtomicBool>,
}
impl RemoteHost {
    fn call(&self, id: &str, name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            session_id: self.request.session_id.clone(),
            turn_id: self.request.turn_id.clone(),
            generation: self.request.generation,
            name: name.into(),
            arguments,
        }
    }
    fn callback(&self, path: &str, body: Value) -> std::result::Result<CallbackReply, String> {
        let mut failures = 0;
        loop {
            if self.cancel.load(Ordering::SeqCst) {
                return Err("runtime cancelled".into());
            }
            let result: Result<CallbackReply> = self
                .handle
                .block_on(self.service.client.post("sessions", path, &body));
            match result {
                Ok(reply) if !reply.pending => return Ok(reply),
                Ok(_) => {
                    failures = 0;
                }
                Err(error) if error.code == "transport" || error.code == "busy" => {
                    failures += 1;
                    if failures >= 10 {
                        return Err(error.to_string());
                    }
                }
                Err(error) => return Err(error.to_string()),
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }
}
impl akh_runtime::Host for RemoteHost {
    fn event(&mut self, event: RuntimeEvent) -> std::result::Result<(), String> {
        self.service
            .store
            .transaction(|tx| {
                let mut job: RuntimeJob = tx
                    .get(&job_key(&self.request.id))?
                    .ok_or_else(|| Fault::missing("runtime job not found"))?;
                if let RuntimeEvent::Bound {
                    provider_session, ..
                } = &event
                {
                    if job
                        .provider_session
                        .as_ref()
                        .is_some_and(|s| s != provider_session)
                    {
                        return Err(Fault::conflict("provider continuation changed"));
                    }
                    job.provider_session = Some(provider_session.clone());
                }
                if job.events.len() >= 100_000 {
                    return Err(Fault::new("capacity", "runtime event limit exceeded"));
                }
                let bytes_key = key("runtime_event_bytes", &[&self.request.id]);
                let old_bytes = tx.get::<usize>(&bytes_key)?.unwrap_or_else(|| {
                    serde_json::to_vec(&job.events)
                        .map(|b| b.len())
                        .unwrap_or(4 * 1024 * 1024)
                });
                let event_bytes = serde_json::to_vec(&event)
                    .map_err(|_| Fault::invalid("cannot encode runtime event"))?
                    .len();
                if old_bytes + event_bytes > 4 * 1024 * 1024 {
                    return Err(Fault::new("capacity", "runtime event byte budget exceeded"));
                }
                tx.put(&bytes_key, &(old_bytes + event_bytes))?;
                job.events.push(event);
                tx.put(&job_key(&self.request.id), &job)
            })
            .map_err(|e| e.to_string())
    }
    fn tool(
        &mut self,
        id: &str,
        name: &str,
        arguments: Value,
    ) -> std::result::Result<ToolResult, String> {
        self.callback("/v1/runtime/tools", json!(self.call(id, name, arguments)))?
            .result
            .ok_or_else(|| "tool callback has no result".into())
    }
    fn approval(
        &mut self,
        id: &str,
        name: &str,
        arguments: &Value,
    ) -> std::result::Result<bool, String> {
        let reply = self.callback("/v1/runtime/input", json!({"call":self.call(id, name, arguments.clone()),"kind":"native_approval","prompt":{"name":name,"arguments":arguments}}))?;
        Ok(reply
            .response
            .as_ref()
            .is_some_and(|r| r["approved"] == true))
    }
    fn question(&mut self, id: &str, questions: &Value) -> std::result::Result<Value, String> {
        self.callback("/v1/runtime/input", json!({"call":self.call(id, "question", questions.clone()),"kind":"question","prompt":questions}))?.response.ok_or_else(||"question has no answer".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn incomplete_model_response_fails_without_quarantining_next_job() {
        use axum::{Router, routing::post};
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = Router::new().route("/", post(move || {
            let nth = counter.fetch_add(1, Ordering::SeqCst);
            let choice = if nth == 1 {
                json!({"index":0,"delta":{"tool_calls":[{"index":0,"id":"bad","function":{"name":"read","arguments":"{broken"}}]},"finish_reason":"tool_calls"})
            } else { json!({"index":0,"delta":{"content":"partial"},"finish_reason":"length"}) };
            let body = format!("data: {}\n\ndata: [DONE]\n\n", json!({"choices":[choice]}));
            async move { ([("content-type", "text/event-stream")], body) }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = (*config(dir.path())).clone();
        cfg.runtimes.insert("local".into(), serde_json::from_value(json!({
            "revision":"1","kind":"internal","working_dir":dir.path(),"binary":"/unused",
            "execution_mode":"brokered","timeout_secs":5,"isolation":{"directory":dir.path().join("profile")},
            "model_endpoint":{"url":url,"max_iterations":2,"max_output_tokens":128}
        })).unwrap());
        let principal = cfg.principals[0].clone();
        let runtime = Runtime::new(Arc::new(cfg), Store::open(&dir.path().join("runtime.redb")).unwrap()).unwrap();
        for id in ["limited-one", "malformed-plan", "limited-two"] {
            let mut req = request(id);
            req.binding.profile = "local".into();
            req.binding.revision = "1".into();
            req.binding.kind = RuntimeKind::Internal;
            req.binding.execution_mode = ExecutionMode::Brokered;
            req.binding.model = Some("fixture".into());
            runtime.submit(&principal, req).unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if runtime.active.lock().unwrap().is_empty() { break; }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }).await.unwrap();
            let job = runtime.get(id).unwrap();
            assert_eq!(job.status, TurnStatus::Failed);
            assert_eq!(job.error.unwrap().code, "model_incomplete");
        }
        server.abort();
    }

    fn config(dir: &std::path::Path) -> Arc<Config> {
        Arc::new(
            serde_json::from_value(json!({
                "service_id": "runtime-1",
                "listen": "127.0.0.1:0",
                "data_dir": dir,
                "principals": [{"id": "sessions", "token_file": "/not-read-by-runtime-tests", "roles": ["session"], "akhs": []}],
                "session_stall_threshold_ms": 1_000
            }))
            .unwrap(),
        )
    }

    fn request(id: &str) -> RuntimeJobRequest {
        RuntimeJobRequest {
            id: id.into(),
            session_id: "session-1".into(),
            turn_id: "turn-1".into(),
            generation: 1,
            binding: RuntimeBinding {
                runtime: "runtime-1".into(),
                profile: "claude".into(),
                revision: "1".into(),
                kind: RuntimeKind::Claude,
                model: None,
                execution_mode: ExecutionMode::Brokered,
            },
            workspace: WorkspaceBinding {
                id: "workspace".into(),
                sandbox_id: "sandbox".into(),
                executor: "executor".into(),
                generation: 1,
                site: "test".into(),
                root: "/workspaces/test".into(),
                platform: "linux".into(),
                confinement: "fixture".into(),
            },
            provider_session: None,
            input: TurnInput {
                origin: InputOrigin::Human {
                    principal: "user".into(),
                },
                text: "hello".into(),
                data: None,
                artifacts: vec![],
            },
            instructions: "be useful".into(),
            tools: vec![],
            approval_policy: ApprovalPolicy::AskWrites,
        }
    }

    fn unknown_job(id: &str) -> RuntimeJob {
        RuntimeJob {
            request: request(id),
            status: TurnStatus::Unknown,
            provider_session: None,
            output: None,
            error: None,
            events: vec![],
        }
    }

    #[test]
    fn shared_store_recovery_never_touches_another_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("shared.redb")).unwrap();
        let mut other = unknown_job("foreign");
        other.request.binding.runtime = "runtime-2".into();
        other.status = TurnStatus::Running;
        store
            .transaction(|tx| tx.put(&job_key("foreign"), &other))
            .unwrap();
        let runtime = Runtime::new(config(dir.path()), store.clone()).unwrap();
        assert_eq!(
            store
                .get::<RuntimeJob>(&job_key("foreign"))
                .unwrap()
                .unwrap()
                .status,
            TurnStatus::Running
        );
        assert_eq!(runtime.get("foreign").unwrap_err().code, "forbidden");
        assert_eq!(runtime.cancel("foreign").unwrap_err().code, "forbidden");
        other.status = TurnStatus::Unknown;
        store
            .transaction(|tx| {
                tx.put(&job_key("foreign"), &other)?;
                tx.put(&unknown_since_key("foreign"), &0u64)
            })
            .unwrap();
        assert!(runtime.resolve_stale_unknown_jobs().unwrap().is_empty());
        assert_eq!(
            store
                .get::<RuntimeJob>(&job_key("foreign"))
                .unwrap()
                .unwrap()
                .status,
            TurnStatus::Unknown
        );
    }

    /// The exact incident this sweep exists for: a job left `Running` when the
    /// process died is marked `Unknown` at the next `Runtime::new`, and stays
    /// that way — quarantining the whole runtime via `submit`'s check — until
    /// this sweep closes it out, well past the stall threshold.
    #[test]
    fn restart_marks_running_jobs_unknown_and_the_sweep_leaves_them_alone_before_the_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        store
            .transaction(|tx| {
                tx.put(
                    &job_key("job-1"),
                    &RuntimeJob {
                        request: request("job-1"),
                        status: TurnStatus::Running,
                        provider_session: None,
                        output: None,
                        error: None,
                        events: vec![],
                    },
                )
            })
            .unwrap();
        let runtime = Runtime::new(cfg, store).unwrap();
        assert_eq!(runtime.get("job-1").unwrap().status, TurnStatus::Unknown);
        assert!(
            runtime.resolve_stale_unknown_jobs().unwrap().is_empty(),
            "an unknown job younger than the stall threshold must not be force-closed"
        );
        assert_eq!(runtime.get("job-1").unwrap().status, TurnStatus::Unknown);
    }

    #[test]
    fn sweep_closes_a_job_unknown_past_the_stall_threshold_and_only_that_one() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        store
            .transaction(|tx| {
                tx.put(&job_key("stale"), &unknown_job("stale"))?;
                tx.put(&unknown_since_key("stale"), &(now_ms() - 60_000))?;
                tx.put(&job_key("fresh"), &unknown_job("fresh"))?;
                tx.put(&unknown_since_key("fresh"), &now_ms())
            })
            .unwrap();
        let runtime = Runtime::new(cfg, store).unwrap();
        let resolved = runtime.resolve_stale_unknown_jobs().unwrap();
        assert_eq!(resolved, vec!["stale".to_string()]);
        assert_eq!(runtime.get("stale").unwrap().status, TurnStatus::Failed);
        assert_eq!(runtime.get("fresh").unwrap().status, TurnStatus::Unknown);
    }

    #[test]
    fn sweep_never_closes_a_job_whose_worker_is_still_active() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        store
            .transaction(|tx| {
                tx.put(&job_key("live"), &unknown_job("live"))?;
                tx.put(&unknown_since_key("live"), &(now_ms() - 60_000))
            })
            .unwrap();
        let runtime = Runtime::new(cfg, store).unwrap();
        runtime
            .active
            .lock()
            .unwrap()
            .insert("live".into(), Arc::new(AtomicBool::new(false)));
        assert!(runtime.resolve_stale_unknown_jobs().unwrap().is_empty());
        assert_eq!(runtime.get("live").unwrap().status, TurnStatus::Unknown);
    }

    /// A crash-looping runtime must not reset its own quarantined jobs' clocks
    /// back to "just now" on every restart, or the stale-job sweep could never
    /// catch up and the quarantine (see `submit`) would never clear.
    #[test]
    fn repeated_restarts_do_not_reset_the_unknown_clock() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        store
            .transaction(|tx| {
                tx.put(&job_key("job-1"), &unknown_job("job-1"))?;
                tx.put(&unknown_since_key("job-1"), &(now_ms() - 60_000))
            })
            .unwrap();
        let store = {
            // Simulate a second restart before reconciliation ever runs.
            drop(Runtime::new(cfg.clone(), store).unwrap());
            Store::open(&dir.path().join("state.redb")).unwrap()
        };
        let runtime = Runtime::new(cfg, store).unwrap();
        let resolved = runtime.resolve_stale_unknown_jobs().unwrap();
        assert_eq!(
            resolved,
            vec!["job-1".to_string()],
            "the original unknown_since must survive repeated restarts"
        );
    }
}
