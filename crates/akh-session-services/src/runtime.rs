//! Durable provider jobs. A restart never repeats a possibly effective turn.
use crate::{
    config::{Config, Principal, RuntimeProfile},
    http::Client,
    store::{Store, key},
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

impl Runtime {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        store.transaction(|tx| {
            for mut job in tx.scan::<RuntimeJob>("job/")? {
                if !job.status.terminal() {
                    job.status = TurnStatus::Unknown;
                    job.error = Some(Fault::new("unknown_outcome", "runtime restarted before a durable terminal result; inspect provider continuation and execution receipts"));
                    tx.put(&job_key(&job.request.id), &job)?;
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
    pub fn get(&self, id: &str) -> Result<RuntimeJob> {
        self.store
            .get(&job_key(id))?
            .ok_or_else(|| Fault::missing("runtime job not found"))
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
        if self
            .store
            .scan::<RuntimeJob>("job/")?
            .iter()
            .any(|j| j.status == TurnStatus::Unknown)
        {
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
                    crate::internal::run(&service.store, &profile, &request, &mut host, &cancel)
                } else {
                    akh_runtime::run(&profile, &request, &mut host, &cancel)
                }
            }))
            .unwrap_or_else(|_| {
                Err("runtime worker panicked; outcome requires reconciliation".into())
            });
            let saved = service.store.transaction(|tx| {
                let mut job: RuntimeJob = tx
                    .get(&job_key(&request.id))?
                    .ok_or_else(|| Fault::missing("runtime job disappeared"))?;
                match result {
                    Ok(output) => {
                        job.status = TurnStatus::Completed;
                        job.output = Some(output);
                    }
                    Err(message) => {
                        job.status = if cancel.load(Ordering::SeqCst) {
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
                            message,
                        ));
                    }
                }
                tx.put(&job_key(&request.id), &job)
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
