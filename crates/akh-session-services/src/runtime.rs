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
/// A job moves here, out of the scanned `job/` prefix, the moment it becomes
/// terminal (Completed/Failed/Cancelled). `submit`'s quarantine scan and
/// `resolve_stale_unknown_jobs` only ever need to see live (Running/Unknown)
/// jobs, so once a job is done, leaving its up-to-4-MiB `output` and
/// up-to-4-MiB `events` sitting in `job/` costs a full deserialize of both on
/// every scan for no reason — see `job_key`'s doc and the commit this
/// accompanies.
fn archived_job_key(id: &str) -> String {
    key("archived_job", &[id])
}
fn event_bytes_key(id: &str) -> String {
    key("runtime_event_bytes", &[id])
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

/// Which prefix a job record currently lives in. Returned by `load_job` so a
/// caller can write back to the same prefix with `save_job` instead of
/// guessing — a late write (e.g. an event appended by a worker racing a
/// reconciliation) must never resurrect an archived job into `job/`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum JobSlot {
    Live,
    Archived,
}

/// Reads a job record by id inside a transaction, checking the live `job/`
/// prefix first and falling back to `archived_job/` for jobs whose terminal
/// outcome has already been archived out of the scanned prefix.
fn load_job(tx: &Transaction<'_>, id: &str) -> Result<Option<(RuntimeJob, JobSlot)>> {
    if let Some(job) = tx.get::<RuntimeJob>(&job_key(id))? {
        return Ok(Some((job, JobSlot::Live)));
    }
    if let Some(job) = tx.get::<RuntimeJob>(&archived_job_key(id))? {
        return Ok(Some((job, JobSlot::Archived)));
    }
    Ok(None)
}

/// Non-transactional twin of `load_job`, for reads outside a transaction
/// (e.g. `Runtime::get` and the idempotent-resubmit check in `submit`).
fn load_job_record(store: &Store, id: &str) -> Result<Option<RuntimeJob>> {
    if let Some(job) = store.get::<RuntimeJob>(&job_key(id))? {
        return Ok(Some(job));
    }
    store.get(&archived_job_key(id))
}

/// Writes a job record back to whichever prefix `load_job` found it in, so a
/// late write can never resurrect an archived job into the scanned `job/`
/// prefix. Does not change slots — use `archive_job` for the terminal
/// transition itself.
fn save_job(tx: &mut Transaction<'_>, job: &RuntimeJob, slot: JobSlot) -> Result<()> {
    match slot {
        JobSlot::Live => tx.put(&job_key(&job.request.id), job),
        JobSlot::Archived => tx.put(&archived_job_key(&job.request.id), job),
    }
}

/// Moves a now-terminal job from `job/` to `archived_job/` in one
/// transaction: put at the archived key, delete the live key, and delete the
/// per-job side keys (`unknown_since`, `runtime_event_bytes`) that only ever
/// matter while the job is live. Callers must only reach this once
/// `job.status.terminal()` is true.
fn archive_job(tx: &mut Transaction<'_>, job: &RuntimeJob) -> Result<()> {
    tx.put(&archived_job_key(&job.request.id), job)?;
    tx.delete(&job_key(&job.request.id))?;
    tx.delete(&unknown_since_key(&job.request.id))?;
    tx.delete(&event_bytes_key(&job.request.id))?;
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
            }
            // One-time migration for records written by a binary from before
            // job archival existed: a terminal job left in `job/` by that
            // older code is otherwise never archived (nothing here re-visits
            // an already-terminal job), so it would keep being deserialized
            // by every future scan forever.
            let mut migrated = 0usize;
            for job in tx.scan::<RuntimeJob>("job/")? {
                if job.request.binding.runtime == config.service_id && job.status.terminal() {
                    archive_job(tx, &job)?;
                    migrated += 1;
                }
            }
            if migrated > 0 {
                tracing::info!(
                    count = migrated,
                    "archived pre-existing terminal jobs out of the scanned job/ prefix"
                );
            }
            Ok(())
        })?;
        Ok(Self {
            client: Client::new(config.clone())?,
            config,
            store,
            active: Default::default(),
        })
    }
    /// Legacy native-runtime expiry. Internal brokered jobs are excluded:
    /// elapsed time cannot establish whether an uncertain tool had effects.
    /// Their owning profile stays quarantined until evidence reconciles it.
    pub fn resolve_stale_unknown_jobs(&self) -> Result<Vec<String>> {
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        let threshold = self.config.session_stall_threshold_ms;
        let now = now_ms();
        self.store.transaction(|tx| {
            let mut resolved = vec![];
            for mut job in tx.scan::<RuntimeJob>("job/")? {
                if job.request.binding.runtime != self.config.service_id || job.status != TurnStatus::Unknown || active.contains_key(&job.request.id) || job.request.binding.kind == RuntimeKind::Internal {
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
                archive_job(tx, &job)?;
                resolved.push(job.request.id.clone());
            }
            Ok(resolved)
        })
    }
    pub fn get(&self, id: &str) -> Result<RuntimeJob> {
        let job = load_job_record(&self.store, id)?
            .ok_or_else(|| Fault::missing("runtime job not found"))?;
        if job.request.binding.runtime != self.config.service_id {
            return Err(Fault::new(
                "forbidden",
                "runtime job belongs to another service",
            ));
        }
        Ok(job)
    }
    /// Atomically fence a dispatch that was cancelled before admission. A late
    /// submit observes this exact tombstone and cannot start a model or tool.
    pub fn cancel_unsubmitted(
        &self,
        principal: &Principal,
        request: RuntimeJobRequest,
    ) -> Result<RuntimeJob> {
        crate::http::role(principal, "session")?;
        validate_key(&request.id)?;
        if request.binding.runtime != self.config.service_id || request.generation == 0 {
            return Err(Fault::new(
                "forbidden",
                "runtime binding is not owned by this service",
            ));
        }
        let _active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        self.store.transaction(|tx| {
            if let Some(old) = tx.get::<RuntimeJob>(&job_key(&request.id))? {
                if old.request != request {
                    return Err(Fault::conflict("job id reused with a different dispatch"));
                }
                return Ok(old);
            }
            let job = RuntimeJob {
                provider_session: request.provider_session.clone(),
                request,
                status: TurnStatus::Cancelled,
                output: None,
                error: Some(Fault::new(
                    "cancelled",
                    "cancelled before runtime admission",
                )),
                events: vec![],
            };
            tx.put(&job_key(&job.request.id), &job)?;
            Ok(job)
        })
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
        if let Some(old) = load_job_record(&self.store, &request.id)? {
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
            j.request.binding.runtime == self.config.service_id
                && j.status == TurnStatus::Unknown
                && (j.request.binding.kind != RuntimeKind::Internal
                    || profile.kind != RuntimeKind::Internal
                    || j.request.binding.profile == request.binding.profile
                    || j.request.session_id == request.session_id)
        }) {
            return Err(Fault::new(
                "unknown_outcome",
                "runtime is quarantined until unknown jobs and possible orphan processes are reconciled",
            ));
        }
        // One provider job per daemon keeps credentials and compute capacity ownership unambiguous.
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
            let span = tracing::info_span!("runtime.turn", session.id=%request.session_id,
                turn.id=%request.turn_id, job.id=%request.id, runtime.id=%request.binding.runtime, runtime.profile=%request.binding.profile,
                model=%request.binding.model.as_deref().unwrap_or("unknown"), generation=request.generation,
                otel.status_code=tracing::field::Empty);
            let _entered = span.enter();
            let mut operation = crate::telemetry::Operation::new("turn");
            tracing::info!(event.name = "runtime.turn.started", "runtime turn started");
            let _profile_lease = profile_lease;
            let mut host = RemoteHost {
                service: service.clone(),
                request: request.clone(),
                handle,
                cancel: cancel.clone(),
                native_tools: HashMap::new(),
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if profile.kind == RuntimeKind::Internal {
                    crate::internal::run_loop(
                        &service.store,
                        &profile,
                        &request,
                        &mut host,
                        &cancel,
                    )
                } else {
                    akh_runtime::run(&profile, &request, &mut host, &cancel)
                        .map_err(|message| Fault::new("unknown_outcome", message))
                }
            }))
            .unwrap_or_else(|_| {
                Err(Fault::new(
                    "unknown_outcome",
                    "runtime worker panicked; outcome requires reconciliation",
                ))
            });
            if result.is_ok() {
                operation.success();
            } else {
                span.record("otel.status_code", "ERROR");
                tracing::error!(
                    event.name = "runtime.turn.failed",
                    "runtime turn failed; see provider and tool events"
                );
            }
            let saved = service.store.transaction(|tx| {
                let (mut job, slot) = load_job(tx, &request.id)?
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
                    Err(error)
                        if profile.kind == RuntimeKind::Internal
                            && matches!(
                                error.code.as_str(),
                                "model_incomplete" | "model_cancelled"
                            ) =>
                    {
                        // These typed errors occur only at model boundaries,
                        // with no callback outstanding and preceding tool
                        // results durable. Retire the continuation
                        // without claiming success or replaying earlier effects.
                        crate::internal::reconcile(tx, &job, false)?;
                        job.status = if error.code == "model_cancelled" {
                            TurnStatus::Cancelled
                        } else {
                            TurnStatus::Failed
                        };
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
                if job.status.terminal() {
                    archive_job(tx, &job)?;
                } else {
                    save_job(tx, &job, slot)?;
                    if job.status == TurnStatus::Unknown {
                        note_unknown_since(tx, &request.id)?;
                    }
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
            // `target` is the reconciliation audit's identity, not a storage
            // key to read/write today: it must stay the `job_key(id)` string
            // even after this job is archived, or a second identical
            // reconcile call (whose `already` check compares `old.target !=
            // target`) would see a mismatch and fault instead of returning
            // the idempotent result.
            let target=job_key(id);
            let Some((mut job, _slot)) = load_job(tx, id)? else {
                return Err(Fault::missing("runtime job not found"));
            };
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
            // `recovery::validate` already required `resolution.status.terminal()`.
            archive_job(tx, &job)?;
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
    native_tools: HashMap<String, (tracing::Span, crate::telemetry::Operation)>,
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
    /// One bounded exchange with the session daemon. `Ok(None)` when the daemon
    /// predates the route (HTTP 404 without a fault body), so newer runtimes
    /// degrade to the old behaviour instead of failing the turn.
    fn post_optional<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: Value,
    ) -> std::result::Result<Option<T>, String> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err("runtime cancelled".into());
        }
        match self
            .handle
            .block_on(self.service.client.post::<T>("sessions", path, &body))
        {
            Ok(reply) => Ok(Some(reply)),
            Err(error)
                if error.code == "not_found" && error.message.contains("returned HTTP 404") =>
            {
                Ok(None)
            }
            Err(error) => Err(error.to_string()),
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
    fn usage(&mut self, usage: &Value) {
        crate::telemetry::usage(usage);
    }
    fn event(&mut self, event: RuntimeEvent) -> std::result::Result<(), String> {
        match &event {
            RuntimeEvent::Bound {
                provider_session,
                model,
            } => {
                tracing::info!(event.name="runtime.bound", provider.session_id=%provider_session, model=?model, "provider session bound")
            }
            RuntimeEvent::Text { text } => crate::telemetry::delta("text", text),
            RuntimeEvent::Reasoning { text } => crate::telemetry::delta("reasoning", text),
            RuntimeEvent::ToolStarted { id, name } => {
                let span = tracing::info_span!("runtime.tool", tool.id=%id, tool.name=%name, otel.status_code=tracing::field::Empty);
                span.in_scope(|| {
                    tracing::info!(event.name = "runtime.tool.started", "native tool started")
                });
                self.native_tools
                    .insert(id.clone(), (span, crate::telemetry::Operation::new("tool")));
            }
            RuntimeEvent::ToolCompleted { id, success, .. } => {
                if let Some((span, mut operation)) = self.native_tools.remove(id) {
                    let _entered = span.enter();
                    if *success {
                        operation.success();
                    } else {
                        span.record("otel.status_code", "ERROR");
                    }
                    tracing::info!(event.name="runtime.tool.completed", tool.id=%id, success, "native tool completed");
                    drop(operation);
                }
            }
        }
        self.service
            .store
            .transaction(|tx| {
                // A job is only ever Running while its own worker can still
                // call back here, so this always finds it in `job/` — but it
                // is routed through the same fallback as every other by-id
                // read so this call site never needs to change if that ever
                // stops being true.
                let (mut job, slot) = load_job(tx, &self.request.id)?
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
                save_job(tx, &job, slot)
            })
            .map_err(|e| e.to_string())
    }
    fn tool(
        &mut self,
        id: &str,
        name: &str,
        arguments: Value,
    ) -> std::result::Result<ToolResult, String> {
        self.event(RuntimeEvent::ToolStarted {
            id: id.into(),
            name: name.into(),
        })?;
        let result = self
            .callback("/v1/runtime/tools", json!(self.call(id, name, arguments)))?
            .result
            .ok_or_else(|| "tool callback has no result".to_owned())?;
        self.event(RuntimeEvent::ToolCompleted {
            id: id.into(),
            success: result.success,
            details: Some(json!({"receipt":result.receipt,"text":result.text})),
        })?;
        Ok(result)
    }
    fn approval(
        &mut self,
        id: &str,
        name: &str,
        arguments: &Value,
    ) -> std::result::Result<bool, String> {
        let span = tracing::info_span!("runtime.approval", tool.id = id, tool.name = name);
        let _entered = span.enter();
        let mut operation = crate::telemetry::Operation::new("approval_wait");
        tracing::info!(
            event.name = "runtime.approval.waiting",
            "waiting for approval"
        );
        let reply = self.callback("/v1/runtime/input", json!({"call":self.call(id, name, arguments.clone()),"kind":"native_approval","prompt":{"name":name,"arguments":arguments}}))?;
        operation.success();
        Ok(reply
            .response
            .as_ref()
            .is_some_and(|r| r["approved"] == true))
    }
    fn question(&mut self, id: &str, questions: &Value) -> std::result::Result<Value, String> {
        let span = tracing::info_span!("runtime.question", question.id = id);
        let _entered = span.enter();
        let mut operation = crate::telemetry::Operation::new("question_wait");
        tracing::info!(event.name = "runtime.question.waiting", "waiting for input");
        let result = self.callback("/v1/runtime/input", json!({"call":self.call(id, "question", questions.clone()),"kind":"question","prompt":questions}))?.response.ok_or_else(||"question has no answer".into());
        if result.is_ok() {
            operation.success();
        }
        result
    }
    fn recall(
        &mut self,
        id: &str,
        space: &neural_spans::SpanSpace,
        vector: &[u8],
        limit: usize,
    ) -> std::result::Result<Option<Vec<neural_spans::SpanHit>>, String> {
        let request = neural_spans::RecallRequest {
            call: self.call(id, neural_spans::RecallRequest::CALL_NAME, json!({})),
            space: space.clone(),
            vector: vector.to_vec(),
            limit,
        };
        Ok(self
            .post_optional::<neural_spans::RecallReply>("/v1/runtime/recall", json!(request))?
            .filter(|reply| reply.supported)
            .map(|reply| reply.hits))
    }
    fn spans(
        &mut self,
        id: &str,
        captures: &[neural_spans::SpanCapture],
    ) -> std::result::Result<bool, String> {
        let delivery = neural_spans::SpanDelivery {
            call: self.call(id, neural_spans::SpanDelivery::CALL_NAME, json!({})),
            captures: captures.to_vec(),
        };
        Ok(self
            .post_optional::<neural_spans::SpanDeliveryReply>("/v1/runtime/spans", json!(delivery))?
            .is_some_and(|reply| reply.accepted))
    }
    fn assemble_context(
        &mut self,
        messages: &[Value],
        window_tokens: usize,
        reserved_output_tokens: usize,
    ) -> std::result::Result<Option<akh_session_types::AssembledContext>, String> {
        let request = akh_session_types::RuntimeContextRequest {
            call: self.call(
                &self.request.turn_id,
                akh_session_types::RuntimeContextRequest::CALL_NAME,
                json!({}),
            ),
            input: self.request.input.clone(),
            messages: messages.to_vec(),
            budget: akh_session_types::ContextBudget {
                window_tokens,
                reserved_output_tokens,
            },
        };
        Ok(self
            .post_optional::<akh_session_types::RuntimeContextReply>(
                "/v1/runtime/context",
                json!(request),
            )?
            .and_then(|reply| reply.assembled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelled_unsubmitted_dispatch_cannot_run_later() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let principal = cfg.principals[0].clone();
        let runtime =
            Runtime::new(cfg, Store::open(&dir.path().join("runtime.redb")).unwrap()).unwrap();
        let req = request("cancelled-before-admission");
        let job = runtime.cancel_unsubmitted(&principal, req.clone()).unwrap();
        assert_eq!(job.status, TurnStatus::Cancelled);
        assert_eq!(runtime.submit(&principal, req.clone()).unwrap(), job);
        let mut changed = req;
        changed.input.text = "different".into();
        assert!(runtime.submit(&principal, changed).is_err());
        assert!(runtime.active.lock().unwrap().is_empty());
    }

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
            async move {
                if nth == 3 { tokio::time::sleep(Duration::from_secs(30)).await; }
                ([("content-type", "text/event-stream")], body)
            }
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
        let runtime = Runtime::new(
            Arc::new(cfg),
            Store::open(&dir.path().join("runtime.redb")).unwrap(),
        )
        .unwrap();
        // An uncertain callback in another provisioned internal profile must
        // remain uncertain without quarantining this independent role.
        let mut uncertain = unknown_job("other-role");
        uncertain.request.binding.kind = RuntimeKind::Internal;
        uncertain.request.binding.profile = "other-role".into();
        uncertain.request.session_id = "other-session".into();
        runtime
            .store
            .transaction(|tx| {
                tx.put(&job_key("other-role"), &uncertain)?;
                tx.put(&unknown_since_key("other-role"), &0u64)
            })
            .unwrap();
        assert!(runtime.resolve_stale_unknown_jobs().unwrap().is_empty());
        assert_eq!(
            runtime.get("other-role").unwrap().status,
            TurnStatus::Unknown
        );
        for id in [
            "limited-one",
            "malformed-plan",
            "limited-two",
            "cancel-model",
        ] {
            let mut req = request(id);
            req.binding.profile = "local".into();
            req.binding.revision = "1".into();
            req.binding.kind = RuntimeKind::Internal;
            req.binding.execution_mode = ExecutionMode::Brokered;
            req.binding.model = Some("fixture".into());
            runtime.submit(&principal, req).unwrap();
            if id == "cancel-model" {
                tokio::time::timeout(Duration::from_secs(5), async {
                    while runtime.get(id).unwrap().events.is_empty() {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .unwrap();
                runtime.cancel(id).unwrap();
            }
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if runtime.active.lock().unwrap().is_empty() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            let job = runtime.get(id).unwrap();
            if id == "cancel-model" {
                assert_eq!(job.status, TurnStatus::Cancelled);
                assert_eq!(job.error.unwrap().code, "model_cancelled");
            } else {
                assert_eq!(job.status, TurnStatus::Failed);
                assert_eq!(job.error.unwrap().code, "model_incomplete");
            }
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
    /// this sweep closes it out, well past the stall threshold. The same
    /// restart also carries a one-time migration: a terminal job written by a
    /// binary from before job archival existed must be swept into
    /// `archived_job/` on this first restart, while the freshly-Unknown job
    /// and the still-Running-turned-Unknown job stay in the scanned `job/`
    /// prefix untouched.
    #[test]
    fn restart_marks_running_jobs_unknown_and_the_sweep_leaves_them_alone_before_the_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        let mut pre_existing_completed = unknown_job("done-before-restart");
        pre_existing_completed.status = TurnStatus::Completed;
        pre_existing_completed.output = Some("already finished".into());
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
                )?;
                // Simulates a record left by a binary predating job archival:
                // already terminal, still sitting in the scanned prefix.
                tx.put(&job_key("done-before-restart"), &pre_existing_completed)
            })
            .unwrap();
        let runtime = Runtime::new(cfg, store.clone()).unwrap();
        assert_eq!(runtime.get("job-1").unwrap().status, TurnStatus::Unknown);
        assert!(
            runtime.resolve_stale_unknown_jobs().unwrap().is_empty(),
            "an unknown job younger than the stall threshold must not be force-closed"
        );
        assert_eq!(runtime.get("job-1").unwrap().status, TurnStatus::Unknown);
        let live = store.scan::<RuntimeJob>("job/").unwrap();
        assert_eq!(
            live.len(),
            1,
            "only the still-live job-1 may remain in the scanned job/ prefix"
        );
        assert_eq!(live[0].request.id, "job-1");
        assert_eq!(
            runtime.get("done-before-restart").unwrap().status,
            TurnStatus::Completed,
            "the pre-existing terminal job must still be reachable by id after migration"
        );
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
        let runtime = Runtime::new(cfg, store.clone()).unwrap();
        let resolved = runtime.resolve_stale_unknown_jobs().unwrap();
        assert_eq!(resolved, vec!["stale".to_string()]);
        assert_eq!(runtime.get("stale").unwrap().status, TurnStatus::Failed);
        assert_eq!(runtime.get("fresh").unwrap().status, TurnStatus::Unknown);
        // The job the sweep just failed is terminal now, so it must leave the
        // scanned prefix (and its unknown_since clock, which only matters
        // while a job is live) — the still-fresh Unknown job must not.
        assert!(
            store
                .get::<RuntimeJob>(&job_key("stale"))
                .unwrap()
                .is_none(),
            "a job the sweep fails must be archived out of job/"
        );
        assert!(
            store
                .get::<u64>(&unknown_since_key("stale"))
                .unwrap()
                .is_none(),
            "an archived job's unknown_since key must be cleaned up"
        );
        assert!(
            store
                .get::<RuntimeJob>(&job_key("fresh"))
                .unwrap()
                .is_some(),
            "a job the sweep does not touch must stay in job/"
        );
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

    fn admin_principal() -> Principal {
        Principal {
            id: "admin-1".into(),
            token_file: "/not-read-by-runtime-tests".into(),
            token_sha256: false,
            roles: vec!["admin".into()],
            akhs: vec![],
            sources: vec![],
        }
    }

    /// The terminal path (worker completion, the stale-unknown sweep, admin
    /// `reconcile`, and the startup migration) all funnel through
    /// `archive_job`. This test exercises that helper directly at the
    /// smallest surface: once a job is terminal, it must vanish from a
    /// `"job/"` scan yet remain reachable through `Runtime::get`.
    #[test]
    fn a_terminal_job_leaves_the_scanned_prefix_and_stays_reachable_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        let runtime = Runtime::new(cfg, store.clone()).unwrap();
        let mut job = unknown_job("job-1");
        job.status = TurnStatus::Completed;
        job.output = Some("done".into());
        store
            .transaction(|tx| {
                tx.put(&job_key("job-1"), &job)?;
                tx.put(&unknown_since_key("job-1"), &now_ms())?;
                tx.put(&event_bytes_key("job-1"), &0usize)?;
                archive_job(tx, &job)
            })
            .unwrap();
        assert!(
            store.scan::<RuntimeJob>("job/").unwrap().is_empty(),
            "a completed job must not be present in a job/ scan"
        );
        assert!(
            store
                .get::<u64>(&unknown_since_key("job-1"))
                .unwrap()
                .is_none()
        );
        assert!(
            store
                .get::<usize>(&event_bytes_key("job-1"))
                .unwrap()
                .is_none()
        );
        let fetched = runtime.get("job-1").unwrap();
        assert_eq!(fetched.status, TurnStatus::Completed);
        assert_eq!(fetched.output.as_deref(), Some("done"));
    }

    /// A resubmission of the same job id, with the exact same request, must
    /// hand back the archived result instead of running the turn again —
    /// callers cannot know or care that this job left `"job/"`.
    #[test]
    fn resubmitting_a_completed_archived_job_with_the_same_request_returns_it_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        let runtime = Runtime::new(cfg.clone(), store.clone()).unwrap();
        let req = request("job-1");
        let job = RuntimeJob {
            request: req.clone(),
            status: TurnStatus::Completed,
            provider_session: None,
            output: Some("done".into()),
            error: None,
            events: vec![],
        };
        store
            .transaction(|tx| {
                tx.put(&job_key("job-1"), &job)?;
                archive_job(tx, &job)
            })
            .unwrap();
        let principal = cfg.principals[0].clone();
        let returned = runtime.submit(&principal, req).unwrap();
        assert_eq!(returned.status, TurnStatus::Completed);
        assert_eq!(returned.output.as_deref(), Some("done"));
        assert!(
            store.scan::<RuntimeJob>("job/").unwrap().is_empty(),
            "an idempotent resubmit must not resurrect the job into job/"
        );
    }

    /// The same job id reused with a materially different request must still
    /// be rejected as a conflict, whether or not the prior job has been
    /// archived.
    #[test]
    fn resubmitting_a_completed_archived_job_with_a_different_request_conflicts() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        let runtime = Runtime::new(cfg.clone(), store.clone()).unwrap();
        let req = request("job-1");
        let job = RuntimeJob {
            request: req.clone(),
            status: TurnStatus::Completed,
            provider_session: None,
            output: Some("done".into()),
            error: None,
            events: vec![],
        };
        store
            .transaction(|tx| {
                tx.put(&job_key("job-1"), &job)?;
                archive_job(tx, &job)
            })
            .unwrap();
        let mut different = req;
        different.instructions = "a materially different instruction".into();
        let principal = cfg.principals[0].clone();
        assert_eq!(
            runtime.submit(&principal, different).unwrap_err().code,
            "conflict"
        );
    }

    /// `reconcile` archives the job it resolves, and a byte-identical repeat
    /// call must stay idempotent via `recovery::already` even though the job
    /// is now found only in `archived_job/`, not `job/`.
    #[test]
    fn reconcile_archives_the_job_and_a_repeat_call_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = config(dir.path());
        let store = Store::open(&dir.path().join("state.redb")).unwrap();
        store
            .transaction(|tx| {
                tx.put(&job_key("job-1"), &unknown_job("job-1"))?;
                tx.put(&unknown_since_key("job-1"), &now_ms())
            })
            .unwrap();
        let runtime = Runtime::new(cfg, store.clone()).unwrap();
        let admin = admin_principal();
        let job = runtime.get("job-1").unwrap();
        let resolution = Reconciliation {
            request_id: "reconcile-job-1".into(),
            expected_request_sha256: crate::recovery::fingerprint(&job.request).unwrap(),
            status: TurnStatus::Failed,
            quiescent: true,
            evidence: vec!["operator checked provider logs".into()],
            output: None,
            result: None,
        };
        let reconciled = runtime.reconcile(&admin, "job-1", &resolution).unwrap();
        assert_eq!(reconciled.status, TurnStatus::Failed);
        assert!(
            store.scan::<RuntimeJob>("job/").unwrap().is_empty(),
            "a reconciled job must leave the scanned job/ prefix"
        );
        assert!(
            store
                .get::<u64>(&unknown_since_key("job-1"))
                .unwrap()
                .is_none()
        );
        let repeat = runtime.reconcile(&admin, "job-1", &resolution).unwrap();
        assert_eq!(
            repeat.status,
            TurnStatus::Failed,
            "an identical repeat reconcile must still succeed idempotently, \
             even though the job now lives only in archived_job/"
        );
    }
}
