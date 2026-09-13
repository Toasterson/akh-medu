//! Allocation intent precedes provisioning; retries address the same named resource.
use crate::{
    config::{Config, Provisioner},
    http::{Client, segment},
    store::{Store, key},
    types::*,
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

pub use akh_session_types::{Allocate, Allocation};

#[derive(Clone)]
pub struct Sandboxes {
    pub config: Arc<Config>,
    pub store: Store,
    client: Client,
    locks: Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>,
        >,
    >,
}
impl Sandboxes {
    fn allocation_lock(&self, session: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap_or_else(|error| error.into_inner());
        locks.retain(|_, lock| lock.strong_count() > 0);
        let entry = locks.entry(session.into()).or_default();
        if let Some(lock) = entry.upgrade() {
            return lock;
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        *entry = Arc::downgrade(&lock);
        lock
    }

    pub async fn reconcile(
        &self,
        p: &crate::config::Principal,
        session: &str,
        r: &AllocationResolution,
    ) -> Result<Allocation> {
        crate::http::role(p, "admin")?;
        let allocation_lock = self.allocation_lock(session);
        let _lock = allocation_lock.lock().await;
        self.store.transaction(|tx| {
            let target = key("allocation", &[session]);
            let mut allocation: Allocation = tx
                .get(&target)?
                .ok_or_else(|| Fault::missing("allocation not found"))?;
            crate::recovery::validate(p, &allocation.request, &r.resolution)?;
            if crate::recovery::already(tx, p, &target, &r.resolution)? {
                if tx
                    .get::<AllocationResolution>(&key(
                        "allocation_reconciliation",
                        &[&r.resolution.request_id],
                    ))?
                    .as_ref()
                    != Some(r)
                {
                    return Err(Fault::conflict(
                        "allocation reconciliation changed its observed binding",
                    ));
                }
                return Ok(allocation);
            }
            if !matches!(
                allocation.state.as_str(),
                "unknown" | "release_unknown" | "creating" | "failed"
            ) {
                return Err(Fault::conflict(
                    "allocation is not awaiting operator reconciliation",
                ));
            }
            if r.resolution.output.is_some() || r.resolution.result.is_some() {
                return Err(Fault::invalid(
                    "allocation recovery uses observed_binding, not model output or tool results",
                ));
            }
            if r.observed_absent
                && (r.observed_binding.is_some() || r.resolution.status != TurnStatus::Completed)
            {
                return Err(Fault::invalid(
                    "observed absence requires completed reconciliation and no binding",
                ));
            }
            if allocation.state == "failed" && !r.observed_absent {
                return Err(Fault::conflict(
                    "failed allocation needs explicit observed absence before closing cleanup",
                ));
            }
            if r.resolution.status == TurnStatus::Completed {
                if allocation.state == "release_unknown" || r.observed_absent {
                    if r.observed_binding.is_some() {
                        return Err(Fault::invalid(
                            "released allocation must have no observed binding",
                        ));
                    }
                    allocation.state = "released".into();
                    allocation.lease_until = 0;
                    allocation.access.clear();
                } else {
                    let binding = r.observed_binding.as_ref().ok_or_else(|| {
                        Fault::invalid("acquired allocation requires an observed executor binding")
                    })?;
                    if binding.id != allocation.id
                        || binding.sandbox_id != allocation.id
                        || binding.generation != 1
                    {
                        return Err(Fault::conflict("observed allocation identity differs"));
                    }
                    self.config.endpoint(&binding.executor)?;
                    allocation.binding = Some(binding.clone());
                    allocation.state = "attaching".into();
                }
            } else {
                allocation.state = "failed".into();
                allocation.error = Some(Fault::new(
                    "operator_resolved",
                    "allocation closed by operator; create an explicit new session",
                ));
            }
            tx.put(&target, &allocation)?;
            crate::recovery::record(tx, p, &target, &r.resolution)?;
            // Include the observed binding in the durable audit, not only in current state.
            tx.put(
                &key("allocation_reconciliation", &[&r.resolution.request_id]),
                r,
            )?;
            Ok(allocation)
        })
    }
    pub async fn renew(&self, session: &str) -> Result<Allocation> {
        let allocation_lock = self.allocation_lock(session);
        let _lock = allocation_lock.lock().await;
        let mut allocation = self.get(session)?;
        if allocation.state != "ready" {
            return Err(Fault::conflict("allocation is not ready for lease renewal"));
        }
        let binding = allocation
            .binding
            .as_ref()
            .ok_or_else(|| Fault::conflict("allocation has no executor binding"))?;
        let lease: WorkspaceLease = self
            .client
            .in_sandbox(&allocation.id)?
            .post(
                &binding.executor,
                &format!("/v1/workspaces/{}/renew", segment(&binding.id)),
                &RenewLease {
                    session_id: session.into(),
                    generation: binding.generation,
                },
            )
            .await?;
        if lease.session_id != session
            || lease.workspace_id != binding.id
            || lease.generation != binding.generation
        {
            return Err(Fault::conflict("executor renewed a different lease"));
        }
        allocation.lease_until = lease.expires_at;
        self.store
            .transaction(|tx| tx.put(&key("allocation", &[session]), &allocation))?;
        Ok(allocation)
    }
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        Ok(Self {
            client: Client::new(config.clone())?,
            config,
            store,
            locks: Default::default(),
        })
    }
    pub fn get(&self, session: &str) -> Result<Allocation> {
        self.store
            .get(&key("allocation", &[session]))?
            .ok_or_else(|| Fault::missing("allocation not found"))
    }
    pub async fn allocate(&self, request: Allocate) -> Result<Allocation> {
        validate_key(&request.session_id)?;
        validate_key(&request.principal)?;
        let allocation_lock = self.allocation_lock(&request.session_id);
        let _lock = allocation_lock.lock().await;
        let profile = self
            .config
            .workspaces
            .get(&request.profile)
            .ok_or_else(|| Fault::missing("workspace profile not found"))?;
        let k = key("allocation", &[&request.session_id]);
        let bootstrap_profile_sha256 = crate::repository_bootstrap::profile_fingerprint(
            profile.repository_bootstrap.as_ref(),
            request.bootstrap.as_ref(),
        )?;
        let mut allocation = match self.store.get::<Allocation>(&k)? {
            Some(old) => {
                if old.request != request
                    || old.profile_revision != profile.revision
                    || old.bootstrap_profile_sha256 != bootstrap_profile_sha256
                {
                    return Err(Fault::conflict(
                        "allocation request or installed profile changed",
                    ));
                }
                old
            }
            None => {
                let allocation = Allocation {
                    bootstrap_profile_sha256,
                    bootstrap_result: None,
                    resource_uid: None,
                    id: format!(
                        "akh-{}",
                        &crate::digest(request.session_id.as_bytes())[..32]
                    ),
                    request: request.clone(),
                    profile_revision: profile.revision.clone(),
                    state: "allocating".into(),
                    binding: None,
                    error: None,
                    lease_until: now_ms() + 300_000,
                    access: vec![],
                };
                self.store.transaction(|tx| tx.put(&k, &allocation))?;
                allocation
            }
        };
        if matches!(
            allocation.state.as_str(),
            "released" | "retained" | "failed" | "release_unknown" | "releasing"
        ) {
            return Err(Fault::conflict(
                "allocation was released; create a new session to allocate again",
            ));
        }
        if allocation.state == "unknown" {
            return Err(Fault::new(
                "unknown_outcome",
                "allocation command may have completed; reconcile its named allocation before retrying",
            ));
        }
        if allocation.state == "ready" {
            if allocation.lease_until <= now_ms() {
                return Err(Fault::new(
                    "expired",
                    "allocation lease expired; no implicit reacquisition",
                ));
            }
            return Ok(allocation);
        }
        let result: Result<WorkspaceBinding> = async {
            if let Some(binding) = &allocation.binding { return Ok(binding.clone()); }
            match &profile.provisioner {
                Provisioner::Existing { executor, root } => Ok(WorkspaceBinding { id: allocation.id.clone(), sandbox_id: allocation.id.clone(), executor: executor.clone(), generation: 1, site: profile.site.clone(), root: format!("{}/{}", root.trim_end_matches('/'), allocation.id), platform: "unobserved".into(), confinement: "unobserved".into() }),
                Provisioner::Command { acquire, .. } => {
                    // Commands must implement named allocation lookup themselves. Once sent, a
                    // crash cannot prove whether provisioning happened; never invoke blindly twice.
                    allocation.state = "unknown".into(); self.store.transaction(|tx|tx.put(&k, &allocation))?;
                    let binding: WorkspaceBinding = command(acquire, &json!({"operation":"acquire","allocation_id":allocation.id,"session_id":request.session_id,"principal":request.principal,"profile":request.profile,"revision":profile.revision})).await?;
                    if binding.id != allocation.id || binding.sandbox_id != allocation.id || binding.generation != 1 { return Err(Fault::conflict("provisioner returned a different allocation identity")); }
                    self.config.endpoint(&binding.executor)?; Ok(binding)
                }
                Provisioner::Kubernetes { api, namespace, resource, template, executor } => {
                    // `resource` is the configured CRD collection path, e.g. /apis/agents.x-k8s.io/v1beta1/namespaces/N/sandboxes.
                    if !resource.starts_with("/apis/") || !resource.contains(&format!("/namespaces/{namespace}/")) { return Err(Fault::invalid("Kubernetes resource must be a namespaced CRD collection path")); }
                    let path = format!("{}/{}", resource.trim_end_matches('/'), segment(&allocation.id));
                    let object: Value = match self.client.get(api, &path).await {
                        Ok(object) => object,
                        Err(error) if error.code == "not_found" && allocation.resource_uid.is_none() => {
                            if allocation.state == "creating" { return Err(Fault::new("unknown_outcome", "Kubernetes create may still complete; lookup or explicitly reconcile it, never create again")); }
                            let mut object = template.clone();
                            object["metadata"]["name"] = json!(allocation.id); object["metadata"]["namespace"] = json!(namespace);
                            object["metadata"]["labels"]["akh.dev/allocation"] = json!(allocation.id);
                            object["metadata"]["annotations"]["akh.dev/session"] = json!(request.session_id);
                            object["metadata"]["annotations"]["akh.dev/profile-revision"] = json!(profile.revision);
                            allocation.state="creating".into();
                            self.store.transaction(|tx|tx.put(&k,&allocation))?;
                            self.client.post(api, resource, &object).await?
                        }
                        Err(error) => return Err(error),
                    };
                    if object["metadata"]["annotations"]["akh.dev/session"] != request.session_id || object["metadata"]["annotations"]["akh.dev/profile-revision"] != profile.revision {
                        return Err(Fault::conflict("existing Kubernetes resource has a different owner/profile"));
                    }
                    let uid=object["metadata"]["uid"].as_str().ok_or_else(||Fault::invalid("Kubernetes resource has no UID"))?;
                    if allocation.resource_uid.as_deref().is_some_and(|old|old!=uid) {return Err(Fault::conflict("Kubernetes allocation was replaced by another resource"));}
                    allocation.resource_uid=Some(uid.into()); allocation.state="allocating".into(); self.store.transaction(|tx|tx.put(&k,&allocation))?;
                    if object["status"]["observedGeneration"].as_u64().zip(object["metadata"]["generation"].as_u64()).is_some_and(|(observed,current)|observed<current) {return Err(Fault::new("busy","Kubernetes controller has not observed this generation"));}
                    let ready = object["status"]["conditions"].as_array().is_some_and(|conditions|conditions.iter().any(|c|c["type"] == "Ready" && c["status"] == "True" && object["metadata"]["generation"].as_u64().is_some_and(|generation|c["observedGeneration"].as_u64().is_some_and(|observed|observed>=generation))));
                    if !ready { return Err(Fault::new("busy", "sandbox resource is not ready")); }
                    Ok(WorkspaceBinding { id: allocation.id.clone(), sandbox_id: allocation.id.clone(), executor: executor.clone(), generation: 1, site: profile.site.clone(), root: format!("/workspaces/{}",allocation.id), platform: "unobserved".into(), confinement: "unobserved".into() })
                }
            }
        }.await;
        match result {
            Ok(binding) => {
                allocation.binding = Some(binding.clone());
                allocation.state = "attaching".into();
                allocation.error = None;
                self.store.transaction(|tx| tx.put(&k, &allocation))?;
                let client = self.client.in_sandbox(&allocation.id)?;
                let binding: WorkspaceBinding = client
                    .post(
                        &binding.executor,
                        "/v1/workspaces",
                        &AttachWorkspace {
                            session_id: request.session_id.clone(),
                            binding: binding.clone(),
                        },
                    )
                    .await?;
                if binding.id != allocation.id || binding.sandbox_id != allocation.id {
                    return Err(Fault::conflict(
                        "executor returned another workspace identity",
                    ));
                }
                if let (Some(git), Some(request)) =
                    (&profile.repository_bootstrap, &request.bootstrap)
                {
                    allocation.bootstrap_result = Some(
                        self.bootstrap_repository(&client, &allocation, &binding, git, request)
                            .await?,
                    );
                }
                allocation.access.clear();
                for service in &profile.services {
                    let status: crate::types::HelperStatus = client
                        .post(
                            &binding.executor,
                            "/v1/services",
                            &crate::types::LaunchService {
                                session_id: request.session_id.clone(),
                                principal: request.principal.clone(),
                                workspace_id: binding.id.clone(),
                                generation: binding.generation,
                                profile: service.clone(),
                            },
                        )
                        .await?;
                    if status.state != "ready" {
                        return Err(status.error.unwrap_or_else(|| {
                            Fault::new("busy", "waiting for upstream service readiness")
                        }));
                    }
                    if let Some(access) = status.access {
                        allocation.access.push(access);
                    }
                }
                allocation.binding = Some(binding);
                allocation.state = "ready".into();
                allocation.lease_until = now_ms() + 300_000;
            }
            Err(error) => {
                allocation.error = Some(error.clone());
                self.store.transaction(|tx| tx.put(&k, &allocation))?;
                return Err(error);
            }
        }
        self.store.transaction(|tx| tx.put(&k, &allocation))?;
        Ok(allocation)
    }
    pub async fn release(&self, session: &str) -> Result<Allocation> {
        let allocation_lock = self.allocation_lock(session);
        let _lock = allocation_lock.lock().await;
        let mut allocation = self.get(session)?;
        if matches!(allocation.state.as_str(), "released" | "retained") {
            return Ok(allocation);
        }
        if matches!(
            allocation.state.as_str(),
            "unknown" | "release_unknown" | "failed"
        ) {
            return Err(Fault::new(
                "unknown_outcome",
                "resolve the allocation outcome before releasing it",
            ));
        }
        let profile = self
            .config
            .workspaces
            .get(&allocation.request.profile)
            .ok_or_else(|| Fault::missing("workspace profile not found"))?;
        if allocation.state != "releasing" {
            if let Some(binding) = &allocation.binding {
                let _: Value = self
                    .client
                    .in_sandbox(&allocation.id)?
                    .post(
                        &binding.executor,
                        &format!("/v1/workspaces/{}/release", segment(&binding.id)),
                        &json!({"session_id":session,"generation":binding.generation}),
                    )
                    .await?;
            } else if matches!(profile.provisioner, Provisioner::Command { .. }) {
                return Err(Fault::conflict(
                    "command allocation needs explicit resource reconciliation",
                ));
            }
        }
        if profile.retain_on_release {
            if allocation.state == "creating" {
                return Err(Fault::new(
                    "unknown_outcome",
                    "observe or reconcile the outstanding Kubernetes create before retention",
                ));
            }
            allocation.state = "retained".into();
        } else {
            match &profile.provisioner {
                Provisioner::Existing { .. } => {} // Files remain for explicit deletion.
                Provisioner::Command { release, .. } => {
                    allocation.state = "release_unknown".into();
                    self.store
                        .transaction(|tx| tx.put(&key("allocation", &[session]), &allocation))?;
                    let _: Value = command(release, &json!({"operation":"release","allocation_id":allocation.id,"session_id":session})).await?;
                }
                Provisioner::Kubernetes { api, resource, .. } => {
                    let path = format!(
                        "{}/{}",
                        resource.trim_end_matches('/'),
                        segment(&allocation.id)
                    );
                    let object: Value = match self.client.get(api, &path).await {
                        Ok(object) => object,
                        Err(error) if error.code == "not_found" => {
                            if allocation.state == "creating" {
                                return Err(Fault::new(
                                    "unknown_outcome",
                                    "a create without a response may still complete; explicitly reconcile before releasing",
                                ));
                            }
                            allocation.state = "released".into();
                            allocation.lease_until = 0;
                            allocation.access.clear();
                            self.store.transaction(|tx| {
                                tx.put(&key("allocation", &[session]), &allocation)
                            })?;
                            return Ok(allocation);
                        }
                        Err(error) => return Err(error),
                    };
                    if object["metadata"]["annotations"]["akh.dev/session"]
                        != allocation.request.session_id
                        || object["metadata"]["annotations"]["akh.dev/profile-revision"]
                            != allocation.profile_revision
                    {
                        return Err(Fault::conflict(
                            "Kubernetes allocation owner or profile differs",
                        ));
                    }
                    let uid = object["metadata"]["uid"]
                        .as_str()
                        .ok_or_else(|| Fault::invalid("Kubernetes resource has no UID"))?;
                    if allocation
                        .resource_uid
                        .as_deref()
                        .is_some_and(|old| old != uid)
                    {
                        return Err(Fault::conflict(
                            "Kubernetes allocation was replaced by another resource",
                        ));
                    }
                    allocation.resource_uid = Some(uid.into());
                    allocation.state = "releasing".into();
                    self.store
                        .transaction(|tx| tx.put(&key("allocation", &[session]), &allocation))?;
                    let result:Result<Value>=self.client.call(api,reqwest::Method::DELETE,&path,
                        Some(json!({"propagationPolicy":"Foreground","preconditions":{"uid":uid}}))).await;
                    if let Err(error) = result {
                        if error.code != "not_found" {
                            return Err(error);
                        }
                    }
                    // DELETE acceptance is not observed foreground deletion. Retain
                    // ownership and skip the already-fenced executor on the next poll.
                    match self.client.get::<Value>(api, &path).await {
                        Err(error) if error.code == "not_found" => {}
                        Err(error) => return Err(error),
                        Ok(object) => {
                            if object["metadata"]["uid"] != uid {
                                return Err(Fault::conflict(
                                    "Kubernetes allocation was replaced during release",
                                ));
                            }
                            return Err(Fault::new(
                                "busy",
                                "waiting for Kubernetes foreground deletion",
                            ));
                        }
                    }
                }
            }
            allocation.state = "released".into();
        }
        allocation.access.clear();
        allocation.lease_until = 0;
        self.store
            .transaction(|tx| tx.put(&key("allocation", &[session]), &allocation))?;
        Ok(allocation)
    }
}

/// Administrative adapter, bounded JSON stdin/stdout; no shell expansion or inherited secrets.
pub async fn command<T: serde::de::DeserializeOwned>(argv: &[String], input: &Value) -> Result<T> {
    let (binary, arguments) = argv
        .split_first()
        .ok_or_else(|| Fault::invalid("adapter command is empty"))?;
    let mut command = tokio::process::Command::new(binary);
    command
        .args(arguments)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let input = serde_json::to_vec(input).map_err(|e| Fault::invalid(e.to_string()))?;
    if input.len() > 1024 * 1024 {
        return Err(Fault::invalid("provisioner input exceeds 1 MiB"));
    }
    let bytes = crate::process::output_with_input(
        command,
        Some(input),
        Duration::from_secs(120),
        1024 * 1024,
    )
    .await
    .map_err(|e| {
        Fault::new(
            "unknown_outcome",
            format!(
                "provisioner {}: inspect the named allocation before retry",
                e.code
            ),
        )
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|_| Fault::invalid("provisioner output does not match the allocation contract"))
}
