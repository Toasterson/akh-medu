//! Allocation intent precedes provisioning; retries address the same named resource.
use crate::{
    config::{Config, Provisioner},
    executor::AttachWorkspace,
    http::{Client, segment},
    store::{Store, key},
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Allocate {
    pub session_id: String,
    pub principal: String,
    pub profile: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub request: Allocate,
    pub id: String,
    pub profile_revision: String,
    pub state: String,
    pub binding: Option<WorkspaceBinding>,
    pub error: Option<Fault>,
    pub lease_until: u64,
    #[serde(default)]
    pub access: Vec<WorkspaceAccessDescriptor>,
}
#[derive(Clone)]
pub struct Sandboxes {
    pub config: Arc<Config>,
    pub store: Store,
    client: Client,
    lock: Arc<tokio::sync::Mutex<()>>,
}
impl Sandboxes {
    pub fn new(config: Arc<Config>, store: Store) -> Result<Self> {
        Ok(Self {
            client: Client::new(config.clone())?,
            config,
            store,
            lock: Default::default(),
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
        let _lock = self.lock.lock().await;
        let profile = self
            .config
            .workspaces
            .get(&request.profile)
            .ok_or_else(|| Fault::missing("workspace profile not found"))?;
        let k = key("allocation", &[&request.session_id]);
        let mut allocation = match self.store.get::<Allocation>(&k)? {
            Some(old) => {
                if old.request != request || old.profile_revision != profile.revision {
                    return Err(Fault::conflict(
                        "allocation request or installed profile changed",
                    ));
                }
                old
            }
            None => {
                let allocation = Allocation {
                    id: format!(
                        "akh-{}",
                        &crate::workspace::digest(request.session_id.as_bytes())[..32]
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
        if allocation.state == "released" || allocation.state == "retained" {
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
            allocation.lease_until = now_ms() + 300_000;
            self.store.transaction(|tx| tx.put(&k, &allocation))?;
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
                    // `resource` is the configured CRD collection path, e.g. /apis/agents.x-k8s.io/v1alpha1/namespaces/N/sandboxes.
                    if !resource.starts_with("/apis/") || !resource.contains(&format!("/namespaces/{namespace}/")) { return Err(Fault::invalid("Kubernetes resource must be a namespaced CRD collection path")); }
                    let path = format!("{}/{}", resource.trim_end_matches('/'), segment(&allocation.id));
                    let object: Value = match self.client.get(api, &path).await {
                        Ok(object) => object,
                        Err(error) if error.code == "not_found" => {
                            let mut object = template.clone();
                            object["metadata"]["name"] = json!(allocation.id); object["metadata"]["namespace"] = json!(namespace);
                            object["metadata"]["labels"]["akh.dev/allocation"] = json!(allocation.id);
                            object["metadata"]["annotations"]["akh.dev/session"] = json!(request.session_id);
                            object["metadata"]["annotations"]["akh.dev/profile-revision"] = json!(profile.revision);
                            self.client.post(api, resource, &object).await?
                        }
                        Err(error) => return Err(error),
                    };
                    if object["metadata"]["annotations"]["akh.dev/session"] != request.session_id || object["metadata"]["annotations"]["akh.dev/profile-revision"] != profile.revision {
                        return Err(Fault::conflict("existing Kubernetes resource has a different owner/profile"));
                    }
                    let ready = object["status"]["conditions"].as_array().is_some_and(|conditions|conditions.iter().any(|c|c["type"] == "Ready" && c["status"] == "True"));
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
                allocation.access.clear();
                for service in &profile.services {
                    let status: crate::helper::Status = client
                        .post(
                            &binding.executor,
                            "/v1/services",
                            &crate::executor_helpers::LaunchService {
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
        let _lock = self.lock.lock().await;
        let mut allocation = self.get(session)?;
        if matches!(allocation.state.as_str(), "released" | "retained") {
            return Ok(allocation);
        }
        let profile = self
            .config
            .workspaces
            .get(&allocation.request.profile)
            .ok_or_else(|| Fault::missing("workspace profile not found"))?;
        if profile.retain_on_release {
            allocation.state = "retained".into();
        } else {
            let binding = allocation
                .binding
                .as_ref()
                .ok_or_else(|| Fault::conflict("cannot release an unresolved allocation"))?;
            let _: Value = self
                .client
                .in_sandbox(&allocation.id)?
                .post(
                    &binding.executor,
                    &format!("/v1/workspaces/{}/release", segment(&binding.id)),
                    &json!({"session_id":session,"generation":binding.generation}),
                )
                .await?;
            match &profile.provisioner {
                Provisioner::Existing { .. } => {} // Retain files for explicit operator-controlled deletion.
                Provisioner::Command { release, .. } => {
                    let _: Value = command(release, &json!({"operation":"release","allocation_id":allocation.id,"session_id":session})).await?;
                }
                Provisioner::Kubernetes { api, resource, .. } => {
                    let result: Result<Value> = self
                        .client
                        .call(
                            api,
                            reqwest::Method::DELETE,
                            &format!(
                                "{}/{}",
                                resource.trim_end_matches('/'),
                                segment(&allocation.id)
                            ),
                            Some(json!({"propagationPolicy":"Foreground"})),
                        )
                        .await;
                    if let Err(error) = result {
                        if error.code != "not_found" {
                            return Err(error);
                        }
                    }
                }
            }
            allocation.state = "released".into();
        }
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
    let mut child = command
        .spawn()
        .map_err(|_| Fault::new("provisioner", "adapter could not start"))?;
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let bytes = serde_json::to_vec(input).map_err(|e| Fault::invalid(e.to_string()))?;
    let reader = tokio::spawn(async move {
        let mut bytes = vec![];
        stdout
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .await
            .map(|_| bytes)
    });
    let operation = async {
        stdin.write_all(&bytes).await?;
        stdin.shutdown().await?;
        drop(stdin);
        child.wait().await
    };
    let status = tokio::time::timeout(Duration::from_secs(120), operation)
        .await
        .map_err(|_| Fault::new("unknown_outcome", "provisioner timed out"))?
        .map_err(|_| Fault::new("unknown_outcome", "provisioner wait failed"))?;
    if !status.success() {
        reader.abort();
        return Err(Fault::new(
            "unknown_outcome",
            "provisioner exited unsuccessfully; inspect the named allocation",
        ));
    }
    let bytes = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .map_err(|_| Fault::new("unknown_outcome", "provisioner output remains open"))?
        .map_err(|_| Fault::new("provisioner", "output reader failed"))?
        .map_err(|_| Fault::new("provisioner", "output read failed"))?;
    if bytes.len() > 1024 * 1024 {
        return Err(Fault::invalid("provisioner output exceeds 1 MiB"));
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| Fault::invalid("provisioner output does not match the allocation contract"))
}
