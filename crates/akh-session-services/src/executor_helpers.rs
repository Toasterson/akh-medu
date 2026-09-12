use crate::{
    executor::Executor,
    helper::{self, Manifest, Status},
    store::key,
    types::*,
};
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};

pub use akh_session_types::LaunchService;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Instance {
    manifest: Manifest,
    launched_at: u64,
}
impl Executor {
    pub fn reconcile_service(
        &self,
        p: &crate::config::Principal,
        id: &str,
        resolution: &Reconciliation,
    ) -> Result<Status> {
        crate::http::role(p, "admin")?;
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        let manifest = self.service_manifest(id)?;
        crate::recovery::validate(p, &manifest, resolution)?;
        if resolution.status == TurnStatus::Completed
            || resolution.output.is_some()
            || resolution.result.is_some()
        {
            return Err(Fault::invalid(
                "helper reconciliation can only certify a quiescent stopped/failed process",
            ));
        }
        let target = key("helper", &[id]);
        if self
            .store
            .transaction(|tx| crate::recovery::already(tx, p, &target, resolution))?
        {
            return self.service_status(id);
        }
        let _owner = helper::owner_lock(&manifest)?;
        let status = Status {
            id: id.into(),
            revision: manifest.profile.revision.clone(),
            state: "stopped".into(),
            heartbeat: now_ms(),
            error: Some(Fault::new(
                "operator_resolved",
                "helper process absence was explicitly reconciled",
            )),
            access: None,
        };
        helper::atomic_json(&manifest.state_dir.join("status.json"), &status)?;
        self.store
            .transaction(|tx| crate::recovery::record(tx, p, &target, resolution))?;
        Ok(status)
    }
    pub fn restart_service(
        &self,
        p: &crate::config::Principal,
        id: &str,
        request: &ServiceRestart,
    ) -> Result<Status> {
        crate::http::role(p, "admin")?;
        validate_key(&request.request_id)?;
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        let k = key("helper", &[id]);
        let restart = key("helper_restart", &[id, &request.request_id]);
        let mut instance: Instance = self
            .store
            .get(&k)?
            .ok_or_else(|| Fault::missing("service not found"))?;
        self.valid_lease(&instance.manifest.workspace.id)?;
        if let Some(old) = self.store.get::<ServiceRestart>(&restart)? {
            if old != *request {
                return Err(Fault::conflict("helper restart ID was reused"));
            }
            return self.service_status(id);
        }
        if crate::recovery::fingerprint(&instance.manifest)? != request.expected_manifest_sha256 {
            return Err(Fault::conflict(
                "service manifest differs from the restart request",
            ));
        }
        let owner = helper::owner_lock(&instance.manifest)?;
        let status = helper::status(&instance.manifest)?;
        if !matches!(status.state.as_str(), "stopped" | "failed")
            || status.heartbeat < instance.launched_at
        {
            return Err(Fault::conflict(
                "stop or reconcile the previous helper before restarting",
            ));
        }
        if let Some(profile) = &request.profile {
            if profile.id != instance.manifest.profile.id
                || profile.client_kind != instance.manifest.profile.client_kind
            {
                return Err(Fault::conflict(
                    "service identity/client kind cannot change during restart",
                ));
            }
            instance.manifest.profile = profile.clone();
        }
        instance.launched_at = now_ms().max(status.heartbeat.saturating_add(1));
        self.store.transaction(|tx| {
            tx.put(&k, &instance)?;
            tx.put(&restart, request)
        })?;
        // After this durable intent, retries observe status; they never spawn again.
        drop(owner);
        spawn_helper(&instance.manifest)?;
        self.service_status(id)
    }
    pub fn launch_service(&self, request: LaunchService) -> Result<Status> {
        self.valid_lease(&request.workspace_id)?;
        let _lock = self.mutations.lock().unwrap_or_else(|e| e.into_inner());
        validate_key(&request.profile.id)?;
        validate_key(&request.profile.revision)?;
        let allocation: crate::executor::AttachWorkspace = self
            .store
            .get(&key("workspace", &[&request.workspace_id]))?
            .ok_or_else(|| Fault::missing("workspace not found"))?;
        if allocation.session_id != request.session_id
            || allocation.binding.generation != request.generation
        {
            return Err(Fault::new(
                "forbidden",
                "service workspace binding mismatch",
            ));
        }
        if self
            .store
            .get::<bool>(&key("released", &[&request.workspace_id]))?
            .unwrap_or(false)
        {
            return Err(Fault::conflict("workspace released"));
        }
        let id =
            crate::digest(key("helper", &[&request.workspace_id, &request.profile.id]).as_bytes());
        let manifest = Manifest {
            schema_version: SCHEMA_VERSION,
            id: id.clone(),
            session_id: request.session_id,
            principal: request.principal,
            workspace: allocation.binding,
            profile: request.profile,
            state_dir: self.config.data_dir.join("helpers").join(&id),
        };
        let k = key("helper", &[&id]);
        let fresh=self.store.transaction(|tx|{
            if let Some(old)=tx.get::<Instance>(&k)?{if old.manifest!=manifest{return Err(Fault::conflict("service already has another profile or owner; drain it before changing revisions"));}return Ok(false);}
            tx.put(&k,&Instance{manifest:manifest.clone(),launched_at:now_ms()})?;Ok(true)
        })?;
        if fresh {
            spawn_helper(&manifest)?;
        }
        self.service_status(&id)
    }
    pub fn service_manifest(&self, id: &str) -> Result<Manifest> {
        self.store
            .get::<Instance>(&key("helper", &[id]))?
            .map(|i| i.manifest)
            .ok_or_else(|| Fault::missing("service not found"))
    }
    pub fn service_status(&self, id: &str) -> Result<Status> {
        let instance: Instance = self
            .store
            .get(&key("helper", &[id]))?
            .ok_or_else(|| Fault::missing("service not found"))?;
        match helper::status(&instance.manifest) {
            Ok(status) if status.heartbeat >= instance.launched_at => Ok(status),
            Ok(_) => Ok(Status {
                id: id.into(),
                revision: instance.manifest.profile.revision,
                state: if now_ms().saturating_sub(instance.launched_at) < 10_000 {
                    "starting"
                } else {
                    "unknown"
                }
                .into(),
                heartbeat: instance.launched_at,
                error: None,
                access: None,
            }),
            Err(_) if now_ms().saturating_sub(instance.launched_at) < 10_000 => Ok(Status {
                id: id.into(),
                revision: instance.manifest.profile.revision,
                state: "starting".into(),
                heartbeat: instance.launched_at,
                error: None,
                access: None,
            }),
            Err(error) => Err(error),
        }
    }
    pub fn stop_services(&self, workspace: &str) -> Result<()> {
        for instance in self
            .store
            .scan::<Instance>("helper/")?
            .into_iter()
            .filter(|i| i.manifest.workspace.id == workspace)
        {
            let status = helper::status(&instance.manifest)?;
            if !matches!(status.state.as_str(), "stopped" | "failed") {
                helper::stop(&instance.manifest)?;
                return Err(Fault::new("busy", "waiting for upstream services to stop"));
            }
        }
        Ok(())
    }
}

fn spawn_helper(manifest: &Manifest) -> Result<()> {
    let path = manifest.state_dir.join("manifest.json");
    helper::atomic_json(&path, &manifest)?;
    let binary = std::env::current_exe()
        .map_err(|e| Fault::new("helper_spawn", e.to_string()))?
        .with_file_name("akh-service");
    match Command::new(binary)
        .args(["--manifest"])
        .arg(&path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(mut child) => {
            std::thread::spawn(move || {
                let _ = child.wait();
            });
        }
        Err(_) => {
            let status = Status {
                id: manifest.id.clone(),
                revision: manifest.profile.revision.clone(),
                state: "failed".into(),
                heartbeat: now_ms(),
                error: Some(Fault::new(
                    "helper_spawn",
                    "akh-service could not start; install the helper beside akhexecd",
                )),
                access: None,
            };
            helper::atomic_json(&manifest.state_dir.join("status.json"), &status)?;
        }
    }
    Ok(())
}
