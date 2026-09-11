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
    pub fn launch_service(&self, request: LaunchService) -> Result<Status> {
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
                        id: id.clone(),
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
            Ok(status) => Ok(status),
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
