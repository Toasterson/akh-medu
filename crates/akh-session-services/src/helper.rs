//! Upstream process supervision and owned configuration. No upstream source patches.
use crate::{
    types::*,
    workspace::{Workspace, digest},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::os::{
    fd::AsRawFd,
    unix::fs::{OpenOptionsExt, PermissionsExt},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u16,
    pub id: String,
    pub session_id: String,
    pub principal: String,
    pub workspace: WorkspaceBinding,
    pub profile: ServiceProfile,
    pub state_dir: PathBuf,
}
pub use akh_session_types::HelperStatus as Status;
fn io(error: impl std::fmt::Display) -> Fault {
    Fault::new("helper_io", error.to_string())
}
pub fn private_directory(path: &Path) -> Result<()> {
    if path.exists() {
        if !path.is_dir()
            || std::fs::symlink_metadata(path)
                .map_err(io)?
                .file_type()
                .is_symlink()
        {
            return Err(Fault::invalid("state path is not a directory"));
        }
        return Ok(());
    }
    std::fs::create_dir_all(path).map_err(io)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(io)
}
pub fn owner_lock(manifest: &Manifest) -> Result<std::fs::File> {
    private_directory(&manifest.state_dir)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(manifest.state_dir.join("owner.lock"))
        .map_err(io)?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(Fault::new("busy", "upstream helper still owns its process"));
    }
    Ok(file)
}

pub fn expand(value: &str, manifest: &Manifest) -> String {
    value
        .replace("{workspace}", &manifest.workspace.root)
        .replace(
            "{config}",
            &manifest.state_dir.join("config").to_string_lossy(),
        )
        .replace("{home}", &manifest.state_dir.join("home").to_string_lossy())
        .replace("{sandbox}", &manifest.workspace.sandbox_id)
}
fn revision(ws: &Workspace, path: &str) -> Result<Option<String>> {
    match ws.read(path) {
        Ok(bytes) => Ok(Some(digest(&bytes))),
        Err(error) if error.code == "not_found" => Ok(None),
        Err(error) => Err(error),
    }
}
pub fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let root = path
        .parent()
        .ok_or_else(|| Fault::invalid("state path has no parent"))?;
    private_directory(root)?;
    let ws = Workspace::open(root)?;
    let file = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Fault::invalid("invalid state filename"))?;
    let old = revision(&ws, file)?;
    let content = serde_json::to_string_pretty(value).map_err(io)?;
    ws.write(file, content.as_bytes(), old.as_deref(), old.is_none())?;
    Ok(())
}
/// Only files previously installed by this helper may be replaced, and only
/// while their content still matches the last installed revision.
pub fn materialize(manifest: &Manifest) -> Result<()> {
    private_directory(&manifest.state_dir)?;
    private_directory(&manifest.state_dir.join("home"))?;
    let files = manifest
        .profile
        .managed_files
        .iter()
        .map(|(path, content)| (path.clone(), expand(content, manifest)))
        .collect();
    install_files(
        &manifest.state_dir.join("config"),
        &manifest.state_dir.join("managed-files.json"),
        &files,
    )
}
pub fn install_files(
    config_dir: &Path,
    index_path: &Path,
    files: &BTreeMap<String, String>,
) -> Result<()> {
    private_directory(&config_dir)?;
    let ws = Workspace::open(&config_dir)?;
    let old: BTreeMap<String, String> = if index_path.exists() {
        serde_json::from_slice(&std::fs::read(&index_path).map_err(io)?).map_err(io)?
    } else {
        BTreeMap::new()
    };
    let mut next = old.clone();
    for (path, content) in files {
        let hash = digest(content.as_bytes());
        let current = revision(&ws, path)?;
        if current.as_ref() == Some(&hash) {
            next.insert(path.clone(), hash);
            continue;
        }
        if current.is_some() && current.as_ref() != old.get(path) {
            return Err(Fault::conflict(format!(
                "managed config {path} was edited outside the helper; preserve and reconcile it"
            )));
        }
        if let Some(parent) = Path::new(path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
        {
            ws.mkdir(
                parent
                    .to_str()
                    .ok_or_else(|| Fault::invalid("non UTF-8 config path"))?,
            )?;
        }
        ws.write(
            path,
            content.as_bytes(),
            current.as_deref(),
            current.is_none(),
        )?;
        next.insert(path.clone(), hash);
    }
    atomic_json(&index_path, &next)
}
pub fn status(manifest: &Manifest) -> Result<Status> {
    let path = manifest.state_dir.join("status.json");
    let mut status: Status =
        serde_json::from_slice(&std::fs::read(path).map_err(io)?).map_err(io)?;
    if !matches!(status.state.as_str(), "stopped" | "failed")
        && now_ms().saturating_sub(status.heartbeat) > 15_000
    {
        status.state = "unknown".into();
        status.access = None;
        status.error = Some(Fault::new(
            "unknown_outcome",
            "helper heartbeat expired; do not launch a duplicate server",
        ));
    }
    Ok(status)
}
pub fn stop(manifest: &Manifest) -> Result<Status> {
    atomic_json(
        &manifest.state_dir.join("stop.json"),
        &json!({"requested_at":now_ms()}),
    )?;
    status(manifest)
}
fn upstream_command(manifest: &Manifest, argv: &[String]) -> Result<tokio::process::Command> {
    let (binary, args) = argv
        .split_first()
        .ok_or_else(|| Fault::invalid("upstream command is empty"))?;
    let mut command = tokio::process::Command::new(expand(binary, manifest));
    command
        .args(args.iter().map(|a| expand(a, manifest)))
        .current_dir(&manifest.workspace.root)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
        .env("HOME", manifest.state_dir.join("home"));
    for (name, value) in &manifest.profile.environment {
        command.env(name, expand(value, manifest));
    }
    Ok(command)
}

pub async fn supervise(manifest: Manifest) -> Result<()> {
    if manifest.schema_version != SCHEMA_VERSION || !manifest.state_dir.is_absolute() {
        return Err(Fault::invalid(
            "unsupported helper manifest or relative state path",
        ));
    }
    private_directory(&manifest.state_dir)?;
    let lock = owner_lock(&manifest)?;
    if let Ok(previous) = status(&manifest) {
        if !matches!(previous.state.as_str(), "stopped" | "failed") {
            return Err(Fault::new(
                "unknown_outcome",
                "a previous upstream process may still exist; reconcile it before replacing its supervisor",
            ));
        }
    }
    materialize(&manifest)?;
    let stop = manifest.state_dir.join("stop.json");
    if stop.exists() {
        std::fs::remove_file(&stop).map_err(io)?;
    }
    let mut status = Status {
        id: manifest.id.clone(),
        revision: manifest.profile.revision.clone(),
        state: "starting".into(),
        heartbeat: now_ms(),
        error: None,
        access: None,
    };
    atomic_json(&manifest.state_dir.join("status.json"), &status)?;
    let mut command = upstream_command(&manifest, &manifest.profile.command)?;
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .kill_on_drop(true);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            status.state = "failed".into();
            status.error = Some(io(error));
            atomic_json(&manifest.state_dir.join("status.json"), &status)?;
            return Err(Fault::new(
                "helper_spawn",
                "upstream process could not start",
            ));
        }
    };
    let child_group = child.id().expect("spawned child has PID") as i32;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(io)?;
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signal = shutdown.clone();
    let signal_task = tokio::spawn(async move {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=term.recv()=>{}}
        signal.store(true, std::sync::atomic::Ordering::SeqCst);
    });
    loop {
        if stop.exists() || shutdown.load(std::sync::atomic::Ordering::SeqCst) {
            if let Some(pid) = child.id() {
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGTERM);
                }
            }
            if tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .is_err()
            {
                if let Some(pid) = child.id() {
                    unsafe {
                        libc::kill(-(pid as i32), libc::SIGKILL);
                    }
                }
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            unsafe {
                libc::kill(-child_group, libc::SIGKILL);
            }
            status.state = "stopped".into();
            status.access = None;
            break;
        }
        if let Some(exit) = child.try_wait().map_err(io)? {
            unsafe {
                libc::kill(-child_group, libc::SIGKILL);
            }
            status.state = "failed".into();
            status.access = None;
            status.error = Some(Fault::new(
                "upstream_exited",
                format!("upstream server exited: {exit}"),
            ));
            break;
        }
        let healthy = if let Some(url) = &manifest.profile.health_url {
            match client.get(expand(url, &manifest)).send().await {
                Ok(response) => response.status().is_success(),
                Err(_) => false,
            }
        } else {
            false
        };
        status.state = if healthy { "ready" } else { "starting" }.into();
        status.heartbeat = now_ms();
        status.access = if healthy {
            descriptor(&manifest).ok()
        } else {
            None
        };
        atomic_json(&manifest.state_dir.join("status.json"), &status)?;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    status.heartbeat = now_ms();
    atomic_json(&manifest.state_dir.join("status.json"), &status)?;
    signal_task.abort();
    drop(lock);
    Ok(())
}
fn descriptor(manifest: &Manifest) -> Result<WorkspaceAccessDescriptor> {
    let endpoint = expand(
        manifest
            .profile
            .endpoint
            .as_ref()
            .ok_or_else(|| Fault::invalid("no advertised endpoint"))?,
        manifest,
    );
    let url = url::Url::parse(&endpoint).map_err(io)?;
    if url.query().is_some()
        || url.fragment().is_some()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(Fault::invalid(
            "access descriptor must not contain credentials or pairing tokens",
        ));
    }
    let environment_id = match &manifest.profile.environment_id_file {
        Some(path) => {
            let path = PathBuf::from(expand(path, manifest));
            let text = std::fs::read_to_string(&path).map_err(io)?;
            let id = serde_json::from_str::<String>(&text).unwrap_or_else(|_| text.trim().into());
            validate_key(&id)?;
            id
        }
        None => manifest.id.clone(),
    };
    Ok(WorkspaceAccessDescriptor {
        id: manifest.id.clone(),
        principal: manifest.principal.clone(),
        session_id: manifest.session_id.clone(),
        workspace_id: manifest.workspace.id.clone(),
        environment_id,
        generation: manifest.workspace.generation,
        client_kind: manifest.profile.client_kind.clone(),
        endpoint,
        version: manifest.profile.upstream_version.clone(),
        central_agent_attachment: manifest.profile.central_agent_attachment,
        state: ConnectionState::Ready,
    })
}

/// Executes the selected upstream account/pairing command in the same private
/// profile. Output is returned directly to the requesting client, never journaled.
pub async fn login(manifest: &Manifest) -> Result<()> {
    let mut command = upstream_command(manifest, &manifest.profile.login_command)?;
    command
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true);
    if !command.status().await.map_err(io)?.success() {
        return Err(Fault::new(
            "login_failed",
            "upstream login did not complete",
        ));
    }
    Ok(())
}
pub async fn action(manifest: &Manifest, action: &str) -> Result<Value> {
    let argv = match action {
        "pair" => &manifest.profile.pair_command,
        "login_status" => &manifest.profile.login_status_command,
        _ => {
            return Err(Fault::invalid(
                "only pair and login_status are available remotely; interactive login runs in the sandbox terminal",
            ));
        }
    };
    let mut command = upstream_command(manifest, argv)?;
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let output = crate::process::output(command, Duration::from_secs(30), 1024 * 1024).await?;
    Ok(json!({"output":String::from_utf8_lossy(&output)}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn managed_configs_preserve_user_edits_and_unmanaged_login_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest:Manifest=serde_json::from_value(json!({"schema_version":1,"id":"service","session_id":"session","principal":"user","state_dir":dir.path(),"workspace":{"id":"w","sandbox_id":"s","executor":"e","generation":1,"site":"test","root":"/work","platform":"test","confinement":"test"},"profile":{"id":"editor","revision":"1","upstream_version":"pinned","command":["upstream"],"managed_files":{"settings.json":"{\"workspace\":\"{workspace}\"}"}}})).unwrap();
        materialize(&manifest).unwrap();
        let config = dir.path().join("config/settings.json");
        assert!(std::fs::read_to_string(&config).unwrap().contains("/work"));
        std::fs::write(dir.path().join("home/login"), "private upstream state").unwrap();
        manifest
            .profile
            .managed_files
            .insert("settings.json".into(), "new managed config".into());
        materialize(&manifest).unwrap();
        std::fs::write(&config, "user edit").unwrap();
        manifest
            .profile
            .managed_files
            .insert("settings.json".into(), "third config".into());
        assert!(materialize(&manifest).is_err());
        assert_eq!(std::fs::read_to_string(config).unwrap(), "user edit");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("home/login")).unwrap(),
            "private upstream state"
        );
    }
}
