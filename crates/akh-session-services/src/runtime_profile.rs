//! An account/config helper uses exactly the worker's private provider profile.
use crate::{helper, types::*};
use std::{
    collections::BTreeMap,
    fs::File,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    time::Duration,
};

pub fn acquire(profile: &ProviderConfig) -> Result<File> {
    let scope = profile.isolation.as_ref().ok_or_else(|| {
        Fault::invalid("dedicated runtimes require an explicit private isolation directory")
    })?;
    if !scope.directory.is_absolute() || !profile.working_dir.is_absolute() {
        return Err(Fault::invalid(
            "runtime state and working directories must be absolute",
        ));
    }
    helper::private_directory(&scope.directory)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(scope.directory.join("owner.lock"))
        .map_err(|_| Fault::new("profile_io", "profile owner lock could not be opened"))?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(Fault::new(
            "busy",
            "runtime or login helper already owns this profile",
        ));
    }
    for path in ["home", "cache", "config", "config/codex", "config/claude"] {
        helper::private_directory(&scope.directory.join(path))?;
    }
    let files: BTreeMap<String, String> = scope
        .managed_files
        .iter()
        .map(|(path, value)| (path.clone(), akh_runtime::profile::expand(value, profile)))
        .collect();
    helper::install_files(
        &scope.directory.join("config"),
        &scope.directory.join("managed-files.json"),
        &files,
    )?;
    akh_runtime::profile::environment(profile).map_err(Fault::invalid)?;
    Ok(lock)
}

pub async fn action(profile: &ProviderConfig, action: &str) -> Result<()> {
    let _lease = acquire(profile)?;
    let scope = profile.isolation.as_ref().unwrap();
    let argv = match action {
        "login" => &scope.login_command,
        "status" => &scope.status_command,
        "version" => &scope.version_command,
        "prepare" => {
            println!(
                "Private runtime configuration prepared: {}",
                profile.revision
            );
            return Ok(());
        }
        _ => return Err(Fault::invalid("unknown profile action")),
    };
    let (binary, args) = argv.split_first().ok_or_else(|| {
        Fault::new(
            "unsupported",
            "this profile has no command for the requested action",
        )
    })?;
    let mut command = tokio::process::Command::new(akh_runtime::profile::expand(binary, profile));
    command
        .args(
            args.iter()
                .map(|v| akh_runtime::profile::expand(v, profile)),
        )
        .current_dir(&profile.working_dir)
        .env_clear()
        .envs(akh_runtime::profile::environment(profile).map_err(Fault::invalid)?);
    if action == "login" {
        // Login stays attached to the user's terminal; the provider owns its credential files.
        command
            .stdin(std::process::Stdio::inherit())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        if !command
            .status()
            .await
            .map_err(|_| Fault::new("login_failed", "login command failed to start"))?
            .success()
        {
            return Err(Fault::new(
                "login_failed",
                "provider login did not complete",
            ));
        }
    } else {
        let output = crate::process::output(command, Duration::from_secs(30), 1024 * 1024).await?;
        use std::io::Write;
        std::io::stdout()
            .write_all(&output)
            .map_err(|_| Fault::new("output", "could not write provider status"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_environment_config_ownership_and_exclusive_login() {
        let dir = tempfile::tempdir().unwrap();
        let mut profile:ProviderConfig=serde_json::from_value(serde_json::json!({"revision":"1","kind":"claude","working_dir":dir.path(),"binary":"/usr/bin/false","isolation":{"directory":dir.path().join("profile"),"managed_files":{"claude/settings.json":"{}"}}})).unwrap();
        let lease = acquire(&profile).unwrap();
        assert_eq!(acquire(&profile).unwrap_err().code, "busy");
        let env = akh_runtime::profile::environment(&profile).unwrap();
        assert_eq!(
            env["HOME"],
            dir.path().join("profile/home").to_string_lossy()
        );
        assert!(!env.contains_key("ANTHROPIC_API_KEY"));
        let login = dir.path().join("profile/config/claude/login.json");
        std::fs::write(&login, "private provider state").unwrap();
        drop(lease);
        std::fs::write(
            dir.path().join("profile/config/claude/settings.json"),
            "user changed config",
        )
        .unwrap();
        assert_eq!(acquire(&profile).unwrap_err().code, "conflict");
        assert_eq!(
            std::fs::read_to_string(login).unwrap(),
            "private provider state"
        );
        profile
            .environment
            .insert("CLAUDE_CONFIG_DIR".into(), "/ambient".into());
        assert!(akh_runtime::profile::environment(&profile).is_err());
    }
}
