//! macOS launchd service management for akhomed.
//!
//! Generates a launchd plist, installs it as a user LaunchAgent, and provides
//! start/stop/status/uninstall operations via `launchctl`. On non-macOS
//! platforms this module compiles to a stub that returns [`ServiceError::NotMacOS`].

use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from launchd service management.
#[derive(Debug, Error, Diagnostic)]
pub enum ServiceError {
    #[error("launchd service management is macOS-only")]
    #[diagnostic(
        code(akh::service::not_macos),
        help("On Linux, use systemd: `systemctl --user enable akhomed`.")
    )]
    NotMacOS,

    #[error("failed to write plist to {path}: {source}")]
    #[diagnostic(
        code(akh::service::plist_write),
        help("Check that ~/Library/LaunchAgents/ exists and you have write permissions.")
    )]
    PlistWriteFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("launchctl failed: {message}")]
    #[diagnostic(
        code(akh::service::launchctl),
        help("Run `launchctl list | grep akh` to check the current state.")
    )]
    LaunchctlFailed { message: String },

    #[error("service already installed at {path}")]
    #[diagnostic(
        code(akh::service::already_installed),
        help("Run `akh service uninstall` first, or `akh service show` to inspect the plist.")
    )]
    AlreadyInstalled { path: String },

    #[error("service not installed — no plist found at {path}")]
    #[diagnostic(
        code(akh::service::not_installed),
        help("Run `akh service install` to create the launchd plist.")
    )]
    NotInstalled { path: String },

    #[error("cannot determine home directory")]
    #[diagnostic(
        code(akh::service::no_home),
        help("Set the HOME environment variable.")
    )]
    NoHome,

    #[error("cannot locate akhomed binary")]
    #[diagnostic(
        code(akh::service::no_binary),
        help(
            "Ensure akhomed is installed. It should be next to the akh binary, \
             or available in $PATH."
        )
    )]
    NoBinary,
}

pub type ServiceResult<T> = Result<T, ServiceError>;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for the launchd plist.
#[derive(Debug, Clone)]
pub struct LaunchdConfig {
    /// Launchd service label (reverse-DNS).
    pub label: String,
    /// Absolute path to the akhomed binary.
    pub program: PathBuf,
    /// Program arguments (after the binary path).
    pub program_args: Vec<String>,
    /// Directory for stdout/stderr logs.
    pub log_dir: PathBuf,
    /// Path where the plist file will be written.
    pub plist_path: PathBuf,
    /// Environment variables to set.
    pub environment: Vec<(String, String)>,
    /// Whether launchd should keep the service alive.
    pub keep_alive: bool,
    /// Nice value (process priority, higher = lower priority).
    pub nice: i32,
}

/// Parsed status from `launchctl list`.
#[derive(Debug, Clone)]
pub struct ServiceStatus {
    /// Whether the service is loaded in launchd.
    pub loaded: bool,
    /// Whether the service is currently running.
    pub running: bool,
    /// PID if running.
    pub pid: Option<u32>,
    /// Last exit status (0 = clean).
    pub last_exit_status: Option<i32>,
}

// ---------------------------------------------------------------------------
// Plist generation
// ---------------------------------------------------------------------------

/// Generate a launchd plist XML string from the given configuration.
pub fn generate_plist(config: &LaunchdConfig) -> String {
    let mut args_xml = String::new();
    args_xml.push_str(&format!(
        "        <string>{}</string>\n",
        xml_escape(&config.program.to_string_lossy())
    ));
    for arg in &config.program_args {
        args_xml.push_str(&format!(
            "        <string>{}</string>\n",
            xml_escape(arg)
        ));
    }

    let mut env_xml = String::new();
    if !config.environment.is_empty() {
        env_xml.push_str("    <key>EnvironmentVariables</key>\n");
        env_xml.push_str("    <dict>\n");
        for (key, val) in &config.environment {
            env_xml.push_str(&format!(
                "        <key>{}</key>\n        <string>{}</string>\n",
                xml_escape(key),
                xml_escape(val)
            ));
        }
        env_xml.push_str("    </dict>\n");
    }

    let stdout_log = config.log_dir.join("akhomed.stdout.log");
    let stderr_log = config.log_dir.join("akhomed.stderr.log");

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
{args}    </array>
    <key>KeepAlive</key>
    <{keep_alive}/>
    <key>RunAtLoad</key>
    <true/>
    <key>ExitTimeOut</key>
    <integer>30</integer>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>Nice</key>
    <integer>{nice}</integer>
    <key>ProcessType</key>
    <string>Background</string>
    <key>StandardOutPath</key>
    <string>{stdout}</string>
    <key>StandardErrorPath</key>
    <string>{stderr}</string>
{env}</dict>
</plist>
"#,
        label = xml_escape(&config.label),
        args = args_xml,
        keep_alive = if config.keep_alive { "true" } else { "false" },
        nice = config.nice,
        stdout = stdout_log.display(),
        stderr = stderr_log.display(),
        env = env_xml,
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ---------------------------------------------------------------------------
// Platform-specific operations
// ---------------------------------------------------------------------------

/// Build a default [`LaunchdConfig`] by locating the akhomed binary relative
/// to the current executable.
///
/// `port` overrides the default server port (8200).
pub fn default_config(port: Option<u16>) -> ServiceResult<LaunchdConfig> {
    let log_dir = crate::paths::AkhPaths::macos_log_dir().ok_or(ServiceError::NoHome)?;
    let plist_dir =
        crate::paths::AkhPaths::macos_launch_agents_dir().ok_or(ServiceError::NoHome)?;

    let program = find_akhomed()?;
    let port_val = port.unwrap_or(8200);

    let mut env = vec![
        ("RUST_LOG".into(), "info,egg=warn,hnsw_rs=warn".into()),
        ("AKH_AUTO_START".into(), "default".into()),
    ];
    if port_val != 8200 {
        env.push(("AKH_SERVER_PORT".into(), port_val.to_string()));
    }

    let mut program_args = Vec::new();
    if port_val != 8200 {
        program_args.push("--port".into());
        program_args.push(port_val.to_string());
    }

    Ok(LaunchdConfig {
        label: "dev.akh-medu.akhomed".into(),
        program,
        program_args,
        log_dir,
        plist_path: plist_dir.join("dev.akh-medu.akhomed.plist"),
        environment: env,
        keep_alive: true,
        nice: 10,
    })
}

/// Find the akhomed binary: first as a sibling of the current exe, then via PATH.
fn find_akhomed() -> ServiceResult<PathBuf> {
    // 1. Sibling of current executable.
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(dir) = current_exe.parent() {
            let sibling = dir.join("akhomed");
            if sibling.is_file() {
                return Ok(sibling);
            }
        }
    }

    // 2. Search PATH via `which`.
    if cfg!(target_os = "macos") {
        if let Ok(output) = std::process::Command::new("which")
            .arg("akhomed")
            .output()
        {
            if output.status.success() {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !path.is_empty() {
                    return Ok(PathBuf::from(path));
                }
            }
        }
    }

    Err(ServiceError::NoBinary)
}

/// Install the launchd service (write plist + `launchctl load`).
#[cfg(target_os = "macos")]
pub fn install(config: &LaunchdConfig) -> ServiceResult<()> {
    if config.plist_path.exists() {
        return Err(ServiceError::AlreadyInstalled {
            path: config.plist_path.display().to_string(),
        });
    }

    // Ensure log directory exists.
    std::fs::create_dir_all(&config.log_dir).map_err(|e| ServiceError::PlistWriteFailed {
        path: config.log_dir.display().to_string(),
        source: e,
    })?;

    // Ensure LaunchAgents directory exists.
    if let Some(parent) = config.plist_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ServiceError::PlistWriteFailed {
            path: parent.display().to_string(),
            source: e,
        })?;
    }

    // Write plist.
    let plist_content = generate_plist(config);
    std::fs::write(&config.plist_path, &plist_content).map_err(|e| {
        ServiceError::PlistWriteFailed {
            path: config.plist_path.display().to_string(),
            source: e,
        }
    })?;

    // Load into launchd.
    let output = std::process::Command::new("launchctl")
        .args(["load", "-w"])
        .arg(&config.plist_path)
        .output()
        .map_err(|e| ServiceError::LaunchctlFailed {
            message: format!("failed to execute launchctl: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ServiceError::LaunchctlFailed {
            message: format!("launchctl load failed: {stderr}"),
        });
    }

    Ok(())
}

/// Uninstall the launchd service (`launchctl unload` + remove plist).
#[cfg(target_os = "macos")]
pub fn uninstall(config: &LaunchdConfig) -> ServiceResult<()> {
    if !config.plist_path.exists() {
        return Err(ServiceError::NotInstalled {
            path: config.plist_path.display().to_string(),
        });
    }

    // Unload from launchd.
    let output = std::process::Command::new("launchctl")
        .args(["unload"])
        .arg(&config.plist_path)
        .output()
        .map_err(|e| ServiceError::LaunchctlFailed {
            message: format!("failed to execute launchctl: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Don't fail if it was already unloaded.
        if !stderr.contains("Could not find specified service") {
            return Err(ServiceError::LaunchctlFailed {
                message: format!("launchctl unload failed: {stderr}"),
            });
        }
    }

    // Remove the plist file.
    std::fs::remove_file(&config.plist_path).map_err(|e| ServiceError::PlistWriteFailed {
        path: config.plist_path.display().to_string(),
        source: e,
    })?;

    Ok(())
}

/// Start the service via `launchctl start`.
#[cfg(target_os = "macos")]
pub fn start(config: &LaunchdConfig) -> ServiceResult<()> {
    if !config.plist_path.exists() {
        return Err(ServiceError::NotInstalled {
            path: config.plist_path.display().to_string(),
        });
    }

    let output = std::process::Command::new("launchctl")
        .args(["start", &config.label])
        .output()
        .map_err(|e| ServiceError::LaunchctlFailed {
            message: format!("failed to execute launchctl: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ServiceError::LaunchctlFailed {
            message: format!("launchctl start failed: {stderr}"),
        });
    }

    Ok(())
}

/// Stop the service via `launchctl stop`.
#[cfg(target_os = "macos")]
pub fn stop(config: &LaunchdConfig) -> ServiceResult<()> {
    if !config.plist_path.exists() {
        return Err(ServiceError::NotInstalled {
            path: config.plist_path.display().to_string(),
        });
    }

    let output = std::process::Command::new("launchctl")
        .args(["stop", &config.label])
        .output()
        .map_err(|e| ServiceError::LaunchctlFailed {
            message: format!("failed to execute launchctl: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ServiceError::LaunchctlFailed {
            message: format!("launchctl stop failed: {stderr}"),
        });
    }

    Ok(())
}

/// Query service status by parsing `launchctl list` output.
#[cfg(target_os = "macos")]
pub fn status(config: &LaunchdConfig) -> ServiceResult<ServiceStatus> {
    if !config.plist_path.exists() {
        return Ok(ServiceStatus {
            loaded: false,
            running: false,
            pid: None,
            last_exit_status: None,
        });
    }

    let output = std::process::Command::new("launchctl")
        .args(["list"])
        .output()
        .map_err(|e| ServiceError::LaunchctlFailed {
            message: format!("failed to execute launchctl: {e}"),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_launchctl_list(&stdout, &config.label)
}

/// Parse `launchctl list` output to extract service status.
///
/// Each line has the format: `PID\tStatus\tLabel`
/// where PID is `-` if not running and Status is the last exit code.
pub fn parse_launchctl_list(output: &str, label: &str) -> ServiceResult<ServiceStatus> {
    for line in output.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 && parts[2] == label {
            let pid = parts[0].parse::<u32>().ok();
            let exit_status = parts[1].parse::<i32>().ok();
            return Ok(ServiceStatus {
                loaded: true,
                running: pid.is_some(),
                pid,
                last_exit_status: exit_status,
            });
        }
    }

    Ok(ServiceStatus {
        loaded: false,
        running: false,
        pid: None,
        last_exit_status: None,
    })
}

// ---------------------------------------------------------------------------
// Non-macOS stubs
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
pub fn install(_config: &LaunchdConfig) -> ServiceResult<()> {
    Err(ServiceError::NotMacOS)
}

#[cfg(not(target_os = "macos"))]
pub fn uninstall(_config: &LaunchdConfig) -> ServiceResult<()> {
    Err(ServiceError::NotMacOS)
}

#[cfg(not(target_os = "macos"))]
pub fn start(_config: &LaunchdConfig) -> ServiceResult<()> {
    Err(ServiceError::NotMacOS)
}

#[cfg(not(target_os = "macos"))]
pub fn stop(_config: &LaunchdConfig) -> ServiceResult<()> {
    Err(ServiceError::NotMacOS)
}

#[cfg(not(target_os = "macos"))]
pub fn status(_config: &LaunchdConfig) -> ServiceResult<ServiceStatus> {
    Err(ServiceError::NotMacOS)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LaunchdConfig {
        LaunchdConfig {
            label: "dev.akh-medu.akhomed".into(),
            program: PathBuf::from("/usr/local/bin/akhomed"),
            program_args: vec!["--port".into(), "8200".into()],
            log_dir: PathBuf::from("/Users/test/Library/Logs/akh-medu"),
            plist_path: PathBuf::from(
                "/Users/test/Library/LaunchAgents/dev.akh-medu.akhomed.plist",
            ),
            environment: vec![
                ("RUST_LOG".into(), "info,egg=warn".into()),
                ("AKH_AUTO_START".into(), "default".into()),
            ],
            keep_alive: true,
            nice: 10,
        }
    }

    #[test]
    fn plist_generates_valid_xml() {
        let config = test_config();
        let plist = generate_plist(&config);

        // Verify XML structure.
        assert!(plist.starts_with("<?xml version=\"1.0\""));
        assert!(plist.contains("<key>Label</key>"));
        assert!(plist.contains("<string>dev.akh-medu.akhomed</string>"));

        // Verify program path is in ProgramArguments.
        assert!(plist.contains("<string>/usr/local/bin/akhomed</string>"));
        assert!(plist.contains("<string>--port</string>"));
        assert!(plist.contains("<string>8200</string>"));

        // Verify KeepAlive.
        assert!(plist.contains("<key>KeepAlive</key>"));
        assert!(plist.contains("<true/>"));

        // Verify RunAtLoad.
        assert!(plist.contains("<key>RunAtLoad</key>"));

        // Verify ExitTimeOut.
        assert!(plist.contains("<key>ExitTimeOut</key>"));
        assert!(plist.contains("<integer>30</integer>"));

        // Verify ThrottleInterval.
        assert!(plist.contains("<key>ThrottleInterval</key>"));
        assert!(plist.contains("<integer>10</integer>"));

        // Verify Nice value.
        assert!(plist.contains("<key>Nice</key>"));

        // Verify ProcessType.
        assert!(plist.contains("<key>ProcessType</key>"));
        assert!(plist.contains("<string>Background</string>"));

        // Verify log paths.
        assert!(plist.contains("akhomed.stdout.log"));
        assert!(plist.contains("akhomed.stderr.log"));

        // Verify environment variables.
        assert!(plist.contains("<key>EnvironmentVariables</key>"));
        assert!(plist.contains("<key>RUST_LOG</key>"));
        assert!(plist.contains("<key>AKH_AUTO_START</key>"));
    }

    #[test]
    fn plist_escapes_xml_entities() {
        let mut config = test_config();
        config.environment = vec![("KEY".into(), "a<b&c".into())];
        let plist = generate_plist(&config);
        assert!(plist.contains("a&lt;b&amp;c"));
    }

    #[test]
    fn plist_no_env_section_when_empty() {
        let mut config = test_config();
        config.environment = vec![];
        let plist = generate_plist(&config);
        assert!(!plist.contains("EnvironmentVariables"));
    }

    #[test]
    fn default_config_uses_correct_label() {
        // This test only checks that default_config produces the right label
        // and paths when HOME is set (which it always is in test environments).
        if let Ok(config) = default_config(None) {
            assert_eq!(config.label, "dev.akh-medu.akhomed");
            assert!(config.plist_path.to_string_lossy().contains("LaunchAgents"));
            assert!(config
                .plist_path
                .to_string_lossy()
                .contains("dev.akh-medu.akhomed.plist"));
            assert!(config.log_dir.to_string_lossy().contains("Logs/akh-medu"));
            assert!(config.keep_alive);
            assert_eq!(config.nice, 10);
        }
        // If default_config fails (e.g. akhomed not found), that's fine for CI.
    }

    #[test]
    fn status_parses_launchctl_running() {
        let output = "PID\tStatus\tLabel\n\
                       -\t0\tcom.apple.something\n\
                       12345\t0\tdev.akh-medu.akhomed\n\
                       -\t0\tcom.apple.other\n";
        let st = parse_launchctl_list(output, "dev.akh-medu.akhomed").unwrap();
        assert!(st.loaded);
        assert!(st.running);
        assert_eq!(st.pid, Some(12345));
        assert_eq!(st.last_exit_status, Some(0));
    }

    #[test]
    fn status_parses_launchctl_stopped() {
        let output = "-\t78\tdev.akh-medu.akhomed\n";
        let st = parse_launchctl_list(output, "dev.akh-medu.akhomed").unwrap();
        assert!(st.loaded);
        assert!(!st.running);
        assert_eq!(st.pid, None);
        assert_eq!(st.last_exit_status, Some(78));
    }

    #[test]
    fn status_parses_launchctl_not_loaded() {
        let output = "-\t0\tcom.apple.something\n";
        let st = parse_launchctl_list(output, "dev.akh-medu.akhomed").unwrap();
        assert!(!st.loaded);
        assert!(!st.running);
        assert_eq!(st.pid, None);
        assert_eq!(st.last_exit_status, None);
    }
}
