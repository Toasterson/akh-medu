//! Server discovery for the akh-medu daemon.
//!
//! akhomed advertises its ACP transport in a PID/discovery record; the CLI and
//! TUI find a running daemon via [`discover_server`] (an ACP socket/TCP liveness
//! probe) and connect over ACP. Also home to [`DaemonStatus`], shared between
//! the daemon and its clients.

use serde::{Deserialize, Serialize};

use crate::paths::AkhPaths;

// ---------------------------------------------------------------------------
// Server discovery
// ---------------------------------------------------------------------------

/// ACP protocol version advertised in the discovery record.
pub const ACP_PROTOCOL_VERSION: &str = "1";

/// Information about a running akhomed instance.
///
/// The `acp_*` fields are a superset added for the ACP transport; they are
/// `#[serde(default)]` so a discovery file written by an older binary (only
/// `pid`/`port`/`bind`) still deserializes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub pid: u32,
    pub port: u16,
    pub bind: String,
    /// ACP Unix socket path, if the daemon serves ACP over a socket.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_socket: Option<String>,
    /// ACP TCP address (`host:port`), if the daemon serves ACP over TCP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_tcp: Option<String>,
    /// ACP protocol version, present iff any ACP transport is served.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acp_version: Option<String>,
}

impl ServerInfo {
    /// Base URL for HTTP requests.
    pub fn base_url(&self) -> String {
        let host = if self.bind == "0.0.0.0" {
            "127.0.0.1"
        } else {
            &self.bind
        };
        format!("http://{host}:{}", self.port)
    }
}

/// Discover a running akhomed server via its PID file.
///
/// Returns `Some(ServerInfo)` when:
/// 1. The PID file exists and parses correctly,
/// 2. the process is still alive (`kill(pid, 0)` succeeds), and
/// 3. the daemon's **ACP transport is reachable** — a connect to the advertised
///    Unix socket (preferred) or TCP address succeeds.
///
/// Step 3 is the ACP-native liveness probe: it confirms the listener is actually
/// accepting connections, not merely that the PID exists. Since REST was retired
/// (ADR-061 P8.4), ACP is the sole liveness signal — a record advertising no
/// `acp_*` transport (a pre-ACP daemon) is treated as not discoverable.
pub fn discover_server(paths: &AkhPaths) -> Option<ServerInfo> {
    let pid_path = paths.pid_file();
    let contents = match std::fs::read_to_string(&pid_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(path = %pid_path.display(), error = %e, "PID file not readable");
            return None;
        }
    };
    let info: ServerInfo = match serde_json::from_str(&contents) {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!(path = %pid_path.display(), error = %e, "PID file corrupt");
            return None;
        }
    };

    // Check process is alive.
    if !process_alive(info.pid) {
        tracing::debug!(pid = info.pid, "stale PID file — process not alive, removing");
        let _ = std::fs::remove_file(&pid_path);
        return None;
    }

    // ACP-native liveness: probe the socket/TCP the daemon advertises.
    if acp_transport_alive(&info) {
        return Some(info);
    }
    tracing::warn!(
        pid = info.pid,
        acp_socket = ?info.acp_socket,
        acp_tcp = ?info.acp_tcp,
        "akhomed process alive but no reachable ACP transport (advertised: \
         socket={:?}, tcp={:?})",
        info.acp_socket,
        info.acp_tcp,
    );
    None
}

/// Probe whether the daemon's ACP transport is accepting connections.
///
/// Prefers the Unix socket (local, cheapest); falls back to TCP with a short
/// connect timeout. A successful connect proves the ACP accept loop is live;
/// the probe drops the connection immediately without an `initialize` turn,
/// which the server's per-connection task tolerates (it simply ends).
fn acp_transport_alive(info: &ServerInfo) -> bool {
    use std::net::ToSocketAddrs;

    #[cfg(unix)]
    if let Some(sock) = &info.acp_socket
        && std::os::unix::net::UnixStream::connect(sock).is_ok()
    {
        return true;
    }

    if let Some(addr) = &info.acp_tcp
        && let Ok(mut addrs) = addr.to_socket_addrs()
        && let Some(sa) = addrs.next()
        && std::net::TcpStream::connect_timeout(&sa, std::time::Duration::from_secs(2)).is_ok()
    {
        return true;
    }

    false
}

/// Write a PID file for the current akhomed process.
///
/// `acp_socket` / `acp_tcp` describe the ACP transports the daemon is serving
/// (pass `None` for either that is disabled); when at least one is present the
/// record also advertises [`ACP_PROTOCOL_VERSION`].
pub fn write_pid_file(
    paths: &AkhPaths,
    port: u16,
    bind: &str,
    acp_socket: Option<&str>,
    acp_tcp: Option<&str>,
) -> std::io::Result<()> {
    let acp_version = (acp_socket.is_some() || acp_tcp.is_some())
        .then(|| ACP_PROTOCOL_VERSION.to_string());
    let info = ServerInfo {
        pid: std::process::id(),
        port,
        bind: bind.to_string(),
        acp_socket: acp_socket.map(str::to_string),
        acp_tcp: acp_tcp.map(str::to_string),
        acp_version,
    };
    let json = serde_json::to_string_pretty(&info).expect("ServerInfo is always serializable");
    std::fs::write(paths.pid_file(), json)
}

/// Remove the PID file on shutdown.
pub fn remove_pid_file(paths: &AkhPaths) {
    let _ = std::fs::remove_file(paths.pid_file());
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill with signal 0 doesn't actually send a signal;
    // it only checks whether the process exists.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    // On non-unix, fall back to trusting the PID file.
    true
}

// ---------------------------------------------------------------------------
// Daemon status (shared between client & server)
// ---------------------------------------------------------------------------

/// Status of a workspace daemon (background agent task manager).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub total_cycles: usize,
    pub started_at: u64,
    pub trigger_count: usize,
    /// Unix timestamp of last successful session persist.
    #[serde(default)]
    pub last_persist_at: Option<u64>,
    /// Unix timestamp of last continuous learning run.
    #[serde(default)]
    pub last_learning_at: Option<u64>,
    /// Unix timestamp of last sleep/consolidation cycle.
    #[serde(default)]
    pub last_sleep_at: Option<u64>,
    /// Unix timestamp of last goal generation run.
    #[serde(default)]
    pub last_goal_gen_at: Option<u64>,
    /// Number of currently active goals.
    #[serde(default)]
    pub active_goals: usize,
    /// Total symbols in knowledge graph.
    #[serde(default)]
    pub kg_symbols: usize,
    /// Total triples in knowledge graph.
    #[serde(default)]
    pub kg_triples: usize,
    /// Knowledge density: knowledge-bearing triples / total triples [0.0, 1.0].
    ///
    /// Updated after each KG compaction pass. `None` until the first compaction
    /// has run.
    #[serde(default)]
    pub knowledge_density: Option<f32>,
    /// Unix timestamp of the last KG compaction run.
    #[serde(default)]
    pub last_compaction_at: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_server_info_without_acp_fields_parses() {
        // A discovery file written by a pre-ACP binary.
        let legacy = r#"{ "pid": 1234, "port": 8200, "bind": "0.0.0.0" }"#;
        let info: ServerInfo = serde_json::from_str(legacy).expect("legacy ServerInfo must parse");
        assert_eq!(info.pid, 1234);
        assert_eq!(info.port, 8200);
        assert!(info.acp_socket.is_none());
        assert!(info.acp_tcp.is_none());
        assert!(info.acp_version.is_none());
        assert_eq!(info.base_url(), "http://127.0.0.1:8200");
    }

    #[test]
    fn server_info_with_acp_round_trips_and_omits_none() {
        let info = ServerInfo {
            pid: 7,
            port: 8200,
            bind: "127.0.0.1".into(),
            acp_socket: Some("/run/akh-medu/akhomed-acp.sock".into()),
            acp_tcp: None,
            acp_version: Some(ACP_PROTOCOL_VERSION.to_string()),
        };
        let json = serde_json::to_string(&info).expect("serialize");
        // `None` fields are omitted from the record.
        assert!(!json.contains("acp_tcp"), "None acp_tcp must be skipped: {json}");
        assert!(json.contains("akhomed-acp.sock"));
        let back: ServerInfo = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(back.acp_socket.as_deref(), Some("/run/akh-medu/akhomed-acp.sock"));
        assert_eq!(back.acp_version.as_deref(), Some(ACP_PROTOCOL_VERSION));
    }

    // ── ACP-native discovery liveness (P8.2) ────────────────────────────────

    /// Build an `AkhPaths` whose `runtime_dir` (hence `pid_file`) is the given
    /// dir; the other roots are unused by discovery.
    fn test_paths(runtime_dir: &std::path::Path) -> AkhPaths {
        AkhPaths {
            config_dir: runtime_dir.join("config"),
            data_dir: runtime_dir.join("data"),
            state_dir: runtime_dir.join("state"),
            runtime_dir: runtime_dir.to_path_buf(),
            cache_dir: runtime_dir.join("cache"),
        }
    }

    /// A live ACP Unix socket ⇒ discovery succeeds: the probe connects to the
    /// advertised socket and confirms the listener is accepting.
    #[cfg(unix)]
    #[test]
    fn discover_succeeds_when_acp_socket_is_live() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).expect("runtime dir");
        let paths = test_paths(&runtime_dir);

        // Bind a real ACP-style Unix socket so a connect succeeds.
        let sock = tmp.path().join("acp.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind socket");
        let sock_str = sock.to_str().unwrap();

        write_pid_file(&paths, 8200, "127.0.0.1", Some(sock_str), None).expect("write pid");

        let found = discover_server(&paths).expect("live ACP socket should be discovered");
        assert_eq!(found.acp_socket.as_deref(), Some(sock_str));
    }

    /// An advertised-but-dead ACP socket ⇒ discovery returns `None` even though
    /// the PID (this process) is alive: liveness is the transport, not the PID.
    #[cfg(unix)]
    #[test]
    fn discover_fails_when_acp_socket_is_dead() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).expect("runtime dir");
        let paths = test_paths(&runtime_dir);

        // A socket path nobody is listening on.
        let dead = tmp.path().join("nobody.sock");
        write_pid_file(&paths, 8200, "127.0.0.1", Some(dead.to_str().unwrap()), None)
            .expect("write pid");

        assert!(
            discover_server(&paths).is_none(),
            "a dead ACP socket must not be discovered even with a live PID"
        );
    }

    /// The TCP branch of the probe: a bound listener is alive; a refused address
    /// is not.
    #[test]
    fn acp_transport_alive_probes_tcp() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind tcp");
        let addr = listener.local_addr().expect("addr").to_string();
        let live = ServerInfo {
            pid: 1,
            port: 8200,
            bind: "127.0.0.1".into(),
            acp_socket: None,
            acp_tcp: Some(addr),
            acp_version: Some(ACP_PROTOCOL_VERSION.into()),
        };
        assert!(acp_transport_alive(&live), "bound TCP listener should probe alive");

        // Port 1 on localhost has no listener ⇒ connection refused ⇒ not alive.
        let dead = ServerInfo {
            pid: 1,
            port: 8200,
            bind: "127.0.0.1".into(),
            acp_socket: None,
            acp_tcp: Some("127.0.0.1:1".into()),
            acp_version: None,
        };
        assert!(!acp_transport_alive(&dead), "refused TCP address should probe dead");
    }
}
