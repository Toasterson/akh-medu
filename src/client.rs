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

    /// Build a `ServerInfo` that targets a **remote** daemon's ACP TCP endpoint
    /// directly, bypassing local PID-file discovery.
    ///
    /// Used by `akh --remote host:port` (or `AKH_REMOTE`) so a client on one
    /// host can drive an `akhomed` on another over the network. `addr` is the
    /// `host:port` the remote daemon advertises (it must be started with
    /// `AKH_ACP_TCP_PORT`/`AKH_ACP_TCP_BIND`). The `pid`/`bind`/`port` fields are
    /// best-effort metadata — the transport only consults `acp_tcp`.
    pub fn remote_tcp(addr: &str) -> Self {
        let (host, port) = addr
            .rsplit_once(':')
            .and_then(|(h, p)| p.parse::<u16>().ok().map(|p| (h.to_string(), p)))
            .unwrap_or_else(|| (addr.to_string(), 0));
        ServerInfo {
            pid: 0,
            port,
            bind: host,
            acp_socket: None,
            acp_tcp: Some(addr.to_string()),
            acp_version: Some(ACP_PROTOCOL_VERSION.to_string()),
        }
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
///
/// A PID that's alive but whose transport never answers is ambiguous on a
/// single probe (it could be mid-startup, between fork and ACP bind), so this
/// doesn't delete the PID file on the first miss. Instead it counts
/// consecutive failures in a sidecar marker (see [`record_stale_probe`]) and
/// only retires the PID file once [`STALE_ACP_PROBE_THRESHOLD`] consecutive
/// probes — necessarily spanning separate `discover_server` calls, i.e.
/// separate CLI invocations, since the ACP bind takes milliseconds once a
/// daemon is actually up — have failed. Until then it warns once per episode
/// rather than on every command.
/// Tries the caller's own runtime directory first, then the system-service one
/// ([`AkhPaths::system_runtime_dir`]).
///
/// The fallback is what lets an unprivileged `akh` find an akhomed running as a
/// system daemon: the daemon advertises itself in `/var/run/akh-medu` (or
/// `/run/akh-medu`), which is nowhere near the user's `~/.local/state`. Without
/// it, every CLI command against a system daemon reports "No running akhomed
/// daemon found" while the daemon is serving normally.
///
/// User-first ordering is deliberate: a developer running their own akhomed in
/// the foreground must keep talking to *theirs*, not to the system service.
pub fn discover_server(paths: &AkhPaths) -> Option<ServerInfo> {
    let user_pid = paths.pid_file();
    if let Some(info) = discover_at(&user_pid) {
        return Some(info);
    }
    let system_pid = AkhPaths::system_runtime_dir().join("akhomed.pid");
    if system_pid != user_pid {
        return discover_at(&system_pid);
    }
    None
}

/// [`discover_server`] against one specific PID file.
fn discover_at(pid_path: &std::path::Path) -> Option<ServerInfo> {
    let pid_path = pid_path.to_path_buf();
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
        clear_stale_probe(&pid_path);
        return None;
    }

    // ACP-native liveness: probe the socket/TCP the daemon advertises.
    if acp_transport_alive(&info) {
        clear_stale_probe(&pid_path);
        return Some(info);
    }

    let failures = record_stale_probe(&pid_path);
    if failures >= STALE_ACP_PROBE_THRESHOLD {
        tracing::warn!(
            pid = info.pid,
            failures,
            path = %pid_path.display(),
            "akhomed PID file's process is alive but ACP has not answered in \
             {failures} consecutive probes across separate invocations — \
             treating it as stale and removing the PID file",
        );
        let _ = std::fs::remove_file(&pid_path);
        clear_stale_probe(&pid_path);
        return None;
    }

    // Only warn on the first miss of an episode; later probes within the
    // grace window are logged at debug so a stuck daemon doesn't spam a
    // warning on every subsequent command.
    if failures <= 1 {
        tracing::warn!(
            pid = info.pid,
            acp_socket = ?info.acp_socket,
            acp_tcp = ?info.acp_tcp,
            "akhomed process alive but no reachable ACP transport (advertised: \
             socket={:?}, tcp={:?}) — will retire the PID file after {} \
             consecutive failed probes",
            info.acp_socket,
            info.acp_tcp,
            STALE_ACP_PROBE_THRESHOLD,
        );
    } else {
        tracing::debug!(
            pid = info.pid,
            failures,
            threshold = STALE_ACP_PROBE_THRESHOLD,
            "akhomed process alive but ACP still unreachable (repeat probe)",
        );
    }
    None
}

/// Consecutive failed ACP-liveness probes (PID alive, transport unreachable)
/// [`discover_server`] tolerates before concluding a PID file is stale and
/// removing it. See [`discover_server`] for the rationale.
const STALE_ACP_PROBE_THRESHOLD: u32 = 3;

/// Sidecar marker tracking consecutive failed ACP-liveness probes for a given
/// PID file. Discovery calls are one-shot CLI invocations with no shared
/// in-process state, so the failure count has to persist to disk to be
/// visible across "N failed probes" — same directory as the PID file, same
/// lifecycle.
#[derive(Debug, Serialize, Deserialize)]
struct StaleProbeMarker {
    /// Consecutive failed probes so far in the current episode.
    count: u32,
}

/// Path to `pid_path`'s stale-probe marker (`<pid_file>.stale-probes`).
fn stale_probe_marker_path(pid_path: &std::path::Path) -> std::path::PathBuf {
    let mut name = pid_path
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_else(|| std::ffi::OsString::from("akhomed.pid"));
    name.push(".stale-probes");
    pid_path.with_file_name(name)
}

/// Record one more failed ACP-liveness probe for `pid_path` and return the
/// updated consecutive-failure count. Best-effort: a marker read/write
/// failure just restarts the count at 1 rather than erroring — this is
/// diagnostic bookkeeping, not correctness-critical state.
fn record_stale_probe(pid_path: &std::path::Path) -> u32 {
    let marker_path = stale_probe_marker_path(pid_path);
    let count = std::fs::read_to_string(&marker_path)
        .ok()
        .and_then(|s| serde_json::from_str::<StaleProbeMarker>(&s).ok())
        .map_or(0, |m| m.count)
        + 1;
    if let Ok(json) = serde_json::to_string(&StaleProbeMarker { count }) {
        let _ = std::fs::write(&marker_path, json);
    }
    count
}

/// Clear `pid_path`'s stale-probe marker (transport answered again, or the
/// PID file itself was just removed).
fn clear_stale_probe(pid_path: &std::path::Path) {
    let _ = std::fs::remove_file(stale_probe_marker_path(pid_path));
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
        && unix_socket_has_listener(std::path::Path::new(sock))
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

/// True if a process is actively accepting connections on the given Unix
/// socket path. Connect-and-drop performs no ACP handshake — the server's
/// per-connection task tolerates an immediate disconnect (it just ends) — so
/// this is a pure liveness probe, safe to call speculatively.
///
/// Shared by [`discover_server`]'s ACP-native liveness check and by
/// akhomed's single-instance startup guard: a socket *file* on disk proves
/// nothing (crashes and unclean shutdowns leave it behind), only a live
/// listener does.
#[cfg(unix)]
pub fn unix_socket_has_listener(path: &std::path::Path) -> bool {
    std::os::unix::net::UnixStream::connect(path).is_ok()
}

#[cfg(not(unix))]
pub fn unix_socket_has_listener(_path: &std::path::Path) -> bool {
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

/// Whether `pid` names a live process.
///
/// `kill(pid, 0)` sends no signal; it only asks whether the process exists and
/// whether we could signal it. Those are two different questions, and
/// conflating them breaks the daemon deployment:
///
/// - `0`      — exists, and we may signal it
/// - `EPERM`  — **exists**, but it belongs to another user
/// - `ESRCH`  — no such process
///
/// A `== 0` test reports `EPERM` as dead. When akhomed runs as a system
/// LaunchDaemon (root) and the `akh` CLI runs as an ordinary user, every
/// probe takes the `EPERM` path — so the client concluded the daemon was
/// stale, **deleted its PID file**, and reported "No running akhomed daemon
/// found" while the daemon was serving ACP on its port the whole time. The
/// deletion also made the failure sticky: the next invocation had no PID file
/// to read at all.
///
/// Only `ESRCH` means dead. Anything else means the process is there.
#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: `kill` with signal 0 delivers nothing; it performs existence and
    // permission checks only, and cannot affect memory.
    if unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
        return true;
    }
    !matches!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    )
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
    fn remote_tcp_targets_tcp_endpoint_without_socket() {
        let info = ServerInfo::remote_tcp("example.internal:8201");
        assert_eq!(info.acp_tcp.as_deref(), Some("example.internal:8201"));
        assert!(
            info.acp_socket.is_none(),
            "remote_tcp must not advertise a socket"
        );
        assert_eq!(info.bind, "example.internal");
        assert_eq!(info.port, 8201);
        assert_eq!(info.acp_version.as_deref(), Some(ACP_PROTOCOL_VERSION));
    }

    #[test]
    fn remote_tcp_tolerates_unparseable_addr() {
        // No port / garbage still yields a usable acp_tcp the transport can try.
        let info = ServerInfo::remote_tcp("host-only");
        assert_eq!(info.acp_tcp.as_deref(), Some("host-only"));
        assert_eq!(info.port, 0);
    }

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
    /// A process owned by another user is alive, not dead.
    ///
    /// pid 1 (`launchd`/`init`) is always running and always root-owned, so on
    /// any non-root test run `kill(1, 0)` takes the `EPERM` path — the exact
    /// path an unprivileged `akh` CLI takes when probing a root akhomed
    /// LaunchDaemon. Reporting that as dead made the client delete the live
    /// daemon's PID file.
    #[cfg(unix)]
    #[test]
    fn process_owned_by_another_user_counts_as_alive() {
        // SAFETY: argument-less credential read.
        if unsafe { libc::geteuid() } == 0 {
            return; // as root the probe succeeds outright; nothing to distinguish
        }
        assert!(
            process_alive(1),
            "pid 1 is root-owned and always running; EPERM must not read as dead"
        );
    }

    /// Only ESRCH means dead.
    #[cfg(unix)]
    #[test]
    fn a_pid_that_does_not_exist_counts_as_dead() {
        // Well above any real pid; `kill` returns ESRCH rather than EPERM.
        assert!(
            !process_alive(0x7FFF_FFFE),
            "a nonexistent pid must be reported dead so stale PID files are cleaned up"
        );
    }

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

    /// A single dead-transport probe must not delete the PID file — it could
    /// just be a startup race — but `STALE_ACP_PROBE_THRESHOLD` consecutive
    /// failures across separate `discover_server` calls means the PID file
    /// (and its stale-probe marker) get retired automatically instead of
    /// warning forever.
    #[cfg(unix)]
    #[test]
    fn discover_removes_stale_pid_file_after_repeated_dead_probes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).expect("runtime dir");
        let paths = test_paths(&runtime_dir);

        let dead = tmp.path().join("nobody.sock");
        write_pid_file(&paths, 8200, "127.0.0.1", Some(dead.to_str().unwrap()), None)
            .expect("write pid");

        for _ in 0..STALE_ACP_PROBE_THRESHOLD - 1 {
            assert!(discover_server(&paths).is_none(), "dead transport ⇒ None");
            assert!(
                paths.pid_file().exists(),
                "PID file must survive fewer than {STALE_ACP_PROBE_THRESHOLD} failed probes"
            );
        }

        // The threshold-th consecutive failure retires the artifact.
        assert!(discover_server(&paths).is_none());
        assert!(
            !paths.pid_file().exists(),
            "PID file must be removed after {STALE_ACP_PROBE_THRESHOLD} consecutive \
             failed ACP probes"
        );
        assert!(
            !stale_probe_marker_path(&paths.pid_file()).exists(),
            "stale-probe marker must be cleaned up alongside the PID file"
        );
    }

    /// A transport that recovers before the threshold clears the marker
    /// instead of carrying a stale failure count into a later, unrelated
    /// episode.
    #[cfg(unix)]
    #[test]
    fn discover_clears_stale_probe_marker_on_recovery() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let runtime_dir = tmp.path().join("run");
        std::fs::create_dir_all(&runtime_dir).expect("runtime dir");
        let paths = test_paths(&runtime_dir);

        let dead = tmp.path().join("nobody.sock");
        write_pid_file(&paths, 8200, "127.0.0.1", Some(dead.to_str().unwrap()), None)
            .expect("write pid");
        assert!(discover_server(&paths).is_none(), "dead transport ⇒ None");
        assert!(
            stale_probe_marker_path(&paths.pid_file()).exists(),
            "one failed probe must record a marker"
        );

        // Now advertise a live socket and re-discover.
        let sock = tmp.path().join("acp.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind socket");
        write_pid_file(&paths, 8200, "127.0.0.1", Some(sock.to_str().unwrap()), None)
            .expect("rewrite pid");

        assert!(discover_server(&paths).is_some(), "live transport ⇒ Some");
        assert!(
            !stale_probe_marker_path(&paths.pid_file()).exists(),
            "a successful probe must clear the stale-probe marker"
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
