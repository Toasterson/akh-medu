//! `RemoteBackend`: the russh-over-SSH executor (Phase 48c, ADR 082).
//!
//! Implements the [`Backend`] seam (`src/backend/mod.rs`) over a real SSH
//! session to a target host — the "russh remote" backend ADR 082 §3/§10
//! describes, feature-gated behind `remote-executor` so the default build
//! (and the lean `akh` client) never links an SSH stack.
//!
//! ## What this is, precisely
//!
//! - **Auth is plain provisioned keys** (ADR 086 §1): a private key file +
//!   a [`HostKeyPolicy`]. 48d's OIDC → short-lived-SSH-certificate path is
//!   explicitly deferred — this backend has no cert/OIDC code at all.
//! - **`exec` runs the SSH `exec` channel request** (one channel per call —
//!   ADR 082 §5's "one SSH connection per session, one channel per agent"),
//!   capturing `ChannelMsg::Data`/`ExtendedData`/`ExitStatus` into the same
//!   [`ExecOutcome`] shape [`InProcessBackend`](super::InProcessBackend)
//!   produces, so `ShellExecTool` behaves identically regardless of backend.
//! - **File verbs run as shell commands over the same exec channel**, not
//!   SFTP. ADR 082 §2 leaves this an open choice ("SFTP subsystem or shell
//!   commands"); shell wins here because the `Filesystem` surface this trait
//!   needs (`read_to_string`/`write`/`read_dir`/`exists`/`file_size`/
//!   `is_symlink`) is entirely text/metadata, so `cat`/`base64`/`ls`/`test`/
//!   `wc` cover it without a second protocol negotiation, a second dependency
//!   (`russh-sftp`), or a second server-side test fixture. A future
//!   refinement can swap in SFTP for large-binary-safe writes; see the
//!   `write` doc comment for the size caveat that motivates it.
//! - **No client-side path sandboxing.** ADR 082 §8: the OS is the *sole*
//!   security boundary. Unlike [`LocalFs`](crate::fs::LocalFs), this backend
//!   does not reject `..` or canonicalize-and-compare against `read_roots`/
//!   `write_roots` — those fields are advisory scope hints for tools that
//!   walk with no explicit path (mirroring `Filesystem::read_roots`'s
//!   existing contract for "tools that scope a walk"). Containment is
//!   whatever the remote `os_user`'s own permissions allow; reproducing a
//!   sandbox here would be a second, driftable policy model, which is
//!   exactly what ADR 082 §8 rejects.
//! - **The synchronous `Backend` trait bridges russh's async API by owning a
//!   dedicated multi-thread [`tokio::runtime::Runtime`]** and calling
//!   `runtime.block_on(..)` inside every trait method — the same
//!   spawn-a-runtime-and-block_on shape `src/main.rs` and `src/tui/remote.rs`
//!   already use for their sync-context ACP calls. This was a live option
//!   ADR 082/Phase-50 flagged ("resolve this explicitly... do NOT change the
//!   `Backend` trait's signature to async without flagging it") — the
//!   alternative (making `Backend::exec` async) would ripple through every
//!   tool and the whole `ToolRegistry` dispatch path for a seam only this one
//!   backend needs; that's out of scope for 48c and is *not* done here.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use miette::Diagnostic;
use russh::client;
use russh::keys::{HashAlg, PrivateKey, PrivateKeyWithHashAlg, PublicKey};
use thiserror::Error;

use crate::backend::{Backend, Confinement, ExecError, ExecLocus, ExecOutcome, ExecRequest, ExecResult, Posture};
use crate::fs::{Filesystem, FsError, FsResult};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from establishing or using a [`RemoteBackend`] SSH session.
///
/// Kept as its own rich subsystem error (per the repo's error-handling
/// convention) rather than folded into [`ExecError`]/[`FsError`] directly;
/// [`ExecError::Remote`] wraps it transparently at the `Backend::exec`
/// boundary, and file verbs flatten it into [`FsError::Io`] (see
/// [`FsError`]'s existing shape — it has no "remote transport failed" variant
/// of its own, and adding one would ripple across every `Filesystem`
/// consumer for a detail only this backend produces).
#[derive(Debug, Error, Diagnostic)]
pub enum RemoteError {
    /// The internal tokio runtime this backend drives russh through could
    /// not be built (thread/fd exhaustion) — an environment problem, not a
    /// remote-host problem.
    #[error("failed to build the internal tokio runtime for the SSH connection: {source}")]
    #[diagnostic(
        code(akh::backend::remote::runtime),
        help(
            "This is a local resource problem (thread or file-descriptor exhaustion), \
             not the remote host's fault. Check the process's thread/fd limits."
        )
    )]
    Runtime {
        #[source]
        source: std::io::Error,
    },

    /// The TCP connect + SSH handshake did not complete before
    /// [`RemoteBackendConfig::connect_timeout`].
    #[error("connecting to {host}:{port} timed out after {}s", timeout.as_secs())]
    #[diagnostic(
        code(akh::backend::remote::connect_timeout),
        help(
            "The SSH handshake did not finish in time. Check the host is reachable and \
             sshd is listening on port {port}; a firewall silently dropping the SYN looks \
             identical to this from here."
        )
    )]
    ConnectTimeout {
        host: String,
        port: u16,
        timeout: Duration,
    },

    /// The private key file could not be loaded / parsed.
    #[error("failed to load private key from \"{}\": {source}", path.display())]
    #[diagnostic(
        code(akh::backend::remote::key_load),
        help(
            "Verify the path points at a readable OpenSSH-format private key. Encrypted \
             (passphrase-protected) keys are not supported by this lab-posture backend \
             (ADR 086 §1) — provision an unencrypted key for the akh's OS principal."
        )
    )]
    KeyLoad {
        path: PathBuf,
        #[source]
        source: russh::keys::Error,
    },

    /// The server rejected the offered key outright (no partial success,
    /// no further methods worth trying with what we have).
    #[error("SSH publickey auth rejected for {user}@{host}:{port}")]
    #[diagnostic(
        code(akh::backend::remote::auth_rejected),
        help(
            "The target did not accept this key for user \"{user}\". Confirm the matching \
             public key is installed in that OS principal's authorized_keys on {host} \
             (ADR 082 §6 — plain provisioned keys, no cert/OIDC in this build)."
        )
    )]
    AuthRejected {
        user: String,
        host: String,
        port: u16,
    },

    /// The server's host key did not satisfy the configured [`HostKeyPolicy`].
    #[error("SSH host-key verification failed for {host}:{port} (policy: {policy})")]
    #[diagnostic(
        code(akh::backend::remote::host_key_rejected),
        help(
            "The server's offered host key did not match the \"{policy}\" policy. If this \
             is a known-good host on its first connection, add its key to the known_hosts \
             file (or switch to `HostKeyPolicy::Pinned` with the expected key) rather than \
             falling back to `InsecureAcceptAny` outside a lab/loopback context."
        )
    )]
    HostKeyRejected {
        host: String,
        port: u16,
        policy: &'static str,
    },

    /// A remote command outlived [`ExecRequest::timeout`]. The process was
    /// signalled and the channel closed on a best-effort basis; SSH gives no
    /// delivery guarantee for either (ADR 082 risk register).
    #[error("remote command timed out after {}s: {command}", timeout.as_secs())]
    #[diagnostic(
        code(akh::backend::remote::exec_timeout),
        help(
            "A SIGKILL channel-signal request and channel close were sent, but SSH does \
             not guarantee the remote process actually received or honored them — unlike \
             a local `kill`, there is no OS-level confirmation from here. Re-check with a \
             status command (e.g. `pgrep`) before assuming the work stopped."
        )
    )]
    ExecTimeout {
        command: String,
        timeout: Duration,
    },

    /// Any other russh transport-level failure (connection reset, channel
    /// open rejected, protocol error, …).
    #[error("SSH transport error: {0}")]
    #[diagnostic(code(akh::backend::remote::transport))]
    Transport(#[from] russh::Error),

    /// Any other russh key-handling failure (known_hosts parsing, key
    /// decoding, …) not already covered by [`RemoteError::KeyLoad`].
    #[error("SSH key error: {0}")]
    #[diagnostic(code(akh::backend::remote::keys))]
    Keys(#[from] russh::keys::Error),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the client verifies the remote host's identity at the SSH transport
/// layer. ADR 086 §1 locks this backend to plain-key auth for the lab; this
/// enum is the parallel decision for the *other* half of SSH trust (who is
/// the client talking to), independent of how the client itself authenticates.
#[derive(Debug, Clone)]
pub enum HostKeyPolicy {
    /// Verify against an OpenSSH `known_hosts` file. `path: None` uses the
    /// default (`~/.ssh/known_hosts` via [`russh::keys::check_known_hosts`]).
    KnownHosts { path: Option<PathBuf> },
    /// Accept only this exact pinned public key — no filesystem lookup. Used
    /// by this module's own loopback test (pins the ephemeral test-fixture
    /// server key) and suits provisioning flows that already know the target
    /// key out of band.
    Pinned(Box<PublicKey>),
    /// Accept any host key, no verification at all. **Lab/loopback only** —
    /// logs a loud warning on every connect. Never point this at a real
    /// exec1 over an untrusted network; it removes SSH's only defense
    /// against a MITM impersonating the target.
    InsecureAcceptAny,
}

impl HostKeyPolicy {
    fn label(&self) -> &'static str {
        match self {
            Self::KnownHosts { .. } => "KnownHosts",
            Self::Pinned(_) => "Pinned",
            Self::InsecureAcceptAny => "InsecureAcceptAny",
        }
    }
}

/// Configuration for a [`RemoteBackend`] connection.
///
/// `read_roots`/`write_roots` are **advisory only** (ADR 082 §8) — see the
/// module doc's "no client-side path sandboxing" note. They exist purely so
/// tools that scope an unrooted walk (Glob/Grep with no explicit path) have
/// something to walk, matching [`Filesystem::read_roots`]'s existing
/// contract; they enforce nothing.
#[derive(Debug, Clone)]
pub struct RemoteBackendConfig {
    /// Target host (DNS name or address).
    pub host: String,
    /// Target port (usually 22).
    pub port: u16,
    /// The remote OS principal to authenticate as — "the akh is a sysadmin
    /// who is already logged in" (ADR 082) as this user.
    pub user: String,
    /// Path to an unencrypted OpenSSH-format private key (ADR 086 §1: plain
    /// provisioned keys, no passphrase support in this build).
    pub private_key_path: PathBuf,
    /// How the server's host key is verified.
    pub host_key_policy: HostKeyPolicy,
    /// Deadline for the TCP connect + SSH handshake + auth.
    pub connect_timeout: Duration,
    /// Deadline for the shell commands the `Filesystem` verbs issue
    /// internally (`cat`/`ls`/`test`/`wc`/`base64`) — independent of
    /// [`ExecRequest::timeout`], which only bounds the `exec` verb.
    pub file_op_timeout: Duration,
    /// Advisory read scope (see struct doc) — not enforced.
    pub read_roots: Vec<PathBuf>,
    /// Advisory write scope (see struct doc) — not enforced. An empty vec
    /// means "no write root configured", so `write()` fails closed with
    /// [`FsError::WritesDisabled`] *without* an SSH round trip, same as
    /// [`crate::fs::LocalFs`]'s `is_writable() == false` case.
    pub write_roots: Vec<PathBuf>,
}

impl RemoteBackendConfig {
    /// A connection to `user@host:22` authenticating with `private_key_path`,
    /// verifying the host key against the default `known_hosts`, 10s connect
    /// timeout, 30s file-op timeout, and no advisory roots (fully closed —
    /// call [`Self::with_roots`] to open reads/writes).
    pub fn new(
        host: impl Into<String>,
        user: impl Into<String>,
        private_key_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            host: host.into(),
            port: 22,
            user: user.into(),
            private_key_path: private_key_path.into(),
            host_key_policy: HostKeyPolicy::KnownHosts { path: None },
            connect_timeout: Duration::from_secs(10),
            file_op_timeout: Duration::from_secs(30),
            read_roots: Vec::new(),
            write_roots: Vec::new(),
        }
    }

    /// Override the port (e.g. a non-standard sshd port for exec1).
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Override the host-key verification policy.
    pub fn with_host_key_policy(mut self, policy: HostKeyPolicy) -> Self {
        self.host_key_policy = policy;
        self
    }

    /// Set the advisory read/write scope roots.
    pub fn with_roots(mut self, read_roots: Vec<PathBuf>, write_roots: Vec<PathBuf>) -> Self {
        self.read_roots = read_roots;
        self.write_roots = write_roots;
        self
    }

    /// Override the connect timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
}

// ---------------------------------------------------------------------------
// Client handler (host-key verification callback)
// ---------------------------------------------------------------------------

struct ClientHandler {
    policy: HostKeyPolicy,
    host: String,
    port: u16,
}

impl ClientHandler {
    fn new(config: &RemoteBackendConfig) -> Self {
        Self {
            policy: config.host_key_policy.clone(),
            host: config.host.clone(),
            port: config.port,
        }
    }
}

impl client::Handler for ClientHandler {
    type Error = RemoteError;

    async fn check_server_key(&mut self, server_public_key: &PublicKey) -> Result<bool, Self::Error> {
        // Returning `Err` (rather than `Ok(false)`) on a mismatch lets our
        // rich `HostKeyRejected` diagnostic reach the caller — russh
        // propagates a handler error verbatim as the connect error, whereas
        // `Ok(false)` collapses to its generic `Error::UnknownKey`.
        let accepted = match &self.policy {
            HostKeyPolicy::InsecureAcceptAny => {
                tracing::warn!(
                    target: "akh::backend::remote",
                    host = %self.host,
                    port = self.port,
                    "accepting SSH host key WITHOUT verification (HostKeyPolicy::InsecureAcceptAny) \
                     — lab/loopback posture only; never point this at a production exec1"
                );
                true
            }
            // Compare on `key_data()` only — `PublicKey`'s derived `PartialEq`
            // also compares the free-text comment, which the server-presented
            // host key carries empty while a pin loaded from a file/authorized
            // key often has one; the cryptographic identity is the key data.
            HostKeyPolicy::Pinned(expected) => expected.key_data() == server_public_key.key_data(),
            HostKeyPolicy::KnownHosts { path: Some(path) } => {
                russh::keys::check_known_hosts_path(&self.host, self.port, server_public_key, path)?
            }
            HostKeyPolicy::KnownHosts { path: None } => {
                russh::keys::check_known_hosts(&self.host, self.port, server_public_key)?
            }
        };
        if accepted {
            Ok(true)
        } else {
            Err(RemoteError::HostKeyRejected {
                host: self.host.clone(),
                port: self.port,
                policy: self.policy.label(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// RemoteBackend
// ---------------------------------------------------------------------------

/// The russh-backed remote executor `Backend` (ADR 082 §3/§10, Phase 48c).
///
/// One `RemoteBackend` == one bound SSH session: every file verb and every
/// `exec` call lands on the same host, as the same OS principal, over the
/// same connection (ADR 082 §10's file/exec coherence requirement) — new
/// channels are opened per call (§5's "one channel per agent" multiplexing),
/// but never a new TCP connection.
///
/// Field order is deliberate: `handle` drops before `runtime` so the
/// connection-driving background task can observe the sender going away and
/// exit *before* the runtime shutdown waits for outstanding tasks.
pub struct RemoteBackend {
    config: RemoteBackendConfig,
    handle: Arc<client::Handle<ClientHandler>>,
    runtime: tokio::runtime::Runtime,
    /// Stable id for *this connection*, minted once at [`Self::connect`] /
    /// [`Self::connect_over_stream`] and reused by every [`Backend::locus`]
    /// call for the lifetime of the session (ADR 082 §7). In a production
    /// OIDC→SSH-cert deployment (48d, deferred) this is exactly the value
    /// that would be minted as the certificate's key-id, so a correlation-
    /// index query can walk straight from an audit entry to the matching
    /// sshd/auditd record for this session. Derived (not random) so it is
    /// reproducible from connection parameters + process identity rather
    /// than requiring a new UUID dependency.
    correlation_id: String,
}

impl RemoteBackend {
    /// Connect to the target described by `config` over a real TCP socket.
    ///
    /// Blocks the calling thread until the TCP connect, SSH handshake, and
    /// publickey authentication all complete (or `config.connect_timeout`
    /// elapses) — this is the one call site where the sync/async bridge is
    /// unavoidable even in spirit, since there is no session to hand work to
    /// yet.
    pub fn connect(config: RemoteBackendConfig) -> Result<Self, RemoteError> {
        let runtime = new_runtime()?;
        let handle = runtime.block_on(Self::connect_tcp(&config))?;
        let correlation_id = mint_correlation_id(&config);
        Ok(Self {
            config,
            handle: Arc::new(handle),
            runtime,
            correlation_id,
        })
    }

    async fn connect_tcp(config: &RemoteBackendConfig) -> Result<client::Handle<ClientHandler>, RemoteError> {
        let ssh_config = Arc::new(client::Config::default());
        let handler = ClientHandler::new(config);
        let addr = (config.host.as_str(), config.port);
        let mut handle = tokio::time::timeout(config.connect_timeout, client::connect(ssh_config, addr, handler))
            .await
            .map_err(|_elapsed| RemoteError::ConnectTimeout {
                host: config.host.clone(),
                port: config.port,
                timeout: config.connect_timeout,
            })??;
        Self::authenticate(&mut handle, config).await?;
        Ok(handle)
    }

    /// Test-only constructor: connect over an already-established duplex
    /// stream (the loopback russh-server fixture below) instead of a TCP
    /// socket, driven by a caller-supplied runtime so the fixture's server
    /// task and this client share one executor.
    #[cfg(test)]
    fn connect_over_stream<S>(
        config: RemoteBackendConfig,
        runtime: tokio::runtime::Runtime,
        stream: S,
    ) -> Result<Self, RemoteError>
    where
        S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let handle = runtime.block_on(async {
            let ssh_config = Arc::new(client::Config::default());
            let handler = ClientHandler::new(&config);
            let mut handle = client::connect_stream(ssh_config, stream, handler).await?;
            Self::authenticate(&mut handle, &config).await?;
            Ok::<_, RemoteError>(handle)
        })?;
        let correlation_id = mint_correlation_id(&config);
        Ok(Self {
            config,
            handle: Arc::new(handle),
            runtime,
            correlation_id,
        })
    }

    async fn authenticate(
        handle: &mut client::Handle<ClientHandler>,
        config: &RemoteBackendConfig,
    ) -> Result<(), RemoteError> {
        let key = load_private_key(&config.private_key_path)?;
        let result = handle.authenticate_publickey(config.user.clone(), key).await?;
        if result.success() {
            Ok(())
        } else {
            Err(RemoteError::AuthRejected {
                user: config.user.clone(),
                host: config.host.clone(),
                port: config.port,
            })
        }
    }

    /// Run `command` (a full shell command line — the SSH `exec` request
    /// payload) over a fresh channel, blocking the calling thread until it
    /// exits or `timeout` elapses.
    fn run_shell(&self, command: String, timeout: Duration) -> Result<ExecOutcome, RemoteError> {
        self.runtime.block_on(Self::run_shell_async(Arc::clone(&self.handle), command, timeout))
    }

    async fn run_shell_async(
        handle: Arc<client::Handle<ClientHandler>>,
        command: String,
        timeout: Duration,
    ) -> Result<ExecOutcome, RemoteError> {
        let mut channel = handle.channel_open_session().await?;
        channel.exec(true, command.clone().into_bytes()).await?;

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut exit_code: Option<u32> = None;

        let drain = async {
            loop {
                match channel.wait().await {
                    Some(russh::ChannelMsg::Data { data }) => stdout.extend_from_slice(&data),
                    Some(russh::ChannelMsg::ExtendedData { data, ext }) => {
                        // SSH_EXTENDED_DATA_STDERR == 1 (RFC 4254 §5.2).
                        if ext == 1 {
                            stderr.extend_from_slice(&data);
                        }
                    }
                    Some(russh::ChannelMsg::ExitStatus { exit_status }) => {
                        exit_code = Some(exit_status);
                    }
                    Some(russh::ChannelMsg::ExitSignal { .. }) => {
                        // Killed by signal rather than exiting normally —
                        // mirror InProcessBackend's `status.code().unwrap_or(-1)`
                        // convention for "no exit code".
                        exit_code = None;
                    }
                    Some(russh::ChannelMsg::Close) | None => break,
                    _ => {}
                }
            }
        };

        match tokio::time::timeout(timeout, drain).await {
            Ok(()) => Ok(ExecOutcome {
                exit_code: exit_code.map(|c| c as i32).unwrap_or(-1),
                success: exit_code == Some(0),
                stdout,
                stderr,
            }),
            Err(_elapsed) => {
                // Best-effort only — SSH gives no delivery guarantee for
                // either of these (ADR 082 risk register, and this error's
                // own diagnostic help text).
                let _ = channel.signal(russh::Sig::KILL).await;
                let _ = channel.close().await;
                Err(RemoteError::ExecTimeout { command, timeout })
            }
        }
    }

    /// Run a shell command for an internal file-verb helper, using the
    /// configured `file_op_timeout` and flattening any `RemoteError` into
    /// [`FsError::Io`] (the file-verb error surface has no "SSH transport
    /// failed" variant of its own — see the module doc).
    fn run_file_op(&self, path: &Path, command: String) -> FsResult<ExecOutcome> {
        self.run_shell(command, self.config.file_op_timeout)
            .map_err(|source| FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(source.to_string()),
            })
    }
}

/// Mint a per-connection correlation id: a blake3 hash of the connection
/// parameters (host, port, user, key path) plus this process's pid and the
/// current time, hex-truncated to 16 characters. Not cryptographically
/// meaningful — just short, unique-enough-in-practice, and dependency-free
/// (no new `uuid` crate for a value that only needs to distinguish sessions
/// in an audit join, not resist an adversary).
fn mint_correlation_id(config: &RemoteBackendConfig) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let material = format!(
        "{}:{}:{}:{}:{}:{}",
        config.host,
        config.port,
        config.user,
        config.private_key_path.display(),
        std::process::id(),
        nanos
    );
    let digest = blake3::hash(material.as_bytes());
    format!("remote-{}", &digest.to_hex()[..16])
}

fn new_runtime() -> Result<tokio::runtime::Runtime, RemoteError> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("akh-remote-backend")
        .enable_all()
        .build()
        .map_err(|source| RemoteError::Runtime { source })
}

fn load_private_key(path: &Path) -> Result<PrivateKeyWithHashAlg, RemoteError> {
    let key = PrivateKey::read_openssh_file(path).map_err(|source| RemoteError::KeyLoad {
        path: path.to_path_buf(),
        source: russh::keys::Error::from(source),
    })?;
    // `hash_alg` is ignored for non-RSA keys (see `PrivateKeyWithHashAlg::new`
    // doc); Sha256 is the modern default when the key *is* RSA.
    Ok(PrivateKeyWithHashAlg::new(Arc::new(key), Some(HashAlg::Sha256)))
}

// ---------------------------------------------------------------------------
// Shell quoting helpers for the file-verb shell commands
// ---------------------------------------------------------------------------

/// POSIX single-quote a string for safe embedding in a shell command line.
fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn shell_quote_path(p: &Path) -> String {
    shell_quote(&p.to_string_lossy())
}

/// Whether `name` is safe to appear on the left of `export NAME=...` without
/// quoting (POSIX shell identifier: `[A-Za-z_][A-Za-z0-9_]*`).
fn is_safe_env_key(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Build the full shell command line for [`Backend::exec`]: an optional
/// `cd`, optional `export`s, then the caller's command — the same shape a
/// hand-typed `ssh host 'cd X && export Y=Z; cmd'` would use, since the SSH
/// exec-request protocol has no first-class cwd/env fields of its own (only
/// an optional, commonly-disabled `SetEnv` request — see the doc on
/// [`ExecRequest`]'s env handling below).
///
/// Environment variables with names that are not safe POSIX shell
/// identifiers are skipped (with a warning) rather than failing the whole
/// command — this is the one place `RemoteBackend` diverges from
/// [`InProcessBackend`](super::InProcessBackend), which passes env as true
/// process attributes via `std::process::Command::env` and never needs to
/// validate the key at all.
fn build_command(request: &ExecRequest) -> String {
    let mut out = String::new();
    if let Some(cwd) = &request.cwd {
        out.push_str("cd ");
        out.push_str(&shell_quote_path(cwd));
        out.push_str(" && ");
    }
    for (key, value) in &request.env {
        if !is_safe_env_key(key) {
            tracing::warn!(
                target: "akh::backend::remote",
                key = %key,
                "skipping environment variable with an unsafe name for remote exec"
            );
            continue;
        }
        out.push_str("export ");
        out.push_str(key);
        out.push('=');
        out.push_str(&shell_quote(value));
        out.push_str("; ");
    }
    out.push_str(&request.command);
    out
}

// ---------------------------------------------------------------------------
// Filesystem impl — shell commands over the same exec channel
// ---------------------------------------------------------------------------

impl Filesystem for RemoteBackend {
    fn read_to_string(&self, path: &Path) -> FsResult<String> {
        let command = format!("cat -- {}", shell_quote_path(path));
        let outcome = self.run_file_op(path, command)?;
        if !outcome.success {
            return Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(String::from_utf8_lossy(&outcome.stderr).into_owned()),
            });
        }
        Ok(String::from_utf8_lossy(&outcome.stdout).into_owned())
    }

    /// Writes go over the exec channel as `printf ... | openssl base64 -d >
    /// path`: the content is base64-encoded client-side and embedded directly
    /// in the command line (base64's alphabet is quote-safe), avoiding a
    /// second stdin-streaming round trip for what is normally config/
    /// text-sized content. **Caveat**: this roughly triples the wire size and
    /// holds the whole payload in one command-line argument, so it is not the
    /// right path for large binary transfers — a future SFTP-backed write
    /// would remove that ceiling (see the module doc's SFTP-vs-shell note).
    ///
    /// **Decoder is `openssl base64 -d`, not GNU coreutils `base64 -d`.**
    /// Found live against the illumos `exec1` guest (Phase 50 W2b): OpenIndiana's
    /// default image ships no `base64` binary at all (`/usr/bin`, `/usr/gnu/bin`
    /// — nowhere on `$PATH`), so the original GNU-coreutils-only invocation
    /// failed closed with "command not found" on every write. `openssl`
    /// (unlike `base64`) is effectively universal — present on both this
    /// illumos image and every Linux distro tested — and its `base64`
    /// subcommand decodes the same alphabet. It needs the input
    /// newline-terminated to flush the last line (hence `printf '%s\n'`
    /// below, not `printf '%s'`); verified byte-exact for both short and
    /// 300-byte payloads on the live guest.
    fn write(&self, path: &Path, contents: &str) -> FsResult<()> {
        if self.config.write_roots.is_empty() {
            return Err(FsError::WritesDisabled);
        }
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(contents.as_bytes());
        let mut command = String::new();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            command.push_str("mkdir -p -- ");
            command.push_str(&shell_quote_path(parent));
            command.push_str(" && ");
        }
        command.push_str("printf '%s\\n' ");
        command.push_str(&shell_quote(&encoded));
        command.push_str(" | openssl base64 -d > ");
        command.push_str(&shell_quote_path(path));
        let outcome = self.run_file_op(path, command)?;
        if !outcome.success {
            return Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(String::from_utf8_lossy(&outcome.stderr).into_owned()),
            });
        }
        Ok(())
    }

    fn read_dir(&self, path: &Path) -> FsResult<Vec<PathBuf>> {
        // `test -d` first so a non-directory path fails closed instead of
        // `ls` silently listing just that one file (GNU/BSD `ls` disagree on
        // this — gating on `test -d` sidesteps the portability gap, which
        // matters since exec1 is illumos). Note: `test`/`[` takes no `--`
        // end-of-options marker (POSIX), so the quoted path stands alone;
        // `ls`, which does accept `--`, gets it for leading-dash safety.
        let command = format!(
            "test -d {p} && ls -1a -- {p}",
            p = shell_quote_path(path)
        );
        let outcome = self.run_file_op(path, command)?;
        if !outcome.success {
            return Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(String::from_utf8_lossy(&outcome.stderr).into_owned()),
            });
        }
        let entries = String::from_utf8_lossy(&outcome.stdout)
            .lines()
            .filter(|name| *name != "." && *name != "..")
            .map(|name| path.join(name))
            .collect();
        Ok(entries)
    }

    fn exists(&self, path: &Path) -> bool {
        let command = format!("test -e {}", shell_quote_path(path));
        self.run_file_op(path, command)
            .map(|outcome| outcome.success)
            .unwrap_or(false)
    }

    fn read_roots(&self) -> &[PathBuf] {
        &self.config.read_roots
    }

    fn is_writable(&self) -> bool {
        !self.config.write_roots.is_empty()
    }

    fn file_size(&self, path: &Path) -> FsResult<u64> {
        let command = format!("wc -c < {}", shell_quote_path(path));
        let outcome = self.run_file_op(path, command)?;
        if !outcome.success {
            return Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(String::from_utf8_lossy(&outcome.stderr).into_owned()),
            });
        }
        String::from_utf8_lossy(&outcome.stdout)
            .trim()
            .parse::<u64>()
            .map_err(|source| FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other(format!("unparsable `wc -c` output: {source}")),
            })
    }

    fn is_symlink(&self, path: &Path) -> bool {
        let command = format!("test -L {}", shell_quote_path(path));
        self.run_file_op(path, command)
            .map(|outcome| outcome.success)
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Backend impl
// ---------------------------------------------------------------------------

impl Backend for RemoteBackend {
    fn exec(&self, request: ExecRequest) -> ExecResult<ExecOutcome> {
        let timeout = request.timeout;
        let command = build_command(&request);
        self.run_shell(command, timeout).map_err(ExecError::from)
    }

    fn handshake(&self) -> Posture {
        // No local privilege drop happens inside this backend — the remote
        // sshd already authenticated `config.user` as itself before we ever
        // ran a command, so this is *not* the in-process "no boundary at
        // all" case. But 48f's full self-report (euid/groups/MAC domain/
        // capability set) doesn't exist yet, so this can't honestly claim a
        // *verified* confined posture either. `RemoteUnverified` names that
        // middle state; policy treats it the same as `Unconfined` until 48f
        // lands (ADR 082 §8: "absent assertion ⇒ fail-safe to
        // unconfined/Critical").
        Posture {
            confinement: Confinement::RemoteUnverified,
        }
    }

    /// `backend = "remote"`, `host`/`os_user` come straight from the
    /// connection config (no round trip needed — they're already known),
    /// `correlation_id` is the id minted once at [`Self::connect`] /
    /// [`Self::connect_over_stream`] and held for the session's lifetime.
    fn locus(&self) -> ExecLocus {
        ExecLocus {
            backend: "remote",
            host: Some(self.config.host.clone()),
            os_user: Some(self.config.user.clone()),
            correlation_id: Some(self.correlation_id.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: a loopback russh **server**, in-process, driving the real wire path
// ---------------------------------------------------------------------------
//
// No mock `Backend` — this stands up an actual `russh::server` on one end of
// a `tokio::io::duplex` pipe and this module's real `RemoteBackend` on the
// other, so the test exercises genuine SSH framing, auth, and channel exec,
// not a stand-in. Pointing `RemoteBackendConfig` at exec1 (once Phase 50/W1
// lands) instead of the loopback pipe is the only change needed for the real
// target — swap `connect_over_stream` for `RemoteBackend::connect` and given
// a real `host`/`port`/`private_key_path`.

#[cfg(test)]
mod tests {
    use super::*;
    use russh::server;
    use russh::keys::ssh_key::LineEnding;
    use std::time::Duration as StdDuration;

    // Ephemeral ed25519 test keypairs generated once for this fixture only
    // (`ssh-keygen -t ed25519 -N ""`) — never used against anything but the
    // in-memory loopback pipe below.
    const SERVER_PRIVATE_KEY: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n\
QyNTUxOQAAACBtaFcwc5MNGRk7OZ+q+bWUYakWnKnbQmuotWPBcs2kTQAAAKgEGCaEBBgm\n\
hAAAAAtzc2gtZWQyNTUxOQAAACBtaFcwc5MNGRk7OZ+q+bWUYakWnKnbQmuotWPBcs2kTQ\n\
AAAEBcT93e6xavXWURBPyee4OwnImuyGPRLRXhk4C5Fb8HvG1oVzBzkw0ZGTs5n6r5tZRh\n\
qRacqdtCa6i1Y8FyzaRNAAAAHmFraC1yZW1vdGUtYmFja2VuZC10ZXN0LXNlcnZlcgECAw\n\
QFBgc=\n\
-----END OPENSSH PRIVATE KEY-----\n";

    const CLIENT_PRIVATE_KEY: &str = "-----BEGIN OPENSSH PRIVATE KEY-----\n\
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n\
QyNTUxOQAAACD/8Or4kjzB2ZEybkEjKgrNx4jWs5cpsRXgyrkpng5VdAAAAKjg0Tjy4NE4\n\
8gAAAAtzc2gtZWQyNTUxOQAAACD/8Or4kjzB2ZEybkEjKgrNx4jWs5cpsRXgyrkpng5VdA\n\
AAAEDGXbtroX/QuY38jR/NJY+uLf+evhd07dzbRLulhnL8AP/w6viSPMHZkTJuQSMqCs3H\n\
iNazlymxFeDKuSmeDlV0AAAAHmFraC1yZW1vdGUtYmFja2VuZC10ZXN0LWNsaWVudAECAw\n\
QFBgc=\n\
-----END OPENSSH PRIVATE KEY-----\n";

    /// The in-process test server: accepts one publickey (the fixture's
    /// client key), opens a session channel, and actually runs the exec
    /// request's command via `/bin/sh -c` on *this* (test) host, returning
    /// real stdout/stderr/exit-status back over the wire.
    struct LoopbackServer {
        expected_client_key: PublicKey,
    }

    type ServerError = Box<dyn std::error::Error + Send + Sync>;

    impl server::Handler for LoopbackServer {
        type Error = ServerError;

        async fn auth_publickey(
            &mut self,
            _user: &str,
            public_key: &PublicKey,
        ) -> Result<server::Auth, Self::Error> {
            // Compare key data only, not the comment (see the client-side
            // `Pinned` note) — the wire-presented key carries no comment.
            if public_key.key_data() == self.expected_client_key.key_data() {
                Ok(server::Auth::Accept)
            } else {
                Ok(server::Auth::reject())
            }
        }

        async fn channel_open_session(
            &mut self,
            _channel: russh::Channel<server::Msg>,
            reply: server::ChannelOpenHandle,
            _session: &mut server::Session,
        ) -> Result<(), Self::Error> {
            reply.accept().await;
            Ok(())
        }

        async fn exec_request(
            &mut self,
            channel: russh::ChannelId,
            data: &[u8],
            session: &mut server::Session,
        ) -> Result<(), Self::Error> {
            let command = String::from_utf8_lossy(data).into_owned();
            session.channel_success(channel)?;
            let output = tokio::process::Command::new("/bin/sh")
                .arg("-c")
                .arg(&command)
                .output()
                .await?;
            if !output.stdout.is_empty() {
                session.data(channel, output.stdout)?;
            }
            if !output.stderr.is_empty() {
                session.extended_data(channel, 1, output.stderr)?;
            }
            let code = output.status.code().unwrap_or(-1) as u32;
            session.exit_status_request(channel, code)?;
            session.eof(channel)?;
            session.close(channel)?;
            Ok(())
        }
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("build test runtime")
    }

    /// Spin up the loopback server + a `RemoteBackend` client talking to it
    /// over an in-memory duplex pipe, pinning the client's `HostKeyPolicy`
    /// to the server's own freshly-loaded host key (real host-key
    /// verification, not a bypass).
    fn connected_backend(runtime: tokio::runtime::Runtime) -> RemoteBackend {
        let server_key = PrivateKey::from_openssh(SERVER_PRIVATE_KEY).expect("parse server key");
        let server_public = server_key.public_key().clone();
        let client_key = PrivateKey::from_openssh(CLIENT_PRIVATE_KEY).expect("parse client key");
        let client_public = client_key.public_key().clone();

        let (client_stream, server_stream) = tokio::io::duplex(1 << 20);

        runtime.spawn(async move {
            let mut server_config = server::Config::default();
            server_config.keys = vec![server_key];
            let running = server::run_stream(
                Arc::new(server_config),
                server_stream,
                LoopbackServer {
                    expected_client_key: client_public,
                },
            )
            .await
            .expect("run_stream should accept the fixture stream");
            let _ = running.await;
        });

        let key_dir = tempfile::tempdir().expect("tempdir for client key");
        let key_path = key_dir.path().join("id_ed25519");
        client_key
            .write_openssh_file(&key_path, LineEnding::LF)
            .expect("write client key");
        // Keep the tempdir alive for the process lifetime of this test by
        // leaking it — it's a handful of bytes in a throwaway test process.
        std::mem::forget(key_dir);

        let config = RemoteBackendConfig::new("loopback", "akh-exec", &key_path)
            .with_host_key_policy(HostKeyPolicy::Pinned(Box::new(server_public)))
            .with_roots(vec![PathBuf::from("/")], vec![PathBuf::from("/")]);

        RemoteBackend::connect_over_stream(config, runtime, client_stream)
            .expect("connect over the loopback fixture")
    }

    #[test]
    fn exec_captures_stdout_stderr_and_exit_code_over_real_ssh_wire() {
        let backend = connected_backend(test_runtime());
        let out = backend
            .exec(ExecRequest::shell(
                "echo out_line; echo err_line >&2; exit 3",
                StdDuration::from_secs(10),
            ))
            .unwrap();
        assert_eq!(out.exit_code, 3);
        assert!(!out.success);
        assert_eq!(String::from_utf8_lossy(&out.stdout), "out_line\n");
        assert_eq!(String::from_utf8_lossy(&out.stderr), "err_line\n");
    }

    #[test]
    fn exec_success_reports_zero_exit() {
        let backend = connected_backend(test_runtime());
        let out = backend
            .exec(ExecRequest::shell("true", StdDuration::from_secs(10)))
            .unwrap();
        assert!(out.success);
        assert_eq!(out.exit_code, 0);
    }

    #[test]
    fn exec_applies_cwd_and_env_via_shell_wrapping() {
        let backend = connected_backend(test_runtime());
        let out = backend
            .exec(
                ExecRequest::shell("pwd && printf '%s' \"$AKH_REMOTE_TEST\"", StdDuration::from_secs(10))
                    .with_cwd("/tmp")
                    .with_env("AKH_REMOTE_TEST", "forty-two"),
            )
            .unwrap();
        assert!(out.success);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(stdout.starts_with("/tmp\n"), "stdout: {stdout:?}");
        assert!(stdout.ends_with("forty-two"), "stdout: {stdout:?}");
    }

    #[test]
    fn write_then_read_round_trips_over_the_remote_filesystem() {
        let backend = connected_backend(test_runtime());
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("roundtrip.txt");

        backend.write(&file, "through the wire\nwith a newline").unwrap();
        let read = backend.read_to_string(&file).unwrap();
        assert_eq!(read, "through the wire\nwith a newline");

        assert!(backend.exists(&file));
        assert!(!backend.is_symlink(&file));
        assert_eq!(backend.file_size(&file).unwrap(), "through the wire\nwith a newline".len() as u64);

        let names = backend.read_dir(dir.path()).unwrap();
        assert!(names.iter().any(|p| p == &file), "read_dir missing the file: {names:?}");
    }

    #[test]
    fn exists_is_false_for_a_missing_path() {
        let backend = connected_backend(test_runtime());
        assert!(!backend.exists(Path::new("/definitely/does/not/exist/akh-remote-test")));
    }

    #[test]
    fn read_missing_file_is_an_error() {
        let backend = connected_backend(test_runtime());
        let err = backend
            .read_to_string(Path::new("/definitely/does/not/exist/akh-remote-test"))
            .unwrap_err();
        assert!(matches!(err, FsError::Io { .. }));
    }

    #[test]
    fn write_disabled_without_a_write_root_is_denied_without_a_round_trip() {
        let runtime = test_runtime();
        let server_key = PrivateKey::from_openssh(SERVER_PRIVATE_KEY).expect("parse server key");
        let server_public = server_key.public_key().clone();
        let client_key = PrivateKey::from_openssh(CLIENT_PRIVATE_KEY).expect("parse client key");
        let client_public = client_key.public_key().clone();
        let (client_stream, server_stream) = tokio::io::duplex(1 << 16);
        runtime.spawn(async move {
            let mut server_config = server::Config::default();
            server_config.keys = vec![server_key];
            let running = server::run_stream(
                Arc::new(server_config),
                server_stream,
                LoopbackServer {
                    expected_client_key: client_public,
                },
            )
            .await
            .expect("run_stream");
            let _ = running.await;
        });
        let key_dir = tempfile::tempdir().unwrap();
        let key_path = key_dir.path().join("id_ed25519");
        client_key.write_openssh_file(&key_path, LineEnding::LF).unwrap();
        std::mem::forget(key_dir);

        // No `with_roots` call: write_roots stays empty.
        let config = RemoteBackendConfig::new("loopback", "akh-exec", &key_path)
            .with_host_key_policy(HostKeyPolicy::Pinned(Box::new(server_public)));
        let backend = RemoteBackend::connect_over_stream(config, runtime, client_stream).unwrap();

        assert!(!backend.is_writable());
        assert!(matches!(
            backend.write(Path::new("/tmp/should-not-be-written"), "nope"),
            Err(FsError::WritesDisabled)
        ));
    }

    #[test]
    fn wrong_client_key_is_rejected_by_the_server() {
        let runtime = test_runtime();
        let server_key = PrivateKey::from_openssh(SERVER_PRIVATE_KEY).expect("parse server key");
        let server_public = server_key.public_key().clone();
        // The server only accepts CLIENT_PRIVATE_KEY's public half; connect
        // with a *different* freshly-parsed identity (the server key reused
        // as a client key) to exercise the auth-rejection path end to end.
        let wrong_client_key = PrivateKey::from_openssh(SERVER_PRIVATE_KEY).expect("parse server key again");
        let real_client_key = PrivateKey::from_openssh(CLIENT_PRIVATE_KEY).expect("parse client key");
        let real_client_public = real_client_key.public_key().clone();

        let (client_stream, server_stream) = tokio::io::duplex(1 << 16);
        runtime.spawn(async move {
            let mut server_config = server::Config::default();
            server_config.keys = vec![server_key];
            let running = server::run_stream(
                Arc::new(server_config),
                server_stream,
                LoopbackServer {
                    expected_client_key: real_client_public,
                },
            )
            .await
            .expect("run_stream");
            let _ = running.await;
        });

        let key_dir = tempfile::tempdir().unwrap();
        let key_path = key_dir.path().join("id_ed25519");
        wrong_client_key.write_openssh_file(&key_path, LineEnding::LF).unwrap();
        std::mem::forget(key_dir);

        let config = RemoteBackendConfig::new("loopback", "akh-exec", &key_path)
            .with_host_key_policy(HostKeyPolicy::Pinned(Box::new(server_public)));
        // Not `unwrap_err()` — `RemoteBackend` (holds a `tokio::Runtime`)
        // isn't `Debug`, and giving it one just to satisfy that bound would
        // add a needless trait obligation to the production type.
        match RemoteBackend::connect_over_stream(config, runtime, client_stream) {
            Err(RemoteError::AuthRejected { .. }) => {}
            Err(other) => panic!("expected AuthRejected, got: {other:?}"),
            Ok(_) => panic!("expected auth rejection, but the connection succeeded"),
        }
    }

    #[test]
    fn handshake_reports_remote_unverified() {
        let backend = connected_backend(test_runtime());
        assert_eq!(
            backend.handshake(),
            Posture {
                confinement: Confinement::RemoteUnverified
            }
        );
    }

    #[test]
    fn locus_reports_remote_backend_host_user_and_a_correlation_id() {
        let backend = connected_backend(test_runtime());
        let locus = backend.locus();
        assert_eq!(locus.backend, "remote");
        assert_eq!(locus.host.as_deref(), Some("loopback"));
        assert_eq!(locus.os_user.as_deref(), Some("akh-exec"));
        assert!(
            locus.correlation_id.as_deref().is_some_and(|id| id.starts_with("remote-")),
            "expected a remote-<hash> correlation id, got {:?}",
            locus.correlation_id
        );
        // Stable across calls on the same connection.
        assert_eq!(locus.correlation_id, backend.locus().correlation_id);
    }

    #[test]
    fn wrong_pinned_host_key_is_rejected() {
        let runtime = test_runtime();
        let server_key = PrivateKey::from_openssh(SERVER_PRIVATE_KEY).expect("parse server key");
        let client_key = PrivateKey::from_openssh(CLIENT_PRIVATE_KEY).expect("parse client key");
        let client_public = client_key.public_key().clone();
        // Pin the *client's* key as the expected host key — guaranteed to
        // mismatch the server's real (different) host key.
        let wrong_pin = client_public.clone();

        let (client_stream, server_stream) = tokio::io::duplex(1 << 16);
        runtime.spawn(async move {
            let mut server_config = server::Config::default();
            server_config.keys = vec![server_key];
            let running = server::run_stream(
                Arc::new(server_config),
                server_stream,
                LoopbackServer {
                    expected_client_key: client_public,
                },
            )
            .await
            .expect("run_stream");
            let _ = running.await;
        });

        let key_dir = tempfile::tempdir().unwrap();
        let key_path = key_dir.path().join("id_ed25519");
        client_key.write_openssh_file(&key_path, LineEnding::LF).unwrap();
        std::mem::forget(key_dir);

        let config = RemoteBackendConfig::new("loopback", "akh-exec", &key_path)
            .with_host_key_policy(HostKeyPolicy::Pinned(Box::new(wrong_pin)));
        // The pinned key mismatches, so our handler returns the rich
        // `HostKeyRejected` diagnostic, which russh propagates as the connect
        // error.
        match RemoteBackend::connect_over_stream(config, runtime, client_stream) {
            Err(RemoteError::HostKeyRejected { policy, .. }) => assert_eq!(policy, "Pinned"),
            Err(other) => panic!("expected HostKeyRejected, got: {other:?}"),
            Ok(_) => panic!("expected host-key rejection, but the connection succeeded"),
        }
    }
}
