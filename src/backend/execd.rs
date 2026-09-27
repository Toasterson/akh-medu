//! `ExecdBackend`: workspace side effects through akhsbxd + akhexecd (Phase 53c,
//! ADR 113 decision 3).
//!
//! ADR 113 moves every side-effecting tool call out of the caller's address
//! space. An akh keeps its cognition in akhomed, but `shell_exec`, `file_io`,
//! `edit`, `glob` and `grep` land in a sandbox that akhsbxd allocates and
//! akhexecd operates. This module implements the ADR 082 [`Backend`] seam
//! against those two services, so the tools themselves do not change.
//!
//! ## Protocol
//!
//! 1. **Allocate** (`POST {sandbox}/v1/allocations`, role `session` at
//!    akhsbxd). The reply carries the [`WorkspaceBinding`]: executor, sandbox
//!    id, generation, the executor-resolved root and the executor's
//!    confinement label. A `busy` reply (a Kubernetes sandbox that is still
//!    starting) is retried until [`ExecdConfig::allocate_timeout`].
//! 2. **Renew** the 300 s lease every [`ExecdConfig::lease_renew_interval`]
//!    (100 s by default) from a background thread
//!    (`POST {sandbox}/v1/allocations/{session}/renew`). A lease the services
//!    report as expired or released is terminal: there is no implicit
//!    reacquisition, and every later verb fails with
//!    [`ExecdError::LeaseExpired`].
//! 3. **Operate** (`POST …/v1/operations`), either through akhsbxd's
//!    `/v1/executors/{sandbox}/…` gateway ([`ExecutorRoute::ViaSandbox`], the
//!    stable per-site address) or straight at akhexecd
//!    ([`ExecutorRoute::Direct`]). Every operation carries a fresh id. The
//!    executor deduplicates on that id, so a transport failure is retried with
//!    the **same** id and can never run an effect twice.
//! 4. **Release** the allocation when the backend drops (best effort; files
//!    stay behind for an `existing` provisioner).
//!
//! `exec` is `exec_start`, then `exec_poll` in a loop (each poll waits up to
//! [`ExecdConfig::poll_wait`] server-side), and `exec_cancel` once the
//! request's deadline passes. Both the executor's own timeout and the
//! client-side deadline map to [`ExecError::Timeout`].
//!
//! ## Honest posture
//!
//! [`Backend::handshake`] reports what the binding's `confinement` label
//! asserts, classified by [`classify_confinement`]. The label is the executor
//! operator's claim (`akhexecd`'s `confinement` setting, default
//! `"unconfined"`). An unknown or absent label fails safe to
//! [`Confinement::Unconfined`], as ADR 082 §8 requires. In particular the Mac
//! `existing` provisioner runs akhexecd as an ordinary host process, so it is
//! never reported as [`Confinement::Sandboxed`].
//!
//! ## Transport
//!
//! The [`Backend`] trait is synchronous and every call here is a short
//! request/response exchange, so the backend uses the blocking `ureq` client
//! the crate already depends on. There is no async runtime to own.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use akh_session_types::{
    Allocate, Allocation, Fault, OperationReceipt, ToolResult, TurnStatus, WorkspaceBinding,
    WorkspaceOperation, WorkspaceStat,
};
use miette::Diagnostic;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use thiserror::Error;

use crate::backend::{
    Backend, Confinement, ExecError, ExecLocus, ExecOutcome, ExecRequest, ExecResult, Posture,
};
use crate::fs::{Filesystem, FsError, FsResult};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from allocating or operating an akhexecd workspace.
///
/// [`ExecError::Execd`] and [`FsError::Execd`] wrap this transparently, so the
/// code and help below reach the caller unchanged.
#[derive(Debug, Error, Diagnostic)]
pub enum ExecdError {
    /// A bearer token file could not be read or is malformed.
    #[error("cannot use the execd credential at {}: {reason}", path.display())]
    #[diagnostic(
        code(akh::backend::execd::credential),
        help(
            "Point `sandbox_token_file` / `executor_token_file` at a readable file holding \
             one bearer token of at least 24 characters with no whitespace, the same \
             format akhsbxd and akhexecd accept."
        )
    )]
    Credential { path: PathBuf, reason: String },

    /// The `[exec] kind = "execd"` settings are unusable.
    #[error("invalid execd configuration: {reason}")]
    #[diagnostic(
        code(akh::backend::execd::config),
        help(
            "Check the `[exec]` table: `sandbox_url` (and `executor_url`, if set) must be \
             https URLs without credentials, query or fragment; plain http is accepted for \
             loopback hosts (localhost, 127.0.0.1, [::1]) or with \
             `allow_insecure_http = true` for a trusted private link, because the bearer \
             token crosses it in clear text. `profile` and `principal` must be non-empty, \
             `poll_wait_ms` must be between 1 and 10000, and `allocate_timeout_secs` (default \
             120) must be between 1 and 3600 — raise it if the provisioner's first image \
             pull routinely takes longer than that."
        )
    )]
    Config { reason: String },

    /// akhsbxd refused or failed the allocation.
    #[error("akhsbxd did not allocate a workspace for profile `{profile}`: {code}: {message}")]
    #[diagnostic(
        code(akh::backend::execd::allocate),
        help(
            "Options: check that akhsbxd has a workspace profile named `{profile}`; that \
             the token's principal holds the `session` role there; and, for a Kubernetes \
             provisioner, that the sandbox becomes Ready within the allocation timeout. \
             `unknown_outcome` means an operator must reconcile the named allocation on \
             akhsbxd before the akh can start."
        )
    )]
    Allocate {
        profile: String,
        code: String,
        message: String,
    },

    /// akhexecd admitted the operation and reported a definite failure.
    #[error("akhexecd `{operation}` failed: {code}: {message}")]
    #[diagnostic(
        code(akh::backend::execd::operation),
        help(
            "The executor ran the check and refused or failed the operation; nothing \
             uncertain happened. `conflict` usually means a stale revision or an \
             unresolved operation fencing mutations; `invalid_request` means a path \
             escaped the workspace or an argument was malformed; `forbidden` means the \
             token's principal lacks the `session` (or, via the gateway, \
             `executor_proxy`) role."
        )
    )]
    Operation {
        operation: &'static str,
        code: String,
        message: String,
    },

    /// akhexecd could not confirm whether the operation took effect.
    #[error("akhexecd cannot confirm the outcome of `{operation}` (receipt {receipt})")]
    #[diagnostic(
        code(akh::backend::execd::unknown_outcome),
        help(
            "The operation may or may not have taken effect, and akhexecd now fences \
             further mutations in this workspace. Do not retry it blindly: inspect the \
             workspace, then have an operator reconcile receipt {receipt} \
             (`POST /v1/operations/{receipt}/reconcile`, admin role) with what actually \
             happened."
        )
    )]
    UnknownOutcome {
        operation: &'static str,
        receipt: String,
    },

    /// `exec_poll` kept failing, so the client stopped following a process it
    /// had started. The process may still run, or may have finished.
    #[error(
        "lost track of execd process {process}: polling failed ({cause}); cancel requested: \
         {cancel_requested}"
    )]
    #[diagnostic(
        code(akh::backend::execd::process_outcome_unknown),
        help(
            "The command's outcome is unknown: it may still be running in the sandbox or \
             may already have finished (with its side effects). The client retried the \
             read-only poll with backoff, then sent a best-effort `exec_cancel`. Do not \
             re-run the command blindly: check akhsbxd/akhexecd reachability, inspect the \
             workspace, and poll process {process} (`exec_poll`) once the executor answers."
        )
    )]
    ProcessOutcomeUnknown {
        process: String,
        cause: String,
        cancel_requested: bool,
    },

    /// The workspace lease lapsed or was released.
    #[error("the execd workspace lease for session {session} expired or was released")]
    #[diagnostic(
        code(akh::backend::execd::lease_expired),
        help(
            "A lapsed lease fences all effects, and there is no implicit reacquisition. \
             Reload the akh (sleep and wake it, or restart akhomed) to allocate a fresh \
             workspace. If this repeats, check that akhsbxd was reachable for the lease \
             renewals, which run every {renew_secs}s."
        )
    )]
    LeaseExpired { session: String, renew_secs: u64 },

    /// The HTTP exchange failed or returned something that is not a receipt.
    #[error("execd transport to {url} failed: {message}")]
    #[diagnostic(
        code(akh::backend::execd::transport),
        help(
            "The request was retried with the same operation id, which akhexecd \
             deduplicates, so nothing ran twice. A mutation may still have taken effect \
             before the connection failed. Check that akhsbxd/akhexecd are reachable \
             from this host before retrying."
        )
    )]
    Transport { url: String, message: String },

    /// A path does not lie inside the allocated workspace.
    #[error("path \"{path}\" is outside the execd workspace {root}")]
    #[diagnostic(
        code(akh::backend::execd::outside_workspace),
        help(
            "Execd workspaces are addressed relative to their root. Use a relative path or \
             an absolute path under {root}. Paths on the akhomed host do not exist inside \
             the sandbox."
        )
    )]
    OutsideWorkspace { path: String, root: String },
}

/// Result alias for execd operations.
pub type ExecdResult<T> = Result<T, ExecdError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Where executor operations are sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorRoute {
    /// Through akhsbxd's `/v1/executors/{sandbox}/…` gateway, the stable
    /// per-site address. The token must be accepted by akhsbxd (role
    /// `executor_proxy`) **and** by akhexecd (role `session`), because the
    /// gateway forwards it unchanged.
    ViaSandbox { token_file: PathBuf },
    /// Straight at one akhexecd, for single-host setups and tests.
    Direct { url: String, token_file: PathBuf },
}

impl ExecutorRoute {
    fn token_file(&self) -> &Path {
        match self {
            Self::ViaSandbox { token_file } | Self::Direct { token_file, .. } => token_file,
        }
    }
}

/// Connection settings for an [`ExecdBackend`].
#[derive(Debug, Clone)]
pub struct ExecdConfig {
    /// akhsbxd base URL, e.g. `https://akhsbxd.lab.internal`.
    pub sandbox_url: String,
    /// Bearer token for akhsbxd (role `session`).
    pub sandbox_token_file: PathBuf,
    /// Where workspace operations go.
    pub executor: ExecutorRoute,
    /// akhsbxd workspace profile to allocate from.
    pub profile: String,
    /// Principal the allocation is made for (recorded by akhsbxd).
    pub principal: String,
    /// Allocation key. akhsbxd allocates at most once per session id, and a
    /// released session cannot be reallocated, so each backend instance mints
    /// its own id.
    pub session_id: String,
    /// Server-side wait per `exec_poll` (akhexecd caps it at 10 s).
    pub poll_wait: Duration,
    /// Deadline for one HTTP exchange. Must exceed `poll_wait`.
    pub request_timeout: Duration,
    /// How long to keep retrying an allocation that reports `busy`.
    pub allocate_timeout: Duration,
    /// Lease renewal cadence. The lease itself lasts 300 s.
    pub lease_renew_interval: Duration,
    /// Accept plain `http://` to a non-loopback host. The bearer tokens then
    /// cross that link in clear text, so this is for trusted private links
    /// only (the same rule as the session services' `Endpoint`).
    pub allow_insecure_http: bool,
    /// How long `exec` keeps retrying a failed `exec_poll` past the process
    /// deadline before it cancels and reports [`ExecdError::UnknownOutcome`].
    pub poll_retry_budget: Duration,
}

impl ExecdConfig {
    /// Default server-side wait per poll.
    pub const DEFAULT_POLL_WAIT: Duration = Duration::from_millis(5_000);
    /// akhexecd rejects longer poll waits.
    pub const MAX_POLL_WAIT: Duration = Duration::from_millis(10_000);
    /// A third of the 300 s lease, so two renewals can fail before it lapses.
    pub const DEFAULT_LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(100);
    /// Keep retrying failed polls for a minute past the process deadline, so a
    /// short executor or gateway restart does not abandon a running process.
    pub const DEFAULT_POLL_RETRY_BUDGET: Duration = Duration::from_secs(60);
    /// Longest accepted `allocate_timeout_secs`: an operator raising this to
    /// tolerate a slow first image pull must still get a bounded wait, not an
    /// effectively-infinite one that hides a genuinely stuck provisioner.
    pub const MAX_ALLOCATE_TIMEOUT: Duration = Duration::from_secs(3_600);

    /// Settings for one akh workspace, routed through the akhsbxd gateway with
    /// the akhsbxd token. The session id is `akhomed-<workspace>-<uuid v7>`.
    pub fn new(
        sandbox_url: impl Into<String>,
        sandbox_token_file: impl Into<PathBuf>,
        profile: impl Into<String>,
        principal: impl Into<String>,
        workspace: &str,
    ) -> Self {
        let sandbox_token_file = sandbox_token_file.into();
        Self {
            sandbox_url: sandbox_url.into(),
            executor: ExecutorRoute::ViaSandbox {
                token_file: sandbox_token_file.clone(),
            },
            sandbox_token_file,
            profile: profile.into(),
            principal: principal.into(),
            session_id: format!("akhomed-{workspace}-{}", akh_session_types::new_id()),
            poll_wait: Self::DEFAULT_POLL_WAIT,
            request_timeout: Duration::from_secs(30),
            allocate_timeout: Duration::from_secs(120),
            lease_renew_interval: Self::DEFAULT_LEASE_RENEW_INTERVAL,
            allow_insecure_http: false,
            poll_retry_budget: Self::DEFAULT_POLL_RETRY_BUDGET,
        }
    }

    /// Route operations through `route` instead of the default gateway.
    pub fn with_executor(mut self, route: ExecutorRoute) -> Self {
        self.executor = route;
        self
    }

    /// Override the per-poll server-side wait.
    pub fn with_poll_wait(mut self, wait: Duration) -> Self {
        self.poll_wait = wait;
        self
    }

    /// Override the lease renewal cadence.
    pub fn with_lease_renew_interval(mut self, interval: Duration) -> Self {
        self.lease_renew_interval = interval;
        self
    }

    /// Override the allocation retry budget.
    pub fn with_allocate_timeout(mut self, timeout: Duration) -> Self {
        self.allocate_timeout = timeout;
        self
    }

    /// Accept plain `http://` to non-loopback hosts (trusted private links).
    pub fn with_insecure_http(mut self, allow: bool) -> Self {
        self.allow_insecure_http = allow;
        self
    }

    /// Override how long failed `exec_poll`s are retried past the deadline.
    pub fn with_poll_retry_budget(mut self, budget: Duration) -> Self {
        self.poll_retry_budget = budget;
        self
    }

    fn validate(&self) -> ExecdResult<()> {
        let invalid = |reason: String| Err(ExecdError::Config { reason });
        validate_base_url("sandbox_url", &self.sandbox_url, self.allow_insecure_http)?;
        if let ExecutorRoute::Direct { url, .. } = &self.executor {
            validate_base_url("executor_url", url, self.allow_insecure_http)?;
        }
        if self.profile.trim().is_empty() || self.principal.trim().is_empty() {
            return invalid("`profile` and `principal` must be non-empty".into());
        }
        if self.poll_wait.is_zero() || self.poll_wait > Self::MAX_POLL_WAIT {
            return invalid(format!(
                "poll_wait_ms must be 1..=10000, got {}",
                self.poll_wait.as_millis()
            ));
        }
        if self.request_timeout <= self.poll_wait {
            return invalid("the request timeout must exceed the poll wait".into());
        }
        if self.lease_renew_interval.is_zero() {
            return invalid("the lease renewal interval must be positive".into());
        }
        if self.allocate_timeout.is_zero() || self.allocate_timeout > Self::MAX_ALLOCATE_TIMEOUT {
            return invalid(format!(
                "allocate_timeout_secs must be 1..={}, got {}",
                Self::MAX_ALLOCATE_TIMEOUT.as_secs(),
                self.allocate_timeout.as_secs()
            ));
        }
        Ok(())
    }
}

/// Check one service base URL. Plain `http://` carries the bearer token in
/// clear text, so it is accepted only for a loopback host (`localhost`,
/// `127.0.0.1`, `[::1]`, exactly the session services' `Endpoint::validate`
/// and the MCP client's rule) or with `allow_insecure_http`.
fn validate_base_url(field: &str, url: &str, allow_insecure_http: bool) -> ExecdResult<()> {
    let scheme_ok = url.starts_with("http://") || url.starts_with("https://");
    let authority = url.split_once("://").map(|(_, rest)| rest).unwrap_or("");
    let host = authority.split('/').next().unwrap_or("");
    if !scheme_ok || host.is_empty() || host.contains('@') || url.contains(['?', '#']) {
        return Err(ExecdError::Config {
            reason: format!(
                "`{field}` must be an http(s) base URL without credentials, query or \
                 fragment, got {url:?}"
            ),
        });
    }
    if let Some(rest) = url.strip_prefix("http://")
        && !allow_insecure_http
        && !crate::agent::tools::mcp_config::is_loopback_authority(rest)
    {
        return Err(ExecdError::Config {
            reason: format!(
                "`{field}` is plain http to a non-loopback host ({url:?}); the bearer token \
                 would cross the network in clear text. Use https, or set \
                 `allow_insecure_http = true` for a trusted private link"
            ),
        });
    }
    Ok(())
}

/// Read a bearer token the way akhsbxd/akhexecd do: trimmed, at least 24
/// characters, no whitespace.
fn read_token(path: &Path) -> ExecdResult<String> {
    let raw = std::fs::read_to_string(path).map_err(|error| ExecdError::Credential {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })?;
    let token = raw.trim();
    if token.len() < 24 || token.chars().any(char::is_whitespace) {
        return Err(ExecdError::Credential {
            path: path.to_path_buf(),
            reason: "the token must be at least 24 characters without whitespace".into(),
        });
    }
    Ok(token.to_owned())
}

// ---------------------------------------------------------------------------
// Pending allocation reuse (idempotent reattach)
// ---------------------------------------------------------------------------

/// akhomed's cache of not-yet-successful [`ExecdBackend::connect`] attempts,
/// keyed by a caller-chosen label (one per exec target: a workspace's
/// primary execd backend, or one `[[exec.sandboxes]]` entry).
///
/// ## The incident this exists to prevent (2026-09-27)
///
/// Every call used to start from a freshly minted `session_id`
/// ([`ExecdConfig::new`]). A slow first image pull (2.5 minutes) outlived the
/// 120 s default [`ExecdConfig::allocate_timeout`], so `connect` gave up —
/// and whatever retried the workspace load next (every ~30 s) minted
/// *another* fresh session id, so akhsbxd provisioned a *second* Kubernetes
/// `Sandbox`, still pulling the very same image, and so on. 25 orphaned
/// `Sandbox` resources later, the node was full and nothing could load.
///
/// ## Idempotent reattach, not release-then-retry
///
/// akhsbxd never reallocates a *released* session id (`"allocation was
/// released; create a new session to allocate again"`), and provisioning
/// keeps running on the server after the client's `allocate_timeout` gives
/// up locally. Releasing on every failed attempt would force a fresh
/// `Sandbox` — and a fresh cold image pull — on *every* retry, so a
/// workspace whose image takes longer to pull than one client is willing to
/// wait could never load, no matter how many times it retried.
///
/// So a failed, retryable attempt is remembered instead: the *next* attempt
/// for the same label reuses the same `session_id` and reattaches to the
/// same (by now further along, possibly already `Ready`) allocation. The
/// pending session is released only when a caller explicitly gives up on it
/// ([`Self::abandon`]) or reconfigures the target underneath it
/// ([`Self::config_for`]) — never on an ordinary retry.
///
/// ## Server-side reaping (deliberately not added here)
///
/// akhsbxd's `Allocation::lease_until` is set on creation but is only
/// enforced once the allocation reaches `"ready"` — a `"creating"` /
/// `"allocating"` allocation nobody ever asks about again (for example this
/// process crashing and never restarting) has no automatic reaper today.
/// The client-side fix above closes the reported incident (every retry that
/// *does* happen reattaches instead of stacking), so a server-side sweep for
/// allocations nobody is retrying is left as an explicit follow-up rather
/// than a rushed change to akhsbxd's allocation state machine.
#[derive(Default)]
pub struct PendingAllocations {
    attempts: Mutex<HashMap<String, ExecdConfig>>,
}

impl PendingAllocations {
    /// A fresh, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The [`ExecdConfig`] to attempt next for `label`: a previously
    /// remembered attempt's config (same `session_id`) when its
    /// `sandbox_url`, `profile` and `principal` still match `fresh` —
    /// otherwise `fresh` itself (minting a new session id), first
    /// abandoning any stale pending attempt (the caller's configuration
    /// changed underneath it, e.g. an operator edited `akh.toml`).
    pub fn config_for(&self, label: &str, fresh: ExecdConfig) -> ExecdConfig {
        let stale = {
            let mut attempts = self.attempts.lock().unwrap_or_else(|e| e.into_inner());
            match attempts.get(label) {
                Some(pending)
                    if pending.sandbox_url == fresh.sandbox_url
                        && pending.profile == fresh.profile
                        && pending.principal == fresh.principal =>
                {
                    return pending.clone();
                }
                Some(_) => attempts.remove(label),
                None => None,
            }
        };
        if let Some(stale) = stale {
            tracing::info!(
                target: "akh::backend::execd",
                label,
                session = %stale.session_id,
                "execd target reconfigured; abandoning its stale pending allocation"
            );
            release_abandoned_allocation(&stale);
        }
        fresh
    }

    /// Record that `config`'s attempt for `label` failed in a way worth
    /// retrying: remember it so the next [`Self::config_for`] reattaches
    /// instead of minting a fresh session.
    pub fn remember(&self, label: &str, config: ExecdConfig) {
        self.attempts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(label.to_owned(), config);
    }

    /// The attempt for `label` succeeded: forget it, so the next
    /// independent load of this target mints a fresh session. (This one now
    /// belongs to the live [`ExecdBackend`], which releases it itself on
    /// drop.)
    pub fn clear(&self, label: &str) {
        self.attempts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(label);
    }

    /// Give up on `label`'s pending allocation for good: forget it and
    /// best-effort release it at akhsbxd, so its still-provisioning
    /// Kubernetes `Sandbox` is reaped now rather than left for an operator
    /// to find.
    pub fn abandon(&self, label: &str) {
        let removed = self
            .attempts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(label);
        if let Some(config) = removed {
            release_abandoned_allocation(&config);
        }
    }
}

/// Best-effort release of an allocation identified by `config.session_id`,
/// for a session this process never successfully connected (so there is no
/// live [`ExecdBackend`] / lease thread to release it itself). Never fails
/// loudly: a release that cannot reach akhsbxd just means the allocation
/// needs an operator's explicit reconcile.
pub fn release_abandoned_allocation(config: &ExecdConfig) {
    let url = format!(
        "{}/v1/allocations/{}/release",
        config.sandbox_url.trim_end_matches('/'),
        segment(&config.session_id)
    );
    let token = match read_token(&config.sandbox_token_file) {
        Ok(token) => token,
        Err(error) => {
            tracing::warn!(
                target: "akh::backend::execd",
                session = %config.session_id,
                %error,
                "could not read the sandbox token to release an abandoned allocation"
            );
            return;
        }
    };
    let http = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout(LEASE_REQUEST_TIMEOUT)
        .redirects(0)
        .build();
    match http
        .post(&url)
        .set("Authorization", &format!("Bearer {token}"))
        .call()
    {
        Ok(_) => tracing::info!(
            target: "akh::backend::execd",
            session = %config.session_id,
            "released an abandoned execd allocation"
        ),
        Err(error) => tracing::warn!(
            target: "akh::backend::execd",
            session = %config.session_id,
            %error,
            "could not release an abandoned execd allocation; an operator may need to \
             reconcile it at akhsbxd"
        ),
    }
}

// ---------------------------------------------------------------------------
// Confinement
// ---------------------------------------------------------------------------

/// Classify an executor's `confinement` label into the posture the backend may
/// claim.
///
/// The label is free text set by the akhexecd operator (its `confinement`
/// setting, default `"unconfined"`). Only the mechanism before an optional
/// `:` is interpreted:
///
/// - `container`, `pod`, `gvisor`, `kata`, `firecracker`, `microvm`, `vm`,
///   `zone`, `jail` → [`Confinement::Sandboxed`];
/// - `user`, `privilege-dropped` → [`Confinement::PrivilegeDropped`];
/// - anything else (including `unconfined`, `unobserved`, an empty label and
///   the Mac `existing` provisioner's host-process executor) →
///   [`Confinement::Unconfined`].
///
/// Unknown labels fail safe to unconfined (ADR 082 §8).
pub fn classify_confinement(label: &str) -> Confinement {
    let mechanism = label
        .split(':')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match mechanism.as_str() {
        "container" | "pod" | "gvisor" | "kata" | "firecracker" | "microvm" | "vm" | "zone"
        | "jail" => Confinement::Sandboxed,
        "user" | "privilege-dropped" => Confinement::PrivilegeDropped,
        _ => Confinement::Unconfined,
    }
}

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

/// The akhexecd verbs this backend issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verb {
    Read,
    Write,
    Edit,
    Mkdir,
    List,
    Search,
    Stat,
    ExecStart,
    ExecPoll,
    ExecCancel,
}

impl Verb {
    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Edit => "edit",
            Self::Mkdir => "mkdir",
            Self::List => "list",
            Self::Search => "search",
            Self::Stat => "stat",
            Self::ExecStart => "exec_start",
            Self::ExecPoll => "exec_poll",
            Self::ExecCancel => "exec_cancel",
        }
    }
}

/// Why an HTTP exchange failed, before it is given operation context.
#[derive(Debug)]
enum HttpFailure {
    /// The service answered with a structured `Fault`.
    Fault(Fault),
    /// The request never completed or the reply could not be decoded.
    Transport(String),
}

/// `exec_poll` result data.
#[derive(Debug, Deserialize)]
struct PollData {
    status: TurnStatus,
    #[serde(default)]
    result: Option<ToolResult>,
}

/// Terminal process data recorded by akhexecd.
#[derive(Debug, Deserialize)]
struct ProcessResult {
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    timed_out: bool,
    #[serde(default)]
    cancelled: bool,
    #[serde(default)]
    truncated: bool,
}

#[derive(Debug, Deserialize)]
struct ReadData {
    content: String,
    revision: String,
}

#[derive(Debug, Deserialize)]
struct ListData {
    entries: Vec<String>,
}

/// Percent-encode one URL path segment (same rule as the services).
fn segment(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// The `Fault` a failed receipt carries, if it recorded one.
fn receipt_fault(result: Option<&ToolResult>) -> Fault {
    result
        .and_then(|r| r.data.as_ref())
        .and_then(|data| data.get("error"))
        .and_then(|error| serde_json::from_value::<Fault>(error.clone()).ok())
        .unwrap_or_else(|| Fault::new("failed", result.map(|r| r.text.clone()).unwrap_or_default()))
}

/// A transport failure is retried this many times with the same operation id.
const TRANSPORT_RETRIES: usize = 2;
/// Extra time past the request deadline before the client cancels, so the
/// executor's own timeout (which reports `timed_out`) normally fires first.
const CANCEL_GRACE: Duration = Duration::from_secs(2);
/// akhexecd accepts process timeouts of 1 ms ..= 1 h.
const MAX_PROCESS_TIMEOUT_MS: u64 = 3_600_000;
/// First and largest pause between retries of a failed `exec_poll`.
const POLL_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(250);
const POLL_RETRY_MAX_BACKOFF: Duration = Duration::from_secs(5);
/// Deadline for one lease renewal or release exchange. Kept short so a stuck
/// akhsbxd cannot hold the renewal thread for the full request timeout.
const LEASE_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// How `exec` treats a failed `exec_poll`.
#[derive(Debug)]
enum PollFailure {
    /// The workspace is fenced or its outcome is already unknown: report it.
    Terminal(ExecdError),
    /// Transport trouble or a transient service fault: poll again.
    Retryable(ExecdError),
    /// A definite refusal that retrying will not change (the process id is
    /// unknown, the token lacks the role, ...): stop following the process.
    Definite(ExecdError),
}

impl PollFailure {
    fn classify(error: ExecdError) -> Self {
        match &error {
            ExecdError::LeaseExpired { .. }
            | ExecdError::UnknownOutcome { .. }
            | ExecdError::ProcessOutcomeUnknown { .. } => Self::Terminal(error),
            ExecdError::Operation { code, .. }
                if matches!(
                    code.as_str(),
                    "not_found" | "forbidden" | "unauthorized" | "invalid_request" | "conflict"
                ) =>
            {
                Self::Definite(error)
            }
            ExecdError::Transport { .. } | ExecdError::Operation { .. } => Self::Retryable(error),
            _ => Self::Definite(error),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct Shared {
    config: ExecdConfig,
    http: ureq::Agent,
    sandbox_token: String,
    executor_token: String,
    operations_url: String,
    binding: WorkspaceBinding,
    /// `binding.root` as the one advisory read root.
    read_roots: Vec<PathBuf>,
    lease_lost: AtomicBool,
    stop: Mutex<bool>,
    wake: Condvar,
}

impl Shared {
    fn root(&self) -> &Path {
        &self.read_roots[0]
    }

    fn lease_expired(&self) -> ExecdError {
        ExecdError::LeaseExpired {
            session: self.config.session_id.clone(),
            renew_secs: self.config.lease_renew_interval.as_secs(),
        }
    }

    fn post<T: DeserializeOwned>(
        &self,
        url: &str,
        token: &str,
        body: Option<&Value>,
        timeout: Option<Duration>,
    ) -> Result<T, HttpFailure> {
        let mut request = self
            .http
            .post(url)
            .set("Authorization", &format!("Bearer {token}"));
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = match body {
            Some(body) => request.send_json(body),
            None => request.call(),
        };
        match response {
            Ok(response) => response
                .into_json::<T>()
                .map_err(|error| HttpFailure::Transport(format!("invalid reply: {error}"))),
            Err(ureq::Error::Status(status, response)) => {
                let text = response.into_string().unwrap_or_default();
                Err(match serde_json::from_str::<Fault>(&text) {
                    Ok(fault) => HttpFailure::Fault(fault),
                    Err(_) => HttpFailure::Transport(format!("HTTP {status}: {text}")),
                })
            }
            Err(ureq::Error::Transport(error)) => Err(HttpFailure::Transport(error.to_string())),
        }
    }

    /// Admit one operation and return its receipt. Transport failures are
    /// retried with the same id; service faults are mapped to typed errors.
    fn operate(&self, verb: Verb, arguments: Value) -> ExecdResult<OperationReceipt> {
        if self.lease_lost.load(Ordering::SeqCst) {
            return Err(self.lease_expired());
        }
        let request = WorkspaceOperation {
            id: akh_session_types::new_id(),
            session_id: self.config.session_id.clone(),
            turn_id: self.config.session_id.clone(),
            workspace_id: self.binding.id.clone(),
            generation: self.binding.generation,
            operation: verb.as_str().into(),
            arguments,
        };
        let body = serde_json::to_value(&request).map_err(|error| ExecdError::Transport {
            url: self.operations_url.clone(),
            message: format!("cannot encode the operation: {error}"),
        })?;
        let mut attempt = 0;
        let receipt: OperationReceipt = loop {
            match self.post(
                &self.operations_url,
                &self.executor_token,
                Some(&body),
                None,
            ) {
                Ok(receipt) => break receipt,
                Err(HttpFailure::Transport(message)) if attempt < TRANSPORT_RETRIES => {
                    attempt += 1;
                    tracing::warn!(
                        target: "akh::backend::execd",
                        operation = verb.as_str(),
                        id = %request.id,
                        attempt,
                        %message,
                        "execd transport failed; retrying with the same operation id"
                    );
                    std::thread::sleep(Duration::from_millis(250 * attempt as u64));
                }
                Err(HttpFailure::Transport(message)) => {
                    return Err(ExecdError::Transport {
                        url: self.operations_url.clone(),
                        message: format!("{message} (operation id {})", request.id),
                    });
                }
                Err(HttpFailure::Fault(fault)) => {
                    return Err(self.fault_error(verb, &request.id, fault));
                }
            }
        };
        if receipt.request != request {
            return Err(ExecdError::Transport {
                url: self.operations_url.clone(),
                message: format!(
                    "the executor returned a receipt for another request than {}",
                    request.id
                ),
            });
        }
        if receipt.status == TurnStatus::Unknown {
            return Err(ExecdError::UnknownOutcome {
                operation: verb.as_str(),
                receipt: receipt.id,
            });
        }
        Ok(receipt)
    }

    fn fault_error(&self, verb: Verb, id: &str, fault: Fault) -> ExecdError {
        match fault.code.as_str() {
            "expired" => {
                self.lease_lost.store(true, Ordering::SeqCst);
                self.lease_expired()
            }
            "unknown_outcome" => ExecdError::UnknownOutcome {
                operation: verb.as_str(),
                receipt: id.to_owned(),
            },
            _ => ExecdError::Operation {
                operation: verb.as_str(),
                code: fault.code,
                message: fault.message,
            },
        }
    }

    /// Run a verb that completes synchronously and return its result data.
    fn call(&self, verb: Verb, arguments: Value) -> ExecdResult<Value> {
        let receipt = self.operate(verb, arguments)?;
        match receipt.status {
            TurnStatus::Completed => Ok(receipt
                .result
                .and_then(|result| result.data)
                .unwrap_or(Value::Null)),
            _ => {
                let fault = receipt_fault(receipt.result.as_ref());
                Err(ExecdError::Operation {
                    operation: verb.as_str(),
                    code: fault.code,
                    message: fault.message,
                })
            }
        }
    }

    fn call_as<T: DeserializeOwned>(&self, verb: Verb, arguments: Value) -> ExecdResult<T> {
        let data = self.call(verb, arguments)?;
        serde_json::from_value(data).map_err(|error| ExecdError::Transport {
            url: self.operations_url.clone(),
            message: format!("unexpected `{}` result: {error}", verb.as_str()),
        })
    }

    fn renew(&self) -> ExecdResult<u64> {
        let url = format!(
            "{}/v1/allocations/{}/renew",
            self.config.sandbox_url.trim_end_matches('/'),
            segment(&self.config.session_id)
        );
        match self.post::<Allocation>(&url, &self.sandbox_token, None, Some(LEASE_REQUEST_TIMEOUT))
        {
            Ok(allocation) if allocation.state == "ready" => Ok(allocation.lease_until),
            Ok(allocation) => {
                self.lease_lost.store(true, Ordering::SeqCst);
                tracing::error!(
                    target: "akh::backend::execd",
                    session = %self.config.session_id,
                    state = %allocation.state,
                    "execd allocation left the ready state; workspace effects are fenced"
                );
                Err(self.lease_expired())
            }
            Err(HttpFailure::Fault(fault))
                if matches!(fault.code.as_str(), "expired" | "conflict" | "not_found") =>
            {
                self.lease_lost.store(true, Ordering::SeqCst);
                tracing::error!(
                    target: "akh::backend::execd",
                    session = %self.config.session_id,
                    code = %fault.code,
                    message = %fault.message,
                    "execd lease renewal refused; workspace effects are fenced"
                );
                Err(self.lease_expired())
            }
            Err(HttpFailure::Fault(fault)) => Err(ExecdError::Transport {
                url,
                message: fault.to_string(),
            }),
            Err(HttpFailure::Transport(message)) => Err(ExecdError::Transport { url, message }),
        }
    }

    /// Wait for a stop request, for at most `timeout` (`None`: indefinitely).
    /// Returns `true` once stopped.
    fn wait_for_stop(&self, timeout: Option<Duration>) -> bool {
        let stopped = self.stop.lock().unwrap_or_else(|e| e.into_inner());
        let stopped = match timeout {
            Some(timeout) => {
                self.wake
                    .wait_timeout_while(stopped, timeout, |s| !*s)
                    .unwrap_or_else(|e| e.into_inner())
                    .0
            }
            None => self
                .wake
                .wait_while(stopped, |s| !*s)
                .unwrap_or_else(|e| e.into_inner()),
        };
        *stopped
    }

    fn release(&self) {
        let url = format!(
            "{}/v1/allocations/{}/release",
            self.config.sandbox_url.trim_end_matches('/'),
            segment(&self.config.session_id)
        );
        match self.post::<Allocation>(
            &url,
            &self.sandbox_token,
            None,
            Some(LEASE_REQUEST_TIMEOUT.min(Duration::from_secs(5))),
        ) {
            Ok(allocation) => tracing::info!(
                target: "akh::backend::execd",
                session = %self.config.session_id,
                state = %allocation.state,
                "released execd workspace allocation"
            ),
            Err(failure) => tracing::warn!(
                target: "akh::backend::execd",
                session = %self.config.session_id,
                ?failure,
                "could not release the execd allocation; its lease will lapse on its own"
            ),
        }
    }
}

/// The lease thread: renew every interval until stopped, then release.
///
/// The wait is a condvar wait, so a stop request wakes it at once. A stop that
/// arrives during a renewal takes effect when that exchange returns (at most
/// [`LEASE_REQUEST_TIMEOUT`]). [`ExecdBackend`]'s `Drop` does not join this
/// thread; the thread owns the release, so dropping a backend never blocks on
/// the network.
fn spawn_renewer(shared: Arc<Shared>) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("akh-execd-lease".into())
        .spawn(move || {
            loop {
                if shared.wait_for_stop(Some(shared.config.lease_renew_interval)) {
                    shared.release();
                    return;
                }
                match shared.renew() {
                    Ok(lease_until) => tracing::debug!(
                        target: "akh::backend::execd",
                        session = %shared.config.session_id,
                        lease_until,
                        "renewed execd workspace lease"
                    ),
                    // Terminal: the lease is gone and every verb now fails closed.
                    Err(ExecdError::LeaseExpired { .. }) => {
                        if shared.wait_for_stop(None) {
                            shared.release();
                        }
                        return;
                    }
                    Err(error) => tracing::warn!(
                        target: "akh::backend::execd",
                        session = %shared.config.session_id,
                        %error,
                        "execd lease renewal failed; retrying at the next interval"
                    ),
                }
            }
        })
}

// ---------------------------------------------------------------------------
// ExecdBackend
// ---------------------------------------------------------------------------

/// The akhsbxd/akhexecd [`Backend`] (ADR 113 decision 3).
///
/// One `ExecdBackend` is one allocation: every file verb and every `exec`
/// lands in the same sandbox workspace, under the same lease.
pub struct ExecdBackend {
    shared: Arc<Shared>,
    renewer: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for ExecdBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecdBackend")
            .field("session", &self.shared.config.session_id)
            .field("binding", &self.shared.binding)
            .finish_non_exhaustive()
    }
}

impl ExecdBackend {
    /// Allocate a workspace and start renewing its lease.
    ///
    /// Blocks until akhsbxd reports the allocation ready (retrying `busy` for
    /// up to [`ExecdConfig::allocate_timeout`]). Fails rather than falling back
    /// to any local execution.
    pub fn connect(config: ExecdConfig) -> ExecdResult<Self> {
        config.validate()?;
        let sandbox_token = read_token(&config.sandbox_token_file)?;
        let executor_token = read_token(config.executor.token_file())?;
        let http = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout(config.request_timeout)
            .redirects(0)
            .build();
        let allocation = Self::allocate(&http, &config, &sandbox_token)?;
        let binding = allocation.binding.ok_or_else(|| ExecdError::Allocate {
            profile: config.profile.clone(),
            code: "invalid_reply".into(),
            message: "the ready allocation has no executor binding".into(),
        })?;
        let operations_url = match &config.executor {
            ExecutorRoute::ViaSandbox { .. } => format!(
                "{}/v1/executors/{}/v1/operations",
                config.sandbox_url.trim_end_matches('/'),
                segment(&binding.sandbox_id)
            ),
            ExecutorRoute::Direct { url, .. } => {
                format!("{}/v1/operations", url.trim_end_matches('/'))
            }
        };
        let confinement = classify_confinement(&binding.confinement);
        tracing::info!(
            target: "akh::backend::execd",
            session = %config.session_id,
            profile = %config.profile,
            sandbox = %binding.sandbox_id,
            executor = %binding.executor,
            site = %binding.site,
            root = %binding.root,
            confinement_label = %binding.confinement,
            ?confinement,
            "allocated execd workspace"
        );
        let shared = Arc::new(Shared {
            read_roots: vec![PathBuf::from(&binding.root)],
            config,
            http,
            sandbox_token,
            executor_token,
            operations_url,
            binding,
            lease_lost: AtomicBool::new(false),
            stop: Mutex::new(false),
            wake: Condvar::new(),
        });
        let renewer = match spawn_renewer(Arc::clone(&shared)) {
            Ok(handle) => handle,
            Err(error) => {
                shared.release();
                return Err(ExecdError::Config {
                    reason: format!("cannot start the lease renewal thread: {error}"),
                });
            }
        };
        Ok(Self {
            shared,
            renewer: Some(renewer),
        })
    }

    fn allocate(http: &ureq::Agent, config: &ExecdConfig, token: &str) -> ExecdResult<Allocation> {
        let url = format!(
            "{}/v1/allocations",
            config.sandbox_url.trim_end_matches('/')
        );
        let body = serde_json::to_value(Allocate {
            bootstrap: None,
            session_id: config.session_id.clone(),
            principal: config.principal.clone(),
            profile: config.profile.clone(),
        })
        .map_err(|error| ExecdError::Config {
            reason: format!("cannot encode the allocation request: {error}"),
        })?;
        let allocate_error = |code: &str, message: String| ExecdError::Allocate {
            profile: config.profile.clone(),
            code: code.into(),
            message,
        };
        let deadline = Instant::now() + config.allocate_timeout;
        let mut backoff = Duration::from_millis(250);
        loop {
            let reply = http
                .post(&url)
                .set("Authorization", &format!("Bearer {token}"))
                .send_json(&body);
            let retry = match reply {
                Ok(response) => {
                    let allocation: Allocation =
                        response
                            .into_json()
                            .map_err(|error| ExecdError::Transport {
                                url: url.clone(),
                                message: format!("invalid allocation reply: {error}"),
                            })?;
                    if allocation.state == "ready" && allocation.binding.is_some() {
                        return Ok(allocation);
                    }
                    format!("allocation is {}", allocation.state)
                }
                Err(ureq::Error::Status(status, response)) => {
                    let text = response.into_string().unwrap_or_default();
                    match serde_json::from_str::<Fault>(&text) {
                        Ok(fault) if fault.code == "busy" => fault.message,
                        Ok(fault) => return Err(allocate_error(&fault.code, fault.message)),
                        Err(_) => {
                            return Err(allocate_error(
                                "http_status",
                                format!("HTTP {status}: {text}"),
                            ));
                        }
                    }
                }
                Err(ureq::Error::Transport(error)) => {
                    return Err(ExecdError::Transport {
                        url: url.clone(),
                        message: error.to_string(),
                    });
                }
            };
            if Instant::now() + backoff > deadline {
                return Err(allocate_error(
                    "busy",
                    format!(
                        "not ready after {}s: {retry}",
                        config.allocate_timeout.as_secs()
                    ),
                ));
            }
            std::thread::sleep(backoff);
            backoff = (backoff * 2).min(Duration::from_secs(5));
        }
    }

    /// The workspace binding akhsbxd returned.
    pub fn binding(&self) -> &WorkspaceBinding {
        &self.shared.binding
    }

    /// The allocation's session id (the correlation key for akhexecd receipts).
    pub fn session_id(&self) -> &str {
        &self.shared.config.session_id
    }

    /// Whether the lease has lapsed, which fences every verb.
    pub fn lease_expired(&self) -> bool {
        self.shared.lease_lost.load(Ordering::SeqCst)
    }

    /// Renew the lease now instead of waiting for the background thread.
    pub fn renew_lease(&self) -> ExecdResult<u64> {
        self.shared.renew()
    }

    /// Replace exactly one occurrence of `old_text` with `new_text`
    /// (akhexecd's revision-checked `edit` verb). Returns the new revision.
    pub fn edit(&self, path: &Path, old_text: &str, new_text: &str) -> ExecdResult<String> {
        let path = self.workspace_path(path)?;
        let current: ReadData = self.shared.call_as(Verb::Read, json!({"path": path}))?;
        let data = self.shared.call(
            Verb::Edit,
            json!({"path": path, "old_text": old_text, "new_text": new_text,
                   "expected_revision": current.revision}),
        )?;
        Ok(data["revision"].as_str().unwrap_or_default().to_owned())
    }

    /// Create a directory and its parents inside the workspace.
    pub fn mkdir(&self, path: &Path) -> ExecdResult<()> {
        let path = self.workspace_path(path)?;
        self.shared
            .call(Verb::Mkdir, json!({"path": path}))
            .map(drop)
    }

    /// akhexecd's server-side regex search (`search` verb): at most 1000
    /// matches, `glob` filters paths. Returns the raw `{matches, truncated}`.
    pub fn search(&self, pattern: &str, glob: Option<&str>, path: &Path) -> ExecdResult<Value> {
        let path = self.workspace_path(path)?;
        let mut arguments = json!({"pattern": pattern, "path": path});
        if let Some(glob) = glob {
            arguments["glob"] = json!(glob);
        }
        self.shared.call(Verb::Search, arguments)
    }

    /// Map a caller path to the workspace-relative form akhexecd takes.
    fn workspace_path(&self, path: &Path) -> ExecdResult<String> {
        let outside = || ExecdError::OutsideWorkspace {
            path: path.display().to_string(),
            root: self.shared.root().display().to_string(),
        };
        let relative = if path.is_absolute() {
            path.strip_prefix(self.shared.root())
                .map_err(|_| outside())?
        } else {
            path
        };
        let relative = relative.to_str().ok_or_else(outside)?;
        Ok(if relative.is_empty() {
            ".".into()
        } else {
            relative.into()
        })
    }

    fn stat(&self, path: &Path) -> ExecdResult<WorkspaceStat> {
        let path = self.workspace_path(path)?;
        self.shared.call_as(Verb::Stat, json!({"path": path}))
    }

    fn fs_error(&self, path: &Path, kind: &'static str, error: ExecdError) -> FsError {
        match error {
            ExecdError::OutsideWorkspace { .. } => FsError::OutsideSandbox {
                path: path.display().to_string(),
                kind,
                roots: self.shared.read_roots.clone(),
            },
            ExecdError::Operation { code, message, .. } if code == "not_found" => FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, message),
            },
            other => FsError::Execd(other),
        }
    }

    /// Settle a terminal process receipt into an outcome.
    fn finish(
        &self,
        request: &ExecRequest,
        process: &str,
        status: TurnStatus,
        result: Option<ToolResult>,
    ) -> ExecResult<ExecOutcome> {
        if status == TurnStatus::Unknown {
            return Err(ExecdError::UnknownOutcome {
                operation: Verb::ExecStart.as_str(),
                receipt: process.to_owned(),
            }
            .into());
        }
        let data = result.as_ref().and_then(|r| r.data.clone());
        let Some(process_result) =
            data.and_then(|data| serde_json::from_value::<ProcessResult>(data).ok())
        else {
            let fault = receipt_fault(result.as_ref());
            return Err(ExecdError::Operation {
                operation: Verb::ExecStart.as_str(),
                code: fault.code,
                message: fault.message,
            }
            .into());
        };
        if process_result.timed_out {
            return Err(ExecError::Timeout {
                command: request.command.clone(),
                timeout: request.timeout,
            });
        }
        if process_result.cancelled {
            return Err(ExecdError::Operation {
                operation: Verb::ExecStart.as_str(),
                code: "cancelled".into(),
                message: format!("process {process} was cancelled before it exited"),
            }
            .into());
        }
        if process_result.truncated {
            tracing::warn!(
                target: "akh::backend::execd",
                process,
                "akhexecd truncated the process output at its 1 MiB capture limit"
            );
        }
        Ok(ExecOutcome {
            exit_code: process_result.exit_code.unwrap_or(-1),
            success: process_result.exit_code == Some(0),
            stdout: process_result.stdout.into_bytes(),
            stderr: process_result.stderr.into_bytes(),
        })
    }

    /// The deadline passed: cancel, give the executor a moment to confirm,
    /// and report the timeout.
    fn cancel_after_deadline(
        &self,
        request: &ExecRequest,
        process: &str,
    ) -> ExecResult<ExecOutcome> {
        tracing::warn!(
            target: "akh::backend::execd",
            process,
            timeout_secs = request.timeout.as_secs(),
            "execd process outlived its deadline; cancelling"
        );
        self.shared
            .call(Verb::ExecCancel, json!({"process_id": process}))?;
        // Best-effort confirmation: the executor kills the process group on
        // its next supervision tick (every 20 ms).
        let confirm = self.shared.call_as::<PollData>(
            Verb::ExecPoll,
            json!({"process_id": process, "wait_ms": 2_000}),
        );
        if let Ok(PollData {
            status: TurnStatus::Unknown,
            ..
        }) = confirm
        {
            return Err(ExecdError::UnknownOutcome {
                operation: Verb::ExecCancel.as_str(),
                receipt: process.to_owned(),
            }
            .into());
        }
        Err(ExecError::Timeout {
            command: request.command.clone(),
            timeout: request.timeout,
        })
    }

    fn exec_inner(&self, request: &ExecRequest) -> ExecResult<ExecOutcome> {
        let cwd = match &request.cwd {
            Some(cwd) => self.workspace_path(cwd)?,
            None => ".".into(),
        };
        let timeout_ms = u64::try_from(request.timeout.as_millis())
            .unwrap_or(u64::MAX)
            .clamp(1, MAX_PROCESS_TIMEOUT_MS);
        let mut arguments = json!({
            "command": request.command,
            "cwd": cwd,
            "timeout_ms": timeout_ms,
        });
        if !request.env.is_empty() {
            let env: serde_json::Map<String, Value> = request
                .env
                .iter()
                .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                .collect();
            arguments["env"] = Value::Object(env);
        }
        let started = self.shared.operate(Verb::ExecStart, arguments)?;
        let process = started.id.clone();
        match started.status {
            TurnStatus::Completed | TurnStatus::Cancelled => {
                return self.finish(request, &process, started.status, started.result);
            }
            TurnStatus::Failed => {
                let fault = receipt_fault(started.result.as_ref());
                return Err(ExecdError::Operation {
                    operation: Verb::ExecStart.as_str(),
                    code: fault.code,
                    message: fault.message,
                }
                .into());
            }
            _ => {}
        }
        let deadline = Instant::now() + Duration::from_millis(timeout_ms) + CANCEL_GRACE;
        // Polls are read-only, so a failed poll is retried with backoff rather
        // than abandoning a process that may still be running.
        let give_up = deadline + self.shared.config.poll_retry_budget;
        let mut backoff = POLL_RETRY_INITIAL_BACKOFF;
        let mut failing = false;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() && !failing {
                return self.cancel_after_deadline(request, &process);
            }
            let wait_ms = self
                .shared
                .config
                .poll_wait
                .min(remaining)
                .as_millis()
                .max(1) as u64;
            let error = match self.shared.call_as::<PollData>(
                Verb::ExecPoll,
                json!({"process_id": process, "wait_ms": wait_ms}),
            ) {
                Ok(poll) if poll.status.terminal() || poll.status == TurnStatus::Unknown => {
                    return self.finish(request, &process, poll.status, poll.result);
                }
                Ok(_) => {
                    failing = false;
                    backoff = POLL_RETRY_INITIAL_BACKOFF;
                    continue;
                }
                Err(error) => error,
            };
            match PollFailure::classify(error) {
                PollFailure::Terminal(error) => return Err(error.into()),
                PollFailure::Retryable(error) if Instant::now() + backoff < give_up => {
                    tracing::warn!(
                        target: "akh::backend::execd",
                        process = %process,
                        %error,
                        backoff_ms = backoff.as_millis() as u64,
                        "exec_poll failed; retrying the read-only poll"
                    );
                    failing = true;
                    std::thread::sleep(backoff);
                    backoff = (backoff * 2).min(POLL_RETRY_MAX_BACKOFF);
                }
                PollFailure::Retryable(error) | PollFailure::Definite(error) => {
                    return Err(self.abandon_process(&process, error).into());
                }
            }
        }
    }

    /// Stop following a process whose polls keep failing: request a
    /// best-effort cancel and report the outcome as unknown.
    fn abandon_process(&self, process: &str, cause: ExecdError) -> ExecdError {
        let cancel_requested = match self
            .shared
            .call(Verb::ExecCancel, json!({"process_id": process}))
        {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(
                    target: "akh::backend::execd",
                    process,
                    %error,
                    "best-effort exec_cancel after lost polls failed"
                );
                false
            }
        };
        tracing::error!(
            target: "akh::backend::execd",
            process,
            %cause,
            cancel_requested,
            "lost track of an execd process; its outcome is unknown"
        );
        ExecdError::ProcessOutcomeUnknown {
            process: process.to_owned(),
            cause: cause.to_string(),
            cancel_requested,
        }
    }
}

impl ExecdBackend {
    /// Ask the lease thread to stop and release the allocation, without
    /// waiting for it. Returns the thread's handle (if it is still owned) so a
    /// caller that must observe the release can join it.
    fn request_stop(&mut self) -> Option<JoinHandle<()>> {
        *self.shared.stop.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.shared.wake.notify_all();
        self.renewer.take()
    }
}

impl Drop for ExecdBackend {
    /// Non-blocking: the lease thread wakes on the stop signal, releases the
    /// allocation (best effort) and exits on its own. Dropping a backend never
    /// waits out a lease renewal or the release request, so evicting an agent
    /// does not stall its caller.
    fn drop(&mut self) {
        // Detach: the thread holds its own `Arc<Shared>`.
        drop(self.request_stop());
    }
}

impl Filesystem for ExecdBackend {
    fn read_to_string(&self, path: &Path) -> FsResult<String> {
        self.workspace_path(path)
            .and_then(|relative| {
                self.shared
                    .call_as::<ReadData>(Verb::Read, json!({"path": relative}))
            })
            .map(|data| data.content)
            .map_err(|error| self.fs_error(path, "read", error))
    }

    /// Overwrite-or-create, like [`crate::fs::LocalFs::write`]: parents are
    /// created, an existing file is replaced under its current revision (so a
    /// concurrent editor's change surfaces as a `conflict`, never a lost
    /// update), and a missing file is created atomically.
    fn write(&self, path: &Path, contents: &str) -> FsResult<()> {
        let attempt = || -> ExecdResult<()> {
            let relative = self.workspace_path(path)?;
            if let Some(parent) = Path::new(&relative).parent()
                && !parent.as_os_str().is_empty()
            {
                self.shared
                    .call(Verb::Mkdir, json!({"path": parent.to_string_lossy()}))?;
            }
            let revision = match self
                .shared
                .call_as::<ReadData>(Verb::Read, json!({"path": relative, "encoding": "base64"}))
            {
                Ok(current) => Some(current.revision),
                Err(ExecdError::Operation { code, .. }) if code == "not_found" => None,
                Err(error) => return Err(error),
            };
            let mut arguments = json!({"path": relative, "content": contents});
            match revision {
                Some(revision) => arguments["expected_revision"] = json!(revision),
                None => arguments["create"] = json!(true),
            }
            self.shared.call(Verb::Write, arguments).map(drop)
        };
        attempt().map_err(|error| self.fs_error(path, "write", error))
    }

    fn read_dir(&self, path: &Path) -> FsResult<Vec<PathBuf>> {
        let relative = self
            .workspace_path(path)
            .map_err(|error| self.fs_error(path, "read", error))?;
        let listing: ListData = self
            .shared
            .call_as(Verb::List, json!({"path": relative}))
            .map_err(|error| self.fs_error(path, "read", error))?;
        let base = if relative == "." {
            self.shared.root().to_path_buf()
        } else {
            self.shared.root().join(&relative)
        };
        Ok(listing
            .entries
            .into_iter()
            .map(|name| base.join(name))
            .collect())
    }

    fn exists(&self, path: &Path) -> bool {
        self.stat(path).is_ok_and(|stat| stat.exists)
    }

    fn read_roots(&self) -> &[PathBuf] {
        &self.shared.read_roots
    }

    fn is_writable(&self) -> bool {
        !self.lease_expired()
    }

    fn file_size(&self, path: &Path) -> FsResult<u64> {
        let stat = self
            .stat(path)
            .map_err(|error| self.fs_error(path, "read", error))?;
        match (stat.exists, stat.size) {
            (true, Some(size)) => Ok(size),
            (true, None) => Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::other("not a regular file"),
            }),
            (false, _) => Err(FsError::Io {
                path: path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
            }),
        }
    }

    fn is_symlink(&self, path: &Path) -> bool {
        self.stat(path).is_ok_and(|stat| stat.is_symlink())
    }
}

impl Backend for ExecdBackend {
    fn exec(&self, request: ExecRequest) -> ExecResult<ExecOutcome> {
        self.exec_inner(&request)
    }

    fn handshake(&self) -> Posture {
        Posture {
            confinement: classify_confinement(&self.shared.binding.confinement),
        }
    }

    /// `backend = "execd"`; `host` is `<site>/<sandbox id>`; the executor's OS
    /// user is not reported by the binding, so `os_user` stays `None` rather
    /// than guessing. `correlation_id` is the allocation's session id, which
    /// every akhexecd receipt carries.
    fn locus(&self) -> ExecLocus {
        let binding = &self.shared.binding;
        ExecLocus {
            backend: "execd",
            host: Some(format!("{}/{}", binding.site, binding.sandbox_id)),
            os_user: None,
            correlation_id: Some(self.shared.config.session_id.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    /// One request as the mock saw it.
    #[derive(Debug, Clone)]
    struct Seen {
        path: String,
        authorization: String,
        body: Value,
    }

    type Handler = dyn Fn(&Seen) -> (u16, Value) + Send + Sync;

    /// A minimal routing HTTP/1.1 server on loopback, one request per
    /// connection (`Connection: close`), recording every request.
    fn spawn_mock(handler: Arc<Handler>) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut writer = stream;
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                let mut content_length = 0;
                let mut authorization = String::new();
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    let (name, value) = line.split_once(':').unwrap_or((&line, ""));
                    match name.trim().to_ascii_lowercase().as_str() {
                        "content-length" => content_length = value.trim().parse().unwrap_or(0),
                        "authorization" => authorization = value.trim().to_string(),
                        _ => {}
                    }
                }
                let mut body = vec![0; content_length];
                reader.read_exact(&mut body).unwrap();
                let request = Seen {
                    path,
                    authorization,
                    body: serde_json::from_slice(&body).unwrap_or(Value::Null),
                };
                let (status, reply) = handler(&request);
                log.lock().unwrap().push(request);
                let reply = reply.to_string();
                let _ = write!(
                    writer,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        (url, seen)
    }

    const ROOT: &str = "/srv/akh-workspaces/akh-fixture";

    fn binding(confinement: &str) -> Value {
        json!({"id":"akh-fixture","sandbox_id":"akh-fixture","executor":"executor","generation":1,
               "site":"mac-mini","root":ROOT,"platform":"macos","confinement":confinement})
    }

    fn allocation(request: &Value, state: &str, confinement: &str) -> Value {
        json!({"id":"akh-fixture","request":request,"profile_revision":"1","state":state,
               "binding":binding(confinement),"error":null,"lease_until":now() + 300_000})
    }

    fn now() -> u64 {
        akh_session_types::now_ms()
    }

    /// A receipt echoing the request with the given status and result.
    fn receipt(request: &Value, status: &str, result: Value) -> Value {
        json!({"id":request["id"],"request":request,"status":status,"result":result,
               "started_at":now(),"finished_at":null,"principal":"akhomed"})
    }

    fn ok(data: Value) -> Value {
        json!({"success":true,"text":"","data":data})
    }

    fn token(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("fixture-{name}-token-0123456789\n")).unwrap();
        path
    }

    fn config(dir: &Path, sandbox_url: &str) -> ExecdConfig {
        ExecdConfig::new(
            sandbox_url,
            token(dir, "sandbox"),
            "mac-local",
            "janus",
            "janus",
        )
        .with_poll_wait(Duration::from_millis(50))
        .with_allocate_timeout(Duration::from_secs(5))
    }

    /// The mock's executor behaviour for one scenario.
    struct Scenario {
        /// How many `exec_poll`s report the process still running.
        running_polls: usize,
        /// Terminal process data once polls finish; `None` = never finishes.
        terminal: Option<(&'static str, Value)>,
        confinement: &'static str,
    }

    fn scenario_handler(scenario: Scenario) -> Arc<Handler> {
        let polls = AtomicUsize::new(0);
        Arc::new(move |seen: &Seen| {
            let body = &seen.body;
            match seen.path.as_str() {
                "/v1/allocations" => (200, allocation(body, "ready", scenario.confinement)),
                path if path.ends_with("/renew") => {
                    (200, allocation(&json!({}), "ready", scenario.confinement))
                }
                path if path.ends_with("/release") => (
                    200,
                    allocation(&json!({}), "released", scenario.confinement),
                ),
                "/v1/executors/akh-fixture/v1/operations" => {
                    match body["operation"].as_str().unwrap() {
                        "exec_start" => (200, receipt(body, "running", Value::Null)),
                        "exec_poll" => {
                            let n = polls.fetch_add(1, Ordering::SeqCst);
                            let (status, result) = match &scenario.terminal {
                                Some((status, data)) if n >= scenario.running_polls => {
                                    (*status, ok(data.clone()))
                                }
                                _ => {
                                    // Emulate the server-side wait so a
                                    // never-ending process is not a busy loop.
                                    let wait = body["arguments"]["wait_ms"].as_u64().unwrap_or(0);
                                    std::thread::sleep(Duration::from_millis(wait.min(50)));
                                    ("running", Value::Null)
                                }
                            };
                            (
                                200,
                                receipt(
                                    body,
                                    "completed",
                                    ok(json!({"process_id":body["arguments"]["process_id"],
                                              "status":status,"result":result,"progress":null})),
                                ),
                            )
                        }
                        "exec_cancel" => (
                            200,
                            receipt(
                                body,
                                "completed",
                                ok(json!({"cancel_requested":true,"terminal":false})),
                            ),
                        ),
                        other => panic!("unexpected operation {other}"),
                    }
                }
                other => (404, json!({"code":"not_found","message":other})),
            }
        })
    }

    fn operations(seen: &Arc<Mutex<Vec<Seen>>>) -> Vec<String> {
        seen.lock()
            .unwrap()
            .iter()
            .filter(|s| s.path.contains("/operations"))
            .map(|s| s.body["operation"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn allocate_exec_poll_collects_output_through_the_gateway() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(scenario_handler(Scenario {
            running_polls: 2,
            terminal: Some((
                "completed",
                json!({"stdout":"built\n","stderr":"warn\n","exit_code":0,
                       "timed_out":false,"cancelled":false,"truncated":false}),
            )),
            confinement: "unconfined",
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let outcome = backend
            .exec(
                ExecRequest::shell("cargo build", Duration::from_secs(30))
                    .with_cwd(format!("{ROOT}/crate"))
                    .with_env("RUST_LOG", "info"),
            )
            .unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.exit_code, 0);
        assert_eq!(outcome.stdout, b"built\n");
        assert_eq!(outcome.stderr, b"warn\n");
        assert_eq!(
            operations(&seen),
            ["exec_start", "exec_poll", "exec_poll", "exec_poll"]
        );
        let requests = seen.lock().unwrap().clone();
        let allocate = &requests[0];
        assert_eq!(allocate.path, "/v1/allocations");
        assert_eq!(allocate.body["profile"], "mac-local");
        assert_eq!(allocate.body["principal"], "janus");
        assert_eq!(
            allocate.authorization,
            "Bearer fixture-sandbox-token-0123456789"
        );
        let start = &requests[1].body;
        assert_eq!(start["workspace_id"], "akh-fixture");
        assert_eq!(start["generation"], 1);
        // The host-side absolute cwd became workspace-relative; env passed through.
        assert_eq!(start["arguments"]["cwd"], "crate");
        assert_eq!(start["arguments"]["env"]["RUST_LOG"], "info");
        assert_eq!(start["arguments"]["timeout_ms"], 30_000);
        assert!(requests[2].body["arguments"]["wait_ms"].as_u64().unwrap() <= 50);
        // Every operation has a fresh id.
        let ids: std::collections::HashSet<_> = requests[1..]
            .iter()
            .map(|r| r.body["id"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn a_process_that_outlives_its_deadline_is_cancelled_and_reported_as_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(scenario_handler(Scenario {
            running_polls: usize::MAX,
            terminal: None,
            confinement: "unconfined",
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let started = Instant::now();
        let error = backend
            .exec(ExecRequest::shell("sleep 3600", Duration::from_millis(1)))
            .unwrap_err();
        assert!(
            matches!(&error, ExecError::Timeout { command, .. } if command == "sleep 3600"),
            "expected Timeout, got {error:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        let ops = operations(&seen);
        assert_eq!(ops.first().map(String::as_str), Some("exec_start"));
        let cancel = ops
            .iter()
            .position(|op| op == "exec_cancel")
            .expect("cancel sent");
        // One confirmation poll follows the cancel.
        assert_eq!(ops[cancel + 1..], ["exec_poll"]);
        let requests = seen.lock().unwrap().clone();
        let cancel_request = requests
            .iter()
            .find(|r| r.body["operation"] == "exec_cancel")
            .unwrap();
        assert_eq!(
            cancel_request.body["arguments"]["process_id"],
            requests[1].body["id"]
        );
    }

    #[test]
    fn the_executors_own_timeout_maps_to_exec_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let (url, _seen) = spawn_mock(scenario_handler(Scenario {
            running_polls: 0,
            terminal: Some((
                "completed",
                json!({"stdout":"","stderr":"","exit_code":null,"timed_out":true,
                       "cancelled":false,"truncated":false}),
            )),
            confinement: "unconfined",
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let error = backend
            .exec(ExecRequest::shell("yes", Duration::from_secs(5)))
            .unwrap_err();
        assert!(matches!(error, ExecError::Timeout { .. }), "{error:?}");
    }

    #[test]
    fn an_unknown_process_outcome_maps_to_unknown_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let (url, _seen) = spawn_mock(scenario_handler(Scenario {
            running_polls: 1,
            terminal: Some(("unknown", json!({"code":"unknown_outcome"}))),
            confinement: "unconfined",
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let error = backend
            .exec(ExecRequest::shell("make install", Duration::from_secs(5)))
            .unwrap_err();
        let ExecError::Execd(ExecdError::UnknownOutcome { operation, receipt }) = &error else {
            panic!("expected UnknownOutcome, got {error:?}");
        };
        assert_eq!(*operation, "exec_start");
        assert!(!receipt.is_empty());
        let report = format!("{:?}", miette::Report::new(error));
        assert!(
            report.contains("akh::backend::execd::unknown_outcome"),
            "{report}"
        );
    }

    #[test]
    fn an_unknown_receipt_and_a_fenced_mutation_both_map_to_unknown_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let (url, _seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "unconfined")),
            path if path.ends_with("/release") => {
                (200, allocation(&json!({}), "released", "unconfined"))
            }
            _ => match seen.body["operation"].as_str().unwrap() {
                // The executor restarted mid-write: the receipt itself is unknown.
                "write" => (
                    200,
                    receipt(
                        &seen.body,
                        "unknown",
                        json!({"success":false,"text":"restarted",
                        "data":{"code":"unknown_outcome"}}),
                    ),
                ),
                "read" => (
                    200,
                    receipt(
                        &seen.body,
                        "failed",
                        json!({"success":false,"text":"missing",
                        "data":{"error":{"code":"not_found","message":"No such file"}}}),
                    ),
                ),
                "mkdir" => (
                    409,
                    json!({"code":"unknown_outcome","message":"workspace fenced"}),
                ),
                other => panic!("unexpected {other}"),
            },
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let error = backend.write(Path::new("new.txt"), "hello").unwrap_err();
        assert!(
            matches!(
                error,
                FsError::Execd(ExecdError::UnknownOutcome {
                    operation: "write",
                    ..
                })
            ),
            "{error:?}"
        );
        assert!(matches!(
            backend.mkdir(Path::new("d")).unwrap_err(),
            ExecdError::UnknownOutcome {
                operation: "mkdir",
                ..
            }
        ));
    }

    #[test]
    fn file_verbs_map_paths_and_revisions() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "unconfined")),
            path if path.ends_with("/release") => {
                (200, allocation(&json!({}), "released", "unconfined"))
            }
            _ => {
                let args = &seen.body["arguments"];
                let data = match (
                    seen.body["operation"].as_str().unwrap(),
                    args["path"].as_str(),
                ) {
                    ("read", Some("src/lib.rs")) => {
                        json!({"content":"fn main() {}","encoding":"utf8","revision":"rev-1"})
                    }
                    ("mkdir", _) | ("write", _) => json!({"revision":"rev-2"}),
                    ("list", Some("src")) => json!({"entries":["lib.rs","link"]}),
                    ("stat", Some("src/lib.rs")) => {
                        json!({"exists":true,"kind":"file","size":12})
                    }
                    ("stat", Some("src/link")) => json!({"exists":true,"kind":"symlink"}),
                    ("stat", _) => json!({"exists":false}),
                    other => panic!("unexpected {other:?}"),
                };
                (200, receipt(&seen.body, "completed", ok(data)))
            }
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let root = PathBuf::from(ROOT);
        assert_eq!(backend.read_roots(), std::slice::from_ref(&root));
        assert_eq!(
            backend.read_to_string(&root.join("src/lib.rs")).unwrap(),
            "fn main() {}"
        );
        assert_eq!(
            backend.read_dir(Path::new("src")).unwrap(),
            [root.join("src/lib.rs"), root.join("src/link")]
        );
        assert_eq!(backend.file_size(Path::new("src/lib.rs")).unwrap(), 12);
        assert!(backend.is_symlink(Path::new("src/link")));
        assert!(!backend.exists(Path::new("absent")));
        assert!(backend.exists(&root.join("src/lib.rs")));
        // An existing file is replaced under its current revision.
        backend
            .write(&root.join("src/lib.rs"), "fn main() { run() }")
            .unwrap();
        let requests = seen.lock().unwrap().clone();
        let write = requests
            .iter()
            .find(|r| r.body["operation"] == "write")
            .unwrap();
        assert_eq!(write.body["arguments"]["expected_revision"], "rev-1");
        assert!(write.body["arguments"].get("create").is_none());
        assert!(
            requests
                .iter()
                .any(|r| r.body["operation"] == "mkdir" && r.body["arguments"]["path"] == "src")
        );
        // Host paths outside the workspace never reach the executor.
        let before = seen.lock().unwrap().len();
        assert!(matches!(
            backend.read_to_string(Path::new("/etc/passwd")),
            Err(FsError::OutsideSandbox { kind: "read", .. })
        ));
        assert!(matches!(
            backend
                .exec(ExecRequest::shell("ls", Duration::from_secs(1)).with_cwd("/Users/someone")),
            Err(ExecError::Execd(ExecdError::OutsideWorkspace { .. }))
        ));
        assert_eq!(seen.lock().unwrap().len(), before);
    }

    #[test]
    fn a_refused_renewal_fences_every_verb_as_lease_expired() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "unconfined")),
            path if path.ends_with("/renew") => (
                400,
                json!({"code":"expired","message":"workspace lease expired or was released"}),
            ),
            path if path.ends_with("/release") => {
                (200, allocation(&json!({}), "released", "unconfined"))
            }
            other => panic!("no operation may be sent after the lease lapsed: {other}"),
        }));
        let backend = ExecdBackend::connect(
            config(dir.path(), &url).with_lease_renew_interval(Duration::from_millis(20)),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !backend.lease_expired() {
            assert!(Instant::now() < deadline, "renewer never ran");
            std::thread::sleep(Duration::from_millis(10));
        }
        let renew_path = format!("/v1/allocations/{}/renew", backend.session_id());
        assert!(seen.lock().unwrap().iter().any(|s| s.path == renew_path));
        assert!(!backend.is_writable());
        let error = backend
            .exec(ExecRequest::shell("true", Duration::from_secs(1)))
            .unwrap_err();
        assert!(
            matches!(error, ExecError::Execd(ExecdError::LeaseExpired { .. })),
            "{error:?}"
        );
        assert!(matches!(
            backend.read_to_string(Path::new("x")),
            Err(FsError::Execd(ExecdError::LeaseExpired { .. }))
        ));
    }

    #[test]
    fn a_busy_allocation_is_retried_and_a_refused_one_is_a_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&attempts);
        let (url, _seen) = spawn_mock(Arc::new(move |seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" if counter.fetch_add(1, Ordering::SeqCst) < 2 => (
                429,
                json!({"code":"busy","message":"sandbox resource is not ready"}),
            ),
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "gvisor:runsc")),
            _ => (200, allocation(&json!({}), "released", "gvisor:runsc")),
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(backend.handshake().confinement, Confinement::Sandboxed);
        drop(backend);

        let (url, _seen) = spawn_mock(Arc::new(|_: &Seen| {
            (
                404,
                json!({"code":"not_found","message":"workspace profile not found"}),
            )
        }));
        let error = ExecdBackend::connect(config(dir.path(), &url)).unwrap_err();
        assert!(
            matches!(&error, ExecdError::Allocate { code, profile, .. }
                if code == "not_found" && profile == "mac-local"),
            "{error:?}"
        );
    }

    /// A provisioner that never becomes ready within `allocate_timeout` (the
    /// 2026-09-27 incident: a first image pull outliving the 120 s default)
    /// gives up with a typed `busy` error instead of hanging forever.
    #[test]
    fn allocate_gives_up_after_the_timeout_with_a_typed_busy_error() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| {
            (200, allocation(&seen.body, "allocating", "unobserved"))
        }));
        let started = Instant::now();
        let error = ExecdBackend::connect(config(dir.path(), &url)).unwrap_err();
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "should give up around the 5s configured timeout, took {:?}",
            started.elapsed()
        );
        assert!(
            matches!(&error, ExecdError::Allocate { code, .. } if code == "busy"),
            "{error:?}"
        );
        // Every retry inside one `connect` reused the same session id: the
        // mock never saw more than one distinct allocation identity.
        let sessions: std::collections::HashSet<_> = seen
            .lock()
            .unwrap()
            .iter()
            .map(|s| s.body["session_id"].clone())
            .collect();
        assert_eq!(sessions.len(), 1);
    }

    /// The core of the 2026-09-27 fix: a [`PendingAllocations`] cache makes a
    /// retried attempt reattach to the same session instead of minting a
    /// fresh one, so a slow provisioner is never asked to stand up a second
    /// (or 25th) `Sandbox` for the same workspace.
    #[test]
    fn pending_allocations_reattach_to_the_same_session_across_retries() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| {
            (200, allocation(&seen.body, "allocating", "unobserved"))
        }));
        let pending = PendingAllocations::new();
        let fresh = config(dir.path(), &url);
        let first = pending.config_for("vesta", fresh.clone());
        pending.remember("vesta", first.clone());
        // A completely independent `ExecdConfig::new` call for the same
        // target still reuses the remembered session id.
        let second_fresh = config(dir.path(), &url);
        let second = pending.config_for("vesta", second_fresh.clone());
        assert_eq!(first.session_id, second.session_id);
        // The independently minted config (as if a brand-new attempt had
        // never consulted the cache) got a different session id — proving
        // `config_for` really did reattach rather than pass it through.
        assert_ne!(second.session_id, second_fresh.session_id);

        // Actually attempt it twice against the mock: still one identity.
        let _ = ExecdBackend::connect(first);
        let _ = ExecdBackend::connect(second);
        let sessions: std::collections::HashSet<_> = seen
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.path == "/v1/allocations")
            .map(|s| s.body["session_id"].clone())
            .collect();
        assert_eq!(
            sessions.len(),
            1,
            "expected exactly one allocated session id, saw {sessions:?}"
        );
    }

    /// Once a target's `sandbox_url`/`profile`/`principal` changes (an
    /// operator edited `akh.toml` while an allocation was still pending),
    /// `config_for` abandons the stale one — releasing it at akhsbxd — and
    /// mints a fresh session for the new configuration.
    #[test]
    fn pending_allocations_abandon_a_stale_target_on_reconfiguration() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            path if path.ends_with("/release") => {
                (200, allocation(&json!({}), "released", "unobserved"))
            }
            _ => (200, allocation(&seen.body, "allocating", "unobserved")),
        }));
        let pending = PendingAllocations::new();
        let old = ExecdConfig::new(
            url.as_str(),
            token(dir.path(), "sandbox"),
            "old-profile",
            "janus",
            "vesta",
        );
        pending.remember("vesta", old.clone());

        let new = ExecdConfig::new(
            url.as_str(),
            token(dir.path(), "sandbox"),
            "new-profile",
            "janus",
            "vesta",
        );
        let resolved = pending.config_for("vesta", new.clone());
        assert_eq!(resolved.session_id, new.session_id);
        assert_eq!(resolved.profile, "new-profile");

        let release_path = format!("/v1/allocations/{}/release", old.session_id);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !seen.lock().unwrap().iter().any(|s| s.path == release_path) {
            assert!(
                Instant::now() < deadline,
                "stale allocation was never released"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// [`PendingAllocations::abandon`] releases the pending session at
    /// akhsbxd and forgets it, so a later `config_for` mints a fresh one.
    #[test]
    fn pending_allocations_abandon_releases_and_forgets() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            path if path.ends_with("/release") => {
                (200, allocation(&json!({}), "released", "unobserved"))
            }
            _ => (200, allocation(&seen.body, "allocating", "unobserved")),
        }));
        let pending = PendingAllocations::new();
        let cfg = config(dir.path(), &url);
        pending.remember("vesta", cfg.clone());
        pending.abandon("vesta");

        let release_path = format!("/v1/allocations/{}/release", cfg.session_id);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !seen.lock().unwrap().iter().any(|s| s.path == release_path) {
            assert!(
                Instant::now() < deadline,
                "abandoned allocation was never released"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        // Forgotten: the next attempt mints a fresh session.
        let fresh = pending.config_for("vesta", config(dir.path(), &url));
        assert_ne!(fresh.session_id, cfg.session_id);
    }

    /// `allocate_timeout_secs` out of the accepted 1..=3600 range is a typed
    /// config error, not a silent clamp.
    #[test]
    fn allocate_timeout_out_of_range_is_a_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let too_long = config(dir.path(), "http://127.0.0.1:9")
            .with_allocate_timeout(Duration::from_secs(3_601));
        assert!(matches!(
            ExecdBackend::connect(too_long),
            Err(ExecdError::Config { .. })
        ));
        let zero =
            config(dir.path(), "http://127.0.0.1:9").with_allocate_timeout(Duration::from_secs(0));
        assert!(matches!(
            ExecdBackend::connect(zero),
            Err(ExecdError::Config { .. })
        ));
    }

    #[test]
    fn handshake_reports_the_binding_label_honestly() {
        assert_eq!(classify_confinement("unconfined"), Confinement::Unconfined);
        // The Mac `existing` provisioner: akhexecd is a host process, never a sandbox.
        assert_eq!(classify_confinement("unobserved"), Confinement::Unconfined);
        assert_eq!(classify_confinement(""), Confinement::Unconfined);
        assert_eq!(
            classify_confinement("local-test-only"),
            Confinement::Unconfined
        );
        assert_eq!(
            classify_confinement("user:akh-janus"),
            Confinement::PrivilegeDropped
        );
        assert_eq!(classify_confinement("Kata:qemu"), Confinement::Sandboxed);
        assert_eq!(classify_confinement("pod"), Confinement::Sandboxed);

        let dir = tempfile::tempdir().unwrap();
        let (url, _seen) = spawn_mock(scenario_handler(Scenario {
            running_polls: 0,
            terminal: None,
            confinement: "unconfined",
        }));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        assert_eq!(backend.handshake().confinement, Confinement::Unconfined);
        let locus = backend.locus();
        assert_eq!(locus.backend, "execd");
        assert_eq!(locus.host.as_deref(), Some("mac-mini/akh-fixture"));
        assert_eq!(locus.correlation_id.as_deref(), Some(backend.session_id()));
    }

    #[test]
    fn plain_http_is_loopback_only_unless_explicitly_allowed() {
        let dir = tempfile::tempdir().unwrap();
        for loopback in [
            "http://localhost:8410",
            "http://127.0.0.1:8410/",
            "http://[::1]:8410",
        ] {
            config(dir.path(), loopback)
                .validate()
                .unwrap_or_else(|error| panic!("{loopback}: {error}"));
        }
        config(dir.path(), "https://akhsbxd.lab.internal")
            .validate()
            .unwrap();
        for remote in [
            "http://akhsbxd.lab.internal",
            "http://10.0.0.5:8410",
            "http://localhost.evil.example",
            "http://127.0.0.1.nip.io",
        ] {
            let error = config(dir.path(), remote).validate().unwrap_err();
            assert!(
                matches!(&error, ExecdError::Config { reason } if reason.contains("clear text")),
                "{remote}: {error:?}"
            );
            config(dir.path(), remote)
                .with_insecure_http(true)
                .validate()
                .unwrap_or_else(|error| panic!("{remote} with opt-in: {error}"));
        }
        // The executor URL follows the same rule.
        let direct = config(dir.path(), "https://akhsbxd.lab.internal").with_executor(
            ExecutorRoute::Direct {
                url: "http://akhexecd.lab.internal".into(),
                token_file: dir.path().join("sandbox"),
            },
        );
        assert!(matches!(
            direct.clone().validate(),
            Err(ExecdError::Config { .. })
        ));
        direct.with_insecure_http(true).validate().unwrap();
    }

    /// A handler for exec scenarios whose polls fail `failing_polls` times
    /// with a transient `unavailable` fault before the process completes
    /// (`usize::MAX`: every poll fails).
    fn flaky_poll_handler(failing_polls: usize) -> Arc<Handler> {
        let polls = AtomicUsize::new(0);
        Arc::new(move |seen: &Seen| {
            let body = &seen.body;
            match seen.path.as_str() {
                "/v1/allocations" => (200, allocation(body, "ready", "unconfined")),
                path if path.ends_with("/renew") => {
                    (200, allocation(&json!({}), "ready", "unconfined"))
                }
                path if path.ends_with("/release") => {
                    (200, allocation(&json!({}), "released", "unconfined"))
                }
                _ => match body["operation"].as_str().unwrap() {
                    "exec_start" => (200, receipt(body, "running", Value::Null)),
                    "exec_poll" if polls.fetch_add(1, Ordering::SeqCst) < failing_polls => (
                        503,
                        json!({"code":"unavailable","message":"executor restarting"}),
                    ),
                    "exec_poll" => (
                        200,
                        receipt(
                            body,
                            "completed",
                            ok(json!({"process_id":body["arguments"]["process_id"],
                                "status":"completed","progress":null,
                                "result":ok(json!({"stdout":"done\n","stderr":"",
                                    "exit_code":0,"timed_out":false,"cancelled":false,
                                    "truncated":false}))})),
                        ),
                    ),
                    "exec_cancel" => (
                        200,
                        receipt(
                            body,
                            "completed",
                            ok(json!({"cancel_requested":true,"terminal":false})),
                        ),
                    ),
                    other => panic!("unexpected operation {other}"),
                },
            }
        })
    }

    #[test]
    fn a_failed_poll_is_retried_instead_of_abandoning_the_process() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(flaky_poll_handler(2));
        let backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let outcome = backend
            .exec(ExecRequest::shell("cargo build", Duration::from_secs(30)))
            .unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.stdout, b"done\n");
        assert_eq!(
            operations(&seen),
            ["exec_start", "exec_poll", "exec_poll", "exec_poll"]
        );
    }

    #[test]
    fn polls_that_keep_failing_cancel_and_report_an_unknown_outcome() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(flaky_poll_handler(usize::MAX));
        let backend = ExecdBackend::connect(
            config(dir.path(), &url).with_poll_retry_budget(Duration::from_millis(500)),
        )
        .unwrap();
        let started = Instant::now();
        let error = backend
            .exec(ExecRequest::shell("make install", Duration::from_millis(1)))
            .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(10));
        let ExecError::Execd(ExecdError::ProcessOutcomeUnknown {
            process,
            cancel_requested,
            cause,
        }) = &error
        else {
            panic!("expected ProcessOutcomeUnknown, got {error:?}");
        };
        assert!(*cancel_requested);
        assert!(cause.contains("unavailable"), "{cause}");
        let ops = operations(&seen);
        assert!(
            ops.iter().filter(|op| *op == "exec_poll").count() >= 2,
            "{ops:?}"
        );
        assert_eq!(ops.last().map(String::as_str), Some("exec_cancel"));
        let requests = seen.lock().unwrap().clone();
        let cancel = requests
            .iter()
            .find(|r| r.body["operation"] == "exec_cancel")
            .unwrap();
        assert_eq!(&cancel.body["arguments"]["process_id"], process.as_str());
        let report = format!("{:?}", miette::Report::new(error));
        assert!(
            report.contains("akh::backend::execd::process_outcome_unknown"),
            "{report}"
        );
    }

    #[test]
    fn dropping_the_backend_does_not_wait_out_a_renewal_or_the_release() {
        let dir = tempfile::tempdir().unwrap();
        let renewing = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&renewing);
        let (url, seen) = spawn_mock(Arc::new(move |seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "unconfined")),
            path if path.ends_with("/renew") => {
                flag.store(true, Ordering::SeqCst);
                // A slow akhsbxd: the renewal is in flight when the backend drops.
                std::thread::sleep(Duration::from_millis(1_500));
                (200, allocation(&json!({}), "ready", "unconfined"))
            }
            _ => (200, allocation(&json!({}), "released", "unconfined")),
        }));
        let backend = ExecdBackend::connect(
            config(dir.path(), &url).with_lease_renew_interval(Duration::from_millis(20)),
        )
        .unwrap();
        let release_path = format!("/v1/allocations/{}/release", backend.session_id());
        let deadline = Instant::now() + Duration::from_secs(5);
        while !renewing.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "renewer never ran");
            std::thread::sleep(Duration::from_millis(5));
        }
        let started = Instant::now();
        drop(backend);
        assert!(
            started.elapsed() < Duration::from_millis(200),
            "drop blocked for {:?}",
            started.elapsed()
        );
        // The lease thread still releases once the renewal returns.
        let deadline = Instant::now() + Duration::from_secs(10);
        while !seen.lock().unwrap().iter().any(|s| s.path == release_path) {
            assert!(Instant::now() < deadline, "allocation never released");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn an_idle_lease_thread_stops_promptly_and_releases() {
        let dir = tempfile::tempdir().unwrap();
        let (url, seen) = spawn_mock(Arc::new(|seen: &Seen| match seen.path.as_str() {
            "/v1/allocations" => (200, allocation(&seen.body, "ready", "unconfined")),
            _ => (200, allocation(&json!({}), "released", "unconfined")),
        }));
        let mut backend = ExecdBackend::connect(config(dir.path(), &url)).unwrap();
        let release_path = format!("/v1/allocations/{}/release", backend.session_id());
        let started = Instant::now();
        let renewer = backend.request_stop().expect("renewer handle");
        renewer.join().unwrap();
        // The default 100 s renewal wait is interrupted, not waited out.
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(seen.lock().unwrap().iter().any(|s| s.path == release_path));
    }

    #[test]
    fn invalid_settings_and_credentials_fail_before_any_request() {
        let dir = tempfile::tempdir().unwrap();
        let bad_url = config(dir.path(), "ftp://sandbox");
        assert!(matches!(
            ExecdBackend::connect(bad_url),
            Err(ExecdError::Config { .. })
        ));
        let slow_poll =
            config(dir.path(), "http://127.0.0.1:9").with_poll_wait(Duration::from_secs(11));
        assert!(matches!(
            ExecdBackend::connect(slow_poll),
            Err(ExecdError::Config { .. })
        ));
        let short = dir.path().join("short");
        std::fs::write(&short, "too-short").unwrap();
        let mut weak = config(dir.path(), "http://127.0.0.1:9");
        weak.sandbox_token_file = short;
        assert!(matches!(
            ExecdBackend::connect(weak),
            Err(ExecdError::Credential { .. })
        ));
    }
}
