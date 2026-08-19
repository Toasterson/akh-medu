//! ACP control-plane — server-level `_akh/{workspace,daemon,config}/*` methods.
//!
//! The `_akh/*` data-plane ([`super::dispatch_akh`]) answers questions about
//! *one* workspace's engine. A handful of genuine client-control operations,
//! though, span the whole daemon process: listing/creating/deleting workspaces,
//! starting/stopping a workspace's background daemon, and reading/writing the
//! global daemon config. Those have no home in a single-workspace [`Engine`],
//! so they are served by akhomed's `ServerState` — which lives in the binary,
//! out of this library's reach.
//!
//! This module defines the seam: a dyn-compatible [`ServerControl`] trait the
//! binary implements over `ServerState`, plus the canonical list of methods the
//! dispatch router peels off and forwards to that backend (everything else
//! falls through to the data-plane). The single async operation is expressed as
//! a boxed future ([`ControlFuture`]) so `dyn ServerControl` is a valid type.
//!
//! [`Engine`]: crate::engine::Engine

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};

/// Per-workspace runtime handles the ACP data-plane + chat turn need.
///
/// Resolved on demand by [`ServerControl::resolve`] so the ACP server can serve
/// *any* workspace per request (not just one bound at startup), and so it holds
/// no permanent engine reference — a permanent hold would pin the workspace's
/// store and block a same-host `sleep` → `wake`.
pub struct WorkspaceHandles {
    pub engine: Arc<crate::engine::Engine>,
    pub agent: Arc<RwLock<crate::agent::Agent>>,
    pub processor: Arc<Mutex<crate::chat::ChatProcessor>>,
    /// This workspace's `AkhConfig::mcp.require_pushed_mcp` — whether a
    /// `session/prompt` turn must fail when a client-pushed MCP server does
    /// not come back from discovery, rather than degrading gracefully.
    /// Threaded from the akh's persisted config so the ACP layer (which has
    /// no direct `AkhConfig` access of its own) can honor the per-akh policy.
    pub require_pushed_mcp: bool,
}

/// The future returned by [`ServerControl::resolve`].
pub type ResolveFuture<'a> =
    Pin<Box<dyn Future<Output = Result<WorkspaceHandles, String>> + Send + 'a>>;

/// Engine-only handle for read-only ACP verbs (ADR-049 split-brain guard).
///
/// Contains just the engine — no agent, no processor, no WAL, no snapshot
/// ticker. Used by data-plane methods that only need engine-derivable data
/// (symbol/triple counts, KG queries, skills, health) so a read on a
/// non-owning host never triggers lazy agent construction.
pub struct EngineHandle {
    pub engine: Arc<crate::engine::Engine>,
}

/// The future returned by [`ServerControl::resolve_engine_only`].
pub type ResolveEngineFuture<'a> =
    Pin<Box<dyn Future<Output = Result<EngineHandle, String>> + Send + 'a>>;

/// The future returned by [`ServerControl::dispatch`].
///
/// `Ok(value)` is the operation's JSON result (surfaced as the JSON-RPC
/// `result`); `Err(msg)` is a human-readable failure (surfaced as a JSON-RPC
/// error). Boxed + `Send` so the trait stays object-safe and the future can
/// cross the async accept loop's task boundary.
pub type ControlFuture<'a> =
    Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send + 'a>>;

/// Server-level control-plane backend for the ACP `_akh/{workspace,daemon,config}/*`
/// methods.
///
/// Implemented in akhomed over its `ServerState`; held by [`super::AkhAcpAgent`]
/// as an `Option<Arc<dyn ServerControl>>` (see [`AkhAcpAgent::with_control`]).
/// When present, the dispatch router forwards every method in [`CONTROL_METHODS`]
/// here; when absent, those methods return a clean "no backend" error rather
/// than being mistaken for an unknown data-plane op.
///
/// [`AkhAcpAgent::with_control`]: super::AkhAcpAgent::with_control
pub trait ServerControl: Send + Sync {
    /// Dispatch one control-plane operation.
    ///
    /// `method` is the full wire method (e.g. `"_akh/workspace/list"`);
    /// `params` is its JSON-RPC params object. Implementors should match on the
    /// `_akh/`-stripped tail and return a structured JSON result.
    fn dispatch<'a>(
        &'a self,
        method: &'a str,
        params: &'a serde_json::Value,
    ) -> ControlFuture<'a>;

    /// Resolve a workspace's runtime handles by name, lazily loading it.
    ///
    /// Lets the ACP server serve any configured workspace per request (the
    /// `-w` data-plane + chat routing) and avoids holding a permanent engine
    /// reference (so a same-host `sleep`/`wake` can fully release the store).
    /// Returns `Err` when the workspace cannot be opened.
    ///
    /// Default: unsupported — in-process embeds/tests without a multi-workspace
    /// backend keep their single held engine/agent instead.
    fn resolve<'a>(&'a self, _workspace: &'a str) -> ResolveFuture<'a> {
        Box::pin(async move {
            Err("workspace resolution not supported by this backend".to_string())
        })
    }

    /// Resolve only the engine for a workspace — never constructs a full
    /// agent (no WAL, no snapshot ticker, no NLU pipeline). Used by
    /// read-only ACP verbs to avoid the split-brain vector from the
    /// 2026-08-08 incident (ADR-049).
    ///
    /// Default: unsupported — in-process embeds/tests without a
    /// multi-workspace backend keep their single held engine instead.
    fn resolve_engine_only<'a>(
        &'a self,
        _workspace: &'a str,
    ) -> ResolveEngineFuture<'a> {
        Box::pin(async move {
            Err("engine-only resolution not supported by this backend".to_string())
        })
    }
}

/// The `_akh/*` methods routed to a [`ServerControl`] backend rather than the
/// per-workspace data-plane. The single source of truth for both the dispatch
/// router ([`super::build_connection`]) and the `_akh/meta/capabilities`
/// advertisement.
///
/// Note `_akh/workspace/status` is intentionally **not** here — it is a
/// data-plane op (the *current* workspace's stats), distinct from the
/// control-plane `_akh/workspace/{list,create,delete}` which span the server.
pub const CONTROL_METHODS: &[&str] = &[
    "_akh/workspace/list",
    "_akh/workspace/create",
    "_akh/workspace/delete",
    "_akh/workspace/info",
    "_akh/daemon/start",
    "_akh/daemon/stop",
    "_akh/daemon/status",
    "_akh/daemon/akhs",
    "_akh/config/get",
    "_akh/config/set",
    "_akh/model/list",
    "_akh/model/set",
    "_akh/akh/snapshot",
    "_akh/akh/wake",
    "_akh/akh/sleep",
    "_akh/akh/relocate",
    "_akh/workspace/join",
    "_akh/workspace/leave",
    "_akh/workspace/list-members",
    "_akh/projection/refresh",
    "_akh/migrate",
    // ADR 081 decision 9 — the Postgres schema-pack registry.
    "_akh/schema/admit",
    "_akh/schema/list",
    "_akh/schema/install",
    "_akh/schema/digest",
    // ADR 081 Slice 5 — promoting a proposal touches the Postgres registry
    // (`_akh/schema/proposals` and `_akh/schema/reject` are engine-only
    // data-plane ops and stay out of this list — see `dispatch_akh`).
    "_akh/schema/promote",
    // Goal CRUD over the ACP control plane (PR #286 follow-up).
    "_akh/goal/list",
    "_akh/goal/create",
    "_akh/goal/update",
    "_akh/goal/cancel",
    "_akh/goal/prioritize",
];

/// Whether `method` is a control-plane method (a member of [`CONTROL_METHODS`]).
#[must_use]
pub fn is_control_method(method: &str) -> bool {
    CONTROL_METHODS.contains(&method)
}
