//! ACP (Agent Client Protocol) transport — P2 server.
//!
//! akh-medu serves as an **ACP agent over Unix domain sockets and TCP**,
//! wrapping the existing [`ChatProcessor`]. P2 increment 1 (this module) lands
//! the transport + protocol surface in the library:
//!
//! - `initialize` → advertise capabilities, echo the protocol version, report
//!   agent name/version.
//! - `session/new` → mint a fresh session id (a handle for one workspace).
//! - `session/prompt` → run [`ChatProcessor::process_input`] under the ADR-027
//!   discipline (lock the agent inside [`tokio::task::spawn_blocking`], never
//!   across an `.await`), then stream each structured [`AkhMessage`] back as a
//!   `session/update` via [`AcpSink`] (visible content → `AgentMessageChunk`,
//!   introspective content → `AgentThoughtChunk`), and respond
//!   [`StopReason::EndTurn`].
//! - Everything else falls through to the crate's default handler.
//!
//! # Transport
//!
//! `agent-client-protocol` is `futures`-based: its [`ByteStreams`] transport
//! wants `futures::io::{AsyncRead, AsyncWrite}`. Tokio's [`UnixStream`] /
//! [`TcpStream`] halves are bridged into those traits with [`tokio_util::compat`],
//! so the same dispatch logic serves both. [`serve_unix_loop`] /
//! [`serve_tcp_loop`] are the production accept loops (task-per-connection, with
//! a watch-channel shutdown).
//!
//! # Deferred to later P2 increments
//!
//! Multi-workspace resolution (a `WorkspaceProvider` over akhomed's
//! `ServerState`), the `_akh/*` extension-method data-plane, the discovery
//! record, and the daemon wiring (Unix + TCP listeners alongside REST during
//! migration). Note: `AgentCapabilities` has no `_meta` field in the crate, so
//! data-plane capability groups will be advertised via a custom
//! `_akh/meta/capabilities` ext method rather than `_meta`.
//!
//! [`ChatProcessor`]: crate::chat::ChatProcessor
//! [`AkhMessage`]: crate::message::AkhMessage
//! [`ByteStreams`]: agent_client_protocol::ByteStreams
//! [`UnixStream`]: tokio::net::UnixStream
//! [`TcpStream`]: tokio::net::TcpStream

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentCapabilities, ClientCapabilities, ContentBlock, ContentChunk, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    ProtocolVersion, SessionId, SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::util::{json_cast, MatchDispatch};
use agent_client_protocol::{
    ActiveSession, Agent as AcpRole, ByteStreams, Client, ConnectTo, ConnectionTo, Dispatch,
    JsonRpcMessage, JsonRpcRequest, JsonRpcResponse, Responder, UntypedMessage,
};
use miette::Diagnostic;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tokio::sync::watch;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::agent::Agent;
use crate::agent::{GoalStatus, PlanStatus, StepStatus};
use crate::chat::ChatProcessor;
use crate::engine::Engine;
use crate::message::{AkhMessage, MessageSink};

pub mod control;
pub use control::{
    is_control_method, ControlFuture, ResolveFuture, ServerControl, WorkspaceHandles,
    CONTROL_METHODS,
};

// ──────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────

/// Errors raised while serving the ACP transport.
///
/// The crate's own protocol errors ([`agent_client_protocol::Error`]) are kept
/// out of this enum and surfaced as their message text, since they are not
/// `miette::Diagnostic` and carry their own JSON-RPC error codes.
#[derive(Debug, Error, Diagnostic)]
pub enum AcpError {
    /// The Unix socket could not be bound.
    #[error("failed to bind ACP Unix socket at {path}: {source}")]
    #[diagnostic(
        code(akh::acp::bind),
        help(
            "Ensure the parent directory exists, you have write permission, and \
             no stale socket file is in the way (remove it before re-binding)."
        )
    )]
    Bind {
        /// The socket path that could not be bound.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Accepting an inbound connection failed.
    #[error("failed to accept ACP connection on {path}: {source}")]
    #[diagnostic(
        code(akh::acp::accept),
        help("The listener is bound but the accept() syscall failed. Check the socket is still valid.")
    )]
    Accept {
        /// The socket path being served.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The ACP protocol loop returned an error.
    #[error("ACP protocol error: {message}")]
    #[diagnostic(
        code(akh::acp::protocol),
        help(
            "The peer sent a malformed message or the connection dropped. \
             Inspect the message text and verify the client speaks ACP v1."
        )
    )]
    Protocol {
        /// Human-readable description from the protocol layer.
        message: String,
    },

    /// A client failed to connect to a daemon's ACP endpoint.
    #[error("failed to connect to ACP daemon at {target}: {source}")]
    #[diagnostic(
        code(akh::acp::connect),
        help(
            "Ensure akhomed is running and the ACP socket/TCP endpoint advertised \
             in the discovery record is correct and reachable."
        )
    )]
    Connect {
        /// The socket path or TCP address that could not be reached.
        target: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// The blocking task that ran the synchronous engine panicked or was cancelled.
    #[error("ACP prompt worker failed: {message}")]
    #[diagnostic(
        code(akh::acp::worker),
        help("spawn_blocking returned a JoinError — the synchronous chat turn panicked. Check the logs.")
    )]
    Worker {
        /// Description of the join failure.
        message: String,
    },
}

impl From<agent_client_protocol::Error> for AcpError {
    fn from(err: agent_client_protocol::Error) -> Self {
        AcpError::Protocol {
            message: err.to_string(),
        }
    }
}

/// Convenience result alias for ACP transport operations.
pub type AcpResult<T> = Result<T, AcpError>;

// ──────────────────────────────────────────────────────────────────────────
// Workspace — the shared, in-process state served over ACP
// ──────────────────────────────────────────────────────────────────────────

/// One in-process ACP workspace: the engine, the shared agent, and the chat
/// processor that turns text prompts into [`AkhMessage`]s.
///
/// Per ADR-027, the [`Agent`] is shared as `Arc<std::sync::Mutex<Agent>>`. The
/// [`ChatProcessor`] requires `&mut self` for [`ChatProcessor::process_input`],
/// so it is likewise wrapped in a `Mutex`. Both guards are acquired and dropped
/// **inside** a [`tokio::task::spawn_blocking`] closure and never held across an
/// `.await` — see [`AkhAcpAgent::run_prompt`].
#[derive(Clone)]
pub struct AkhAcpAgent {
    /// Held engine/agent/processor for the single in-process workspace (tests,
    /// embedded use with no [`ServerControl`]). `None` in *provider mode* (the
    /// daemon): every request resolves its workspace's handles on demand via
    /// [`ServerControl::resolve`], so the server serves any akh per request and
    /// pins no engine (a permanent hold would block same-host sleep/wake).
    engine: Option<Arc<Engine>>,
    agent: Option<Arc<Mutex<Agent>>>,
    processor: Option<Arc<Mutex<ChatProcessor>>>,
    /// Default workspace used when a request names none (the daemon's bound
    /// `[acp].workspace`; the held workspace's name in embedded mode).
    default_ws: String,
    /// Chat session → workspace map. `session/new` records the workspace (from
    /// the request `cwd` sentinel) so each `session/prompt` routes to the right
    /// akh — the data-plane `-w` for chat.
    sessions: Arc<Mutex<std::collections::HashMap<String, String>>>,
    /// Agent name advertised during `initialize`.
    name: String,
    /// Client filesystem/terminal capabilities captured at `initialize`.
    /// Consulted before issuing client-side `fs/*` or `terminal/*` requests
    /// (the Remote-FS path); `None` until the client initializes.
    client_caps: Arc<Mutex<Option<ClientCapabilities>>>,
    /// Server-level control-plane backend (akhomed's `ServerState`). Handles the
    /// `_akh/{workspace,daemon,config}/*` methods that span the whole daemon
    /// rather than one workspace. `None` for in-process single-workspace setups
    /// (tests, embedded use) — those methods then return a clean "no backend"
    /// error. See [`ServerControl`].
    control: Option<Arc<dyn ServerControl>>,
}

impl AkhAcpAgent {
    /// Build a new ACP agent wrapper for a single workspace.
    ///
    /// The caller owns construction of the [`Engine`], [`Agent`], and
    /// [`ChatProcessor`] so this module stays decoupled from how a workspace is
    /// seeded (tests seed in-memory; a real daemon would open a persistent
    /// store).
    pub fn new(
        engine: Arc<Engine>,
        agent: Arc<Mutex<Agent>>,
        processor: ChatProcessor,
    ) -> Self {
        Self {
            engine: Some(engine),
            agent: Some(agent),
            processor: Some(Arc::new(Mutex::new(processor))),
            default_ws: "akh-medu".to_string(),
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            name: "akh-medu".to_string(),
            client_caps: Arc::new(Mutex::new(None)),
            control: None,
        }
    }

    /// Build an ACP agent in **provider mode**: it holds no workspace engine of
    /// its own; every request resolves its workspace's handles on demand from the
    /// [`ServerControl`] backend. `default_ws` is used when a request names none.
    /// This is what the daemon uses so one server serves all configured akhs
    /// (the `-w` data-plane + chat routing) and pins no engine.
    pub fn with_provider(control: Arc<dyn ServerControl>, default_ws: impl Into<String>) -> Self {
        let default_ws = default_ws.into();
        Self {
            engine: None,
            agent: None,
            processor: None,
            name: default_ws.clone(),
            default_ws,
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            client_caps: Arc::new(Mutex::new(None)),
            control: Some(control),
        }
    }

    /// Override the agent name advertised during `initialize`. In held
    /// (single-workspace) mode this is also the default workspace, so set both.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.default_ws = name.clone();
        self.name = name;
        self
    }

    /// Resolve a workspace's runtime handles for one request. In provider mode
    /// (daemon) this loads them on demand via [`ServerControl::resolve`]; in
    /// held mode (embedded/tests) it returns the single held workspace.
    async fn resolve_handles(
        &self,
        workspace: &str,
    ) -> Result<(Arc<Engine>, Arc<Mutex<Agent>>, Arc<Mutex<ChatProcessor>>), String> {
        if let Some(ctrl) = &self.control {
            match ctrl.resolve(workspace).await {
                Ok(h) => return Ok((h.engine, h.agent, h.processor)),
                // A control wired only for the control-plane (e.g. tests, or an
                // embed that keeps a single held engine) returns the default
                // "not supported" — fall back to the held workspace if there is
                // one; only error when there's nothing to fall back to.
                Err(e) if self.engine.is_none() => return Err(e),
                Err(_) => {}
            }
        }
        match (&self.engine, &self.agent, &self.processor) {
            (Some(e), Some(a), Some(p)) => Ok((Arc::clone(e), Arc::clone(a), Arc::clone(p))),
            _ => Err("no workspace handles available (no engine held and no control backend)".into()),
        }
    }

    /// The workspace a chat session is bound to (recorded at `session/new` from
    /// the request `cwd` sentinel), or the default when unrecorded.
    fn session_workspace(&self, session_id: &str) -> String {
        self.sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| self.default_ws.clone())
    }

    /// Attach a server-level control-plane backend.
    ///
    /// Wires the `_akh/{workspace,daemon,config}/*` methods to akhomed's
    /// `ServerState` (or any [`ServerControl`] impl). Without it, those methods
    /// return a "no backend wired" error and `_akh/meta/capabilities` omits the
    /// control groups.
    #[must_use]
    pub fn with_control(mut self, control: Arc<dyn ServerControl>) -> Self {
        self.control = Some(control);
        self
    }

    /// Run a single prompt turn under the ADR-027 lock discipline, **streaming**
    /// its narration to `sink` as the turn unfolds.
    ///
    /// The turn runs inside [`spawn_blocking`]: the agent + processor `Mutex`es
    /// are acquired and released entirely inside the closure, and each
    /// [`AkhMessage`] is handed to `sink` (a non-blocking channel send) the
    /// instant it is produced — answer text, reasoning, and tool steps stream
    /// out live rather than being batched at turn-end. Because the send is
    /// synchronous (never an `.await`), no `std::sync::Mutex` guard crosses an
    /// await, preserving ADR-027.
    ///
    /// Returns the blocking task's [`JoinHandle`]; the caller drains the paired
    /// channel into a [`AcpSink`] concurrently, then awaits the handle. The
    /// `sink` is dropped when the closure ends, closing the channel so the drain
    /// loop terminates.
    ///
    /// [`spawn_blocking`]: tokio::task::spawn_blocking
    /// [`JoinHandle`]: tokio::task::JoinHandle
    fn run_turn(
        engine: Arc<Engine>,
        agent: Arc<Mutex<Agent>>,
        processor: Arc<Mutex<ChatProcessor>>,
        agent_id: String,
        text: String,
        asker: Option<Arc<dyn crate::agent::permission::PermissionAsker>>,
        sink: ChannelSink,
    ) -> tokio::task::JoinHandle<()> {
        tokio::task::spawn_blocking(move || {
            // ── ADR-027: lock, process (emitting via the channel sink), drop —
            // all synchronous, no await while a guard is held. ──
            let mut agent = agent.lock().unwrap_or_else(|e| e.into_inner());
            let mut processor = processor.lock().unwrap_or_else(|e| e.into_inner());
            // Install the per-turn permission approver (if any) so tool calls
            // made during the turn — by the LLM loop or the symbolic pipeline —
            // can prompt the client; removed before the closure returns so its
            // channel closes and the approver task ends.
            if let Some(asker) = asker {
                agent.set_permission_asker(asker);
            }
            // Prefer the LLM agentic loop (the akh's pinned Ollama model + tools),
            // which streams directly into `sink`; fall back to the symbolic
            // ChatProcessor when no model backend is reachable, emitting its
            // (batched) output through the same sink so the wire shape matches.
            if agent
                .run_agentic_turn(&engine, &agent_id, &text, &sink)
                .is_none()
            {
                let out = processor.process_input(&text, &mut agent, &engine);
                sink.emit_batch(&out);
            }
            agent.clear_permission_asker();
            // Guards + sink drop here; the dropped sink closes the channel.
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────
// ChannelSink — bridge the synchronous turn to the async session/update drain
// ──────────────────────────────────────────────────────────────────────────

/// A [`MessageSink`] that forwards each [`AkhMessage`] over an unbounded channel
/// to an async drain task.
///
/// The chat turn runs inside [`spawn_blocking`](tokio::task::spawn_blocking) and
/// must not touch the ACP connection directly (that lives on the async side), so
/// it streams its messages through this sink; the prompt handler receives them
/// and re-emits each as a `session/update` via [`AcpSink`]. The send is
/// non-blocking, so emitting while the agent `Mutex` is held is ADR-027-safe.
struct ChannelSink {
    tx: tokio::sync::mpsc::UnboundedSender<AkhMessage>,
}

impl MessageSink for ChannelSink {
    fn emit(&self, msg: &AkhMessage) {
        // A closed receiver (client gone) just drops the message — best-effort,
        // matching the `MessageSink::emit` infallible contract.
        let _ = self.tx.send(msg.clone());
    }
}

// ──────────────────────────────────────────────────────────────────────────
// AcpSink — stream AkhMessages to the client as session/update notifications
// ──────────────────────────────────────────────────────────────────────────

/// A [`MessageSink`] that streams each [`AkhMessage`] to the connected ACP
/// client as a `session/update` notification on a live [`ConnectionTo`].
///
/// Construct one per prompt turn from the request's [`ConnectionTo<Client>`] and
/// [`SessionId`], then [`MessageSink::emit_batch`] the turn's responses. Sends
/// are best-effort: a failure (client gone mid-turn) is logged, not propagated,
/// since [`MessageSink::emit`] is infallible by contract.
///
/// [`ConnectionTo`]: agent_client_protocol::ConnectionTo
#[derive(Clone)]
pub struct AcpSink {
    conn: ConnectionTo<Client>,
    session_id: SessionId,
}

impl AcpSink {
    /// Build a sink bound to one client connection + session.
    pub fn new(conn: ConnectionTo<Client>, session_id: SessionId) -> Self {
        Self { conn, session_id }
    }

    /// Send one mapped `session/update`, logging (not raising) on failure.
    fn send(&self, update: SessionUpdate) {
        if let Err(e) = self
            .conn
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
        {
            tracing::warn!("ACP session/update send failed (client gone?): {e}");
        }
    }
}

impl MessageSink for AcpSink {
    fn emit(&self, msg: &AkhMessage) {
        self.send(to_session_update(msg));
    }
}

/// Map one [`AkhMessage`] to the ACP [`SessionUpdate`] variant that best fits it.
///
/// Visible answer content (facts, narrative, system notes, tool output, prompts,
/// errors) becomes an `AgentMessageChunk` so a client's `read_to_string()`
/// collects it; introspective content (reasoning steps, knowledge gaps, goal
/// progress, audit entries) becomes an `AgentThoughtChunk` so it is surfaced as
/// agent "thinking" rather than the answer. Structured fields beyond the text
/// (confidence, provenance, tool-call lifecycle) are folded into the rendered
/// text for now; richer `ToolCallUpdate`/`Plan` mappings are a later increment.
fn to_session_update(msg: &AkhMessage) -> SessionUpdate {
    let chunk = |text: String| ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
    match msg {
        AkhMessage::Fact { text, confidence, .. } => {
            let body = match confidence {
                Some(c) => format!("{text} (confidence {c:.2})"),
                None => text.clone(),
            };
            SessionUpdate::AgentMessageChunk(chunk(body))
        }
        AkhMessage::Narrative { text, .. } => SessionUpdate::AgentMessageChunk(chunk(text.clone())),
        AkhMessage::System { text } => SessionUpdate::AgentMessageChunk(chunk(text.clone())),
        AkhMessage::ToolResult { tool, success, output } => {
            // Tool calls + their results are turn "steps", not the answer — map
            // them to thought chunks so the client folds them into the
            // collapsible thinking section rather than the response body.
            // Trailing newline keeps each tool step on its own line when the
            // client coalesces a turn's thought chunks into one block.
            let status = if *success { "ok" } else { "failed" };
            SessionUpdate::AgentThoughtChunk(chunk(format!("[{tool}:{status}] {output}\n")))
        }
        AkhMessage::Prompt { question } => SessionUpdate::AgentMessageChunk(chunk(question.clone())),
        AkhMessage::Error { code, message, .. } => {
            SessionUpdate::AgentMessageChunk(chunk(format!("error[{code}]: {message}")))
        }
        AkhMessage::Reasoning { step, expression } => {
            let body = match expression {
                Some(expr) => format!("{step} :: {expr}"),
                None => step.clone(),
            };
            SessionUpdate::AgentThoughtChunk(chunk(body))
        }
        AkhMessage::Gap { entity, description } => {
            SessionUpdate::AgentThoughtChunk(chunk(format!("gap: {entity} — {description}")))
        }
        AkhMessage::GoalProgress { goal, status, detail } => {
            let body = match detail {
                Some(d) => format!("goal {goal} [{status}]: {d}"),
                None => format!("goal {goal} [{status}]"),
            };
            SessionUpdate::AgentThoughtChunk(chunk(body))
        }
        AkhMessage::AuditLog { kind, summary, .. } => {
            SessionUpdate::AgentThoughtChunk(chunk(format!("audit[{kind}]: {summary}")))
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Serving the agent over a transport
// ──────────────────────────────────────────────────────────────────────────

/// Build the crate's agent-role connection builder, wiring the three ACP
/// methods this spike supports.
///
/// Returns a [`Builder`](agent_client_protocol::Builder) that is itself a
/// `ConnectTo<Client>` — i.e. it serves the [`Client`] counterpart over whatever
/// transport it is connected to.
// ──────────────────────────────────────────────────────────────────────────
// `_akh/*` data-plane — custom JSON-RPC extension methods
// ──────────────────────────────────────────────────────────────────────────
//
// ACP routes any method starting with `_` to an extension channel. Because the
// method name is dynamic, it does not fit the crate's fixed-method typed-request
// model, so the data-plane rides a single custom request type
// ([`AkhExtRequest`]) whose `matches_method` claims the `_akh/` prefix — the
// same pattern as the crate's `EchoRequestResponse` test. The agent dispatches
// these via `on_receive_dispatch` (the catch-all after the typed
// initialize/session handlers); the client sends them with [`ext_request_unix`]
// / [`ext_request_tcp`]. New data-plane operations are added by extending
// [`dispatch_akh`] (server) — the transport stays fixed.

/// A `_akh/<group>/<op>` data-plane request: carries the full wire method and
/// its JSON params.
#[derive(Debug, Clone)]
struct AkhExtRequest {
    method: String,
    params: serde_json::Value,
}

impl JsonRpcMessage for AkhExtRequest {
    fn matches_method(method: &str) -> bool {
        method.starts_with("_akh/")
    }
    fn method(&self) -> &str {
        &self.method
    }
    fn to_untyped_message(&self) -> Result<UntypedMessage, agent_client_protocol::Error> {
        UntypedMessage::new(&self.method, &self.params)
    }
    fn parse_message(
        method: &str,
        params: &impl serde::Serialize,
    ) -> Result<Self, agent_client_protocol::Error> {
        if !Self::matches_method(method) {
            return Err(agent_client_protocol::Error::method_not_found());
        }
        Ok(Self {
            method: method.to_string(),
            params: json_cast(params)?,
        })
    }
}

/// The JSON result of an [`AkhExtRequest`].
#[derive(Debug, Clone)]
struct AkhExtResponse(serde_json::Value);

impl JsonRpcResponse for AkhExtResponse {
    fn into_json(self, _method: &str) -> Result<serde_json::Value, agent_client_protocol::Error> {
        Ok(self.0)
    }
    fn from_value(
        _method: &str,
        value: serde_json::Value,
    ) -> Result<Self, agent_client_protocol::Error> {
        Ok(Self(value))
    }
}

impl JsonRpcRequest for AkhExtRequest {
    type Response = AkhExtResponse;
}

/// Dispatch one `_akh/*` data-plane operation against the engine.
///
/// `method` is the full wire method (e.g. `"_akh/health"`). Read-only engine
/// access needs no agent lock (engine methods take `&self`); operations that
/// mutate the agent would route through `spawn_blocking` per ADR-027 (none yet).
/// Returns the JSON result, or an error string surfaced as a JSON-RPC error.
///
/// NOTE: agent-mutating data-plane methods (e.g. `_akh/permission/mode`) are
/// NOT handled here — they take the agent lock and run in `spawn_blocking`
/// (ADR-027) from the dispatch handler. This function is read-only engine work.
///
/// `has_control` reflects whether a [`ServerControl`] backend is wired (it owns
/// the control-plane methods, dispatched *before* this function). It only
/// affects `_akh/meta/capabilities`, which advertises the control groups +
/// methods so a client never offers an op the server can't honour.
fn dispatch_akh(
    engine: &Engine,
    workspace: &str,
    has_control: bool,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let op = method.strip_prefix("_akh/").unwrap_or(method);
    match op {
        // ── meta group ──────────────────────────────────────────────────
        "health" => Ok(serde_json::json!({
            "status": "ok",
            "workspace": workspace,
            "symbols": engine.all_symbols().len(),
            "triples": engine.all_triples().len(),
        })),
        "meta/capabilities" => {
            // Per-workspace data-plane (always present).
            let mut groups = vec![
                "meta", "workspace", "symbols", "triples", "query", "memory", "infer",
                "permission", "goals", "tools", "skills", "projects", "status", "usage",
                "reasoning", "evidence",
            ];
            let mut methods = vec![
                "_akh/health",
                "_akh/meta/capabilities",
                "_akh/workspace/status",
                "_akh/symbols/list",
                "_akh/triples/from",
                "_akh/symbols/create",
                "_akh/triples/add",
                "_akh/triples/remove",
                "_akh/query/sparql",
                "_akh/memory/recall",
                "_akh/infer/activate",
                "_akh/permission/mode",
                "_akh/goals",
                "_akh/tools",
                "_akh/skills",
                "_akh/projects",
                "_akh/status",
                "_akh/usage",
                "_akh/reasoning",
                "_akh/evidence",
            ];
            // Server-level control-plane (only when a backend is wired).
            if has_control {
                groups.push("daemon");
                groups.push("config");
                // Phase 38 operator groups: akh lifecycle/relocation, AGE
                // projection refresh (stub), and the migration runner.
                groups.push("akh");
                groups.push("projection");
                groups.push("migrate");
                methods.extend(control::CONTROL_METHODS.iter().copied());
            }
            Ok(serde_json::json!({
                "protocol": "acp",
                "version": crate::client::ACP_PROTOCOL_VERSION,
                "control_plane": has_control,
                "groups": groups,
                "methods": methods,
            }))
        }

        // ── workspace group ─────────────────────────────────────────────
        "workspace/status" => Ok(serde_json::json!({
            "workspace": workspace,
            "symbols": engine.all_symbols().len(),
            "triples": engine.all_triples().len(),
        })),

        // ── skills group (engine-derivable; no agent lock needed, so it is
        //    served here rather than in `dispatch_akh_agent`) ──────────────
        "skills" => Ok(serde_json::json!({
            "workspace": workspace,
            "skills": skills_json(engine),
        })),

        // ── symbols group (read-only) ───────────────────────────────────
        "symbols/list" => {
            let limit = params
                .get("limit")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(100) as usize;
            let all = engine.all_symbols();
            let total = all.len();
            let symbols: Vec<_> = all.into_iter().take(limit).collect();
            serde_json::to_value(serde_json::json!({
                "count": total,
                "returned": symbols.len(),
                "symbols": symbols,
            }))
            .map_err(|e| e.to_string())
        }

        // ── triples group (read-only) ───────────────────────────────────
        "triples/from" => {
            let sym = params
                .get("symbol")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: symbol".to_string())?;
            let id = engine
                .resolve_symbol(sym)
                .map_err(|e| format!("unknown symbol '{sym}': {e}"))?;
            let triples: Vec<_> = engine
                .triples_from(id)
                .into_iter()
                .map(|t| {
                    serde_json::json!({
                        "predicate": engine.resolve_label(t.predicate),
                        "object": engine.resolve_label(t.object),
                        "confidence": t.confidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({
                "symbol": engine.resolve_label(id),
                "triples": triples,
            }))
        }

        // ── write group (engine KG mutations; &self + internal sync, no agent
        //    lock; persisted immediately). Unguarded for now — a permission
        //    layer (request_permission) lands in P4. ─────────────────────────
        "symbols/create" => {
            let label = params
                .get("label")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: label".to_string())?;
            let kind = params
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("entity");
            let id = match kind {
                "relation" => engine.resolve_or_create_relation(label),
                _ => engine.resolve_or_create_entity(label),
            }
            .map_err(|e| format!("create failed: {e}"))?;
            engine.persist().map_err(|e| format!("persist failed: {e}"))?;
            Ok(serde_json::json!({
                "id": id.get(),
                "label": engine.resolve_label(id),
                "kind": kind,
            }))
        }

        "triples/add" => {
            let subject = params
                .get("subject")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: subject".to_string())?;
            let predicate = params
                .get("predicate")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: predicate".to_string())?;
            let object = params
                .get("object")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: object".to_string())?;
            let confidence = params
                .get("confidence")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let (symbols_added, triples_added) = engine
                .ingest_label_triples(&[(
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                    confidence,
                )])
                .map_err(|e| format!("ingest failed: {e}"))?;
            engine.persist().map_err(|e| format!("persist failed: {e}"))?;
            Ok(serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "confidence": confidence,
                "symbols_added": symbols_added,
                "triples_added": triples_added,
            }))
        }

        "triples/remove" => {
            let subject = params
                .get("subject")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: subject".to_string())?;
            let predicate = params
                .get("predicate")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: predicate".to_string())?;
            let object = params
                .get("object")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: object".to_string())?;
            // Resolve labels to *existing* symbols; an unknown label means there
            // is nothing to remove (report removed=false rather than erroring, so
            // cleanup scripts are idempotent).
            let resolve = |label: &str| engine.resolve_symbol(label).ok();
            let (Some(s), Some(p), Some(o)) =
                (resolve(subject), resolve(predicate), resolve(object))
            else {
                return Ok(serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "removed": false,
                    "retracted_count": 0,
                    "note": "one or more labels not present; nothing to remove",
                }));
            };
            // Pre-check existence: RetractionResult only reflects TMS cascade, so
            // a plain removal with no cascade would otherwise look like a no-op.
            let existed = engine
                .triples_from(s)
                .iter()
                .any(|t| t.predicate == p && t.object == o);
            let result = engine
                .remove_triple(s, p, o)
                .map_err(|e| format!("remove failed: {e}"))?;
            engine.persist().map_err(|e| format!("persist failed: {e}"))?;
            Ok(serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "removed": existed,
                "cascade_retracted": result.retracted.len(),
                "cascade_depth": result.cascade_depth,
            }))
        }

        // ── query group (read-only SPARQL SELECT; requires a persistent
        //    engine — the daemon's workspaces are persistent). ──────────────
        "query/sparql" => {
            let query = params
                .get("query")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: query".to_string())?;
            let rows = engine
                .sparql_query(query)
                .map_err(|e| format!("sparql query failed: {e}"))?;
            // Each row is a solution: a list of (variable, value) bindings.
            let solutions: Vec<serde_json::Value> = rows
                .into_iter()
                .map(|row| {
                    let obj: serde_json::Map<String, serde_json::Value> = row
                        .into_iter()
                        .map(|(var, val)| (var, serde_json::Value::String(val)))
                        .collect();
                    serde_json::Value::Object(obj)
                })
                .collect();
            Ok(serde_json::json!({ "count": solutions.len(), "solutions": solutions }))
        }

        // ── memory group (read-only episodic recall). `AgentPredicates::init`
        //    is resolve-or-create on `&Engine` (idempotent — the agent already
        //    created these relations at init), so no agent lock is needed. This
        //    mirrors the `memory_recall` tool the agent uses on itself: resolve
        //    the requested query symbols and recall by them, or fall back to a
        //    pure recency scan when none resolve.
        "memory/recall" => {
            let top_k = params
                .get("top_k")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(5) as usize;

            // Accept a comma-separated `query` string and/or a `symbols` array.
            let mut requested: Vec<String> = Vec::new();
            if let Some(s) = params.get("query").and_then(serde_json::Value::as_str) {
                requested.extend(
                    s.split(',')
                        .map(|p| p.trim().to_string())
                        .filter(|p| !p.is_empty()),
                );
            }
            if let Some(arr) = params.get("symbols").and_then(serde_json::Value::as_array) {
                requested.extend(
                    arr.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string),
                );
            }

            let predicates = crate::agent::AgentPredicates::init(engine)
                .map_err(|e| format!("failed to resolve agent predicates: {e}"))?;

            // Resolve requested symbols; report unresolvable ones so the caller
            // can tell "no episodes" apart from "typo in the query".
            let mut query_ids = Vec::new();
            let mut unresolved = Vec::new();
            for label in &requested {
                match engine.resolve_symbol(label) {
                    Ok(id) => query_ids.push(id),
                    Err(_) => unresolved.push(label.clone()),
                }
            }

            let (episodes, mode) = if query_ids.is_empty() {
                let eps = crate::agent::memory::recall_recent_episodes(engine, &predicates, top_k)
                    .map_err(|e| format!("recall failed: {e}"))?;
                (eps, "recent")
            } else {
                let eps =
                    crate::agent::memory::recall_episodes(engine, &query_ids, &predicates, top_k, None)
                        .map_err(|e| format!("recall failed: {e}"))?;
                (eps, "matched")
            };

            let episodes_json: Vec<serde_json::Value> =
                episodes.iter().map(|ep| episode_to_json(engine, ep)).collect();

            Ok(serde_json::json!({
                "mode": mode,
                "count": episodes_json.len(),
                "unresolved": unresolved,
                "episodes": episodes_json,
            }))
        }

        // ── infer group (read-only spreading-activation inference). `engine.infer`
        //    takes `&self`; the optional provenance-ledger write is internally
        //    synchronized, so no agent lock is needed. Seeds are required —
        //    unlike recall there is no recency fallback, so an all-unresolvable
        //    seed set is a clean error rather than a silent empty result.
        "infer/activate" => {
            // Seeds: a comma-separated `seeds`/`query` string and/or a `seeds` array.
            let mut requested: Vec<String> = Vec::new();
            for key in ["seeds", "query"] {
                if let Some(s) = params.get(key).and_then(serde_json::Value::as_str) {
                    requested.extend(
                        s.split(',')
                            .map(|p| p.trim().to_string())
                            .filter(|p| !p.is_empty()),
                    );
                }
            }
            if let Some(arr) = params.get("seeds").and_then(serde_json::Value::as_array) {
                requested.extend(
                    arr.iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_string),
                );
            }

            let top_k = params
                .get("top_k")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(10) as usize;
            let max_depth = params
                .get("max_depth")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1) as usize;
            let min_confidence = params
                .get("min_confidence")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.1) as f32;

            let mut seed_ids = Vec::new();
            let mut unresolved = Vec::new();
            for label in &requested {
                match engine.resolve_symbol(label) {
                    Ok(id) => seed_ids.push(id),
                    Err(_) => unresolved.push(label.clone()),
                }
            }

            if seed_ids.is_empty() {
                Err(format!(
                    "infer/activate requires at least one resolvable seed symbol \
                     (none of {} requested resolved: {unresolved:?})",
                    requested.len()
                ))
            } else {
                let query = crate::infer::InferenceQuery {
                    seeds: seed_ids,
                    top_k,
                    max_depth,
                    min_confidence,
                    ..Default::default()
                };
                let result = engine
                    .infer(&query)
                    .map_err(|e| format!("inference failed: {e}"))?;

                let activations: Vec<serde_json::Value> = result
                    .activations
                    .iter()
                    .map(|(id, score)| {
                        serde_json::json!({
                            "symbol": engine.resolve_label(*id),
                            "score": score,
                        })
                    })
                    .collect();

                Ok(serde_json::json!({
                    "count": activations.len(),
                    "max_depth": max_depth,
                    "unresolved": unresolved,
                    "provenance_records": result.provenance.len(),
                    "activations": activations,
                }))
            }
        }

        other => Err(format!("unknown _akh data-plane method: _akh/{other}")),
    }
}

/// Agent-read `_akh/*` methods — `goals`, `tools`, `projects`, and the combined
/// `status` snapshot. Distinct from [`dispatch_akh`] because these read the
/// [`Agent`] (behind a `Mutex`), not just the [`Engine`]. (`_akh/skills` is
/// engine-derivable and stays in [`dispatch_akh`].)
///
/// To honour ADR-027 ("the turn is the priority") the agent is taken with
/// `try_lock`: when a turn holds the lock, the agent-derived fields are reported
/// empty with `stale: true`, and only the engine-derivable fields (skills in the
/// `status` snapshot) are filled — so a sidebar refresh never blocks behind, nor
/// stalls, a live turn (ADR 063 §2).
///
/// Returns `None` when `method` is not an agent-read method, so the caller falls
/// through to [`dispatch_akh`].
fn dispatch_akh_agent(
    engine: &Engine,
    agent: &Arc<Mutex<Agent>>,
    workspace: &str,
    method: &str,
    _params: &serde_json::Value,
) -> Option<Result<serde_json::Value, String>> {
    let op = method.strip_prefix("_akh/").unwrap_or(method);
    match op {
        "goals" | "tools" | "projects" | "status" | "usage" | "reasoning" | "evidence" => {}
        _ => return None,
    }

    // Non-blocking acquire: a held lock (turn in flight) ⇒ no agent snapshot.
    // Poisoned ⇒ recover the inner agent (a prior panic left it valid to read).
    let guard = match agent.try_lock() {
        Ok(g) => Some(g),
        Err(std::sync::TryLockError::WouldBlock) => None,
        Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
    };
    let stale = guard.is_none();
    let a = guard.as_deref();

    let value = match op {
        "goals" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "goals": a.map(goals_json).unwrap_or_default(),
        }),
        "tools" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "tools": a.map(tools_json).unwrap_or_default(),
        }),
        "projects" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "projects": a.map(projects_json).unwrap_or_default(),
        }),
        "usage" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "usage": a.and_then(usage_json),
        }),
        "reasoning" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "reasoning": a.map(reasoning_json).unwrap_or_default(),
        }),
        "evidence" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "evidence": a.map(|ag| evidence_json(engine, ag)),
        }),
        // The combined snapshot backing the whole TUI sidebar in one round-trip.
        "status" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "cycle": a.map(Agent::cycle_count),
            "goals": a.map(goals_json).unwrap_or_default(),
            "plan": a.and_then(active_plan_json),
            "tools": a.map(tools_json).unwrap_or_default(),
            "projects": a.map(projects_json).unwrap_or_default(),
            "skills": skills_json(engine), // engine-derivable: present even when stale
            "kg": { // engine-derivable KG size (cheap counts): present even when stale
                "symbols": engine.symbol_count(),
                "triples": engine.triple_count(),
            },
            "active_project": a.and_then(active_project_name),
            "reasoning": a.map(reasoning_json).unwrap_or_default(),
            "evidence": a.map(|ag| evidence_json(engine, ag)),
            "usage": a.and_then(usage_json), // None when no model manager attached
        }),
        _ => unreachable!("op was filtered to the agent-read set above"),
    };
    Some(Ok(value))
}

/// JSON projection of the agent's token usage, or `None` when no model manager
/// is attached (e.g. a daemon with no Ollama, or test agents). `context_window`
/// is lazily fetched once via Ollama `/api/show`; until it lands (or if the
/// model omits it) the client renders absolute token counts without a percentage.
fn usage_json(agent: &Agent) -> Option<serde_json::Value> {
    let u = agent.model_manager()?.usage();
    Some(serde_json::json!({
        "total_tokens": u.total_tokens(),
        "context_tokens": u.context_tokens,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "turns": u.turns,
        "context_window": u.context_window,
    }))
}

/// Name of the currently-active project (microtheory the agent is focused on),
/// resolved from the agenda's `active_project` id, or `None` when none is active.
fn active_project_name(agent: &Agent) -> Option<String> {
    let id = agent.agenda().active_project?;
    agent
        .projects()
        .iter()
        .find(|p| p.id == id)
        .map(|p| p.name.clone())
}

/// Compact JSON projection of the agent's recent reasoning trace — the most
/// recent working-memory entries (observations, decisions, inferences, tool
/// results), newest first, capped for the sidebar.
fn reasoning_json(agent: &Agent) -> Vec<serde_json::Value> {
    const MAX: usize = 6;
    agent
        .working_memory()
        .recent(MAX)
        .iter()
        .map(|e| {
            serde_json::json!({
                "kind": format!("{:?}", e.kind),
                "content": e.content,
            })
        })
        .collect()
}

/// JSON projection of the agent's evidence state: total tracked claims plus the
/// most uncertain ones (widest belief interval) with their Dempster-Shafer
/// verdict and belief/plausibility bounds.
fn evidence_json(engine: &Engine, agent: &Agent) -> serde_json::Value {
    let em = agent.evidence_manager();
    let uncertain: Vec<_> = em
        .most_uncertain_claims(3)
        .into_iter()
        .map(|c| {
            serde_json::json!({
                "claim": engine.resolve_label(c.claim_entity),
                "verdict": c.verdict.as_label(),
                "belief": c.interval.belief,
                "plausibility": c.interval.plausibility,
                "evidence_count": c.evidence_count,
            })
        })
        .collect();
    serde_json::json!({
        "claims": em.claim_count(),
        "uncertain": uncertain,
    })
}

/// Compact JSON projection of the agent's goals for the sidebar (capped at 50):
/// description, status (the serialized [`GoalStatus`]), priority, cycles worked.
fn goals_json(agent: &Agent) -> Vec<serde_json::Value> {
    const MAX: usize = 50;
    agent
        .goals()
        .iter()
        .take(MAX)
        .map(|g| {
            serde_json::json!({
                "description": g.description,
                "status": serde_json::to_value(&g.status).unwrap_or(serde_json::Value::Null),
                "priority": g.priority,
                "cycles_worked": g.cycles_worked,
            })
        })
        .collect()
}

/// Compact JSON projection of the registered tools (name + description).
fn tools_json(agent: &Agent) -> Vec<serde_json::Value> {
    agent
        .list_tools()
        .into_iter()
        .map(|t| serde_json::json!({ "name": t.name, "description": t.description }))
        .collect()
}

/// Compact JSON projection of the agent's projects (name + status + goal count).
fn projects_json(agent: &Agent) -> Vec<serde_json::Value> {
    agent
        .projects()
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "status": p.status.to_string(),
                "goals": p.goals.len(),
            })
        })
        .collect()
}

/// Engine-derivable JSON projection of loaded skills (id, name, state, counts).
fn skills_json(engine: &Engine) -> Vec<serde_json::Value> {
    engine
        .list_skills()
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "name": s.name,
                "state": format!("{:?}", s.state),
                "triples": s.triple_count,
                "rules": s.rule_count,
                "actions": s.action_count,
                "plans": s.plan_count,
            })
        })
        .collect()
}

/// The active goal's plan (if any) projected to JSON for the sidebar: the goal
/// description, plan strategy/status, and each step's tool + status + rationale.
fn active_plan_json(agent: &Agent) -> Option<serde_json::Value> {
    let active = agent.goals().iter().find(|g| g.status == GoalStatus::Active)?;
    let plan = agent.plan_for_goal(active.symbol_id)?;
    let steps: Vec<_> = plan
        .steps
        .iter()
        .map(|s| {
            serde_json::json!({
                "tool": s.tool_name,
                "status": step_status_str(&s.status),
                "rationale": s.rationale,
            })
        })
        .collect();
    Some(serde_json::json!({
        "goal": active.description,
        "strategy": plan.strategy,
        "status": plan_status_str(&plan.status),
        "steps": steps,
    }))
}

/// Short, stable label for a [`StepStatus`] (the data-carrying variants collapse
/// to their kind).
fn step_status_str(s: &StepStatus) -> &'static str {
    match s {
        StepStatus::Pending => "pending",
        StepStatus::Active => "active",
        StepStatus::Completed => "completed",
        StepStatus::Failed { .. } => "failed",
        StepStatus::Skipped => "skipped",
    }
}

/// Short, stable label for a [`PlanStatus`].
fn plan_status_str(s: &PlanStatus) -> &'static str {
    match s {
        PlanStatus::Active => "active",
        PlanStatus::Completed => "completed",
        PlanStatus::Failed { .. } => "failed",
        PlanStatus::Superseded => "superseded",
    }
}

/// Serialize an [`EpisodicEntry`] to JSON, resolving its symbol references to
/// human-readable labels. `EpisodicEntry` is not `Serialize` (it holds raw
/// [`SymbolId`]s), so the data-plane projects it to a label-anchored object.
fn episode_to_json(engine: &Engine, ep: &crate::agent::memory::EpisodicEntry) -> serde_json::Value {
    serde_json::json!({
        "symbol": engine.resolve_label(ep.symbol_id),
        "summary": ep.summary,
        "consolidation_reason": ep.consolidation_reason,
        "relevance_score": ep.relevance_score,
        "timestamp": ep.timestamp,
        "affective_valence": ep.affective_valence,
        "learnings": ep
            .learnings
            .iter()
            .map(|id| engine.resolve_label(*id))
            .collect::<Vec<_>>(),
        "tags": ep
            .tags
            .iter()
            .map(|id| engine.resolve_label(*id))
            .collect::<Vec<_>>(),
    })
}

// ──────────────────────────────────────────────────────────────────────────
// Permission bridge — sync tool path ↔ async session/request_permission
// ──────────────────────────────────────────────────────────────────────────

/// A pending approval, sent from the (synchronous, blocking) tool path to the
/// async approver task that owns the client connection.
struct PermissionAskRequest {
    tool_name: String,
    danger: String,
    input_summary: String,
    /// The approver replies here with allow (`true`) / reject (`false`).
    reply: tokio::sync::oneshot::Sender<bool>,
}

/// A [`PermissionAsker`](crate::agent::permission::PermissionAsker) that bridges
/// the blocking tool-execution path (inside `spawn_blocking`) to the async ACP
/// handler: `ask` sends the request over a channel and blocks on the reply.
/// Failing to reach the approver (channel closed) denies — fail closed.
struct AcpPermissionAsker {
    tx: tokio::sync::mpsc::Sender<PermissionAskRequest>,
}

impl crate::agent::permission::PermissionAsker for AcpPermissionAsker {
    fn ask(
        &self,
        manifest: &crate::agent::tool_manifest::ToolManifest,
        input_summary: &str,
    ) -> bool {
        let (reply, reply_rx) = tokio::sync::oneshot::channel();
        let req = PermissionAskRequest {
            tool_name: manifest.name.clone(),
            danger: manifest.danger.level.to_string(),
            input_summary: input_summary.to_string(),
            reply,
        };
        // We are inside spawn_blocking, so blocking_send / blocking_recv are
        // correct (never call these on an async runtime thread).
        if self.tx.blocking_send(req).is_err() {
            return false;
        }
        reply_rx.blocking_recv().unwrap_or(false)
    }
}

/// Service permission asks for one prompt turn: for each request, call the
/// client's `session/request_permission` and relay the decision. Ends when the
/// channel closes (the asker is dropped at end of turn).
async fn serve_permission_asks(
    mut rx: tokio::sync::mpsc::Receiver<PermissionAskRequest>,
    cx: ConnectionTo<Client>,
    session_id: SessionId,
) {
    use agent_client_protocol::schema::{
        PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
        ToolCallId, ToolCallUpdate, ToolCallUpdateFields,
    };
    while let Some(req) = rx.recv().await {
        let title = format!(
            "Run tool \"{}\" [{}] — {}",
            req.tool_name, req.danger, req.input_summary
        );
        let tool_call = ToolCallUpdate::new(
            ToolCallId::new(req.tool_name.clone()),
            ToolCallUpdateFields::new().title(title),
        );
        let options = vec![
            PermissionOption::new("allow", "Allow", PermissionOptionKind::AllowOnce),
            PermissionOption::new("reject", "Reject", PermissionOptionKind::RejectOnce),
        ];
        let request = RequestPermissionRequest::new(session_id.clone(), tool_call, options);
        let approved = match cx.send_request(request).block_task().await {
            Ok(resp) => match resp.outcome {
                RequestPermissionOutcome::Selected(sel) => sel.option_id.0.as_ref() == "allow",
                _ => false, // Cancelled / future variants → deny.
            },
            Err(_) => false, // client error → deny.
        };
        let _ = req.reply.send(approved);
    }
}

/// Parse a wire permission-mode string into a [`PermissionMode`].
fn parse_permission_mode(s: &str) -> Option<crate::agent::permission::PermissionMode> {
    use crate::agent::permission::PermissionMode;
    match s.trim().to_ascii_lowercase().as_str() {
        "bypass" => Some(PermissionMode::Bypass),
        "standard" => Some(PermissionMode::Standard),
        "accept_edits" | "acceptedits" | "accept-edits" => Some(PermissionMode::AcceptEdits),
        "plan" => Some(PermissionMode::Plan),
        _ => None,
    }
}

/// Render a [`PermissionMode`] as its wire string.
fn permission_mode_str(mode: crate::agent::permission::PermissionMode) -> &'static str {
    use crate::agent::permission::PermissionMode;
    match mode {
        PermissionMode::Bypass => "bypass",
        PermissionMode::Standard => "standard",
        PermissionMode::AcceptEdits => "accept_edits",
        PermissionMode::Plan => "plan",
    }
}

fn build_connection(
    me: AkhAcpAgent,
) -> impl agent_client_protocol::ConnectTo<Client> {
    AcpRole
        .builder()
        .name("akh-medu-acp")
        // ── initialize ──────────────────────────────────────────────────
        .on_receive_request_from(
            Client,
            {
                let me = me.clone();
                move |req: InitializeRequest, responder: Responder<InitializeResponse>, _cx| {
                    let me = me.clone();
                    async move {
                        // Capture the client's fs/terminal capabilities for the
                        // Remote-FS path (P4.5 (c)); gating fs/* + terminal/*
                        // requests on what the client advertised.
                        *me.client_caps.lock().unwrap_or_else(|e| e.into_inner()) =
                            Some(req.client_capabilities.clone());
                        // Echo the client's protocol version; advertise default
                        // (minimal) agent capabilities for the spike.
                        let resp = InitializeResponse::new(req.protocol_version)
                            .agent_capabilities(AgentCapabilities::default())
                            .agent_info(agent_client_protocol::schema::Implementation::new(
                                me.name.clone(),
                                env!("CARGO_PKG_VERSION"),
                            ));
                        responder.respond(resp)
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        // ── session/new ─────────────────────────────────────────────────
        .on_receive_request_from(
            Client,
            {
                let me = me.clone();
                move |req: NewSessionRequest, responder: Responder<NewSessionResponse>, _cx| {
                    let me = me.clone();
                    async move {
                        let session_id = SessionId::new(format!("akh-{}", next_session_suffix()));
                        // The client encodes the target workspace in the session
                        // `cwd` as the sentinel `akh-ws:<name>` (akhomed otherwise
                        // ignores the client cwd). Record session → workspace so
                        // each prompt routes to that akh — the chat half of `-w`.
                        if let Some(ws) = req
                            .cwd
                            .to_str()
                            .and_then(|s| s.strip_prefix("akh-ws:"))
                            .filter(|s| !s.is_empty())
                        {
                            me.sessions
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .insert(session_id.to_string(), ws.to_string());
                        }
                        responder.respond(NewSessionResponse::new(session_id))
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        // ── session/prompt ──────────────────────────────────────────────
        .on_receive_request_from(
            Client,
            {
                let me = me.clone();
                move |req: PromptRequest,
                      responder: Responder<PromptResponse>,
                      cx: ConnectionTo<Client>| {
                    let me = me.clone();
                    async move {
                        let session_id = req.session_id.clone();
                        let text = concat_prompt(&req.prompt);

                        // Route to the session's workspace (recorded at
                        // session/new) and resolve its handles on demand — the
                        // chat half of `-w`: one daemon serves any configured akh.
                        let ws = me.session_workspace(&session_id.to_string());
                        let (engine, agent, processor) = match me.resolve_handles(&ws).await {
                            Ok(h) => h,
                            Err(e) => {
                                AcpSink::new(cx.clone(), session_id.clone()).emit(
                                    &AkhMessage::error(
                                        "acp",
                                        format!("cannot open workspace \"{ws}\": {e}"),
                                    ),
                                );
                                return responder
                                    .respond(PromptResponse::new(StopReason::EndTurn));
                            }
                        };

                        // Per-turn permission bridge: a sync asker (consulted
                        // inside the blocking chat turn when the policy says
                        // "Ask") relays to this async approver task, which calls
                        // session/request_permission on the client. Under the
                        // default Bypass mode the asker is simply never invoked.
                        let (perm_tx, perm_rx) =
                            tokio::sync::mpsc::channel::<PermissionAskRequest>(8);
                        let approver = tokio::spawn(serve_permission_asks(
                            perm_rx,
                            cx.clone(),
                            session_id.clone(),
                        ));
                        let asker: Arc<dyn crate::agent::permission::PermissionAsker> =
                            Arc::new(AcpPermissionAsker { tx: perm_tx });

                        // Run the synchronous chat turn off the async runtime
                        // (ADR-027: locks live only inside the closure). It
                        // streams its narration through an unbounded channel; we
                        // forward each message to the client as a `session/update`
                        // as it arrives, so the answer appears live rather than
                        // in one batch at turn-end.
                        let (msg_tx, mut msg_rx) =
                            tokio::sync::mpsc::unbounded_channel::<AkhMessage>();
                        let handle = AkhAcpAgent::run_turn(
                            engine,
                            agent,
                            processor,
                            ws,
                            text,
                            Some(asker),
                            ChannelSink { tx: msg_tx },
                        );

                        let sink = AcpSink::new(cx, session_id.clone());
                        while let Some(msg) = msg_rx.recv().await {
                            sink.emit(&msg);
                        }

                        // Channel closed ⇒ the turn finished and dropped its sink.
                        // Join to surface a panicked turn task, and let the
                        // approver (whose Sender was dropped with the asker at
                        // turn-end) drain and finish.
                        if let Err(e) = handle.await {
                            sink.emit(&AkhMessage::error(
                                "acp",
                                format!("turn task failed: {e}"),
                            ));
                        }
                        let _ = approver.await;

                        responder.respond(PromptResponse::new(StopReason::EndTurn))
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        // ── _akh/* data-plane (catch-all, after the typed handlers) ─────────
        .on_receive_dispatch_from(
            Client,
            {
                let me = me.clone();
                move |message: Dispatch, _cx: ConnectionTo<Client>| {
                    let me = me.clone();
                    let control = me.control.clone();
                    let has_control = control.is_some();
                    let default_ws = me.default_ws.clone();
                    async move {
                        MatchDispatch::new(message)
                            .if_request(
                                async move |req: AkhExtRequest,
                                            responder: Responder<AkhExtResponse>| {
                                    let method: &str = &req.method;
                                    // Server-level control-plane (workspace/
                                    // daemon/config ops that span the whole
                                    // daemon). Routed to the ServerControl
                                    // backend before the per-workspace data-plane.
                                    if is_control_method(method) {
                                        return match &control {
                                            Some(ctrl) => match ctrl
                                                .dispatch(&req.method, &req.params)
                                                .await
                                            {
                                                Ok(value) => {
                                                    responder.respond(AkhExtResponse(value))
                                                }
                                                Err(e) => responder.respond_with_error(
                                                    agent_client_protocol::Error::internal_error()
                                                        .data(e),
                                                ),
                                            },
                                            None => responder.respond_with_error(
                                                agent_client_protocol::Error::method_not_found()
                                                    .data(format!(
                                                        "control-plane method {method} requires a \
                                                         ServerControl backend (none wired)"
                                                    )),
                                            ),
                                        };
                                    }
                                    // Resolve the target workspace for this
                                    // data-plane op: a `workspace` param selects
                                    // the akh (the `-w` data-plane); absent ⇒ the
                                    // bound default. Loaded on demand so any
                                    // configured akh is reachable from one daemon.
                                    let workspace = req
                                        .params
                                        .get("workspace")
                                        .and_then(serde_json::Value::as_str)
                                        .map(str::to_string)
                                        .unwrap_or_else(|| default_ws.clone());
                                    let (engine, agent, _processor) =
                                        match me.resolve_handles(&workspace).await {
                                            Ok(h) => h,
                                            Err(e) => {
                                                return responder.respond_with_error(
                                                    agent_client_protocol::Error::internal_error()
                                                        .data(e),
                                                );
                                            }
                                        };
                                    // Agent-mutating method: take the agent lock
                                    // inside spawn_blocking (ADR-027). GET when no
                                    // "mode" param; SET otherwise.
                                    if method == "_akh/permission/mode" {
                                        let params = req.params.clone();
                                        let result = tokio::task::spawn_blocking(move || {
                                            let mut a =
                                                agent.lock().unwrap_or_else(|e| e.into_inner());
                                            if let Some(mode_str) = params
                                                .get("mode")
                                                .and_then(serde_json::Value::as_str)
                                            {
                                                match parse_permission_mode(mode_str) {
                                                    Some(mode) => a.set_permission_mode(mode),
                                                    None => return Err(format!(
                                                        "unknown permission mode \"{mode_str}\" \
                                                         (expected bypass|standard|accept_edits|plan)"
                                                    )),
                                                }
                                            }
                                            Ok(permission_mode_str(a.permission_mode()))
                                        })
                                        .await;
                                        return match result {
                                            Ok(Ok(mode)) => responder.respond(AkhExtResponse(
                                                serde_json::json!({ "mode": mode }),
                                            )),
                                            Ok(Err(e)) => responder.respond_with_error(
                                                agent_client_protocol::Error::invalid_params()
                                                    .data(e),
                                            ),
                                            Err(join_err) => responder.respond_with_error(
                                                agent_client_protocol::Error::internal_error()
                                                    .data(join_err.to_string()),
                                            ),
                                        };
                                    }
                                    // Agent-read methods (goals/tools/projects/
                                    // status) snapshot the agent under try_lock,
                                    // so a sidebar refresh never blocks a live
                                    // turn (ADR 063 §2). Engine-only methods fall
                                    // through to dispatch_akh below.
                                    if let Some(result) = dispatch_akh_agent(
                                        &engine,
                                        &agent,
                                        &workspace,
                                        &req.method,
                                        &req.params,
                                    ) {
                                        return match result {
                                            Ok(value) => {
                                                responder.respond(AkhExtResponse(value))
                                            }
                                            Err(e) => responder.respond_with_error(
                                                agent_client_protocol::Error::internal_error()
                                                    .data(e),
                                            ),
                                        };
                                    }
                                    match dispatch_akh(
                                        &engine,
                                        &workspace,
                                        has_control,
                                        &req.method,
                                        &req.params,
                                    ) {
                                        Ok(value) => responder.respond(AkhExtResponse(value)),
                                        Err(e) => responder.respond_with_error(
                                            agent_client_protocol::Error::method_not_found()
                                                .data(e),
                                        ),
                                    }
                                },
                            )
                            .await
                            .done()
                    }
                }
            },
            agent_client_protocol::on_receive_dispatch!(),
        )
}

/// Concatenate the text content of a prompt's content blocks.
///
/// Non-text blocks (images, audio, resources) are ignored in the spike.
fn concat_prompt(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Monotonic-ish suffix for session ids within a process.
fn next_session_suffix() -> u128 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    nanos.wrapping_add(u128::from(n))
}

/// Serve the ACP agent over a tokio [`UnixStream`], adapting it into the
/// crate's `futures`-based [`ByteStreams`] transport.
///
/// The stream is split into read/write halves and bridged with
/// [`tokio_util::compat`]. This consumes the connection and runs until the
/// peer disconnects or the protocol loop ends.
pub async fn serve_stream(me: AkhAcpAgent, stream: UnixStream) -> AcpResult<()> {
    // tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf} → futures io via compat.
    let (read_half, write_half) = stream.into_split();
    let incoming = read_half.compat();
    let outgoing = write_half.compat_write();

    // ByteStreams::new(outgoing_writer, incoming_reader)
    let transport = ByteStreams::new(outgoing, incoming);

    // The agent builder is a ConnectTo<Client>; connecting it TO the transport
    // (a role-agnostic ConnectTo<Agent>) runs the agent's dispatch loop over the
    // socket, mirroring how `Stdio`/`AcpAgent` wire byte streams in the crate.
    use agent_client_protocol::ConnectTo;
    build_connection(me).connect_to(transport).await?;
    Ok(())
}

/// Bind a Unix domain socket at `path`, accept exactly ONE connection, and serve
/// the ACP agent over it.
///
/// This is the single-workspace spike entry point. A real server would loop on
/// `accept()` and spawn a task per connection; here we accept one and return.
///
/// Any stale socket file at `path` is removed first (best-effort).
pub async fn serve_unix(me: AkhAcpAgent, path: impl AsRef<Path>) -> AcpResult<()> {
    let path = path.as_ref();

    // Best-effort cleanup of a stale socket so bind() doesn't fail with EADDRINUSE.
    let _ = std::fs::remove_file(path);

    let listener = UnixListener::bind(path).map_err(|source| AcpError::Bind {
        path: path.to_path_buf(),
        source,
    })?;

    let (stream, _addr) = listener.accept().await.map_err(|source| AcpError::Accept {
        path: path.to_path_buf(),
        source,
    })?;

    serve_stream(me, stream).await
}

/// Serve the ACP agent over a tokio [`TcpStream`].
///
/// Identical to [`serve_stream`] but for TCP: the stream's halves are split and
/// bridged into the crate's `futures`-based [`ByteStreams`] transport. The byte
/// protocol is transport-agnostic, so a TCP client and a Unix client are served
/// by the exact same dispatch logic.
pub async fn serve_tcp_stream(me: AkhAcpAgent, stream: TcpStream) -> AcpResult<()> {
    // Disable Nagle so small JSON-RPC frames are not delayed.
    let _ = stream.set_nodelay(true);
    let (read_half, write_half) = stream.into_split();
    let incoming = read_half.compat();
    let outgoing = write_half.compat_write();

    let transport = ByteStreams::new(outgoing, incoming);

    use agent_client_protocol::ConnectTo;
    build_connection(me).connect_to(transport).await?;
    Ok(())
}

/// Accept connections on a bound [`UnixListener`] until `shutdown` flips to
/// `true`, serving each on its own task.
///
/// This is the production accept loop: the cheap-to-clone [`AkhAcpAgent`]
/// (`Arc`-backed) is cloned per connection, so many clients are served
/// concurrently against the same shared engine/agent. A failed connection logs
/// and is dropped; it does not tear down the loop.
pub async fn serve_unix_loop(
    me: AkhAcpAgent,
    listener: UnixListener,
    mut shutdown: watch::Receiver<bool>,
) -> AcpResult<()> {
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _addr) = accepted.map_err(|source| AcpError::Accept {
                    path: PathBuf::from("<acp-unix-listener>"),
                    source,
                })?;
                let conn = me.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_stream(conn, stream).await {
                        tracing::warn!("ACP unix connection ended with error: {e}");
                    }
                });
            }
            res = shutdown.changed() => {
                // Sender dropped (Err) or signalled true → stop accepting.
                if res.is_err() || *shutdown.borrow() {
                    tracing::info!("ACP unix accept loop shutting down");
                    return Ok(());
                }
            }
        }
    }
}

/// Accept connections on a bound [`TcpListener`] until `shutdown` flips to
/// `true`, serving each on its own task. TCP analogue of [`serve_unix_loop`].
pub async fn serve_tcp_loop(
    me: AkhAcpAgent,
    listener: TcpListener,
    mut shutdown: watch::Receiver<bool>,
) -> AcpResult<()> {
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _addr) = accepted.map_err(|source| AcpError::Accept {
                    path: PathBuf::from("<acp-tcp-listener>"),
                    source,
                })?;
                let conn = me.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_tcp_stream(conn, stream).await {
                        tracing::warn!("ACP tcp connection ended with error: {e}");
                    }
                });
            }
            res = shutdown.changed() => {
                if res.is_err() || *shutdown.borrow() {
                    tracing::info!("ACP tcp accept loop shutting down");
                    return Ok(());
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Client — connect to a daemon and drive sessions over ACP
// ──────────────────────────────────────────────────────────────────────────

/// Connect over `transport`, `initialize`, start ONE persistent session, then
/// run `f` with the live [`ActiveSession`] and a clone of the agent
/// [`ConnectionTo`].
///
/// `f` owns the session for its whole duration, so it may issue many
/// `send_prompt` turns — and later `_akh/*` ext requests via the connection —
/// over the single connection. The crate keeps the connection alive until `f`
/// returns; on return the protocol loop drains and the transport closes. This
/// is the lifetime constraint of the ACP client: the connection cannot outlive
/// `f`, so the interactive loop must live inside it.
async fn run_session<T, R, F>(transport: T, name: &str, workspace: &str, f: F) -> AcpResult<R>
where
    T: ConnectTo<Client> + 'static,
    F: AsyncFnOnce(
        ActiveSession<'static, AcpRole>,
        ConnectionTo<AcpRole>,
    ) -> Result<R, agent_client_protocol::Error>,
{
    // Encode the target workspace in the session `cwd` as `akh-ws:<name>`; the
    // daemon reads it at `session/new` to route this session's prompts to that
    // akh (the chat half of `-w`). akhomed otherwise ignores the client cwd.
    let cwd = format!("akh-ws:{workspace}");
    Client
        .builder()
        .name(name.to_string())
        .connect_with(transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .build_session(cwd)
                .block_task()
                .start_session()
                .await?;
            f(session, cx).await
        })
        .await
        .map_err(AcpError::from)
}

/// Retry budget for connecting to a daemon whose listener may still be coming
/// up. Bounded so a genuinely-down daemon still fails within a couple seconds.
const ACP_CONNECT_RETRY_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);

/// `true` for the transient errors seen while a just-spawned daemon finishes
/// binding its listener: ECONNREFUSED (socket bound, not yet accepting) and
/// ENOENT (socket file not created yet). Anything else is a real failure.
///
/// This is the race the in-process ACP tests hit under full-suite CPU load: the
/// socket file exists (`bind` ran) but the accept loop isn't ready, so the first
/// `connect` is refused. Retrying briefly turns that into a clean connection —
/// and equally hardens a real client racing a daemon that's still starting up.
fn acp_connect_transient(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
    )
}

/// Connect to a Unix socket, retrying [`acp_connect_transient`] failures with
/// capped backoff until [`ACP_CONNECT_RETRY_BUDGET`] is spent.
async fn connect_unix_retrying(path: &Path) -> std::io::Result<UnixStream> {
    let mut delay = std::time::Duration::from_millis(20);
    let mut elapsed = std::time::Duration::ZERO;
    loop {
        match UnixStream::connect(path).await {
            Ok(s) => return Ok(s),
            Err(e) if elapsed < ACP_CONNECT_RETRY_BUDGET && acp_connect_transient(&e) => {
                tokio::time::sleep(delay).await;
                elapsed += delay;
                delay = (delay * 2).min(std::time::Duration::from_millis(150));
            }
            Err(e) => return Err(e),
        }
    }
}

/// TCP analogue of [`connect_unix_retrying`].
async fn connect_tcp_retrying(addr: &str) -> std::io::Result<TcpStream> {
    let mut delay = std::time::Duration::from_millis(20);
    let mut elapsed = std::time::Duration::ZERO;
    loop {
        match TcpStream::connect(addr).await {
            Ok(s) => return Ok(s),
            Err(e) if elapsed < ACP_CONNECT_RETRY_BUDGET && acp_connect_transient(&e) => {
                tokio::time::sleep(delay).await;
                elapsed += delay;
                delay = (delay * 2).min(std::time::Duration::from_millis(150));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Connect to an ACP daemon over a Unix socket and run `f` over one session.
///
/// `f` receives a started [`ActiveSession`] (call `send_prompt` / `read_to_string`
/// / `read_update` on it, repeatedly) and a [`ConnectionTo`] for ext requests.
pub async fn chat_over_unix<R, F>(
    path: impl AsRef<Path>,
    name: &str,
    workspace: &str,
    f: F,
) -> AcpResult<R>
where
    F: AsyncFnOnce(
        ActiveSession<'static, AcpRole>,
        ConnectionTo<AcpRole>,
    ) -> Result<R, agent_client_protocol::Error>,
{
    let path = path.as_ref();
    let stream = connect_unix_retrying(path)
        .await
        .map_err(|source| AcpError::Connect {
            target: path.display().to_string(),
            source,
        })?;
    let (read_half, write_half) = stream.into_split();
    let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
    run_session(transport, name, workspace, f).await
}

/// Connect to an ACP daemon over TCP (`host:port`) and run `f` over one session
/// bound to `workspace`.
pub async fn chat_over_tcp<R, F>(addr: &str, name: &str, workspace: &str, f: F) -> AcpResult<R>
where
    F: AsyncFnOnce(
        ActiveSession<'static, AcpRole>,
        ConnectionTo<AcpRole>,
    ) -> Result<R, agent_client_protocol::Error>,
{
    let stream = connect_tcp_retrying(addr)
        .await
        .map_err(|source| AcpError::Connect {
            target: addr.to_string(),
            source,
        })?;
    let _ = stream.set_nodelay(true);
    let (read_half, write_half) = stream.into_split();
    let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
    run_session(transport, name, workspace, f).await
}

/// Send one `_akh/*` data-plane request over `transport` and return its JSON
/// result. Connects, `initialize`s, sends the request, and disconnects — the
/// one-shot shape that fits stateless CLI data commands (no chat session).
async fn ext_request<T>(transport: T, method: &str, params: serde_json::Value) -> AcpResult<serde_json::Value>
where
    T: ConnectTo<Client> + 'static,
{
    let method = method.to_string();
    Client
        .builder()
        .name("akh-ext-client")
        .connect_with(transport, async move |cx| {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let resp: AkhExtResponse = cx
                .send_request(AkhExtRequest { method, params })
                .block_task()
                .await?;
            Ok(resp.0)
        })
        .await
        .map_err(AcpError::from)
}

/// Send a `_akh/*` data-plane request to a daemon over a Unix socket.
///
/// `method` must include the leading `_akh/` (e.g. `"_akh/health"`).
pub async fn ext_request_unix(
    path: impl AsRef<Path>,
    method: &str,
    params: serde_json::Value,
) -> AcpResult<serde_json::Value> {
    let path = path.as_ref();
    let stream = connect_unix_retrying(path)
        .await
        .map_err(|source| AcpError::Connect {
            target: path.display().to_string(),
            source,
        })?;
    let (read_half, write_half) = stream.into_split();
    let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
    ext_request(transport, method, params).await
}

/// Send a `_akh/*` data-plane request to a daemon over TCP (`host:port`).
pub async fn ext_request_tcp(
    addr: &str,
    method: &str,
    params: serde_json::Value,
) -> AcpResult<serde_json::Value> {
    let stream = connect_tcp_retrying(addr)
        .await
        .map_err(|source| AcpError::Connect {
            target: addr.to_string(),
            source,
        })?;
    let _ = stream.set_nodelay(true);
    let (read_half, write_half) = stream.into_split();
    let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
    ext_request(transport, method, params).await
}

/// Interactive headless chat loop over one ACP session: read a line from stdin,
/// send it as a prompt, print the turn's response; repeat until EOF or `/quit`.
///
/// `label` is the akh/workspace name shown in the banner. Lives in the library
/// so the CLI need not depend on `agent-client-protocol` directly.
async fn headless_loop(
    session: &mut ActiveSession<'static, AcpRole>,
    label: &str,
) -> Result<(), agent_client_protocol::Error> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let to_err =
        |e: std::io::Error| agent_client_protocol::Error::internal_error().data(e.to_string());

    let mut stderr = tokio::io::stderr();
    let _ = stderr
        .write_all(
            format!("Connected to akh '{label}' over ACP. Type a message; Ctrl-D or /quit to exit.\n")
                .as_bytes(),
        )
        .await;
    let _ = stderr.flush().await;

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        let _ = stderr.write_all(b"> ").await;
        let _ = stderr.flush().await;

        let Some(line) = lines.next_line().await.map_err(to_err)? else {
            break;
        };
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if matches!(msg, "/quit" | "/exit" | "quit" | "exit") {
            break;
        }

        session.send_prompt(msg)?;
        let resp = session.read_to_string().await?;
        if !resp.trim().is_empty() {
            println!("{resp}");
        }
    }
    Ok(())
}

/// Connect to a daemon over a Unix socket and run an interactive headless chat
/// (stdin/stdout) over one persistent ACP session.
pub async fn run_headless_chat_unix(
    path: impl AsRef<Path>,
    name: &str,
    label: &str,
) -> AcpResult<()> {
    let label = label.to_string();
    chat_over_unix(path, name, &label.clone(), async move |mut session, _cx| {
        headless_loop(&mut session, &label).await
    })
    .await
}

/// Connect to a daemon over TCP (`host:port`) and run an interactive headless
/// chat over one persistent ACP session.
pub async fn run_headless_chat_tcp(addr: &str, name: &str, label: &str) -> AcpResult<()> {
    let label = label.to_string();
    chat_over_tcp(addr, name, &label.clone(), async move |mut session, _cx| {
        headless_loop(&mut session, &label).await
    })
    .await
}

// ──────────────────────────────────────────────────────────────────────────
// End-to-end test: real Unix socket, real ACP client ↔ ACP agent.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
// NOTE: the `tokio::time::timeout(..)` guards in these socket tests are
// deliberately generous (60s for an operation, 30s for a post-shutdown join).
// They exist only to bound a genuine hang (a wedged server / deadlock), NOT to
// assert latency — a prompt turn finishes in well under a second. Each test
// runs a multi-thread runtime, so under full-suite parallel load the runtimes
// oversubscribe the CPU and a tight (e.g. 15s) budget flakes spuriously. Do not
// tighten these back; if a test legitimately needs >60s, something is wrong.
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::engine::EngineConfig;
    use crate::nlu::NluPipeline;
    use crate::vsa::Dimension;
    use agent_client_protocol::schema::{InitializeRequest, ProtocolVersion};
    use agent_client_protocol::Client;

    /// Build a tiny in-process workspace seeded with one fact.
    fn workspace() -> AkhAcpAgent {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .expect("engine");

        // Seed: ("Sun", "is-a", "Star", 1.0)
        engine
            .ingest_label_triples(&[(
                "Sun".to_string(),
                "is-a".to_string(),
                "Star".to_string(),
                1.0,
            )])
            .expect("seed triple");

        let engine = Arc::new(engine);

        let processor = ChatProcessor::new(&engine, NluPipeline::new());
        let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        let shared_agent = Arc::new(Mutex::new(agent));

        AkhAcpAgent::new(engine, shared_agent, processor)
    }

    /// Build a bare engine + shared agent (no ACP wrapper) for exercising the
    /// agent-read dispatch helpers directly.
    fn engine_and_agent() -> (Arc<Engine>, Arc<Mutex<Agent>>) {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .expect("engine");
        let engine = Arc::new(engine);
        let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        (engine, Arc::new(Mutex::new(agent)))
    }

    #[test]
    fn agent_read_status_snapshot_is_fresh_when_idle() {
        let (engine, agent) = engine_and_agent();

        let v = dispatch_akh_agent(&engine, &agent, "default", "_akh/status", &serde_json::json!({}))
            .expect("status is an agent-read method")
            .expect("ok");
        assert_eq!(v["stale"], serde_json::json!(false));
        // All sidebar arrays are present (possibly empty) for a fresh agent.
        assert!(v["goals"].is_array());
        assert!(v["tools"].is_array());
        assert!(v["projects"].is_array());
        assert!(v["skills"].is_array());
        // KG counts are engine-derivable, always present.
        assert!(v["kg"]["symbols"].is_number());
        assert!(v["kg"]["triples"].is_number());
        // Cognitive panels: reasoning is an array, evidence is present (fresh).
        assert!(v["reasoning"].is_array());
        assert!(v["evidence"]["claims"].is_number());
        // No model manager on a test agent ⇒ usage is null.
        assert_eq!(v["usage"], serde_json::Value::Null);
    }

    #[test]
    fn agent_read_reports_stale_when_agent_is_busy() {
        let (engine, agent) = engine_and_agent();

        // Simulate a turn holding the agent lock; try_lock must not block.
        let _busy = agent.lock().expect("hold the agent lock");
        let v = dispatch_akh_agent(&engine, &agent, "default", "_akh/status", &serde_json::json!({}))
            .unwrap()
            .unwrap();
        assert_eq!(v["stale"], serde_json::json!(true));
        // Agent-derived fields are empty under contention…
        assert_eq!(v["goals"], serde_json::json!([]));
        assert_eq!(v["tools"], serde_json::json!([]));
        assert_eq!(v["cycle"], serde_json::Value::Null);
        // …but engine-derivable skills are still served.
        assert!(v["skills"].is_array());
    }

    #[test]
    fn agent_read_ignores_non_agent_methods() {
        let (engine, agent) = engine_and_agent();
        // Engine-only / unknown methods fall through (None) to dispatch_akh.
        assert!(
            dispatch_akh_agent(&engine, &agent, "default", "_akh/health", &serde_json::json!({}))
                .is_none()
        );
        assert!(
            dispatch_akh_agent(&engine, &agent, "default", "_akh/skills", &serde_json::json!({}))
                .is_none()
        );
    }

    /// Full ACP lifecycle over a REAL Unix domain socket:
    /// server binds + accepts one connection; an in-process ACP **client**
    /// connects over the same socket and drives initialize → session/new →
    /// session/prompt, asserting the prompt turn completes with non-empty text.
    ///
    /// This exercises the byte-stream transport (tokio↔futures bridge) AND the
    /// ADR-027 lock discipline end to end.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_unix_socket_prompt_turn() {
        // Unique socket path inside the OS temp dir (short, to stay under the
        // sun_path length limit on macOS/Linux).
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp.sock");

        // ── Server: bind + accept one connection, serve the agent. ─────────
        let me = workspace();
        let server_sock = sock.clone();
        let server = tokio::spawn(async move { serve_unix(me, &server_sock).await });

        // Give the listener a moment to bind before the client connects.
        // (Poll-connect loop avoids a fixed sleep.)
        let stream = connect_with_retry(&sock).await;

        // ── Client: drive the full ACP lifecycle over the same socket. ─────
        let (read_half, write_half) = stream.into_split();
        let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());

        let client_lifecycle = Client
            .builder()
            .name("akh-acp-test-client")
            .connect_with(transport, async move |cx| {
                // 1) initialize
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;

                // 2) session/new + 3) session/prompt, reading the turn to a string.
                cx.build_session_cwd()?
                    .block_task()
                    .run_until(async move |mut session| {
                        session.send_prompt("What is the Sun?")?;
                        let text = session.read_to_string().await?;
                        Ok(text)
                    })
                    .await
            });

        // Hard bound: a transport/protocol deadlock must fail fast, never hang.
        let answer: String = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            client_lifecycle,
        )
        .await
        .expect("ACP client lifecycle timed out after 15s — likely a transport/protocol deadlock")
        .expect("client lifecycle completed");

        // The prompt turn must complete and return non-empty response content.
        assert!(
            !answer.trim().is_empty(),
            "expected non-empty ACP prompt response, got: {answer:?}"
        );

        // The client already received its answer (asserted above). The single-accept
        // server may still be parked on the dispatch loop; bound the join so the test
        // can never hang, and don't fail the spike if the loop hasn't unwound yet
        // (runtime shutdown at test end cancels any still-running spawned task).
        match tokio::time::timeout(std::time::Duration::from_secs(30), server).await {
            Ok(joined) => joined
                .expect("server task joined")
                .expect("server served without protocol error"),
            Err(_) => eprintln!(
                "note: ACP server task still parked 5s after client completed \
                 (single-accept dispatch loop) — acceptable for the spike"
            ),
        }
    }

    /// Connect to the Unix socket, retrying briefly while the server binds.
    async fn connect_with_retry(path: &Path) -> UnixStream {
        for _ in 0..200 {
            match UnixStream::connect(path).await {
                Ok(s) => return s,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(5)).await,
            }
        }
        panic!("could not connect to ACP Unix socket at {path:?}");
    }

    /// Regression (ADR-061 P8.8): the *production* `connect_unix_retrying` —
    /// used by `chat_over_unix`/`ext_request_unix` — must ride through the
    /// daemon-startup races, ENOENT (socket file not created yet) then
    /// ECONNREFUSED (bound, not yet accepting). Before this retry a single early
    /// `connect` aborted the session, which flaked the TUI relay test under
    /// full-suite CPU load (the in-process socket tests above never saw it
    /// because they use the `connect_with_retry` test helper).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connect_unix_retrying_rides_through_late_listener() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("late.sock");

        // Listener comes up ~120ms late — far inside the 2s retry budget. The
        // first connects hit ENOENT/ECONNREFUSED; the retry must still land.
        let bind_sock = sock.clone();
        let server = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(120)).await;
            let listener = tokio::net::UnixListener::bind(&bind_sock).expect("late bind");
            // Hold the listener open long enough for the client to connect.
            let _ =
                tokio::time::timeout(std::time::Duration::from_secs(5), listener.accept()).await;
        });

        let connected = connect_unix_retrying(&sock).await;
        assert!(
            connected.is_ok(),
            "connect_unix_retrying must ride through a late listener: {connected:?}"
        );
        let _ = server.await;
    }

    /// Drive a full ACP prompt lifecycle (initialize → session/new →
    /// session/prompt → read) over an already-connected Unix stream, returning
    /// the concatenated agent message text. Hard-bounded so a deadlock fails
    /// fast instead of hanging.
    async fn unix_client_prompt(stream: UnixStream, prompt: &'static str) -> String {
        let (read_half, write_half) = stream.into_split();
        let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
        let lifecycle = Client
            .builder()
            .name("akh-acp-test-client")
            .connect_with(transport, async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                cx.build_session_cwd()?
                    .block_task()
                    .run_until(async move |mut session| {
                        session.send_prompt(prompt)?;
                        session.read_to_string().await
                    })
                    .await
            });
        tokio::time::timeout(std::time::Duration::from_secs(60), lifecycle)
            .await
            .expect("unix client lifecycle timed out (transport/protocol deadlock)")
            .expect("unix client lifecycle completed")
    }

    /// TCP analogue of [`unix_client_prompt`].
    async fn tcp_client_prompt(stream: TcpStream, prompt: &'static str) -> String {
        let (read_half, write_half) = stream.into_split();
        let transport = ByteStreams::new(write_half.compat_write(), read_half.compat());
        let lifecycle = Client
            .builder()
            .name("akh-acp-tcp-test-client")
            .connect_with(transport, async move |cx| {
                cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                cx.build_session_cwd()?
                    .block_task()
                    .run_until(async move |mut session| {
                        session.send_prompt(prompt)?;
                        session.read_to_string().await
                    })
                    .await
            });
        tokio::time::timeout(std::time::Duration::from_secs(60), lifecycle)
            .await
            .expect("tcp client lifecycle timed out (transport/protocol deadlock)")
            .expect("tcp client lifecycle completed")
    }

    /// Same prompt turn as the Unix test, but over a real **TCP** socket —
    /// proves the byte protocol is transport-agnostic (Unix/TCP parity).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_tcp_prompt_turn() {
        let me = workspace();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind tcp");
        let addr = listener.local_addr().expect("local addr");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept tcp");
            serve_tcp_stream(me, stream).await
        });

        let stream = TcpStream::connect(addr).await.expect("connect tcp");
        let answer = tcp_client_prompt(stream, "What is the Sun?").await;
        assert!(
            !answer.trim().is_empty(),
            "expected non-empty ACP/TCP response, got: {answer:?}"
        );

        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The accept loop serves more than one connection against the same shared
    /// agent, then exits cleanly when the shutdown watch flips.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_unix_accept_loop_serves_multiple_connections() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-loop.sock");

        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let loop_task = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // Two clients, served sequentially against the same loop + shared agent.
        let a = unix_client_prompt(connect_with_retry(&sock).await, "What is the Sun?").await;
        assert!(!a.trim().is_empty(), "first connection got empty answer: {a:?}");
        let b = unix_client_prompt(connect_with_retry(&sock).await, "What is the Sun?").await;
        assert!(!b.trim().is_empty(), "second connection got empty answer: {b:?}");

        // Signal shutdown; the loop must return Ok promptly.
        shutdown_tx.send(true).expect("send shutdown");
        let joined = tokio::time::timeout(std::time::Duration::from_secs(30), loop_task)
            .await
            .expect("accept loop did not shut down within 5s")
            .expect("accept loop task joined");
        joined.expect("accept loop returned Ok");
    }

    /// The reusable client (`chat_over_unix`) holds ONE connection + session and
    /// issues TWO prompt turns over it — proving persistent multi-turn sessions
    /// (the basis for `akh chat` over ACP).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_client_persistent_session_two_prompts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-client.sock");

        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let (a, b) = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            chat_over_unix(&sock, "akh-acp-test-client", "akh-medu", async move |mut session, _cx| {
                session.send_prompt("What is the Sun?")?;
                let a = session.read_to_string().await?;
                session.send_prompt("What is the Sun?")?;
                let b = session.read_to_string().await?;
                Ok::<(String, String), agent_client_protocol::Error>((a, b))
            }),
        )
        .await
        .expect("client timed out (transport/protocol deadlock)")
        .expect("client session completed");

        assert!(!a.trim().is_empty(), "first turn empty: {a:?}");
        assert!(!b.trim().is_empty(), "second turn empty: {b:?}");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The `_akh/*` data-plane: a client `ext_request` reaches the server's
    /// dispatch handler and gets a structured JSON result back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_health_and_capabilities() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-ext.sock");

        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // _akh/health → status ok + counts (the seeded triple ⇒ ≥1 triple).
        let health = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/health", serde_json::json!({})),
        )
        .await
        .expect("ext health timed out")
        .expect("ext health ok");
        assert_eq!(health["status"], "ok");
        assert!(health["triples"].as_u64().unwrap() >= 1, "expected seeded triples: {health}");

        // _akh/meta/capabilities → advertises the data-plane.
        let caps = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/meta/capabilities", serde_json::json!({})),
        )
        .await
        .expect("ext caps timed out")
        .expect("ext caps ok");
        assert_eq!(caps["protocol"], "acp");
        assert!(caps["methods"].as_array().unwrap().iter().any(|m| m == "_akh/health"));

        // An unknown method is a clean error, not a hang or panic.
        let unknown = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/does/not/exist", serde_json::json!({})),
        )
        .await
        .expect("ext unknown timed out");
        assert!(unknown.is_err(), "unknown _akh method must error");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The read-only introspection group (`workspace/status`, `symbols/list`,
    /// `triples/from`) returns engine-backed JSON over the data-plane.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_introspection_group() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-introspect.sock");

        let me = workspace().with_name("default");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let call = |method: &'static str, params: serde_json::Value| {
            let sock = sock.clone();
            async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    ext_request_unix(&sock, method, params),
                )
                .await
                .expect("ext request timed out")
                .expect("ext request ok")
            }
        };

        // workspace/status
        let st = call("_akh/workspace/status", serde_json::json!({})).await;
        assert_eq!(st["workspace"], "default");
        assert!(st["symbols"].as_u64().unwrap() >= 3, "Sun/Star/is-a ⇒ ≥3 symbols: {st}");

        // symbols/list — the engine also registers many built-in symbols, so
        // request all of them (count == returned) to be sure the seeded "Sun"
        // is in the set. (A bounded `limit` returns an arbitrary subset, by
        // design — that is what `symbols/list` does.)
        let syms = call("_akh/symbols/list", serde_json::json!({ "limit": 1_000_000 })).await;
        assert_eq!(
            syms["count"], syms["returned"],
            "limit was meant to return every symbol: {syms}"
        );
        let labels: Vec<&str> = syms["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|s| s["label"].as_str())
            .collect();
        assert!(labels.contains(&"Sun"), "symbols/list missing seeded Sun");

        // triples/from Sun → is-a Star
        let tr = call("_akh/triples/from", serde_json::json!({ "symbol": "Sun" })).await;
        assert_eq!(tr["symbol"], "Sun");
        assert!(
            tr["triples"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["object"] == "Star"),
            "triples/from Sun missing object Star: {tr}"
        );

        // missing required param → clean error.
        let bad = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/triples/from", serde_json::json!({})),
        )
        .await
        .expect("timed out");
        assert!(bad.is_err(), "triples/from without symbol must error");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The write group: `_akh/triples/add` mutates the KG and the change is
    /// immediately visible via `_akh/triples/from`; `_akh/symbols/create`
    /// resolves/creates a symbol.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_write_group() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-write.sock");

        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let call = |method: &'static str, params: serde_json::Value| {
            let sock = sock.clone();
            async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    ext_request_unix(&sock, method, params),
                )
                .await
                .expect("ext request timed out")
                .expect("ext request ok")
            }
        };

        // add a triple → at least one triple asserted.
        let add = call(
            "_akh/triples/add",
            serde_json::json!({ "subject": "Moon", "predicate": "orbits", "object": "Earth" }),
        )
        .await;
        assert!(
            add["triples_added"].as_u64().unwrap() >= 1,
            "expected ≥1 triple added: {add}"
        );

        // it is immediately readable back.
        let back = call("_akh/triples/from", serde_json::json!({ "symbol": "Moon" })).await;
        assert!(
            back["triples"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t["object"] == "Earth" && t["predicate"] == "orbits"),
            "added triple not visible: {back}"
        );

        // create a symbol.
        let created = call("_akh/symbols/create", serde_json::json!({ "label": "Mars" })).await;
        assert_eq!(created["label"], "Mars");
        assert!(created["id"].as_u64().is_some(), "symbol id expected: {created}");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The query group: `_akh/query/sparql` runs a SELECT over a persistent
    /// (SPARQL-enabled) engine and returns solution bindings. Uses a tempdir
    /// data-dir because `sparql_query` requires persistence — exactly what the
    /// daemon's workspaces have.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_query_sparql() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().join("kg");
        std::fs::create_dir_all(&data_dir).expect("data dir");

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(data_dir),
            ..Default::default()
        })
        .expect("persistent engine");
        engine
            .ingest_label_triples(&[("Sun".to_string(), "is-a".to_string(), "Star".to_string(), 1.0)])
            .expect("seed");
        engine.persist().expect("persist");
        let engine = Arc::new(engine);
        let processor = ChatProcessor::new(&engine, NluPipeline::new());
        let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        let me = AkhAcpAgent::new(engine, Arc::new(Mutex::new(agent)), processor);

        let sock = dir.path().join("akh-acp-query.sock");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let res = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(
                &sock,
                "_akh/query/sparql",
                serde_json::json!({ "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 50" }),
            ),
        )
        .await
        .expect("query timed out")
        .expect("query ok");

        assert!(
            res["count"].as_u64().unwrap() >= 1,
            "expected ≥1 SPARQL solution for the seeded triple: {res}"
        );
        assert!(res["solutions"].as_array().is_some_and(|s| !s.is_empty()));

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The memory group: `_akh/memory/recall` returns episodic memories. With no
    /// query it falls back to a recency scan; an unresolvable query symbol is
    /// reported in `unresolved` (so the caller can distinguish a typo from an
    /// empty memory). Seeds one episode via `index_cycle_episode` — the same
    /// path the OODA loop uses, so what we recall is what the agent would.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_memory_recall() {
        use crate::agent::memory::index_cycle_episode;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .expect("engine");

        let predicates = crate::agent::AgentPredicates::init(&engine).expect("predicates");
        index_cycle_episode(
            &engine,
            &predicates,
            1,
            "learn about stars",
            None,
            "memory_recall",
            "made progress",
        )
        .expect("seed episode");

        let engine = Arc::new(engine);
        let processor = ChatProcessor::new(&engine, NluPipeline::new());
        let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        let me = AkhAcpAgent::new(engine, Arc::new(Mutex::new(agent)), processor);

        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-memory.sock");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // No query → recency fallback returns the seeded episode.
        let recent = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/memory/recall", serde_json::json!({ "top_k": 5 })),
        )
        .await
        .expect("recall timed out")
        .expect("recall ok");
        assert_eq!(recent["mode"], "recent");
        assert!(
            recent["count"].as_u64().unwrap() >= 1,
            "expected ≥1 recent episode: {recent}"
        );
        assert!(
            recent["episodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["summary"].as_str().is_some_and(|s| s.contains("memory_recall"))),
            "seeded episode summary missing: {recent}"
        );

        // Unresolvable query → reported in `unresolved`, recency fallback still runs.
        let typo = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(
                &sock,
                "_akh/memory/recall",
                serde_json::json!({ "query": "no_such_symbol_xyz" }),
            ),
        )
        .await
        .expect("recall timed out")
        .expect("recall ok");
        assert_eq!(typo["mode"], "recent", "unresolved query falls back to recency");
        assert!(
            typo["unresolved"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == "no_such_symbol_xyz"),
            "unresolved symbol should be reported: {typo}"
        );

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The infer group: `_akh/infer/activate` runs spreading-activation from a
    /// seed and returns activated symbols with scores. An all-unresolvable seed
    /// set is a clean error (no recency fallback, unlike recall).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_infer_activate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-infer.sock");

        // workspace() seeds ("Sun", "is-a", "Star").
        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // Seeding from "Sun" activates at least one symbol (cf. integration's
        // inference test on the same Sun/is-a/Star shape).
        let act = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(
                &sock,
                "_akh/infer/activate",
                serde_json::json!({ "seeds": "Sun", "max_depth": 1, "top_k": 10 }),
            ),
        )
        .await
        .expect("infer timed out")
        .expect("infer ok");
        assert!(
            act["count"].as_u64().unwrap() >= 1,
            "expected ≥1 activation from Sun: {act}"
        );
        assert!(
            act["activations"]
                .as_array()
                .unwrap()
                .iter()
                .all(|a| a["symbol"].is_string() && a["score"].is_number()),
            "each activation must carry a label + score: {act}"
        );

        // No resolvable seed → clean error, not an empty success.
        let bad = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(
                &sock,
                "_akh/infer/activate",
                serde_json::json!({ "seeds": "no_such_symbol_xyz" }),
            ),
        )
        .await
        .expect("infer timed out");
        assert!(bad.is_err(), "all-unresolvable seeds must error: {bad:?}");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The permission group: `_akh/permission/mode` reports the active mode and
    /// sets a new one (agent-locked via spawn_blocking). Default is bypass;
    /// an unknown mode is a clean error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_permission_mode_get_set() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-perm.sock");
        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let call = |params: serde_json::Value| {
            let sock = sock.clone();
            async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    ext_request_unix(&sock, "_akh/permission/mode", params),
                )
                .await
                .expect("ext request timed out")
                .expect("ext request ok")
            }
        };

        // GET (no mode param) → default bypass.
        let got = call(serde_json::json!({})).await;
        assert_eq!(got["mode"], "bypass");

        // SET → plan, echoed back.
        let set = call(serde_json::json!({ "mode": "plan" })).await;
        assert_eq!(set["mode"], "plan");

        // GET again → the change persisted on the agent.
        let got2 = call(serde_json::json!({})).await;
        assert_eq!(got2["mode"], "plan", "mode should persist: {got2}");

        // Unknown mode → clean error (not a silent no-op).
        let bad = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/permission/mode", serde_json::json!({ "mode": "nope" })),
        )
        .await
        .expect("timed out");
        assert!(bad.is_err(), "unknown permission mode must error");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// The permission bridge relays a blocking `ask` (from inside the tool
    /// path) to an async approver and returns its decision — the spawn_blocking
    /// ↔ async hop that lets a tool prompt the client mid-turn.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn permission_asker_bridge_relays_decision() {
        use crate::agent::permission::PermissionAsker;
        use crate::agent::tool_manifest::{
            Capability, DangerInfo, DangerLevel, ToolManifest, ToolSource,
        };
        use std::collections::HashSet;

        fn manifest(name: &str) -> ToolManifest {
            ToolManifest {
                name: name.into(),
                description: "t".into(),
                parameters: vec![],
                danger: DangerInfo {
                    level: DangerLevel::Dangerous,
                    capabilities: HashSet::from([Capability::WriteFilesystem]),
                    description: "t".into(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel::<PermissionAskRequest>(4);
        // Simulated client approver: approve "edit", reject everything else.
        let approver = tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                let _ = req.reply.send(req.tool_name == "edit");
            }
        });

        let asker = Arc::new(AcpPermissionAsker { tx });

        let a = asker.clone();
        let approved_edit =
            tokio::task::spawn_blocking(move || a.ask(&manifest("edit"), "path=x"))
                .await
                .unwrap();
        let a = asker.clone();
        let approved_shell =
            tokio::task::spawn_blocking(move || a.ask(&manifest("shell_exec"), "cmd=y"))
                .await
                .unwrap();

        assert!(approved_edit, "edit should be approved");
        assert!(!approved_shell, "shell_exec should be rejected");

        drop(asker); // close the channel so the approver task ends
        let _ = approver.await;
    }

    /// The structured mapping routes visible answers to AgentMessageChunk and
    /// introspective content to AgentThoughtChunk.
    #[test]
    fn to_session_update_routes_variants() {
        let is_msg = |u: SessionUpdate| matches!(u, SessionUpdate::AgentMessageChunk(_));
        let is_thought = |u: SessionUpdate| matches!(u, SessionUpdate::AgentThoughtChunk(_));

        assert!(is_msg(to_session_update(&AkhMessage::Fact {
            text: "the Sun is a star".into(),
            confidence: Some(0.9),
            provenance: None,
        })));
        assert!(is_msg(to_session_update(&AkhMessage::Narrative {
            text: "hello".into(),
            grammar: "default".into(),
        })));
        assert!(is_msg(to_session_update(&AkhMessage::Error {
            code: "x".into(),
            message: "boom".into(),
            help: None,
        })));
        assert!(is_thought(to_session_update(&AkhMessage::Reasoning {
            step: "consider".into(),
            expression: None,
        })));
        assert!(is_thought(to_session_update(&AkhMessage::Gap {
            entity: "E".into(),
            description: "missing".into(),
        })));
    }

    // ── control-plane routing (the ServerControl seam) ──────────────────────
    //
    // These exercise the *router*: that `_akh/{workspace,daemon,config}/*` is
    // peeled off and forwarded to the wired backend, that the per-workspace
    // data-plane still answers alongside it, and that `meta/capabilities`
    // reflects whether a backend is present. The real `ServerControl` impl over
    // akhomed's `ServerState` is tested in that binary; here a mock stands in.

    /// A stand-in [`ServerControl`] backend: it never touches a real
    /// `ServerState`, just returns canned JSON keyed by the control method (and
    /// echoes a param, to prove params flow through the router).
    struct MockControl;

    impl ServerControl for MockControl {
        fn dispatch<'a>(
            &'a self,
            method: &'a str,
            params: &'a serde_json::Value,
        ) -> ControlFuture<'a> {
            let op = method.strip_prefix("_akh/").unwrap_or(method).to_string();
            let params = params.clone();
            Box::pin(async move {
                match op.as_str() {
                    "workspace/list" => Ok(serde_json::json!({ "workspaces": ["alpha", "beta"] })),
                    "workspace/create" => Ok(serde_json::json!({
                        "name": params.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "created": true,
                    })),
                    "daemon/status" => Ok(serde_json::json!({ "running": false })),
                    "config/get" => Ok(serde_json::json!({ "config": { "mock": true } })),
                    other => Err(format!("mock backend has no op: {other}")),
                }
            })
        }
    }

    /// With a control backend wired, the `_akh/{workspace,daemon,config}/*`
    /// methods route to it; the per-workspace data-plane still answers its own
    /// methods; and `_akh/meta/capabilities` advertises the control groups.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_control_plane_routes_to_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-control.sock");
        let me = workspace().with_control(Arc::new(MockControl));
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let call = |method: &'static str, params: serde_json::Value| {
            let sock = sock.clone();
            async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(60),
                    ext_request_unix(&sock, method, params),
                )
                .await
                .expect("ext request timed out")
            }
        };

        // control-plane: workspace/list → backend's canned answer.
        let list = call("_akh/workspace/list", serde_json::json!({}))
            .await
            .expect("list ok");
        assert_eq!(list["workspaces"][0], "alpha");

        // control-plane: params flow through the router to the backend.
        let created = call("_akh/workspace/create", serde_json::json!({ "name": "gamma" }))
            .await
            .expect("create ok");
        assert_eq!(created["name"], "gamma");
        assert_eq!(created["created"], true);

        // control-plane: daemon/status routes (not mistaken for data-plane).
        let st = call("_akh/daemon/status", serde_json::json!({}))
            .await
            .expect("status ok");
        assert_eq!(st["running"], false);

        // the per-workspace data-plane still works alongside the control-plane.
        let health = call("_akh/health", serde_json::json!({}))
            .await
            .expect("health ok");
        assert_eq!(health["status"], "ok");

        // meta/capabilities advertises the control groups + methods.
        let caps = call("_akh/meta/capabilities", serde_json::json!({}))
            .await
            .expect("caps ok");
        assert_eq!(caps["control_plane"], true);
        let methods = caps["methods"].as_array().unwrap();
        assert!(methods.iter().any(|m| m == "_akh/workspace/list"));
        assert!(methods.iter().any(|m| m == "_akh/daemon/start"));
        assert!(methods.iter().any(|m| m == "_akh/config/set"));
        let groups = caps["groups"].as_array().unwrap();
        assert!(groups.iter().any(|g| g == "daemon"));
        assert!(groups.iter().any(|g| g == "config"));

        // a control method the backend doesn't implement → clean error.
        let err = call("_akh/config/set", serde_json::json!({})).await;
        assert!(err.is_err(), "unimplemented control op must error: {err:?}");

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Without a control backend, the control-plane methods return a clean
    /// "no backend" error (never silently mis-routed to the data-plane), and
    /// `_akh/meta/capabilities` omits the control groups + methods.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_control_plane_absent_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-nocontrol.sock");
        let me = workspace(); // no .with_control(..)
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // control method without a backend → error, not a data-plane fall-through.
        let err = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/workspace/list", serde_json::json!({})),
        )
        .await
        .expect("request timed out");
        assert!(err.is_err(), "control method must error with no backend: {err:?}");

        // capabilities omit the control surface.
        let caps = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(&sock, "_akh/meta/capabilities", serde_json::json!({})),
        )
        .await
        .expect("caps timed out")
        .expect("caps ok");
        assert_eq!(caps["control_plane"], false);
        let methods = caps["methods"].as_array().unwrap();
        assert!(!methods.iter().any(|m| m == "_akh/workspace/list"));
        let groups = caps["groups"].as_array().unwrap();
        assert!(!groups.iter().any(|g| g == "daemon"));

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }
}
