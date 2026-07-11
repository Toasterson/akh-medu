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
    /// Watchdog for a prompt turn: if the turn emits **nothing** (no answer
    /// delta, no reasoning, no tool step) for this long, the prompt handler
    /// stops waiting, sends the client an `error[acp]` message, and ends the
    /// turn — so a wedged step (a blocked lock, an unanswered client
    /// round-trip, a hung backend) returns an error instead of pinning the
    /// gateway slot forever. Zero disables the watchdog. Defaults from
    /// `AKH_TURN_IDLE_TIMEOUT_SECS` (see [`turn_idle_timeout_from_env`]).
    turn_idle_timeout: std::time::Duration,
}

/// Default prompt-turn idle watchdog (seconds). Generous: a healthy turn
/// streams progress messages (answer deltas, tool steps, audit lines) far more
/// often than this; only a genuinely stuck turn is silent for minutes.
const DEFAULT_TURN_IDLE_TIMEOUT_SECS: u64 = 300;

/// The prompt-turn idle watchdog from `AKH_TURN_IDLE_TIMEOUT_SECS` (seconds;
/// `0` disables), defaulting to [`DEFAULT_TURN_IDLE_TIMEOUT_SECS`].
fn turn_idle_timeout_from_env() -> std::time::Duration {
    let secs = std::env::var("AKH_TURN_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_TURN_IDLE_TIMEOUT_SECS);
    std::time::Duration::from_secs(secs)
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
            turn_idle_timeout: turn_idle_timeout_from_env(),
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
            turn_idle_timeout: turn_idle_timeout_from_env(),
        }
    }

    /// Override the prompt-turn idle watchdog (zero disables). Tests use a tiny
    /// value to exercise the stall path; production reads
    /// `AKH_TURN_IDLE_TIMEOUT_SECS` at construction instead.
    #[must_use]
    pub fn with_turn_idle_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.turn_idle_timeout = timeout;
        self
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
    engine: &Arc<Engine>,
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
                // Phase 29 capability-exposure groups (plan
                // docs/ai/plans/2026-03-26-phase29-capability-exposure.md, ADR 039).
                "microtheory", "psyche", "argue", "explain", "trust",
                // ADR 081 Slice 9 — schema-coverage report.
                "schema",
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
                // Phase 29a — microtheory (compartment) management.
                "_akh/microtheory/list",
                "_akh/microtheory/create",
                "_akh/microtheory/inspect",
                "_akh/microtheory/activate",
                "_akh/microtheory/deactivate",
                // Phase 29b — skillpack lifecycle.
                "_akh/skills/inspect",
                "_akh/skills/activate",
                "_akh/skills/deactivate",
                // Phase 29c — psyche inspection + archetype tuning.
                "_akh/psyche",
                "_akh/psyche/archetypes",
                // Phase 29d — argumentation.
                "_akh/argue",
                // Phase 29e — transparent reasoning traces.
                "_akh/explain",
                "_akh/provenance",
                // Phase 29h — evidence assessment.
                "_akh/evidence/assess",
                // Phase 29i — source trust.
                "_akh/trust",
                // ADR 081 Slice 9 — schema-coverage report.
                "_akh/schema/coverage",
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

        // ── microtheory group (Phase 29a — capability exposure over ACP; plan
        //    docs/ai/plans/2026-03-26-phase29-capability-exposure.md, ADR 039).
        //    Compartments (microtheories) are engine-side state guarded by the
        //    CompartmentManager's own locks, so no agent lock is needed. A
        //    workspace without a data_dir has no manager — clean error. ─────
        "microtheory/list" => {
            let mgr = compartment_manager(engine)?;
            let compartments: Vec<serde_json::Value> = mgr
                .all_compartments()
                .into_iter()
                .map(|(manifest, state, triples)| {
                    serde_json::json!({
                        "id": manifest.id,
                        "name": manifest.name,
                        "kind": format!("{:?}", manifest.kind),
                        "description": manifest.description,
                        "parent": manifest.parent,
                        "tags": manifest.tags,
                        "state": state.to_string(),
                        "triples": triples,
                    })
                })
                .collect();
            Ok(serde_json::json!({
                "workspace": workspace,
                "count": compartments.len(),
                "microtheories": compartments,
            }))
        }

        "microtheory/create" => {
            let mgr = compartment_manager(engine)?;
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            let name = params
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(id);
            let kind_str = params
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("project");
            let kind = parse_compartment_kind(kind_str)?;
            let manifest = crate::compartment::CompartmentManifest {
                id: id.to_string(),
                name: name.to_string(),
                kind,
                description: params
                    .get("description")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                triples_file: None,
                rules_file: None,
                grammar_ref: None,
                tags: params
                    .get("tags")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(serde_json::Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                parent: params
                    .get("parent")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                scope: Default::default(),
            };
            mgr.create_compartment(manifest)
                .map_err(|e| format!("create microtheory failed: {e}"))?;
            Ok(serde_json::json!({
                "id": id,
                "name": name,
                "kind": kind_str,
                "state": "Dormant",
                "created": true,
            }))
        }

        "microtheory/inspect" => {
            let mgr = compartment_manager(engine)?;
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            let limit = params
                .get("limit")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(50) as usize;
            let (manifest, state, triple_count) = mgr
                .all_compartments()
                .into_iter()
                .find(|(m, _, _)| m.id == id)
                .ok_or_else(|| format!("unknown microtheory \"{id}\""))?;
            // Sample the compartment's triples (label-projected, capped).
            let triples: Vec<serde_json::Value> = engine
                .all_triples()
                .into_iter()
                .filter(|t| t.compartment_id.as_deref() == Some(id))
                .take(limit)
                .map(|t| {
                    serde_json::json!({
                        "subject": engine.resolve_label(t.subject),
                        "predicate": engine.resolve_label(t.predicate),
                        "object": engine.resolve_label(t.object),
                        "confidence": t.confidence,
                    })
                })
                .collect();
            Ok(serde_json::json!({
                "id": manifest.id,
                "name": manifest.name,
                "kind": format!("{:?}", manifest.kind),
                "description": manifest.description,
                "parent": manifest.parent,
                "tags": manifest.tags,
                "state": state.to_string(),
                "triples": triple_count,
                "sample": triples,
            }))
        }

        "microtheory/activate" => {
            let mgr = compartment_manager(engine)?;
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            mgr.load(id, engine)
                .map_err(|e| format!("load microtheory \"{id}\" failed: {e}"))?;
            mgr.activate(id)
                .map_err(|e| format!("activate microtheory \"{id}\" failed: {e}"))?;
            let triples = mgr
                .all_compartments()
                .into_iter()
                .find(|(m, _, _)| m.id == id)
                .map(|(_, _, n)| n)
                .unwrap_or(0);
            Ok(serde_json::json!({ "id": id, "state": "Active", "triples": triples }))
        }

        "microtheory/deactivate" => {
            let mgr = compartment_manager(engine)?;
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            // `unload: true` additionally removes the compartment's triples from
            // the KG (back to Dormant); default just stops OODA influence.
            let unload = params
                .get("unload")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            mgr.deactivate(id)
                .map_err(|e| format!("deactivate microtheory \"{id}\" failed: {e}"))?;
            let state = if unload {
                mgr.unload(id, engine)
                    .map_err(|e| format!("unload microtheory \"{id}\" failed: {e}"))?;
                "Dormant"
            } else {
                "Loaded"
            };
            Ok(serde_json::json!({ "id": id, "state": state }))
        }

        // ── schema group (ADR 081 Slice 9 — migration: grandfather +
        //    schema-coverage report). Pure engine read (`&self`), no agent
        //    lock: classifies the live triple set into typed-core / frontier
        //    / schema-infra without touching data (see
        //    `crate::schema::coverage` module docs). ─────────────────────
        "schema/coverage" => {
            let top_n = params
                .get("top_n")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(crate::schema::coverage::DEFAULT_TOP_FRONTIER);
            let report = crate::schema::coverage_report_top_n(engine, top_n);
            serde_json::to_value(&report).map_err(|e| e.to_string())
        }

        // ── skillpack lifecycle (Phase 29b). `_akh/skills` (list) already
        //    exists above; these add inspect/activate/deactivate. All engine
        //    methods (`&self`), no agent lock. ─────────────────────────────
        "skills/inspect" => {
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            let info = engine
                .skill_info(id)
                .map_err(|e| format!("unknown skill \"{id}\": {e}"))?;
            let actions: Vec<serde_json::Value> = engine
                .skill_actions(id)
                .into_iter()
                .map(|a| serde_json::json!({ "action": a.action, "description": a.description }))
                .collect();
            let plans: Vec<serde_json::Value> = engine
                .skill_plans(id)
                .into_iter()
                .map(|p| serde_json::json!({ "goal": p.goal, "steps": p.steps }))
                .collect();
            Ok(serde_json::json!({
                "id": info.id,
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "state": format!("{:?}", info.state),
                "domains": info.domains,
                "actions": actions,
                "plans": plans,
            }))
        }

        "skills/activate" => {
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            let activation = engine
                .load_skill(id)
                .map_err(|e| format!("activate skill \"{id}\" failed: {e}"))?;
            Ok(serde_json::json!({
                "id": activation.skill_id,
                "state": "Hot",
                "triples_loaded": activation.triples_loaded,
                "rules_loaded": activation.rules_loaded,
                "actions_loaded": activation.actions_loaded,
                "plans_loaded": activation.plans_loaded,
                "memory_bytes": activation.memory_bytes,
            }))
        }

        "skills/deactivate" => {
            let id = params
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: id".to_string())?;
            engine
                .unload_skill(id)
                .map_err(|e| format!("deactivate skill \"{id}\" failed: {e}"))?;
            Ok(serde_json::json!({ "id": id, "state": "Cold" }))
        }

        // ── argue (Phase 29d). Pro/con argumentation over the KG's provenance
        //    chains with meta-rule scoring; read-only engine work. ──────────
        "argue" => {
            let subject = params
                .get("subject")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: subject".to_string())?;
            let predicate = params
                .get("predicate")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: predicate".to_string())?;
            let s = engine
                .resolve_symbol(subject)
                .map_err(|e| format!("unknown subject \"{subject}\": {e}"))?;
            let p = engine
                .resolve_symbol(predicate)
                .map_err(|e| format!("unknown predicate \"{predicate}\": {e}"))?;
            let set = engine
                .argue(s, p)
                .map_err(|e| format!("argumentation failed: {e}"))?;
            let v = &set.verdict;
            Ok(serde_json::json!({
                "subject": engine.resolve_label(set.subject),
                "predicate": engine.resolve_label(set.predicate),
                "candidates": set.candidates.len(),
                "verdict": {
                    "answer": v.answer.map(|id| engine.resolve_label(id)),
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "decisive_rule": v.decisive_rule.map(|r| format!("{r:?}")),
                    "pro": v.pro.iter().map(|a| argument_json(engine, a)).collect::<Vec<_>>(),
                    "con": v.con.iter().map(|a| argument_json(engine, a)).collect::<Vec<_>>(),
                },
            }))
        }

        // ── explain group (Phase 29e). Provenance-to-prose derivation traces
        //    (src/agent/explain.rs) + raw provenance records. ───────────────
        "explain" => {
            use crate::agent::explain;
            // Either a free-text `question` ("why is the sky blue?") routed via
            // ExplanationQuery::parse, or a bare `subject` (entity label).
            if let Some(q) = params.get("question").and_then(serde_json::Value::as_str) {
                let query = explain::ExplanationQuery::parse(q).ok_or_else(|| {
                    format!(
                        "could not parse an explanation question from {q:?} — try \
                         \"why <entity>?\", \"what do you know about <entity>?\", or \
                         \"how confident are you about <entity>?\" (or pass `subject`)"
                    )
                })?;
                let prose = explain::execute_query(engine, &query, None)
                    .map_err(|e| format!("explain failed: {e}"))?;
                Ok(serde_json::json!({ "question": q, "prose": prose }))
            } else {
                let subject = params
                    .get("subject")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| "missing required param: subject (or question)".to_string())?;
                let max_depth = params
                    .get("max_depth")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(5) as usize;
                let prose = explain::explain_entity(engine, subject)
                    .map_err(|e| format!("explain failed: {e}"))?;
                let id = engine
                    .resolve_symbol(subject)
                    .map_err(|e| format!("unknown symbol \"{subject}\": {e}"))?;
                let tree = explain::build_derivation_tree(engine, id, max_depth)
                    .map_err(|e| format!("derivation tree failed: {e}"))?;
                Ok(serde_json::json!({
                    "subject": subject,
                    "prose": prose,
                    "derivation": derivation_node_json(&tree),
                }))
            }
        }

        "provenance" => {
            let symbol = params
                .get("symbol")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "missing required param: symbol".to_string())?;
            let id = engine
                .resolve_symbol(symbol)
                .map_err(|e| format!("unknown symbol \"{symbol}\": {e}"))?;
            let records: Vec<serde_json::Value> = engine
                .provenance_of(id)
                .map_err(|e| format!("provenance lookup failed: {e}"))?
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "derived": engine.resolve_label(r.derived_id),
                        "kind": crate::agent::explain::derivation_kind_prose(&r.kind),
                        "confidence": r.confidence,
                        "depth": r.depth,
                        "timestamp": r.timestamp,
                        "sources": r
                            .sources
                            .iter()
                            .map(|s| engine.resolve_label(*s))
                            .collect::<Vec<_>>(),
                    })
                })
                .collect();
            Ok(serde_json::json!({
                "symbol": engine.resolve_label(id),
                "count": records.len(),
                "records": records,
            }))
        }

        other => Err(format!("unknown _akh data-plane method: _akh/{other}")),
    }
}

/// The workspace's [`CompartmentManager`], or a clean error for workspaces
/// without a data_dir (compartments are disk-backed).
///
/// [`CompartmentManager`]: crate::compartment::CompartmentManager
fn compartment_manager(engine: &Engine) -> Result<&crate::compartment::CompartmentManager, String> {
    engine.compartments().ok_or_else(|| {
        "this workspace has no compartment manager (no data_dir configured) — \
         microtheories require a persistent workspace"
            .to_string()
    })
}

/// Parse a wire compartment-kind string into a [`CompartmentKind`].
///
/// [`CompartmentKind`]: crate::compartment::CompartmentKind
fn parse_compartment_kind(s: &str) -> Result<crate::compartment::CompartmentKind, String> {
    use crate::compartment::CompartmentKind;
    match s.trim().to_ascii_lowercase().as_str() {
        "core" => Ok(CompartmentKind::Core),
        "skill" => Ok(CompartmentKind::Skill),
        "project" => Ok(CompartmentKind::Project),
        "library" => Ok(CompartmentKind::Library),
        "person" => Ok(CompartmentKind::Person),
        "workspace" => Ok(CompartmentKind::Workspace),
        other => Err(format!(
            "unknown microtheory kind \"{other}\" \
             (expected core|skill|project|library|person|workspace)"
        )),
    }
}

/// Label-anchored JSON projection of one [`Argument`] — the conclusion, its
/// polarity/strength, and the provenance-chain shape (not the raw records; the
/// `_akh/provenance` method serves those).
///
/// [`Argument`]: crate::argumentation::Argument
fn argument_json(engine: &Engine, a: &crate::argumentation::Argument) -> serde_json::Value {
    serde_json::json!({
        "conclusion": engine.resolve_label(a.conclusion),
        "polarity": format!("{:?}", a.polarity),
        "strength": a.strength,
        "monotonic": a.monotonic,
        "chain_len": a.chain.len(),
    })
}

/// Recursive JSON projection of a [`DerivationNode`] tree (labels + prose
/// derivation kinds, so the client needs no symbol table).
///
/// [`DerivationNode`]: crate::agent::explain::DerivationNode
fn derivation_node_json(node: &crate::agent::explain::DerivationNode) -> serde_json::Value {
    serde_json::json!({
        "label": node.label,
        "confidence": node.confidence,
        "depth": node.depth,
        "derivation": node
            .derivation
            .as_ref()
            .map(crate::agent::explain::derivation_kind_prose),
        "children": node.children.iter().map(derivation_node_json).collect::<Vec<_>>(),
    })
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
    params: &serde_json::Value,
) -> Option<Result<serde_json::Value, String>> {
    let op = method.strip_prefix("_akh/").unwrap_or(method);
    match op {
        "goals" | "tools" | "projects" | "status" | "usage" | "reasoning" | "evidence"
        | "psyche" | "evidence/assess" | "trust" => {}
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
        // Phase 29c — psyche inspection (read-only; archetype tuning is the
        // agent-mutating `_akh/psyche/archetypes`, handled in the dispatch
        // closure like `_akh/permission/mode`).
        "psyche" => serde_json::json!({
            "workspace": workspace,
            "stale": stale,
            "psyche": a.and_then(|ag| ag.psyche().map(psyche_json)),
        }),
        // Phase 29h — Dempster-Shafer assessment of one claim. An unresolvable
        // claim label is a clean error, not an empty success.
        "evidence/assess" => {
            let claim = match params.get("claim").and_then(serde_json::Value::as_str) {
                Some(c) => c,
                None => return Some(Err("missing required param: claim".to_string())),
            };
            let claim_id = match engine.resolve_symbol(claim) {
                Ok(id) => id,
                Err(e) => return Some(Err(format!("unknown claim \"{claim}\": {e}"))),
            };
            serde_json::json!({
                "workspace": workspace,
                "stale": stale,
                "assessment": a.map(|ag| {
                    let c = ag.evidence_manager().assess_claim(claim_id);
                    serde_json::json!({
                        "claim": engine.resolve_label(c.claim_entity),
                        "verdict": c.verdict.as_label(),
                        "belief": c.interval.belief,
                        "plausibility": c.interval.plausibility,
                        "conflict": c.conflict_degree,
                        "evidence_count": c.evidence_count,
                        "reasoning": c.reasoning,
                    })
                }),
            })
        }
        // Phase 29i — source-trust leaderboard (Admiralty + Bayesian trust).
        "trust" => {
            let top_k = params
                .get("top_k")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(10) as usize;
            let project = |entries: Vec<(u64, f32, crate::agent::source_reliability::SourceReliability)>| {
                entries
                    .into_iter()
                    .filter_map(|(raw, trust, rating)| {
                        let label = crate::symbol::SymbolId::new(raw)
                            .map(|id| engine.resolve_label(id))?;
                        Some(serde_json::json!({
                            "source": label,
                            "trust": trust,
                            "rating": rating.as_label(),
                        }))
                    })
                    .collect::<Vec<_>>()
            };
            serde_json::json!({
                "workspace": workspace,
                "stale": stale,
                "most_reliable": a
                    .map(|ag| project(ag.reliability_manager().most_reliable(top_k)))
                    .unwrap_or_default(),
                "least_reliable": a
                    .map(|ag| project(ag.reliability_manager().least_reliable(top_k)))
                    .unwrap_or_default(),
            })
        }
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

/// JSON projection of the agent's [`Psyche`] (Phase 29c): persona, archetype
/// weights, OCEAN profile, integration metrics, and shadow-pattern names (the
/// patterns themselves stay server-side — only their names + severities are
/// surfaced, enough for "why did you hesitate?" transparency).
///
/// [`Psyche`]: crate::compartment::psyche::Psyche
fn psyche_json(p: &crate::compartment::psyche::Psyche) -> serde_json::Value {
    let pattern = |sp: &crate::compartment::psyche::ShadowPattern| {
        serde_json::json!({ "name": sp.name, "severity": sp.severity, "explanation": sp.explanation })
    };
    serde_json::json!({
        "persona": {
            "name": p.persona.name,
            "grammar": p.persona.grammar_preference,
            "traits": p.persona.traits,
            "tone": p.persona.tone,
        },
        "archetypes": archetypes_json(&p.archetypes),
        "dominant_archetype": p.dominant_archetype(),
        "ocean": p.ocean.as_ref().map(|o| serde_json::json!({
            "openness": o.openness,
            "conscientiousness": o.conscientiousness,
            "extraversion": o.extraversion,
            "agreeableness": o.agreeableness,
            "neuroticism": o.neuroticism,
        })),
        "integration": {
            "individuation_level": p.self_integration.individuation_level,
            "shadow_encounters": p.self_integration.shadow_encounters,
            "rebalance_count": p.self_integration.rebalance_count,
        },
        "shadow": {
            "veto_patterns": p.shadow.veto_patterns.iter().map(pattern).collect::<Vec<_>>(),
            "bias_patterns": p.shadow.bias_patterns.iter().map(pattern).collect::<Vec<_>>(),
        },
        "awakened": p.awakened,
        "ennead_role": p.ennead_role,
    })
}

/// JSON projection of [`ArchetypeWeights`] (shared by `_akh/psyche` and the
/// `_akh/psyche/archetypes` GET/SET echo).
///
/// [`ArchetypeWeights`]: crate::compartment::psyche::ArchetypeWeights
fn archetypes_json(w: &crate::compartment::psyche::ArchetypeWeights) -> serde_json::Value {
    serde_json::json!({
        "healer": w.healer,
        "sage": w.sage,
        "guardian": w.guardian,
        "explorer": w.explorer,
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
    /// A `std` sync channel so the blocking asker can wait with
    /// [`recv_timeout`](std::sync::mpsc::Receiver::recv_timeout) — a client
    /// that never answers `session/request_permission` must deny (fail
    /// closed), not wedge the turn forever.
    reply: std::sync::mpsc::SyncSender<bool>,
}

/// Default bound on one `session/request_permission` round-trip (seconds).
/// Long enough for a human to answer an interactive prompt; short enough that
/// a client which does not implement the method (e.g. a headless gateway)
/// denies instead of hanging the chat turn. `AKH_PERMISSION_ASK_TIMEOUT_SECS`
/// overrides; `0` disables the bound.
const DEFAULT_PERMISSION_ASK_TIMEOUT_SECS: u64 = 120;

/// The permission-ask bound from `AKH_PERMISSION_ASK_TIMEOUT_SECS`.
fn permission_ask_timeout() -> std::time::Duration {
    let secs = std::env::var("AKH_PERMISSION_ASK_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(DEFAULT_PERMISSION_ASK_TIMEOUT_SECS);
    std::time::Duration::from_secs(secs)
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
        let (reply, reply_rx) = std::sync::mpsc::sync_channel(1);
        let req = PermissionAskRequest {
            tool_name: manifest.name.clone(),
            danger: manifest.danger.level.to_string(),
            input_summary: input_summary.to_string(),
            reply,
        };
        // We are inside spawn_blocking, so blocking_send / a blocking recv are
        // correct (never call these on an async runtime thread).
        if self.tx.blocking_send(req).is_err() {
            return false;
        }
        // Bound the wait: a client that never answers the permission request
        // (or does not implement it) denies instead of wedging the turn.
        let bound = permission_ask_timeout();
        if bound.is_zero() {
            return reply_rx.recv().unwrap_or(false);
        }
        match reply_rx.recv_timeout(bound) {
            Ok(approved) => approved,
            Err(_) => {
                tracing::warn!(
                    tool = %manifest.name,
                    timeout_secs = bound.as_secs(),
                    "permission ask unanswered within the bound — denying (fail closed)"
                );
                false
            }
        }
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
        // Mirror the asker-side bound: even though the blocking asker already
        // times out and denies, this task must not stay parked forever on a
        // request the client will never answer (that would wedge the prompt
        // handler's final `approver.await`).
        let bound = permission_ask_timeout();
        let response = cx.send_request(request).block_task();
        let outcome = if bound.is_zero() {
            response.await.ok()
        } else {
            match tokio::time::timeout(bound, response).await {
                Ok(resp) => resp.ok(),
                Err(_) => {
                    tracing::warn!(
                        tool = %req.tool_name,
                        timeout_secs = bound.as_secs(),
                        "client did not answer session/request_permission — denying"
                    );
                    None
                }
            }
        };
        let approved = match outcome {
            Some(resp) => match resp.outcome {
                RequestPermissionOutcome::Selected(sel) => sel.option_id.0.as_ref() == "allow",
                _ => false, // Cancelled / future variants → deny.
            },
            None => false, // client error / timeout → deny.
        };
        // `try_send`: the asker may have timed out and gone; never block here.
        let _ = req.reply.try_send(approved);
    }
}

/// GET/SET the agent's archetype weights (Phase 29c, `_akh/psyche/archetypes`).
///
/// With no weight params this reports the current weights. With any of
/// `healer`/`sage`/`guardian`/`explorer` present it applies a **weights-only
/// delta**: the psyche is cloned, only the given weights are replaced (clamped
/// to `[0, 1]`), and the result is force-set on the agent *and* persisted via
/// the compartment manager (when the workspace has one) so the tuning survives
/// a restart. `force_set` is deliberate — plan 29c allows archetype calibration
/// within the awakened identity, and nothing outside the four weights changes.
///
/// Errors when the akh has no psyche at all (nothing to tune) or a weight is
/// not a number.
fn set_archetype_weights(
    engine: &Engine,
    agent: &mut Agent,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    const KEYS: [&str; 4] = ["healer", "sage", "guardian", "explorer"];
    let wants_set = KEYS.iter().any(|k| params.get(k).is_some());

    if wants_set {
        let mut psyche = agent.psyche().cloned().ok_or_else(|| {
            "this akh has no psyche loaded — run the ritual of awakening (or load \
             the psyche compartment) before tuning archetype weights"
                .to_string()
        })?;
        let weight = |key: &str, current: f32| -> Result<f32, String> {
            match params.get(key) {
                None => Ok(current),
                Some(v) => v
                    .as_f64()
                    .map(|f| (f as f32).clamp(0.0, 1.0))
                    .ok_or_else(|| format!("archetype weight \"{key}\" must be a number in [0,1]")),
            }
        };
        psyche.archetypes.healer = weight("healer", psyche.archetypes.healer)?;
        psyche.archetypes.sage = weight("sage", psyche.archetypes.sage)?;
        psyche.archetypes.guardian = weight("guardian", psyche.archetypes.guardian)?;
        psyche.archetypes.explorer = weight("explorer", psyche.archetypes.explorer)?;
        psyche.self_integration.rebalance_count += 1;

        // Persist to the psyche compartment first (best-effort; a workspace
        // without a data_dir has no manager), then apply on the live agent.
        if let Some(cm) = engine.compartments() {
            cm.force_set_psyche(psyche.clone());
        }
        agent.force_set_psyche(psyche);
    }

    match agent.psyche() {
        Some(p) => Ok(serde_json::json!({
            "archetypes": archetypes_json(&p.archetypes),
            "dominant_archetype": p.dominant_archetype(),
        })),
        None => Ok(serde_json::json!({ "archetypes": serde_json::Value::Null })),
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
                        // Resolution can wake the akh from its backing store, so
                        // it too rides the idle watchdog: a hung wake (e.g. a
                        // wedged Postgres connection) must error, not pin the
                        // client's slot forever.
                        let ws = me.session_workspace(&session_id.to_string());
                        let idle = me.turn_idle_timeout;
                        let resolved = if idle.is_zero() {
                            me.resolve_handles(&ws).await
                        } else {
                            match tokio::time::timeout(idle, me.resolve_handles(&ws)).await {
                                Ok(r) => r,
                                Err(_) => Err(format!(
                                    "workspace resolution stalled for {}s (wake from \
                                     the backing store may be blocked)",
                                    idle.as_secs()
                                )),
                            }
                        };
                        let (engine, agent, processor) = match resolved {
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
                            ws.clone(),
                            text,
                            Some(asker),
                            ChannelSink { tx: msg_tx },
                        );

                        // Drain the turn's messages under the idle watchdog: a
                        // healthy turn keeps emitting (deltas, tool steps, audit
                        // lines); one that goes silent for the whole window is
                        // wedged — a blocked lock, a hung backend, an unanswered
                        // client round-trip. Fail the turn instead of pinning
                        // the client's slot forever.
                        let sink = AcpSink::new(cx, session_id.clone());
                        let mut stalled = false;
                        loop {
                            let next = if idle.is_zero() {
                                msg_rx.recv().await
                            } else {
                                match tokio::time::timeout(idle, msg_rx.recv()).await {
                                    Ok(next) => next,
                                    Err(_) => {
                                        stalled = true;
                                        break;
                                    }
                                }
                            };
                            match next {
                                Some(msg) => sink.emit(&msg),
                                // Channel closed ⇒ the turn finished and dropped
                                // its sink.
                                None => break,
                            }
                        }

                        if stalled {
                            // The blocking turn thread cannot be killed; leave it
                            // behind (it holds the agent lock until whatever it is
                            // blocked on resolves) and free the client's slot with
                            // an explicit error.
                            tracing::error!(
                                workspace = %ws,
                                idle_secs = idle.as_secs(),
                                "chat turn stalled (no progress within the idle \
                                 watchdog) — ending the turn with an error; the \
                                 turn's worker thread may still be blocked"
                            );
                            sink.emit(&AkhMessage::error(
                                "acp",
                                format!(
                                    "chat turn stalled: no progress for {}s — \
                                     aborting this turn. The stuck step may still \
                                     be blocked inside the daemon; check akhomed \
                                     logs (workspace \"{ws}\").",
                                    idle.as_secs()
                                ),
                            ));
                            return responder
                                .respond(PromptResponse::new(StopReason::EndTurn));
                        }

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
                                    // Agent-mutating method (Phase 29c): archetype
                                    // tuning. GET with no weight params; SET applies
                                    // the given weights (clamped to [0,1]) as a
                                    // weights-only delta — per plan 29c the tuning
                                    // calibrates *within* the identity established
                                    // at awakening, so it deliberately bypasses the
                                    // wholesale-replace immutability guard while
                                    // touching nothing but the four weights. Runs
                                    // under the agent lock in spawn_blocking
                                    // (ADR-027), mirroring `_akh/permission/mode`.
                                    if method == "_akh/psyche/archetypes" {
                                        let params = req.params.clone();
                                        let engine = Arc::clone(&engine);
                                        let result = tokio::task::spawn_blocking(move || {
                                            let mut a =
                                                agent.lock().unwrap_or_else(|e| e.into_inner());
                                            set_archetype_weights(&engine, &mut a, &params)
                                        })
                                        .await;
                                        return match result {
                                            Ok(Ok(value)) => {
                                                responder.respond(AkhExtResponse(value))
                                            }
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

/// Grace given to a connection after the peer's read side reaches EOF, so an
/// in-flight response can still be flushed (a half-closing client). After it
/// elapses the connection future is dropped, which closes the socket and frees
/// its fds.
const POST_EOF_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// How long the accept loops sleep after a failed `accept()` before retrying.
const ACCEPT_RETRY_BACKOFF: std::time::Duration = std::time::Duration::from_millis(500);

/// A `futures` [`AsyncRead`](futures::io::AsyncRead) adapter that fires a
/// oneshot the first time the inner reader reaches EOF or errors — i.e. when
/// the peer has disconnected or shut down its write side.
struct SignalOnEof<R> {
    inner: R,
    signal: Option<tokio::sync::oneshot::Sender<()>>,
}

impl<R> SignalOnEof<R> {
    fn new(inner: R) -> (Self, tokio::sync::oneshot::Receiver<()>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (
            Self {
                inner,
                signal: Some(tx),
            },
            rx,
        )
    }
}

impl<R: futures::io::AsyncRead + Unpin> futures::io::AsyncRead for SignalOnEof<R> {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        let me = self.get_mut();
        let res = std::pin::Pin::new(&mut me.inner).poll_read(cx, buf);
        match &res {
            std::task::Poll::Ready(Ok(0)) if !buf.is_empty() => {
                if let Some(tx) = me.signal.take() {
                    let _ = tx.send(());
                }
            }
            std::task::Poll::Ready(Err(_)) => {
                if let Some(tx) = me.signal.take() {
                    let _ = tx.send(());
                }
            }
            _ => {}
        }
        res
    }
}

/// Drive one connection's protocol future until it finishes on its own, or
/// until the peer disconnects (read EOF) plus [`POST_EOF_GRACE`].
///
/// The protocol crate's `connect_to` does NOT resolve on read-EOF alone: its
/// background actors are `try_join!`ed and the outgoing/task actors keep
/// waiting for traffic that will never come, so the future pends forever.
/// Without this watchdog every disconnected peer leaks its task + socket fds
/// until the process hits EMFILE — observed in production (archibald akh-thoth,
/// 2026-06/07: ~1k closed-socket fds from the gateway's 15s discovery polls,
/// then a dead accept loop).
async fn drive_until_disconnect(
    conn: impl std::future::Future<Output = Result<(), agent_client_protocol::Error>>,
    eof: tokio::sync::oneshot::Receiver<()>,
) -> AcpResult<()> {
    tokio::pin!(conn);
    tokio::select! {
        res = &mut conn => {
            res?;
            Ok(())
        }
        _ = eof => {
            match tokio::time::timeout(POST_EOF_GRACE, &mut conn).await {
                Ok(res) => {
                    res?;
                    Ok(())
                }
                // Timed out after EOF: nothing left to read and the protocol
                // loop didn't end on its own — drop it, closing the socket.
                Err(_) => Ok(()),
            }
        }
    }
}

/// Serve the ACP agent over a tokio [`UnixStream`], adapting it into the
/// crate's `futures`-based [`ByteStreams`] transport.
///
/// The stream is split into read/write halves and bridged with
/// [`tokio_util::compat`]. This consumes the connection and runs until the
/// protocol loop ends or the peer disconnects (read EOF + a short flush
/// grace) — see [`drive_until_disconnect`].
pub async fn serve_stream(me: AkhAcpAgent, stream: UnixStream) -> AcpResult<()> {
    // tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf} → futures io via compat.
    let (read_half, write_half) = stream.into_split();
    let (incoming, eof_rx) = SignalOnEof::new(read_half.compat());
    let outgoing = write_half.compat_write();

    // ByteStreams::new(outgoing_writer, incoming_reader)
    let transport = ByteStreams::new(outgoing, incoming);

    // The agent builder is a ConnectTo<Client>; connecting it TO the transport
    // (a role-agnostic ConnectTo<Agent>) runs the agent's dispatch loop over the
    // socket, mirroring how `Stdio`/`AcpAgent` wire byte streams in the crate.
    use agent_client_protocol::ConnectTo;
    drive_until_disconnect(build_connection(me).connect_to(transport), eof_rx).await
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
    let (incoming, eof_rx) = SignalOnEof::new(read_half.compat());
    let outgoing = write_half.compat_write();

    let transport = ByteStreams::new(outgoing, incoming);

    use agent_client_protocol::ConnectTo;
    drive_until_disconnect(build_connection(me).connect_to(transport), eof_rx).await
}

/// Accept connections on a bound [`UnixListener`] until `shutdown` flips to
/// `true`, serving each on its own task.
///
/// This is the production accept loop: the cheap-to-clone [`AkhAcpAgent`]
/// (`Arc`-backed) is cloned per connection, so many clients are served
/// concurrently against the same shared engine/agent. A failed connection logs
/// and is dropped; it does not tear down the loop. A failed `accept()` (EMFILE
/// under fd exhaustion, ECONNABORTED from a half-open peer, …) logs and retries
/// after [`ACCEPT_RETRY_BACKOFF`] — returning instead would leave the daemon up
/// but permanently deaf.
pub async fn serve_unix_loop(
    me: AkhAcpAgent,
    listener: UnixListener,
    mut shutdown: watch::Receiver<bool>,
) -> AcpResult<()> {
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        let conn = me.clone();
                        tokio::spawn(async move {
                            if let Err(e) = serve_stream(conn, stream).await {
                                tracing::warn!("ACP unix connection ended with error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "ACP unix accept failed; retrying");
                        tokio::time::sleep(ACCEPT_RETRY_BACKOFF).await;
                    }
                }
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
                match accepted {
                    Ok((stream, _addr)) => {
                        let conn = me.clone();
                        tokio::spawn(async move {
                            if let Err(e) = serve_tcp_stream(conn, stream).await {
                                tracing::warn!("ACP tcp connection ended with error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "ACP tcp accept failed; retrying");
                        tokio::time::sleep(ACCEPT_RETRY_BACKOFF).await;
                    }
                }
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

    /// Regression: the production companion health-check prompt (observed on
    /// akh-thoth, develop 75086fa) hung the turn forever after a successful
    /// tier-1 NLU parse in a `server,delegation`-only build. The turn must
    /// complete — answer or error — never hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prompt_turn_completes_for_companion_health_check() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-hc.sock");

        let me = workspace();
        let server_sock = sock.clone();
        let _server = tokio::spawn(async move { serve_unix(me, &server_sock).await });

        let stream = connect_with_retry(&sock).await;
        let answer = unix_client_prompt(
            stream,
            "Health check after the ACP fd-leak fix deploy: please reply with \
             a one-line status of your knowledge graph.",
        )
        .await;
        assert!(
            !answer.trim().is_empty(),
            "expected non-empty ACP prompt response, got: {answer:?}"
        );
    }

    /// Regression: a wedged chat turn (here: the agent lock is held forever, so
    /// the turn's `spawn_blocking` closure blocks before emitting anything) must
    /// NOT hang the prompt forever — the idle watchdog ends the turn with an
    /// `error[acp]` message so the client's slot is freed. This is the guard for
    /// the akh-thoth production hang (develop 75086fa): a turn that blocked
    /// silently after its tier-1 NLU parse pinned the gateway slot for good.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stalled_prompt_turn_fails_fast_instead_of_hanging() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-stall.sock");

        let me = workspace().with_turn_idle_timeout(std::time::Duration::from_secs(2));

        // Wedge the workspace: hold the agent lock on another thread, so the
        // prompt turn blocks at `agent.lock()` and never emits a message. The
        // lock is released after the assertion (via the release flag): the
        // wedged turn thread must be allowed to finish, because dropping the
        // test runtime waits for in-flight blocking tasks.
        let blocker = Arc::clone(me.agent.as_ref().expect("held-mode agent"));
        let release = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let release_flag = Arc::clone(&release);
        let blocker_thread = std::thread::spawn(move || {
            let _guard = blocker.lock().unwrap_or_else(|e| e.into_inner());
            while !release_flag.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::park_timeout(std::time::Duration::from_millis(25));
            }
        });
        // Give the blocker thread a moment to actually take the lock.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let server_sock = sock.clone();
        let _server = tokio::spawn(async move { serve_unix(me, &server_sock).await });

        let stream = connect_with_retry(&sock).await;
        // unix_client_prompt is hard-bounded at 60s: if the watchdog does not
        // fire, this test fails by timeout instead of hanging the suite.
        let answer = unix_client_prompt(stream, "hello?").await;

        // Unwedge before asserting so the turn thread can finish even if the
        // assertion fails.
        release.store(true, std::sync::atomic::Ordering::Relaxed);
        blocker_thread.thread().unpark();
        let _ = blocker_thread.join();

        assert!(
            answer.contains("error[acp]") && answer.contains("stalled"),
            "expected the idle-watchdog error in the turn output, got: {answer:?}"
        );
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

    /// The `_akh/*` data-plane over a real **TCP** socket — exactly the path
    /// `akh --remote host:port acp ...` drives. Proves a remote client reaches a
    /// daemon's data-plane across the network, and that a `remote_tcp` discovery
    /// record routes to that endpoint. (`acp_ext_health_and_capabilities` covers
    /// the same surface over a Unix socket.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_data_plane_over_tcp() {
        let me = workspace();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind tcp");
        let addr = listener.local_addr().expect("local addr").to_string();

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_tcp_loop(me, listener, shutdown_rx).await });

        // A remote ServerInfo points the client at this TCP endpoint (no socket).
        let info = crate::client::ServerInfo::remote_tcp(&addr);
        assert_eq!(info.acp_tcp.as_deref(), Some(addr.as_str()));
        assert!(
            info.acp_socket.is_none(),
            "remote_tcp must not advertise a socket"
        );

        // _akh/meta/capabilities reaches the dispatch handler over TCP.
        let caps = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_tcp(
                info.acp_tcp.as_deref().unwrap(),
                "_akh/meta/capabilities",
                serde_json::json!({}),
            ),
        )
        .await
        .expect("ext caps over tcp timed out")
        .expect("ext caps over tcp ok");
        assert_eq!(caps["protocol"], "acp");
        assert!(caps["methods"].as_array().unwrap().iter().any(|m| m == "_akh/health"));

        // An unknown method is a clean error over TCP too — no hang/panic.
        let unknown = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_tcp(&addr, "_akh/does/not/exist", serde_json::json!({})),
        )
        .await
        .expect("ext unknown over tcp timed out");
        assert!(unknown.is_err(), "unknown _akh method over tcp must error");

        shutdown_tx.send(true).expect("send shutdown");
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

    /// Regression (archibald 2026-06/07 fd leak): the per-connection serve
    /// future must END when the peer disconnects, releasing the socket fds.
    /// The protocol crate's `connect_to` pends forever after read-EOF, so
    /// without the `drive_until_disconnect` watchdog this task lives (and
    /// holds its fds) until the process dies — 1k+ leaked fds in production.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_tcp_serve_ends_when_peer_disconnects() {
        let me = workspace();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind tcp");
        let addr = listener.local_addr().expect("local addr");

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept tcp");
            serve_tcp_stream(me, stream).await
        });

        // Connect and immediately hang up — the shape of a gateway discovery
        // poll that errors out, and of any client that finished its business.
        let stream = TcpStream::connect(addr).await.expect("connect tcp");
        drop(stream);

        let joined = tokio::time::timeout(std::time::Duration::from_secs(30), server)
            .await
            .expect(
                "serve_tcp_stream did not end after peer disconnect — \
                 the connection task (and its socket fds) would leak",
            )
            .expect("server task joined");
        joined.expect("serve returned Ok after peer disconnect");
    }

    /// Unix-socket twin of `acp_tcp_serve_ends_when_peer_disconnects`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_unix_serve_ends_when_peer_disconnects() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-eof.sock");

        let me = workspace();
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept unix");
            serve_stream(me, stream).await
        });

        let stream = connect_with_retry(&sock).await;
        drop(stream);

        let joined = tokio::time::timeout(std::time::Duration::from_secs(30), server)
            .await
            .expect(
                "serve_stream did not end after peer disconnect — \
                 the connection task (and its socket fds) would leak",
            )
            .expect("server task joined");
        joined.expect("serve returned Ok after peer disconnect");
    }

    /// Regression (archibald 2026-06/07): connection churn must not wedge the
    /// accept loop. 20 clients connect and hang up without speaking a byte
    /// (the discovery-poll storm shape); every serve task must end, and the
    /// loop must still serve a real request afterwards.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_tcp_accept_loop_survives_connect_disconnect_churn() {
        let me = workspace();
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind tcp");
        let addr = listener.local_addr().expect("local addr").to_string();

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_tcp_loop(me, listener, shutdown_rx).await });

        for _ in 0..20 {
            let stream = TcpStream::connect(&addr).await.expect("connect tcp");
            drop(stream);
        }

        // The loop still answers a real data-plane request after the churn.
        let caps = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_tcp(&addr, "_akh/meta/capabilities", serde_json::json!({})),
        )
        .await
        .expect("ext caps after churn timed out")
        .expect("ext caps after churn ok");
        assert_eq!(caps["protocol"], "acp");

        shutdown_tx.send(true).expect("send shutdown");
        let joined = tokio::time::timeout(std::time::Duration::from_secs(30), server)
            .await
            .expect("accept loop did not shut down")
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

    // ── Phase 29 capability-exposure groups (plan
    //    docs/ai/plans/2026-03-26-phase29-capability-exposure.md, ADR 039) ──

    /// Build a **persistent** in-process workspace (data_dir on disk) so the
    /// compartment + skill managers exist — the Phase 29a/29b groups need them.
    fn persistent_workspace(data_dir: &Path) -> AkhAcpAgent {
        std::fs::create_dir_all(data_dir).expect("data dir");
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(data_dir.to_path_buf()),
            ..Default::default()
        })
        .expect("persistent engine");
        let engine = Arc::new(engine);
        let processor = ChatProcessor::new(&engine, NluPipeline::new());
        let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        AkhAcpAgent::new(engine, Arc::new(Mutex::new(agent)), processor)
    }

    /// One bounded `ext_request_unix` that must succeed.
    async fn ext_ok(
        sock: &Path,
        method: &str,
        params: serde_json::Value,
    ) -> serde_json::Value {
        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(sock, method, params),
        )
        .await
        .expect("ext request timed out")
        .expect("ext request ok")
    }

    /// One bounded `ext_request_unix` whose *error* is under test.
    async fn ext_err(
        sock: &Path,
        method: &str,
        params: serde_json::Value,
    ) -> AcpResult<serde_json::Value> {
        tokio::time::timeout(
            std::time::Duration::from_secs(60),
            ext_request_unix(sock, method, params),
        )
        .await
        .expect("ext request timed out")
    }

    /// Phase 29a: the microtheory group — create → list → inspect → activate →
    /// deactivate(+unload) round-trip, capability advertisement, and clean
    /// errors for unknown ids.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_microtheory_lifecycle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let me = persistent_workspace(&dir.path().join("ws"));
        let sock = dir.path().join("akh-acp-mt.sock");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // The new groups + methods are advertised.
        let caps = ext_ok(&sock, "_akh/meta/capabilities", serde_json::json!({})).await;
        let methods = caps["methods"].as_array().unwrap();
        for m in [
            "_akh/microtheory/list",
            "_akh/skills/activate",
            "_akh/psyche/archetypes",
            "_akh/argue",
            "_akh/explain",
            "_akh/provenance",
            "_akh/evidence/assess",
            "_akh/trust",
        ] {
            assert!(methods.iter().any(|v| v == m), "capabilities missing {m}: {caps}");
        }
        assert!(caps["groups"].as_array().unwrap().iter().any(|g| g == "microtheory"));

        // create → Dormant.
        let created = ext_ok(
            &sock,
            "_akh/microtheory/create",
            serde_json::json!({ "id": "cooking", "description": "my cooking knowledge" }),
        )
        .await;
        assert_eq!(created["created"], true);
        assert_eq!(created["state"], "Dormant");

        // list contains it.
        let list = ext_ok(&sock, "_akh/microtheory/list", serde_json::json!({})).await;
        assert!(
            list["microtheories"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["id"] == "cooking"),
            "created microtheory missing from list: {list}"
        );

        // inspect echoes the manifest.
        let ins = ext_ok(
            &sock,
            "_akh/microtheory/inspect",
            serde_json::json!({ "id": "cooking" }),
        )
        .await;
        assert_eq!(ins["state"], "Dormant");
        assert_eq!(ins["description"], "my cooking knowledge");

        // activate → Active; deactivate(+unload) → Dormant.
        let act = ext_ok(
            &sock,
            "_akh/microtheory/activate",
            serde_json::json!({ "id": "cooking" }),
        )
        .await;
        assert_eq!(act["state"], "Active");
        let deact = ext_ok(
            &sock,
            "_akh/microtheory/deactivate",
            serde_json::json!({ "id": "cooking", "unload": true }),
        )
        .await;
        assert_eq!(deact["state"], "Dormant");

        // duplicate create + unknown-id inspect are clean errors.
        assert!(ext_err(&sock, "_akh/microtheory/create", serde_json::json!({ "id": "cooking" }))
            .await
            .is_err());
        assert!(ext_err(&sock, "_akh/microtheory/inspect", serde_json::json!({ "id": "nope" }))
            .await
            .is_err());
        // bad kind is a clean error too.
        assert!(ext_err(
            &sock,
            "_akh/microtheory/create",
            serde_json::json!({ "id": "x", "kind": "galaxy" })
        )
        .await
        .is_err());

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Phase 29b: the skillpack lifecycle — activate loads a disk skillpack's
    /// label triples into the KG (visible via `triples/from`), inspect shows
    /// the manifest, deactivate returns it to Cold. Unknown skills error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_skills_lifecycle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().join("ws");

        // A minimal on-disk skillpack with label-based triples.
        let skill_dir = data_dir.join("skills").join("astronomy");
        std::fs::create_dir_all(&skill_dir).expect("skill dir");
        std::fs::write(
            skill_dir.join("skill.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "id": "astronomy",
                "name": "Astronomy Pack",
                "version": "1.0.0",
                "description": "Basic astronomical knowledge",
                "domains": ["astronomy"],
                "weight_size_bytes": 0,
                "triples_file": "triples.json",
                "depends": [],
                "conflicts": [],
                "provides": ["celestial_bodies"],
                "requires": []
            }))
            .unwrap(),
        )
        .expect("skill manifest");
        std::fs::write(
            skill_dir.join("triples.json"),
            serde_json::to_string(&serde_json::json!([
                { "subject": "Sol", "predicate": "is-a", "object": "Star", "confidence": 1.0 }
            ]))
            .unwrap(),
        )
        .expect("skill triples");

        let me = persistent_workspace(&data_dir);
        let sock = dir.path().join("akh-acp-skill.sock");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // activate: Cold → Hot, its triples land in the KG.
        let act = ext_ok(&sock, "_akh/skills/activate", serde_json::json!({ "id": "astronomy" }))
            .await;
        assert_eq!(act["id"], "astronomy");
        assert!(
            act["triples_loaded"].as_u64().unwrap() >= 1,
            "expected the label triple to load: {act}"
        );

        // the loaded triple is visible through the ordinary data-plane.
        let from = ext_ok(&sock, "_akh/triples/from", serde_json::json!({ "symbol": "Sol" })).await;
        assert!(
            from["triples"].as_array().unwrap().iter().any(|t| t["object"] == "Star"),
            "skill triple not visible: {from}"
        );

        // list + inspect report the Hot skill.
        let list = ext_ok(&sock, "_akh/skills", serde_json::json!({})).await;
        assert!(
            list["skills"].as_array().unwrap().iter().any(|s| s["id"] == "astronomy"),
            "skill missing from list: {list}"
        );
        let ins = ext_ok(&sock, "_akh/skills/inspect", serde_json::json!({ "id": "astronomy" }))
            .await;
        assert_eq!(ins["name"], "Astronomy Pack");
        assert_eq!(ins["state"], "Hot");

        // deactivate: Hot → Cold.
        let de = ext_ok(&sock, "_akh/skills/deactivate", serde_json::json!({ "id": "astronomy" }))
            .await;
        assert_eq!(de["state"], "Cold");

        // unknown skill → clean error.
        assert!(ext_err(&sock, "_akh/skills/activate", serde_json::json!({ "id": "nope" }))
            .await
            .is_err());

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Phase 29c: psyche inspection + archetype tuning. GET reports the loaded
    /// psyche; SET applies clamped weights and persists across requests; an akh
    /// without a psyche reads as null and rejects tuning.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_psyche_group() {
        let dir = tempfile::tempdir().expect("tempdir");

        // ── an akh WITHOUT a psyche: read null, tuning is a clean error ──
        let bare = workspace();
        let bare_sock = dir.path().join("akh-acp-psyche0.sock");
        let listener = UnixListener::bind(&bare_sock).expect("bind unix");
        let (bare_tx, bare_rx) = watch::channel(false);
        let bare_server =
            tokio::spawn(async move { serve_unix_loop(bare, listener, bare_rx).await });

        let show = ext_ok(&bare_sock, "_akh/psyche", serde_json::json!({})).await;
        assert_eq!(show["stale"], serde_json::json!(false));
        assert_eq!(show["psyche"], serde_json::Value::Null);
        assert!(
            ext_err(&bare_sock, "_akh/psyche/archetypes", serde_json::json!({ "sage": 0.9 }))
                .await
                .is_err(),
            "tuning without a psyche must error"
        );
        bare_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), bare_server).await;

        // ── an akh WITH a psyche: full read + clamped tuning round-trip ──
        let me = workspace();
        me.agent
            .as_ref()
            .expect("held-mode agent")
            .lock()
            .unwrap()
            .force_set_psyche(crate::compartment::psyche::Psyche::default());
        let sock = dir.path().join("akh-acp-psyche.sock");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let show = ext_ok(&sock, "_akh/psyche", serde_json::json!({})).await;
        assert!(show["psyche"]["persona"]["name"].is_string(), "psyche read: {show}");
        assert!(show["psyche"]["archetypes"]["sage"].is_number());
        assert!(show["psyche"]["dominant_archetype"].is_string());

        // GET weights.
        let got = ext_ok(&sock, "_akh/psyche/archetypes", serde_json::json!({})).await;
        assert!(got["archetypes"]["healer"].is_number());

        // SET: sage=0.9 applied, explorer=1.7 clamped to 1.0.
        let set = ext_ok(
            &sock,
            "_akh/psyche/archetypes",
            serde_json::json!({ "sage": 0.9, "explorer": 1.7 }),
        )
        .await;
        assert!(
            (set["archetypes"]["sage"].as_f64().unwrap() - 0.9).abs() < 1e-6,
            "sage weight not applied: {set}"
        );
        assert!(
            (set["archetypes"]["explorer"].as_f64().unwrap() - 1.0).abs() < 1e-6,
            "explorer weight not clamped: {set}"
        );

        // The change persisted on the agent.
        let got2 = ext_ok(&sock, "_akh/psyche/archetypes", serde_json::json!({})).await;
        assert!((got2["archetypes"]["sage"].as_f64().unwrap() - 0.9).abs() < 1e-6);

        // A non-numeric weight is a clean error.
        assert!(
            ext_err(&sock, "_akh/psyche/archetypes", serde_json::json!({ "sage": "high" }))
                .await
                .is_err()
        );

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Phase 29d: `_akh/argue` marshals pro/con arguments for
    /// (subject, predicate, ?) and renders a verdict naming the winning answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_argue_verdict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-argue.sock");
        let me = workspace(); // seeds ("Sun", "is-a", "Star")
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let v = ext_ok(
            &sock,
            "_akh/argue",
            serde_json::json!({ "subject": "Sun", "predicate": "is-a" }),
        )
        .await;
        assert_eq!(v["subject"], "Sun");
        assert!(v["candidates"].as_u64().unwrap() >= 1, "expected ≥1 candidate: {v}");
        assert_eq!(v["verdict"]["answer"], "Star", "verdict should name Star: {v}");
        assert!(v["verdict"]["reasoning"].is_string());
        assert!(v["verdict"]["pro"].is_array());

        // missing predicate + unknown subject → clean errors.
        assert!(ext_err(&sock, "_akh/argue", serde_json::json!({ "subject": "Sun" }))
            .await
            .is_err());
        assert!(ext_err(
            &sock,
            "_akh/argue",
            serde_json::json!({ "subject": "NoSuchThing", "predicate": "is-a" })
        )
        .await
        .is_err());

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Phase 29e: `_akh/explain` returns prose + a derivation tree for an
    /// entity, routes free-text questions through `ExplanationQuery`, and
    /// `_akh/provenance` serves the raw records. Unknown entities error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_explain_and_provenance() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-explain.sock");
        // Persistent workspace: the provenance ledger (which explain walks)
        // requires a durable store — same requirement as the daemon's akhs.
        let me = persistent_workspace(&dir.path().join("ws"));
        me.engine
            .as_ref()
            .expect("held-mode engine")
            .ingest_label_triples(&[(
                "Sun".to_string(),
                "is-a".to_string(),
                "Star".to_string(),
                1.0,
            )])
            .expect("seed triple");
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        // entity form: prose + structured derivation tree.
        let ex = ext_ok(&sock, "_akh/explain", serde_json::json!({ "subject": "Sun" })).await;
        assert!(
            ex["prose"].as_str().unwrap().contains("Sun"),
            "explanation prose should mention the subject: {ex}"
        );
        assert_eq!(ex["derivation"]["label"], "Sun");
        assert!(ex["derivation"]["children"].is_array());

        // question form: parsed via ExplanationQuery.
        let q = ext_ok(
            &sock,
            "_akh/explain",
            serde_json::json!({ "question": "what do you know about Sun?" }),
        )
        .await;
        assert!(!q["prose"].as_str().unwrap().is_empty(), "question prose empty: {q}");

        // provenance records for the seeded symbol.
        let prov = ext_ok(&sock, "_akh/provenance", serde_json::json!({ "symbol": "Sun" })).await;
        assert_eq!(prov["symbol"], "Sun");
        assert!(prov["records"].is_array());

        // unknown entity / unparseable question / missing params → clean errors.
        assert!(ext_err(&sock, "_akh/explain", serde_json::json!({ "subject": "NoSuchThing" }))
            .await
            .is_err());
        assert!(ext_err(
            &sock,
            "_akh/explain",
            serde_json::json!({ "question": "gibberish with no pattern" })
        )
        .await
        .is_err());
        assert!(ext_err(&sock, "_akh/explain", serde_json::json!({})).await.is_err());
        assert!(ext_err(&sock, "_akh/provenance", serde_json::json!({ "symbol": "NoSuchThing" }))
            .await
            .is_err());

        shutdown_tx.send(true).expect("send shutdown");
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), server).await;
    }

    /// Phase 29h/29i: `_akh/evidence/assess` returns a Dempster-Shafer
    /// assessment for a resolvable claim (vacuous when no evidence is tracked)
    /// and errors on unknown claims; `_akh/trust` serves the source-trust
    /// leaderboard shape.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn acp_ext_evidence_assess_and_trust() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-acp-evidence.sock");
        let me = workspace(); // seeds ("Sun", "is-a", "Star")
        let listener = UnixListener::bind(&sock).expect("bind unix");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server = tokio::spawn(async move { serve_unix_loop(me, listener, shutdown_rx).await });

        let a = ext_ok(&sock, "_akh/evidence/assess", serde_json::json!({ "claim": "Sun" })).await;
        assert_eq!(a["stale"], serde_json::json!(false));
        let assessment = &a["assessment"];
        assert_eq!(assessment["claim"], "Sun");
        assert!(assessment["verdict"].is_string(), "verdict label expected: {a}");
        assert!(assessment["belief"].is_number());
        assert!(assessment["plausibility"].is_number());

        // unknown claim / missing param → clean errors.
        assert!(ext_err(&sock, "_akh/evidence/assess", serde_json::json!({ "claim": "NoSuchThing" }))
            .await
            .is_err());
        assert!(ext_err(&sock, "_akh/evidence/assess", serde_json::json!({})).await.is_err());

        // trust leaderboard: both directions present (empty for a fresh agent).
        let t = ext_ok(&sock, "_akh/trust", serde_json::json!({ "top_k": 5 })).await;
        assert!(t["most_reliable"].is_array());
        assert!(t["least_reliable"].is_array());

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
