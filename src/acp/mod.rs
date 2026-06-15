//! ACP (Agent Client Protocol) transport — P0 spike.
//!
//! This module proves that akh-medu can serve as an **ACP agent over a Unix
//! domain socket**, wrapping the existing [`ChatProcessor`] for ONE in-process
//! workspace. It is intentionally minimal: it de-risks the byte-stream transport
//! and the ADR-027 concurrency invariant, not the full P2 server.
//!
//! # What it does
//!
//! - `initialize` → advertise minimal (default) capabilities, echo the protocol
//!   version, and report agent name/version.
//! - `session/new` → mint a fresh session id (the spike serves a single
//!   workspace, so the id is purely a handle).
//! - `session/prompt` → run [`ChatProcessor::process_input`] under the ADR-027
//!   discipline (lock the agent inside [`tokio::task::spawn_blocking`], never
//!   across an `.await`), stream the concatenated [`AkhMessage`] text back as a
//!   `session/update` `AgentMessageChunk`, then respond with
//!   [`StopReason::EndTurn`].
//! - Everything else falls through to the crate's default handler.
//!
//! # Transport
//!
//! `agent-client-protocol` is `futures`-based: its [`ByteStreams`] transport
//! wants `futures::io::{AsyncRead, AsyncWrite}`. We bridge tokio's [`UnixStream`]
//! halves into those traits with [`tokio_util::compat`].
//!
//! [`ChatProcessor`]: crate::chat::ChatProcessor
//! [`AkhMessage`]: crate::message::AkhMessage
//! [`ByteStreams`]: agent_client_protocol::ByteStreams
//! [`UnixStream`]: tokio::net::UnixStream

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use agent_client_protocol::schema::{
    AgentCapabilities, ContentBlock, ContentChunk, InitializeRequest, InitializeResponse,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, SessionId,
    SessionNotification, SessionUpdate, StopReason, TextContent,
};
use agent_client_protocol::{
    Agent as AcpRole, ByteStreams, Client, ConnectionTo, Responder,
};
use miette::Diagnostic;
use thiserror::Error;
use tokio::net::{UnixListener, UnixStream};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::agent::Agent;
use crate::chat::ChatProcessor;
use crate::engine::Engine;
use crate::message::AkhMessage;

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
    engine: Arc<Engine>,
    agent: Arc<Mutex<Agent>>,
    processor: Arc<Mutex<ChatProcessor>>,
    /// Agent name advertised during `initialize`.
    name: String,
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
            engine,
            agent,
            processor: Arc::new(Mutex::new(processor)),
            name: "akh-medu".to_string(),
        }
    }

    /// Override the agent name advertised during `initialize`.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Run a single prompt turn under the ADR-027 lock discipline.
    ///
    /// Locks (engine clone is cheap `Arc`, agent + processor `Mutex`es) are
    /// acquired and released entirely inside the [`spawn_blocking`] closure, so
    /// no `std::sync::Mutex` guard ever crosses the `.await`. The returned
    /// `String` is the rendered, concatenated response — the caller emits any
    /// ACP messages only after this future resolves.
    ///
    /// [`spawn_blocking`]: tokio::task::spawn_blocking
    async fn run_prompt(&self, text: String) -> AcpResult<String> {
        let engine = Arc::clone(&self.engine);
        let agent = Arc::clone(&self.agent);
        let processor = Arc::clone(&self.processor);

        let responses: Vec<AkhMessage> = tokio::task::spawn_blocking(move || {
            // ── ADR-027: lock, process, drop — all synchronous, no await. ──
            let mut agent = agent.lock().unwrap_or_else(|e| e.into_inner());
            let mut processor = processor.lock().unwrap_or_else(|e| e.into_inner());
            processor.process_input(&text, &mut agent, &engine)
            // Guards drop here, at end of closure scope.
        })
        .await
        .map_err(|e| AcpError::Worker {
            message: e.to_string(),
        })?;

        Ok(render_messages(&responses))
    }
}

// ──────────────────────────────────────────────────────────────────────────
// AkhMessage rendering (spike-grade: a simple match → display text)
// ──────────────────────────────────────────────────────────────────────────

/// Render a slice of [`AkhMessage`] into a single display string for the spike.
///
/// This is deliberately a thin `format!`/match — the full structured mapping to
/// rich ACP content blocks is deferred to the P2 server.
fn render_messages(messages: &[AkhMessage]) -> String {
    messages
        .iter()
        .map(render_message)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render a single [`AkhMessage`] variant to display text.
fn render_message(msg: &AkhMessage) -> String {
    match msg {
        AkhMessage::Fact { text, confidence, .. } => match confidence {
            Some(c) => format!("{text} (confidence {c:.2})"),
            None => text.clone(),
        },
        AkhMessage::Reasoning { step, expression } => match expression {
            Some(expr) => format!("{step} :: {expr}"),
            None => step.clone(),
        },
        AkhMessage::Gap { entity, description } => format!("gap: {entity} — {description}"),
        AkhMessage::ToolResult { tool, success, output } => {
            let status = if *success { "ok" } else { "failed" };
            format!("[{tool}:{status}] {output}")
        }
        AkhMessage::Narrative { text, .. } => text.clone(),
        AkhMessage::System { text } => text.clone(),
        AkhMessage::Error { code, message, .. } => format!("error[{code}]: {message}"),
        AkhMessage::GoalProgress { goal, status, detail } => match detail {
            Some(d) => format!("goal {goal} [{status}]: {d}"),
            None => format!("goal {goal} [{status}]"),
        },
        AkhMessage::Prompt { question } => question.clone(),
        AkhMessage::AuditLog { kind, summary, .. } => format!("audit[{kind}]: {summary}"),
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
            move |_req: NewSessionRequest, responder: Responder<NewSessionResponse>, _cx| async move {
                // Single-workspace spike: any unique handle works. Use a
                // monotonic-ish id derived from the wall clock.
                let session_id = SessionId::new(format!("akh-{}", next_session_suffix()));
                responder.respond(NewSessionResponse::new(session_id))
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

                        // Run the synchronous chat turn off the async runtime,
                        // honoring ADR-027 (locks live only inside the closure).
                        let rendered = me.run_prompt(text).await.map_err(|e| {
                            agent_client_protocol::Error::internal_error()
                                .data(e.to_string())
                        })?;

                        // Emit the rendered answer as an agent message chunk so
                        // the client's `read_to_string()` collects it. Sent only
                        // AFTER the blocking guard is gone (it is — `run_prompt`
                        // already awaited and dropped its guards).
                        if !rendered.is_empty() {
                            cx.send_notification(SessionNotification::new(
                                session_id,
                                SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                    ContentBlock::Text(TextContent::new(rendered)),
                                )),
                            ))?;
                        }

                        responder.respond(PromptResponse::new(StopReason::EndTurn))
                    }
                }
            },
            agent_client_protocol::on_receive_request!(),
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

// ──────────────────────────────────────────────────────────────────────────
// End-to-end test: real Unix socket, real ACP client ↔ ACP agent.
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
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
            std::time::Duration::from_secs(15),
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
        match tokio::time::timeout(std::time::Duration::from_secs(5), server).await {
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
}
