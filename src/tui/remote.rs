//! Remote backend for TUI chat.
//!
//! Exposes a sync-friendly [`RemoteChat`] handle the TUI event loop can poll
//! (`send_input`/`send_command` + `try_recv`, all non-blocking). The transport
//! is selected at compile time:
//!
//! - With the **`acp`** feature: an Agent Client Protocol session over the
//!   daemon's Unix socket / TCP endpoint (the pure-ACP path).
//! - With **`daemon`** only (no `acp`): the legacy WebSocket connection, kept as
//!   a fallback during the migration to ACP-only (removed in P8).
//!
//! Both implementations present the same `RemoteChat` API, so the TUI is
//! transport-agnostic.

// ──────────────────────────────────────────────────────────────────────────
// ACP backend (preferred)
// ──────────────────────────────────────────────────────────────────────────
#[cfg(feature = "acp")]
mod inner {
    use std::sync::mpsc;
    use std::thread;

    use agent_client_protocol::schema::{ContentBlock, ContentChunk, SessionNotification, SessionUpdate};
    use agent_client_protocol::util::MatchDispatch;
    use agent_client_protocol::SessionMessage;

    use crate::client::ServerInfo;
    use crate::message::AkhMessage;

    /// A user action queued from the TUI to the background ACP session.
    enum Outbound {
        Input(String),
        Command(String),
    }

    /// One incremental piece of a streaming chat turn, surfaced to the TUI event
    /// loop via [`RemoteChat::try_recv`].
    ///
    /// A turn streams as a sequence of deltas terminated by [`ChatUpdate::TurnEnd`]:
    /// answer text arrives as [`AnswerDelta`](ChatUpdate::AnswerDelta) chunks (to
    /// be concatenated), chain-of-thought + tool steps as
    /// [`ThoughtDelta`](ChatUpdate::ThoughtDelta) chunks, and out-of-band whole
    /// messages (errors, connection loss) as [`Notice`](ChatUpdate::Notice).
    pub enum ChatUpdate {
        /// A delta of the agent's answer text (ACP `AgentMessageChunk`).
        AnswerDelta(String),
        /// A delta of the agent's thinking / a tool step (ACP `AgentThoughtChunk`).
        ThoughtDelta(String),
        /// A complete out-of-band message (error, system notice, disconnect).
        Notice(AkhMessage),
        /// The current turn finished (ACP `StopReason`) — stop the spinner.
        TurnEnd,
    }

    /// Which ACP transport the daemon advertised in its discovery record.
    enum Endpoint {
        Unix(String),
        Tcp(String),
    }

    /// Handle to a background ACP session.
    ///
    /// `send_input`/`send_command` dispatch user actions and `try_recv` polls for
    /// streaming [`ChatUpdate`]s — all non-blocking, so the TUI event loop never
    /// stalls.
    pub struct RemoteChat {
        tx: mpsc::Sender<Outbound>,
        rx: mpsc::Receiver<ChatUpdate>,
        disconnected: bool,
    }

    impl RemoteChat {
        /// Connect to the daemon's ACP endpoint on a background tokio runtime.
        ///
        /// The endpoint comes from the discovery record (`acp_socket` preferred,
        /// then `acp_tcp`). Note: the ACP daemon serves its configured akh, so
        /// `workspace` is advisory here — multi-workspace session routing is a
        /// later increment.
        pub fn connect(info: &ServerInfo, workspace: &str) -> Result<Self, String> {
            let _ = workspace;
            let endpoint = if let Some(sock) = info.acp_socket.clone() {
                Endpoint::Unix(sock)
            } else if let Some(tcp) = info.acp_tcp.clone() {
                Endpoint::Tcp(tcp)
            } else {
                return Err("akhomed is running but advertises no ACP endpoint — \
                            rebuild the daemon with --features acp"
                    .to_string());
            };

            // Channels: TUI → session (outbound), session → TUI (inbound).
            let (out_tx, out_rx) = mpsc::channel::<Outbound>();
            let (in_tx, in_rx) = mpsc::channel::<ChatUpdate>();

            thread::Builder::new()
                .name("acp-remote".into())
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(2)
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            let _ = in_tx.send(ChatUpdate::Notice(AkhMessage::error(
                                "acp",
                                format!("failed to create tokio runtime: {e}"),
                            )));
                            return;
                        }
                    };
                    rt.block_on(acp_task(endpoint, out_rx, in_tx));
                })
                .map_err(|e| format!("failed to spawn ACP thread: {e}"))?;

            Ok(Self {
                tx: out_tx,
                rx: in_rx,
                disconnected: false,
            })
        }

        /// Send user input to the akh as a prompt (non-blocking).
        pub fn send_input(&self, text: &str) {
            let _ = self.tx.send(Outbound::Input(text.to_string()));
        }

        /// Send a server-side command (non-blocking). Data-plane commands over
        /// ACP (`_akh/*`) are a later increment; for now this reports back.
        pub fn send_command(&self, cmd: &str) {
            let _ = self.tx.send(Outbound::Command(cmd.to_string()));
        }

        /// Poll for the next streaming update (non-blocking).
        pub fn try_recv(&mut self) -> Option<ChatUpdate> {
            match self.rx.try_recv() {
                Ok(update) => Some(update),
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !self.disconnected {
                        self.disconnected = true;
                        Some(ChatUpdate::Notice(AkhMessage::error(
                            "acp",
                            "server connection lost",
                        )))
                    } else {
                        None
                    }
                }
                Err(mpsc::TryRecvError::Empty) => None,
            }
        }

        /// Whether the ACP connection has been lost.
        pub fn is_disconnected(&self) -> bool {
            self.disconnected
        }
    }

    /// Background task: open ONE ACP session and relay between the TUI channels
    /// and the akh. User inputs become `session/prompt` turns; each turn's
    /// `session/update` notifications are forwarded back to the TUI as streaming
    /// [`ChatUpdate`]s as they arrive, terminated by [`ChatUpdate::TurnEnd`].
    async fn acp_task(
        endpoint: Endpoint,
        out_rx: mpsc::Receiver<Outbound>,
        in_tx: mpsc::Sender<ChatUpdate>,
    ) {
        let in_err = in_tx.clone();

        // The whole interactive loop lives inside the session closure, because
        // the ACP connection is alive only for its duration.
        let run = async move |mut session: agent_client_protocol::ActiveSession<
            'static,
            agent_client_protocol::Agent,
        >,
                              _cx| {
            loop {
                match out_rx.try_recv() {
                    Ok(Outbound::Input(text)) => {
                        if let Err(e) = session.send_prompt(&text) {
                            let _ = in_tx.send(ChatUpdate::Notice(AkhMessage::error(
                                "acp",
                                e.to_string(),
                            )));
                            let _ = in_tx.send(ChatUpdate::TurnEnd);
                            break;
                        }
                        // Stream this turn's updates until its stop reason,
                        // forwarding answer/thought deltas live.
                        if let Err(e) = relay_turn(&mut session, &in_tx).await {
                            let _ = in_tx.send(ChatUpdate::Notice(AkhMessage::error(
                                "acp",
                                e.to_string(),
                            )));
                            let _ = in_tx.send(ChatUpdate::TurnEnd);
                            break;
                        }
                    }
                    Ok(Outbound::Command(cmd)) => {
                        let _ = in_tx.send(ChatUpdate::Notice(AkhMessage::system(format!(
                            "command '{cmd}' is not yet available over ACP"
                        ))));
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => break, // TUI closed
                }
            }
            Ok::<(), agent_client_protocol::Error>(())
        };

        let result = match endpoint {
            Endpoint::Unix(path) => crate::acp::chat_over_unix(&path, "akh-tui", run).await,
            Endpoint::Tcp(addr) => crate::acp::chat_over_tcp(&addr, "akh-tui", run).await,
        };
        if let Err(e) = result {
            let _ = in_err.send(ChatUpdate::Notice(AkhMessage::error(
                "acp",
                format!("ACP session ended: {e}"),
            )));
        }
    }

    /// Read one turn's `session/update` notifications until its stop reason,
    /// forwarding each text delta to the TUI as a [`ChatUpdate`].
    ///
    /// `AgentMessageChunk`s become [`ChatUpdate::AnswerDelta`] (the live answer);
    /// `AgentThoughtChunk`s become [`ChatUpdate::ThoughtDelta`] (thinking + tool
    /// steps). A [`StopReason`](agent_client_protocol::schema::StopReason) ends
    /// the turn and emits [`ChatUpdate::TurnEnd`]. Non-text / unhandled updates
    /// are ignored.
    async fn relay_turn(
        session: &mut agent_client_protocol::ActiveSession<'static, agent_client_protocol::Agent>,
        in_tx: &mpsc::Sender<ChatUpdate>,
    ) -> Result<(), agent_client_protocol::Error> {
        loop {
            match session.read_update().await? {
                SessionMessage::SessionMessage(dispatch) => {
                    MatchDispatch::new(dispatch)
                        .if_notification(async |notif: SessionNotification| {
                            match notif.update {
                                SessionUpdate::AgentMessageChunk(ContentChunk {
                                    content: ContentBlock::Text(text),
                                    ..
                                }) => {
                                    let _ = in_tx.send(ChatUpdate::AnswerDelta(text.text));
                                }
                                SessionUpdate::AgentThoughtChunk(ContentChunk {
                                    content: ContentBlock::Text(text),
                                    ..
                                }) => {
                                    let _ = in_tx.send(ChatUpdate::ThoughtDelta(text.text));
                                }
                                _ => {}
                            }
                            Ok(())
                        })
                        .await
                        .otherwise_ignore()?;
                }
                SessionMessage::StopReason(_) => {
                    let _ = in_tx.send(ChatUpdate::TurnEnd);
                    break;
                }
                // `SessionMessage` is `#[non_exhaustive]`; ignore future variants.
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(feature = "acp")]
pub use inner::*;

#[cfg(all(test, feature = "acp"))]
mod tests {
    use super::{ChatUpdate, RemoteChat};
    use crate::agent::{Agent, AgentConfig};
    use crate::chat::ChatProcessor;
    use crate::client::ServerInfo;
    use crate::engine::{Engine, EngineConfig};
    use crate::message::AkhMessage;
    use crate::nlu::NluPipeline;
    use crate::vsa::Dimension;
    use std::sync::{Arc, Mutex};

    /// The ACP `RemoteChat` backend relays a user prompt to an in-process ACP
    /// server over a real Unix socket and surfaces the akh's answer back through
    /// its sync `try_recv` channel — the exact path the TUI event loop uses.
    #[test]
    fn remote_chat_relays_prompt_over_acp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("akh-tui-remote.sock");
        let sock_str = sock.display().to_string();

        // ── In-process ACP server on its own runtime/thread. ──
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let sock_for_server = sock.clone();
        let server_thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("server rt");
            rt.block_on(async move {
                let engine = Engine::new(EngineConfig {
                    dimension: Dimension::TEST,
                    ..Default::default()
                })
                .expect("engine");
                engine
                    .ingest_label_triples(&[(
                        "Sun".to_string(),
                        "is-a".to_string(),
                        "Star".to_string(),
                        1.0,
                    )])
                    .expect("seed");
                let engine = Arc::new(engine);
                let processor = ChatProcessor::new(&engine, NluPipeline::new());
                let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
                let me = crate::acp::AkhAcpAgent::new(engine, Arc::new(Mutex::new(agent)), processor);
                let listener = tokio::net::UnixListener::bind(&sock_for_server).expect("bind");
                let _ = crate::acp::serve_unix_loop(me, listener, shutdown_rx).await;
            });
        });

        // Wait for the socket to be created before connecting (≤4s — generous
        // so heavy parallel test load can't lose the race to a short budget).
        for _ in 0..800 {
            if sock.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let info = ServerInfo {
            pid: std::process::id(),
            port: 0,
            bind: "127.0.0.1".into(),
            acp_socket: Some(sock_str),
            acp_tcp: None,
            acp_version: Some("1".into()),
        };

        let mut chat = RemoteChat::connect(&info, "default").expect("RemoteChat connect");
        chat.send_input("What is the Sun?");

        // Poll the sync channel like the TUI does, accumulating the streamed
        // answer deltas until the turn ends (bounded so it can never hang). The
        // budget is ≤30s (1500 × 20ms): the relay+turn finishes in well under a
        // second normally, so this only bounds a genuine hang — generous enough
        // that CPU starvation under full-suite parallel load cannot trip it.
        let mut answer = String::new();
        'poll: for _ in 0..1500 {
            while let Some(update) = chat.try_recv() {
                match update {
                    ChatUpdate::AnswerDelta(s) => answer.push_str(&s),
                    ChatUpdate::ThoughtDelta(_) => {}
                    ChatUpdate::Notice(AkhMessage::Error { code, message, .. }) => {
                        panic!("ACP relay surfaced an error [{code}]: {message}");
                    }
                    ChatUpdate::Notice(_) => {}
                    ChatUpdate::TurnEnd => break 'poll,
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        shutdown_tx.send(true).ok();
        let _ = server_thread.join();

        assert!(
            !answer.trim().is_empty(),
            "expected the ACP RemoteChat backend to relay a non-empty answer"
        );
    }
}
