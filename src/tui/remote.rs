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
#[cfg(all(feature = "daemon", feature = "acp"))]
mod inner {
    use std::sync::mpsc;
    use std::thread;

    use crate::client::ServerInfo;
    use crate::message::AkhMessage;

    /// A user action queued from the TUI to the background ACP session.
    enum Outbound {
        Input(String),
        Command(String),
    }

    /// Which ACP transport the daemon advertised in its discovery record.
    enum Endpoint {
        Unix(String),
        Tcp(String),
    }

    /// Handle to a background ACP session.
    ///
    /// The API mirrors the legacy WebSocket backend exactly so the TUI event
    /// loop is unchanged: `send_input`/`send_command` dispatch user actions and
    /// `try_recv` polls for responses — all non-blocking.
    pub struct RemoteChat {
        tx: mpsc::Sender<Outbound>,
        rx: mpsc::Receiver<AkhMessage>,
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
            let (in_tx, in_rx) = mpsc::channel::<AkhMessage>();

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
                            let _ = in_tx.send(AkhMessage::error(
                                "acp",
                                format!("failed to create tokio runtime: {e}"),
                            ));
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

        /// Poll for the next response (non-blocking).
        pub fn try_recv(&mut self) -> Option<AkhMessage> {
            match self.rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !self.disconnected {
                        self.disconnected = true;
                        Some(AkhMessage::error("acp", "server connection lost"))
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
    /// answer text is forwarded back to the TUI as a narrative message.
    async fn acp_task(
        endpoint: Endpoint,
        out_rx: mpsc::Receiver<Outbound>,
        in_tx: mpsc::Sender<AkhMessage>,
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
                            let _ = in_tx.send(AkhMessage::error("acp", e.to_string()));
                            break;
                        }
                        match session.read_to_string().await {
                            Ok(resp) => {
                                if !resp.trim().is_empty() {
                                    let _ = in_tx.send(AkhMessage::narrative(resp, "acp"));
                                }
                            }
                            Err(e) => {
                                let _ = in_tx.send(AkhMessage::error("acp", e.to_string()));
                                break;
                            }
                        }
                    }
                    Ok(Outbound::Command(cmd)) => {
                        let _ = in_tx.send(AkhMessage::system(format!(
                            "command '{cmd}' is not yet available over ACP"
                        )));
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
            let _ = in_err.send(AkhMessage::error("acp", format!("ACP session ended: {e}")));
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Legacy WebSocket backend (fallback while `acp` is not enabled)
// ──────────────────────────────────────────────────────────────────────────
#[cfg(all(feature = "daemon", not(feature = "acp")))]
mod inner {
    use std::sync::mpsc;
    use std::thread;

    use crate::client::ServerInfo;
    use crate::message::AkhMessage;

    /// JSON message sent to the server over WebSocket.
    #[derive(Debug, serde::Serialize)]
    pub struct WsInput {
        #[serde(rename = "type")]
        pub msg_type: String,
        pub text: String,
    }

    /// Handle to a background WS connection.
    ///
    /// The TUI calls `send()` to dispatch user input and `try_recv()` to
    /// poll for server responses — both non-blocking from the TUI's
    /// crossterm event loop perspective.
    pub struct RemoteChat {
        tx: mpsc::Sender<WsInput>,
        rx: mpsc::Receiver<AkhMessage>,
        /// Set to true when the background thread has exited.
        disconnected: bool,
    }

    impl RemoteChat {
        /// Spawn a background tokio runtime + WS connection.
        ///
        /// Returns a `RemoteChat` handle or an error string.
        pub fn connect(info: &ServerInfo, workspace: &str) -> Result<Self, String> {
            let host = if info.bind == "0.0.0.0" {
                "127.0.0.1"
            } else {
                &info.bind
            };
            let url = format!("ws://{}:{}/ws/{}", host, info.port, workspace);

            // Channels: TUI → WS thread (outbound), WS thread → TUI (inbound).
            let (out_tx, out_rx) = mpsc::channel::<WsInput>();
            let (in_tx, in_rx) = mpsc::channel::<AkhMessage>();

            let url_clone = url.clone();
            thread::Builder::new()
                .name("ws-remote".into())
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            let _ = in_tx.send(AkhMessage::error(
                                "ws",
                                format!("failed to create tokio runtime: {e}"),
                            ));
                            return;
                        }
                    };

                    rt.block_on(async move {
                        ws_task(url_clone, out_rx, in_tx).await;
                    });
                })
                .map_err(|e| format!("failed to spawn WS thread: {e}"))?;

            Ok(Self {
                tx: out_tx,
                rx: in_rx,
                disconnected: false,
            })
        }

        /// Send user input to the server (non-blocking).
        pub fn send_input(&self, text: &str) {
            let _ = self.tx.send(WsInput {
                msg_type: "input".into(),
                text: text.to_string(),
            });
        }

        /// Send a command to the server (non-blocking).
        pub fn send_command(&self, cmd: &str) {
            let _ = self.tx.send(WsInput {
                msg_type: "command".into(),
                text: cmd.to_string(),
            });
        }

        /// Poll for the next server message (non-blocking).
        pub fn try_recv(&mut self) -> Option<AkhMessage> {
            match self.rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !self.disconnected {
                        self.disconnected = true;
                        Some(AkhMessage::error("ws", "server connection lost"))
                    } else {
                        None
                    }
                }
                Err(mpsc::TryRecvError::Empty) => None,
            }
        }

        /// Whether the WS connection has been lost.
        pub fn is_disconnected(&self) -> bool {
            self.disconnected
        }
    }

    /// Background async task: connects to WS, relays messages in both directions.
    async fn ws_task(
        url: String,
        outbound: mpsc::Receiver<WsInput>,
        inbound: mpsc::Sender<AkhMessage>,
    ) {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite;

        let ws_stream = match tokio_tungstenite::connect_async(&url).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                let _ = inbound.send(AkhMessage::error(
                    "ws",
                    format!("failed to connect to {url}: {e}"),
                ));
                return;
            }
        };

        let (mut sink, mut stream) = ws_stream.split();

        // Spawn a task to forward outbound messages (TUI → server).
        let inbound_err = inbound.clone();
        let send_handle = tokio::spawn(async move {
            // Poll the sync mpsc receiver from an async context.
            loop {
                // Yield to avoid busy-spinning; check every 50ms.
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                match outbound.try_recv() {
                    Ok(msg) => {
                        let json = match serde_json::to_string(&msg) {
                            Ok(j) => j,
                            Err(_) => continue,
                        };
                        if sink
                            .send(tungstenite::Message::Text(json.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => break,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
        });

        // Read inbound messages (server → TUI).
        while let Some(result) = stream.next().await {
            match result {
                Ok(tungstenite::Message::Text(text)) => {
                    match serde_json::from_str::<AkhMessage>(&text) {
                        Ok(msg) => {
                            if inbound.send(msg).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = inbound.send(AkhMessage::system(format!(
                                "[ws] unparseable message: {e}"
                            )));
                        }
                    }
                }
                Ok(tungstenite::Message::Close(_)) => break,
                Err(e) => {
                    let _ = inbound_err.send(AkhMessage::error(
                        "ws",
                        format!("connection error: {e}"),
                    ));
                    break;
                }
                _ => {}
            }
        }

        send_handle.abort();
    }
}

#[cfg(feature = "daemon")]
pub use inner::*;

#[cfg(all(test, feature = "daemon", feature = "acp"))]
mod tests {
    use super::RemoteChat;
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

        // Poll the sync channel like the TUI does (bounded so it can never
        // hang). The budget is ≤30s (1500 × 20ms): the relay+turn finishes in
        // well under a second normally, so this only bounds a genuine hang —
        // generous enough that CPU starvation under full-suite parallel load
        // cannot trip it (was 10s, which flaked).
        let mut answer: Option<String> = None;
        for _ in 0..1500 {
            if let Some(msg) = chat.try_recv() {
                if let AkhMessage::Error { code, message, .. } = &msg {
                    panic!("ACP relay surfaced an error [{code}]: {message}");
                }
                let text = msg.to_plain_text();
                if !text.trim().is_empty() {
                    answer = Some(text);
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        shutdown_tx.send(true).ok();
        let _ = server_thread.join();

        assert!(
            answer.is_some(),
            "expected the ACP RemoteChat backend to relay a non-empty answer"
        );
    }
}
