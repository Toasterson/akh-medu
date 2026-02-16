//! Remote WebSocket backend for TUI chat.
//!
//! Connects to an `akhomed` server's WebSocket endpoint and exposes a
//! sync-friendly channel interface that the TUI event loop can poll.

#[cfg(feature = "daemon")]
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
