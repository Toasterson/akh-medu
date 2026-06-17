//! Unified ratatui-based TUI — a pure ACP (remote) chat client.
//!
//! The TUI provides: scrollable message output, input area, status bar,
//! and TUI commands (e.g., `/grammar`, `/goals`, `/quit`). It connects to a
//! running akhomed daemon over the Agent Client Protocol; there is no longer a
//! local-engine backend (`akh` is a pure ACP client).
//!
//! The whole app is gated behind the `acp` feature: without it, this module
//! exports nothing.

pub mod remote;
pub mod sink;
pub mod widgets;

#[cfg(feature = "acp")]
use std::sync::Arc;
#[cfg(feature = "acp")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "acp")]
use std::time::Duration;

#[cfg(feature = "acp")]
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
#[cfg(feature = "acp")]
use miette::IntoDiagnostic;

#[cfg(feature = "acp")]
use crate::message::AkhMessage;

/// A centered model-picker overlay (the `/model` popup).
#[cfg(feature = "acp")]
pub struct ModelPicker {
    /// All model names returned by `_akh/model/list`, sorted (may be empty for a
    /// cloud endpoint that lists nothing). This is the unfiltered universe; the
    /// rows actually shown are [`ModelPicker::filtered`].
    pub models: Vec<String>,
    /// Type-ahead filter: only models whose name starts with this (case-
    /// insensitive) are shown. Empty = show all.
    pub filter: String,
    /// Index into the *filtered* list of the currently highlighted row.
    pub selected: usize,
}

#[cfg(feature = "acp")]
impl ModelPicker {
    /// The models currently visible given the type-ahead [`filter`]: a
    /// case-insensitive prefix match over the (already sorted) `models`. An
    /// empty filter shows everything.
    ///
    /// [`filter`]: ModelPicker::filter
    #[must_use]
    pub fn filtered(&self) -> Vec<&String> {
        if self.filter.is_empty() {
            return self.models.iter().collect();
        }
        let needle = self.filter.to_ascii_lowercase();
        self.models
            .iter()
            .filter(|m| m.to_ascii_lowercase().starts_with(&needle))
            .collect()
    }
}

/// TUI application state (ACP client).
#[cfg(feature = "acp")]
pub struct AkhTui {
    workspace: String,
    remote: remote::RemoteChat,
    /// Discovery record for the connected daemon — used to issue one-shot
    /// `_akh/*` control calls (model list/set) over a fresh short-lived ACP
    /// connection, independent of the persistent chat session.
    info: crate::client::ServerInfo,
    messages: Vec<AkhMessage>,
    input_buffer: String,
    scroll_offset: usize,
    should_quit: bool,
    /// Whether audit log entries are displayed in the message stream.
    show_audit: bool,
    /// Whether tool/agent "thinking" steps are expanded (default: collapsed).
    show_thinking: bool,
    /// Active model-picker overlay (the `/model` popup). When `Some`, key input
    /// is routed to the picker instead of the chat input.
    picker: Option<ModelPicker>,
}

#[cfg(feature = "acp")]
impl AkhTui {
    /// Create a new TUI instance with a remote ACP backend.
    ///
    /// `info` is the daemon's discovery record; it is retained so the TUI can
    /// issue one-shot `_akh/*` control calls (e.g. model switching) over a fresh
    /// connection without disturbing the persistent chat session.
    pub fn new_remote(
        workspace: String,
        remote: remote::RemoteChat,
        info: crate::client::ServerInfo,
    ) -> Self {
        Self {
            workspace,
            remote,
            info,
            messages: vec![AkhMessage::system(
                "Connected to akhomed. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
            show_audit: true,
            show_thinking: false,
            picker: None,
        }
    }

    /// Run the TUI event loop.
    pub fn run(&mut self) -> miette::Result<()> {
        // Register SIGTERM handler so `kill <pid>` triggers a clean exit
        // (terminal restore + session persist) instead of leaving raw mode.
        let sigterm_flag = Arc::new(AtomicBool::new(false));
        #[cfg(unix)]
        {
            use signal_hook::consts::SIGTERM;
            signal_hook::flag::register(SIGTERM, Arc::clone(&sigterm_flag))
                .into_diagnostic()?;
        }

        let mut terminal = ratatui::init();

        // Enable mouse capture for scroll wheel support.
        crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)
            .into_diagnostic()?;

        loop {
            // Drain pending messages from backend.
            self.drain_backend_messages();

            terminal
                .draw(|frame| {
                    widgets::render(
                        frame,
                        &self.workspace,
                        &self.messages,
                        &self.input_buffer,
                        self.scroll_offset,
                        self.show_thinking,
                        self.picker.as_ref(),
                    );
                })
                .into_diagnostic()?;

            if self.should_quit || sigterm_flag.load(Ordering::Relaxed) {
                break;
            }

            // Poll for events.
            if event::poll(Duration::from_millis(100)).into_diagnostic()? {
                match event::read().into_diagnostic()? {
                    Event::Key(key) => {
                        if key.kind != KeyEventKind::Press {
                            continue;
                        }
                        self.handle_key(key.code, key.modifiers);
                    }
                    Event::Mouse(mouse) => match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            self.scroll_offset = self.scroll_offset.saturating_sub(3);
                        }
                        MouseEventKind::ScrollDown => {
                            self.scroll_offset = (self.scroll_offset + 3)
                                .min(self.messages.len().saturating_sub(1));
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }

        self.on_exit()?;

        // Disable mouse capture before restoring terminal.
        let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);

        ratatui::restore();
        Ok(())
    }

    /// Drain pending messages from the backend.
    fn drain_backend_messages(&mut self) {
        let show_audit = self.show_audit;
        while let Some(msg) = self.remote.try_recv() {
            if show_audit || !matches!(msg, AkhMessage::AuditLog { .. }) {
                self.messages.push(msg);
            }
        }
    }

    /// Called on exit — persist session, etc.
    fn on_exit(&mut self) -> miette::Result<()> {
        Ok(())
    }

    fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        // When the model picker is open it owns all key input: arrows move the
        // selection, Enter applies the highlighted model, Esc dismisses. Normal
        // chat keys (typing / send / Tab-thinking) only apply when it is closed.
        if self.picker.is_some() {
            self.handle_picker_key(code);
            return;
        }
        match code {
            KeyCode::Enter => {
                let input = self.input_buffer.trim().to_string();
                self.input_buffer.clear();

                if input.is_empty() {
                    return;
                }

                self.process_input(&input);
            }
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Tab => {
                // Expand/collapse the tool/agent "thinking" steps.
                self.show_thinking = !self.show_thinking;
            }
            KeyCode::Char(c) => {
                self.input_buffer.push(c);
            }
            KeyCode::Backspace => {
                self.input_buffer.pop();
            }
            KeyCode::PageUp => {
                self.scroll_offset = self.scroll_offset.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.scroll_offset =
                    (self.scroll_offset + 10).min(self.messages.len().saturating_sub(1));
            }
            KeyCode::Home => {
                self.scroll_offset = 0;
            }
            KeyCode::End => {
                self.scroll_offset = self.messages.len().saturating_sub(1);
            }
            KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    /// Route a key to the open model picker. Up/Down move the (clamped)
    /// selection within the filtered view; printable chars extend the type-ahead
    /// filter and Backspace trims it (both reset the selection to the top);
    /// Enter applies the highlighted model via `_akh/model/set` and closes the
    /// picker; Esc closes without applying.
    fn handle_picker_key(&mut self, code: KeyCode) {
        let Some(picker) = self.picker.as_mut() else {
            return;
        };
        match code {
            KeyCode::Up => {
                picker.selected = picker.selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let len = picker.filtered().len();
                if len > 0 {
                    picker.selected = (picker.selected + 1).min(len - 1);
                }
            }
            KeyCode::Char(c) => {
                // Type to filter; a narrower list invalidates the old index.
                picker.filter.push(c);
                picker.selected = 0;
            }
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.selected = 0;
            }
            KeyCode::Esc => {
                self.picker = None;
            }
            KeyCode::Enter => {
                // Take the chosen model out of the *filtered* view, then close
                // the picker before the (blocking) control call so the overlay
                // isn't left up.
                let chosen = picker.filtered().get(picker.selected).map(|m| (*m).clone());
                self.picker = None;
                if let Some(model) = chosen {
                    self.set_model(&model);
                }
            }
            _ => {}
        }
    }

    /// Issue one `_akh/model/set` to switch the running model, reporting the
    /// outcome as a System message.
    fn set_model(&mut self, model: &str) {
        let params = serde_json::json!({ "workspace": self.workspace, "model": model });
        match self.control_call("_akh/model/set", params) {
            Ok(v) => {
                let m = v.get("model").and_then(serde_json::Value::as_str).unwrap_or(model);
                self.messages
                    .push(AkhMessage::system(format!("Model set to '{m}'.")));
            }
            Err(e) => self
                .messages
                .push(AkhMessage::error("model", format!("model set failed: {e}"))),
        }
    }

    /// List the akh's models via `_akh/model/list` and open the picker.
    ///
    /// Pre-selects the current model when present in the list. A cloud endpoint
    /// that lists no models (the common `ollama.com` case) still gets a picker —
    /// it shows an informational line — so the user is told to type
    /// `/model <name>` instead of the picker silently failing.
    fn open_model_picker(&mut self) {
        let params = serde_json::json!({ "workspace": self.workspace });
        match self.control_call("_akh/model/list", params) {
            Ok(v) => {
                if let Some(err) = v.get("error").and_then(serde_json::Value::as_str) {
                    self.messages.push(AkhMessage::system(format!(
                        "(could not list models: {err})"
                    )));
                }
                let current = v.get("current").and_then(serde_json::Value::as_str);
                let mut models: Vec<String> = v
                    .get("models")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(serde_json::Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                models.sort();

                if models.is_empty() {
                    // Nothing to pick from — guide the user to the direct form.
                    let cur = current.unwrap_or("(none)");
                    self.messages.push(AkhMessage::system(format!(
                        "No models listed (current: {cur}). Use /model <name> to switch."
                    )));
                    return;
                }

                let selected = current
                    .and_then(|c| models.iter().position(|m| m == c))
                    .unwrap_or(0);
                self.picker = Some(ModelPicker {
                    models,
                    filter: String::new(),
                    selected,
                });
            }
            Err(e) => self.messages.push(AkhMessage::error(
                "model",
                format!("model list failed: {e}"),
            )),
        }
    }

    /// Issue one short-lived `_akh/*` control call over a fresh ACP connection
    /// (Unix socket preferred, else TCP), blocking the event loop for its
    /// duration. This does NOT touch the persistent chat session. Errors —
    /// including "no ACP endpoint advertised" — are mapped to a `String`.
    fn control_call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("failed to build runtime: {e}"))?;
        let method = method.to_string();
        if let Some(sock) = self.info.acp_socket.clone() {
            rt.block_on(crate::acp::ext_request_unix(&sock, &method, params))
                .map_err(|e| e.to_string())
        } else if let Some(tcp) = self.info.acp_tcp.clone() {
            rt.block_on(crate::acp::ext_request_tcp(&tcp, &method, params))
                .map_err(|e| e.to_string())
        } else {
            Err("daemon advertises no ACP endpoint for control calls".to_string())
        }
    }

    fn process_input(&mut self, input: &str) {
        // TUI commands start with '/'.
        if let Some(cmd) = input.strip_prefix('/') {
            self.handle_command(cmd);
            return;
        }

        // Show user input in the messages area.
        self.messages.push(AkhMessage::Prompt {
            question: format!("> {input}"),
        });

        self.remote.send_input(input);

        // Auto-scroll to bottom only if user was already near the bottom
        // (within 5 messages). This preserves the scroll position when reading
        // older messages and typing a new question.
        let at_bottom = self.scroll_offset + 5 >= self.messages.len().saturating_sub(1);
        if at_bottom {
            self.scroll_offset = self.messages.len().saturating_sub(1);
        }
    }

    fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "quit" | "q" | "exit" => {
                self.should_quit = true;
            }
            "help" | "h" => {
                self.messages.push(AkhMessage::system(
                    "Commands: /model [name], /effort <low|medium|high|off>, /goals, /status, \
                     /seed <pack>, /audit, /quit. /model with no name opens a picker. \
                     Tab toggles thinking steps."
                        .to_string(),
                ));
            }
            "model" => {
                let arg = parts.get(1).map(|s| s.trim()).filter(|s| !s.is_empty());
                match arg {
                    // `/model <name>` — set directly.
                    Some(name) => self.set_model(name),
                    // `/model` — list and open the picker.
                    None => self.open_model_picker(),
                }
            }
            "effort" => {
                let level = parts.get(1).map(|s| s.trim()).unwrap_or("");
                match level.to_ascii_lowercase().as_str() {
                    "low" | "medium" | "high" | "off" => {
                        let params = serde_json::json!({
                            "workspace": self.workspace,
                            "effort": level,
                        });
                        match self.control_call("_akh/model/set", params) {
                            Ok(v) => {
                                let e = v
                                    .get("effort")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("default");
                                self.messages.push(AkhMessage::system(format!(
                                    "Reasoning effort set to '{e}'."
                                )));
                            }
                            Err(err) => self.messages.push(AkhMessage::error(
                                "model",
                                format!("effort set failed: {err}"),
                            )),
                        }
                    }
                    _ => self.messages.push(AkhMessage::system(
                        "Usage: /effort <low|medium|high|off>".to_string(),
                    )),
                }
            }
            "audit" => {
                self.show_audit = !self.show_audit;
                let state = if self.show_audit { "ON" } else { "OFF" };
                self.messages.push(AkhMessage::system(format!(
                    "Audit log display: {state}"
                )));
            }
            "goals" | "status" | "seed" => {
                // Forward as a command to the server.
                self.remote.send_command(cmd);
            }
            _ => {
                self.messages.push(AkhMessage::system(format!(
                    "Unknown command: /{cmd}. Type /help for commands."
                )));
            }
        }
    }
}

/// Launch the TUI connected to a remote akhomed server.
#[cfg(feature = "acp")]
pub fn launch_remote(
    workspace: &str,
    info: &crate::client::ServerInfo,
) -> miette::Result<()> {
    let remote = remote::RemoteChat::connect(info, workspace)
        .map_err(|e| miette::miette!("failed to connect: {e}"))?;

    let mut tui = AkhTui::new_remote(workspace.to_string(), remote, info.clone());
    tui.run()
}
