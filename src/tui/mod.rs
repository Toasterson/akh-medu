//! Unified ratatui-based TUI — a pure ACP (remote) chat client.
//!
//! The TUI provides: scrollable message output, input area, status bar,
//! and TUI commands (e.g., `/grammar`, `/goals`, `/quit`). It connects to a
//! running akhomed daemon over the Agent Client Protocol; there is no longer a
//! local-engine backend (`akh` is a pure ACP client).
//!
//! The whole app is gated behind the `acp` feature: without it, this module
//! exports nothing.

#[cfg(feature = "acp")]
pub mod markdown;
pub mod remote;
pub mod sink;
#[cfg(feature = "acp")]
pub mod status;
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

/// One selectable entry in the [`CommandPalette`] — a slash command plus a
/// one-line description shown beside it.
#[cfg(feature = "acp")]
pub struct PaletteItem {
    /// The command token written after `/` (e.g. `model`).
    pub name: &'static str,
    /// One-line description shown beside the name in the palette.
    pub desc: &'static str,
    /// Whether the command takes an argument. `true` ⇒ selecting it pre-fills the
    /// input with `/name ` so the user can type the argument; `false` ⇒ it runs
    /// immediately on Enter.
    pub needs_arg: bool,
}

/// The full slash-command set — the single source of truth shared by the
/// `ctrl+p` palette, `/`-autocomplete, and `/help`.
#[cfg(feature = "acp")]
pub const COMMANDS: &[PaletteItem] = &[
    PaletteItem { name: "model", desc: "switch the active model (picker if no name)", needs_arg: false },
    PaletteItem { name: "effort", desc: "reasoning effort: low | medium | high | off", needs_arg: true },
    PaletteItem { name: "goals", desc: "show the akh's current goals", needs_arg: false },
    PaletteItem { name: "status", desc: "show workspace status", needs_arg: false },
    PaletteItem { name: "seed", desc: "seed a knowledge pack by name", needs_arg: true },
    PaletteItem { name: "audit", desc: "toggle audit-log display", needs_arg: false },
    PaletteItem { name: "sidebar", desc: "toggle the right sidebar (ctrl+b)", needs_arg: false },
    PaletteItem { name: "help", desc: "list commands", needs_arg: false },
    PaletteItem { name: "quit", desc: "exit the TUI", needs_arg: false },
];

/// A centered command-palette overlay (the `ctrl+p` popup): a substring-filtered
/// list of [`COMMANDS`] with a type-ahead filter, mirroring [`ModelPicker`].
#[cfg(feature = "acp")]
pub struct CommandPalette {
    /// Type-ahead filter: only commands whose name contains this (case-
    /// insensitive) are shown. Empty = show all.
    pub filter: String,
    /// Index into the *filtered* list of the currently highlighted row.
    pub selected: usize,
}

#[cfg(feature = "acp")]
impl CommandPalette {
    /// The commands currently visible given the type-ahead [`filter`]: a
    /// case-insensitive *substring* match over [`COMMANDS`] (substring, not
    /// prefix, so `eff` finds `effort` and `dit` finds `audit`). An empty filter
    /// shows everything.
    ///
    /// [`filter`]: CommandPalette::filter
    #[must_use]
    pub fn filtered(&self) -> Vec<&'static PaletteItem> {
        if self.filter.is_empty() {
            return COMMANDS.iter().collect();
        }
        let needle = self.filter.to_ascii_lowercase();
        COMMANDS
            .iter()
            .filter(|c| c.name.contains(needle.as_str()))
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
    /// Whether a prompt turn is in flight — drives the progress spinner. Set when
    /// a prompt is sent, cleared on [`remote::ChatUpdate::TurnEnd`].
    awaiting: bool,
    /// Animation tick for the spinner; advanced each event-loop iteration while
    /// [`awaiting`](Self::awaiting).
    spinner_frame: usize,
    /// Index in `messages` of the in-progress streamed answer, so successive
    /// answer deltas append to one growing message. `None` between turns.
    stream_idx: Option<usize>,
    /// Whether the right sidebar (context / goals / capabilities) is shown.
    /// Toggled with `ctrl+b`; auto-hidden by the renderer below a min width.
    show_sidebar: bool,
    /// A short title derived from the first user prompt this session, shown in
    /// the header in place of the bare workspace name. `None` until the first
    /// prompt is sent.
    session_title: Option<String>,
    /// The active model name, fetched once at startup via `_akh/model/list`
    /// (`current`) and refreshed when `/model` switches it. Shown in the footer.
    model_label: Option<String>,
    /// The reasoning-effort level last set this session via `/effort`, shown in
    /// the footer. `None` means the daemon default (unknown to the client).
    effort_label: Option<String>,
    /// `repo-branch[*]` of the client's working directory (the `*` marks a dirty
    /// tree), computed once at startup. `None` when not in a git repo.
    git_branch: Option<String>,
    /// Active command-palette overlay (the `ctrl+p` popup). When `Some`, key
    /// input is routed to the palette instead of the chat input.
    palette: Option<CommandPalette>,
    /// Latest sidebar snapshot from the status poller (`_akh/status`). `None`
    /// until the first poll lands; drives the live Context/Goals/Capabilities
    /// panels.
    snapshot: Option<status::StatusSnapshot>,
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
            awaiting: false,
            spinner_frame: 0,
            stream_idx: None,
            show_sidebar: true,
            session_title: None,
            model_label: None,
            effort_label: None,
            git_branch: detect_git_branch(),
            palette: None,
            snapshot: None,
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

        // Fetch the active model once for the footer (best-effort — a daemon
        // that lists nothing just leaves the footer model blank).
        self.model_label = self.fetch_current_model();

        loop {
            // Drain pending messages from backend.
            self.drain_backend_messages();

            // Pick up the latest sidebar snapshot from the status poller.
            if let Some(snap) = self.remote.try_recv_snapshot() {
                self.snapshot = Some(snap);
            }

            // Advance the spinner while a turn is in flight (the event poll
            // below ticks ~10×/s when idle, giving a steady animation).
            if self.awaiting {
                self.spinner_frame = self.spinner_frame.wrapping_add(1);
            }

            terminal
                .draw(|frame| widgets::render(frame, self))
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

    /// Drain pending streaming updates from the backend, accumulating answer /
    /// thought deltas into growing messages and stopping the spinner at the end
    /// of the turn.
    fn drain_backend_messages(&mut self) {
        while let Some(update) = self.remote.try_recv() {
            match update {
                remote::ChatUpdate::AnswerDelta(delta) => self.append_answer(&delta),
                remote::ChatUpdate::ThoughtDelta(delta) => self.append_thought(&delta),
                remote::ChatUpdate::Notice(msg) => {
                    // A discrete out-of-band message breaks the streaming runs.
                    self.stream_idx = None;
                    if self.show_audit || !matches!(msg, AkhMessage::AuditLog { .. }) {
                        self.messages.push(msg);
                        self.autoscroll();
                    }
                }
                remote::ChatUpdate::TurnEnd => {
                    self.awaiting = false;
                    self.stream_idx = None;
                    self.autoscroll();
                }
            }
        }
    }

    /// Append a delta to the in-progress streamed answer, creating the answer
    /// message on the first delta of a run.
    fn append_answer(&mut self, delta: &str) {
        match self.stream_idx {
            Some(i) if matches!(self.messages.get(i), Some(AkhMessage::Narrative { .. })) => {
                if let Some(AkhMessage::Narrative { text, .. }) = self.messages.get_mut(i) {
                    text.push_str(delta);
                }
            }
            _ => {
                self.messages
                    .push(AkhMessage::narrative(delta.to_string(), "acp"));
                self.stream_idx = Some(self.messages.len() - 1);
            }
        }
        self.autoscroll();
    }

    /// Append a delta to the current "thinking" block (reasoning + tool steps),
    /// coalescing consecutive thought deltas into one collapsible message. A
    /// thought breaks the answer run, so a later answer delta starts fresh.
    fn append_thought(&mut self, delta: &str) {
        self.stream_idx = None;
        let coalesce = matches!(
            self.messages.last(),
            Some(AkhMessage::ToolResult { tool, .. }) if tool == "thinking"
        );
        if coalesce {
            if let Some(AkhMessage::ToolResult { output, .. }) = self.messages.last_mut() {
                output.push_str(delta);
            }
        } else {
            self.messages
                .push(AkhMessage::tool_result("thinking", true, delta.to_string()));
        }
        self.autoscroll();
    }

    /// Snap the view to the newest message if the user was already near the
    /// bottom; preserves scroll position when reading back through history.
    fn autoscroll(&mut self) {
        let last = self.messages.len().saturating_sub(1);
        if self.scroll_offset + 5 >= last {
            self.scroll_offset = last;
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
        if self.palette.is_some() {
            self.handle_palette_key(code);
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
            KeyCode::Char('b') if modifiers.contains(KeyModifiers::CONTROL) => {
                // Toggle the right sidebar.
                self.show_sidebar = !self.show_sidebar;
            }
            KeyCode::Char('p') if modifiers.contains(KeyModifiers::CONTROL) => {
                // Open the command palette.
                self.palette = Some(CommandPalette {
                    filter: String::new(),
                    selected: 0,
                });
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

    /// Route a key to the open command palette. Mirrors [`handle_picker_key`]:
    /// Up/Down move the (clamped) selection within the filtered view; printable
    /// chars extend the substring filter and Backspace trims it (both reset the
    /// selection to the top); Enter runs the highlighted command (see
    /// [`run_palette_selection`]); Esc closes without running.
    ///
    /// [`handle_picker_key`]: Self::handle_picker_key
    /// [`run_palette_selection`]: Self::run_palette_selection
    fn handle_palette_key(&mut self, code: KeyCode) {
        let Some(palette) = self.palette.as_mut() else {
            return;
        };
        match code {
            KeyCode::Up => {
                palette.selected = palette.selected.saturating_sub(1);
            }
            KeyCode::Down => {
                let len = palette.filtered().len();
                if len > 0 {
                    palette.selected = (palette.selected + 1).min(len - 1);
                }
            }
            KeyCode::Char(c) => {
                palette.filter.push(c);
                palette.selected = 0;
            }
            KeyCode::Backspace => {
                palette.filter.pop();
                palette.selected = 0;
            }
            KeyCode::Esc => {
                self.palette = None;
            }
            KeyCode::Enter => {
                // Resolve the chosen command out of the *filtered* view, close
                // the palette, then act on it.
                let chosen = palette
                    .filtered()
                    .get(palette.selected)
                    .map(|item| (item.name, item.needs_arg));
                self.palette = None;
                if let Some((name, needs_arg)) = chosen {
                    self.run_palette_selection(name, needs_arg);
                }
            }
            _ => {}
        }
    }

    /// Act on a palette choice: a command that takes an argument pre-fills the
    /// input with `/name ` (so the user can type the argument and press Enter);
    /// an argument-less command runs immediately via [`handle_command`].
    ///
    /// [`handle_command`]: Self::handle_command
    fn run_palette_selection(&mut self, name: &str, needs_arg: bool) {
        if needs_arg {
            self.input_buffer = format!("/{name} ");
        } else {
            self.handle_command(name);
        }
    }

    /// Fetch the daemon's current model name via `_akh/model/list` (`current`)
    /// for the footer. Best-effort: any error or a daemon that lists no current
    /// model yields `None` (the footer then omits the model).
    fn fetch_current_model(&self) -> Option<String> {
        let params = serde_json::json!({ "workspace": self.workspace });
        let v = self.control_call("_akh/model/list", params).ok()?;
        v.get("current")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    }

    /// Fetch a fresh `_akh/status` snapshot synchronously for the `/goals` and
    /// `/status` commands. Best-effort: `None` on any transport/parse error.
    fn fetch_status(&self) -> Option<status::StatusSnapshot> {
        let v = self.control_call("_akh/status", serde_json::json!({})).ok()?;
        serde_json::from_value(v).ok()
    }

    /// Print the akh's current goals into the message stream (also shown live in
    /// the sidebar).
    fn show_goals(&mut self) {
        let Some(snap) = self.fetch_status() else {
            self.messages
                .push(AkhMessage::error("status", "could not fetch goals"));
            return;
        };
        if snap.goals.is_empty() {
            let note = if snap.stale {
                " (agent busy — try again)"
            } else {
                ""
            };
            self.messages
                .push(AkhMessage::system(format!("No goals.{note}")));
            return;
        }
        self.messages
            .push(AkhMessage::system(format!("Goals ({}):", snap.goals.len())));
        for g in &snap.goals {
            self.messages.push(AkhMessage::system(format!(
                "  • [{}] {} (p{})",
                g.status_label(),
                g.description,
                g.priority
            )));
        }
        self.autoscroll();
    }

    /// Print a one-line workspace/agent status summary into the message stream.
    fn show_status(&mut self) {
        let Some(snap) = self.fetch_status() else {
            self.messages
                .push(AkhMessage::error("status", "could not fetch status"));
            return;
        };
        let cycle = snap
            .cycle
            .map_or_else(|| "—".to_string(), |c| c.to_string());
        let stale = if snap.stale { " (stale: agent busy)" } else { "" };
        self.messages.push(AkhMessage::system(format!(
            "Status — workspace {ws} · cycle {cycle} · {g} goals · {t} tools · {s} skills · {p} projects{stale}",
            ws = self.workspace,
            g = snap.goals.len(),
            t = snap.tools.len(),
            s = snap.skills.len(),
            p = snap.projects.len(),
        )));
        self.autoscroll();
    }

    /// Issue one `_akh/model/set` to switch the running model, reporting the
    /// outcome as a System message.
    fn set_model(&mut self, model: &str) {
        let params = serde_json::json!({ "workspace": self.workspace, "model": model });
        match self.control_call("_akh/model/set", params) {
            Ok(v) => {
                let m = v
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(model)
                    .to_string();
                self.model_label = Some(m.clone());
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

        // Derive the session title from the first real prompt (shown in the
        // header in place of the bare workspace name).
        if self.session_title.is_none() {
            self.session_title = Some(summarize_title(input));
        }

        // Show user input in the messages area.
        self.messages.push(AkhMessage::Prompt {
            question: format!("> {input}"),
        });

        // Begin a new turn: a fresh answer run, spinner on.
        self.stream_idx = None;
        self.awaiting = true;
        self.spinner_frame = 0;
        self.remote.send_input(input);
        self.autoscroll();
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
                     /seed <pack>, /audit, /sidebar, /quit. /model with no name opens a picker. \
                     Keys: Tab toggles thinking · ctrl+b toggles sidebar · ctrl+p opens the \
                     command palette."
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
                                    .unwrap_or("default")
                                    .to_string();
                                self.effort_label = Some(e.clone());
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
            "sidebar" => {
                self.show_sidebar = !self.show_sidebar;
                let state = if self.show_sidebar { "shown" } else { "hidden" };
                self.messages
                    .push(AkhMessage::system(format!("Sidebar {state}.")));
            }
            "goals" => self.show_goals(),
            "status" => self.show_status(),
            "seed" => {
                // Seed packs over ACP are a later increment.
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

/// Truncate a prompt into a short session title for the header: first line,
/// whitespace-collapsed, capped at ~40 chars with an ellipsis.
#[cfg(feature = "acp")]
fn summarize_title(prompt: &str) -> String {
    const MAX: usize = 40;
    let first_line = prompt.lines().next().unwrap_or(prompt);
    let collapsed = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > MAX {
        let head: String = collapsed.chars().take(MAX - 1).collect();
        format!("{head}…")
    } else {
        collapsed
    }
}

/// Best-effort `branch[*]` of the client's current git repo for the footer: the
/// abbreviated branch name, suffixed with `*` when the working tree is dirty.
/// `None` outside a repo or if `git` is unavailable.
#[cfg(feature = "acp")]
fn detect_git_branch() -> Option<String> {
    use std::process::Command;
    let branch_out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !branch_out.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&branch_out.stdout).trim().to_string();
    if branch.is_empty() {
        return None;
    }
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|o| !o.stdout.is_empty());
    Some(if dirty { format!("{branch}*") } else { branch })
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

#[cfg(all(test, feature = "acp"))]
mod tests {
    use super::{CommandPalette, COMMANDS, summarize_title};

    #[test]
    fn title_keeps_short_prompts_verbatim() {
        assert_eq!(summarize_title("what is the sun"), "what is the sun");
    }

    #[test]
    fn title_collapses_whitespace_and_takes_first_line() {
        assert_eq!(
            summarize_title("  hello   world \n second line"),
            "hello world"
        );
    }

    #[test]
    fn title_truncates_long_prompts_with_ellipsis() {
        let long = "a".repeat(80);
        let title = summarize_title(&long);
        assert_eq!(title.chars().count(), 40);
        assert!(title.ends_with('…'));
    }

    #[test]
    fn palette_empty_filter_shows_all_commands() {
        let p = CommandPalette { filter: String::new(), selected: 0 };
        assert_eq!(p.filtered().len(), COMMANDS.len());
    }

    #[test]
    fn palette_substring_filter_matches_anywhere_in_name() {
        // "dit" is a substring of "audit" but not a prefix.
        let p = CommandPalette { filter: "dit".to_string(), selected: 0 };
        let names: Vec<&str> = p.filtered().iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["audit"]);
    }

    #[test]
    fn palette_no_match_yields_empty() {
        let p = CommandPalette { filter: "zzz".to_string(), selected: 0 };
        assert!(p.filtered().is_empty());
    }

    #[test]
    fn palette_filter_is_case_insensitive() {
        let p = CommandPalette { filter: "MODEL".to_string(), selected: 0 };
        let names: Vec<&str> = p.filtered().iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["model"]);
    }
}
