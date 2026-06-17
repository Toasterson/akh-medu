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

/// TUI application state (ACP client).
#[cfg(feature = "acp")]
pub struct AkhTui {
    workspace: String,
    remote: remote::RemoteChat,
    messages: Vec<AkhMessage>,
    input_buffer: String,
    scroll_offset: usize,
    should_quit: bool,
    /// Whether audit log entries are displayed in the message stream.
    show_audit: bool,
    /// Whether tool/agent "thinking" steps are expanded (default: collapsed).
    show_thinking: bool,
}

#[cfg(feature = "acp")]
impl AkhTui {
    /// Create a new TUI instance with a remote ACP backend.
    pub fn new_remote(workspace: String, remote: remote::RemoteChat) -> Self {
        Self {
            workspace,
            remote,
            messages: vec![AkhMessage::system(
                "Connected to akhomed. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
            show_audit: true,
            show_thinking: false,
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
            "quit" | "q" => {
                self.should_quit = true;
            }
            "help" | "h" => {
                self.messages.push(AkhMessage::system(
                    "Commands: /goals, /status, /seed <pack>, /audit, /quit. Tab toggles thinking steps."
                        .to_string(),
                ));
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

    let mut tui = AkhTui::new_remote(workspace.to_string(), remote);
    tui.run()
}
