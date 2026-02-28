//! Unified ratatui-based TUI replacing all three chat modes.
//!
//! The TUI provides: scrollable message output, input area, status bar,
//! and TUI commands (e.g., `/grammar`, `/workspace`, `/goals`, `/quit`).
//!
//! Supports two backends:
//! - **Local**: direct agent + engine (current behavior)
//! - **Remote**: WebSocket connection to akhomed (feature-gated behind `daemon`)

pub mod remote;
pub mod sink;
pub mod widgets;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use miette::IntoDiagnostic;

use crate::agent::{Agent, AgentConfig, IdleScheduler, InboundHandle};
use crate::chat::ChatProcessor;
use crate::engine::Engine;
use crate::message::AkhMessage;

/// Local backend state: agent + engine bundled for the TUI.
struct LocalBackend {
    agent: Box<Agent>,
    engine: Arc<Engine>,
    tui_sink: Arc<sink::TuiSink>,
    idle_scheduler: IdleScheduler,
    operator_handle: InboundHandle,
    chat_processor: ChatProcessor,
}

/// The chat backend: either a local agent+engine or a remote WS connection.
enum ChatBackend {
    /// Direct local engine and agent.
    Local(Box<LocalBackend>),
    /// WebSocket connection to akhomed.
    #[cfg(feature = "daemon")]
    Remote {
        remote: remote::RemoteChat,
    },
}

/// TUI application state.
pub struct AkhTui {
    workspace: String,
    backend: ChatBackend,
    messages: Vec<AkhMessage>,
    input_buffer: String,
    scroll_offset: usize,
    should_quit: bool,
    /// Whether audit log entries are displayed in the message stream.
    show_audit: bool,
}

impl AkhTui {
    /// Create a new TUI instance with a local engine backend.
    pub fn new_local(workspace: String, engine: Arc<Engine>, mut agent: Agent) -> Self {
        let tui_sink = Arc::new(sink::TuiSink::new());

        // Set up the operator channel so input flows through the channel abstraction.
        agent.set_sink(tui_sink.clone());
        let operator_handle = agent.setup_operator_channel();

        // Wire audit ledger to TUI sink so entries appear live.
        if let Some(ledger) = engine.audit_ledger() {
            ledger.set_sink(tui_sink.clone());
        }

        // Restore NLU ranker from durable storage if available,
        // and attempt to load ML/LLM models from data_dir.
        let data_dir = engine.config().data_dir.as_deref();
        let nlu_pipeline = engine
            .store()
            .get_meta(b"nlu_ranker_state")
            .ok()
            .flatten()
            .and_then(|bytes| crate::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
            .map(|ranker| crate::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir))
            .unwrap_or_else(|| crate::nlu::NluPipeline::new_with_models(data_dir));

        let chat_processor = ChatProcessor::new(&engine, nlu_pipeline);

        Self {
            workspace,
            backend: ChatBackend::Local(Box::new(LocalBackend {
                agent: Box::new(agent),
                engine,
                tui_sink,
                idle_scheduler: IdleScheduler::default(),
                operator_handle,
                chat_processor,
            })),
            messages: vec![AkhMessage::system(
                "Welcome to akh. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
            show_audit: true,
        }
    }

    /// Create a new TUI instance with a remote WS backend.
    #[cfg(feature = "daemon")]
    pub fn new_remote(workspace: String, remote: remote::RemoteChat) -> Self {
        Self {
            workspace,
            backend: ChatBackend::Remote { remote },
            messages: vec![AkhMessage::system(
                "Connected to akhomed. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
            show_audit: true,
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

        loop {
            // Drain pending messages from backend.
            self.drain_backend_messages();

            terminal
                .draw(|frame| {
                    let (cycle_count, symbol_count, goal_count) = self.status_counts();

                    let grammar = self.grammar();
                    widgets::render(
                        frame,
                        &self.workspace,
                        &grammar,
                        &self.messages,
                        &self.input_buffer,
                        self.scroll_offset,
                        cycle_count,
                        symbol_count,
                        goal_count,
                    );
                })
                .into_diagnostic()?;

            if self.should_quit || sigterm_flag.load(Ordering::Relaxed) {
                break;
            }

            // Poll for events.
            if event::poll(Duration::from_millis(100)).into_diagnostic()? {
                if let Event::Key(key) = event::read().into_diagnostic()? {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    self.handle_key(key.code, key.modifiers);
                }
            } else {
                self.on_idle();
            }
        }

        self.on_exit()?;

        ratatui::restore();
        Ok(())
    }

    /// Drain pending messages from the backend.
    fn drain_backend_messages(&mut self) {
        let show_audit = self.show_audit;
        match self.backend {
            ChatBackend::Local(ref local) => {
                let pending = local.tui_sink.drain();
                self.messages.extend(
                    pending
                        .into_iter()
                        .filter(|m| show_audit || !matches!(m, AkhMessage::AuditLog { .. })),
                );
            }
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { ref mut remote } => {
                while let Some(msg) = remote.try_recv() {
                    if show_audit || !matches!(msg, AkhMessage::AuditLog { .. }) {
                        self.messages.push(msg);
                    }
                }
            }
        }
    }

    /// Get (cycle_count, symbol_count, goal_count) for the status bar.
    fn status_counts(&self) -> (u64, usize, usize) {
        match self.backend {
            ChatBackend::Local(ref local) => {
                let symbol_count = local.engine.all_symbols().len();
                let goal_count = local
                    .agent
                    .goals()
                    .iter()
                    .filter(|g| matches!(g.status, crate::agent::GoalStatus::Active))
                    .count();
                (local.agent.cycle_count(), symbol_count, goal_count)
            }
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { .. } => {
                // Remote mode: status info comes from server messages.
                (0, 0, 0)
            }
        }
    }

    /// Get the current grammar archetype name.
    fn grammar(&self) -> String {
        match self.backend {
            ChatBackend::Local(ref local) => local.chat_processor.grammar().to_string(),
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { .. } => "narrative".to_string(),
        }
    }

    /// Called on idle (no key events).
    #[allow(irrefutable_let_patterns)]
    fn on_idle(&mut self) {
        if let ChatBackend::Local(ref mut local) = self.backend
            && let Some(result) = local.idle_scheduler.tick(&mut local.agent)
        {
            self.messages.push(AkhMessage::system(format!(
                "[idle:{}] {}",
                result.task, result.summary,
            )));
        }
    }

    /// Called on exit — persist session, etc.
    #[allow(irrefutable_let_patterns)]
    fn on_exit(&mut self) -> miette::Result<()> {
        if let ChatBackend::Local(ref mut local) = self.backend {
            local.agent.persist_session().into_diagnostic()?;
            local.chat_processor.persist_nlu_state(&local.engine);
        }
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

        match self.backend {
            ChatBackend::Local(ref local) => {
                // Push through the operator channel, then process locally.
                local.operator_handle.push_text(input);
                self.process_inbound_local();
            }
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { ref remote, .. } => {
                remote.send_input(input);
            }
        }

        // Auto-scroll to bottom.
        self.scroll_offset = self.messages.len().saturating_sub(1);
    }

    /// Drain all pending inbound messages from the agent's channel registry
    /// and dispatch each one through intent classification.
    #[allow(irrefutable_let_patterns)]
    fn process_inbound_local(&mut self) {
        // Drain and process inbound messages in two passes to avoid borrow conflicts.
        let texts: Vec<String> = {
            let ChatBackend::Local(ref mut local) = self.backend else {
                return;
            };

            let inbound = local.agent.drain_inbound();
            let mut texts = Vec::new();
            for msg in &inbound {
                // Auto-register interlocutor on first interaction.
                let _ = local.agent.ensure_interlocutor(
                    &msg.sender,
                    &msg.channel_id,
                    crate::agent::ChannelKind::Operator,
                );
                if let Some(text) = msg.text() {
                    texts.push(text.to_string());
                }
            }
            texts
        };

        for text in &texts {
            self.process_input_local(text);
        }
    }

    /// Process input against the local engine/agent via the unified ChatProcessor.
    #[allow(irrefutable_let_patterns)]
    fn process_input_local(&mut self, input: &str) {
        let ChatBackend::Local(ref mut local) = self.backend else {
            return;
        };
        let responses =
            local
                .chat_processor
                .process_input(input, &mut local.agent, &local.engine);
        self.messages.extend(responses);
    }

    fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        match parts[0] {
            "quit" | "q" => {
                self.should_quit = true;
            }
            "help" | "h" => {
                self.messages.push(AkhMessage::system(
                    "Commands: /grammar <name>, /goals, /status, /seed <pack>, /audit, /quit".to_string(),
                ));
            }
            "grammar" | "g" => {
                if let Some(name) = parts.get(1) {
                    #[allow(irrefutable_let_patterns)]
                    if let ChatBackend::Local(ref mut local) = self.backend {
                        local.chat_processor.set_grammar(name);
                    }
                    self.messages
                        .push(AkhMessage::system(format!("Grammar switched to: {name}")));
                } else {
                    let current = self.grammar();
                    self.messages.push(AkhMessage::system(format!(
                        "Current grammar: {current}. Use /grammar <name> to switch.",
                    )));
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
                match self.backend {
                    ChatBackend::Local(..) => self.handle_command_local(parts[0], parts.get(1).copied()),
                    #[cfg(feature = "daemon")]
                    ChatBackend::Remote { ref remote, .. } => {
                        // Forward as a command to the server.
                        remote.send_command(cmd);
                    }
                }
            }
            _ => {
                self.messages.push(AkhMessage::system(format!(
                    "Unknown command: /{cmd}. Type /help for commands."
                )));
            }
        }
    }

    /// Handle /goals, /status, /seed locally.
    #[allow(irrefutable_let_patterns)]
    fn handle_command_local(&mut self, cmd: &str, arg: Option<&str>) {
        let ChatBackend::Local(ref local) = self.backend else {
            return;
        };
        let agent = &local.agent;
        let engine = &local.engine;

        match cmd {
            "goals" => {
                let goals = agent.goals();
                if goals.is_empty() {
                    self.messages
                        .push(AkhMessage::system("No goals.".to_string()));
                } else {
                    for g in goals {
                        self.messages.push(AkhMessage::goal_progress(
                            &g.description,
                            format!(
                                "[P{}] {} ({}, {} cycles)",
                                g.priority, g.description, g.status, g.cycles_worked,
                            ),
                        ));
                    }
                }
            }
            "status" => {
                let info = engine.info();
                self.messages.push(AkhMessage::system(format!("{info}")));
            }
            "seed" => {
                if let Some(pack_name) = arg {
                    let registry = crate::seeds::SeedRegistry::bundled();
                    match registry.apply(pack_name, engine) {
                        Ok(report) => {
                            if report.already_applied {
                                self.messages.push(AkhMessage::system(format!(
                                    "Seed \"{}\" already applied.",
                                    report.id
                                )));
                            } else {
                                self.messages.push(AkhMessage::system(format!(
                                    "Applied seed \"{}\": {} triples.",
                                    report.id, report.triples_applied,
                                )));
                            }
                        }
                        Err(e) => {
                            self.messages.push(AkhMessage::error("seed", e.to_string()));
                        }
                    }
                } else {
                    self.messages
                        .push(AkhMessage::system("Usage: /seed <pack-name>".to_string()));
                }
            }
            _ => {}
        }
    }

}

/// Launch the TUI with a local engine and workspace.
pub fn launch(
    workspace: &str,
    engine: Arc<Engine>,
    agent_config: AgentConfig,
    fresh: bool,
) -> miette::Result<()> {
    let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
        Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
    } else {
        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
    };

    if fresh {
        agent.clear_goals();
    }

    let mut tui = AkhTui::new_local(workspace.to_string(), engine, agent);
    tui.run()
}

/// Launch the TUI connected to a remote akhomed server.
#[cfg(feature = "daemon")]
pub fn launch_remote(
    workspace: &str,
    info: &crate::client::ServerInfo,
) -> miette::Result<()> {
    let remote = remote::RemoteChat::connect(info, workspace)
        .map_err(|e| miette::miette!("failed to connect: {e}"))?;

    let mut tui = AkhTui::new_remote(workspace.to_string(), remote);
    tui.run()
}
