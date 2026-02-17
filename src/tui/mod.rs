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
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use miette::IntoDiagnostic;

use crate::agent::{Agent, AgentConfig, IdleScheduler};
use crate::engine::Engine;
use crate::message::AkhMessage;

/// The chat backend: either a local agent+engine or a remote WS connection.
enum ChatBackend {
    /// Direct local engine and agent.
    Local {
        agent: Agent,
        engine: Arc<Engine>,
        tui_sink: Arc<sink::TuiSink>,
        idle_scheduler: IdleScheduler,
    },
    /// WebSocket connection to akhomed.
    #[cfg(feature = "daemon")]
    Remote {
        remote: remote::RemoteChat,
    },
}

/// TUI application state.
pub struct AkhTui {
    workspace: String,
    grammar: String,
    backend: ChatBackend,
    messages: Vec<AkhMessage>,
    input_buffer: String,
    scroll_offset: usize,
    should_quit: bool,
}

impl AkhTui {
    /// Create a new TUI instance with a local engine backend.
    pub fn new_local(workspace: String, engine: Arc<Engine>, agent: Agent) -> Self {
        let grammar = engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .map(|p| p.persona.grammar_preference.clone())
            .unwrap_or_else(|| "narrative".to_string());

        let tui_sink = Arc::new(sink::TuiSink::new());

        Self {
            workspace,
            grammar,
            backend: ChatBackend::Local {
                agent,
                engine,
                tui_sink,
                idle_scheduler: IdleScheduler::default(),
            },
            messages: vec![AkhMessage::system(
                "Welcome to akh. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
        }
    }

    /// Create a new TUI instance with a remote WS backend.
    #[cfg(feature = "daemon")]
    pub fn new_remote(workspace: String, remote: remote::RemoteChat) -> Self {
        Self {
            workspace,
            grammar: "narrative".to_string(),
            backend: ChatBackend::Remote { remote },
            messages: vec![AkhMessage::system(
                "Connected to akhomed. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
        }
    }

    /// Run the TUI event loop.
    pub fn run(&mut self) -> miette::Result<()> {
        let mut terminal = ratatui::init();

        // Set the agent's sink (local only).
        if let ChatBackend::Local {
            ref mut agent,
            ref tui_sink,
            ..
        } = self.backend
        {
            agent.set_sink(tui_sink.clone());
        }

        loop {
            // Drain pending messages from backend.
            self.drain_backend_messages();

            terminal
                .draw(|frame| {
                    let (cycle_count, symbol_count, goal_count) = self.status_counts();

                    widgets::render(
                        frame,
                        &self.workspace,
                        &self.grammar,
                        &self.messages,
                        &self.input_buffer,
                        self.scroll_offset,
                        cycle_count,
                        symbol_count,
                        goal_count,
                    );
                })
                .into_diagnostic()?;

            if self.should_quit {
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
        match self.backend {
            ChatBackend::Local { ref tui_sink, .. } => {
                let pending = tui_sink.drain();
                self.messages.extend(pending);
            }
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { ref mut remote } => {
                while let Some(msg) = remote.try_recv() {
                    self.messages.push(msg);
                }
            }
        }
    }

    /// Get (cycle_count, symbol_count, goal_count) for the status bar.
    fn status_counts(&self) -> (u64, usize, usize) {
        match self.backend {
            ChatBackend::Local {
                ref agent,
                ref engine,
                ..
            } => {
                let symbol_count = engine.all_symbols().len();
                let goal_count = agent
                    .goals()
                    .iter()
                    .filter(|g| matches!(g.status, crate::agent::GoalStatus::Active))
                    .count();
                (agent.cycle_count(), symbol_count, goal_count)
            }
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { .. } => {
                // Remote mode: status info comes from server messages.
                (0, 0, 0)
            }
        }
    }

    /// Called on idle (no key events).
    fn on_idle(&mut self) {
        if let ChatBackend::Local {
            ref mut agent,
            ref mut idle_scheduler,
            ..
        } = self.backend
        {
            if let Some(result) = idle_scheduler.tick(agent) {
                self.messages.push(AkhMessage::system(format!(
                    "[idle:{}] {}",
                    result.task, result.summary,
                )));
            }
        }
    }

    /// Called on exit — persist session, etc.
    fn on_exit(&mut self) -> miette::Result<()> {
        if let ChatBackend::Local { ref mut agent, .. } = self.backend {
            agent.persist_session().into_diagnostic()?;
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
            ChatBackend::Local { .. } => self.process_input_local(input),
            #[cfg(feature = "daemon")]
            ChatBackend::Remote { ref remote, .. } => {
                remote.send_input(input);
            }
        }

        // Auto-scroll to bottom.
        self.scroll_offset = self.messages.len().saturating_sub(1);
    }

    /// Process input against the local engine/agent.
    fn process_input_local(&mut self, input: &str) {
        let ChatBackend::Local {
            ref mut agent,
            ref engine,
            ..
        } = self.backend
        else {
            return;
        };

        let intent = crate::agent::classify_intent(input);

        match intent {
            crate::agent::UserIntent::SetGoal { description } => {
                match agent.add_goal(&description, 128, "User-directed goal") {
                    Ok(id) => {
                        self.messages.push(AkhMessage::system(format!(
                            "Goal added: \"{description}\" (id: {})",
                            id.get()
                        )));
                        // Run a few cycles.
                        for _ in 0..5 {
                            match agent.run_cycle() {
                                Ok(result) => {
                                    self.messages.push(AkhMessage::tool_result(
                                        &result.decision.chosen_tool,
                                        result.action_result.tool_output.success,
                                        &result.action_result.tool_output.result,
                                    ));
                                }
                                Err(_) => break,
                            }
                            // Stop if no more active goals.
                            let active = crate::agent::goal::active_goals(agent.goals());
                            if active.is_empty() {
                                break;
                            }
                        }
                        // Synthesize findings.
                        let summary = agent
                            .synthesize_findings_with_grammar(&description, &self.grammar);
                        if !summary.overview.is_empty() {
                            self.messages
                                .push(AkhMessage::narrative(&summary.overview, &self.grammar));
                        }
                        for section in &summary.sections {
                            self.messages.push(AkhMessage::narrative(
                                format!("## {}\n{}", section.heading, section.prose),
                                &self.grammar,
                            ));
                        }
                        for gap in &summary.gaps {
                            self.messages.push(AkhMessage::gap("(unknown)", gap));
                        }
                    }
                    Err(e) => {
                        self.messages.push(AkhMessage::error("goal", e.to_string()));
                    }
                }
            }
            crate::agent::UserIntent::Query { subject, original_input, question_word } => {
                // Try discourse-aware response first, fall back to synthesis.
                let discourse_result = crate::grammar::discourse::resolve_discourse(
                    &subject,
                    question_word,
                    &original_input,
                    engine,
                );
                let handled = if let Ok(ref ctx) = discourse_result {
                    let from_triples = engine.triples_from(ctx.subject_id);
                    let to_triples = engine.triples_to(ctx.subject_id);
                    let mut all_triples = from_triples;
                    all_triples.extend(to_triples);
                    if let Some(discourse_tree) =
                        crate::grammar::discourse::build_discourse_response(
                            &all_triples, ctx, engine,
                        )
                    {
                        let registry = crate::grammar::GrammarRegistry::new();
                        if let Ok(prose) = registry.linearize(&self.grammar, &discourse_tree) {
                            if !prose.trim().is_empty() {
                                self.messages.push(AkhMessage::narrative(&prose, &self.grammar));
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !handled {
                    // Fallback: existing synthesis path.
                    match engine.resolve_symbol(&subject) {
                        Ok(sym_id) => {
                            let from_triples = engine.triples_from(sym_id);
                            let to_triples = engine.triples_to(sym_id);
                            if from_triples.is_empty() && to_triples.is_empty() {
                                self.messages.push(AkhMessage::system(format!(
                                    "No facts found for \"{subject}\"."
                                )));
                            } else {
                                let mut all_triples = from_triples;
                                all_triples.extend(to_triples);
                                let summary =
                                    crate::agent::synthesize::synthesize_from_triples(
                                        &subject,
                                        &all_triples,
                                        engine,
                                        &self.grammar,
                                    );
                                if !summary.overview.is_empty() {
                                    self.messages.push(AkhMessage::narrative(
                                        &summary.overview,
                                        &self.grammar,
                                    ));
                                }
                                for section in &summary.sections {
                                    self.messages.push(AkhMessage::narrative(
                                        format!("## {}\n{}", section.heading, section.prose),
                                        &self.grammar,
                                    ));
                                }
                                for gap in &summary.gaps {
                                    self.messages.push(AkhMessage::gap("(unknown)", gap));
                                }
                            }
                        }
                        Err(_) => {
                            self.messages.push(AkhMessage::system(format!(
                                "Symbol \"{subject}\" not found."
                            )));
                        }
                    }
                }
            }
            crate::agent::UserIntent::Assert { text } => {
                use crate::agent::tool::Tool;
                let tool_input = crate::agent::ToolInput::new().with_param("text", &text);
                match crate::agent::tools::TextIngestTool.execute(engine, tool_input) {
                    Ok(output) => {
                        self.messages.push(AkhMessage::tool_result(
                            "text_ingest",
                            output.success,
                            &output.result,
                        ));
                    }
                    Err(e) => {
                        self.messages
                            .push(AkhMessage::error("ingest", e.to_string()));
                    }
                }
            }
            crate::agent::UserIntent::RunAgent { cycles } => {
                let n = cycles.unwrap_or(1);
                for _ in 0..n {
                    match agent.run_cycle() {
                        Ok(result) => {
                            self.messages.push(AkhMessage::tool_result(
                                &result.decision.chosen_tool,
                                result.action_result.tool_output.success,
                                &result.action_result.tool_output.result,
                            ));
                        }
                        Err(e) => {
                            self.messages
                                .push(AkhMessage::error("cycle", e.to_string()));
                            break;
                        }
                    }
                }
            }
            crate::agent::UserIntent::ShowStatus => {
                let goals = agent.goals();
                if goals.is_empty() {
                    self.messages
                        .push(AkhMessage::system("No active goals.".to_string()));
                } else {
                    for g in goals {
                        self.messages.push(AkhMessage::goal_progress(
                            &g.description,
                            format!("{}", g.status),
                        ));
                    }
                }
                self.messages.push(AkhMessage::system(format!(
                    "Cycles: {}, WM entries: {}, Triples: {}",
                    agent.cycle_count(),
                    agent.working_memory().len(),
                    engine.all_triples().len(),
                )));
            }
            crate::agent::UserIntent::RenderHiero { entity } => {
                let render_config = crate::glyph::RenderConfig {
                    color: false,
                    notation: crate::glyph::NotationConfig {
                        use_pua: crate::glyph::catalog::font_available(),
                        show_confidence: true,
                        show_provenance: false,
                        show_sigils: true,
                        compact: false,
                    },
                    ..Default::default()
                };

                if let Some(ref name) = entity {
                    match engine.resolve_symbol(name) {
                        Ok(sym_id) => match engine.extract_subgraph(&[sym_id], 1) {
                            Ok(result) if !result.triples.is_empty() => {
                                let rendered = crate::glyph::render::render_to_terminal(
                                    engine,
                                    &result.triples,
                                    &render_config,
                                );
                                self.messages.push(AkhMessage::system(rendered));
                            }
                            _ => {
                                self.messages.push(AkhMessage::system(format!(
                                    "No triples found around \"{name}\"."
                                )));
                            }
                        },
                        Err(_) => {
                            self.messages
                                .push(AkhMessage::system(format!("Symbol \"{name}\" not found.")));
                        }
                    }
                } else {
                    let triples = engine.all_triples();
                    if triples.is_empty() {
                        self.messages.push(AkhMessage::system(
                            "No triples in knowledge graph.".to_string(),
                        ));
                    } else {
                        let rendered = crate::glyph::render::render_to_terminal(
                            engine,
                            &triples,
                            &render_config,
                        );
                        self.messages.push(AkhMessage::system(rendered));
                    }
                }
            }
            crate::agent::UserIntent::Help => {
                self.messages.push(AkhMessage::system(
                    "Type a question (\"What is X?\"), assert a fact (\"X is a Y\"), \
                     or set a goal (\"find X\"). Commands: /help, /grammar, /goals, \
                     /status, /seed, /quit"
                        .to_string(),
                ));
            }
            crate::agent::UserIntent::Freeform { text: _ } => {
                self.messages.push(AkhMessage::system(
                    "I don't understand that. Type /help for commands.".to_string(),
                ));
            }
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
                    "Commands: /grammar <name>, /goals, /status, /seed <pack>, /quit".to_string(),
                ));
            }
            "grammar" | "g" => {
                if let Some(name) = parts.get(1) {
                    self.grammar = name.to_string();
                    self.messages
                        .push(AkhMessage::system(format!("Grammar switched to: {name}")));
                } else {
                    self.messages.push(AkhMessage::system(format!(
                        "Current grammar: {}. Use /grammar <name> to switch.",
                        self.grammar
                    )));
                }
            }
            "goals" | "status" | "seed" => {
                match self.backend {
                    ChatBackend::Local { .. } => self.handle_command_local(parts[0], parts.get(1).copied()),
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
    fn handle_command_local(&mut self, cmd: &str, arg: Option<&str>) {
        let ChatBackend::Local {
            ref agent,
            ref engine,
            ..
        } = self.backend
        else {
            return;
        };

        match cmd {
            "goals" => {
                let goals = agent.goals();
                if goals.is_empty() {
                    self.messages
                        .push(AkhMessage::system("No active goals.".to_string()));
                } else {
                    for g in goals {
                        self.messages.push(AkhMessage::goal_progress(
                            &g.description,
                            format!("{:?}", g.status),
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
