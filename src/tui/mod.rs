//! Unified ratatui-based TUI replacing all three chat modes.
//!
//! The TUI provides: scrollable message output, input area, status bar,
//! and TUI commands (e.g., `/grammar`, `/workspace`, `/goals`, `/quit`).

pub mod sink;
pub mod widgets;

use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use miette::IntoDiagnostic;

use crate::agent::{Agent, AgentConfig, IdleScheduler};
use crate::engine::Engine;
use crate::message::AkhMessage;

/// TUI application state.
pub struct AkhTui {
    workspace: String,
    grammar: String,
    agent: Agent,
    engine: Arc<Engine>,
    messages: Vec<AkhMessage>,
    input_buffer: String,
    scroll_offset: usize,
    should_quit: bool,
    tui_sink: Arc<sink::TuiSink>,
    idle_scheduler: IdleScheduler,
}

impl AkhTui {
    /// Create a new TUI instance.
    pub fn new(workspace: String, engine: Arc<Engine>, agent: Agent) -> Self {
        let grammar = engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .map(|p| p.persona.grammar_preference.clone())
            .unwrap_or_else(|| "narrative".to_string());

        let tui_sink = Arc::new(sink::TuiSink::new());

        Self {
            workspace,
            grammar,
            agent,
            engine,
            messages: vec![AkhMessage::system(
                "Welcome to akh-medu. Type a question or command. /help for commands, /quit to exit.",
            )],
            input_buffer: String::new(),
            scroll_offset: 0,
            should_quit: false,
            tui_sink,
            idle_scheduler: IdleScheduler::default(),
        }
    }

    /// Run the TUI event loop.
    pub fn run(&mut self) -> miette::Result<()> {
        let mut terminal = ratatui::init();

        // Set the agent's sink to our TUI sink.
        self.agent.set_sink(self.tui_sink.clone());

        loop {
            // Drain any pending messages from the sink.
            let pending = self.tui_sink.drain();
            self.messages.extend(pending);

            terminal
                .draw(|frame| {
                    let symbol_count = self.engine.all_symbols().len();
                    let goal_count = self
                        .agent
                        .goals()
                        .iter()
                        .filter(|g| matches!(g.status, crate::agent::GoalStatus::Active))
                        .count();

                    widgets::render(
                        frame,
                        &self.workspace,
                        &self.grammar,
                        &self.messages,
                        &self.input_buffer,
                        self.scroll_offset,
                        self.agent.cycle_count(),
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
                // Idle — run a background task if one is due.
                if let Some(result) = self.idle_scheduler.tick(&mut self.agent) {
                    self.messages.push(AkhMessage::system(format!(
                        "[idle:{}] {}",
                        result.task, result.summary,
                    )));
                }
            }
        }

        // Persist session on exit.
        self.agent.persist_session().into_diagnostic()?;

        ratatui::restore();
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

        // User message — show it in the messages area.
        self.messages.push(AkhMessage::Prompt {
            question: format!("> {input}"),
        });

        // Classify intent and process.
        let intent = crate::agent::classify_intent(input);

        match intent {
            crate::agent::UserIntent::SetGoal { description } => {
                match self.agent.add_goal(&description, 128, "User-directed goal") {
                    Ok(id) => {
                        self.messages.push(AkhMessage::system(format!(
                            "Goal added: \"{description}\" (id: {})",
                            id.get()
                        )));
                        // Run a few cycles.
                        for _ in 0..5 {
                            match self.agent.run_cycle() {
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
                            let active = crate::agent::goal::active_goals(self.agent.goals());
                            if active.is_empty() {
                                break;
                            }
                        }
                        // Synthesize findings.
                        let summary = self
                            .agent
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
            crate::agent::UserIntent::Query { subject } => {
                // Direct KG lookup.
                match self.engine.resolve_symbol(&subject) {
                    Ok(sym_id) => {
                        let from_triples = self.engine.triples_from(sym_id);
                        let to_triples = self.engine.triples_to(sym_id);

                        if from_triples.is_empty() && to_triples.is_empty() {
                            self.messages.push(AkhMessage::system(format!(
                                "No facts found for \"{subject}\"."
                            )));
                        } else {
                            for t in &from_triples {
                                self.messages.push(AkhMessage::fact(format!(
                                    "{} {} {}",
                                    self.engine.resolve_label(t.subject),
                                    self.engine.resolve_label(t.predicate),
                                    self.engine.resolve_label(t.object),
                                )));
                            }
                            for t in &to_triples {
                                self.messages.push(AkhMessage::fact(format!(
                                    "{} {} {}",
                                    self.engine.resolve_label(t.subject),
                                    self.engine.resolve_label(t.predicate),
                                    self.engine.resolve_label(t.object),
                                )));
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
            crate::agent::UserIntent::Assert { text } => {
                use crate::agent::tool::Tool;
                let tool_input = crate::agent::ToolInput::new().with_param("text", &text);
                match crate::agent::tools::TextIngestTool.execute(&self.engine, tool_input) {
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
                    match self.agent.run_cycle() {
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
                let goals = self.agent.goals();
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
                    self.agent.cycle_count(),
                    self.agent.working_memory().len(),
                    self.engine.all_triples().len(),
                )));
            }
            crate::agent::UserIntent::RenderHiero { entity } => {
                let render_config = crate::glyph::RenderConfig {
                    color: false, // TUI handles its own colors
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
                    match self.engine.resolve_symbol(name) {
                        Ok(sym_id) => match self.engine.extract_subgraph(&[sym_id], 1) {
                            Ok(result) if !result.triples.is_empty() => {
                                let rendered = crate::glyph::render::render_to_terminal(
                                    &self.engine,
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
                    let triples = self.engine.all_triples();
                    if triples.is_empty() {
                        self.messages.push(AkhMessage::system(
                            "No triples in knowledge graph.".to_string(),
                        ));
                    } else {
                        let rendered = crate::glyph::render::render_to_terminal(
                            &self.engine,
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

        // Auto-scroll to bottom.
        self.scroll_offset = self.messages.len().saturating_sub(1);
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
            "goals" => {
                let goals = self.agent.goals();
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
                let info = self.engine.info();
                self.messages.push(AkhMessage::system(format!("{info}")));
            }
            "seed" => {
                if let Some(pack_name) = parts.get(1) {
                    let registry = crate::seeds::SeedRegistry::bundled();
                    match registry.apply(pack_name, &self.engine) {
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
            _ => {
                self.messages.push(AkhMessage::system(format!(
                    "Unknown command: /{cmd}. Type /help for commands."
                )));
            }
        }
    }
}

/// Launch the TUI with a given engine and workspace.
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

    let mut tui = AkhTui::new(workspace.to_string(), engine, agent);
    tui.run()
}
