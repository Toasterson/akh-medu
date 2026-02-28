//! Unified input processing for all chat surfaces (TUI, daemon WS, headless CLI).
//!
//! `ChatProcessor` replaces the duplicated intent handling that previously lived
//! separately in TUI's `process_input_local()`, akhomed's `process_ws_input()`,
//! and the headless CLI's inline dispatch. All non-command user input flows
//! through `ChatProcessor::process_input()`, which implements:
//!
//! 1. Structural command fast-path (help, status, goals, etc.)
//! 2. NLU pipeline parse attempt (confidence ≥ tier threshold → ingest as fact)
//! 3. `classify_intent()` fallback with full intent handling chain
//! 4. Conversational classification for `Freeform` intent
//! 5. Escalation to autonomous goal when all else fails

use std::sync::Arc;

use crate::agent::nlp::{classify_conversational, ConversationalKind};
use crate::agent::{Agent, classify_intent, UserIntent};
use crate::engine::Engine;
use crate::grammar::concrete::ParseContext;
use crate::grammar::lexer::Lexicon;
use crate::message::AkhMessage;
use crate::nlu::NluPipeline;

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the chat processor.
pub struct ChatProcessorConfig {
    /// Grammar archetype for narrative output (e.g. "narrative", "clinical").
    pub grammar: String,
    /// Maximum OODA cycles when escalating to a goal in chat context.
    pub max_escalation_cycles: usize,
}

impl Default for ChatProcessorConfig {
    fn default() -> Self {
        Self {
            grammar: "narrative".to_string(),
            max_escalation_cycles: 10,
        }
    }
}

// ── ChatProcessor ───────────────────────────────────────────────────────

/// Unified input processor for all chat surfaces.
///
/// Owns the NLU pipeline and grammar preference, and contains all intent
/// handlers migrated from the TUI's `process_input_local()`.
pub struct ChatProcessor {
    nlu_pipeline: NluPipeline,
    config: ChatProcessorConfig,
    lexicon: Lexicon,
}

impl ChatProcessor {
    /// Create a new `ChatProcessor` with the given NLU pipeline.
    ///
    /// Reads the grammar preference from the engine's Psyche compartment,
    /// falling back to `"narrative"`.
    pub fn new(engine: &Engine, nlu_pipeline: NluPipeline) -> Self {
        let grammar = engine
            .compartments()
            .and_then(|mgr| mgr.psyche())
            .map(|p| p.persona.grammar_preference.clone())
            .unwrap_or_else(|| "narrative".to_string());

        let lexicon = Lexicon::for_language(crate::grammar::lexer::Language::default());

        Self {
            nlu_pipeline,
            config: ChatProcessorConfig {
                grammar,
                ..Default::default()
            },
            lexicon,
        }
    }

    /// Process a line of user input and return response messages.
    ///
    /// Slash commands (`/quit`, `/grammar`, etc.) are NOT handled here — the
    /// caller (TUI, WS handler, headless loop) should intercept those before
    /// calling this method.
    pub fn process_input(
        &mut self,
        text: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // ── Structural command fast-path (skip NLU) ─────────────────
        if is_structural_command(trimmed) {
            let intent = classify_intent(trimmed);
            return self.dispatch_intent(intent, trimmed, agent, engine);
        }

        // ── Conversational fast-path (skip NLU + classify_intent) ──
        // Greetings, meta-questions ("who are you?"), acknowledgments,
        // and follow-ups go directly to conversational handling.
        // This prevents "Hello, who are you?" from being parsed as
        // Query { subject: "you" } or ingested as a fact.
        let conv_kind = classify_conversational(trimmed, &self.lexicon);
        if !matches!(conv_kind, ConversationalKind::Unrecognized) {
            let grammar = self.config.grammar.clone();
            let mut msgs = Vec::new();
            self.handle_freeform(&mut msgs, agent, engine, trimmed, trimmed, &grammar);
            return msgs;
        }

        // ── NLU pipeline ────────────────────────────────────────────
        let parse_ctx = ParseContext::with_engine(
            engine.registry(),
            engine.ops(),
            engine.item_memory(),
        );
        if let Ok(nlu_result) = self.nlu_pipeline.parse(trimmed, &parse_ctx) {
            // Only ingest assertable structures (triples, compounds).
            // Query-like parses (bare entities) fall through to classify_intent.
            if nlu_result.tree.is_assertable() {
                return self.ingest_as_fact(trimmed, engine, "nlu_parse");
            }
        }

        // ── classify_intent fallback ────────────────────────────────
        let intent = classify_intent(trimmed);
        self.dispatch_intent(intent, trimmed, agent, engine)
    }

    /// Current grammar archetype name.
    pub fn grammar(&self) -> &str {
        &self.config.grammar
    }

    /// Switch the grammar archetype.
    pub fn set_grammar(&mut self, name: &str) {
        self.config.grammar = name.to_string();
    }

    /// Persist NLU pipeline state (parse ranker) to the engine's durable store.
    pub fn persist_nlu_state(&self, engine: &Engine) {
        let ranker_bytes = self.nlu_pipeline.ranker().to_bytes();
        let _ = engine.store().put_meta(b"nlu_ranker_state", &ranker_bytes);
    }

    /// Access the NLU pipeline (e.g. for model loading status checks).
    pub fn nlu_pipeline(&self) -> &NluPipeline {
        &self.nlu_pipeline
    }

    // ── Private intent dispatch ─────────────────────────────────────

    fn dispatch_intent(
        &mut self,
        intent: UserIntent,
        raw_input: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let mut msgs = Vec::new();
        let grammar = self.config.grammar.clone();

        match intent {
            UserIntent::SetGoal { description } => {
                self.handle_set_goal(&mut msgs, agent, engine, &description, &grammar);
            }
            UserIntent::Query {
                subject,
                original_input,
                question_word,
                capability_signal,
            } => {
                self.handle_query(
                    &mut msgs,
                    agent,
                    engine,
                    &subject,
                    &original_input,
                    question_word,
                    capability_signal,
                    &grammar,
                );
            }
            UserIntent::Assert { text } => {
                msgs.extend(self.ingest_as_fact(&text, engine, "text_ingest"));
            }
            UserIntent::RunAgent { cycles } => {
                Self::handle_run_agent(&mut msgs, agent, cycles);
            }
            UserIntent::ShowStatus => {
                Self::handle_show_status(&mut msgs, agent, engine);
            }
            UserIntent::RenderHiero { entity } => {
                Self::handle_render_hiero(&mut msgs, engine, entity.as_deref());
            }
            UserIntent::SetDetail { level } => {
                Self::handle_set_detail(&mut msgs, agent, &level);
            }
            UserIntent::Help => {
                msgs.push(AkhMessage::system(
                    "Type a question (\"What is X?\"), assert a fact (\"X is a Y\"), \
                     or set a goal (\"find X\"). Commands: /help, /grammar, /goals, \
                     /status, /seed, /quit, detail <level>"
                        .to_string(),
                ));
            }
            UserIntent::Explain { ref query } => {
                match crate::agent::explain::execute_query(engine, query, None) {
                    Ok(explanation) => msgs.push(AkhMessage::system(explanation)),
                    Err(e) => msgs.push(AkhMessage::system(format!("{e}"))),
                }
            }
            UserIntent::AgentProtocol { ref message } => {
                msgs.push(AkhMessage::system(format!(
                    "[agent protocol] received: {:?}",
                    std::mem::discriminant(message),
                )));
            }
            UserIntent::PimCommand { subcommand, args } => {
                msgs.push(AkhMessage::system(format!(
                    "PIM commands are available via the CLI: akh pim {subcommand} {args}",
                )));
            }
            UserIntent::CalCommand { subcommand, args } => {
                msgs.push(AkhMessage::system(format!(
                    "Calendar commands are available via the CLI: akh cal {subcommand} {args}",
                )));
            }
            UserIntent::PrefCommand { subcommand, args } => {
                msgs.push(AkhMessage::system(format!(
                    "Preference commands are available via the CLI: akh pref {subcommand} {args}",
                )));
            }
            UserIntent::CausalQuery { subcommand, args } => {
                msgs.push(AkhMessage::system(format!(
                    "Causal commands are available via the CLI: akh causal {subcommand} {args}",
                )));
            }
            UserIntent::AwakenCommand { subcommand, args } => {
                if subcommand == "status" {
                    Self::handle_awaken_status(&mut msgs, engine);
                } else {
                    msgs.push(AkhMessage::system(format!(
                        "Awaken commands are available via the CLI: akh awaken {subcommand} {args}",
                    )));
                }
            }
            UserIntent::Freeform { ref text } => {
                self.handle_freeform(&mut msgs, agent, engine, text, raw_input, &grammar);
            }
        }

        msgs
    }

    // ── Intent handlers ─────────────────────────────────────────────

    fn handle_set_goal(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        description: &str,
        grammar: &str,
    ) {
        match agent.add_goal(description, 128, "User-directed goal") {
            Ok(id) => {
                msgs.push(AkhMessage::system(format!(
                    "Goal added: \"{description}\" (id: {})",
                    id.get()
                )));
                // Run a few cycles.
                for _ in 0..5 {
                    match agent.run_cycle() {
                        Ok(result) => {
                            msgs.push(AkhMessage::tool_result(
                                &result.decision.chosen_tool,
                                result.action_result.tool_output.success,
                                &result.action_result.tool_output.result,
                            ));
                        }
                        Err(_) => break,
                    }
                    if crate::agent::goal::active_goals(agent.goals()).is_empty() {
                        break;
                    }
                }
                // Synthesize findings.
                let summary = agent.synthesize_findings_with_grammar(description, grammar);
                Self::push_summary(msgs, &summary, grammar);
            }
            Err(e) => {
                msgs.push(AkhMessage::error("goal", e.to_string()));
            }
        }
        let _ = engine;
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_query(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        subject: &str,
        original_input: &str,
        question_word: Option<crate::agent::QuestionWord>,
        capability_signal: bool,
        grammar: &str,
    ) {
        // Phase 12b+12c: Try grounded dialogue with constraint checking.
        let grounded = crate::agent::conversation::ground_query(subject, engine, grammar);
        if let Some(ref gr) = grounded {
            let (out_msg, decision) = agent.check_and_wrap_grounded(
                gr,
                "operator",
                crate::agent::ChannelKind::Operator,
            );
            if decision == crate::agent::constraint_check::EmissionDecision::Emit {
                let detail = agent.conversation_state().response_detail;
                let rendered = gr.render(detail);
                agent.conversation_state_mut().record_agent_turn(&rendered);
                agent
                    .conversation_state_mut()
                    .track_referent(subject.to_string());
                for akh_msg in out_msg.to_akh_messages() {
                    msgs.push(akh_msg);
                }
                if !out_msg.constraint_check.is_passed() {
                    msgs.push(AkhMessage::system(
                        "[constraint check: some violations detected]".to_string(),
                    ));
                }
            }
        }

        if grounded.is_none() {
            // Fallback: discourse-aware response, then synthesis.
            let discourse_result = crate::grammar::discourse::resolve_discourse(
                subject,
                question_word,
                original_input,
                engine,
                capability_signal,
                Some(agent.conversation_state()),
                Some(&self.lexicon),
            );
            let handled = if let Ok(ref ctx) = discourse_result {
                let from_triples = engine.triples_from(ctx.subject_id);
                let to_triples = engine.triples_to(ctx.subject_id);
                let mut all_triples = from_triples;
                all_triples.extend(to_triples);
                if let Some(discourse_tree) =
                    crate::grammar::discourse::build_discourse_response(&all_triples, ctx, engine)
                {
                    let registry = crate::grammar::GrammarRegistry::new();
                    if let Ok(prose) = registry.linearize(grammar, &discourse_tree) {
                        if !prose.trim().is_empty() {
                            msgs.push(AkhMessage::narrative(&prose, grammar));
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
                match engine.resolve_symbol(subject) {
                    Ok(sym_id) => {
                        let from_triples = engine.triples_from(sym_id);
                        let to_triples = engine.triples_to(sym_id);
                        if from_triples.is_empty() && to_triples.is_empty() {
                            self.handle_unknown_subject(
                                msgs,
                                agent,
                                grammar,
                                subject,
                                original_input,
                            );
                        } else {
                            let mut all_triples = from_triples;
                            all_triples.extend(to_triples);
                            let summary =
                                crate::agent::synthesize::synthesize_from_triples(
                                    subject,
                                    &all_triples,
                                    engine,
                                    grammar,
                                );
                            Self::push_summary(msgs, &summary, grammar);
                        }
                    }
                    Err(_) => {
                        self.handle_unknown_subject(
                            msgs,
                            agent,
                            grammar,
                            subject,
                            original_input,
                        );
                    }
                }
            }
        }
    }

    fn handle_freeform(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        text: &str,
        raw_input: &str,
        grammar: &str,
    ) {
        let kind = classify_conversational(text, &self.lexicon);
        match kind {
            ConversationalKind::Greeting | ConversationalKind::Acknowledgment => {
                let (name, traits) = Self::persona_name_and_traits(engine);
                let resp =
                    agent
                        .conversation_state()
                        .respond_conversational(&kind, &name, &traits);
                if !resp.text.is_empty() {
                    msgs.push(AkhMessage::narrative(&resp.text, grammar));
                    agent.conversation_state_mut().record_agent_turn(&resp.text);
                }
            }
            ConversationalKind::FollowUp => {
                let (name, traits) = Self::persona_name_and_traits(engine);
                let resp =
                    agent
                        .conversation_state()
                        .respond_conversational(&kind, &name, &traits);
                if resp.text.is_empty() {
                    // Active topic exists — re-query at Full detail.
                    if let Some(topic_id) = agent.conversation_state().topic() {
                        let label = engine.resolve_label(topic_id);
                        if let Some(gr) =
                            crate::agent::conversation::ground_query(&label, engine, grammar)
                        {
                            let rendered = gr.render(
                                crate::agent::conversation::ResponseDetail::Full,
                            );
                            msgs.push(AkhMessage::narrative(&rendered, grammar));
                            agent.conversation_state_mut().record_agent_turn(&rendered);
                        }
                    }
                } else {
                    msgs.push(AkhMessage::narrative(&resp.text, grammar));
                    agent.conversation_state_mut().record_agent_turn(&resp.text);
                }
            }
            ConversationalKind::MetaQuestion => {
                // Route to grounded self-description.
                if let Some(gr) =
                    crate::agent::conversation::ground_query("self", engine, grammar)
                {
                    let detail = agent.conversation_state().response_detail;
                    let rendered = gr.render(detail);
                    msgs.push(AkhMessage::narrative(&rendered, grammar));
                    agent.conversation_state_mut().record_agent_turn(&rendered);
                } else {
                    let (name, _) = Self::persona_name_and_traits(engine);
                    msgs.push(AkhMessage::narrative(
                        &format!(
                            "I am {name}. I can answer questions about what I know, \
                             learn new facts, and investigate topics autonomously."
                        ),
                        grammar,
                    ));
                }
            }
            ConversationalKind::Unrecognized => {
                // NLU already tried in process_input() and failed.
                // Escalate to goal.
                self.escalate_to_goal(
                    msgs,
                    agent,
                    grammar,
                    raw_input,
                    &format!("Exploring \"{text}\"."),
                );
            }
        }
    }

    fn handle_run_agent(msgs: &mut Vec<AkhMessage>, agent: &mut Agent, cycles: Option<usize>) {
        let n = cycles.unwrap_or(1);
        for _ in 0..n {
            match agent.run_cycle() {
                Ok(result) => {
                    msgs.push(AkhMessage::tool_result(
                        &result.decision.chosen_tool,
                        result.action_result.tool_output.success,
                        &result.action_result.tool_output.result,
                    ));
                }
                Err(e) => {
                    msgs.push(AkhMessage::error("cycle", e.to_string()));
                    break;
                }
            }
        }
    }

    fn handle_show_status(msgs: &mut Vec<AkhMessage>, agent: &Agent, engine: &Engine) {
        let goals = agent.goals();
        if goals.is_empty() {
            msgs.push(AkhMessage::system("No active goals.".to_string()));
        } else {
            for g in goals {
                msgs.push(AkhMessage::goal_progress(
                    &g.description,
                    format!("{}", g.status),
                ));
            }
        }
        msgs.push(AkhMessage::system(format!(
            "Cycles: {}, WM entries: {}, Triples: {}",
            agent.cycle_count(),
            agent.working_memory().len(),
            engine.all_triples().len(),
        )));
    }

    fn handle_render_hiero(msgs: &mut Vec<AkhMessage>, engine: &Engine, entity: Option<&str>) {
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

        if let Some(name) = entity {
            match engine.resolve_symbol(name) {
                Ok(sym_id) => match engine.extract_subgraph(&[sym_id], 1) {
                    Ok(result) if !result.triples.is_empty() => {
                        let rendered = crate::glyph::render::render_to_terminal(
                            engine,
                            &result.triples,
                            &render_config,
                        );
                        msgs.push(AkhMessage::system(rendered));
                    }
                    _ => {
                        msgs.push(AkhMessage::system(format!(
                            "No triples found around \"{name}\"."
                        )));
                    }
                },
                Err(_) => {
                    msgs.push(AkhMessage::system(format!(
                        "Symbol \"{name}\" not found."
                    )));
                }
            }
        } else {
            let triples = engine.all_triples();
            if triples.is_empty() {
                msgs.push(AkhMessage::system(
                    "No triples in knowledge graph.".to_string(),
                ));
            } else {
                let rendered = crate::glyph::render::render_to_terminal(
                    engine,
                    &triples,
                    &render_config,
                );
                msgs.push(AkhMessage::system(rendered));
            }
        }
    }

    fn handle_set_detail(msgs: &mut Vec<AkhMessage>, agent: &mut Agent, level: &str) {
        match crate::agent::conversation::ResponseDetail::from_str_loose(level) {
            Some(detail) => {
                agent.set_response_detail(detail);
                msgs.push(AkhMessage::system(format!(
                    "Response detail set to: {detail:?}"
                )));
            }
            None => {
                msgs.push(AkhMessage::system(
                    "Unknown detail level. Use: concise, normal, or full.".to_string(),
                ));
            }
        }
    }

    fn handle_awaken_status(msgs: &mut Vec<AkhMessage>, engine: &Engine) {
        let psyche = engine.compartments().and_then(|m| m.psyche());
        if let Some(p) = psyche {
            msgs.push(AkhMessage::system(format!(
                "Psyche:\n\
                 \x20 Persona:    {}\n\
                 \x20 Grammar:    {}\n\
                 \x20 Traits:     {:?}\n\
                 \x20 Archetypes: sage={:.2} explorer={:.2} healer={:.2} guardian={:.2}\n\
                 \x20 Shadow:     {} veto, {} bias patterns\n\
                 \x20 Integration: {:.1}% (dominant: {})",
                p.persona.name,
                p.persona.grammar_preference,
                p.persona.traits,
                p.archetypes.sage,
                p.archetypes.explorer,
                p.archetypes.healer,
                p.archetypes.guardian,
                p.shadow.veto_patterns.len(),
                p.shadow.bias_patterns.len(),
                p.self_integration.individuation_level * 100.0,
                p.self_integration.dominant_archetype,
            )));
        } else {
            msgs.push(AkhMessage::system(
                "No psyche loaded. Run `akh awaken resolve <name>` to awaken.".to_string(),
            ));
        }
    }

    // ── Shared helpers ──────────────────────────────────────────────

    /// Ingest text as a fact using the TextIngestTool.
    fn ingest_as_fact(
        &self,
        text: &str,
        engine: &Engine,
        tool_label: &str,
    ) -> Vec<AkhMessage> {
        use crate::agent::tool::Tool;
        let tool_input = crate::agent::ToolInput::new().with_param("text", text);
        match crate::agent::tools::TextIngestTool.execute(engine, tool_input) {
            Ok(output) => {
                vec![AkhMessage::tool_result(tool_label, output.success, &output.result)]
            }
            Err(e) => {
                vec![AkhMessage::error("ingest", e.to_string())]
            }
        }
    }

    /// Escalate unresolved input to a goal, run OODA cycles, and synthesize findings.
    fn escalate_to_goal(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        grammar: &str,
        description: &str,
        display_prefix: &str,
    ) {
        msgs.push(AkhMessage::system(format!(
            "{display_prefix} Investigating..."
        )));

        let goal_id = match agent.add_goal(
            &format!("investigate: {description}"),
            180,
            "Find or derive relevant knowledge",
        ) {
            Ok(id) => id,
            Err(e) => {
                msgs.push(AkhMessage::error("goal", e.to_string()));
                return;
            }
        };

        msgs.push(AkhMessage::system(format!(
            "Goal created (id: {})",
            goal_id.get()
        )));

        let max = agent
            .config
            .max_cycles
            .clamp(1, self.config.max_escalation_cycles);
        for _ in 0..max {
            match agent.run_cycle() {
                Ok(result) => {
                    msgs.push(AkhMessage::tool_result(
                        &result.decision.chosen_tool,
                        result.action_result.tool_output.success,
                        &result.action_result.tool_output.result,
                    ));
                }
                Err(_) => break,
            }
            if crate::agent::goal::active_goals(agent.goals()).is_empty() {
                break;
            }
        }

        let summary = agent.synthesize_findings_with_grammar(description, grammar);
        Self::push_summary(msgs, &summary, grammar);
    }

    /// Handle an unknown subject: friendly message then escalate.
    fn handle_unknown_subject(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        grammar: &str,
        subject: &str,
        original_input: &str,
    ) {
        msgs.push(AkhMessage::narrative(
            &format!("I don't know about \"{subject}\" yet. Let me investigate."),
            grammar,
        ));
        self.escalate_to_goal(msgs, agent, grammar, original_input, "");
    }

    /// Extract persona name and traits from the engine's Psyche compartment.
    fn persona_name_and_traits(engine: &Engine) -> (String, Vec<String>) {
        engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .map(|p| (p.persona.name.clone(), p.persona.traits.clone()))
            .unwrap_or_else(|| ("Akh".to_string(), Vec::new()))
    }

    /// Push a NarrativeSummary's overview, sections, and gaps onto a message vec.
    fn push_summary(
        msgs: &mut Vec<AkhMessage>,
        summary: &crate::agent::NarrativeSummary,
        grammar: &str,
    ) {
        if !summary.overview.is_empty() {
            msgs.push(AkhMessage::narrative(&summary.overview, grammar));
        }
        for section in &summary.sections {
            msgs.push(AkhMessage::narrative(
                format!("## {}\n{}", section.heading, section.prose),
                grammar,
            ));
        }
        for gap in &summary.gaps {
            msgs.push(AkhMessage::gap("(unknown)", gap));
        }
    }
}

// ── Structural command detection ────────────────────────────────────────

/// Fast-path detection for inputs that should bypass NLU and go directly
/// to `classify_intent()`. These are structural commands where NLU parsing
/// would be wasteful or incorrect.
fn is_structural_command(text: &str) -> bool {
    let lower = text.trim().to_lowercase();

    // Exact matches.
    matches!(
        lower.as_str(),
        "help" | "?" | "status" | "goals" | "show status" | "show goals"
    ) ||
    // Prefix matches for multi-word commands.
    lower.starts_with("help ")
        || lower.starts_with("run ")
        || lower.starts_with("run")  // "run" alone or "run5"
        || lower.starts_with("cycle")
        || lower.starts_with("show ")
        || lower.starts_with("render ")
        || lower.starts_with("graph ")
        || lower.starts_with("pim ")
        || lower.starts_with("pim")
        || lower.starts_with("cal ")
        || lower.starts_with("cal")
        || lower.starts_with("pref ")
        || lower.starts_with("pref")
        || lower.starts_with("causal ")
        || lower.starts_with("causal")
        || lower.starts_with("awaken ")
        || lower.starts_with("awaken")
        || lower.starts_with("set detail ")
        || lower.starts_with("detail ")
        || lower.starts_with("list goals")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_commands_detected() {
        assert!(is_structural_command("help"));
        assert!(is_structural_command("status"));
        assert!(is_structural_command("goals"));
        assert!(is_structural_command("run 5"));
        assert!(is_structural_command("show Dog"));
        assert!(is_structural_command("pim inbox"));
        assert!(is_structural_command("set detail concise"));
        assert!(is_structural_command("awaken status"));
    }

    #[test]
    fn non_structural_inputs() {
        assert!(!is_structural_command("What is a dog?"));
        assert!(!is_structural_command("hello"));
        assert!(!is_structural_command("Rust is a programming language"));
        assert!(!is_structural_command("find similar animals"));
    }

    #[test]
    fn conversational_inputs_skip_nlu() {
        use crate::grammar::lexer::{Language, Lexicon};

        let lexicon = Lexicon::for_language(Language::default());

        // Greetings should be recognized as conversational, not unrecognized.
        let hello = classify_conversational("hello", &lexicon);
        assert!(!matches!(hello, ConversationalKind::Unrecognized), "hello should be Greeting");

        let hi = classify_conversational("hi there", &lexicon);
        assert!(!matches!(hi, ConversationalKind::Unrecognized), "hi there should be Greeting");

        // Meta-questions should be recognized as conversational.
        let who = classify_conversational("who are you?", &lexicon);
        assert!(!matches!(who, ConversationalKind::Unrecognized), "who are you should be MetaQuestion");

        // Factual statements should NOT be conversational.
        let fact = classify_conversational("Rust is a programming language", &lexicon);
        assert!(matches!(fact, ConversationalKind::Unrecognized), "fact should be Unrecognized");
    }
}
