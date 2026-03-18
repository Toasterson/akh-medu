//! Unified input processing for all chat surfaces (TUI, daemon WS, headless CLI).
//!
//! `ChatProcessor` replaces the duplicated intent handling that previously lived
//! separately in TUI's `process_input_local()`, akhomed's `process_ws_input()`,
//! and the headless CLI's inline dispatch. All non-command user input flows
//! through `ChatProcessor::process_input()`, which implements:
//!
//! 1. Structural command fast-path (help, status, goals, etc.)
//! 2. NLU pipeline parse → AbsTree dispatch (dialogue acts, facts, queries)
//! 3. Escalation to autonomous goal when all else fails

use std::sync::Arc;

use crate::agent::Agent;
use crate::agent::nlp::QuestionWord;
#[allow(deprecated)]
use crate::agent::{classify_intent, UserIntent};
use crate::agent::conversation::Speaker;
use crate::engine::Engine;
use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::ParseContext;
use crate::grammar::discourse::{QueryFocus, build_discourse_response, classify_focus_with_modal, resolve_discourse};
use crate::grammar::lexer::Lexicon;
use crate::graph::Triple;
use crate::infer::InferenceQuery;
use crate::infer::backward::{BackwardConfig, infer_backward};
use crate::message::AkhMessage;
use crate::nlu::NluPipeline;
use crate::symbol::SymbolId;

// ── Query decomposition ─────────────────────────────────────────────────

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
/// Owns the NLU pipeline and grammar preference. All user input (except
/// slash commands) is parsed through the NLU pipeline into AbsTree nodes,
/// then dispatched by variant: dialogue acts go to the DialogueManager,
/// assertable structures are ingested as facts, queries are grounded, and
/// unrecognized input is escalated to an autonomous goal.
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

        let status = nlu_pipeline.tier_status();
        tracing::info!(%status, "NLU pipeline initialized");
        if !status.tier3_llm {
            tracing::warn!(
                "LLM translator not loaded — complex queries will fall through to OODA. \
                 Run `akh setup models` to download."
            );
        }

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
    ///
    /// Processing flow:
    /// 1. Structural commands (help, status, run, show, etc.) bypass NLU
    /// 2. All other input goes through NLU pipeline → AbsTree
    /// 3. AbsTree variant dispatch: dialogue acts, facts, queries, goals
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
        // These are system commands where NLU parsing would be wasteful.
        if is_structural_command(trimmed) {
            #[allow(deprecated)]
            let intent = classify_intent(trimmed);
            return self.dispatch_structural(intent, trimmed, agent, engine);
        }

        // ── NLU pipeline (all other input) ──────────────────────────
        let parse_ctx = ParseContext::with_engine(
            engine.registry(),
            engine.ops(),
            engine.item_memory(),
        );

        match self.nlu_pipeline.parse(trimmed, &parse_ctx) {
            Ok(nlu_result) => {
                // Record turn in dialogue manager (no resolved entities at parse time;
                // entity resolution happens downstream in dispatch handlers).
                agent.dialogue_manager().record_turn(
                    Speaker::Operator,
                    trimmed,
                    &nlu_result.tree,
                    engine,
                    &[],
                );

                // Dispatch on AbsTree variant
                self.dispatch_abstree(&nlu_result.tree, trimmed, agent, engine)
            }
            Err(_) => {
                // NLU pipeline completely failed — escalate to goal
                let grammar = self.config.grammar.clone();
                let mut msgs = Vec::new();
                self.escalate_to_goal(
                    &mut msgs,
                    agent,
                    &grammar,
                    trimmed,
                    &format!("Exploring \"{trimmed}\"."),
                );
                msgs
            }
        }
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

    // ── AbsTree dispatch ────────────────────────────────────────────

    /// Dispatch on an NLU-parsed AbsTree variant.
    fn dispatch_abstree(
        &mut self,
        tree: &AbsTree,
        raw_input: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let grammar = self.config.grammar.clone();
        let mut msgs = Vec::new();

        match tree {
            // ── Dialogue acts ───────────────────────────────────────
            AbsTree::Greeting { .. } => {
                self.handle_dialogue_greeting(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::Farewell { .. } => {
                self.handle_dialogue_farewell(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::Acknowledgment { .. } => {
                self.handle_dialogue_ack(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::FollowUpRequest { .. } => {
                self.handle_dialogue_follow_up(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::MetaQuery { .. } => {
                self.handle_dialogue_meta_query(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::GoalRequest { description } => {
                let desc = description.label()
                    .map(String::from)
                    .unwrap_or_else(|| description.collect_labels().join(" "));
                self.handle_set_goal(&mut msgs, agent, engine, &desc, &grammar);
            }
            AbsTree::StructuralCommand { command, args } => {
                self.handle_structural_command(&mut msgs, agent, engine, command, args);
            }

            // ── Assertable structures (triples, compounds) ──────────
            tree if tree.is_assertable() => {
                msgs.extend(self.ingest_as_fact(raw_input, engine, "nlu_parse"));
            }

            // ── Free-form text that parsed but isn't assertable ─────
            AbsTree::Freeform(_) => {
                self.escalate_to_goal(
                    &mut msgs,
                    agent,
                    &grammar,
                    raw_input,
                    &format!("Exploring \"{raw_input}\"."),
                );
            }

            // ── Query-like trees (entity refs, etc.) ────────────────
            _ => {
                // Extract a subject label from the tree for query handling.
                let subject = tree.label()
                    .map(String::from)
                    .unwrap_or_else(|| tree.collect_labels().join(" "));
                if subject.is_empty() {
                    self.escalate_to_goal(
                        &mut msgs,
                        agent,
                        &grammar,
                        raw_input,
                        &format!("Exploring \"{raw_input}\"."),
                    );
                } else {
                    self.handle_query_from_tree(
                        &mut msgs,
                        agent,
                        engine,
                        &subject,
                        raw_input,
                        &grammar,
                    );
                }
            }
        }

        msgs
    }

    // ── Dialogue act handlers ───────────────────────────────────────

    fn handle_dialogue_greeting(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let active_topic = Self::active_topic_label(agent, engine);
        // Try LLM-generated greeting — but validate it mentions the persona name.
        // Small LLMs often ignore the system prompt and produce generic greetings.
        let text = self
            .nlu_pipeline
            .generate_dialogue("greeting", &name, &traits, active_topic.as_deref())
            .filter(|t| t.to_lowercase().contains(&name.to_lowercase()))
            .unwrap_or_else(|| {
                agent
                    .dialogue_manager()
                    .handle_greeting(engine, &name, &traits)
            });
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state_mut().record_agent_turn(&text);
    }

    fn handle_dialogue_farewell(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let text = self
            .nlu_pipeline
            .generate_dialogue("farewell", &name, &traits, None)
            .unwrap_or_else(|| agent.dialogue_manager().handle_farewell(&name));
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state_mut().record_agent_turn(&text);
    }

    fn handle_dialogue_ack(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let text = self
            .nlu_pipeline
            .generate_dialogue("acknowledgment", &name, &traits, None)
            .unwrap_or_else(|| agent.dialogue_manager().handle_ack(&traits));
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state_mut().record_agent_turn(&text);
    }

    fn handle_dialogue_follow_up(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        let result = agent.dialogue_manager().handle_follow_up(engine);
        match result {
            Some(text) => {
                // No active topic — inform the user.
                msgs.push(AkhMessage::narrative(&text, grammar));
                agent.conversation_state_mut().record_agent_turn(&text);
            }
            None => {
                // Active topic exists in KG — resolve and re-query at Full detail.
                let topic_id = agent.dialogue_manager().query_active_topic(engine)
                    .or_else(|| agent.conversation_state().topic());
                if let Some(tid) = topic_id {
                    let label = engine.resolve_label(tid);
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
            }
        }
    }

    fn handle_dialogue_meta_query(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        // Route through the standard query pipeline with subject "self".
        // This produces a first-person, discourse-ranked response from the KG
        // using the same grammar pipeline as any other query.
        self.handle_query_from_tree(msgs, agent, engine, "self", "who are you?", grammar);
    }

    fn handle_structural_command(
        &self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        command: &str,
        args: &[String],
    ) {
        match command {
            "help" => {
                msgs.push(AkhMessage::system(
                    "Type a question (\"What is X?\"), assert a fact (\"X is a Y\"), \
                     or set a goal (\"find X\"). Commands: /help, /grammar, /goals, \
                     /status, /seed, /quit, detail <level>"
                        .to_string(),
                ));
            }
            "status" | "goals" => {
                Self::handle_show_status(msgs, agent, engine);
            }
            "run" => {
                let cycles = args.first().and_then(|a| a.parse::<usize>().ok());
                Self::handle_run_agent(msgs, agent, cycles);
            }
            "show" | "render" | "graph" => {
                let entity = args.first().map(|s| s.as_str());
                Self::handle_render_hiero(msgs, engine, entity);
            }
            "detail" => {
                if let Some(level) = args.first() {
                    Self::handle_set_detail(msgs, agent, level);
                } else {
                    msgs.push(AkhMessage::system(
                        "Usage: detail <concise|normal|full>".to_string(),
                    ));
                }
            }
            "explain" => {
                let query_text = args.join(" ");
                if let Some(eq) = crate::agent::explain::ExplanationQuery::parse(&query_text) {
                    match crate::agent::explain::execute_query(engine, &eq, None) {
                        Ok(explanation) => msgs.push(AkhMessage::system(explanation)),
                        Err(e) => msgs.push(AkhMessage::system(format!("{e}"))),
                    }
                } else {
                    msgs.push(AkhMessage::system(format!(
                        "Could not parse explanation query: \"{query_text}\". \
                         Try: explain why <entity>, explain how <entity>, explain what changed"
                    )));
                }
            }
            "pim" | "cal" | "pref" | "causal" => {
                let sub = args.first().map(|s| s.as_str()).unwrap_or("");
                let rest = if args.len() > 1 { args[1..].join(" ") } else { String::new() };
                msgs.push(AkhMessage::system(format!(
                    "{command} commands are available via the CLI: akh {command} {sub} {rest}",
                )));
            }
            "awaken" => {
                let sub = args.first().map(|s| s.as_str()).unwrap_or("");
                if sub == "status" {
                    Self::handle_awaken_status(msgs, engine);
                } else {
                    let rest = if args.len() > 1 { args[1..].join(" ") } else { String::new() };
                    msgs.push(AkhMessage::system(format!(
                        "Awaken commands are available via the CLI: akh awaken {sub} {rest}",
                    )));
                }
            }
            _ => {
                msgs.push(AkhMessage::system(format!(
                    "Unknown command: {command}. Type \"help\" for available commands.",
                )));
            }
        }
    }

    // ── Structural command dispatch (legacy, for is_structural_command fast-path) ──

    #[allow(deprecated)]
    fn dispatch_structural(
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
                question_word: _,
                capability_signal: _,
            } => {
                self.handle_query_from_tree(
                    &mut msgs,
                    agent,
                    engine,
                    &subject,
                    &original_input,
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
                // For structural command fallthrough, escalate to goal.
                self.escalate_to_goal(
                    &mut msgs,
                    agent,
                    &grammar,
                    raw_input,
                    &format!("Exploring \"{text}\"."),
                );
            }
        }

        msgs
    }

    // ── Intent handlers ─────────────────────────────────────────────

    fn handle_set_goal(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        description: &str,
        grammar: &str,
    ) {
        let wm_before = agent.working_memory().len();

        match agent.add_goal(description, 128, "User-directed goal") {
            Ok(id) => {
                // Goal creation → audit log only
                msgs.push(AkhMessage::audit_log(
                    id.get(),
                    "goal",
                    format!("Added: {description}"),
                ));
                // Run a few cycles.
                for _ in 0..5 {
                    match agent.run_cycle() {
                        Ok(result) => {
                            // Tool results → audit log only
                            msgs.push(AkhMessage::audit_log(
                                0,
                                &result.decision.chosen_tool,
                                &result.action_result.tool_output.result,
                            ));
                        }
                        Err(_) => break,
                    }
                    if crate::agent::goal::active_goals(agent.goals()).is_empty() {
                        break;
                    }
                }
                // Synthesize only from entries added during this goal's cycles.
                let all_entries = agent.working_memory().entries();
                let new_entries = if wm_before < all_entries.len() {
                    &all_entries[wm_before..]
                } else {
                    &[]
                };
                let summary = crate::agent::synthesize::synthesize_with_grammar(
                    description,
                    new_entries,
                    agent.engine(),
                    grammar,
                );
                self.push_summary(msgs, &summary, grammar);
            }
            Err(e) => {
                msgs.push(AkhMessage::error("goal", e.to_string()));
            }
        }
        let _ = engine;
    }

    /// Handle a query derived from an AbsTree parse (entity ref, etc.).
    fn handle_query_from_tree(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        subject: &str,
        original_input: &str,
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

                // Persist active topic to KG via dialogue manager.
                if let Ok(sym_id) = engine.resolve_symbol(subject) {
                    let _ = agent.dialogue_manager().set_active_topic(engine, sym_id);
                    agent.conversation_state_mut().active_topic = Some(sym_id);
                }

                for akh_msg in out_msg.to_akh_messages() {
                    msgs.push(akh_msg);
                }
                if !out_msg.constraint_check.is_passed() {
                    msgs.push(AkhMessage::system(
                        "[constraint check: some violations detected]".to_string(),
                    ));
                }
            }
            return;
        }

        // ── Step 1: Classify query focus from raw input ──────────────────
        let question_word = Self::extract_question_word(original_input, &self.lexicon);
        let capability_signal = Self::detect_capability_signal(original_input, &self.lexicon);
        let focus = classify_focus_with_modal(question_word, capability_signal);

        // ── Step 2: Resolve discourse context (pronoun resolution, POV) ──
        let discourse_result = resolve_discourse(
            subject,
            question_word,
            original_input,
            engine,
            capability_signal,
            Some(agent.conversation_state()),
            Some(&self.lexicon),
        );

        // Determine the subject symbol ID (from discourse or direct resolution).
        let subject_id = discourse_result
            .as_ref()
            .ok()
            .map(|ctx| ctx.subject_id)
            .or_else(|| engine.resolve_symbol(subject).ok());

        // If exact resolution failed, try compound subject decomposition.
        let (subject_id, was_provisional) = if let Some(id) = subject_id {
            (Some(id), false)
        } else {
            self.try_decomposed_resolution(subject, engine)
                .map(|(id, prov)| (Some(id), prov))
                .unwrap_or((None, false))
        };

        let Some(sym_id) = subject_id else {
            // Subject unknown even after decomposition — escalate to goal.
            self.handle_unknown_subject(msgs, agent, grammar, subject, original_input);
            return;
        };

        // ── Step 3: Collect base triples from KG ─────────────────────────
        let from_triples = engine.triples_from(sym_id);
        let to_triples = engine.triples_to(sym_id);
        let mut all_triples: Vec<Triple> = from_triples;
        all_triples.extend(to_triples);

        // ── Step 4: Spreading activation — discover related facts ────────
        let infer_query = InferenceQuery::default()
            .with_seeds(vec![sym_id])
            .with_max_depth(2)
            .with_min_confidence(0.3);

        if let Ok(infer_result) = engine.infer(&infer_query) {
            let activation_count = infer_result.activations.len();
            for (activated_sym, _confidence) in &infer_result.activations {
                // Skip the seed itself — we already have its triples.
                if *activated_sym == sym_id {
                    continue;
                }
                let activated_triples = engine.triples_from(*activated_sym);
                all_triples.extend(activated_triples);
            }
            if activation_count > 0 {
                tracing::info!(
                    subject = %subject,
                    activations = activation_count,
                    "Spreading activation discovered related concepts"
                );
            }
        }

        // ── Step 5: For why/how focus — backward chaining ────────────────
        if matches!(focus, QueryFocus::Cause | QueryFocus::Method) {
            let bc_config = BackwardConfig {
                max_depth: 3,
                min_confidence: 0.1,
                vsa_verify: true,
            };
            if let Ok(chains) = infer_backward(engine, sym_id, &bc_config) {
                let chain_count = chains.len();
                for chain in &chains {
                    all_triples.extend(chain.supporting_triples.iter().cloned());
                }
                if chain_count > 0 {
                    tracing::info!(
                        subject = %subject,
                        chains = chain_count,
                        "Backward chaining found supporting evidence"
                    );
                }
            }
        }

        // Escalate when knowledge is too thin to give a useful answer:
        // - No triples at all
        // - Only provisional triples from decomposition
        // - Only low-confidence triples (≤ 0.70) suggesting provisional-only knowledge
        let is_thin_knowledge = if all_triples.is_empty() {
            true
        } else if was_provisional && all_triples.len() <= 2 {
            true
        } else {
            let high_conf = all_triples.iter().any(|t| t.confidence > 0.70);
            !high_conf && all_triples.len() <= 3
        };

        if is_thin_knowledge {
            let primary = subject.split_whitespace().last().unwrap_or(subject);
            self.handle_unknown_subject(msgs, agent, grammar, primary, original_input);
            return;
        }

        // ── Step 6: Discourse-ranked response via grammar ────────────────
        // If discourse resolved, use it for focus-ranked, POV-aware AbsTree.
        if let Ok(ref ctx) = discourse_result {
            if let Some(discourse_tree) = build_discourse_response(&all_triples, ctx, engine) {
                let registry = crate::grammar::GrammarRegistry::new();
                if let Ok(prose) = registry.linearize(grammar, &discourse_tree) {
                    if !prose.trim().is_empty() {
                        // Grammar output is the final answer — no LLM involvement.
                        msgs.push(AkhMessage::narrative(&prose, grammar));
                        return;
                    }
                }
            }
        }

        // ── Step 7: Fallback — synthesize_from_triples + push_summary ────
        let summary = crate::agent::synthesize::synthesize_from_triples(
            subject,
            &all_triples,
            engine,
            grammar,
        );
        self.push_summary(msgs, &summary, grammar);
    }

    /// Extract question word from raw input using the lexicon.
    fn extract_question_word(input: &str, lexicon: &Lexicon) -> Option<QuestionWord> {
        let lower = input.trim().to_lowercase();
        let first_word = lower.split_whitespace().next()?;
        let category = lexicon.classify_question_word(first_word)?;
        match category {
            "what" => Some(QuestionWord::What),
            "who" => Some(QuestionWord::Who),
            "where" => Some(QuestionWord::Where),
            "when" => Some(QuestionWord::When),
            "how" => Some(QuestionWord::How),
            "why" => Some(QuestionWord::Why),
            "which" => Some(QuestionWord::Which),
            "yesno" => Some(QuestionWord::YesNo),
            _ => None,
        }
    }

    /// Detect capability modal signal ("can", "could", etc.) in raw input.
    fn detect_capability_signal(input: &str, lexicon: &Lexicon) -> bool {
        let lower = input.trim().to_lowercase();
        // Check if any word in the input is a capability modal.
        lower.split_whitespace().any(|w| lexicon.is_capability_modal(w))
    }

    // ── KG-based query decomposition ──────────────────────────────────

    /// Try to resolve a compound subject via KG-based token classification.
    ///
    /// Algorithm:
    /// 1. Split into tokens, filter void words
    /// 2. Resolve each token against the KG (known vs unknown)
    /// 3. Pick primary entity: single unknown → it; else head-final (last token)
    /// 4. For known tokens, assert context triples (`is-a` or `part-of`)
    /// 5. If nothing resolves at all → last token as escalation target (no triples)
    fn try_decomposed_resolution(
        &self,
        subject: &str,
        engine: &Arc<Engine>,
    ) -> Option<(SymbolId, bool)> {
        let tokens: Vec<String> = subject
            .split_whitespace()
            .filter(|t| !self.lexicon.is_void(t))
            .map(|t| t.to_lowercase())
            .collect();

        if tokens.len() <= 1 {
            return None;
        }

        // Partition tokens into known (resolves in KG) and unknown.
        let mut known: Vec<(usize, SymbolId)> = Vec::new();
        let mut unknown: Vec<usize> = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            if let Ok(sym_id) = engine.resolve_symbol(token) {
                known.push((i, sym_id));
            } else {
                unknown.push(i);
            }
        }

        // If nothing resolves at all, use last token as escalation target.
        if known.is_empty() && unknown.len() == tokens.len() {
            return None;
        }

        // Primary entity selection.
        let primary_idx = match unknown.len() {
            1 => unknown[0],                   // single unknown → it's the primary
            0 => tokens.len() - 1,             // all known → head-final
            _ => *unknown.last().unwrap(),      // multiple unknowns → head-final among unknowns
        };

        let primary_label = &tokens[primary_idx];

        // Resolve or create the primary entity.
        let primary_id = if let Some((_, sym_id)) = known.iter().find(|(i, _)| *i == primary_idx) {
            *sym_id
        } else {
            engine.resolve_or_create_entity(primary_label).ok()?
        };

        // Assert context triples for known tokens that are NOT the primary.
        let mut created = false;
        for &(i, known_id) in &known {
            if i == primary_idx {
                continue;
            }
            if let Some(pred_id) = Self::pick_context_predicate(engine, known_id) {
                let triple = Triple::new(primary_id, pred_id, known_id).with_confidence(0.65);
                let _ = engine.add_triple(&triple);
                created = true;
            }
        }

        if created {
            let known_labels: Vec<&str> = known
                .iter()
                .filter(|(i, _)| *i != primary_idx)
                .map(|(i, _)| tokens[*i].as_str())
                .collect();
            tracing::info!(
                primary = %primary_label,
                context = ?known_labels,
                "KG-based decomposition — provisional triples created"
            );
        }

        Some((primary_id, created))
    }

    /// Determine the context predicate for a known token relative to a primary entity.
    ///
    /// Walks `is-a` / `subclass-of` ancestors of `known_id`:
    /// - If any ancestor is `artifact`, `tool`, `concept`, or `abstract-thing` → `is-a`
    /// - Otherwise → `part-of` (generic ecosystem/context relation)
    fn pick_context_predicate(engine: &Arc<Engine>, known_id: SymbolId) -> Option<SymbolId> {
        let type_ancestors = ["artifact", "tool", "concept", "abstract-thing"];
        let triples = engine.triples_from(known_id);

        // Walk up to 3 levels of is-a/subclass-of.
        let mut frontier: Vec<SymbolId> = Vec::new();
        for t in &triples {
            let pred_label = engine.resolve_label(t.predicate);
            if pred_label == "is-a" || pred_label == "subclass-of" {
                let obj_label = engine.resolve_label(t.object);
                if type_ancestors.contains(&obj_label.as_str()) {
                    return engine.resolve_or_create_relation("is-a").ok();
                }
                frontier.push(t.object);
            }
        }

        // Second hop.
        for hop_id in &frontier {
            let hop_triples = engine.triples_from(*hop_id);
            for t in &hop_triples {
                let pred_label = engine.resolve_label(t.predicate);
                if pred_label == "is-a" || pred_label == "subclass-of" {
                    let obj_label = engine.resolve_label(t.object);
                    if type_ancestors.contains(&obj_label.as_str()) {
                        return engine.resolve_or_create_relation("is-a").ok();
                    }
                }
            }
        }

        // Default: generic context relation.
        engine.resolve_or_create_relation("part-of").ok()
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
    ///
    /// Tool results and goal metadata are routed to audit log — only the
    /// persona-framed investigation message and final narrative appear in chat.
    fn escalate_to_goal(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        grammar: &str,
        description: &str,
        _display_prefix: &str,
    ) {
        // Investigation framing — always use template. The LLM hallucinates
        // answers instead of acknowledgments, so we don't use it here.
        let framing = format!("Let me look into {description} for you.");
        msgs.push(AkhMessage::narrative(&framing, grammar));

        // Snapshot WM size before adding the goal so we can scope synthesis.
        let wm_before = agent.working_memory().len();

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

        // Goal creation → audit log only
        msgs.push(AkhMessage::audit_log(
            goal_id.get(),
            "goal",
            format!("Created: {description}"),
        ));

        let max = agent
            .config
            .max_cycles
            .clamp(1, self.config.max_escalation_cycles);
        for _ in 0..max {
            match agent.run_cycle() {
                Ok(result) => {
                    // Tool results → audit log only
                    msgs.push(AkhMessage::audit_log(
                        0,
                        &result.decision.chosen_tool,
                        &result.action_result.tool_output.result,
                    ));
                }
                Err(_) => break,
            }
            if crate::agent::goal::active_goals(agent.goals()).is_empty() {
                break;
            }
        }

        // Synthesize only from entries added during this escalation.
        let all_entries = agent.working_memory().entries();
        let new_entries = if wm_before < all_entries.len() {
            &all_entries[wm_before..]
        } else {
            &[]
        };
        let summary = crate::agent::synthesize::synthesize_with_grammar(
            description,
            new_entries,
            agent.engine(),
            grammar,
        );

        self.push_summary(msgs, &summary, grammar);
    }

    /// Handle an unknown subject: friendly message then escalate.
    fn handle_unknown_subject(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        grammar: &str,
        _subject: &str,
        original_input: &str,
    ) {
        // The investigation framing in escalate_to_goal handles the message
        self.escalate_to_goal(msgs, agent, grammar, original_input, "");
    }

    /// Extract persona name and traits from the engine's Psyche compartment.
    fn persona_name_and_traits(engine: &Engine) -> (String, Vec<String>) {
        engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .map(|p| (p.persona.name.clone(), p.persona.traits.clone()))
            .unwrap_or_else(|| {
                // Default Psyche name is "Scholar" — "Akh" is the species, not the name.
                ("Scholar".to_string(), vec!["precise".to_string(), "curious".to_string()])
            })
    }

    /// Get the label of the active dialogue topic (if any).
    fn active_topic_label(agent: &Agent, engine: &Engine) -> Option<String> {
        agent
            .dialogue_manager()
            .query_active_topic(engine)
            .map(|tid| engine.resolve_label(tid))
    }

    /// Push a NarrativeSummary as a single cohesive narrative message.
    ///
    /// Uses grammar output directly — no LLM involvement. The narrative grammar
    /// already produces correct, focused prose.
    fn push_summary(
        &self,
        msgs: &mut Vec<AkhMessage>,
        summary: &crate::agent::NarrativeSummary,
        grammar: &str,
    ) {
        let has_content = !summary.overview.is_empty()
            || !summary.sections.is_empty()
            || !summary.gaps.is_empty();

        if !has_content {
            msgs.push(AkhMessage::narrative(
                "I couldn't find enough information to answer that. \
                 Try teaching me about it first, or rephrase your question.",
                grammar,
            ));
            return;
        }

        // Build grammar prose from sections (the grammar already did the thinking).
        let mut parts = Vec::new();
        // Only include meta-overview if there are no sections.
        if !summary.overview.is_empty() && summary.sections.is_empty() {
            parts.push(summary.overview.clone());
        }
        for section in &summary.sections {
            if section.heading.to_lowercase() != "overview" && !section.prose.is_empty() {
                parts.push(format!(
                    "Regarding {}: {}",
                    section.heading.to_lowercase(),
                    section.prose
                ));
            } else if !section.prose.is_empty() {
                parts.push(section.prose.clone());
            }
        }
        if !summary.gaps.is_empty() {
            let gap_text = if summary.gaps.len() == 1 {
                format!("I'm not sure about {} yet.", summary.gaps[0])
            } else {
                format!(
                    "There are a few things I don't know yet: {}.",
                    summary.gaps.join(", ")
                )
            };
            parts.push(gap_text);
        }

        let grammar_prose = parts.join(" ");

        if grammar_prose.trim().is_empty() {
            msgs.push(AkhMessage::narrative(
                "I couldn't find enough information to answer that. \
                 Try teaching me about it first, or rephrase your question.",
                grammar,
            ));
            return;
        }

        msgs.push(AkhMessage::narrative(&grammar_prose, grammar));
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
}
