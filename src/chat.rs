//! Unified input processing for all chat surfaces (TUI, daemon WS, headless CLI).
//!
//! `ChatProcessor` replaces the duplicated intent handling that previously lived
//! separately in TUI's `process_input_local()`, akhomed's `process_ws_input()`,
//! and the headless CLI's inline dispatch. All non-command user input flows
//! through `ChatProcessor::process_input()`, which implements:
//!
//! 1. Structural command fast-path (help, status, goals, etc.)
//! 2. ELIZA routing — classify before NLU; creative queries → LLM, factual → KG
//! 3. NLU pipeline parse → AbsTree dispatch (dialogue acts, facts, queries)
//! 4. Escalation to autonomous goal when all else fails

use std::sync::Arc;

use crate::agent::Agent;
use crate::agent::eliza::{ElizaRouter, ElizaRouting, FallbackReason};
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
use crate::nlu::kg_context::{KgContextCache, build_kg_context};
use crate::symbol::SymbolId;

// ── Query decomposition ─────────────────────────────────────────────────

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for the chat processor.
pub struct ChatProcessorConfig {
    /// Grammar archetype for narrative output (e.g. "narrative", "clinical").
    pub grammar: String,
    /// Maximum OODA cycles when escalating to a goal in chat context.
    pub max_escalation_cycles: usize,
    /// When `false`, the processor operates in pure-read mode: no goals are created
    /// and no OODA cycles run. Unresolved input synthesises from existing knowledge
    /// instead of escalating to a new goal. Defaults to `true` for TUI/WS sessions.
    pub create_goal: bool,
    /// Phase 34: communicative intelligence on the grounded response path
    /// (strategy selection + psyche style transform + shadow filter).
    /// Defaults to `true`; set env `AKH_COMM_INTEL=0` to disable.
    pub communicative: bool,
}

impl Default for ChatProcessorConfig {
    fn default() -> Self {
        Self {
            grammar: "narrative".to_string(),
            max_escalation_cycles: 10,
            create_goal: true,
            communicative: communicative_default(),
        }
    }
}

/// Communicative pipeline default: on, unless `AKH_COMM_INTEL=0`.
fn communicative_default() -> bool {
    std::env::var("AKH_COMM_INTEL").map(|v| v != "0").unwrap_or(true)
}

// ── SessionContext ──────────────────────────────────────────────────────

/// Per-session cognitive context, isolated from the shared [`ChatProcessor`].
///
/// Each session (TUI, WS handler, headless, ACP chat) owns its own
/// `SessionContext` so the KG context cache, ELIZA routing state, and
/// discourse trajectory don't bleed across sessions sharing one
/// `ChatProcessor` (which is `Arc<Mutex<ChatProcessor>>` — one per workspace).
///
/// This is the foundation for delegation: a delegated session gets its own
/// `SessionContext` so it can't pollute the calling session's cognitive state.
pub struct SessionContext {
    /// Session-scoped KG context cache for LLM prompt injection.
    pub kg_cache: KgContextCache,
    /// ELIZA routing classifier (KG vs LLM decision).
    pub eliza: ElizaRouter,
    /// Discourse trajectory (Phase 34d): topic repetition, discourse mode,
    /// strategy history, engagement estimate.
    pub trajectory: crate::agent::trajectory::DiscourseTrajectory,
}

impl Default for SessionContext {
    fn default() -> Self {
        Self {
            kg_cache: KgContextCache::new(),
            eliza: ElizaRouter::new(),
            trajectory: crate::agent::trajectory::DiscourseTrajectory::default(),
        }
    }
}

impl SessionContext {
    /// Create a fresh per-session context.
    pub fn new() -> Self {
        Self::default()
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
///
/// ELIZA routing is applied before NLU: creative/open-ended queries go to
/// the LLM with KG-fact context, factual queries go to the KG pipeline.
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
        ctx: &mut SessionContext,
        text: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // ── Discourse trajectory (Phase 34d) ────────────────────────
        // Update discourse mode + engagement from every user turn.
        ctx.trajectory.observe_input(trimmed);

        // ── Structural command fast-path (skip NLU) ─────────────────
        // These are system commands where NLU parsing would be wasteful.
        if is_structural_command(trimmed) {
            #[allow(deprecated)]
            let intent = classify_intent(trimmed);
            return self.dispatch_structural(ctx, intent, trimmed, agent, engine);
        }

        // ── Polar (yes/no) + relational-wh fast-path (read-only KG) ─────
        // Answer "Is/Are X Y?" and "Where is X?" / "What is X in?" straight from
        // a bounded transitive KG walk — instant, no LLM, no writes — before NLU
        // can fall to the slow LLM tier and mis-store the question as a fact.
        if let Some(msgs) = self.try_answer_polar(engine, trimmed) {
            return msgs;
        }
        if let Some(msgs) = self.try_answer_wh_relational(engine, trimmed) {
            return msgs;
        }

        // ── ELIZA routing — classify before NLU ─────────────────────
        // Creative/open-ended queries bypass the KG path and go to the LLM
        // with KG-fact context injected. Factual queries continue to the NLU
        // pipeline which will ground them against the knowledge graph.
        let eliza_routing = ctx.eliza.classify(trimmed);
        if let ElizaRouting::LlmFallback { reason } = &eliza_routing {
            if matches!(reason, FallbackReason::Creative) {
                return self.handle_creative_llm_fallback(ctx, trimmed, agent, engine);
            }
            // LowConfidence: let NLU try — it may still parse well enough.
        }

        // ── NLU pipeline (all other input) ──────────────────────────
        let parse_ctx = ParseContext::with_engine(
            engine.registry(),
            engine.ops(),
            engine.item_memory(),
        );

        // Build KG context for LLM tiers (gracefully absent when KG is empty).
        let kg_context = build_kg_context(engine, trimmed, &mut ctx.kg_cache);

        match self.nlu_pipeline.parse_with_context(trimmed, &parse_ctx, kg_context.as_deref()) {
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
                self.dispatch_abstree(ctx, &nlu_result.tree, trimmed, agent, engine)
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

    /// Enable or disable goal creation (pure-read mode).
    ///
    /// When `false`, unresolved input synthesises from existing knowledge instead
    /// of escalating to a new goal. TUI and WebSocket sessions leave this `true`.
    pub fn set_create_goal(&mut self, create_goal: bool) {
        self.config.create_goal = create_goal;
    }

    // ── AbsTree dispatch ────────────────────────────────────────────

    /// Dispatch on an NLU-parsed AbsTree variant.
    fn dispatch_abstree(
        &mut self,
        ctx: &mut SessionContext,
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
                self.handle_dialogue_greeting(ctx, &mut msgs, agent, engine, &grammar, raw_input);
            }
            AbsTree::Farewell { .. } => {
                self.handle_dialogue_farewell(ctx, &mut msgs, agent, engine, &grammar, raw_input);
            }
            AbsTree::Acknowledgment { .. } => {
                self.handle_dialogue_ack(ctx, &mut msgs, agent, engine, &grammar, raw_input);
            }
            AbsTree::FollowUpRequest { .. } => {
                self.handle_dialogue_follow_up(&mut msgs, agent, engine, &grammar);
            }
            AbsTree::MetaQuery { .. } => {
                self.handle_dialogue_meta_query(ctx, &mut msgs, agent, engine, &grammar);
            }
            AbsTree::GoalRequest { description } => {
                if self.config.create_goal {
                    let desc = description.label()
                        .map(String::from)
                        .unwrap_or_else(|| description.collect_labels().join(" "));
                    self.handle_set_goal(&mut msgs, agent, engine, &desc, &grammar);
                } else {
                    msgs.push(AkhMessage::system(
                        "Goal creation is disabled in read-only mode. \
                         Pass create_goal=true to enable investigation."
                            .to_string(),
                    ));
                }
            }
            AbsTree::StructuralCommand { command, args } => {
                self.handle_structural_command(&mut msgs, agent, engine, command, args);
            }

            // ── Assertable structures (triples, compounds) ──────────
            tree if tree.is_assertable() => {
                // Prefer asserting the parsed AbsTree directly (reuses the
                // high-quality NLU parse) over re-running the lossy regex
                // ingester; fall back to text ingest only if nothing landed.
                let asserted = Self::assert_abstree_triples(tree, engine, 0.9);
                if asserted > 0 {
                    let plural = if asserted == 1 { "" } else { "s" };
                    msgs.push(AkhMessage::tool_result(
                        "nlu_assert",
                        true,
                        format!("Learned {asserted} fact{plural}."),
                    ));
                } else {
                    msgs.extend(self.ingest_as_fact(raw_input, engine, "nlu_parse"));
                }
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
                        ctx,
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
        ctx: &mut SessionContext,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
        _raw_input: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let active_topic = Self::active_topic_label(agent, engine);
        // Build KG context for the greeting (topic-aware).
        let kg_ctx = active_topic
            .as_deref()
            .and_then(|topic| build_kg_context(engine, topic, &mut ctx.kg_cache));
        // Try LLM-generated greeting — but validate it mentions the persona name.
        // Small LLMs often ignore the system prompt and produce generic greetings.
        let text = self
            .nlu_pipeline
            .generate_dialogue_with_kg("greeting", &name, &traits, active_topic.as_deref(), kg_ctx.as_deref())
            .filter(|t| t.to_lowercase().contains(&name.to_lowercase()))
            .unwrap_or_else(|| {
                agent
                    .dialogue_manager()
                    .handle_greeting(engine, &name, &traits)
            });
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state().record_agent_turn(&text);
    }

    fn handle_dialogue_farewell(
        &mut self,
        ctx: &mut SessionContext,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
        raw_input: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let kg_ctx = build_kg_context(engine, raw_input, &mut ctx.kg_cache);
        let text = self
            .nlu_pipeline
            .generate_dialogue("farewell", &name, &traits, None, kg_ctx.as_deref())
            .unwrap_or_else(|| agent.dialogue_manager().handle_farewell(&name));
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state().record_agent_turn(&text);
    }

    fn handle_dialogue_ack(
        &mut self,
        ctx: &mut SessionContext,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
        raw_input: &str,
    ) {
        let (name, traits) = Self::persona_name_and_traits(engine);
        let kg_ctx = build_kg_context(engine, raw_input, &mut ctx.kg_cache);
        let text = self
            .nlu_pipeline
            .generate_dialogue("acknowledgment", &name, &traits, None, kg_ctx.as_deref())
            .unwrap_or_else(|| agent.dialogue_manager().handle_ack(&traits));
        msgs.push(AkhMessage::narrative(&text, grammar));
        agent.conversation_state().record_agent_turn(&text);
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
                agent.conversation_state().record_agent_turn(&text);
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
                        agent.conversation_state().record_agent_turn(&rendered);
                    }
                }
            }
        }
    }

    fn handle_dialogue_meta_query(
        &mut self,
        ctx: &mut SessionContext,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        grammar: &str,
    ) {
        // Route through the standard query pipeline with subject "self".
        // This produces a first-person, discourse-ranked response from the KG
        // using the same grammar pipeline as any other query.
        self.handle_query_from_tree(ctx, msgs, agent, engine, "self", "who are you?", grammar);
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
        ctx: &mut SessionContext,
        intent: UserIntent,
        raw_input: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let mut msgs = Vec::new();
        let grammar = self.config.grammar.clone();

        match intent {
            UserIntent::SetGoal { description } => {
                if self.config.create_goal {
                    self.handle_set_goal(&mut msgs, agent, engine, &description, &grammar);
                } else {
                    msgs.push(AkhMessage::system(
                        "Goal creation is disabled in read-only mode. \
                         Pass create_goal=true to enable investigation."
                            .to_string(),
                    ));
                }
            }
            UserIntent::Query {
                subject,
                original_input,
                question_word: _,
                capability_signal: _,
            } => {
                self.handle_query_from_tree(
                    ctx,
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
        ctx: &mut SessionContext,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        engine: &Arc<Engine>,
        subject: &str,
        original_input: &str,
        grammar: &str,
    ) {
        // Phase 12b+12c: Try grounded dialogue with constraint checking.
        let grounded = crate::agent::conversation::ground_query(subject, engine, grammar);
        if let Some(gr) = grounded {
            // ── Phase 34f: communicative pipeline on the grounded path ──
            // Strategy selection (34a) → shadow filter (34e) → content
            // filter (34c) → psyche style transform (34b), then the
            // composed prose flows through the existing constraint check
            // and rendering — one wiring point for all chat surfaces.
            let (gr, shadow_journal) =
                self.apply_communicative(ctx, gr, subject, original_input, agent, engine);
            let (out_msg, decision) = agent.check_and_wrap_grounded(
                &gr,
                "operator",
                crate::agent::ChannelKind::Operator,
            );
            if decision == crate::agent::constraint_check::EmissionDecision::Emit {
                let detail = agent.conversation_state().response_detail;
                let rendered = gr.render(detail);
                agent.conversation_state().record_agent_turn(&rendered);
                agent
                    .conversation_state()
                    .track_referent(subject.to_string());

                // Persist active topic to KG via dialogue manager.
                if let Ok(sym_id) = engine.resolve_symbol(subject) {
                    let _ = agent.dialogue_manager().set_active_topic(engine, sym_id);
                    agent.conversation_state().active_topic = Some(sym_id);
                }

                // Conversational boundary: rephrase the grounded answer into
                // natural chat prose if a verbalizer (Candle) is loaded,
                // constrained to the grounded content. Falls back to the
                // rendered messages.
                if !self.maybe_verbalize(msgs, engine, original_input, &rendered) {
                    for akh_msg in out_msg.to_akh_messages() {
                        msgs.push(akh_msg);
                    }
                }
                if !out_msg.constraint_check.is_passed() {
                    msgs.push(AkhMessage::system(
                        "[constraint check: some violations detected]".to_string(),
                    ));
                }
            }
            // Shadow encounters are journaled to the audit log (Phase 34e),
            // visible via psyche exposure — never in the chat stream.
            if let Some(journal) = shadow_journal {
                msgs.push(AkhMessage::audit_log(0, "shadow", journal));
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
                        // Log as NLG training pair (Phase 27f).
                        let tree_desc = format!("{discourse_tree:?}");
                        self.nlu_pipeline.log_nlg_pair(&tree_desc, &prose, grammar);
                        // Conversational boundary: if a verbalizer (Candle) is
                        // available, rephrase the grounded prose into chat-like
                        // text; otherwise emit the grammar prose directly.
                        if self.maybe_verbalize(msgs, engine, original_input, &prose) {
                            return;
                        }
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
        // Conversational boundary: verbalize the grounded summary if a
        // verbalizer (Candle) is available, else emit the synthesized summary.
        let mut facts_text = summary.overview.clone();
        for section in &summary.sections {
            if !section.prose.is_empty() {
                if !facts_text.is_empty() {
                    facts_text.push('\n');
                }
                facts_text.push_str(&section.prose);
            }
        }
        if self.maybe_verbalize(msgs, engine, original_input, &facts_text) {
            return;
        }
        self.push_summary(msgs, &summary, grammar);
    }

    /// Phase 34f: run the communicative pipeline over a grounded response.
    ///
    /// Selects a response strategy (34a) from the psyche + trajectory +
    /// affective state, runs the shadow filter (34e), filters content (34c),
    /// and applies the psyche style transform (34b). Returns the transformed
    /// response plus an optional shadow-journal entry.
    ///
    /// Skipped (returns the response untouched) when the pipeline is
    /// disabled (`AKH_COMM_INTEL=0` / `config.communicative == false`) or
    /// the operator asked for `concise` detail — terse output should stay
    /// terse.
    fn apply_communicative(
        &mut self,
        ctx: &mut SessionContext,
        gr: crate::agent::conversation::GroundedResponse,
        subject: &str,
        original_input: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> (crate::agent::conversation::GroundedResponse, Option<String>) {
        use crate::agent::communicative::{
            classify_query_type, compose_grounded, context_from_grounded,
        };
        use crate::agent::conversation::ResponseDetail;
        use crate::agent::response_strategy::Sentiment;

        if !self.config.communicative
            || agent.conversation_state().response_detail == ResponseDetail::Concise
        {
            return (gr, None);
        }

        // Psyche: loaded compartment, or the default Scholar psyche.
        let psyche = engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .unwrap_or_default();

        // Affective signal (Phase 23): current mood valence → user-facing tone.
        let sentiment = Sentiment::from_valence(agent.affective_system.current_affect().valence);

        let query_type = classify_query_type(original_input, subject);
        let comm_ctx = context_from_grounded(
            &gr,
            subject,
            query_type,
            &ctx.trajectory,
            Some(sentiment),
        );
        let composed = compose_grounded(&gr, subject, &psyche, &comm_ctx);

        tracing::debug!(
            subject = %subject,
            strategy = composed.strategy,
            companion = ?composed.companion,
            mode = %ctx.trajectory.discourse_mode(),
            "communicative strategy selected"
        );

        // Record this response turn in the trajectory (34d).
        ctx.trajectory.record_turn(
            Some(subject),
            engine.resolve_symbol(subject).ok(),
            composed.strategy,
        );

        let shadow_journal = composed.shadow_journal.clone();
        let transformed = crate::agent::conversation::GroundedResponse {
            prose: composed.text,
            ..gr
        };
        (transformed, shadow_journal)
    }

    /// Enable or disable the communicative pipeline (Phase 34f).
    pub fn set_communicative(&mut self, enabled: bool) {
        self.config.communicative = enabled;
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

    // ── ELIZA LLM fallback ──────────────────────────────────────────

    /// Handle a creative or open-ended query via LLM with KG-fact context.
    ///
    /// Flow:
    /// 1. Extract noun-like tokens from the input and query the KG for each.
    /// 2. Summarize any found triples as short text context.
    /// 3. Call the LLM with persona + KG context + the original query.
    /// 4. If LLM unavailable, fall through to OODA goal escalation.
    fn handle_creative_llm_fallback(
        &mut self,
        _ctx: &mut SessionContext,
        original_input: &str,
        agent: &mut Agent,
        engine: &Arc<Engine>,
    ) -> Vec<AkhMessage> {
        let grammar = self.config.grammar.clone();
        let mut msgs = Vec::new();

        // Collect KG facts for any nouns in the input.
        let kg_context = self.collect_kg_context_for_input(original_input, engine);
        let kg_context_str = kg_context.join("; ");
        let kg_context_opt = if kg_context_str.is_empty() {
            None
        } else {
            Some(kg_context_str.as_str())
        };

        let (name, traits) = Self::persona_name_and_traits(engine);

        tracing::info!(
            query = original_input,
            kg_facts = kg_context.len(),
            "ELIZA: routing creative query to LLM fallback"
        );

        // Try LLM generation with persona + KG context.
        if let Some(text) = self
            .nlu_pipeline
            .generate_open_ended(original_input, &name, &traits, kg_context_opt)
        {
            if !text.trim().is_empty() {
                agent.conversation_state().record_agent_turn(&text);
                msgs.push(AkhMessage::narrative(&text, &grammar));
                return msgs;
            }
        }

        // LLM unavailable or returned empty — fall back to OODA goal escalation.
        tracing::debug!("LLM fallback returned nothing; escalating to goal");
        self.escalate_to_goal(
            &mut msgs,
            agent,
            &grammar,
            original_input,
            &format!("Exploring \"{original_input}\"."),
        );
        msgs
    }

    /// Extract KG facts for noun-like tokens in `input`.
    ///
    /// Filters stop words and tries each token against the KG. Returns a vec
    /// of short fact strings like "Osiris is-a deity (confidence: 0.95)".
    fn collect_kg_context_for_input(&self, input: &str, engine: &Arc<Engine>) -> Vec<String> {
        let mut facts = Vec::new();
        let stop_words = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "to", "of", "for", "in", "on", "at", "by", "with", "from", "into",
            "and", "or", "but", "if", "then", "that", "this", "it", "me", "you",
            "we", "they", "he", "she", "what", "which", "who", "write", "about",
            "tell", "please", "create", "make", "compose", "imagine", "poem",
            "story", "essay", "blog", "joke",
        ];

        let seen_tokens: std::collections::HashSet<String> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| !w.is_empty() && !stop_words.contains(&w.as_str()))
            .collect();

        for token in &seen_tokens {
            if let Ok(sym_id) = engine.resolve_symbol(token) {
                let triples = engine.triples_from(sym_id);
                for triple in triples.iter().take(3) {
                    let subj_label = engine.resolve_label(triple.subject);
                    let pred_label = engine.resolve_label(triple.predicate);
                    let obj_label = engine.resolve_label(triple.object);
                    facts.push(format!(
                        "{subj_label} {pred_label} {obj_label} (confidence: {:.2})",
                        triple.confidence
                    ));
                }
            }
        }

        facts.truncate(10); // Cap context size to avoid prompt overflow.
        facts
    }

    // ── Shared helpers ──────────────────────────────────────────────

    /// Ingest text as a fact using the pattern-based text extractor.
    fn ingest_as_fact(
        &self,
        text: &str,
        engine: &Engine,
        tool_label: &str,
    ) -> Vec<AkhMessage> {
        let (n, _symbols) = crate::agent::text_extract::ingest_text(engine, text, 100);
        let msg = format!("Text ingest: extracted {n} triple(s).");
        vec![AkhMessage::tool_result(tool_label, true, &msg)]
    }

    /// Escalate unresolved input to a goal, run OODA cycles, and synthesize findings.
    ///
    /// Tool results and goal metadata are routed to audit log — only the
    /// persona-framed investigation message and final narrative appear in chat.
    ///
    /// When `create_goal` is `false` (pure-read mode), no goal is created and no
    /// OODA cycles run. The method synthesises from existing working memory instead,
    /// returning an "I don't know" response when the KG has nothing relevant.
    fn escalate_to_goal(
        &mut self,
        msgs: &mut Vec<AkhMessage>,
        agent: &mut Agent,
        grammar: &str,
        description: &str,
        _display_prefix: &str,
    ) {
        if !self.config.create_goal {
            // Pure-read mode: synthesise from whatever is already in working memory
            // without creating a goal or running any OODA cycles.
            let entries = agent.working_memory().entries().to_vec();
            let summary = crate::agent::synthesize::synthesize_with_grammar(
                description,
                &entries,
                agent.engine(),
                grammar,
            );
            self.push_summary(msgs, &summary, grammar);
            return;
        }

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
    ///
    /// Public so UIs (e.g. the TUI chat) can label assistant turns with the
    /// engine's persona name.
    pub fn persona_name_and_traits(engine: &Engine) -> (String, Vec<String>) {
        engine
            .compartments()
            .and_then(|cm| cm.psyche())
            .map(|p| (p.persona.name.clone(), p.persona.traits.clone()))
            .unwrap_or_else(|| {
                // Default Psyche name is "Scholar" — "Akh" is the species, not the name.
                ("Scholar".to_string(), vec!["precise".to_string(), "curious".to_string()])
            })
    }

    /// Answer a polar (yes/no) question read-only from the KG, before NLU.
    ///
    /// Intercepts "Is/Are/Does X …?" so the answer comes straight from a bounded
    /// transitive KG walk (instant, no LLM, no writes) instead of falling to the
    /// slow LLM tier and getting mis-stored as a provisional fact. Returns `None`
    /// when `input` is not a polar question.
    fn try_answer_polar(&self, engine: &Engine, input: &str) -> Option<Vec<AkhMessage>> {
        let (subject, aux, object) = parse_polar_question(input)?;
        let grammar = &self.config.grammar;
        let text = match polar_verdict(engine, &subject, &object) {
            PolarVerdict::Yes { direct } => {
                if direct {
                    format!("Yes, {subject} {aux} {object}.")
                } else {
                    // Held via a chain (e.g. dogs → mammals → animals).
                    format!("Yes, {subject} {aux} {object} — by what I know.")
                }
            }
            PolarVerdict::No => format!("No, {subject} {aux} not {object}."),
            PolarVerdict::Unknown { subject_known } => {
                if subject_known {
                    format!(
                        "Not that I know of — I don't have anything linking {subject} to {object}."
                    )
                } else {
                    format!("I don't know about {subject} yet, so I can't say whether it {aux} {object}.")
                }
            }
        };
        Some(vec![AkhMessage::narrative(text, grammar)])
    }

    /// Answer a relational wh-question ("What is X in?", "Where is X?") by
    /// walking the KG transitively (read-only). Returns `None` if `input` is not
    /// such a question or the subject is unknown.
    fn try_answer_wh_relational(&self, engine: &Engine, input: &str) -> Option<Vec<AkhMessage>> {
        let subject = parse_wh_relational(input)?;
        let grammar = &self.config.grammar;
        let objs = transitive_object_labels(engine, &subject);
        if objs.is_empty() {
            // Subject unknown or no relations — let the normal query path try.
            engine.resolve_symbol(&subject).ok()?;
            return Some(vec![AkhMessage::narrative(
                format!("I don't have anything on where {subject} is."),
                grammar,
            )]);
        }
        let objs: Vec<String> = objs.iter().map(|o| capitalize_first(o)).collect();
        let list = humanize_list(&objs);
        let subject = capitalize_first(&subject);
        Some(vec![AkhMessage::narrative(
            format!("{subject} is in {list}."),
            grammar,
        )])
    }

    /// Assert the triples in an already-parsed assertable [`AbsTree`] directly
    /// into the KG, resolving entity/relation labels to symbols.
    ///
    /// This reuses the NLU parse (especially the high-quality LLM tier) instead
    /// of re-running lossy regex extraction over the raw text — so teaching a
    /// fact the parser understood actually lands in the graph. Returns the
    /// number of triples asserted.
    fn assert_abstree_triples(tree: &AbsTree, engine: &Engine, confidence: f32) -> usize {
        match tree {
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let (Some(s), Some(p), Some(o)) =
                    (subject.label(), predicate.label(), object.label())
                else {
                    return 0;
                };
                let (s, p, o) = (s.trim(), p.trim(), o.trim());
                if s.is_empty() || p.is_empty() || o.is_empty() {
                    return 0;
                }
                match crate::agent::text_extract::ingest_extracted(
                    engine, s, p, o, confidence,
                ) {
                    Ok(_) => 1,
                    Err(e) => {
                        tracing::warn!(error = %e, "assert_abstree: triple insert failed");
                        0
                    }
                }
            }
            // Conjunctions assert each conjunct; disjunctions are ambiguous to assert.
            AbsTree::Conjunction { items, is_and: true } => items
                .iter()
                .map(|i| Self::assert_abstree_triples(i, engine, confidence))
                .sum(),
            // Modifier wrappers: descend, honoring an explicit confidence.
            AbsTree::WithConfidence { inner, confidence: c } => {
                Self::assert_abstree_triples(inner, engine, *c)
            }
            AbsTree::WithProvenance { inner, .. }
            | AbsTree::Modal { inner, .. }
            | AbsTree::Temporal { inner, .. } => {
                Self::assert_abstree_triples(inner, engine, confidence)
            }
            AbsTree::Quantified { scope, .. } => {
                Self::assert_abstree_triples(scope, engine, confidence)
            }
            AbsTree::RelativeClause { clause, .. } => {
                Self::assert_abstree_triples(clause, engine, confidence)
            }
            // Negation/Comparison/Conditional/Similarity carry semantics that a
            // plain positive triple can't represent — leave those to the
            // dedicated ingest path rather than asserting them incorrectly.
            _ => 0,
        }
    }

    /// Conversational output boundary: if a verbalizer (the Candle backend) is
    /// loaded, rephrase the grounded `facts` into a natural, chat-like reply to
    /// `question` and push it as the narrative. Returns `true` if it produced the
    /// reply, `false` if the caller should fall back to the grammar prose.
    ///
    /// The verbalizer is constrained to `facts` (which come from the KG /
    /// grammar), so this stays grounded — the LLM only changes phrasing, not
    /// content. Opt out per-request with `AKH_LLM_VERBALIZE=0`.
    fn maybe_verbalize(
        &self,
        msgs: &mut Vec<AkhMessage>,
        engine: &Engine,
        question: &str,
        facts: &str,
    ) -> bool {
        if !self.nlu_pipeline.has_verbalizer() {
            return false;
        }
        match std::env::var("AKH_LLM_VERBALIZE").as_deref() {
            Ok("0") | Ok("off") | Ok("false") | Ok("no") => return false,
            _ => {}
        }
        if facts.trim().is_empty() {
            return false;
        }
        let (name, traits) = Self::persona_name_and_traits(engine);
        // Channel-aware: surface the boundary LLM's reasoning (`thought`) as a
        // collapsible trace distinct from the answer, rather than dropping it
        // (ADR 069).
        if let Some(stream) =
            self.nlu_pipeline
                .verbalize_channels(question, facts, &name, &traits)
        {
            let answer = stream.answer();
            if !answer.trim().is_empty() {
                Self::push_thoughts(msgs, &stream);
                msgs.push(AkhMessage::narrative(answer.trim(), "conversational"));
                return true;
            }
        }
        false
    }

    /// Push the boundary LLM's internal reasoning (if any) as a dim, collapsible
    /// [`AkhMessage::Reasoning`] trace immediately before the answer narrative.
    /// The Ctrl-R reasoning-collapse machinery folds it by default (ADR 069).
    fn push_thoughts(msgs: &mut Vec<AkhMessage>, stream: &crate::nlu::channel::ChannelStream) {
        if stream.has_thoughts() {
            let thoughts = stream.thoughts();
            if !thoughts.trim().is_empty() {
                msgs.push(AkhMessage::Reasoning {
                    step: thoughts,
                    expression: None,
                });
            }
        }
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

// ── Polar (yes/no) question handling ────────────────────────────────────────

/// Verdict for a yes/no question, checked read-only against the KG.
enum PolarVerdict {
    /// The relation holds. `direct` is true if found on the subject itself,
    /// false if reached through a transitive chain.
    Yes { direct: bool },
    /// A negation/contradiction holds.
    No,
    /// No supporting fact found. `subject_known` distinguishes "I have the
    /// subject but no link" from "I don't know the subject at all".
    Unknown { subject_known: bool },
}

/// Auxiliaries that open a yes/no question.
const POLAR_AUX: &[&str] = &[
    "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "has", "have", "had",
];

/// Prepositions that act as the relation in an "X <prep> Y" polar question.
const POLAR_PREPS: &[&str] = &[
    "in", "at", "on", "of", "into", "inside", "within", "from", "near", "under", "above",
];

fn is_polar_void(w: &str) -> bool {
    matches!(w, "a" | "an" | "the")
}

/// Common irregular singular/plural pairs the simple `+s`/`-s` rule misses.
const IRREGULAR_PLURALS: &[(&str, &str)] = &[
    ("man", "men"),
    ("woman", "women"),
    ("person", "people"),
    ("child", "children"),
    ("foot", "feet"),
    ("tooth", "teeth"),
    ("mouse", "mice"),
    ("goose", "geese"),
    ("leaf", "leaves"),
    ("life", "lives"),
];

/// Surface variants of `word` to try when matching/resolving KG labels —
/// the word itself plus regular and irregular singular/plural forms. Lets
/// "Socrates is a **man**" connect to "All **men** are mortal".
fn inflect_variants(word: &str) -> Vec<String> {
    let w = word.trim().to_lowercase();
    if w.is_empty() {
        return Vec::new();
    }
    let mut out = vec![w.clone()];
    for (s, p) in IRREGULAR_PLURALS {
        if w == *s {
            out.push((*p).to_string());
        } else if w == *p {
            out.push((*s).to_string());
        }
    }
    if let Some(stem) = w.strip_suffix("es") {
        out.push(stem.to_string());
    }
    if let Some(stem) = w.strip_suffix('s') {
        out.push(stem.to_string());
    }
    if !w.ends_with('s') {
        out.push(format!("{w}s"));
    }
    out.sort();
    out.dedup();
    out
}

/// Parse a yes/no question into `(subject, auxiliary, object)`, or `None` if
/// `input` is not a polar question. The auxiliary (e.g. "are", "is") is kept so
/// the reply can be phrased with the question's own verb ("Yes, dogs are …").
fn parse_polar_question(input: &str) -> Option<(String, String, String)> {
    let lower = input.trim().to_lowercase();
    let body = lower.strip_suffix('?')?; // must end with '?'
    let mut toks: Vec<&str> = body.split_whitespace().collect();
    if toks.is_empty() || !POLAR_AUX.contains(&toks[0]) {
        return None;
    }
    let aux = toks.remove(0).to_string(); // drop + keep the auxiliary
    let toks: Vec<&str> = toks.into_iter().filter(|w| !is_polar_void(w)).collect();
    if toks.len() < 2 {
        return None;
    }
    // Preposition split: "paris in europe" → subj="paris", obj="europe".
    if let Some(pos) = toks.iter().position(|w| POLAR_PREPS.contains(w)) {
        if pos > 0 && pos + 1 < toks.len() {
            let subject = toks[..pos].join(" ");
            let rel = toks[pos];
            let object = toks[pos + 1..].join(" ");
            if !subject.is_empty() && !object.is_empty() {
                // Fold the preposition into the verb: "is in".
                return Some((subject, format!("{aux} {rel}"), object));
            }
        }
    }
    // Copula form: "dogs reptiles" → subj="dogs", obj="reptiles".
    Some((toks[0].to_string(), aux, toks[1..].join(" ")))
}

/// Read-only KG check: does `subject` relate to `object`? Bounded transitive
/// walk over taxonomic/locative predicates; the object is matched on any
/// predicate. No symbols or triples are created.
fn polar_verdict(engine: &Engine, subject: &str, object: &str) -> PolarVerdict {
    use std::collections::HashSet;
    // Resolve the subject, tolerating singular/plural ("dog"↔"dogs", "man"↔"men").
    let Some(subj_id) = inflect_variants(subject)
        .iter()
        .find_map(|v| engine.resolve_symbol(v).ok())
    else {
        return PolarVerdict::Unknown {
            subject_known: false,
        };
    };
    let obj_variants: HashSet<String> = inflect_variants(object).into_iter().collect();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    let mut frontier = vec![subj_id];
    let mut found_negation = false;

    for depth in 0..5 {
        let mut next = Vec::new();
        for id in frontier.drain(..) {
            if !seen.insert(id) {
                continue;
            }
            for t in engine.triples_from(id) {
                let pred = engine.resolve_label(t.predicate).to_lowercase();
                let obj_label = engine.resolve_label(t.object).to_lowercase();
                if obj_variants.contains(&obj_label) {
                    if pred.contains("not") || pred.contains("negat") || pred.contains("opposite")
                    {
                        found_negation = true;
                    } else {
                        return PolarVerdict::Yes { direct: depth == 0 };
                    }
                }
                if is_transitive_pred(&pred) {
                    next.push(t.object);
                    // Bridge singular/plural at intermediate hops (man → men),
                    // so chains like Socrates→man, "men are mortal" connect.
                    for v in inflect_variants(&obj_label) {
                        if let Ok(vid) = engine.resolve_symbol(&v) {
                            next.push(vid);
                        }
                    }
                }
            }
        }
        frontier = next;
    }
    if found_negation {
        PolarVerdict::No
    } else {
        PolarVerdict::Unknown {
            subject_known: true,
        }
    }
}

/// Parse a relational wh-question — "Where is X?" or "What is X <prep>?" —
/// into its subject. Returns `None` for plain "What is X?" (handled by the
/// normal query path) and for non-wh input.
fn parse_wh_relational(input: &str) -> Option<String> {
    let lower = input.trim().to_lowercase();
    let body = lower.strip_suffix('?')?;
    let toks: Vec<&str> = body.split_whitespace().collect();
    if toks.len() < 3 {
        return None;
    }
    let wh = toks[0];
    if wh != "what" && wh != "where" {
        return None;
    }
    if !POLAR_AUX.contains(&toks[1]) {
        return None;
    }
    let mut body: Vec<&str> = toks[2..]
        .iter()
        .copied()
        .filter(|w| !is_polar_void(w))
        .collect();
    let had_prep = body.last().is_some_and(|w| POLAR_PREPS.contains(w));
    if had_prep {
        body.pop();
    }
    if body.is_empty() {
        return None;
    }
    // Only handle the relational forms here; plain "what is X?" → normal query.
    if wh == "where" || had_prep {
        Some(body.join(" "))
    } else {
        None
    }
}

/// Collect the labels reachable from `subject` over transitive predicates
/// (read-only), nearest first — for answering "What is X in?" / "Where is X?".
fn transitive_object_labels(engine: &Engine, subject: &str) -> Vec<String> {
    use std::collections::HashSet;
    let Some(subj_id) = inflect_variants(subject)
        .iter()
        .find_map(|v| engine.resolve_symbol(v).ok())
    else {
        return Vec::new();
    };
    let subj_low = subject.trim().to_lowercase();
    let mut seen: HashSet<SymbolId> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    let mut frontier = vec![subj_id];
    for _ in 0..5 {
        let mut next = Vec::new();
        for id in frontier.drain(..) {
            if !seen.insert(id) {
                continue;
            }
            for t in engine.triples_from(id) {
                let pred = engine.resolve_label(t.predicate).to_lowercase();
                if !is_transitive_pred(&pred) {
                    continue;
                }
                let lab = engine.resolve_label(t.object);
                let low = lab.to_lowercase();
                // Skip infrastructure labels and the subject's own forms.
                if !lab.contains(':')
                    && low != subj_low
                    && !out.iter().any(|o| o.eq_ignore_ascii_case(&lab))
                {
                    out.push(lab.clone());
                }
                next.push(t.object);
                for v in inflect_variants(&low) {
                    if let Ok(vid) = engine.resolve_symbol(&v) {
                        next.push(vid);
                    }
                }
            }
        }
        frontier = next;
    }
    out
}

/// Capitalize the first character of `s` (for sentence-initial subjects).
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Join labels into a readable list: "France", "France and Europe",
/// "A, B and C".
fn humanize_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [a] => a.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Predicates that are transitive enough to follow for chains like
/// Paris→France→Europe or dog→mammal→animal.
///
/// Short copula/locative predicates are matched **exactly** (so "is"/"in"/"at"
/// chain, but "consists"/"this" don't accidentally match); longer relational
/// predicates are matched as substrings.
fn is_transitive_pred(pred: &str) -> bool {
    const EXACT: &[&str] = &[
        "is", "isa", "are", "be", "in", "at", "on", "was", "were",
    ];
    const CONTAINS: &[&str] = &[
        "is-a", "is a", "type", "kind", "subclass", "child-of", "instance",
        "member", "part-of", "part of", "located", "within", "inside", "subset",
    ];
    EXACT.contains(&pred) || CONTAINS.iter().any(|p| pred.contains(p))
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
