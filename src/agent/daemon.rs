//! Background daemon for autonomous learning.
//!
//! `AgentDaemon` runs a tokio event loop that periodically fires background
//! tasks: equivalence learning, reflection, consolidation, schema discovery,
//! rule inference, gap analysis, goal generation, sleep cycles, and idle OODA
//! cycles. The agent itself stays synchronous — tokio only drives scheduling
//! and signal handling.
//!
//! Both the standalone `akh-medu daemon` and the `akhomed` server reuse this
//! same `AgentDaemon`, so every background task added here automatically works
//! in both deployment modes.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::Agent;
use super::error::AgentResult;
use crate::client::DaemonStatus;
use crate::message::AkhMessage;

/// Current unix timestamp in seconds.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Intervals for each daemon background task.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Equivalence learning interval (default: 5 min).
    pub equivalence_interval: Duration,
    /// Reflection interval (default: 3 min).
    pub reflection_interval: Duration,
    /// Consolidation check interval (default: 1 min).
    pub consolidation_interval: Duration,
    /// Schema discovery interval (default: 30 min).
    pub schema_discovery_interval: Duration,
    /// Rule inference interval (default: 10 min).
    pub rule_inference_interval: Duration,
    /// Gap analysis interval (default: 15 min).
    pub gap_analysis_interval: Duration,
    /// Session persistence interval (default: 60s).
    pub persist_interval: Duration,
    /// Idle OODA cycle interval (default: 30s).
    pub idle_cycle_interval: Duration,
    /// Continuous learning interval (default: 2 hours).
    pub continuous_learning_interval: Duration,
    /// Goal generation interval (default: 5 min).
    ///
    /// Runs independently of the OODA cycle so that drives can propose goals
    /// even when no active goals exist — breaking the autonomy deadlock.
    pub goal_generation_interval: Duration,
    /// Sleep/consolidation cycle interval (default: 60 min).
    ///
    /// Runs KG reorganization (duplicate merge, orphan prune, dream
    /// connections) as a standalone timer so consolidation happens even when
    /// the OODA cycle is idle.
    pub sleep_cycle_interval: Duration,
    /// Trigger evaluation interval (default: 15s).
    ///
    /// Evaluates user-defined triggers against the current agent state and
    /// fires any whose conditions are met.
    pub trigger_evaluation_interval: Duration,
    /// Knowledge extraction interval (default: 45 min).
    ///
    /// Runs LLM-backed triple extraction on recently-fetched content that
    /// hasn't been processed yet. Uses the three-tier fallback: local LLM,
    /// external API, regex.
    pub knowledge_extraction_interval: Duration,
    /// Neural bridge training interval (default: 2 hours).
    ///
    /// Runs bridge encoder alignment training during idle periods.
    /// Only starts after sufficient training data has accumulated.
    pub bridge_training_interval: Duration,
    /// Maximum OODA cycles (0 = unlimited).
    pub max_cycles: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            equivalence_interval: Duration::from_secs(300),
            reflection_interval: Duration::from_secs(180),
            consolidation_interval: Duration::from_secs(60),
            schema_discovery_interval: Duration::from_secs(1800),
            rule_inference_interval: Duration::from_secs(600),
            gap_analysis_interval: Duration::from_secs(900),
            persist_interval: Duration::from_secs(60),
            idle_cycle_interval: Duration::from_secs(30),
            continuous_learning_interval: Duration::from_secs(7200),
            goal_generation_interval: Duration::from_secs(300),
            sleep_cycle_interval: Duration::from_secs(3600),
            trigger_evaluation_interval: Duration::from_secs(15),
            knowledge_extraction_interval: Duration::from_secs(2700), // 45 min
            bridge_training_interval: Duration::from_secs(7200), // 2 hours
            max_cycles: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------------

/// Long-running daemon that schedules background learning tasks.
///
/// The agent is shared via `Arc<Mutex<Agent>>` so that HTTP handlers,
/// WebSocket sessions, and the daemon all operate on the same agent
/// instance. Each daemon method locks briefly, does synchronous work,
/// and releases — near-zero contention since ticks are 15s–2h apart.
pub struct AgentDaemon {
    agent: Arc<Mutex<Agent>>,
    config: DaemonConfig,
    total_cycles: usize,
    /// Optional external shutdown signal (replaces Ctrl+C when set).
    ///
    /// Used by `akhomed` to integrate daemon shutdown with the HTTP server's
    /// watch channel instead of relying on process-level Ctrl+C.
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
    /// Optional shared status for real-time monitoring from HTTP handlers.
    status: Option<Arc<tokio::sync::Mutex<DaemonStatus>>>,
}

impl AgentDaemon {
    /// Create a new daemon wrapping the given shared agent.
    pub fn new(agent: Arc<Mutex<Agent>>, config: DaemonConfig) -> Self {
        Self {
            agent,
            config,
            total_cycles: 0,
            shutdown: None,
            status: None,
        }
    }

    /// Attach an external shutdown receiver (replaces Ctrl+C handling).
    pub fn with_shutdown(mut self, rx: tokio::sync::watch::Receiver<bool>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    /// Attach a shared status for real-time monitoring.
    pub fn with_status(mut self, status: Arc<tokio::sync::Mutex<DaemonStatus>>) -> Self {
        self.status = Some(status);
        self
    }

    /// Run the main daemon loop.
    ///
    /// Schedules all background tasks as tokio intervals and selects on them.
    /// Shuts down cleanly on Ctrl+C (or external shutdown signal), persisting
    /// the session before exit.
    pub async fn run(&mut self) -> AgentResult<()> {
        use tokio::time::interval;

        let mut equivalence_tick = interval(self.config.equivalence_interval);
        let mut reflection_tick = interval(self.config.reflection_interval);
        let mut consolidation_tick = interval(self.config.consolidation_interval);
        let mut schema_tick = interval(self.config.schema_discovery_interval);
        let mut rules_tick = interval(self.config.rule_inference_interval);
        let mut gaps_tick = interval(self.config.gap_analysis_interval);
        let mut persist_tick = interval(self.config.persist_interval);
        let mut idle_cycle_tick = interval(self.config.idle_cycle_interval);
        let mut continuous_learning_tick = interval(self.config.continuous_learning_interval);
        let mut goal_gen_tick = interval(self.config.goal_generation_interval);
        let mut sleep_tick = interval(self.config.sleep_cycle_interval);
        let mut trigger_tick = interval(self.config.trigger_evaluation_interval);
        let mut extraction_tick = interval(self.config.knowledge_extraction_interval);
        let mut bridge_training_tick = interval(self.config.bridge_training_interval);

        self.agent.lock().unwrap().sink().emit(&AkhMessage::system(
            "daemon started — background learning active",
        ));

        // Startup burst: seed goals so the akh can start acting immediately.
        // Continuous learning is deliberately NOT run at startup — it holds the
        // agent mutex for 30-120s (network I/O + ingestion), blocking all HTTP
        // handlers.  The interval timer will kick it off naturally.
        self.run_goal_generation();

        loop {
            tokio::select! {
                _ = equivalence_tick.tick() => {
                    self.run_equivalence_learning();
                }
                _ = reflection_tick.tick() => {
                    self.run_reflection();
                }
                _ = consolidation_tick.tick() => {
                    self.run_consolidation_check();
                }
                _ = schema_tick.tick() => {
                    self.run_schema_discovery();
                }
                _ = rules_tick.tick() => {
                    self.run_rule_inference();
                }
                _ = gaps_tick.tick() => {
                    self.run_gap_analysis();
                }
                _ = persist_tick.tick() => {
                    self.persist();
                }
                _ = idle_cycle_tick.tick() => {
                    self.run_idle_cycle();
                    if self.config.max_cycles > 0 && self.total_cycles >= self.config.max_cycles {
                        self.agent.lock().unwrap().sink().emit(&AkhMessage::system(format!(
                            "daemon: max cycles ({}) reached, shutting down",
                            self.config.max_cycles,
                        )));
                        break;
                    }
                }
                _ = continuous_learning_tick.tick() => {
                    self.run_continuous_learning();
                }
                _ = goal_gen_tick.tick() => {
                    self.run_goal_generation();
                }
                _ = sleep_tick.tick() => {
                    self.run_sleep_cycle();
                }
                _ = trigger_tick.tick() => {
                    self.run_trigger_evaluation();
                }
                _ = extraction_tick.tick() => {
                    self.run_knowledge_extraction();
                }
                _ = bridge_training_tick.tick() => {
                    self.run_bridge_training();
                }
                _ = self.wait_for_shutdown() => {
                    self.agent.lock().unwrap().sink().emit(&AkhMessage::system(
                        "daemon: received shutdown signal, persisting session...",
                    ));
                    break;
                }
            }
        }

        // Final persist on shutdown.
        self.persist();
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.running = false;
        }
        self.agent
            .lock()
            .unwrap()
            .sink()
            .emit(&AkhMessage::system("daemon stopped"));
        Ok(())
    }

    /// Wait for either the external shutdown receiver, Ctrl+C, or SIGTERM.
    async fn wait_for_shutdown(&mut self) {
        if let Some(ref mut rx) = self.shutdown {
            let _ = rx.changed().await;
        } else {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to register SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c.await.ok();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Background task methods
    // -----------------------------------------------------------------------

    fn run_equivalence_learning(&mut self) {
        let engine = self.agent.lock().unwrap().engine.clone();
        match engine.learn_equivalences() {
            Ok(count) => {
                if count > 0 {
                    if let Ok(agent) = self.agent.lock() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:equivalence] {count} new equivalences discovered",
                        )));
                    }
                }
                tracing::info!(count, "daemon: equivalence learning complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: equivalence learning failed");
            }
        }
    }

    fn run_reflection(&mut self) {
        let mut agent = self.agent.lock().unwrap();
        match agent.reflect() {
            Ok(result) => {
                let adj_count = result.adjustments.len();
                // Auto-apply safe priority adjustments.
                let safe: Vec<_> = result
                    .adjustments
                    .iter()
                    .filter(|a| {
                        matches!(
                            a,
                            super::reflect::Adjustment::IncreasePriority { .. }
                                | super::reflect::Adjustment::DecreasePriority { .. }
                        )
                    })
                    .cloned()
                    .collect();
                let applied = agent.apply_adjustments(&safe).unwrap_or(0);
                if adj_count > 0 {
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:reflection] {adj_count} insights, {applied} adjustments applied",
                    )));
                }
                tracing::info!(adj_count, applied, "daemon: reflection complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: reflection failed");
            }
        }
    }

    fn run_consolidation_check(&mut self) {
        let mut agent = self.agent.lock().unwrap();
        let wm_len = agent.working_memory().len();
        let threshold = agent.config.consolidation.auto_consolidate_at;
        if wm_len < threshold {
            return;
        }

        match agent.consolidate() {
            Ok(result) => {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:consolidation] {} entries evicted, {} episodes stored",
                    result.entries_evicted,
                    result.episodes_created.len(),
                )));
                tracing::info!(
                    evicted = result.entries_evicted,
                    episodes = result.episodes_created.len(),
                    "daemon: consolidation complete",
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: consolidation failed");
            }
        }
    }

    fn run_schema_discovery(&self) {
        let engine = self.agent.lock().unwrap().engine.clone();
        let config = crate::autonomous::schema::SchemaDiscoveryConfig::default();
        match engine.discover_schema(config) {
            Ok(result) => {
                let types = result.types.len();
                let hierarchies = result.relation_hierarchies.len();
                if types > 0 || hierarchies > 0 {
                    if let Ok(agent) = self.agent.lock() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:schema] discovered {types} types, {hierarchies} relation hierarchies",
                        )));
                    }
                }
                tracing::info!(types, hierarchies, "daemon: schema discovery complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: schema discovery failed");
            }
        }
    }

    fn run_rule_inference(&self) {
        let engine = self.agent.lock().unwrap().engine.clone();
        let config = crate::autonomous::rule_engine::RuleEngineConfig::default();
        match engine.run_rules(config) {
            Ok(result) => {
                let derived_count = result.derived.len();
                if derived_count > 0 {
                    if let Ok(agent) = self.agent.lock() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:rules] {} new triples from {} iterations",
                            derived_count, result.iterations,
                        )));
                    }
                }
                tracing::info!(
                    triples = derived_count,
                    iterations = result.iterations,
                    "daemon: rule inference complete",
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: rule inference failed");
            }
        }
    }

    fn run_gap_analysis(&self) {
        use super::goal;

        let (engine, goal_ids) = {
            let agent = self.agent.lock().unwrap();
            let active = goal::active_goals(agent.goals());
            if active.is_empty() {
                return;
            }
            let ids: Vec<_> = active.iter().map(|g| g.symbol_id).collect();
            (agent.engine.clone(), ids)
        };

        let config = crate::autonomous::gap::GapAnalysisConfig::default();
        match engine.analyze_gaps(&goal_ids, config) {
            Ok(result) => {
                let gaps = result.gaps.len();
                if gaps > 0 {
                    if let Ok(agent) = self.agent.lock() {
                        agent.sink().emit(&AkhMessage::system(format!(
                            "[daemon:gaps] {gaps} knowledge gaps identified around active goals",
                        )));
                    }
                }
                tracing::info!(gaps, "daemon: gap analysis complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: gap analysis failed");
            }
        }
    }

    fn run_continuous_learning(&self) {
        // Extract what we need from the agent, then DROP the lock before the
        // expensive learning pipeline runs.  `run_continuous_learning` only
        // needs an `&Arc<Engine>` and a curiosity config — it does not mutate
        // the Agent, so holding the mutex during network I/O + ingestion would
        // needlessly block every HTTP handler for 30-120s.
        let (engine, curiosity_cfg) = {
            let agent = self.agent.lock().unwrap();
            (agent.engine.clone(), agent.curiosity_config.clone())
        };
        // Lock is released here — HTTP handlers can proceed.

        let config = super::continuous_learning::ContinuousLearningConfig::default();
        match super::continuous_learning::run_continuous_learning(
            &engine,
            &curiosity_cfg,
            &config,
        ) {
            Ok(result) => {
                // Re-acquire lock briefly just for the status message.
                if let Ok(agent) = self.agent.lock() {
                    agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:learning] {} targets, {} resources, {} ingested, Dreyfus: {}",
                        result.targets_found,
                        result.resources_discovered,
                        result.concepts_ingested,
                        result.dreyfus_level,
                    )));
                }
                tracing::info!(
                    targets = result.targets_found,
                    resources = result.resources_discovered,
                    ingested = result.concepts_ingested,
                    dreyfus = %result.dreyfus_level,
                    "daemon: continuous learning complete",
                );
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.last_learning_at = Some(now_secs());
                }
            }
            Err(e) => {
                tracing::info!(error = %e, "daemon: continuous learning skipped");
            }
        }
    }

    fn run_goal_generation(&mut self) {
        // Phase 1: Brief lock — snapshot inputs + update drives (needs &WorkingMemory).
        let (engine, goals_snapshot, drives_snapshot, gen_config, predicates,
             cycle_count, impasse, reflection, watch_firings, curiosity_config) = {
            let mut guard = self.agent.lock().unwrap();
            // Explicit reborrow so the borrow checker can split field access.
            let agent: &mut Agent = &mut *guard;

            // Update drives while we hold the lock (needs &WorkingMemory which is !Clone).
            let goal_symbols: Vec<crate::symbol::SymbolId> =
                super::goal::active_goals(&agent.goals)
                    .iter()
                    .map(|g| g.symbol_id)
                    .collect();
            agent.drives.update_with_curiosity(
                &agent.engine,
                &agent.working_memory,
                &goal_symbols,
                agent.cycle_count,
                Some(&agent.curiosity_config),
            );

            (
                agent.engine.clone(),
                agent.goals.clone(),
                agent.drives.clone(),
                agent.config.goal_generation.clone(),
                agent.predicates.clone(),
                agent.cycle_count,
                agent.last_impasse.clone(),
                agent.last_reflection.clone(),
                agent.last_watch_firings.clone(),
                agent.curiosity_config.clone(),
            )
        };
        // Lock released — HTTP handlers can proceed.

        // Phase 2: Expensive computation (no lock held).
        let curiosity_proposals = super::curiosity::compute_directed_curiosity(
            &engine,
            &curiosity_config,
        )
        .map(|report| report.proposals)
        .unwrap_or_default();

        let gen_result = match super::goal_generation::generate_goals(
            &engine,
            &goals_snapshot,
            &drives_snapshot,
            &gen_config,
            &predicates,
            cycle_count,
            impasse.as_ref(),
            reflection.as_ref(),
            &watch_firings,
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(error = %e, "daemon: goal generation skipped");
                return;
            }
        };

        // Merge curiosity proposals.
        let mut gen_result = gen_result;
        for cp in curiosity_proposals {
            if gen_result.activated.len() < gen_config.max_activations {
                gen_result.activated.push(cp);
            } else if gen_result.dormant.len() < gen_config.max_dormant {
                gen_result.dormant.push(cp);
            }
        }

        // Phase 3: Brief lock — apply results (push goals, store provenance).
        let mut agent = self.agent.lock().unwrap();

        for proposal in &gen_result.activated {
            match super::goal::create_goal(
                &agent.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &agent.predicates,
            ) {
                Ok(mut g) => {
                    g.source = Some(proposal.source.clone());
                    let (drive_name, drive_strength) =
                        super::goal_generation::provenance_from_source(&proposal.source);
                    let mut prov = crate::provenance::ProvenanceRecord::new(
                        g.symbol_id,
                        crate::provenance::DerivationKind::AutonomousGoalGeneration {
                            drive: drive_name,
                            strength: drive_strength,
                        },
                    )
                    .with_confidence(proposal.feasibility);
                    if let Err(e) = agent.engine.store_provenance(&mut prov) {
                        tracing::debug!("failed to store goal provenance: {e}");
                    }
                    agent.goals.push(g);
                }
                Err(e) => tracing::debug!("failed to create activated goal: {e}"),
            }
        }

        for proposal in &gen_result.dormant {
            match super::goal::create_goal(
                &agent.engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                &agent.predicates,
            ) {
                Ok(mut g) => {
                    g.source = Some(proposal.source.clone());
                    let _ = super::goal::update_goal_status(
                        &agent.engine,
                        &mut g,
                        super::goal::GoalStatus::Dormant,
                        &agent.predicates,
                    );
                    agent.goals.push(g);
                }
                Err(e) => tracing::debug!("failed to create dormant goal: {e}"),
            }
        }

        // Clear impasse after it's been consumed.
        agent.last_impasse = None;

        let activated = gen_result.activated.len();
        let dormant = gen_result.dormant.len();
        if activated > 0 || dormant > 0 {
            agent.sink().emit(&AkhMessage::system(format!(
                "[daemon:goals] {activated} activated, {dormant} dormant",
            )));
        }
        tracing::info!(activated, dormant, "daemon: goal generation complete");

        let goal_count = super::goal::active_goals(&agent.goals).len();
        drop(agent); // Release agent lock before status lock.
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.last_goal_gen_at = Some(now_secs());
            st.active_goals = goal_count;
        }
    }

    fn run_sleep_cycle(&mut self) {
        let mut guard = self.agent.lock().unwrap();
        // Explicit reborrow so the borrow checker can see partial field access.
        let agent: &mut Agent = &mut *guard;
        match super::sleep::run_sleep_cycle(
            &agent.engine,
            &agent.working_memory,
            &mut agent.sleep_cycle,
            agent.cycle_count,
        ) {
            Ok(metrics) => {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:sleep] replayed {}, merged {}, pruned {}, dream {}",
                    metrics.episodes_replayed,
                    metrics.duplicates_merged,
                    metrics.orphans_pruned,
                    metrics.dream_connections_found,
                )));
                tracing::info!(?metrics, "daemon: sleep cycle complete");
                drop(guard); // Release agent lock before status lock.
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.last_sleep_at = Some(now_secs());
                }
            }
            Err(e) => tracing::debug!(error = %e, "daemon: sleep cycle skipped"),
        }
    }

    fn run_trigger_evaluation(&mut self) {
        use super::trigger::{self as trigger_mod, TriggerStore};

        let mut guard = self.agent.lock().unwrap();
        // Explicit reborrow for partial field access.
        let agent: &mut Agent = &mut *guard;

        let triggers = TriggerStore::new(&agent.engine).list();
        if triggers.is_empty() {
            return;
        }

        let trigger_count = triggers.len();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Collect which triggers should fire (immutable access).
        let to_fire: Vec<usize> = triggers
            .iter()
            .enumerate()
            .filter(|(_, t)| trigger_mod::should_fire(t, agent, now))
            .map(|(i, _)| i)
            .collect();

        // Execute and update last-fired (mutable access).
        for idx in to_fire {
            let trigger = &triggers[idx];
            match trigger_mod::execute_trigger(trigger, agent) {
                Ok(msg) => tracing::info!("{msg}"),
                Err(e) => tracing::warn!(error = %e, "trigger execution failed"),
            }
            TriggerStore::new(&agent.engine).update_last_fired(&trigger.id, now);
        }

        // Sync trigger count in shared status.
        drop(guard); // Release agent lock before status lock.
        if let Some(ref status) = self.status
            && let Ok(mut st) = status.try_lock()
        {
            st.trigger_count = trigger_count;
        }
    }

    fn run_idle_cycle(&mut self) {
        use super::goal;

        let mut agent = self.agent.lock().unwrap();
        if goal::active_goals(agent.goals()).is_empty() {
            return;
        }

        match agent.run_cycle() {
            Ok(result) => {
                self.total_cycles += 1;
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:cycle] tool={}, success={}",
                    result.decision.chosen_tool, result.action_result.tool_output.success,
                )));
                tracing::info!(
                    tool = %result.decision.chosen_tool,
                    cycle = self.total_cycles,
                    "daemon: OODA cycle complete",
                );
                drop(agent); // Release agent lock before status lock.
                // Sync status for real-time HTTP monitoring.
                if let Some(ref status) = self.status
                    && let Ok(mut st) = status.try_lock()
                {
                    st.total_cycles = self.total_cycles;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: OODA cycle failed");
            }
        }
    }

    /// Run autonomous knowledge extraction on recently-fetched content.
    ///
    /// The akh decides what to extract based on its own curiosity targets:
    /// 1. Query gap analysis for ZPD-proximal concepts with knowledge gaps
    /// 2. For each target, gather any text associated with it (doc:mentions triples)
    /// 3. Run the TripleExtractor (local LLM → external API → regex fallback)
    /// 4. Store extracted triples with LlmTripleExtraction provenance
    ///
    /// Drops the agent lock during extraction (expensive I/O) to avoid blocking
    /// HTTP handlers.
    fn run_knowledge_extraction(&self) {
        use crate::extraction::{ExtractionConfig, TripleExtractor};

        // Extract engine ref and drop agent lock.
        let engine = {
            let agent = self.agent.lock().unwrap();
            agent.engine.clone()
        };

        // Load extraction config (use defaults for now — config.toml integration
        // reads from the file at daemon startup and passes through DaemonConfig).
        let config = ExtractionConfig::default();
        if !config.enabled {
            return;
        }

        let extractor = TripleExtractor::new(config);

        // Find recently-added content paragraphs that reference gap entities.
        // Strategy: query for `doc:mentions` triples pointing to entities that
        // have few outgoing relations (knowledge gaps).
        let doc_mentions = engine.resolve_symbol("doc:mentions").ok();
        let Some(mentions_pred) = doc_mentions else {
            return; // No content has been ingested yet.
        };

        let mention_triples = engine
            .knowledge_graph()
            .triples_for_predicate(mentions_pred);

        if mention_triples.is_empty() {
            return;
        }

        // Collect paragraph entities that mention under-connected concepts.
        let mut extraction_targets: Vec<(crate::symbol::SymbolId, String)> = Vec::new();
        for (para_id, entity_id) in &mention_triples {
            // Check if the mentioned entity has few outgoing relations (gap signal).
            let outgoing = engine.triples_from(*entity_id);
            if outgoing.len() < 3 {
                let para_label = engine.resolve_label(*para_id);
                // Only process paragraph-like entities (from content ingest).
                if para_label.starts_with("para:") {
                    extraction_targets.push((*para_id, para_label));
                }
            }
        }

        // Limit to avoid excessive extraction in a single cycle.
        extraction_targets.truncate(10);

        if extraction_targets.is_empty() {
            return;
        }

        let mut total_extracted = 0usize;

        for (para_id, para_label) in &extraction_targets {
            // Get the paragraph text from the KG (stored as outgoing triples).
            let para_triples = engine.triples_from(*para_id);
            // Build text from the paragraph's doc:content triple if it exists,
            // or reconstruct from mentioned entity labels.
            let text: String = para_triples
                .iter()
                .filter(|t| {
                    let pred = engine.resolve_label(t.predicate);
                    pred == "doc:mentions"
                })
                .map(|t| engine.resolve_label(t.object))
                .collect::<Vec<_>>()
                .join(", ");

            if text.len() < 20 {
                continue; // Too short to extract from.
            }

            match extractor.extract_and_store(&text, &engine) {
                Ok(count) => total_extracted += count,
                Err(e) => {
                    tracing::debug!(para = %para_label, error = %e, "extraction failed");
                }
            }
        }

        if total_extracted > 0 {
            let _ = engine.persist();
            if let Ok(agent) = self.agent.lock() {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:extraction] {total_extracted} triples extracted from {} paragraphs",
                    extraction_targets.len(),
                )));
            }
            tracing::info!(
                triples = total_extracted,
                paragraphs = extraction_targets.len(),
                "daemon: knowledge extraction complete"
            );
        }
    }

    fn run_bridge_training(&self) {
        use crate::training::{BridgeTrainingSession, TrainerConfig};
        use crate::vsa::neural_bridge::NeuralVsaBridge;
        use crate::vsa::Dimension;

        let engine = {
            let agent = self.agent.lock().unwrap();
            agent.engine.clone()
        };

        // Check if we have enough grounded symbols to train.
        let symbol_count = engine.item_memory().all_symbols().len();
        if symbol_count < 50 {
            tracing::debug!(
                symbols = symbol_count,
                "daemon: bridge training skipped — insufficient grounded symbols (need ≥50)"
            );
            return;
        }

        // Create or reuse a bridge with the appropriate hidden dim.
        // For now, use a small proxy dim since we don't have real hidden states.
        let hidden_dim = 128;
        let mut bridge = NeuralVsaBridge::new(hidden_dim, Dimension::DEFAULT);

        let config = TrainerConfig {
            max_steps: 100,
            ..Default::default()
        };
        let session = BridgeTrainingSession::new(config);
        let result = session.train_from_grounded_symbols(&mut bridge, &engine);

        if result.improved {
            if let Ok(agent) = self.agent.lock() {
                agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:bridge-training] {} steps, match rate {:.1}% → {:.1}% ({} symbols)",
                    result.steps,
                    result.initial_match_rate * 100.0,
                    result.final_match_rate * 100.0,
                    result.pairs_used,
                )));
            }
        }
        tracing::info!(
            steps = result.steps,
            pairs = result.pairs_used,
            initial = result.initial_match_rate,
            final_rate = result.final_match_rate,
            improved = result.improved,
            "daemon: bridge training complete"
        );
    }

    fn persist(&self) {
        let agent = self.agent.lock().unwrap();
        if let Err(e) = agent.persist_session() {
            tracing::warn!(error = %e, "daemon: session persist failed");
        } else {
            tracing::debug!("daemon: session persisted");
            drop(agent); // Release agent lock before status lock.
            if let Some(ref status) = self.status
                && let Ok(mut st) = status.try_lock()
            {
                st.last_persist_at = Some(now_secs());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_reasonable_intervals() {
        let config = DaemonConfig::default();
        assert_eq!(config.equivalence_interval, Duration::from_secs(300));
        assert_eq!(config.reflection_interval, Duration::from_secs(180));
        assert_eq!(config.consolidation_interval, Duration::from_secs(60));
        assert_eq!(config.schema_discovery_interval, Duration::from_secs(1800));
        assert_eq!(config.rule_inference_interval, Duration::from_secs(600));
        assert_eq!(config.gap_analysis_interval, Duration::from_secs(900));
        assert_eq!(config.persist_interval, Duration::from_secs(60));
        assert_eq!(config.idle_cycle_interval, Duration::from_secs(30));
        assert_eq!(config.continuous_learning_interval, Duration::from_secs(7200));
        assert_eq!(config.goal_generation_interval, Duration::from_secs(300));
        assert_eq!(config.sleep_cycle_interval, Duration::from_secs(3600));
        assert_eq!(config.trigger_evaluation_interval, Duration::from_secs(15));
        assert_eq!(
            config.knowledge_extraction_interval,
            Duration::from_secs(2700)
        );
        assert_eq!(config.max_cycles, 0);
    }
}
