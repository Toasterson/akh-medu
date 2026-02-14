//! Background daemon for autonomous learning.
//!
//! `AgentDaemon` runs a tokio event loop that periodically fires background
//! tasks: equivalence learning, reflection, consolidation, schema discovery,
//! rule inference, gap analysis, and idle OODA cycles. The agent itself stays
//! synchronous — tokio only drives scheduling and signal handling.

use std::time::Duration;

use super::Agent;
use super::error::AgentResult;
use crate::message::AkhMessage;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Intervals for each daemon background task.
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
            max_cycles: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Daemon
// ---------------------------------------------------------------------------

/// Long-running daemon that schedules background learning tasks.
pub struct AgentDaemon {
    agent: Agent,
    config: DaemonConfig,
    total_cycles: usize,
}

impl AgentDaemon {
    /// Create a new daemon wrapping the given agent.
    pub fn new(agent: Agent, config: DaemonConfig) -> Self {
        Self {
            agent,
            config,
            total_cycles: 0,
        }
    }

    /// Run the main daemon loop.
    ///
    /// Schedules all background tasks as tokio intervals and selects on them.
    /// Shuts down cleanly on Ctrl+C, persisting the session before exit.
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

        self.agent.sink().emit(&AkhMessage::system(
            "daemon started — background learning active",
        ));

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
                        self.agent.sink().emit(&AkhMessage::system(format!(
                            "daemon: max cycles ({}) reached, shutting down",
                            self.config.max_cycles,
                        )));
                        break;
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    self.agent.sink().emit(&AkhMessage::system(
                        "daemon: received shutdown signal, persisting session...",
                    ));
                    break;
                }
            }
        }

        // Final persist on shutdown.
        self.persist();
        self.agent
            .sink()
            .emit(&AkhMessage::system("daemon stopped"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Background task methods
    // -----------------------------------------------------------------------

    fn run_equivalence_learning(&mut self) {
        match self.agent.engine().learn_equivalences() {
            Ok(count) => {
                if count > 0 {
                    self.agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:equivalence] {count} new equivalences discovered",
                    )));
                }
                tracing::info!(count, "daemon: equivalence learning complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: equivalence learning failed");
            }
        }
    }

    fn run_reflection(&mut self) {
        match self.agent.reflect() {
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
                let applied = self.agent.apply_adjustments(&safe).unwrap_or(0);
                if adj_count > 0 {
                    self.agent.sink().emit(&AkhMessage::system(format!(
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
        let wm_len = self.agent.working_memory().len();
        let threshold = self.agent.config.consolidation.auto_consolidate_at;
        if wm_len < threshold {
            return;
        }

        match self.agent.consolidate() {
            Ok(result) => {
                self.agent.sink().emit(&AkhMessage::system(format!(
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
        let config = crate::autonomous::schema::SchemaDiscoveryConfig::default();
        match self.agent.engine().discover_schema(config) {
            Ok(result) => {
                let types = result.types.len();
                let hierarchies = result.relation_hierarchies.len();
                if types > 0 || hierarchies > 0 {
                    self.agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:schema] discovered {types} types, {hierarchies} relation hierarchies",
                    )));
                }
                tracing::info!(types, hierarchies, "daemon: schema discovery complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: schema discovery failed");
            }
        }
    }

    fn run_rule_inference(&self) {
        let config = crate::autonomous::rule_engine::RuleEngineConfig::default();
        match self.agent.engine().run_rules(config) {
            Ok(result) => {
                let derived_count = result.derived.len();
                if derived_count > 0 {
                    self.agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:rules] {} new triples from {} iterations",
                        derived_count, result.iterations,
                    )));
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

        let active = goal::active_goals(self.agent.goals());
        if active.is_empty() {
            return;
        }

        let goal_ids: Vec<_> = active.iter().map(|g| g.symbol_id).collect();
        let config = crate::autonomous::gap::GapAnalysisConfig::default();
        match self.agent.engine().analyze_gaps(&goal_ids, config) {
            Ok(result) => {
                let gaps = result.gaps.len();
                if gaps > 0 {
                    self.agent.sink().emit(&AkhMessage::system(format!(
                        "[daemon:gaps] {gaps} knowledge gaps identified around active goals",
                    )));
                }
                tracing::info!(gaps, "daemon: gap analysis complete");
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: gap analysis failed");
            }
        }
    }

    fn run_idle_cycle(&mut self) {
        use super::goal;

        if goal::active_goals(self.agent.goals()).is_empty() {
            return;
        }

        match self.agent.run_cycle() {
            Ok(result) => {
                self.total_cycles += 1;
                self.agent.sink().emit(&AkhMessage::system(format!(
                    "[daemon:cycle] tool={}, success={}",
                    result.decision.chosen_tool, result.action_result.tool_output.success,
                )));
                tracing::info!(
                    tool = %result.decision.chosen_tool,
                    cycle = self.total_cycles,
                    "daemon: OODA cycle complete",
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "daemon: OODA cycle failed");
            }
        }
    }

    fn persist(&self) {
        if let Err(e) = self.agent.persist_session() {
            tracing::warn!(error = %e, "daemon: session persist failed");
        } else {
            tracing::debug!("daemon: session persisted");
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
        assert_eq!(config.max_cycles, 0);
    }
}
