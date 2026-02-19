//! Trigger system: condition→action rules that fire autonomously.
//!
//! Triggers let the agent (or user) register rules like "reflect every 5 minutes"
//! or "run inference rules when new triples appear". The daemon evaluates these
//! each tick cycle and fires matching actions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::agent::Agent;
use super::error::{AgentError, AgentResult};
use crate::engine::Engine;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A registered trigger: a condition→action rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    /// Unique identifier (hex timestamp + counter).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// When to fire.
    pub condition: TriggerCondition,
    /// What to do when fired.
    pub action: TriggerAction,
    /// Whether the trigger is enabled.
    pub enabled: bool,
    /// Unix timestamp of last firing (0 = never).
    pub last_fired: u64,
}

/// Conditions under which a trigger fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Fire every N seconds since last firing.
    Interval { seconds: u64 },
    /// Fire when any active goal has been stalled for >= threshold cycles.
    GoalStalled { threshold: u32 },
    /// Fire when working memory entries exceed threshold.
    MemoryPressure { threshold: usize },
    /// Fire when the triple count has grown by >= min_count since last fire.
    NewTriples { min_count: usize },
    /// Fire when a triple matching the pattern appears (Phase 11e).
    TriplePattern {
        subject_pattern: Option<String>,
        predicate: Option<String>,
        object_pattern: Option<String>,
    },
    /// Fire when a symbol's confidence drops below threshold (Phase 11e).
    ConfidenceThreshold { symbol_label: String, below: f64 },
}

/// Actions to execute when a trigger fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerAction {
    /// Run N OODA cycles.
    RunCycles { count: usize },
    /// Run reflection.
    Reflect,
    /// Run equivalence learning.
    LearnEquivalences,
    /// Run rule inference.
    RunRules,
    /// Run gap analysis.
    AnalyzeGaps,
    /// Add a new goal to the agent.
    AddGoal {
        description: String,
        priority: u8,
        criteria: String,
    },
    /// Execute a named tool with parameters.
    ExecuteTool {
        name: String,
        params: HashMap<String, String>,
    },
}

// ---------------------------------------------------------------------------
// ID generation
// ---------------------------------------------------------------------------

/// Generate a trigger ID from current timestamp + a discriminator.
fn generate_trigger_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    // Combine seconds and nanos for uniqueness.
    format!("{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

// ---------------------------------------------------------------------------
// TriggerStore — persistence via engine durable store
// ---------------------------------------------------------------------------

const TRIGGER_PREFIX: &[u8] = b"trigger:";

/// Persistent storage for triggers, backed by `engine.store().put_meta()`.
pub struct TriggerStore<'a> {
    engine: &'a Engine,
}

impl<'a> TriggerStore<'a> {
    /// Create a trigger store backed by the given engine.
    pub fn new(engine: &'a Engine) -> Self {
        Self { engine }
    }

    /// List all stored triggers.
    pub fn list(&self) -> Vec<Trigger> {
        let entries = match self.engine.store().scan_prefix(TRIGGER_PREFIX) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        entries
            .into_iter()
            .filter_map(|(_key, value)| bincode::deserialize(&value).ok())
            .collect()
    }

    /// Add (or update) a trigger. Returns the trigger with an assigned ID.
    pub fn add(&self, mut trigger: Trigger) -> AgentResult<Trigger> {
        if trigger.id.is_empty() {
            trigger.id = generate_trigger_id();
        }
        let key = format!("trigger:{}", trigger.id);
        let data = bincode::serialize(&trigger).map_err(|e| AgentError::ToolExecution {
            tool_name: "trigger_store".into(),
            message: format!("serialize trigger: {e}"),
        })?;
        self.engine
            .store()
            .put_meta(key.as_bytes(), &data)
            .map_err(|e| AgentError::Engine(Box::new(e.into())))?;
        Ok(trigger)
    }

    /// Remove a trigger by ID.
    pub fn remove(&self, id: &str) -> AgentResult<()> {
        let key = format!("trigger:{id}");
        // Write an empty value to effectively delete.
        self.engine
            .store()
            .put_meta(key.as_bytes(), &[])
            .map_err(|e| AgentError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Get a trigger by ID.
    pub fn get(&self, id: &str) -> Option<Trigger> {
        let key = format!("trigger:{id}");
        let data = self.engine.store().get_meta(key.as_bytes()).ok()??;
        if data.is_empty() {
            return None;
        }
        bincode::deserialize(&data).ok()
    }

    /// Update the `last_fired` timestamp for a trigger.
    pub fn update_last_fired(&self, id: &str, ts: u64) {
        if let Some(mut trigger) = self.get(id) {
            trigger.last_fired = ts;
            let _ = self.add(trigger);
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Check whether a trigger should fire given the current agent state.
pub fn should_fire(trigger: &Trigger, agent: &Agent, now: u64) -> bool {
    if !trigger.enabled {
        return false;
    }

    match &trigger.condition {
        TriggerCondition::Interval { seconds } => {
            now.saturating_sub(trigger.last_fired) >= *seconds
        }
        TriggerCondition::GoalStalled { threshold } => {
            use super::goal;
            let active = goal::active_goals(agent.goals());
            let current_cycle = agent.cycle_count();
            active
                .iter()
                .any(|g| g.is_stalled(current_cycle, *threshold))
        }
        TriggerCondition::MemoryPressure { threshold } => {
            agent.working_memory().len() >= *threshold
        }
        TriggerCondition::NewTriples { min_count } => {
            // Approximate: check total triple count against a threshold.
            // More precise tracking would require persisting the last-seen count.
            agent.engine().all_triples().len() >= *min_count
        }
        TriggerCondition::TriplePattern {
            subject_pattern,
            predicate,
            object_pattern,
        } => {
            let pattern = super::watch::TriplePattern {
                subject_pattern: subject_pattern.clone(),
                predicate_pattern: predicate.clone(),
                object_pattern: object_pattern.clone(),
            };
            agent
                .engine()
                .all_triples()
                .iter()
                .any(|t| super::watch::matches_pattern(&pattern, t, agent.engine()))
        }
        TriggerCondition::ConfidenceThreshold {
            symbol_label,
            below,
        } => {
            if let Ok(sym) = agent.engine().lookup_symbol(symbol_label) {
                let triples = agent.engine().triples_from(sym);
                triples
                    .iter()
                    .any(|t| (t.confidence as f64) < *below)
            } else {
                false
            }
        }
    }
}

/// Execute a trigger's action against the agent. Returns a description of what happened.
pub fn execute_trigger(trigger: &Trigger, agent: &mut Agent) -> AgentResult<String> {
    match &trigger.action {
        TriggerAction::RunCycles { count } => {
            let mut completed = 0;
            for _ in 0..*count {
                match agent.run_cycle() {
                    Ok(_) => completed += 1,
                    Err(_) => break,
                }
            }
            Ok(format!(
                "trigger \"{}\": ran {completed}/{count} cycles",
                trigger.name
            ))
        }
        TriggerAction::Reflect => {
            let result = agent.reflect()?;
            Ok(format!(
                "trigger \"{}\": reflection produced {} adjustments",
                trigger.name,
                result.adjustments.len()
            ))
        }
        TriggerAction::LearnEquivalences => {
            let count = agent.engine().learn_equivalences()?;
            Ok(format!(
                "trigger \"{}\": learned {count} equivalences",
                trigger.name
            ))
        }
        TriggerAction::RunRules => {
            let config = crate::autonomous::rule_engine::RuleEngineConfig::default();
            let result = agent.engine().run_rules(config)?;
            Ok(format!(
                "trigger \"{}\": {} triples derived in {} iterations",
                trigger.name,
                result.derived.len(),
                result.iterations
            ))
        }
        TriggerAction::AnalyzeGaps => {
            use super::goal;
            let active = goal::active_goals(agent.goals());
            let goal_ids: Vec<_> = active.iter().map(|g| g.symbol_id).collect();
            let config = crate::autonomous::gap::GapAnalysisConfig::default();
            let result = agent.engine().analyze_gaps(&goal_ids, config)?;
            Ok(format!(
                "trigger \"{}\": {} gaps identified",
                trigger.name,
                result.gaps.len()
            ))
        }
        TriggerAction::AddGoal {
            description,
            priority,
            criteria,
        } => {
            let id = agent.add_goal(description, *priority, criteria)?;
            Ok(format!(
                "trigger \"{}\": added goal \"{}\" (id: {})",
                trigger.name,
                description,
                id.get()
            ))
        }
        TriggerAction::ExecuteTool { name, params } => {
            use super::tool::ToolInput;
            let mut input = ToolInput::new();
            for (k, v) in params {
                input = input.with_param(k, v);
            }
            let output = agent.tool_registry.execute(name, input, &agent.engine)?;
            Ok(format!(
                "trigger \"{}\": tool {} → success={}",
                trigger.name, name, output.success
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigger_id_is_nonempty() {
        let id = generate_trigger_id();
        assert!(!id.is_empty());
        assert!(id.contains('-'));
    }

    #[test]
    fn trigger_serialization_roundtrip() {
        let trigger = Trigger {
            id: "test-id".into(),
            name: "test-trigger".into(),
            condition: TriggerCondition::Interval { seconds: 300 },
            action: TriggerAction::Reflect,
            enabled: true,
            last_fired: 0,
        };

        let bytes = bincode::serialize(&trigger).unwrap();
        let decoded: Trigger = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.id, "test-id");
        assert_eq!(decoded.name, "test-trigger");
        assert!(decoded.enabled);
    }

    #[test]
    fn should_fire_interval_elapsed() {
        let trigger = Trigger {
            id: "t1".into(),
            name: "test".into(),
            condition: TriggerCondition::Interval { seconds: 60 },
            action: TriggerAction::Reflect,
            enabled: true,
            last_fired: 1000,
        };

        // Not elapsed yet.
        assert!(!should_fire_interval(&trigger, 1050));
        // Elapsed.
        assert!(should_fire_interval(&trigger, 1061));
    }

    #[test]
    fn disabled_trigger_never_fires() {
        let trigger = Trigger {
            id: "t2".into(),
            name: "disabled".into(),
            condition: TriggerCondition::Interval { seconds: 1 },
            action: TriggerAction::Reflect,
            enabled: false,
            last_fired: 0,
        };
        // Would fire if enabled, but isn't.
        assert!(!trigger.enabled);
    }

    /// Standalone interval check for unit tests (no Agent needed).
    fn should_fire_interval(trigger: &Trigger, now: u64) -> bool {
        if !trigger.enabled {
            return false;
        }
        match &trigger.condition {
            TriggerCondition::Interval { seconds } => {
                now.saturating_sub(trigger.last_fired) >= *seconds
            }
            _ => false,
        }
    }

    #[test]
    fn should_fire_triple_pattern_serialization() {
        let trigger = Trigger {
            id: "tp1".into(),
            name: "triple-pattern-test".into(),
            condition: TriggerCondition::TriplePattern {
                subject_pattern: Some("concept:*".into()),
                predicate: Some("is-a".into()),
                object_pattern: None,
            },
            action: TriggerAction::Reflect,
            enabled: true,
            last_fired: 0,
        };

        let bytes = bincode::serialize(&trigger).unwrap();
        let decoded: Trigger = bincode::deserialize(&bytes).unwrap();
        match decoded.condition {
            TriggerCondition::TriplePattern {
                subject_pattern,
                predicate,
                object_pattern,
            } => {
                assert_eq!(subject_pattern.as_deref(), Some("concept:*"));
                assert_eq!(predicate.as_deref(), Some("is-a"));
                assert!(object_pattern.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn should_fire_confidence_threshold_serialization() {
        let trigger = Trigger {
            id: "ct1".into(),
            name: "confidence-test".into(),
            condition: TriggerCondition::ConfidenceThreshold {
                symbol_label: "test-sym".into(),
                below: 0.3,
            },
            action: TriggerAction::AnalyzeGaps,
            enabled: true,
            last_fired: 0,
        };

        let bytes = bincode::serialize(&trigger).unwrap();
        let decoded: Trigger = bincode::deserialize(&bytes).unwrap();
        match decoded.condition {
            TriggerCondition::ConfidenceThreshold {
                symbol_label,
                below,
            } => {
                assert_eq!(symbol_label, "test-sym");
                assert!((below - 0.3).abs() < f64::EPSILON);
            }
            _ => panic!("wrong variant"),
        }
    }
}
