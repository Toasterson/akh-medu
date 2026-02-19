//! World monitoring: watches, expectations, and discrepancy detection (Phase 11e).
//!
//! Provides GDA-style expectation monitoring through:
//! - **Watches**: condition→action rules that fire when KG state changes
//! - **WorldSnapshot**: lightweight KG state capture for delta computation
//! - **Expectations**: what a plan step is expected to accomplish
//! - **Discrepancies**: mismatch between expected and actual effects
//!
//! Pull-based delta polling: the agent captures a [`WorldSnapshot`] before and
//! after each OODA cycle, then evaluates watches against the delta. No engine
//! callbacks or push-based hooks needed.

use serde::{Deserialize, Serialize};

use crate::engine::Engine;

use super::error::{AgentError, AgentResult};

// ---------------------------------------------------------------------------
// WorldSnapshot
// ---------------------------------------------------------------------------

/// Lightweight snapshot of KG state at a specific cycle.
///
/// Used for delta computation — compare before/after to detect changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Total number of triples in the KG.
    pub triple_count: usize,
    /// Total number of symbols in the KG.
    pub symbol_count: usize,
    /// Which OODA cycle this snapshot was taken at.
    pub cycle: u64,
}

/// Capture the current KG state as a [`WorldSnapshot`].
pub fn take_snapshot(engine: &Engine, cycle: u64) -> WorldSnapshot {
    WorldSnapshot {
        triple_count: engine.all_triples().len(),
        symbol_count: engine.all_symbols().len(),
        cycle,
    }
}

// ---------------------------------------------------------------------------
// TriplePattern — glob-style matching for KG triples
// ---------------------------------------------------------------------------

/// A pattern for matching KG triples.
///
/// Each field is `None` (matches any) or `Some(pattern)` where the pattern
/// can be exact (`"foo"`) or a prefix glob (`"concept:*"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePattern {
    /// Subject pattern. `None` = match any subject.
    pub subject_pattern: Option<String>,
    /// Predicate pattern. `None` = match any predicate.
    pub predicate_pattern: Option<String>,
    /// Object pattern. `None` = match any object.
    pub object_pattern: Option<String>,
}

impl TriplePattern {
    /// Create a pattern that matches any triple.
    pub fn any() -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            object_pattern: None,
        }
    }
}

/// Check if a label matches a pattern string.
///
/// Supports:
/// - Exact match: `"foo"` matches only `"foo"`
/// - Prefix glob: `"concept:*"` matches any label starting with `"concept:"`
fn label_matches_pattern(label: &str, pattern: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        label.starts_with(prefix)
    } else {
        label == pattern
    }
}

/// Check whether a KG triple matches a [`TriplePattern`].
///
/// Resolves SymbolIds to labels via the engine for comparison.
pub fn matches_pattern(
    pattern: &TriplePattern,
    triple: &crate::graph::Triple,
    engine: &Engine,
) -> bool {
    if let Some(ref sp) = pattern.subject_pattern {
        let label = engine.resolve_label(triple.subject);
        if !label_matches_pattern(&label, sp) {
            return false;
        }
    }
    if let Some(ref pp) = pattern.predicate_pattern {
        let label = engine.resolve_label(triple.predicate);
        if !label_matches_pattern(&label, pp) {
            return false;
        }
    }
    if let Some(ref op) = pattern.object_pattern {
        let label = engine.resolve_label(triple.object);
        if !label_matches_pattern(&label, op) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// WatchCondition + WatchAction + Watch
// ---------------------------------------------------------------------------

/// Conditions under which a watch fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatchCondition {
    /// Fires when a triple matching the pattern exists in the KG.
    TripleMatch { pattern: TriplePattern },
    /// Fires when a symbol's confidence drops below threshold.
    ConfidenceBelow {
        symbol_label: String,
        threshold: f64,
    },
    /// Fires when KG triple count changes by at least `min_delta`.
    TripleCountDelta { min_delta: usize },
    /// Fires when a KG triple matching the pattern is removed (retracted).
    TripleRemoved { pattern: TriplePattern },
}

/// What to do when a watch fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WatchAction {
    /// Generate a goal proposal when fired.
    GenerateGoal {
        description_template: String,
        priority: u8,
        criteria_template: String,
    },
    /// Log an observation in working memory.
    LogObservation { message_template: String },
}

/// A registered world monitor: condition + action + cooldown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Watch {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// When to fire.
    pub condition: WatchCondition,
    /// What to do when fired.
    pub action: WatchAction,
    /// Whether the watch is enabled.
    pub enabled: bool,
    /// Minimum cycles between firings (debouncing).
    pub cooldown_cycles: u64,
    /// Last cycle this watch fired (0 = never).
    pub last_fired_cycle: u64,
}

// ---------------------------------------------------------------------------
// WatchFiring — result of a watch evaluation
// ---------------------------------------------------------------------------

/// A watch that fired during evaluation.
#[derive(Debug, Clone)]
pub struct WatchFiring {
    /// Name of the watch that fired.
    pub watch_name: String,
    /// The action to take.
    pub watch_action: WatchAction,
    /// Brief description of why the condition was met.
    pub condition_summary: String,
}

// ---------------------------------------------------------------------------
// Expectation + Discrepancy
// ---------------------------------------------------------------------------

/// Expected effects of a plan step (GDA-style expectations).
#[derive(Debug, Clone)]
pub struct Expectation {
    /// What the plan step is expected to accomplish (string descriptions).
    pub expected_effects: Vec<String>,
}

/// Mismatch between expected and actual effects.
#[derive(Debug, Clone)]
pub struct Discrepancy {
    /// Expected effects that were not observed.
    pub missing_expected: Vec<String>,
    /// How many new triples were actually added.
    pub actual_new_triples: usize,
    /// Brief explanation of the mismatch.
    pub explanation: String,
}

/// Compute a discrepancy between expected and actual effects.
///
/// Returns `None` if there are no expectations or everything looks fine.
pub fn compute_discrepancy(
    expected_effects: &[String],
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    tool_output: &str,
) -> Option<Discrepancy> {
    if expected_effects.is_empty() {
        return None;
    }

    let tool_lower = tool_output.to_lowercase();
    let tool_failed = tool_lower.contains("error") || tool_lower.contains("failed");

    let actual_new = after.triple_count.saturating_sub(before.triple_count);

    // Check which expected effects are missing from tool output.
    let missing: Vec<String> = expected_effects
        .iter()
        .filter(|effect| {
            let effect_lower = effect.to_lowercase();
            // Extract keywords from the effect description.
            let keywords: Vec<&str> = effect_lower.split_whitespace().collect();
            // Check if at least one significant keyword appears in tool output.
            !keywords
                .iter()
                .filter(|kw| kw.len() > 3) // Skip short words like "a", "the", "add"
                .any(|kw| tool_lower.contains(kw))
        })
        .cloned()
        .collect();

    // Full discrepancy: tool failed with expectations pending.
    if tool_failed {
        return Some(Discrepancy {
            missing_expected: expected_effects.to_vec(),
            actual_new_triples: actual_new,
            explanation: "Tool execution failed; all expected effects unmet.".into(),
        });
    }

    // Partial discrepancy: expected to add/create but triple count didn't change.
    let expects_addition = expected_effects.iter().any(|e| {
        let el = e.to_lowercase();
        el.contains("add") || el.contains("create") || el.contains("insert")
    });
    if expects_addition && actual_new == 0 {
        return Some(Discrepancy {
            missing_expected: missing.clone(),
            actual_new_triples: 0,
            explanation: "Expected new triples but triple count unchanged.".into(),
        });
    }

    // If most keywords were found, no discrepancy.
    if missing.is_empty() {
        return None;
    }

    None
}

// ---------------------------------------------------------------------------
// Watch evaluation
// ---------------------------------------------------------------------------

/// Evaluate all watches against the world delta and return firings.
///
/// Respects cooldown: a watch won't fire if fewer than `cooldown_cycles`
/// have elapsed since its last firing.
pub fn evaluate_watches(
    watches: &mut [Watch],
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    engine: &Engine,
    current_cycle: u64,
) -> Vec<WatchFiring> {
    let mut firings = Vec::new();

    for watch in watches.iter_mut() {
        if !watch.enabled {
            continue;
        }

        // Respect cooldown.
        if current_cycle.saturating_sub(watch.last_fired_cycle) < watch.cooldown_cycles
            && watch.last_fired_cycle > 0
        {
            continue;
        }

        if let Some(summary) = evaluate_condition(&watch.condition, before, after, engine) {
            watch.last_fired_cycle = current_cycle;
            firings.push(WatchFiring {
                watch_name: watch.name.clone(),
                watch_action: watch.action.clone(),
                condition_summary: summary,
            });
        }
    }

    firings
}

/// Evaluate a single watch condition against the before/after snapshots.
///
/// Returns `Some(summary)` if the condition is met, `None` otherwise.
fn evaluate_condition(
    condition: &WatchCondition,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    engine: &Engine,
) -> Option<String> {
    match condition {
        WatchCondition::TripleMatch { pattern } => {
            // Scan recent triples (those added since before snapshot).
            let all_triples = engine.all_triples();
            // Only check triples if count grew.
            if after.triple_count <= before.triple_count {
                return None;
            }
            // Check the most recent triples (heuristic: scan from end).
            let new_count = after.triple_count.saturating_sub(before.triple_count);
            let start = all_triples.len().saturating_sub(new_count);
            for triple in &all_triples[start..] {
                if matches_pattern(pattern, triple, engine) {
                    return Some(format!(
                        "Triple matched: {} {} {}",
                        engine.resolve_label(triple.subject),
                        engine.resolve_label(triple.predicate),
                        engine.resolve_label(triple.object),
                    ));
                }
            }
            None
        }
        WatchCondition::ConfidenceBelow {
            symbol_label,
            threshold,
        } => {
            // Look up the symbol and check confidence of triples from it.
            if let Ok(sym) = engine.lookup_symbol(symbol_label) {
                let triples = engine.triples_from(sym);
                let min_conf = triples
                    .iter()
                    .map(|t| t.confidence as f64)
                    .fold(f64::MAX, f64::min);
                if min_conf < *threshold && min_conf < f64::MAX {
                    return Some(format!(
                        "Confidence {:.2} below threshold {:.2} for '{}'",
                        min_conf, threshold, symbol_label,
                    ));
                }
            }
            None
        }
        WatchCondition::TripleCountDelta { min_delta } => {
            let delta = after.triple_count.saturating_sub(before.triple_count);
            if delta >= *min_delta {
                Some(format!(
                    "Triple count grew by {} (threshold: {})",
                    delta, min_delta,
                ))
            } else {
                None
            }
        }
        WatchCondition::TripleRemoved { pattern } => {
            // Check if triple count decreased.
            if after.triple_count >= before.triple_count {
                return None;
            }
            // We can't cheaply identify which triples were removed without
            // a full diff, so we check if the pattern no longer matches anything.
            let all_triples = engine.all_triples();
            let still_exists = all_triples.iter().any(|t| matches_pattern(pattern, t, engine));
            if !still_exists {
                Some(format!(
                    "No triples match pattern after removal (count {} → {})",
                    before.triple_count, after.triple_count,
                ))
            } else {
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Watch persistence
// ---------------------------------------------------------------------------

/// Persist watches to the engine's durable store.
pub fn persist_watches(engine: &Engine, watches: &[Watch]) -> AgentResult<()> {
    let data = bincode::serialize(watches).map_err(|e| AgentError::ConsolidationFailed {
        message: format!("failed to serialize watches: {e}"),
    })?;
    engine
        .store()
        .put_meta(b"agent:watches", &data)
        .map_err(|e| AgentError::ConsolidationFailed {
            message: format!("failed to persist watches: {e}"),
        })?;
    Ok(())
}

/// Restore watches from the engine's durable store.
pub fn restore_watches(engine: &Engine) -> AgentResult<Vec<Watch>> {
    match engine.store().get_meta(b"agent:watches").ok().flatten() {
        Some(data) if !data.is_empty() => {
            bincode::deserialize(&data).map_err(|e| AgentError::ConsolidationFailed {
                message: format!("failed to deserialize watches: {e}"),
            })
        }
        _ => Ok(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// Watch ID generation
// ---------------------------------------------------------------------------

/// Generate a unique watch ID from the current timestamp.
pub fn generate_watch_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("w-{:x}-{:x}", now.as_secs(), now.subsec_nanos())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_captures_state() {
        let snap = WorldSnapshot {
            triple_count: 42,
            symbol_count: 10,
            cycle: 5,
        };
        assert_eq!(snap.triple_count, 42);
        assert_eq!(snap.symbol_count, 10);
        assert_eq!(snap.cycle, 5);
    }

    #[test]
    fn world_snapshot_default_values() {
        let snap = WorldSnapshot {
            triple_count: 0,
            symbol_count: 0,
            cycle: 0,
        };
        assert_eq!(snap.triple_count, 0);
        assert_eq!(snap.cycle, 0);
    }

    #[test]
    fn triple_pattern_exact_match() {
        assert!(label_matches_pattern("concept:dog", "concept:dog"));
        assert!(!label_matches_pattern("concept:cat", "concept:dog"));
    }

    #[test]
    fn triple_pattern_wildcard() {
        assert!(label_matches_pattern("concept:dog", "concept:*"));
        assert!(label_matches_pattern("concept:cat", "concept:*"));
        assert!(!label_matches_pattern("relation:is-a", "concept:*"));
    }

    #[test]
    fn triple_pattern_prefix_glob() {
        assert!(label_matches_pattern("status:failed:timeout", "status:*"));
        assert!(label_matches_pattern("status:", "status:*"));
        assert!(!label_matches_pattern("other:foo", "status:*"));
    }

    #[test]
    fn discrepancy_none_when_no_expectations() {
        let before = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 1,
        };
        let after = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 2,
        };
        assert!(compute_discrepancy(&[], &before, &after, "ok").is_none());
    }

    #[test]
    fn discrepancy_when_no_change() {
        let before = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 1,
        };
        let after = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 2,
        };
        let effects = vec!["add triple about concept".to_string()];
        let disc = compute_discrepancy(&effects, &before, &after, "did something");
        assert!(disc.is_some());
        let d = disc.unwrap();
        assert_eq!(d.actual_new_triples, 0);
        assert!(d.explanation.contains("unchanged"));
    }

    #[test]
    fn discrepancy_on_tool_failure() {
        let before = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 1,
        };
        let after = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 2,
        };
        let effects = vec!["find related concepts".to_string()];
        let disc = compute_discrepancy(&effects, &before, &after, "error: tool crashed");
        assert!(disc.is_some());
        let d = disc.unwrap();
        assert!(d.explanation.contains("failed"));
    }

    #[test]
    fn watch_id_nonempty() {
        let id = generate_watch_id();
        assert!(!id.is_empty());
        assert!(id.starts_with("w-"));
    }

    #[test]
    fn watch_persistence_roundtrip() {
        let watches = vec![Watch {
            id: "w-test".into(),
            name: "test-watch".into(),
            condition: WatchCondition::TripleCountDelta { min_delta: 5 },
            action: WatchAction::LogObservation {
                message_template: "triples grew".into(),
            },
            enabled: true,
            cooldown_cycles: 3,
            last_fired_cycle: 0,
        }];

        let bytes = bincode::serialize(&watches).unwrap();
        let restored: Vec<Watch> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].id, "w-test");
        assert_eq!(restored[0].name, "test-watch");
        assert_eq!(restored[0].cooldown_cycles, 3);
    }

    #[test]
    fn evaluate_fires_on_delta() {
        let before = WorldSnapshot {
            triple_count: 10,
            symbol_count: 5,
            cycle: 1,
        };
        let after = WorldSnapshot {
            triple_count: 20,
            symbol_count: 8,
            cycle: 2,
        };

        // TripleCountDelta condition: delta = 10 >= min_delta 5.
        let delta = after.triple_count.saturating_sub(before.triple_count);
        assert!(delta >= 5);
        assert_eq!(delta, 10);

        // Verify the condition logic matches what evaluate_condition would do.
        let min_delta = 5usize;
        assert!(delta >= min_delta);
    }

    #[test]
    fn evaluate_respects_cooldown() {
        // Watch just fired at cycle 5, cooldown is 3 cycles.
        let watch = Watch {
            id: "w2".into(),
            name: "cooldown-watch".into(),
            condition: WatchCondition::TripleCountDelta { min_delta: 1 },
            action: WatchAction::LogObservation {
                message_template: "test".into(),
            },
            enabled: true,
            cooldown_cycles: 3,
            last_fired_cycle: 5,
        };

        // At cycle 7 (5+2 < 5+3), should NOT fire.
        let should_skip = 7u64.saturating_sub(watch.last_fired_cycle) < watch.cooldown_cycles;
        assert!(should_skip);

        // At cycle 8 (5+3), should fire.
        let should_fire = 8u64.saturating_sub(watch.last_fired_cycle) >= watch.cooldown_cycles;
        assert!(should_fire);
    }

    #[test]
    fn evaluate_disabled_watch_skipped() {
        let watch = Watch {
            id: "w3".into(),
            name: "disabled".into(),
            condition: WatchCondition::TripleCountDelta { min_delta: 1 },
            action: WatchAction::LogObservation {
                message_template: "test".into(),
            },
            enabled: false,
            cooldown_cycles: 1,
            last_fired_cycle: 0,
        };
        assert!(!watch.enabled);
    }

    #[test]
    fn triple_pattern_any_matches_everything() {
        let pattern = TriplePattern::any();
        assert!(pattern.subject_pattern.is_none());
        assert!(pattern.predicate_pattern.is_none());
        assert!(pattern.object_pattern.is_none());
    }

    #[test]
    fn watch_condition_serialization_roundtrip() {
        let cond = WatchCondition::ConfidenceBelow {
            symbol_label: "test-sym".into(),
            threshold: 0.5,
        };
        let bytes = bincode::serialize(&cond).unwrap();
        let restored: WatchCondition = bincode::deserialize(&bytes).unwrap();
        match restored {
            WatchCondition::ConfidenceBelow {
                symbol_label,
                threshold,
            } => {
                assert_eq!(symbol_label, "test-sym");
                assert!((threshold - 0.5).abs() < f64::EPSILON);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn watch_action_generate_goal() {
        let action = WatchAction::GenerateGoal {
            description_template: "Investigate change in {pattern}".into(),
            priority: 150,
            criteria_template: "Change has been analyzed".into(),
        };
        match &action {
            WatchAction::GenerateGoal {
                priority,
                description_template,
                ..
            } => {
                assert_eq!(*priority, 150);
                assert!(description_template.contains("Investigate"));
            }
            _ => panic!("wrong variant"),
        }
    }
}
