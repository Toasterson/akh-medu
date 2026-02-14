//! Idle-time background learning for the TUI.
//!
//! The `IdleScheduler` runs lightweight background tasks during crossterm poll
//! timeouts (100ms windows). Each `tick()` runs at most one task — the most
//! overdue — to avoid blocking the TUI event loop.

use std::time::{Duration, Instant};

use super::Agent;

/// Result of a single idle background task.
pub struct IdleTaskResult {
    /// Task name (e.g., "equivalence", "reflection").
    pub task: &'static str,
    /// Human-readable summary of what happened.
    pub summary: String,
}

/// Scheduler that tracks when each background task last ran and selects the
/// most overdue task on each `tick()`.
pub struct IdleScheduler {
    last_equivalence: Instant,
    last_reflection: Instant,
    last_consolidation_check: Instant,
    last_rules: Instant,
    equivalence_interval: Duration,
    reflection_interval: Duration,
    consolidation_interval: Duration,
    rules_interval: Duration,
}

impl Default for IdleScheduler {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            last_equivalence: now,
            last_reflection: now,
            last_consolidation_check: now,
            last_rules: now,
            // Longer intervals than daemon — TUI is interactive, don't steal CPU.
            equivalence_interval: Duration::from_secs(600),  // 10 min
            reflection_interval: Duration::from_secs(300),   // 5 min
            consolidation_interval: Duration::from_secs(120), // 2 min
            rules_interval: Duration::from_secs(900),        // 15 min
        }
    }
}

impl IdleScheduler {
    /// Create a scheduler with custom intervals.
    pub fn with_intervals(
        equivalence: Duration,
        reflection: Duration,
        consolidation: Duration,
        rules: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            last_equivalence: now,
            last_reflection: now,
            last_consolidation_check: now,
            last_rules: now,
            equivalence_interval: equivalence,
            reflection_interval: reflection,
            consolidation_interval: consolidation,
            rules_interval: rules,
        }
    }

    /// Run the most overdue background task, if any is due.
    ///
    /// Returns `Some(result)` if a task ran, `None` if nothing was due.
    /// Each task is wrapped in error handling — failures are reported
    /// in the result summary rather than propagated.
    pub fn tick(&mut self, agent: &mut Agent) -> Option<IdleTaskResult> {
        let now = Instant::now();

        // Compute how overdue each task is (None if not yet due).
        let overdue = [
            (
                "consolidation",
                now.checked_duration_since(self.last_consolidation_check)
                    .and_then(|d| d.checked_sub(self.consolidation_interval)),
            ),
            (
                "reflection",
                now.checked_duration_since(self.last_reflection)
                    .and_then(|d| d.checked_sub(self.reflection_interval)),
            ),
            (
                "equivalence",
                now.checked_duration_since(self.last_equivalence)
                    .and_then(|d| d.checked_sub(self.equivalence_interval)),
            ),
            (
                "rules",
                now.checked_duration_since(self.last_rules)
                    .and_then(|d| d.checked_sub(self.rules_interval)),
            ),
        ];

        // Pick the most overdue task.
        let most_overdue = overdue
            .iter()
            .filter_map(|(name, over)| over.map(|d| (*name, d)))
            .max_by_key(|(_, d)| *d);

        let (task_name, _) = most_overdue?;

        match task_name {
            "consolidation" => {
                self.last_consolidation_check = now;
                let wm_len = agent.working_memory().len();
                let threshold = agent.config.consolidation.auto_consolidate_at;
                if wm_len >= threshold {
                    match agent.consolidate() {
                        Ok(result) => Some(IdleTaskResult {
                            task: "consolidation",
                            summary: format!(
                                "consolidated {} entries, {} episodes created",
                                result.entries_evicted, result.episodes_created.len(),
                            ),
                        }),
                        Err(e) => Some(IdleTaskResult {
                            task: "consolidation",
                            summary: format!("failed: {e}"),
                        }),
                    }
                } else {
                    Some(IdleTaskResult {
                        task: "consolidation",
                        summary: format!("WM pressure low ({wm_len}/{threshold}), skipped"),
                    })
                }
            }
            "reflection" => {
                self.last_reflection = now;
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
                        Some(IdleTaskResult {
                            task: "reflection",
                            summary: format!(
                                "{adj_count} insights, {applied} adjustments applied",
                            ),
                        })
                    }
                    Err(e) => Some(IdleTaskResult {
                        task: "reflection",
                        summary: format!("failed: {e}"),
                    }),
                }
            }
            "equivalence" => {
                self.last_equivalence = now;
                match agent.engine().learn_equivalences() {
                    Ok(count) => Some(IdleTaskResult {
                        task: "equivalence",
                        summary: format!("{count} new equivalences discovered"),
                    }),
                    Err(e) => Some(IdleTaskResult {
                        task: "equivalence",
                        summary: format!("failed: {e}"),
                    }),
                }
            }
            "rules" => {
                self.last_rules = now;
                let config = crate::autonomous::rule_engine::RuleEngineConfig::default();
                match agent.engine().run_rules(config) {
                    Ok(result) => Some(IdleTaskResult {
                        task: "rules",
                        summary: format!(
                            "{} new triples from {} iterations",
                            result.derived.len(), result.iterations,
                        ),
                    }),
                    Err(e) => Some(IdleTaskResult {
                        task: "rules",
                        summary: format!("failed: {e}"),
                    }),
                }
            }
            _ => None,
        }
    }

    /// Check if any task is overdue without running it.
    pub fn has_pending_task(&self) -> bool {
        let now = Instant::now();
        now.duration_since(self.last_consolidation_check) >= self.consolidation_interval
            || now.duration_since(self.last_reflection) >= self.reflection_interval
            || now.duration_since(self.last_equivalence) >= self.equivalence_interval
            || now.duration_since(self.last_rules) >= self.rules_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_intervals_are_reasonable() {
        let sched = IdleScheduler::default();
        assert!(sched.equivalence_interval >= Duration::from_secs(60));
        assert!(sched.reflection_interval >= Duration::from_secs(60));
        assert!(sched.consolidation_interval >= Duration::from_secs(30));
        assert!(sched.rules_interval >= Duration::from_secs(60));
    }

    #[test]
    fn nothing_due_initially() {
        let sched = IdleScheduler::default();
        // Nothing should be overdue right after creation.
        assert!(!sched.has_pending_task());
    }

    #[test]
    fn overdue_detection_after_interval() {
        let mut sched = IdleScheduler::default();
        // Simulate that consolidation happened long ago.
        sched.last_consolidation_check = Instant::now() - Duration::from_secs(300);
        assert!(sched.has_pending_task());
    }
}
