//! Skill quality metrics — Phase 30e.
//!
//! Tracks per-skill effectiveness so the agent can:
//! - Surface stale/ineffective skills to the user
//! - Suggest deactivation when a skill is unused or underperforming
//! - Prioritise refinements for the most frequently failing skills
//!
//! Metrics are kept in memory and can be persisted as KG triples with
//! `skill:quality_*` predicates.
//!
//! # Staleness detection
//!
//! A skill is considered *stale* when it hasn't been used for `stale_days`
//! days AND has a triple coverage below `min_coverage`.  Stale skills are
//! candidates for deactivation to free the memory budget.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// FailureMode
// ─────────────────────────────────────────────────────────────────────────────

/// A recurring failure pattern observed during skill execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMode {
    /// The action that failed.
    pub action: String,
    /// Category of failure.
    pub failure_type: String,
    /// How many times this failure has been recorded.
    pub count: u32,
    /// Unix timestamp of the most recent occurrence.
    pub last_occurrence: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// SkillQuality
// ─────────────────────────────────────────────────────────────────────────────

/// Per-skill quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillQuality {
    /// Skill identifier.
    pub skill_id: String,

    /// Plans completed successfully / plans attempted.
    pub success_rate: f32,
    /// Total plan executions (completed + failed).
    pub total_executions: u32,
    /// Plans that completed successfully.
    pub successful_executions: u32,

    /// Recurring failure patterns.
    pub failure_modes: Vec<FailureMode>,

    /// Per-rule hit rates: rule name → fraction of inferences it participated in.
    pub rule_hit_rates: HashMap<String, f32>,
    /// Total inference passes across all rules.
    pub total_inference_passes: u64,

    /// Fraction of the skill's triples that have been queried at least once.
    pub triple_coverage: f32,
    /// Number of distinct triples queried.
    pub triples_queried: usize,
    /// Total triples the skill declared.
    pub triples_total: usize,

    /// Unix timestamp of the last successful execution (None if never used).
    pub last_used: Option<u64>,

    /// How many refinement suggestions are currently pending.
    pub refinements_pending: u32,
    /// How many refinements have been applied to this skill.
    pub refinements_applied: u32,
}

impl SkillQuality {
    /// Create a fresh quality record for a skill.
    pub fn new(skill_id: impl Into<String>, triples_total: usize) -> Self {
        Self {
            skill_id: skill_id.into(),
            success_rate: 0.0,
            total_executions: 0,
            successful_executions: 0,
            failure_modes: Vec::new(),
            rule_hit_rates: HashMap::new(),
            total_inference_passes: 0,
            triple_coverage: 0.0,
            triples_queried: 0,
            triples_total,
            last_used: None,
            refinements_pending: 0,
            refinements_applied: 0,
        }
    }

    // ─── Mutation helpers ────────────────────────────────────────────────────

    /// Record a successful plan execution.
    pub fn record_success(&mut self) {
        self.total_executions += 1;
        self.successful_executions += 1;
        self.recompute_success_rate();
        let now = now_secs();
        self.last_used = Some(now);
    }

    /// Record a failed plan execution.
    pub fn record_failure(&mut self, action: &str, failure_type: &str) {
        self.total_executions += 1;
        self.recompute_success_rate();

        let now = now_secs();
        if let Some(mode) = self
            .failure_modes
            .iter_mut()
            .find(|m| m.action == action && m.failure_type == failure_type)
        {
            mode.count += 1;
            mode.last_occurrence = now;
        } else {
            self.failure_modes.push(FailureMode {
                action: action.to_string(),
                failure_type: failure_type.to_string(),
                count: 1,
                last_occurrence: now,
            });
        }
    }

    /// Record that a specific inference rule fired during reasoning.
    pub fn record_rule_hit(&mut self, rule_name: &str) {
        self.total_inference_passes += 1;
        let hits = self
            .rule_hit_rates
            .entry(rule_name.to_string())
            .or_insert(0.0);
        // Running average: new_rate = (old_rate * (n-1) + 1) / n
        let n = self.total_inference_passes as f32;
        *hits = (*hits * (n - 1.0) + 1.0) / n;
    }

    /// Record that a rule pass occurred but this rule did NOT fire.
    pub fn record_rule_miss(&mut self, rule_name: &str) {
        self.total_inference_passes += 1;
        let hits = self
            .rule_hit_rates
            .entry(rule_name.to_string())
            .or_insert(0.0);
        let n = self.total_inference_passes as f32;
        // Rule did not fire: hit contribution is 0.
        *hits = *hits * (n - 1.0) / n;
    }

    /// Record that a triple was queried, updating coverage.
    pub fn record_triple_query(&mut self, is_new_triple: bool) {
        if is_new_triple {
            self.triples_queried += 1;
        }
        self.recompute_coverage();
    }

    /// Update the pending refinements count.
    pub fn set_refinements_pending(&mut self, count: u32) {
        self.refinements_pending = count;
    }

    /// Record that a refinement was applied.
    pub fn record_refinement_applied(&mut self) {
        self.refinements_applied += 1;
        if self.refinements_pending > 0 {
            self.refinements_pending -= 1;
        }
    }

    // ─── Derived properties ──────────────────────────────────────────────────

    /// Return the most common failure mode, if any.
    pub fn top_failure_mode(&self) -> Option<&FailureMode> {
        self.failure_modes.iter().max_by_key(|m| m.count)
    }

    /// Return rules whose hit rate is below `threshold` (likely dead rules).
    pub fn low_hit_rate_rules(&self, threshold: f32) -> Vec<(&str, f32)> {
        self.rule_hit_rates
            .iter()
            .filter(|(_, rate)| **rate < threshold)
            .map(|(name, rate)| (name.as_str(), *rate))
            .collect()
    }

    // ─── Internal ────────────────────────────────────────────────────────────

    fn recompute_success_rate(&mut self) {
        if self.total_executions == 0 {
            self.success_rate = 0.0;
        } else {
            self.success_rate = self.successful_executions as f32 / self.total_executions as f32;
        }
    }

    fn recompute_coverage(&mut self) {
        if self.triples_total == 0 {
            self.triple_coverage = 1.0; // vacuously full coverage
        } else {
            self.triple_coverage =
                self.triples_queried.min(self.triples_total) as f32 / self.triples_total as f32;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StalenessConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for staleness detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StalenessConfig {
    /// Number of days without use before a skill is considered stale.
    pub stale_days: u32,
    /// Minimum triple coverage below which a skill is considered under-utilised.
    pub min_coverage: f32,
    /// Minimum success rate below which a skill is flagged for review.
    pub min_success_rate: f32,
}

impl Default for StalenessConfig {
    fn default() -> Self {
        Self {
            stale_days: 30,
            min_coverage: 0.1,
            min_success_rate: 0.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QualityRegistry
// ─────────────────────────────────────────────────────────────────────────────

/// Registry of quality metrics for all known skills.
#[derive(Debug, Default)]
pub struct QualityRegistry {
    records: HashMap<String, SkillQuality>,
    config: StalenessConfig,
}

impl QualityRegistry {
    /// Create an empty registry with default staleness config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with custom staleness config.
    pub fn with_config(config: StalenessConfig) -> Self {
        Self {
            records: HashMap::new(),
            config,
        }
    }

    /// Ensure a quality record exists for `skill_id`.
    pub fn ensure(&mut self, skill_id: &str, triples_total: usize) {
        self.records
            .entry(skill_id.to_string())
            .or_insert_with(|| SkillQuality::new(skill_id, triples_total));
    }

    /// Get an immutable reference to a skill's quality record.
    pub fn get(&self, skill_id: &str) -> Option<&SkillQuality> {
        self.records.get(skill_id)
    }

    /// Get a mutable reference to a skill's quality record.
    pub fn get_mut(&mut self, skill_id: &str) -> Option<&mut SkillQuality> {
        self.records.get_mut(skill_id)
    }

    /// Return skill IDs that are stale under the current config.
    pub fn stale_skills(&self) -> Vec<&str> {
        let threshold_secs = self.config.stale_days as u64 * 86_400;
        let now = now_secs();

        self.records
            .values()
            .filter(|q| {
                let stale_by_time = match q.last_used {
                    None => true, // never used
                    Some(last) => now.saturating_sub(last) > threshold_secs,
                };
                let low_coverage = q.triple_coverage < self.config.min_coverage;
                stale_by_time && low_coverage
            })
            .map(|q| q.skill_id.as_str())
            .collect()
    }

    /// Return skill IDs with success rate below the configured threshold.
    pub fn underperforming_skills(&self) -> Vec<&str> {
        self.records
            .values()
            .filter(|q| q.total_executions > 0 && q.success_rate < self.config.min_success_rate)
            .map(|q| q.skill_id.as_str())
            .collect()
    }

    /// Return all quality records.
    pub fn all(&self) -> Vec<&SkillQuality> {
        self.records.values().collect()
    }

    /// Remove a skill's quality record (e.g. on full uninstall).
    pub fn remove(&mut self, skill_id: &str) {
        self.records.remove(skill_id);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_rate_tracking() {
        let mut q = SkillQuality::new("my-skill", 10);
        q.record_success();
        q.record_success();
        q.record_failure("act_a", "precondition_unmet");
        assert_eq!(q.total_executions, 3);
        assert!((q.success_rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn failure_mode_dedup() {
        let mut q = SkillQuality::new("s", 5);
        q.record_failure("act_a", "timeout");
        q.record_failure("act_a", "timeout");
        q.record_failure("act_b", "precondition_unmet");
        assert_eq!(q.failure_modes.len(), 2);
        let top = q.top_failure_mode().unwrap();
        assert_eq!(top.action, "act_a");
        assert_eq!(top.count, 2);
    }

    #[test]
    fn triple_coverage() {
        let mut q = SkillQuality::new("s", 4);
        q.record_triple_query(true);
        q.record_triple_query(true);
        assert!((q.triple_coverage - 0.5).abs() < 1e-6);
        // Querying same triple again (not new) doesn't change coverage.
        q.record_triple_query(false);
        assert!((q.triple_coverage - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rule_hit_rates() {
        let mut q = SkillQuality::new("s", 0);
        q.record_rule_hit("rule-0");
        q.record_rule_hit("rule-0");
        q.record_rule_miss("rule-1");
        // rule-0 hit 2 out of 3 passes
        let rate = q.rule_hit_rates["rule-0"];
        assert!(rate > 0.0 && rate <= 1.0, "rate={rate}");
    }

    #[test]
    fn staleness_detection() {
        let mut reg = QualityRegistry::with_config(StalenessConfig {
            stale_days: 1,
            min_coverage: 0.5,
            min_success_rate: 0.5,
        });
        reg.ensure("never-used", 100);
        // A skill never used has coverage 0 < 0.5, so stale.
        let stale = reg.stale_skills();
        assert!(stale.contains(&"never-used"));
    }

    #[test]
    fn underperforming_skills() {
        let mut reg = QualityRegistry::new();
        reg.ensure("bad-skill", 5);
        let q = reg.get_mut("bad-skill").unwrap();
        q.record_failure("act", "timeout");
        q.record_failure("act", "timeout");

        let under = reg.underperforming_skills();
        assert!(under.contains(&"bad-skill"));
    }

    #[test]
    fn refinement_tracking() {
        let mut q = SkillQuality::new("s", 0);
        q.set_refinements_pending(3);
        q.record_refinement_applied();
        assert_eq!(q.refinements_pending, 2);
        assert_eq!(q.refinements_applied, 1);
    }
}
