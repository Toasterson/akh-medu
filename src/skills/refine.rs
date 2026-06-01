//! Rule refinement loop — Phase 30d.
//!
//! Analyses plan execution failures involving skill procedures and suggests
//! corrections to action schemas, preconditions, plan methods, and inference
//! rules.  Suggestions are stored as *pending* KG triples so the user can
//! review and approve them before they are applied.
//!
//! # Failure analysis pipeline
//!
//! 1. Collect recent plan execution failures that involved a skill's actions.
//! 2. Extract the relevant context (expected state, actual state, failed
//!    preconditions, last rule that fired).
//! 3. Pass context to the LLM (Phase 26b) with the skill's `procedures.md` for
//!    grounding.
//! 4. Parse `RefinementSuggestion`s from the LLM response.
//! 5. Store each suggestion as a pending triple in the KG:
//!    `(suggestion_id, skill:pending_refinement, skill_id)`.
//! 6. Expose pending suggestions for user review via MCP / CLI.
//!
//! When the LLM backend is unavailable, `SkillRefiner::analyze_failure` still
//! stores a heuristic suggestion based on the failure metadata alone.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{LabelTriple, SkillAction, SkillEffects, SkillPlan};

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, Diagnostic)]
pub enum RefineError {
    #[error("skill not found for refinement: {skill_id}")]
    #[diagnostic(
        code(akh::skill::refine::not_found),
        help("The skill must be Hot (activated) before refinements can be generated for it.")
    )]
    SkillNotFound { skill_id: String },

    #[error("no failure context available for skill `{skill_id}`")]
    #[diagnostic(
        code(akh::skill::refine::no_failures),
        help(
            "There are no recorded plan failures for this skill. \
             Execute the skill's plans at least once before requesting refinements."
        )
    )]
    NoFailures { skill_id: String },

    #[error("LLM backend unavailable: {reason}")]
    #[diagnostic(
        code(akh::skill::refine::backend_unavailable),
        help(
            "Full LLM-assisted refinement requires the `candle-backend` feature (Phase 26b). \
             Heuristic refinements are still generated without the LLM."
        )
    )]
    BackendUnavailable { reason: String },

    #[error("serialization error: {message}")]
    #[diagnostic(
        code(akh::skill::refine::serialize),
        help("Internal JSON serialization failure; this is a bug.")
    )]
    Serialize { message: String },
}

pub type RefineResult<T> = std::result::Result<T, RefineError>;

// ─────────────────────────────────────────────────────────────────────────────
// FailureContext — input to the refiner
// ─────────────────────────────────────────────────────────────────────────────

/// The type of plan execution failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureType {
    /// A required precondition fluent was not present in the world state.
    PreconditionUnmet,
    /// An action executed but produced unexpected effects.
    UnexpectedEffect,
    /// The action timed out before completing.
    Timeout,
    /// A plan step referenced an action that doesn't exist in the registry.
    UnknownAction,
    /// The goal was not satisfied after all plan steps completed.
    GoalNotAchieved,
    /// A custom / catch-all failure.
    Other(String),
}

/// Context describing a single plan execution failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureContext {
    /// The skill whose procedures were being executed.
    pub skill_id: String,
    /// The goal that was being pursued.
    pub goal: String,
    /// The action that failed.
    pub failed_action: String,
    /// What kind of failure occurred.
    pub failure_type: FailureType,
    /// Fluents that were present in the world state when the failure occurred.
    pub world_state_fluents: Vec<String>,
    /// Preconditions that were expected but not met.
    pub unmet_preconditions: Vec<String>,
    /// Effects that were expected but did not materialise.
    pub missing_effects: Vec<String>,
    /// Unix timestamp of the failure.
    pub timestamp: u64,
    /// Optional identifier of the provenance record (for tracing back through
    /// the KG provenance chain).
    pub provenance_id: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// RefinementKind — what kind of change is suggested
// ─────────────────────────────────────────────────────────────────────────────

/// A concrete refinement suggestion targeting a specific part of the skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RefinementKind {
    /// Add a new inference rule to `rules.txt`.
    NewRule {
        /// Left-hand side of the rule in egg syntax.
        lhs: String,
        /// Right-hand side of the rule in egg syntax.
        rhs: String,
    },
    /// Modify the preconditions of an existing action.
    ModifiedPrecondition {
        /// The action to update.
        action: String,
        /// Fluent names to add as preconditions.
        add: Vec<String>,
        /// Fluent names to remove from preconditions.
        remove: Vec<String>,
    },
    /// Modify the effects of an existing action.
    ModifiedEffect {
        /// The action to update.
        action: String,
        /// Additional fluents to initiate.
        add_initiates: Vec<String>,
        /// Fluents to stop initiating.
        remove_initiates: Vec<String>,
        /// Additional fluents to terminate.
        add_terminates: Vec<String>,
        /// Fluents to stop terminating.
        remove_terminates: Vec<String>,
    },
    /// Provide an alternative decomposition path for a goal.
    AlternativePlan {
        /// The goal this alternative plan achieves.
        goal: String,
        /// The new plan template to add.
        new_plan: SkillPlan,
    },
    /// Add a completely new action to `actions.json`.
    NewAction {
        /// The action schema to add.
        action: SkillAction,
    },
    /// Add a missing triple to `triples.json`.
    MissingTriple {
        /// The triple to add.
        triple: LabelTriple,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// RefinementSuggestion
// ─────────────────────────────────────────────────────────────────────────────

/// A pending refinement suggestion waiting for user approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementSuggestion {
    /// Unique identifier for this suggestion.
    pub id: String,
    /// The skill this suggestion targets.
    pub skill_id: String,
    /// What kind of change is being suggested.
    pub refinement: RefinementKind,
    /// Human-readable explanation of why this refinement is needed.
    pub rationale: String,
    /// Confidence score [0.0, 1.0] — higher means LLM is more certain.
    pub confidence: f32,
    /// Which failure context triggered this suggestion.
    pub source_failure_action: String,
    /// Unix timestamp when the suggestion was created.
    pub created_at: u64,
    /// Whether this suggestion has been reviewed.
    pub reviewed: bool,
    /// User decision: `Some(true)` = accepted, `Some(false)` = rejected.
    pub accepted: Option<bool>,
}

impl RefinementSuggestion {
    fn new(
        skill_id: &str,
        refinement: RefinementKind,
        rationale: impl Into<String>,
        confidence: f32,
        source_action: &str,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Generate a deterministic ID from skill + action + refinement content + timestamp.
        let refinement_tag = serde_json::to_string(&refinement)
            .unwrap_or_default()
            .chars()
            .take(64)
            .collect::<String>();
        let id = format!("refine:{skill_id}:{source_action}:{now}:{refinement_tag}");

        Self {
            id,
            skill_id: skill_id.to_string(),
            refinement,
            rationale: rationale.into(),
            confidence,
            source_failure_action: source_action.to_string(),
            created_at: now,
            reviewed: false,
            accepted: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PendingRefinements — in-memory store for pending suggestions
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory store for pending refinement suggestions.
///
/// In production this is persisted to KG triples via `skill:pending_refinement`
/// predicates.  This struct manages the hot-path in-memory view.
#[derive(Debug, Default)]
pub struct PendingRefinements {
    /// `skill_id` → list of pending suggestions.
    suggestions: HashMap<String, Vec<RefinementSuggestion>>,
}

impl PendingRefinements {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a suggestion (replaces any existing suggestion with the same id).
    pub fn add(&mut self, suggestion: RefinementSuggestion) {
        let list = self
            .suggestions
            .entry(suggestion.skill_id.clone())
            .or_default();
        // Deduplicate by id.
        list.retain(|s| s.id != suggestion.id);
        list.push(suggestion);
    }

    /// Return all pending (unreviewed) suggestions for a skill.
    pub fn pending_for(&self, skill_id: &str) -> Vec<&RefinementSuggestion> {
        self.suggestions
            .get(skill_id)
            .map(|list| list.iter().filter(|s| !s.reviewed).collect())
            .unwrap_or_default()
    }

    /// Return all suggestions for all skills.
    pub fn all(&self) -> Vec<&RefinementSuggestion> {
        self.suggestions.values().flatten().collect()
    }

    /// Mark a suggestion as accepted.  Returns `false` if not found.
    pub fn accept(&mut self, suggestion_id: &str) -> bool {
        for list in self.suggestions.values_mut() {
            if let Some(s) = list.iter_mut().find(|s| s.id == suggestion_id) {
                s.reviewed = true;
                s.accepted = Some(true);
                return true;
            }
        }
        false
    }

    /// Mark a suggestion as rejected.  Returns `false` if not found.
    pub fn reject(&mut self, suggestion_id: &str) -> bool {
        for list in self.suggestions.values_mut() {
            if let Some(s) = list.iter_mut().find(|s| s.id == suggestion_id) {
                s.reviewed = true;
                s.accepted = Some(false);
                return true;
            }
        }
        false
    }

    /// Remove all suggestions for a skill (e.g. on deactivation).
    pub fn clear_skill(&mut self, skill_id: &str) {
        self.suggestions.remove(skill_id);
    }

    /// Count of unreviewed suggestions across all skills.
    pub fn pending_count(&self) -> usize {
        self.suggestions
            .values()
            .flatten()
            .filter(|s| !s.reviewed)
            .count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SkillRefiner
// ─────────────────────────────────────────────────────────────────────────────

/// Analyses plan failures and generates `RefinementSuggestion`s.
pub struct SkillRefiner {
    llm_available: bool,
}

impl std::fmt::Debug for SkillRefiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillRefiner")
            .field("llm_available", &self.llm_available)
            .finish()
    }
}

impl SkillRefiner {
    /// Create a refiner.  Pass `llm_available = true` when the Candle backend
    /// has been initialised (Phase 26b).
    pub fn new(llm_available: bool) -> Self {
        Self { llm_available }
    }

    /// Analyse a single plan failure and generate refinement suggestions.
    ///
    /// Returns suggestions that should be added to `PendingRefinements` for
    /// user review.
    pub fn analyze_failure(
        &self,
        failure: &FailureContext,
        existing_actions: &[SkillAction],
    ) -> RefineResult<Vec<RefinementSuggestion>> {
        let mut suggestions = Vec::new();

        if self.llm_available {
            // Full LLM-assisted analysis (Phase 26b wiring needed).
            // For now fall through to heuristic path with a warning.
            tracing::warn!(
                skill_id = %failure.skill_id,
                "LLM-assisted refinement not yet wired to Phase 26b; using heuristics"
            );
        }

        // Heuristic analysis: generate suggestions from failure metadata alone.
        suggestions.extend(self.heuristic_suggestions(failure, existing_actions));
        Ok(suggestions)
    }

    /// Analyse a batch of failures for one skill, deduplicating similar
    /// suggestions.
    pub fn analyze_batch(
        &self,
        failures: &[FailureContext],
        existing_actions: &[SkillAction],
    ) -> RefineResult<Vec<RefinementSuggestion>> {
        let mut all: Vec<RefinementSuggestion> = Vec::new();

        for failure in failures {
            let mut batch = self.analyze_failure(failure, existing_actions)?;
            // Simple dedup: skip if a suggestion with the same rationale already
            // exists (avoids flooding the user with identical suggestions).
            batch.retain(|new| !all.iter().any(|old| old.rationale == new.rationale));
            all.extend(batch);
        }

        Ok(all)
    }

    // ─── Heuristic rules ────────────────────────────────────────────────────

    fn heuristic_suggestions(
        &self,
        failure: &FailureContext,
        existing_actions: &[SkillAction],
    ) -> Vec<RefinementSuggestion> {
        let mut suggestions = Vec::new();

        match &failure.failure_type {
            FailureType::PreconditionUnmet => {
                // Suggest removing each unmet precondition from the action.
                if !failure.unmet_preconditions.is_empty() {
                    suggestions.push(RefinementSuggestion::new(
                        &failure.skill_id,
                        RefinementKind::ModifiedPrecondition {
                            action: failure.failed_action.clone(),
                            add: Vec::new(),
                            remove: failure.unmet_preconditions.clone(),
                        },
                        format!(
                            "Action `{}` failed because precondition(s) [{}] were not met. \
                             Consider removing them or adding a preceding action that \
                             establishes them.",
                            failure.failed_action,
                            failure.unmet_preconditions.join(", ")
                        ),
                        0.6,
                        &failure.failed_action,
                    ));

                    // Suggest adding a rule that derives the missing precondition
                    // from existing world state.
                    for unmet in &failure.unmet_preconditions {
                        if let Some(fluent) = failure.world_state_fluents.first() {
                            suggestions.push(RefinementSuggestion::new(
                                &failure.skill_id,
                                RefinementKind::NewRule {
                                    lhs: format!("({fluent} ?x)"),
                                    rhs: format!("({unmet} ?x)"),
                                },
                                format!(
                                    "When `{fluent}` holds it may imply `{unmet}`. \
                                     If correct, this rule bridges the precondition gap."
                                ),
                                0.4,
                                &failure.failed_action,
                            ));
                        }
                    }
                }
            }

            FailureType::UnexpectedEffect => {
                // Suggest updating the effects list to match observed outcomes.
                if !failure.missing_effects.is_empty() {
                    suggestions.push(RefinementSuggestion::new(
                        &failure.skill_id,
                        RefinementKind::ModifiedEffect {
                            action: failure.failed_action.clone(),
                            add_initiates: failure.missing_effects.clone(),
                            remove_initiates: Vec::new(),
                            add_terminates: Vec::new(),
                            remove_terminates: Vec::new(),
                        },
                        format!(
                            "Action `{}` produced different effects than expected. \
                             Adding [{}] to its `initiates` list.",
                            failure.failed_action,
                            failure.missing_effects.join(", ")
                        ),
                        0.55,
                        &failure.failed_action,
                    ));
                }
            }

            FailureType::UnknownAction => {
                // Suggest adding a stub action with no preconditions.
                let has_action = existing_actions
                    .iter()
                    .any(|a| a.action == failure.failed_action);
                if !has_action {
                    suggestions.push(RefinementSuggestion::new(
                        &failure.skill_id,
                        RefinementKind::NewAction {
                            action: SkillAction {
                                action: failure.failed_action.clone(),
                                description: format!(
                                    "Auto-generated stub for `{}`; fill in details.",
                                    failure.failed_action
                                ),
                                preconditions: Vec::new(),
                                effects: SkillEffects {
                                    initiates: Vec::new(),
                                    terminates: Vec::new(),
                                },
                                estimated_duration: super::ActionDuration::Fast,
                            },
                        },
                        format!(
                            "Plan referenced action `{}` which is not in actions.json. \
                             A stub has been generated.",
                            failure.failed_action
                        ),
                        0.8,
                        &failure.failed_action,
                    ));
                }
            }

            FailureType::GoalNotAchieved => {
                // Suggest an alternative plan with the current steps.
                suggestions.push(RefinementSuggestion::new(
                    &failure.skill_id,
                    RefinementKind::AlternativePlan {
                        goal: failure.goal.clone(),
                        new_plan: SkillPlan {
                            goal: failure.goal.clone(),
                            method: super::PlanMethod::Sequential,
                            steps: existing_actions.iter().map(|a| a.action.clone()).collect(),
                            preconditions: Vec::new(),
                            alternatives: Vec::new(),
                            quality_gate: None,
                        },
                    },
                    format!(
                        "Goal `{}` was not achieved by the current plan. \
                         An alternative plan using all known actions is suggested.",
                        failure.goal
                    ),
                    0.35,
                    &failure.failed_action,
                ));
            }

            FailureType::Timeout | FailureType::Other(_) => {
                // No heuristic refinement available.
                tracing::debug!(
                    skill_id = %failure.skill_id,
                    failure_type = ?failure.failure_type,
                    "no heuristic refinement available for this failure type"
                );
            }
        }

        suggestions
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_failure(skill_id: &str, action: &str, failure_type: FailureType) -> FailureContext {
        FailureContext {
            skill_id: skill_id.to_string(),
            goal: "triage_bug".to_string(),
            failed_action: action.to_string(),
            failure_type,
            world_state_fluents: vec!["bug_reported".to_string()],
            unmet_preconditions: vec!["severity_assessed".to_string()],
            missing_effects: vec!["bug_triaged".to_string()],
            timestamp: 0,
            provenance_id: None,
        }
    }

    #[test]
    fn heuristic_precondition_unmet_generates_suggestions() {
        let refiner = SkillRefiner::new(false);
        let failure = sample_failure("bug-triage", "assign_team", FailureType::PreconditionUnmet);
        let suggestions = refiner.analyze_failure(&failure, &[]).unwrap();

        assert!(
            !suggestions.is_empty(),
            "should generate at least one suggestion"
        );
        let has_precond = suggestions.iter().any(|s| {
            matches!(
                &s.refinement,
                RefinementKind::ModifiedPrecondition { action, .. }
                if action == "assign_team"
            )
        });
        assert!(has_precond, "should suggest modifying precondition");
    }

    #[test]
    fn heuristic_unknown_action_generates_stub() {
        let refiner = SkillRefiner::new(false);
        let failure = sample_failure("bug-triage", "missing_action", FailureType::UnknownAction);
        let suggestions = refiner.analyze_failure(&failure, &[]).unwrap();

        let has_new_action = suggestions
            .iter()
            .any(|s| matches!(&s.refinement, RefinementKind::NewAction { .. }));
        assert!(has_new_action, "should suggest adding the missing action");
    }

    #[test]
    fn heuristic_unknown_action_skipped_if_exists() {
        let refiner = SkillRefiner::new(false);
        let existing = vec![SkillAction {
            action: "existing_action".to_string(),
            description: String::new(),
            preconditions: Vec::new(),
            effects: SkillEffects {
                initiates: Vec::new(),
                terminates: Vec::new(),
            },
            estimated_duration: super::super::ActionDuration::Fast,
        }];
        let failure = sample_failure("s", "existing_action", FailureType::UnknownAction);
        let suggestions = refiner.analyze_failure(&failure, &existing).unwrap();
        let has_new = suggestions
            .iter()
            .any(|s| matches!(&s.refinement, RefinementKind::NewAction { .. }));
        assert!(
            !has_new,
            "should not suggest new action if it already exists"
        );
    }

    #[test]
    fn pending_refinements_accept_reject() {
        let mut pending = PendingRefinements::new();
        let sug = RefinementSuggestion::new(
            "skill-a",
            RefinementKind::NewRule {
                lhs: "(a ?x)".to_string(),
                rhs: "(b ?x)".to_string(),
            },
            "test",
            0.9,
            "action_x",
        );
        let id = sug.id.clone();
        pending.add(sug);
        assert_eq!(pending.pending_count(), 1);

        pending.accept(&id);
        assert_eq!(pending.pending_count(), 0);
        let all = pending.all();
        assert_eq!(all[0].accepted, Some(true));
    }

    #[test]
    fn pending_refinements_clear_skill() {
        let mut pending = PendingRefinements::new();
        for i in 0..3 {
            pending.add(RefinementSuggestion::new(
                "skill-a",
                RefinementKind::NewRule {
                    lhs: format!("(a{i} ?x)"),
                    rhs: format!("(b{i} ?x)"),
                },
                "test",
                0.5,
                "act",
            ));
        }
        assert_eq!(pending.pending_count(), 3);
        pending.clear_skill("skill-a");
        assert_eq!(pending.pending_count(), 0);
    }

    #[test]
    fn batch_analysis_deduplicates() {
        let refiner = SkillRefiner::new(false);
        let f1 = sample_failure("s", "act", FailureType::PreconditionUnmet);
        let f2 = sample_failure("s", "act", FailureType::PreconditionUnmet);
        let suggestions = refiner.analyze_batch(&[f1, f2], &[]).unwrap();

        // Duplicate rationale should be filtered.
        let unique_rationales: std::collections::HashSet<&str> =
            suggestions.iter().map(|s| s.rationale.as_str()).collect();
        assert_eq!(suggestions.len(), unique_rationales.len());
    }
}
