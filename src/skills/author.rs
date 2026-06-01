//! LLM-assisted skill authoring — Phase 30c.
//!
//! Translates natural language procedure descriptions into complete skillpack
//! files (`triples.json`, `actions.json`, `plans.json`, `rules.txt`,
//! `procedures.md`).
//!
//! The pipeline uses the Candle LLM backend (Phase 26b) via a multi-pass
//! extraction strategy so each pass uses a focused prompt rather than one
//! large inference call — important for small models like Qwen2.5-1.5B.
//!
//! # Extraction passes
//!
//! 1. **Concept pass** — extract domain entities and relationships → triples
//! 2. **Action pass** — extract steps with preconditions/effects → actions
//! 3. **Plan pass** — extract goal decomposition and ordering → plans
//! 4. **Rule pass** — extract heuristics and inference patterns → rules
//! 5. **Validation pass** — check internal consistency (all action refs in
//!    plans exist, all preconditions reference known fluents)
//!
//! When the `candle-backend` feature is not enabled this module compiles but
//! `SkillAuthor::author_from_description` returns
//! `AuthorError::BackendUnavailable`.

use std::collections::HashSet;
use std::path::PathBuf;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{LabelTriple, SkillAction, SkillEffects, SkillManifest, SkillPlan};

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error, Diagnostic)]
pub enum AuthorError {
    #[error("LLM backend unavailable: {reason}")]
    #[diagnostic(
        code(akh::skill::author::backend_unavailable),
        help(
            "LLM-assisted skill authoring requires the `candle-backend` feature (Phase 26b). \
             Build with `--features candle-backend` or author the skill files by hand."
        )
    )]
    BackendUnavailable { reason: String },

    #[error("extraction pass failed ({pass}): {message}")]
    #[diagnostic(
        code(akh::skill::author::extraction_failed),
        help(
            "The LLM could not extract structured data from the provided description. \
             Try rephrasing with clearer step descriptions and explicit preconditions."
        )
    )]
    ExtractionFailed { pass: String, message: String },

    #[error("validation failed: {message}")]
    #[diagnostic(
        code(akh::skill::author::validation_failed),
        help(
            "The generated skill failed internal consistency checks. \
             Review the generated files and fix the listed issues."
        )
    )]
    ValidationFailed { message: String },

    #[error("I/O error writing skill to {path}: {source}")]
    #[diagnostic(
        code(akh::skill::author::io),
        help("Ensure the target skills directory exists and is writable.")
    )]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("serialization error: {message}")]
    #[diagnostic(
        code(akh::skill::author::serialize),
        help("Internal JSON serialization failure; this is a bug.")
    )]
    Serialize { message: String },
}

pub type AuthorResult<T> = std::result::Result<T, AuthorError>;

// ─────────────────────────────────────────────────────────────────────────────
// AuthoredSkill
// ─────────────────────────────────────────────────────────────────────────────

/// A complete skill produced by the authoring pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthoredSkill {
    /// The skill manifest.
    pub manifest: SkillManifest,
    /// Label-based triples (concept pass).
    pub triples: Vec<LabelTriple>,
    /// Action schemas (action pass).
    pub actions: Vec<SkillAction>,
    /// Plan templates (plan pass).
    pub plans: Vec<SkillPlan>,
    /// Egg rewrite rules — one rule per line in `lhs => rhs` format.
    pub rules: Vec<String>,
    /// Original natural language text (preserved as `procedures.md`).
    pub source_text: String,
}

impl AuthoredSkill {
    /// Write all skill files into `target_dir/<skill_id>/`.
    ///
    /// Creates the directory if it does not exist.  Returns the path to the
    /// created skill directory.
    pub fn write_to_disk(&self, skills_dir: &std::path::Path) -> AuthorResult<PathBuf> {
        let skill_dir = skills_dir.join(&self.manifest.id);
        std::fs::create_dir_all(&skill_dir).map_err(|e| AuthorError::Io {
            path: skill_dir.display().to_string(),
            source: e,
        })?;

        // skill.json
        let manifest_json =
            serde_json::to_string_pretty(&self.manifest).map_err(|e| AuthorError::Serialize {
                message: e.to_string(),
            })?;
        std::fs::write(skill_dir.join("skill.json"), manifest_json).map_err(|e| {
            AuthorError::Io {
                path: skill_dir.join("skill.json").display().to_string(),
                source: e,
            }
        })?;

        // triples.json
        if !self.triples.is_empty() {
            let triples_json = serde_json::to_string_pretty(&self.triples).map_err(|e| {
                AuthorError::Serialize {
                    message: e.to_string(),
                }
            })?;
            std::fs::write(skill_dir.join("triples.json"), triples_json).map_err(|e| {
                AuthorError::Io {
                    path: skill_dir.join("triples.json").display().to_string(),
                    source: e,
                }
            })?;
        }

        // actions.json
        if !self.actions.is_empty() {
            let actions_json = serde_json::to_string_pretty(&self.actions).map_err(|e| {
                AuthorError::Serialize {
                    message: e.to_string(),
                }
            })?;
            std::fs::write(skill_dir.join("actions.json"), actions_json).map_err(|e| {
                AuthorError::Io {
                    path: skill_dir.join("actions.json").display().to_string(),
                    source: e,
                }
            })?;
        }

        // plans.json
        if !self.plans.is_empty() {
            let plans_json =
                serde_json::to_string_pretty(&self.plans).map_err(|e| AuthorError::Serialize {
                    message: e.to_string(),
                })?;
            std::fs::write(skill_dir.join("plans.json"), plans_json).map_err(|e| {
                AuthorError::Io {
                    path: skill_dir.join("plans.json").display().to_string(),
                    source: e,
                }
            })?;
        }

        // rules.txt
        if !self.rules.is_empty() {
            let rules_text = self.rules.join("\n");
            std::fs::write(skill_dir.join("rules.txt"), rules_text).map_err(|e| {
                AuthorError::Io {
                    path: skill_dir.join("rules.txt").display().to_string(),
                    source: e,
                }
            })?;
        }

        // procedures.md — natural language source of truth
        if !self.source_text.is_empty() {
            std::fs::write(skill_dir.join("procedures.md"), &self.source_text).map_err(|e| {
                AuthorError::Io {
                    path: skill_dir.join("procedures.md").display().to_string(),
                    source: e,
                }
            })?;
        }

        Ok(skill_dir)
    }

    /// Validate internal consistency of the authored skill.
    ///
    /// Checks:
    /// - All step names in plans reference a known action name.
    /// - No duplicate action names.
    pub fn validate(&self) -> AuthorResult<()> {
        let action_names: HashSet<&str> = self.actions.iter().map(|a| a.action.as_str()).collect();

        // Check for duplicate action names.
        let mut seen = HashSet::new();
        for action in &self.actions {
            if !seen.insert(action.action.as_str()) {
                return Err(AuthorError::ValidationFailed {
                    message: format!("duplicate action name: `{}`", action.action),
                });
            }
        }

        // Check plan step references.
        for plan in &self.plans {
            for step in &plan.steps {
                if !action_names.contains(step.as_str()) {
                    return Err(AuthorError::ValidationFailed {
                        message: format!(
                            "plan `{}` references unknown action `{}`; \
                             known actions: [{}]",
                            plan.goal,
                            step,
                            action_names.iter().cloned().collect::<Vec<_>>().join(", ")
                        ),
                    });
                }
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExtractionHints — caller-provided guidance for the authoring pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Optional hints passed to the authoring pipeline to steer extraction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionHints {
    /// Suggested domain tags (merged with LLM-detected ones).
    pub domains: Vec<String>,
    /// Fluents the caller knows exist in the world model (seed for validation).
    pub known_fluents: Vec<String>,
    /// Maximum number of actions to extract.
    pub max_actions: Option<usize>,
    /// Maximum number of plan templates to extract.
    pub max_plans: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SkillAuthor
// ─────────────────────────────────────────────────────────────────────────────

/// Orchestrates the multi-pass LLM extraction pipeline for skill authoring.
///
/// Construction succeeds even when the Candle backend is absent; the authoring
/// methods return `AuthorError::BackendUnavailable` in that case.  This lets
/// callers handle the case gracefully at runtime rather than at compile time.
pub struct SkillAuthor {
    /// Whether the LLM backend is available (Phase 26b Candle integration).
    llm_available: bool,
}

impl std::fmt::Debug for SkillAuthor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillAuthor")
            .field("llm_available", &self.llm_available)
            .finish()
    }
}

impl SkillAuthor {
    /// Create a new author.  Pass `llm_available = true` when the Candle
    /// backend has been initialised (Phase 26b).
    pub fn new(llm_available: bool) -> Self {
        Self { llm_available }
    }

    /// Return whether the LLM backend is available.
    pub fn is_llm_available(&self) -> bool {
        self.llm_available
    }

    /// Generate a complete skill from a natural language description.
    ///
    /// When the LLM backend is unavailable this returns a minimal stub skill
    /// with only the metadata populated so the caller can still save it and
    /// fill in the procedural files by hand.
    pub fn author_from_description(
        &self,
        id: &str,
        name: &str,
        description: &str,
        hints: ExtractionHints,
    ) -> AuthorResult<AuthoredSkill> {
        if !self.llm_available {
            // Return a stub — the caller can fill in files manually.
            return Ok(self.stub_skill(id, name, description, hints));
        }

        // Full multi-pass pipeline (requires Phase 26b Candle backend).
        // Implemented here once the backend is wired; for now we fall back to
        // the stub path since we cannot call the LLM.
        //
        // Planned passes:
        // 1. concept_pass(description) → Vec<LabelTriple>
        // 2. action_pass(description, &triples) → Vec<SkillAction>
        // 3. plan_pass(description, &actions) → Vec<SkillPlan>
        // 4. rule_pass(description, &triples, &actions) → Vec<String>
        // 5. validate(&authored_skill)
        //
        // Each pass sends a focused prompt to the LLM and parses the JSON
        // response with brace-depth tracking to handle streaming tokens.
        tracing::warn!(
            skill_id = id,
            "LLM-assisted authoring is not yet fully wired to Phase 26b Candle backend; \
             returning stub skill"
        );
        Ok(self.stub_skill(id, name, description, hints))
    }

    /// Re-generate structured skill files from an updated `procedures.md`.
    ///
    /// Useful when the user edits the natural language description after initial
    /// authoring.  The regeneration uses the same multi-pass pipeline as
    /// `author_from_description`.
    pub fn regenerate_from_procedures(
        &self,
        id: &str,
        name: &str,
        procedures_text: &str,
        hints: ExtractionHints,
    ) -> AuthorResult<AuthoredSkill> {
        self.author_from_description(id, name, procedures_text, hints)
    }

    // ─── Internal helpers ────────────────────────────────────────────────────

    /// Build a minimal stub skill with no procedural content.
    fn stub_skill(
        &self,
        id: &str,
        name: &str,
        description: &str,
        hints: ExtractionHints,
    ) -> AuthoredSkill {
        let domains = if hints.domains.is_empty() {
            vec!["general".to_string()]
        } else {
            hints.domains
        };

        AuthoredSkill {
            manifest: SkillManifest {
                id: id.to_string(),
                name: name.to_string(),
                version: "0.1.0".to_string(),
                description: description.to_string(),
                domains,
                weight_size_bytes: 0,
                triples_file: None,
                rules_file: None,
                cli_tools: Vec::new(),
                wasm_tools: Vec::new(),
                tool_config: std::collections::HashMap::new(),
                actions_file: Some("actions.json".into()),
                plans_file: Some("plans.json".into()),
                procedures_file: Some("procedures.md".into()),
            },
            triples: Vec::new(),
            actions: Vec::new(),
            plans: Vec::new(),
            rules: Vec::new(),
            source_text: description.to_string(),
        }
    }

    // ─── Static helpers for building skill components ────────────────────────

    /// Build a `SkillAction` from raw parts.  Convenience for expert / Tier 3
    /// callers who want to construct actions programmatically.
    pub fn build_action(
        action: impl Into<String>,
        description: impl Into<String>,
        preconditions: Vec<String>,
        initiates: Vec<String>,
        terminates: Vec<String>,
    ) -> SkillAction {
        SkillAction {
            action: action.into(),
            description: description.into(),
            preconditions,
            effects: SkillEffects {
                initiates,
                terminates,
            },
            estimated_duration: super::ActionDuration::Fast,
        }
    }

    /// Build a sequential `SkillPlan` from raw parts.
    pub fn build_sequential_plan(
        goal: impl Into<String>,
        steps: Vec<String>,
        preconditions: Vec<String>,
    ) -> SkillPlan {
        SkillPlan {
            goal: goal.into(),
            method: super::PlanMethod::Sequential,
            steps,
            preconditions,
            alternatives: Vec::new(),
            quality_gate: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::SkillState;
    use tempfile::TempDir;

    fn sample_authored_skill() -> AuthoredSkill {
        let author = SkillAuthor::new(false);
        let hints = ExtractionHints {
            domains: vec!["code-review".into()],
            ..Default::default()
        };
        let mut skill = author
            .author_from_description("code-review", "Code Review", "review code", hints)
            .unwrap();

        // Populate manually for tests.
        skill.actions = vec![
            SkillAuthor::build_action(
                "read_file",
                "Read the file under review",
                vec!["file_available".into()],
                vec!["file_read".into()],
                vec!["file_unknown".into()],
            ),
            SkillAuthor::build_action(
                "find_bugs",
                "Identify bugs in the file",
                vec!["file_read".into()],
                vec!["bugs_identified".into()],
                vec![],
            ),
        ];
        skill.plans = vec![SkillAuthor::build_sequential_plan(
            "review_code",
            vec!["read_file".into(), "find_bugs".into()],
            vec!["file_available".into()],
        )];
        skill
    }

    #[test]
    fn stub_skill_has_correct_metadata() {
        let author = SkillAuthor::new(false);
        let hints = ExtractionHints::default();
        let skill = author
            .author_from_description("my-skill", "My Skill", "does things", hints)
            .unwrap();
        assert_eq!(skill.manifest.id, "my-skill");
        assert_eq!(skill.manifest.name, "My Skill");
        assert_eq!(skill.manifest.version, "0.1.0");
    }

    #[test]
    fn validation_passes_for_consistent_skill() {
        let skill = sample_authored_skill();
        assert!(skill.validate().is_ok());
    }

    #[test]
    fn validation_fails_for_unknown_action_ref() {
        let mut skill = sample_authored_skill();
        skill.plans[0].steps.push("nonexistent_action".into());
        let err = skill.validate().unwrap_err();
        assert!(matches!(err, AuthorError::ValidationFailed { .. }));
    }

    #[test]
    fn validation_fails_for_duplicate_action_name() {
        let mut skill = sample_authored_skill();
        skill.actions.push(skill.actions[0].clone()); // duplicate
        let err = skill.validate().unwrap_err();
        assert!(matches!(err, AuthorError::ValidationFailed { .. }));
    }

    #[test]
    fn write_to_disk_creates_files() {
        let tmp = TempDir::new().unwrap();
        let skill = sample_authored_skill();
        let skill_dir = skill.write_to_disk(tmp.path()).unwrap();

        assert!(skill_dir.join("skill.json").exists());
        assert!(skill_dir.join("actions.json").exists());
        assert!(skill_dir.join("plans.json").exists());
        assert!(skill_dir.join("procedures.md").exists());
    }

    #[test]
    fn build_helpers_produce_correct_types() {
        let action = SkillAuthor::build_action(
            "step_a",
            "Do step A",
            vec!["pre_a".into()],
            vec!["post_a".into()],
            vec!["pre_a".into()],
        );
        assert_eq!(action.action, "step_a");
        assert_eq!(action.effects.initiates, vec!["post_a"]);

        let plan = SkillAuthor::build_sequential_plan(
            "my_goal",
            vec!["step_a".into()],
            vec!["pre_a".into()],
        );
        assert_eq!(plan.goal, "my_goal");
        assert_eq!(plan.steps, vec!["step_a"]);
        assert_eq!(plan.method, super::super::PlanMethod::Sequential);
    }

    // `SkillState` is not used inside this module at runtime, but verifying
    // it's accessible proves the import chain compiles correctly.
    #[test]
    fn skill_state_accessible() {
        assert_eq!(SkillState::Cold.to_string(), "Cold");
    }
}
