//! Skillpack manager: MoE-style modular expert loading.
//!
//! Skillpacks are loadable modules containing domain-specific knowledge
//! (triples), rewrite rules, metadata, and procedural knowledge (Phase 30).
//! They follow a lifecycle: Cold (on disk) → Warm (manifest parsed) → Hot
//! (active in memory).
//!
//! Phase 30 extends skills with:
//! - `actions.json`  — causal action schemas (preconditions + effects)
//! - `plans.json`    — HTN decomposition templates
//! - `procedures.md` — natural language source of truth (for LLM re-generation)

pub mod author;
pub mod manager;
pub mod quality;
pub mod refine;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::SkillResult;

/// The lifecycle state of a skillpack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillState {
    /// On disk, not loaded.
    Cold,
    /// Manifest loaded, ready to activate.
    Warm,
    /// Fully loaded and active in memory.
    Hot,
}

impl std::fmt::Display for SkillState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cold => write!(f, "Cold"),
            Self::Warm => write!(f, "Warm"),
            Self::Hot => write!(f, "Hot"),
        }
    }
}

/// Metadata describing a skillpack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique skillpack identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Version string.
    pub version: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Domain tags (e.g., "astronomy", "linguistics").
    pub domains: Vec<String>,
    /// Size of weight data in bytes (0 if no neural component).
    #[serde(default)]
    pub weight_size_bytes: u64,
    /// Path to triples JSON file relative to skill directory.
    #[serde(default)]
    pub triples_file: Option<String>,
    /// Path to rules file relative to skill directory.
    #[serde(default)]
    pub rules_file: Option<String>,
    /// Paths to CLI tool manifest JSON files relative to skill directory.
    #[serde(default)]
    pub cli_tools: Vec<String>,
    /// Paths to WASM component modules relative to skill directory.
    #[serde(default)]
    pub wasm_tools: Vec<String>,
    /// Key-value configuration passed to tools (e.g., API keys, base URLs).
    #[serde(default)]
    pub tool_config: HashMap<String, String>,

    // ── Phase 30: Procedural knowledge ──────────────────────────────────────
    /// Path to action schemas JSON file relative to skill directory (default: "actions.json").
    #[serde(default)]
    pub actions_file: Option<String>,
    /// Path to HTN plan templates JSON file relative to skill directory (default: "plans.json").
    #[serde(default)]
    pub plans_file: Option<String>,
    /// Path to natural language procedures source file relative to skill directory
    /// (default: "procedures.md"). This is the human-editable source of truth used
    /// for LLM-assisted re-generation (Phase 30c).
    #[serde(default)]
    pub procedures_file: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 30: Procedural knowledge types
// ─────────────────────────────────────────────────────────────────────────────

/// Effects that a skill action has on world state (event-calculus style).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SkillEffects {
    /// Fluent names initiated (started) by this action.
    #[serde(default)]
    pub initiates: Vec<String>,
    /// Fluent names terminated (ended) by this action.
    #[serde(default)]
    pub terminates: Vec<String>,
}

/// Estimated duration class for a skill action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ActionDuration {
    /// Completes within a single OODA cycle.
    #[default]
    Fast,
    /// Completes within a few cycles.
    Medium,
    /// Takes many cycles or involves external I/O.
    Slow,
}

/// A causal action schema stored in `actions.json`.
///
/// Maps to `CausalManager::register_action_schema()` when activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillAction {
    /// Action name — becomes the entity label in the KG (`skill:<skill_id>:<action>`).
    pub action: String,
    /// Human-readable purpose.
    #[serde(default)]
    pub description: String,
    /// Fluent names that must hold before this action is applicable.
    #[serde(default)]
    pub preconditions: Vec<String>,
    /// Fluents initiated/terminated by this action.
    #[serde(default)]
    pub effects: SkillEffects,
    /// Rough execution cost hint for planning.
    #[serde(default)]
    pub estimated_duration: ActionDuration,
}

/// One conditional branch within a `PlanMethod::Conditional` plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConditionalStep {
    /// Fluent to test at runtime.
    pub condition: String,
    /// Action name to execute when the fluent holds.
    pub then_step: String,
    /// Action name to execute when the fluent does not hold (optional).
    #[serde(default)]
    pub else_step: Option<String>,
}

/// How the steps within a plan template are organized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase", tag = "type")]
pub enum PlanMethod {
    /// Steps must execute in listed order.
    #[default]
    Sequential,
    /// Steps may execute concurrently.
    Parallel,
    /// Steps depend on runtime conditions.
    Conditional {
        /// The conditional branches.
        branches: Vec<ConditionalStep>,
    },
}

/// An HTN plan template stored in `plans.json`.
///
/// Maps to `MethodRegistry::register()` when activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPlan {
    /// Goal name — used to match incoming goal descriptions.
    pub goal: String,
    /// How steps within this plan are organized.
    #[serde(default)]
    pub method: PlanMethod,
    /// Ordered list of action names (references into `actions.json`).
    #[serde(default)]
    pub steps: Vec<String>,
    /// Fluent names that must hold for this plan template to apply.
    #[serde(default)]
    pub preconditions: Vec<String>,
    /// Names of alternative plan templates that achieve the same goal.
    #[serde(default)]
    pub alternatives: Vec<String>,
    /// Optional quality gate: fluent that must hold after the last step.
    #[serde(default)]
    pub quality_gate: Option<String>,
}

/// A skillpack with its manifest and current state.
#[derive(Debug, Clone)]
pub struct LoadedSkill {
    /// The parsed manifest.
    pub manifest: SkillManifest,
    /// Current lifecycle state.
    pub state: SkillState,
    /// Number of triples loaded from this skill.
    pub triple_count: usize,
    /// Number of rules loaded from this skill.
    pub rule_count: usize,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Loaded action schemas (populated when Hot; empty otherwise).
    pub actions: Vec<SkillAction>,
    /// Loaded plan templates (populated when Hot; empty otherwise).
    pub plans: Vec<SkillPlan>,
}

/// Summary info for listing skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInfo {
    /// Skill identifier.
    pub id: String,
    /// Skill name.
    pub name: String,
    /// Version.
    pub version: String,
    /// Description.
    pub description: String,
    /// Current state.
    pub state: SkillState,
    /// Domain tags.
    pub domains: Vec<String>,
    /// Triples loaded.
    pub triple_count: usize,
    /// Rules loaded.
    pub rule_count: usize,
    /// Number of action schemas loaded.
    pub action_count: usize,
    /// Number of plan templates loaded.
    pub plan_count: usize,
}

/// Result of activating a skillpack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillActivation {
    /// Skill identifier.
    pub skill_id: String,
    /// Number of triples loaded.
    pub triples_loaded: usize,
    /// Number of rules loaded.
    pub rules_loaded: usize,
    /// Memory used in bytes.
    pub memory_bytes: usize,
    /// Number of action schemas loaded (Phase 30).
    pub actions_loaded: usize,
    /// Number of plan templates loaded (Phase 30).
    pub plans_loaded: usize,
}

/// Payload for installing a skill remotely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInstallPayload {
    pub manifest: SkillManifest,
    pub triples: Vec<LabelTriple>,
    #[serde(default)]
    pub rules: String,
}

/// A single label-based triple for skill installation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default = "default_label_confidence")]
    pub confidence: f32,
}

fn default_label_confidence() -> f32 {
    1.0
}

/// Result type alias for skill operations.
pub type SkillResultType<T> = SkillResult<T>;
