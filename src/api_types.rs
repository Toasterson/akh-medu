//! Shared request/response types for the akh-medu HTTP API.
//!
//! These types are used by both `akhomed` server handlers and `AkhClient`
//! remote methods. Having them in the library crate ensures wire-format
//! compatibility between client and server.

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "client-only"))]
use crate::symbol::SymbolId;

// ── Seeds ─────────────────────────────────────────────────────────────────

/// A seed pack entry returned by `GET /seeds`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedPackInfo {
    pub id: String,
    pub version: String,
    pub description: String,
    pub source: String,
    pub triple_count: usize,
}

/// Status of a single seed pack for a workspace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedStatusEntry {
    pub id: String,
    pub applied: bool,
}

/// Response for `GET /workspaces/{ws}/seeds/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedStatusResponse {
    pub workspace: String,
    pub seeds: Vec<SeedStatusEntry>,
}

// ── Render ────────────────────────────────────────────────────────────────

/// Request for `POST /workspaces/{ws}/render`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderRequest {
    /// Entity name/ID to render (optional — if absent, render all).
    pub entity: Option<String>,
    /// Depth of subgraph to render.
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Render all triples (when entity is None).
    #[serde(default)]
    pub all: bool,
    /// Show glyph legend only.
    #[serde(default)]
    pub legend: bool,
}

fn default_depth() -> usize {
    1
}

/// Response for `POST /workspaces/{ws}/render`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResponse {
    /// The rendered output (terminal-ready text with optional ANSI).
    pub output: String,
}

// ── Agent run/resume ──────────────────────────────────────────────────────

/// Request for `POST /workspaces/{ws}/agent/run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunRequest {
    pub goals: Vec<String>,
    #[serde(default = "default_max_cycles")]
    pub max_cycles: usize,
    #[serde(default)]
    pub fresh: bool,
}

fn default_max_cycles() -> usize {
    10
}

/// Response for `POST /workspaces/{ws}/agent/run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunResponse {
    pub cycles_completed: usize,
    pub goals: Vec<GoalSummary>,
    pub overview: String,
}

/// Request for `POST /workspaces/{ws}/agent/resume`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResumeRequest {
    #[serde(default = "default_max_cycles")]
    pub max_cycles: usize,
}

/// Response for `POST /workspaces/{ws}/agent/resume`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResumeResponse {
    pub cycles_completed: usize,
    pub goals: Vec<GoalSummary>,
}

/// Summary of a single goal (used in multiple responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSummary {
    pub symbol_id: u64,
    pub label: String,
    pub status: String,
    pub description: String,
}

// ── PIM ───────────────────────────────────────────────────────────────────

/// Item in PIM task lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimTaskItem {
    pub symbol_id: u64,
    pub label: String,
    pub quadrant: String,
    pub gtd_state: Option<String>,
    pub energy: Option<String>,
    pub overdue_days: Option<u64>,
}

/// Request for `POST /workspaces/{ws}/pim/next`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimNextRequest {
    pub context: Option<String>,
    pub energy: Option<String>,
}

/// Response for `GET /workspaces/{ws}/pim/inbox` and similar list endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimTaskList {
    pub tasks: Vec<PimTaskItem>,
}

/// Request for `POST /workspaces/{ws}/pim/add`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimAddRequest {
    pub goal: u64,
    #[serde(default = "default_gtd")]
    pub gtd: String,
    #[serde(default = "default_half")]
    pub urgency: f32,
    #[serde(default = "default_half")]
    pub importance: f32,
    pub para: Option<String>,
    pub contexts: Option<Vec<String>>,
    pub recur: Option<String>,
    pub deadline: Option<u64>,
}

fn default_gtd() -> String {
    "inbox".into()
}

fn default_half() -> f32 {
    0.5
}

/// Response for `POST /workspaces/{ws}/pim/add`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimAddResponse {
    pub goal: u64,
    pub gtd_state: String,
    pub quadrant: String,
}

/// Request for `POST /workspaces/{ws}/pim/transition`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimTransitionRequest {
    pub goal: u64,
    pub to: String,
}

/// Response for `GET /workspaces/{ws}/pim/review`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimReviewResponse {
    pub summary: String,
    pub overdue: Vec<PimTaskItem>,
    pub stale_inbox: Vec<PimTaskItem>,
    pub stalled_projects: Vec<PimTaskItem>,
    pub adjustment_count: usize,
}

/// Response for `GET /workspaces/{ws}/pim/matrix`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimMatrixResponse {
    pub do_tasks: Vec<PimTaskItem>,
    pub schedule_tasks: Vec<PimTaskItem>,
    pub delegate_tasks: Vec<PimTaskItem>,
    pub eliminate_tasks: Vec<PimTaskItem>,
}

/// Response for `GET /workspaces/{ws}/pim/deps`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimDepsResponse {
    pub order: Vec<PimTaskItem>,
}

/// PIM project info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimProjectResponse {
    pub name: String,
    pub status: String,
    pub goals: Vec<PimTaskItem>,
}

// ── Causal ────────────────────────────────────────────────────────────────

/// Summary of a causal action schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalSchemaSummary {
    pub name: String,
    pub precondition_count: usize,
    pub effect_count: usize,
    pub success_rate: f32,
    pub execution_count: usize,
}

/// Detailed schema info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalSchemaDetail {
    pub name: String,
    pub action_id: u64,
    pub precondition_count: usize,
    pub effect_count: usize,
    pub success_rate: f32,
    pub execution_count: usize,
}

/// Request for `POST /workspaces/{ws}/causal/predict`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPredictRequest {
    pub name: String,
}

/// An element in a state transition prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// A confidence-change entry in a state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionConfidenceChange {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub delta: f32,
}

/// Response for `POST /workspaces/{ws}/causal/predict`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPredictResponse {
    pub assertions: Vec<TransitionTriple>,
    pub retractions: Vec<TransitionTriple>,
    pub confidence_changes: Vec<TransitionConfidenceChange>,
}

/// Response for `POST /workspaces/{ws}/causal/bootstrap`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalBootstrapResponse {
    pub schemas_created: usize,
    pub tools_scanned: usize,
}

// ── Pref ──────────────────────────────────────────────────────────────────

/// Response for `GET /workspaces/{ws}/pref/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefStatusResponse {
    pub interaction_count: usize,
    pub proactivity_level: String,
    pub decay_rate: f32,
    pub suggestions_offered: u64,
    pub suggestions_accepted: u64,
    pub acceptance_rate: f32,
    pub prototype_active: bool,
}

/// Request for `POST /workspaces/{ws}/pref/train`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefTrainRequest {
    pub entity: u64,
    #[serde(default = "default_one")]
    pub weight: f32,
}

fn default_one() -> f32 {
    1.0
}

/// Response for `POST /workspaces/{ws}/pref/train`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefTrainResponse {
    pub entity_label: String,
    pub weight: f32,
    pub total_interactions: usize,
}

/// Request for `PUT /workspaces/{ws}/pref/level`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefLevelRequest {
    pub level: String,
}

/// Response for `GET /workspaces/{ws}/pref/interests`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefInterest {
    pub label: String,
    pub similarity: f32,
}

// ── Calendar ──────────────────────────────────────────────────────────────

/// Calendar event summary for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalEventSummary {
    pub symbol_id: u64,
    pub summary: String,
    pub duration_minutes: u64,
    pub location: Option<String>,
}

/// Response for event list endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalEventList {
    pub events: Vec<CalEventSummary>,
}

/// Request for `POST /workspaces/{ws}/cal/add`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalAddRequest {
    pub summary: String,
    pub start: u64,
    pub end: u64,
    pub location: Option<String>,
}

/// Response for `POST /workspaces/{ws}/cal/add`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalAddResponse {
    pub symbol_id: u64,
    pub summary: String,
    pub duration_minutes: u64,
}

/// A scheduling conflict pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalConflict {
    pub event_a: String,
    pub event_b: String,
}

/// Request for `POST /workspaces/{ws}/cal/import`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalImportRequest {
    pub ical_data: String,
}

/// Response for `POST /workspaces/{ws}/cal/import`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalImportResponse {
    pub imported_count: usize,
}

/// Request for `POST /workspaces/{ws}/cal/sync`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalSyncRequest {
    pub url: String,
    pub user: String,
    pub pass: String,
}

// ── Ingest ───────────────────────────────────────────────────────────────

/// Request for `POST /workspaces/{ws}/ingest/csv`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvIngestRequest {
    /// CSV content as a string.
    pub content: String,
    /// Format: "spo" (default) or "entity".
    #[serde(default = "default_csv_format")]
    pub format: String,
}

fn default_csv_format() -> String {
    "spo".into()
}

/// Request for `POST /workspaces/{ws}/ingest/text`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextIngestRequest {
    /// Natural language text to extract triples from.
    pub text: String,
    /// Maximum sentences to process (default: 100).
    #[serde(default = "default_max_sentences")]
    pub max_sentences: usize,
}

fn default_max_sentences() -> usize {
    100
}

/// Generic ingest response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub success: bool,
    pub message: String,
}

/// Request for `POST /workspaces/{ws}/library/scan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryScanRequest {
    /// Optional inbox directory override. Uses default if not provided.
    pub inbox_dir: Option<String>,
}

/// Response for `POST /workspaces/{ws}/library/scan`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryScanResponse {
    pub files_processed: usize,
    pub files_failed: usize,
}

// ── Awaken ────────────────────────────────────────────────────────────────

/// Request for `POST /workspaces/{ws}/awaken/parse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenParseRequest {
    pub statement: String,
}

/// Response for `POST /workspaces/{ws}/awaken/parse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenParseResponse {
    pub domain: String,
    pub competence_level: String,
    pub seed_concepts: Vec<String>,
    pub identity_name: Option<String>,
    pub identity_type: Option<String>,
}

/// Request for `POST /workspaces/{ws}/awaken/resolve`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenResolveRequest {
    pub name: String,
}

/// Response for `POST /workspaces/{ws}/awaken/resolve`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenResolveResponse {
    pub name: String,
    pub entity_type: String,
    pub culture: String,
    pub description: String,
    pub domains: Vec<String>,
    pub traits: Vec<String>,
    pub archetypes: Vec<String>,
    pub chosen_name: Option<String>,
    pub persona: Option<String>,
}

/// Request for `POST /workspaces/{ws}/awaken/expand`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenExpandRequest {
    pub seeds: Option<Vec<String>>,
    pub purpose: Option<String>,
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_max_concepts")]
    pub max_concepts: usize,
    #[serde(default)]
    pub no_conceptnet: bool,
}

fn default_threshold() -> f32 {
    0.6
}

fn default_max_concepts() -> usize {
    200
}

/// Response for `POST /workspaces/{ws}/awaken/expand`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenExpandResponse {
    pub concept_count: usize,
    pub relation_count: usize,
    pub rejected_count: usize,
    pub api_calls: usize,
    pub accepted_labels: Vec<String>,
}

/// Request for `POST /workspaces/{ws}/awaken/prerequisite`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenPrerequisiteRequest {
    pub seeds: Option<Vec<String>>,
    pub purpose: Option<String>,
    #[serde(default = "default_known_threshold")]
    pub known_threshold: usize,
    #[serde(default = "default_zpd_low")]
    pub zpd_low: f32,
    #[serde(default = "default_zpd_high")]
    pub zpd_high: f32,
}

fn default_known_threshold() -> usize {
    5
}

fn default_zpd_low() -> f32 {
    0.3
}

fn default_zpd_high() -> f32 {
    0.7
}

/// Response for `POST /workspaces/{ws}/awaken/prerequisite`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenPrerequisiteResponse {
    pub concepts_analyzed: usize,
    pub edge_count: usize,
    pub cycles_broken: usize,
    pub max_tier: usize,
    pub zone_distribution: Vec<(String, usize)>,
    pub curriculum: Vec<CurriculumEntry>,
}

/// A single curriculum entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumEntry {
    pub tier: usize,
    pub zone: String,
    pub label: String,
    pub prereq_coverage: f32,
    pub similarity_to_known: f32,
}

/// Request for `POST /workspaces/{ws}/awaken/resources`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenResourcesRequest {
    pub seeds: Option<Vec<String>>,
    pub purpose: Option<String>,
    #[serde(default = "default_min_quality")]
    pub min_quality: f32,
    #[serde(default = "default_max_api_calls")]
    pub max_api_calls: usize,
    #[serde(default)]
    pub no_semantic_scholar: bool,
    #[serde(default)]
    pub no_openalex: bool,
    #[serde(default)]
    pub no_open_library: bool,
}

fn default_min_quality() -> f32 {
    0.2
}

fn default_max_api_calls() -> usize {
    60
}

/// Response for `POST /workspaces/{ws}/awaken/resources`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenResourcesResponse {
    pub resources_discovered: usize,
    pub api_calls_used: usize,
    pub concepts_covered: usize,
}

/// Request for `POST /workspaces/{ws}/awaken/ingest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenIngestRequest {
    pub seeds: Option<Vec<String>>,
    pub purpose: Option<String>,
    #[serde(default = "default_ingest_cycles")]
    pub max_cycles: usize,
    #[serde(default = "default_saturation")]
    pub saturation: usize,
    #[serde(default = "default_xval_boost")]
    pub xval_boost: f32,
    #[serde(default)]
    pub no_url: bool,
    pub catalog_dir: Option<String>,
}

fn default_ingest_cycles() -> usize {
    500
}

fn default_saturation() -> usize {
    3
}

fn default_xval_boost() -> f32 {
    0.15
}

/// Response for `POST /workspaces/{ws}/awaken/ingest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenIngestResponse {
    pub triples_added: usize,
    pub concepts_covered: usize,
    pub cycles_used: usize,
}

/// Request for `POST /workspaces/{ws}/awaken/assess`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenAssessRequest {
    pub seeds: Option<Vec<String>>,
    pub purpose: Option<String>,
    #[serde(default = "default_min_triples")]
    pub min_triples: usize,
    #[serde(default = "default_bloom_depth")]
    pub bloom_depth: usize,
    #[serde(default)]
    pub verbose: bool,
}

fn default_min_triples() -> usize {
    3
}

fn default_bloom_depth() -> usize {
    4
}

/// Response for `POST /workspaces/{ws}/awaken/assess`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenAssessResponse {
    pub overall_dreyfus: String,
    pub overall_score: f32,
    pub recommendation: String,
    pub knowledge_areas: Vec<KnowledgeAreaSummary>,
}

/// Summary of a knowledge area assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeAreaSummary {
    pub name: String,
    pub dreyfus_level: String,
    pub score: f32,
}

/// Request for `POST /workspaces/{ws}/awaken/bootstrap`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenBootstrapRequest {
    pub statement: Option<String>,
    #[serde(default)]
    pub plan_only: bool,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub status: bool,
    #[serde(default = "default_max_cycles")]
    pub max_cycles: usize,
    pub identity: Option<String>,
}

/// Response for `POST /workspaces/{ws}/awaken/bootstrap`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwakenBootstrapResponse {
    pub domain: String,
    pub target_level: String,
    pub chosen_name: Option<String>,
    pub learning_cycles: usize,
    pub target_reached: bool,
    pub final_dreyfus: Option<String>,
    pub final_score: Option<f32>,
    pub recommendation: Option<String>,
}

// ── Workspace ─────────────────────────────────────────────────────────────

/// Response for workspace creation via `POST /workspaces/{name}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCreateResponse {
    pub name: String,
    pub created: bool,
}

/// Response for workspace deletion via `DELETE /workspaces/{name}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceDeleteResponse {
    pub deleted: String,
}

/// Request for `POST /workspaces/{name}` with optional role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCreateRequest {
    pub role: Option<String>,
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Parse a GTD state string to the internal enum.
pub fn parse_gtd_state(s: &str) -> Option<crate::agent::GtdState> {
    match s.to_lowercase().as_str() {
        "inbox" => Some(crate::agent::GtdState::Inbox),
        "next" => Some(crate::agent::GtdState::Next),
        "waiting" => Some(crate::agent::GtdState::Waiting),
        "someday" => Some(crate::agent::GtdState::Someday),
        "reference" => Some(crate::agent::GtdState::Reference),
        "done" => Some(crate::agent::GtdState::Done),
        _ => None,
    }
}

/// Parse an energy level string.
pub fn parse_energy_level(s: &str) -> Option<crate::agent::EnergyLevel> {
    match s.to_lowercase().as_str() {
        "low" => Some(crate::agent::EnergyLevel::Low),
        "medium" => Some(crate::agent::EnergyLevel::Medium),
        "high" => Some(crate::agent::EnergyLevel::High),
        _ => None,
    }
}

/// Parse a proactivity level string.
pub fn parse_proactivity_level(s: &str) -> Option<crate::agent::ProactivityLevel> {
    match s.to_lowercase().as_str() {
        "ambient" => Some(crate::agent::ProactivityLevel::Ambient),
        "nudge" => Some(crate::agent::ProactivityLevel::Nudge),
        "offer" => Some(crate::agent::ProactivityLevel::Offer),
        "scheduled" => Some(crate::agent::ProactivityLevel::Scheduled),
        "autonomous" => Some(crate::agent::ProactivityLevel::Autonomous),
        _ => None,
    }
}

/// Helper: build a `PimTaskItem` from a SymbolId.
///
/// Returns `None` for callers without an engine (client-only mode).
#[cfg(not(feature = "client-only"))]
pub fn pim_task_item(
    engine: &crate::engine::Engine,
    agent: &crate::agent::Agent,
    id: SymbolId,
) -> PimTaskItem {
    let label = engine.resolve_label(id);
    let meta = agent.pim_manager().get_metadata(id.get());
    PimTaskItem {
        symbol_id: id.get(),
        label,
        quadrant: meta
            .map(|m| m.quadrant.to_string())
            .unwrap_or_default(),
        gtd_state: meta.map(|m| m.gtd_state.to_string()),
        energy: meta.and_then(|m| m.energy).map(|e| format!("{e}")),
        overdue_days: None,
    }
}
