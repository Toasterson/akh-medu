//! Stateful synthetic curricula and exams for Akh-native learning workflows.
//!
//! These are not one-shot trivia prompts. An episode makes the model check
//! memory, diagnose a gap, acquire or retrieve textbook evidence, preserve
//! provenance in an append-only note, and only then answer. Separate scenarios
//! exercise restart recall, edition conflicts, broken providers, and grounded
//! abstention. Every generated call is checked against a content-addressed
//! runtime tool snapshot before it can enter a corpus.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::adaptive_exam::{
    CapabilitySpec, CheckpointRef, ExamObservation, ExamPlan, ExamTier, TrainingSiblingPlan,
};
use crate::tool_contract::{SchemaIssue, ToolContractSnapshot};

pub const KNOWLEDGE_WORKFLOW_SCHEMA_VERSION: u32 = 1;
pub const KNOWLEDGE_WORKFLOW_GENERATOR_VERSION: &str = "akh-knowledge-workflow-v2";

pub const ACQUIRE_FAMILY: &str = "knowledge.acquire_textbook";
pub const RESTART_FAMILY: &str = "knowledge.recall_after_restart";
pub const REVISION_FAMILY: &str = "knowledge.revise_from_new_edition";
pub const FALLBACK_FAMILY: &str = "knowledge.provider_fallback";
pub const ABSTAIN_FAMILY: &str = "knowledge.grounded_abstention";
pub const TEXTBOOK_INVENTORY_FAMILY: &str = "knowledge.textbook_inventory";
pub const NOTE_NAVIGATION_FAMILY: &str = "knowledge.note_navigation";

/// Capability matrix for the adaptive exam controller. The lanes trend
/// independently so an average cannot hide collapsing citation or fallback
/// behavior.
pub fn knowledge_capability_specs() -> Vec<CapabilitySpec> {
    [
        ACQUIRE_FAMILY,
        RESTART_FAMILY,
        REVISION_FAMILY,
        FALLBACK_FAMILY,
        ABSTAIN_FAMILY,
        TEXTBOOK_INVENTORY_FAMILY,
        NOTE_NAVIGATION_FAMILY,
    ]
    .into_iter()
    .map(|id| CapabilitySpec::new(id, 0, 5).with_initial_difficulty(1))
    .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowLanguage {
    English,
    German,
    French,
}

impl WorkflowLanguage {
    pub const fn all() -> [Self; 3] {
        [Self::English, Self::German, Self::French]
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::English => "english",
            Self::German => "german",
            Self::French => "french",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::English => "en",
            Self::German => "de",
            Self::French => "fr",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::English => Self::German,
            Self::German => Self::French,
            Self::French => Self::English,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeScenario {
    AcquireTextbook,
    RecallAfterRestart,
    ReviseFromNewEdition,
    ProviderFallback,
    GroundedAbstention,
    TextbookInventory,
    NoteNavigation,
}

impl KnowledgeScenario {
    pub fn family(self) -> &'static str {
        match self {
            Self::AcquireTextbook => ACQUIRE_FAMILY,
            Self::RecallAfterRestart => RESTART_FAMILY,
            Self::ReviseFromNewEdition => REVISION_FAMILY,
            Self::ProviderFallback => FALLBACK_FAMILY,
            Self::GroundedAbstention => ABSTAIN_FAMILY,
            Self::TextbookInventory => TEXTBOOK_INVENTORY_FAMILY,
            Self::NoteNavigation => NOTE_NAVIGATION_FAMILY,
        }
    }

    fn all() -> [Self; 7] {
        [
            Self::AcquireTextbook,
            Self::RecallAfterRestart,
            Self::ReviseFromNewEdition,
            Self::ProviderFallback,
            Self::GroundedAbstention,
            Self::TextbookInventory,
            Self::NoteNavigation,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStage {
    CheckMemory,
    DiagnoseGap,
    ListTextbooks,
    ListNotes,
    ReadNote,
    SearchCorpus,
    SearchProvider,
    IngestTextbook,
    EmbedCorpus,
    RetrieveEvidence,
    DistillNote,
    ReviseNote,
    Answer,
    DeclareGap,
}

impl WorkflowStage {
    pub const fn all() -> [Self; 14] {
        [
            Self::CheckMemory,
            Self::DiagnoseGap,
            Self::ListTextbooks,
            Self::ListNotes,
            Self::ReadNote,
            Self::SearchCorpus,
            Self::SearchProvider,
            Self::IngestTextbook,
            Self::EmbedCorpus,
            Self::RetrieveEvidence,
            Self::DistillNote,
            Self::ReviseNote,
            Self::Answer,
            Self::DeclareGap,
        ]
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CheckMemory => "check_memory",
            Self::DiagnoseGap => "diagnose_gap",
            Self::ListTextbooks => "list_textbooks",
            Self::ListNotes => "list_notes",
            Self::ReadNote => "read_note",
            Self::SearchCorpus => "search_corpus",
            Self::SearchProvider => "search_provider",
            Self::IngestTextbook => "ingest_textbook",
            Self::EmbedCorpus => "embed_corpus",
            Self::RetrieveEvidence => "retrieve_evidence",
            Self::DistillNote => "distill_note",
            Self::ReviseNote => "revise_note",
            Self::Answer => "answer",
            Self::DeclareGap => "declare_gap",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceState {
    Unknown,
    Unavailable,
    Verified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowCitation {
    pub source_id: String,
    pub locator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KnowledgeAction {
    ToolCall {
        tool: String,
        arguments: Value,
    },
    Answer {
        text: String,
        citations: Vec<WorkflowCitation>,
    },
    Gap {
        missing: Vec<String>,
        reason: String,
    },
}

impl KnowledgeAction {
    /// Parse the tagged representation, OpenAI-style name/arguments, compact
    /// answer/gap JSON, or a plain uncited answer.
    pub fn parse(raw: &str) -> Result<Self, KnowledgeWorkflowError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(KnowledgeWorkflowError::InvalidPrediction {
                message: "prediction is blank".into(),
            });
        }
        if let Some(start) = trimmed.find("<tool_call>") {
            let body = &trimmed[start + "<tool_call>".len()..];
            let body = body
                .split_once("</tool_call>")
                .map(|(body, _)| body)
                .unwrap_or(body)
                .trim();
            let value: Value = serde_json::from_str(body).map_err(|error| {
                KnowledgeWorkflowError::InvalidPrediction {
                    message: format!("invalid tool-call block: {error}"),
                }
            })?;
            let name = value.get("name").and_then(Value::as_str).ok_or_else(|| {
                KnowledgeWorkflowError::InvalidPrediction {
                    message: "tool-call block has no name".into(),
                }
            })?;
            return Ok(Self::ToolCall {
                tool: name.into(),
                arguments: value
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
            });
        }
        if trimmed
            .get(.."Knowledge gap:".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("Knowledge gap:"))
        {
            let rest = trimmed["Knowledge gap:".len()..].trim();
            let (missing, reason) = rest
                .split_once("\nReason:")
                .map(|(missing, reason)| (missing, reason))
                .unwrap_or((rest, "No verified evidence is available."));
            return Ok(Self::Gap {
                missing: missing
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(String::from)
                    .collect(),
                reason: reason.trim().into(),
            });
        }
        let value: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(_) => {
                return Ok(parse_plain_answer(trimmed));
            }
        };
        if let Ok(action) = serde_json::from_value(value.clone()) {
            return Ok(action);
        }
        if let Some(name) = value.get("name").and_then(Value::as_str) {
            return Ok(Self::ToolCall {
                tool: name.into(),
                arguments: value
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
            });
        }
        if let Some(answer) = value.get("answer").and_then(Value::as_str) {
            let citations = value
                .get("citations")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| KnowledgeWorkflowError::InvalidPrediction {
                    message: format!("invalid answer citations: {error}"),
                })?
                .unwrap_or_default();
            return Ok(Self::Answer {
                text: answer.into(),
                citations,
            });
        }
        if value.get("gap").and_then(Value::as_bool) == Some(true) {
            let missing = value
                .get("missing")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| KnowledgeWorkflowError::InvalidPrediction {
                    message: format!("invalid gap list: {error}"),
                })?
                .unwrap_or_default();
            return Ok(Self::Gap {
                missing,
                reason: value
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("evidence unavailable")
                    .into(),
            });
        }
        Err(KnowledgeWorkflowError::InvalidPrediction {
            message: "JSON is not a recognized knowledge action".into(),
        })
    }

    /// Parse exactly the wire shape accepted by Akh-Medu's BabyLLM runtime.
    ///
    /// [`Self::parse`] remains intentionally liberal for importing historical
    /// predictions. Release exams and serving must use this method so a model
    /// cannot pass evaluation with an envelope the daemon would reject.
    pub fn parse_runtime(raw: &str) -> Result<Self, KnowledgeWorkflowError> {
        let action = Self::parse(raw)?;
        match &action {
            Self::ToolCall { .. } if !strict_single_tool_envelope(raw) => {
                Err(KnowledgeWorkflowError::InvalidPrediction {
                    message: "tool call is not one unambiguous tagged envelope".into(),
                })
            }
            Self::Answer { text, .. }
                if raw.contains("<tool_call>")
                    || raw.contains("</tool_call>")
                    || !valid_runtime_text(text) =>
            {
                Err(KnowledgeWorkflowError::InvalidPrediction {
                    message: "answer is empty, malformed, or mixed with a tool call".into(),
                })
            }
            Self::Gap { missing, reason }
                if raw.contains("<tool_call>")
                    || raw.contains("</tool_call>")
                    || missing.is_empty()
                    || !valid_runtime_text(reason) =>
            {
                Err(KnowledgeWorkflowError::InvalidPrediction {
                    message: "knowledge gap is empty, malformed, or mixed with a tool call".into(),
                })
            }
            _ => Ok(action),
        }
    }

    /// Render exactly what Akh's Candle backend consumes: tool calls use its
    /// tool-call tags; final text remains ordinary assistant content with
    /// machine-readable source lines.
    pub fn render_runtime(&self) -> Result<String, KnowledgeWorkflowError> {
        match self {
            Self::ToolCall { tool, arguments } => {
                let call = serde_json::to_string(&serde_json::json!({
                    "name": tool,
                    "arguments": arguments
                }))
                .map_err(|error| KnowledgeWorkflowError::Serialize {
                    message: error.to_string(),
                })?;
                Ok(format!("<tool_call>\n{call}\n</tool_call>"))
            }
            Self::Answer { text, citations } => {
                let mut output = text.clone();
                if !citations.is_empty() {
                    output.push_str("\n\nSources:");
                    for citation in citations {
                        output
                            .push_str(&format!("\n- {}#{}", citation.source_id, citation.locator));
                    }
                }
                Ok(output)
            }
            Self::Gap { missing, reason } => Ok(format!(
                "Knowledge gap: {}\nReason: {reason}",
                missing.join(", ")
            )),
        }
    }
}

fn strict_single_tool_envelope(raw: &str) -> bool {
    if raw.matches("<tool_call>").count() != 1 || raw.matches("</tool_call>").count() != 1 {
        return false;
    }
    let Some((prefix, rest)) = raw.split_once("<tool_call>") else {
        return false;
    };
    let Some((_body, suffix)) = rest.split_once("</tool_call>") else {
        return false;
    };
    prefix.trim().is_empty() && suffix.trim().is_empty()
}

fn valid_runtime_text(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
        && !text.contains("<|im_start|>")
        && !text.contains("<|im_end|>")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeStep {
    pub stage: WorkflowStage,
    /// User or tool observation immediately before the expected action.
    pub input: String,
    pub evidence_state: EvidenceState,
    pub expected: KnowledgeAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mock_result: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextbookFixture {
    pub concept: String,
    pub attribute: String,
    pub answer: String,
    pub prior_answer: String,
    pub source_id: String,
    pub prior_source_id: String,
    pub title: String,
    pub edition: u8,
    pub locator: String,
    pub source_language: WorkflowLanguage,
    pub passage: String,
    pub target_sentence: String,
    pub question: String,
    pub note_id: String,
    /// Head token for the synthetic edition-1 note used by safe revision
    /// exercises.
    pub prior_revision: String,
    /// Head token for the synthetic current note used by browse/read exercises
    /// and mock write receipts.
    pub current_revision: String,
    /// Intervening head token injected into high-difficulty revision episodes
    /// to teach conflict recovery.
    pub concurrent_revision: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeEpisode {
    pub schema_version: u32,
    pub id: String,
    pub contract_hash: String,
    pub scenario: KnowledgeScenario,
    pub family: String,
    pub difficulty: u8,
    pub language: WorkflowLanguage,
    pub fixture: TextbookFixture,
    pub steps: Vec<KnowledgeStep>,
}

impl KnowledgeEpisode {
    pub fn to_jsonl(&self) -> Result<String, KnowledgeWorkflowError> {
        serde_json::to_string(self).map_err(|error| KnowledgeWorkflowError::Serialize {
            message: error.to_string(),
        })
    }

    /// Expand a multi-step episode into target-masked SFT records. Each record
    /// carries the exact conversation state available before one model action.
    pub fn to_sft_records(
        &self,
        contracts: &ToolContractSnapshot,
        bindings: &KnowledgeToolBindings,
    ) -> Result<Vec<KnowledgeSftRecord>, KnowledgeWorkflowError> {
        if self.contract_hash != contracts.manifest_hash {
            return Err(KnowledgeWorkflowError::ContractMismatch {
                episode: self.contract_hash.clone(),
                loaded: contracts.manifest_hash.clone(),
            });
        }
        let mut conversation = vec![serde_json::json!({
            "role": "user",
            "content": self.steps.first().map(|step| step.input.as_str()).unwrap_or("")
        })];
        let mut records = Vec::with_capacity(self.steps.len());
        for (index, step) in self.steps.iter().enumerate() {
            let tool_names = offered_tools(index, step, bindings);
            let tools: Vec<Value> = tool_names
                .iter()
                .map(|name| {
                    let contract = contracts.tool(name).ok_or_else(|| {
                        KnowledgeWorkflowError::MissingToolBinding {
                            name: name.clone(),
                            contract_hash: contracts.manifest_hash.clone(),
                        }
                    })?;
                    Ok(serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": contract.name,
                            "description": contract.description,
                            "parameters": contract.input_schema()
                        }
                    }))
                })
                .collect::<Result<_, KnowledgeWorkflowError>>()?;
            let prompt = render_knowledge_prompt(&tools, &conversation);
            let target = step.expected.render_runtime()?;
            records.push(KnowledgeSftRecord {
                schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
                id: format!("{}-step-{index:02}", self.id),
                episode_id: self.id.clone(),
                step: index,
                stage: step.stage,
                family: self.family.clone(),
                difficulty: self.difficulty,
                language: self.language,
                contract_hash: self.contract_hash.clone(),
                tools: tools.clone(),
                conversation: conversation.clone(),
                system_context: None,
                prompt,
                target: target.clone(),
            });

            match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    conversation.push(serde_json::json!({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "function": {"name": tool, "arguments": arguments}
                        }]
                    }));
                    conversation.push(serde_json::json!({
                        "role": "tool",
                        "content": step.mock_result.as_deref().unwrap_or("")
                    }));
                }
                KnowledgeAction::Answer { .. } | KnowledgeAction::Gap { .. } => break,
            }
        }
        Ok(records)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeSftRecord {
    pub schema_version: u32,
    pub id: String,
    pub episode_id: String,
    pub step: usize,
    pub stage: WorkflowStage,
    pub family: String,
    pub difficulty: u8,
    pub language: WorkflowLanguage,
    pub contract_hash: String,
    /// OpenAI-compatible tool definitions used to render the prompt.
    pub tools: Vec<Value>,
    pub conversation: Vec<Value>,
    /// Optional psyche/operator context inserted into the canonical system
    /// turn. Dialogue/character SFT uses it; pure workflow records omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_context: Option<String>,
    /// Canonical flat prompt accepted by grug-train's target-masked encoder.
    pub prompt: String,
    /// Runtime-native assistant output.
    pub target: String,
}

impl KnowledgeSftRecord {
    pub fn to_jsonl(&self) -> Result<String, KnowledgeWorkflowError> {
        serde_json::to_string(self).map_err(|error| KnowledgeWorkflowError::Serialize {
            message: error.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeTrainingCorpus {
    pub schema_version: u32,
    pub generator_version: String,
    pub seed: u64,
    pub contract_hash: String,
    /// Present only for a separately-derived frontier sibling. The exact exam
    /// fingerprint remains quarantine metadata, never training text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_exam_fingerprint: Option<String>,
    pub episodes: Vec<KnowledgeEpisode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeExamSet {
    pub schema_version: u32,
    pub generator_version: String,
    pub contract_hash: String,
    pub plan: ExamPlan,
    pub exam_fingerprint: String,
    pub episodes: Vec<KnowledgeEpisode>,
}

/// Runtime names used by the current Akh and Seshat learning loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeToolBindings {
    pub memory_recall: String,
    pub gap_analysis: String,
    pub corpus_search: String,
    pub corpus_documents: String,
    pub corpus_ingest: String,
    pub corpus_embed: String,
    pub archival_search: String,
    pub study_note_append: String,
    pub study_note_revise: String,
    pub study_note_list: String,
    pub study_note_read: String,
    pub provider_search: String,
}

impl Default for KnowledgeToolBindings {
    fn default() -> Self {
        Self {
            memory_recall: "memory_recall".into(),
            gap_analysis: "gap_analysis".into(),
            corpus_search: "mcp__seshat__seshat_search".into(),
            corpus_documents: "mcp__seshat__seshat_documents".into(),
            corpus_ingest: "mcp__seshat__seshat_ingest".into(),
            corpus_embed: "mcp__seshat__seshat_embed_now".into(),
            archival_search: "archival_search".into(),
            study_note_append: "study_note_append".into(),
            study_note_revise: "study_note_revise".into(),
            study_note_list: "study_note_list".into(),
            study_note_read: "study_note_read".into(),
            provider_search: "web_search".into(),
        }
    }
}

impl KnowledgeToolBindings {
    fn names(&self) -> [&str; 12] {
        [
            &self.memory_recall,
            &self.gap_analysis,
            &self.corpus_search,
            &self.corpus_documents,
            &self.corpus_ingest,
            &self.corpus_embed,
            &self.archival_search,
            &self.study_note_append,
            &self.study_note_revise,
            &self.study_note_list,
            &self.study_note_read,
            &self.provider_search,
        ]
    }
}

#[derive(Debug, Error)]
pub enum KnowledgeWorkflowError {
    #[error("tool binding {name:?} is absent from contract snapshot {contract_hash}")]
    MissingToolBinding { name: String, contract_hash: String },
    #[error("adaptive family {family:?} is not a knowledge-workflow family")]
    UnknownFamily { family: String },
    #[error("generated call to {tool:?} violates the runtime schema: {issues:?}")]
    GeneratedCallInvalid {
        tool: String,
        issues: Vec<SchemaIssue>,
    },
    #[error("episode contract {episode} does not match loaded contract {loaded}")]
    ContractMismatch { episode: String, loaded: String },
    #[error("could not serialize knowledge workflow: {message}")]
    Serialize { message: String },
    #[error("invalid model prediction: {message}")]
    InvalidPrediction { message: String },
}

pub struct KnowledgeWorkflowGenerator {
    contracts: ToolContractSnapshot,
    bindings: KnowledgeToolBindings,
}

impl KnowledgeWorkflowGenerator {
    pub fn new(
        contracts: ToolContractSnapshot,
        bindings: KnowledgeToolBindings,
    ) -> Result<Self, KnowledgeWorkflowError> {
        for name in bindings.names() {
            if contracts.tool(name).is_none() {
                return Err(KnowledgeWorkflowError::MissingToolBinding {
                    name: name.into(),
                    contract_hash: contracts.manifest_hash.clone(),
                });
            }
        }
        Ok(Self {
            contracts,
            bindings,
        })
    }

    pub fn contracts(&self) -> &ToolContractSnapshot {
        &self.contracts
    }

    pub fn generate_training(
        &self,
        seed: u64,
        episode_count: usize,
        max_difficulty: u8,
    ) -> Result<KnowledgeTrainingCorpus, KnowledgeWorkflowError> {
        let seed = stable_seed(&["knowledge-training", &seed.to_string()]);
        let mut episodes = Vec::with_capacity(episode_count);
        for index in 0..episode_count {
            let scenario = KnowledgeScenario::all()[index % KnowledgeScenario::all().len()];
            let language = language_for(index);
            let difficulty = (index as u8) % max_difficulty.min(5).saturating_add(1);
            episodes
                .push(self.generate_episode(seed, index, scenario, language, difficulty, "train")?);
        }
        Ok(KnowledgeTrainingCorpus {
            schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            generator_version: KNOWLEDGE_WORKFLOW_GENERATOR_VERSION.into(),
            seed,
            contract_hash: self.contracts.manifest_hash.clone(),
            source_exam_fingerprint: None,
            episodes,
        })
    }

    /// Materialize an exact adaptive exam. Callers must quarantine the result.
    pub fn generate_exam(
        &self,
        plan: &ExamPlan,
    ) -> Result<KnowledgeExamSet, KnowledgeWorkflowError> {
        let scenario = scenario_from_family(&plan.family)?;
        let mut episodes = Vec::with_capacity(plan.sample_count as usize);
        for index in 0..plan.sample_count as usize {
            episodes.push(self.generate_episode(
                plan.seed,
                index,
                scenario,
                language_for(index),
                plan.difficulty,
                tier_label(plan.tier),
            )?);
        }
        Ok(KnowledgeExamSet {
            schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            generator_version: KNOWLEDGE_WORKFLOW_GENERATOR_VERSION.into(),
            contract_hash: self.contracts.manifest_hash.clone(),
            plan: plan.clone(),
            exam_fingerprint: plan.fingerprint(),
            episodes,
        })
    }

    /// Generate a separately namespaced frontier training sibling. The caller
    /// still applies the controller's separation policy before admission.
    pub fn generate_training_sibling(
        &self,
        plan: &TrainingSiblingPlan,
    ) -> Result<KnowledgeTrainingCorpus, KnowledgeWorkflowError> {
        let scenario = scenario_from_family(&plan.family)?;
        let mut episodes = Vec::with_capacity(plan.sample_count as usize);
        for index in 0..plan.sample_count as usize {
            episodes.push(self.generate_episode(
                plan.seed,
                index,
                scenario,
                language_for(index + 1),
                plan.difficulty,
                "training-sibling",
            )?);
        }
        Ok(KnowledgeTrainingCorpus {
            schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            generator_version: KNOWLEDGE_WORKFLOW_GENERATOR_VERSION.into(),
            seed: plan.seed,
            contract_hash: self.contracts.manifest_hash.clone(),
            source_exam_fingerprint: Some(plan.source_exam_fingerprint.clone()),
            episodes,
        })
    }

    fn generate_episode(
        &self,
        seed: u64,
        index: usize,
        scenario: KnowledgeScenario,
        language: WorkflowLanguage,
        difficulty: u8,
        partition: &str,
    ) -> Result<KnowledgeEpisode, KnowledgeWorkflowError> {
        let episode_seed = stable_seed(&[
            KNOWLEDGE_WORKFLOW_GENERATOR_VERSION,
            &seed.to_string(),
            &index.to_string(),
            scenario.family(),
            language.code(),
            &difficulty.to_string(),
            partition,
        ]);
        let fixture = make_fixture(episode_seed, language, difficulty);
        let steps = self.steps_for(scenario, language, difficulty, &fixture);
        for step in &steps {
            if let KnowledgeAction::ToolCall { tool, arguments } = &step.expected {
                let validation = self.contracts.validate_call(tool, arguments);
                if !validation.valid {
                    return Err(KnowledgeWorkflowError::GeneratedCallInvalid {
                        tool: tool.clone(),
                        issues: validation.issues,
                    });
                }
            }
        }
        Ok(KnowledgeEpisode {
            schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            id: format!("kw-{partition}-{episode_seed:016x}-{index:06}"),
            contract_hash: self.contracts.manifest_hash.clone(),
            scenario,
            family: scenario.family().into(),
            difficulty,
            language,
            fixture,
            steps,
        })
    }

    fn steps_for(
        &self,
        scenario: KnowledgeScenario,
        language: WorkflowLanguage,
        difficulty: u8,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        match scenario {
            KnowledgeScenario::AcquireTextbook => self.acquire_steps(language, fixture),
            KnowledgeScenario::RecallAfterRestart => self.restart_steps(language, fixture),
            KnowledgeScenario::ReviseFromNewEdition => {
                self.revision_steps(language, difficulty, fixture)
            }
            KnowledgeScenario::ProviderFallback => self.fallback_steps(language, fixture),
            KnowledgeScenario::GroundedAbstention => self.abstention_steps(language, fixture),
            KnowledgeScenario::TextbookInventory => {
                self.textbook_inventory_steps(language, fixture)
            }
            KnowledgeScenario::NoteNavigation => self.note_navigation_steps(language, fixture),
        }
    }
}

fn scenario_from_family(family: &str) -> Result<KnowledgeScenario, KnowledgeWorkflowError> {
    match family {
        ACQUIRE_FAMILY => Ok(KnowledgeScenario::AcquireTextbook),
        RESTART_FAMILY => Ok(KnowledgeScenario::RecallAfterRestart),
        REVISION_FAMILY => Ok(KnowledgeScenario::ReviseFromNewEdition),
        FALLBACK_FAMILY => Ok(KnowledgeScenario::ProviderFallback),
        ABSTAIN_FAMILY => Ok(KnowledgeScenario::GroundedAbstention),
        TEXTBOOK_INVENTORY_FAMILY => Ok(KnowledgeScenario::TextbookInventory),
        NOTE_NAVIGATION_FAMILY => Ok(KnowledgeScenario::NoteNavigation),
        _ => Err(KnowledgeWorkflowError::UnknownFamily {
            family: family.into(),
        }),
    }
}

fn tier_label(tier: ExamTier) -> &'static str {
    match tier {
        ExamTier::Sentinel => "sentinel",
        ExamTier::RollingProbe => "rolling-probe",
        ExamTier::Frontier => "frontier",
        ExamTier::Audit => "audit",
    }
}

fn language_for(index: usize) -> WorkflowLanguage {
    match index % 3 {
        0 => WorkflowLanguage::English,
        1 => WorkflowLanguage::German,
        _ => WorkflowLanguage::French,
    }
}

fn parse_plain_answer(raw: &str) -> KnowledgeAction {
    let (text, sources) = raw
        .split_once("\n\nSources:")
        .map(|(text, sources)| (text.trim(), Some(sources)))
        .unwrap_or((raw.trim(), None));
    let citations = sources
        .into_iter()
        .flat_map(str::lines)
        .filter_map(|line| line.trim().strip_prefix("- "))
        .filter_map(|line| line.split_once('#'))
        .map(|(source_id, locator)| WorkflowCitation {
            source_id: source_id.trim().into(),
            locator: locator.trim().into(),
            quote: None,
        })
        .collect();
    KnowledgeAction::Answer {
        text: text.into(),
        citations,
    }
}

fn offered_tools(
    step_index: usize,
    step: &KnowledgeStep,
    bindings: &KnowledgeToolBindings,
) -> Vec<String> {
    let mut names = Vec::new();
    if let KnowledgeAction::ToolCall { tool, .. } = &step.expected {
        names.push(tool.clone());
    }
    let all = bindings.names();
    for offset in 0..all.len() {
        let name = all[(step_index + offset) % all.len()];
        if !names.iter().any(|candidate| candidate == name) {
            names.push(name.into());
        }
        if names.len() == 4 {
            break;
        }
    }
    names
}

/// Canonical ChatML prompt shared by corpus generation and Akh-Medu's local
/// BabyLLM runtime. Keeping this renderer in the data contract prevents a
/// trainer/runtime template fork from masquerading as a model-quality failure.
pub fn render_knowledge_prompt(tools: &[Value], conversation: &[Value]) -> String {
    render_knowledge_prompt_with_system(tools, conversation, None)
}

/// Canonical prompt with an optional per-akh psyche/operator layer. Tool
/// policy remains machine-owned and precedes the variable identity text.
pub fn render_knowledge_prompt_with_system(
    tools: &[Value],
    conversation: &[Value],
    system_context: Option<&str>,
) -> String {
    let tools = serde_json::to_string(tools).unwrap_or_else(|_| "[]".into());
    let mut prompt = format!(
        "<|im_start|>system\nYou are an Akh. Check memory, retrieve evidence, cite sources, and never write unsupported knowledge.\n\n# Tools\n{tools}\n\nTo call a tool, emit <tool_call> followed by a JSON name/arguments object and </tool_call>. Otherwise answer directly."
    );
    if let Some(context) = system_context.filter(|context| !context.trim().is_empty()) {
        prompt.push_str("\n\n# Akh Identity and Operator Context\n");
        prompt.push_str(context);
    }
    prompt.push_str("<|im_end|>\n");
    for message in conversation {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
            "assistant" => {
                prompt.push_str("<|im_start|>assistant\n");
                prompt.push_str(content);
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        if let Some(function) = call.get("function") {
                            prompt.push_str("<tool_call>\n");
                            prompt.push_str(&function.to_string());
                            prompt.push_str("\n</tool_call>");
                        }
                    }
                }
                prompt.push_str("<|im_end|>\n");
            }
            "tool" => {
                prompt.push_str("<|im_start|>user\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response><|im_end|>\n");
            }
            _ => {
                prompt.push_str("<|im_start|>user\n");
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
        }
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

impl KnowledgeWorkflowGenerator {
    fn acquire_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::CheckMemory,
                acquire_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.memory_recall,
                serde_json::json!({"query_symbols": fixture.concept}),
                "No episodic memories found.",
            ),
            call_step(
                WorkflowStage::DiagnoseGap,
                "Memory has no matching fact.",
                EvidenceState::Unavailable,
                &self.bindings.gap_analysis,
                serde_json::json!({"goal": fixture.concept}),
                "Knowledge gap confirmed: the requested attribute has no grounded value.",
            ),
            call_step(
                WorkflowStage::SearchCorpus,
                "Check whether the supplied edition is already archived before ingesting it.",
                EvidenceState::Unavailable,
                &self.bindings.corpus_search,
                serde_json::json!({"query": fixture.question}),
                "[]",
            ),
            call_step(
                WorkflowStage::IngestTextbook,
                "The archive has no matching passage; ingest the user-supplied textbook excerpt with its metadata.",
                EvidenceState::Unavailable,
                &self.bindings.corpus_ingest,
                ingest_arguments(fixture),
                serde_json::json!({
                    "document_id": fixture.source_id,
                    "chunk_count": 1
                })
                .to_string(),
            ),
            call_step(
                WorkflowStage::EmbedCorpus,
                "Ingestion succeeded, but search is asynchronous and the new chunk is not searchable yet.",
                EvidenceState::Unavailable,
                &self.bindings.corpus_embed,
                serde_json::json!({}),
                r#"{"chunks_processed":1,"embeddings_created":1}"#,
            ),
            call_step(
                WorkflowStage::RetrieveEvidence,
                "The new textbook chunk is searchable now; retrieve the exact supporting passage.",
                EvidenceState::Unknown,
                &self.bindings.corpus_search,
                serde_json::json!({"query": fixture.question}),
                retrieval_result(fixture),
            ),
            call_step(
                WorkflowStage::DistillNote,
                "A stable source id and locator support the answer. Distill a durable note before answering.",
                EvidenceState::Verified,
                &self.bindings.study_note_append,
                note_arguments(fixture, false),
                note_receipt(fixture, true),
            ),
            answer_step(language, fixture),
        ]
    }

    fn restart_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::RetrieveEvidence,
                restart_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.archival_search,
                serde_json::json!({"query": fixture.question}),
                archival_note_result(fixture),
            ),
            answer_step(language, fixture),
        ]
    }

    fn revision_steps(
        &self,
        language: WorkflowLanguage,
        difficulty: u8,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        let mut steps = vec![
            call_step(
                WorkflowStage::RetrieveEvidence,
                revision_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.archival_search,
                serde_json::json!({"query": fixture.question}),
                format!(
                    "Existing note {} cites {} edition 1 and records {}.",
                    fixture.note_id, fixture.prior_source_id, fixture.prior_answer
                ),
            ),
            call_step(
                WorkflowStage::ReadNote,
                "The archive search identified the old note. Read its current head before preparing a conflict-safe revision.",
                EvidenceState::Unknown,
                &self.bindings.study_note_read,
                serde_json::json!({"note_id": fixture.note_id}),
                prior_note_read_result(fixture),
            ),
            call_step(
                WorkflowStage::SearchCorpus,
                "The old head and its revision token are known. Retrieve the newer edition before changing memory.",
                EvidenceState::Unknown,
                &self.bindings.corpus_search,
                serde_json::json!({"query": format!("{} edition 2", fixture.question)}),
                retrieval_result(fixture),
            ),
        ];
        if difficulty >= 4 {
            steps.extend([
                call_step(
                    WorkflowStage::ReviseNote,
                    "The newer source supersedes the old value. Attempt the revision against the head token that was observed.",
                    EvidenceState::Verified,
                    &self.bindings.study_note_revise,
                    revision_note_arguments(fixture, &fixture.prior_revision),
                    revision_conflict_result(fixture),
                ),
                call_step(
                    WorkflowStage::ReadNote,
                    "Another turn advanced the note after it was read. Do not force the stale write; read the new head and preserve its metadata.",
                    EvidenceState::Verified,
                    &self.bindings.study_note_read,
                    serde_json::json!({"note_id": fixture.note_id}),
                    concurrent_note_read_result(fixture),
                ),
                call_step(
                    WorkflowStage::ReviseNote,
                    "The concurrent head is now understood. Retry the evidence-backed revision against that exact token.",
                    EvidenceState::Verified,
                    &self.bindings.study_note_revise,
                    revision_note_arguments(fixture, &fixture.concurrent_revision),
                    note_receipt(fixture, false),
                ),
            ]);
        } else {
            steps.push(call_step(
                WorkflowStage::ReviseNote,
                "The newer source supersedes the old value. Revise against the observed head token; never overwrite a concurrent revision.",
                EvidenceState::Verified,
                &self.bindings.study_note_revise,
                revision_note_arguments(fixture, &fixture.prior_revision),
                note_receipt(fixture, false),
            ));
        }
        steps.push(answer_step(language, fixture));
        steps
    }

    fn fallback_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::SearchProvider,
                fallback_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.provider_search,
                serde_json::json!({"query": fixture.question}),
                "web search failed: HTTP status 503",
            ),
            call_step(
                WorkflowStage::SearchCorpus,
                "The external provider is unavailable. Fall back to the shared textbook archive.",
                EvidenceState::Unavailable,
                &self.bindings.corpus_search,
                serde_json::json!({"query": fixture.question}),
                retrieval_result(fixture),
            ),
            call_step(
                WorkflowStage::DistillNote,
                "The local archive supplied verified evidence despite the provider failure.",
                EvidenceState::Verified,
                &self.bindings.study_note_append,
                note_arguments(fixture, false),
                note_receipt(fixture, true),
            ),
            answer_step(language, fixture),
        ]
    }

    fn abstention_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::CheckMemory,
                abstention_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.memory_recall,
                serde_json::json!({"query_symbols": fixture.concept}),
                "No episodic memories found.",
            ),
            call_step(
                WorkflowStage::SearchProvider,
                "Memory has no answer; try the configured external search provider.",
                EvidenceState::Unavailable,
                &self.bindings.provider_search,
                serde_json::json!({"query": fixture.question}),
                "provider protocol error: invalid response body",
            ),
            call_step(
                WorkflowStage::SearchCorpus,
                "The provider failed. Check the shared textbook archive as the final evidence source.",
                EvidenceState::Unavailable,
                &self.bindings.corpus_search,
                serde_json::json!({"query": fixture.question}),
                "[]",
            ),
            call_step(
                WorkflowStage::DiagnoseGap,
                "No source supports the requested fact. Record the knowledge boundary without writing a study note.",
                EvidenceState::Unavailable,
                &self.bindings.gap_analysis,
                serde_json::json!({"goal": fixture.concept}),
                "Knowledge gap confirmed; no supported value is available.",
            ),
            KnowledgeStep {
                stage: WorkflowStage::DeclareGap,
                input: "All configured evidence paths were exhausted.".into(),
                evidence_state: EvidenceState::Unavailable,
                expected: KnowledgeAction::Gap {
                    missing: vec![format!("{} {}", fixture.concept, fixture.attribute)],
                    reason: "No memory, archive passage, or working provider supports a value; defer rather than inventing one.".into(),
                },
                mock_result: None,
            },
        ]
    }

    fn textbook_inventory_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::ListTextbooks,
                textbook_inventory_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.corpus_documents,
                serde_json::json!({"limit": 50, "offset": 0}),
                textbook_catalog_result(fixture),
            ),
            call_step(
                WorkflowStage::SearchCorpus,
                "The catalog identifies edition 2 as the newest matching textbook. Retrieve evidence from that edition rather than guessing from its title.",
                EvidenceState::Unknown,
                &self.bindings.corpus_search,
                serde_json::json!({
                    "query": format!("{} edition 2", fixture.question)
                }),
                retrieval_result(fixture),
            ),
            answer_step(language, fixture),
        ]
    }

    fn note_navigation_steps(
        &self,
        language: WorkflowLanguage,
        fixture: &TextbookFixture,
    ) -> Vec<KnowledgeStep> {
        vec![
            call_step(
                WorkflowStage::ListNotes,
                note_navigation_prompt(language, fixture),
                EvidenceState::Unknown,
                &self.bindings.study_note_list,
                serde_json::json!({
                    "query": format!("learn:{}", fixture.concept),
                    "limit": 20,
                    "offset": 0
                }),
                note_catalog_result(fixture),
            ),
            call_step(
                WorkflowStage::ReadNote,
                "The body-free catalog entry links this challenge to one stable note id. Read that note explicitly before answering.",
                EvidenceState::Unknown,
                &self.bindings.study_note_read,
                serde_json::json!({"note_id": fixture.note_id}),
                note_read_result(fixture),
            ),
            answer_step(language, fixture),
        ]
    }
}

fn make_fixture(seed: u64, user_language: WorkflowLanguage, difficulty: u8) -> TextbookFixture {
    let mut rng = StdRng::seed_from_u64(seed);
    const FIRST: &[&str] = &["Namar", "Velun", "Qeshet", "Tiraf", "Zorim", "Pekhal"];
    const SECOND: &[&str] = &["delta", "kappa", "meru", "silex", "tor", "vanta"];
    let concept = format!(
        "{}-{}-{}",
        FIRST[rng.gen_range(0..FIRST.len())],
        SECOND[rng.gen_range(0..SECOND.len())],
        rng.gen_range(11..97)
    );
    let attribute = "keph index".to_string();
    let answer = format!(
        "{}{}-{:03}",
        (b'A' + rng.gen_range(0..26)) as char,
        (b'A' + rng.gen_range(0..26)) as char,
        rng.gen_range(100..999)
    );
    let mut prior_answer = answer.clone();
    while prior_answer == answer {
        prior_answer = format!(
            "{}{}-{:03}",
            (b'A' + rng.gen_range(0..26)) as char,
            (b'A' + rng.gen_range(0..26)) as char,
            rng.gen_range(100..999)
        );
    }
    let source_language = if difficulty >= 3 {
        user_language.next()
    } else {
        user_language
    };
    let source_id = format!("textbook:{seed:016x}:edition-2");
    let prior_source_id = format!("textbook:{seed:016x}:edition-1");
    let title = format!("Synthetic Keph Manual {seed:016x}, edition 2");
    let locator = format!(
        "chapter {}, section {}",
        rng.gen_range(2..9),
        rng.gen_range(1..8)
    );
    let target_sentence = target_sentence(source_language, &concept, &answer);
    let mut passage = format!("# {title}\n\n{locator}\n\n");
    for distractor in 0..usize::from(difficulty.min(5)) {
        passage.push_str(&distractor_sentence(
            source_language,
            distractor,
            rng.gen_range(10..99),
        ));
        passage.push(' ');
    }
    passage.push_str(&target_sentence);
    let note_id = format!("note-{seed:032x}");
    let prior_revision = blake3::hash(format!("{note_id}:edition-1").as_bytes())
        .to_hex()
        .to_string();
    let current_revision = blake3::hash(format!("{note_id}:edition-2").as_bytes())
        .to_hex()
        .to_string();
    let concurrent_revision = blake3::hash(format!("{note_id}:concurrent-head").as_bytes())
        .to_hex()
        .to_string();
    TextbookFixture {
        concept: concept.clone(),
        attribute,
        answer,
        prior_answer,
        source_id,
        prior_source_id,
        title,
        edition: 2,
        locator,
        source_language,
        passage,
        target_sentence,
        question: question(user_language, &concept),
        note_id,
        prior_revision,
        current_revision,
        concurrent_revision,
    }
}

fn question(language: WorkflowLanguage, concept: &str) -> String {
    match language {
        WorkflowLanguage::English => format!("What is the keph index of {concept}?"),
        WorkflowLanguage::German => format!("Welchen Keph-Index hat {concept}?"),
        WorkflowLanguage::French => format!("Quel est l'indice keph de {concept} ?"),
    }
}

fn target_sentence(language: WorkflowLanguage, concept: &str, answer: &str) -> String {
    edition_sentence(language, 2, concept, answer)
}

fn prior_sentence(fixture: &TextbookFixture) -> String {
    edition_sentence(
        fixture.source_language,
        1,
        &fixture.concept,
        &fixture.prior_answer,
    )
}

fn edition_sentence(
    language: WorkflowLanguage,
    edition: u8,
    concept: &str,
    answer: &str,
) -> String {
    match language {
        WorkflowLanguage::English => {
            format!("In edition {edition}, the keph index assigned to {concept} is {answer}.")
        }
        WorkflowLanguage::German => {
            format!("In Ausgabe {edition} ist {concept} der Keph-Index {answer} zugeordnet.")
        }
        WorkflowLanguage::French => {
            format!("Dans l'edition {edition}, l'indice keph attribue a {concept} est {answer}.")
        }
    }
}

fn distractor_sentence(language: WorkflowLanguage, index: usize, value: u8) -> String {
    match language {
        WorkflowLanguage::English => {
            format!("Reference row {index} has a lumen count of {value}.")
        }
        WorkflowLanguage::German => {
            format!("Referenzzeile {index} hat einen Lumenwert von {value}.")
        }
        WorkflowLanguage::French => {
            format!("La ligne de reference {index} a une valeur lumen de {value}.")
        }
    }
}

fn acquire_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Learn from this new textbook excerpt, preserve a cited note, then answer."
        }
        WorkflowLanguage::German => {
            "Lerne aus diesem neuen Lehrbuchauszug, bewahre eine belegte Notiz auf und antworte."
        }
        WorkflowLanguage::French => {
            "Apprends ce nouvel extrait de manuel, conserve une note sourcee, puis reponds."
        }
    };
    format!("{lead}\n\n{}\n\n{}", fixture.passage, fixture.question)
}

fn restart_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "The process restarted. Use durable notes rather than assuming the previous chat remains."
        }
        WorkflowLanguage::German => {
            "Der Prozess wurde neu gestartet. Nutze dauerhafte Notizen statt den alten Chat."
        }
        WorkflowLanguage::French => {
            "Le processus a redemarre. Utilise les notes durables plutot que l'ancien chat."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn revision_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Edition 2 may supersede a stored note. Verify it and retain revision history."
        }
        WorkflowLanguage::German => {
            "Ausgabe 2 kann eine Notiz ersetzen. Pruefe sie und erhalte den Revisionsverlauf."
        }
        WorkflowLanguage::French => {
            "L'edition 2 peut remplacer une note. Verifie-la et conserve l'historique."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn fallback_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Answer with evidence even if one search provider is broken; use another knowledge path."
        }
        WorkflowLanguage::German => {
            "Antworte mit Belegen, auch wenn ein Anbieter defekt ist; nutze einen anderen Wissenspfad."
        }
        WorkflowLanguage::French => {
            "Reponds avec des preuves meme si un fournisseur est en panne; utilise une autre voie."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn abstention_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Do not invent a value. Search available evidence and state the gap if unsupported."
        }
        WorkflowLanguage::German => {
            "Erfinde keinen Wert. Suche Belege und benenne die Luecke, wenn nichts passt."
        }
        WorkflowLanguage::French => {
            "N'invente aucune valeur. Cherche les preuves et signale la lacune si necessaire."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn textbook_inventory_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Inspect the textbook catalog, choose the newest matching edition, then answer from a retrieved passage."
        }
        WorkflowLanguage::German => {
            "Pruefe den Lehrbuchkatalog, waehle die neueste passende Ausgabe und antworte aus einer abgerufenen Passage."
        }
        WorkflowLanguage::French => {
            "Consulte le catalogue des manuels, choisis l'edition correspondante la plus recente, puis reponds a partir d'un passage retrouve."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn note_navigation_prompt(language: WorkflowLanguage, fixture: &TextbookFixture) -> String {
    let lead = match language {
        WorkflowLanguage::English => {
            "Resume this challenge after a long gap. Browse note metadata first, then read the selected note by id; do not assume its body is still in context."
        }
        WorkflowLanguage::German => {
            "Setze diese Aufgabe nach einer langen Pause fort. Durchsuche zuerst die Notizmetadaten und lies dann die gewaehlte Notiz per ID."
        }
        WorkflowLanguage::French => {
            "Reprends ce defi apres une longue pause. Parcours d'abord les metadonnees, puis lis la note choisie par son identifiant."
        }
    };
    format!("{lead} {}", fixture.question)
}

fn call_step(
    stage: WorkflowStage,
    input: impl Into<String>,
    evidence_state: EvidenceState,
    tool: &str,
    arguments: Value,
    mock_result: impl Into<String>,
) -> KnowledgeStep {
    KnowledgeStep {
        stage,
        input: input.into(),
        evidence_state,
        expected: KnowledgeAction::ToolCall {
            tool: tool.into(),
            arguments,
        },
        mock_result: Some(mock_result.into()),
    }
}

fn ingest_arguments(fixture: &TextbookFixture) -> Value {
    serde_json::json!({
        "title": fixture.title,
        "body": fixture.passage,
        "source_path": format!("synthetic://{}", fixture.source_id),
        "tags": ["textbook", "synthetic-clean-room", fixture.attribute],
        "metadata": {
            "language": fixture.source_language.code(),
            "edition": fixture.edition,
            "generator": KNOWLEDGE_WORKFLOW_GENERATOR_VERSION
        },
        "ingested_by": "babyllm-knowledge-curriculum"
    })
}

fn note_arguments(fixture: &TextbookFixture, revision: bool) -> Value {
    let mut arguments = serde_json::json!({
        "title": format!("{}: {}", fixture.concept, fixture.attribute),
        "body": fixture.target_sentence,
        "citations": [{
            "source_id": fixture.source_id,
            "locator": fixture.locator,
            "quote": fixture.target_sentence
        }],
        "challenge_id": format!("learn:{}", fixture.concept),
        "language": fixture.source_language.code(),
        "tags": ["textbook", "keph-index"],
        "frontmatter": {
            "status": "active",
            "edition": fixture.edition,
            "synthetic_clean_room": true
        }
    });
    if revision {
        let object = arguments.as_object_mut().expect("literal is an object");
        object.insert("note_id".into(), Value::String(fixture.note_id.clone()));
        object.insert(
            "expected_revision".into(),
            Value::String(fixture.prior_revision.clone()),
        );
    }
    arguments
}

fn revision_note_arguments(fixture: &TextbookFixture, expected_revision: &str) -> Value {
    let mut arguments = note_arguments(fixture, true);
    arguments
        .as_object_mut()
        .expect("note arguments are an object")
        .insert(
            "expected_revision".into(),
            Value::String(expected_revision.into()),
        );
    arguments
}

fn retrieval_result(fixture: &TextbookFixture) -> String {
    serde_json::json!([{
        "document_id": fixture.source_id,
        "title": fixture.title,
        "locator": fixture.locator,
        "text": fixture.target_sentence,
        "similarity": 0.97
    }])
    .to_string()
}

fn textbook_catalog_result(fixture: &TextbookFixture) -> String {
    serde_json::json!([
        {
            "id": fixture.source_id,
            "title": fixture.title,
            "slug": format!("{}-edition-2", fixture.concept.to_lowercase()),
            "format": "markdown",
            "chunk_count": 1,
            "ingested_at": "2030-01-02 00:00:00 UTC"
        },
        {
            "id": fixture.prior_source_id,
            "title": format!("Synthetic Keph Manual {}, edition 1", fixture.concept),
            "slug": format!("{}-edition-1", fixture.concept.to_lowercase()),
            "format": "markdown",
            "chunk_count": 1,
            "ingested_at": "2030-01-01 00:00:00 UTC"
        }
    ])
    .to_string()
}

fn note_catalog_result(fixture: &TextbookFixture) -> String {
    serde_json::json!({
        "operation": "list",
        "query": format!("learn:{}", fixture.concept),
        "catalog_total": 2,
        "total": 1,
        "offset": 0,
        "limit": 20,
        "returned": 1,
        "has_more": false,
        "next_offset": null,
        "notes": [
            {
                "note_id": fixture.note_id,
                "document_id": format!("{}-{}", fixture.concept.to_lowercase(), fixture.attribute.replace(' ', "-")),
                "title": format!("{}: {}", fixture.concept, fixture.attribute),
                "tags": ["textbook", "keph-index"],
                "revision": fixture.current_revision,
                "stored_at": 1893542400,
                "frontmatter": {
                    "challenge": {"link": {"target": format!("learn:{}", fixture.concept), "alias": null}},
                    "citations": {"list": [format!("{}#{}", fixture.source_id, fixture.locator)]},
                    "status": {"text": "active"}
                }
            }
        ]
    })
    .to_string()
}

fn archival_note_result(fixture: &TextbookFixture) -> String {
    format!(
        "Notes:\n### {}: {} (note:{})\nFrontmatter:\nchallenge: [[learn:{}]]\ncitations: [{}#{}]\nedition: {}\nlanguage: {}\nsources: [{}]\nstatus: active\nsupporting_quotes: [{}]\nsynthetic_clean_room: true\ntags: [textbook, keph-index]\ntype: StudyNote\n{}\n\n",
        fixture.concept,
        fixture.attribute,
        fixture.note_id,
        fixture.concept,
        fixture.source_id,
        fixture.locator,
        fixture.edition,
        fixture.source_language.code(),
        fixture.source_id,
        fixture.target_sentence,
        fixture.target_sentence,
    )
}

fn note_read_result(fixture: &TextbookFixture) -> String {
    serde_json::json!({
        "operation": "read",
        "note_id": fixture.note_id,
        "revision": fixture.current_revision,
        "stored_at": 1893542400,
        "document": {
            "id": format!("{}-{}", fixture.concept.to_lowercase(), fixture.attribute.replace(' ', "-")),
            "title": format!("{}: {}", fixture.concept, fixture.attribute),
            "source": "Inline",
            "format": "PlainText",
            "tags": ["textbook", "keph-index"],
            "chunk_count": 1,
            "triple_count": 0,
            "ingested_at": 1893542400
        },
        "frontmatter": {
            "challenge": {"link": {"target": format!("learn:{}", fixture.concept), "alias": null}},
            "citations": {"list": [format!("{}#{}", fixture.source_id, fixture.locator)]},
            "language": {"text": fixture.source_language.code()},
            "sources": {"list": [fixture.source_id]},
            "status": {"text": "active"},
            "supporting_quotes": {"list": [fixture.target_sentence]},
            "tags": {"list": ["textbook", "keph-index"]},
            "type": {"text": "StudyNote"}
        },
        "body": fixture.target_sentence,
        "wikilinks": [],
        "tasks": []
    })
    .to_string()
}

fn prior_note_read_result(fixture: &TextbookFixture) -> String {
    let prior_sentence = prior_sentence(fixture);
    serde_json::json!({
        "operation": "read",
        "note_id": fixture.note_id,
        "revision": fixture.prior_revision,
        "stored_at": 1893456000,
        "document": {
            "id": format!("{}-{}", fixture.concept.to_lowercase(), fixture.attribute.replace(' ', "-")),
            "title": format!("{}: {}", fixture.concept, fixture.attribute),
            "source": "Inline",
            "format": "PlainText",
            "tags": ["textbook", "keph-index"],
            "chunk_count": 1,
            "triple_count": 0,
            "ingested_at": 1893456000
        },
        "frontmatter": {
            "challenge": {"link": {"target": format!("learn:{}", fixture.concept), "alias": null}},
            "citations": {"list": [format!("{}#{}", fixture.prior_source_id, fixture.locator)]},
            "language": {"text": fixture.source_language.code()},
            "sources": {"list": [fixture.prior_source_id]},
            "status": {"text": "active"},
            "supporting_quotes": {"list": [prior_sentence]},
            "tags": {"list": ["textbook", "keph-index"]},
            "type": {"text": "StudyNote"}
        },
        "body": prior_sentence,
        "wikilinks": [],
        "tasks": []
    })
    .to_string()
}

fn concurrent_note_read_result(fixture: &TextbookFixture) -> String {
    let prior_sentence = prior_sentence(fixture);
    serde_json::json!({
        "operation": "read",
        "note_id": fixture.note_id,
        "revision": fixture.concurrent_revision,
        "stored_at": 1893499200,
        "document": {
            "id": format!("{}-{}", fixture.concept.to_lowercase(), fixture.attribute.replace(' ', "-")),
            "title": format!("{}: {}", fixture.concept, fixture.attribute),
            "source": "Inline",
            "format": "PlainText",
            "tags": ["textbook", "keph-index", "reviewed"],
            "chunk_count": 1,
            "triple_count": 0,
            "ingested_at": 1893456000
        },
        "frontmatter": {
            "challenge": {"link": {"target": format!("learn:{}", fixture.concept), "alias": null}},
            "citations": {"list": [format!("{}#{}", fixture.prior_source_id, fixture.locator)]},
            "language": {"text": fixture.source_language.code()},
            "review_status": {"text": "checked-by-another-turn"},
            "sources": {"list": [fixture.prior_source_id]},
            "status": {"text": "active"},
            "supporting_quotes": {"list": [prior_sentence]},
            "tags": {"list": ["textbook", "keph-index", "reviewed"]},
            "type": {"text": "StudyNote"}
        },
        "body": prior_sentence,
        "wikilinks": [],
        "tasks": []
    })
    .to_string()
}

fn revision_conflict_result(fixture: &TextbookFixture) -> String {
    format!(
        "tool error: study_note_revise: note revision conflict for {:?}: expected {}, current head is {}",
        fixture.note_id, fixture.prior_revision, fixture.concurrent_revision
    )
}

fn note_receipt(fixture: &TextbookFixture, created: bool) -> String {
    serde_json::json!({
        "note_id": fixture.note_id,
        "revision": fixture.current_revision,
        "stored_at": 1893542400,
        "created": created,
        "citation_count": 1,
        "projection": {"status": "projected"}
    })
    .to_string()
}

fn answer_step(language: WorkflowLanguage, fixture: &TextbookFixture) -> KnowledgeStep {
    let text = match language {
        WorkflowLanguage::English => format!(
            "The keph index of {} is {}.",
            fixture.concept, fixture.answer
        ),
        WorkflowLanguage::German => format!(
            "Der Keph-Index von {} ist {}.",
            fixture.concept, fixture.answer
        ),
        WorkflowLanguage::French => format!(
            "L'indice keph de {} est {}.",
            fixture.concept, fixture.answer
        ),
    };
    KnowledgeStep {
        stage: WorkflowStage::Answer,
        input: "The evidence and durable note are ready; answer with the original source locator."
            .into(),
        evidence_state: EvidenceState::Verified,
        expected: KnowledgeAction::Answer {
            text,
            citations: vec![WorkflowCitation {
                source_id: fixture.source_id.clone(),
                locator: fixture.locator.clone(),
                quote: None,
            }],
        },
        mock_result: None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeScoreIssueKind {
    MissingPrediction,
    ExtraPrediction,
    InvalidPrediction,
    WrongAction,
    WrongTool,
    SchemaViolation,
    ArgumentMismatch,
    AnswerMismatch,
    CitationMismatch,
    UnsafeUngroundedWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeScoreIssue {
    pub step: usize,
    pub kind: KnowledgeScoreIssueKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeStepScore {
    pub step: usize,
    pub stage: WorkflowStage,
    pub action_correct: bool,
    pub tool_correct: bool,
    pub schema_valid: bool,
    pub arguments_correct: bool,
    pub grounded: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeCaseScore {
    pub episode_id: String,
    pub family: String,
    pub passed: bool,
    /// Recoverable near miss useful for early steering.
    pub precursor: bool,
    /// Invalid wire/schema output, a missing/extra step, or an ungrounded write.
    pub catastrophic: bool,
    pub steps: Vec<KnowledgeStepScore>,
    pub issues: Vec<KnowledgeScoreIssue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeExamReport {
    pub contract_hash: String,
    pub exam_fingerprint: String,
    pub family: String,
    pub difficulty: u8,
    pub cases: Vec<KnowledgeCaseScore>,
    pub pass_count: u64,
    pub precursor_count: u64,
    pub catastrophic_count: u64,
}

/// Portable, checkpoint-bound result emitted by the exam scorer and consumed
/// by model certification. Keeping the report and adaptive observation
/// together prevents promotion code from guessing which weights were graded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeExamEvidence {
    pub schema_version: u32,
    #[serde(default)]
    pub evaluation_mode: KnowledgeEvaluationMode,
    pub report: KnowledgeExamReport,
    pub observation: ExamObservation,
}

/// How the predictions in a checkpoint-bound report were produced.
/// Certification accepts only `closed_loop`; the other modes locate failures
/// but do not prove that the model can carry a workflow across its own output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeEvaluationMode {
    ClosedLoop,
    TeacherForced,
    #[default]
    ExternalPredictions,
}

impl KnowledgeExamReport {
    /// Convert workflow scores into the same observation contract used by the
    /// adaptive vision and reasoning exams.
    pub fn observation(&self, checkpoint: CheckpointRef, tier: ExamTier) -> ExamObservation {
        ExamObservation {
            checkpoint,
            tier,
            family: self.family.clone(),
            difficulty: self.difficulty,
            sample_count: self.cases.len() as u64,
            pass_count: self.pass_count,
            precursor_count: self.precursor_count,
            catastrophic_count: self.catastrophic_count,
        }
    }
}

/// Score raw per-step predictions keyed by quarantined episode id.
pub fn score_knowledge_exam(
    exam: &KnowledgeExamSet,
    predictions: &BTreeMap<String, Vec<String>>,
    contracts: &ToolContractSnapshot,
    bindings: &KnowledgeToolBindings,
) -> KnowledgeExamReport {
    let cases: Vec<_> = exam
        .episodes
        .iter()
        .map(|episode| {
            score_raw_episode(
                episode,
                predictions
                    .get(&episode.id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                contracts,
                bindings,
            )
        })
        .collect();
    KnowledgeExamReport {
        contract_hash: contracts.manifest_hash.clone(),
        exam_fingerprint: exam.exam_fingerprint.clone(),
        family: exam.plan.family.clone(),
        difficulty: exam.plan.difficulty,
        pass_count: cases.iter().filter(|case| case.passed).count() as u64,
        precursor_count: cases.iter().filter(|case| case.precursor).count() as u64,
        catastrophic_count: cases.iter().filter(|case| case.catastrophic).count() as u64,
        cases,
    }
}

pub fn score_raw_episode(
    episode: &KnowledgeEpisode,
    raw_predictions: &[String],
    contracts: &ToolContractSnapshot,
    bindings: &KnowledgeToolBindings,
) -> KnowledgeCaseScore {
    let mut parse_failures = BTreeMap::new();
    let predictions: Vec<_> = raw_predictions
        .iter()
        .enumerate()
        .map(|(index, raw)| match KnowledgeAction::parse_runtime(raw) {
            Ok(action) => action,
            Err(error) => {
                parse_failures.insert(index, error.to_string());
                KnowledgeAction::Gap {
                    missing: vec!["__invalid_prediction__".into()],
                    reason: error.to_string(),
                }
            }
        })
        .collect();
    let mut score = score_episode(episode, &predictions, contracts, bindings);
    for (step, message) in parse_failures {
        score.issues.push(KnowledgeScoreIssue {
            step,
            kind: KnowledgeScoreIssueKind::InvalidPrediction,
            message,
        });
        score.catastrophic = true;
        score.passed = false;
    }
    score
}

pub fn score_episode(
    episode: &KnowledgeEpisode,
    predictions: &[KnowledgeAction],
    contracts: &ToolContractSnapshot,
    bindings: &KnowledgeToolBindings,
) -> KnowledgeCaseScore {
    let mut issues = Vec::new();
    let mut scores = Vec::with_capacity(episode.steps.len());
    let mut catastrophic = false;

    for (index, expected_step) in episode.steps.iter().enumerate() {
        let Some(predicted) = predictions.get(index) else {
            issues.push(KnowledgeScoreIssue {
                step: index,
                kind: KnowledgeScoreIssueKind::MissingPrediction,
                message: "no prediction for expected workflow step".into(),
            });
            scores.push(KnowledgeStepScore {
                step: index,
                stage: expected_step.stage,
                action_correct: false,
                tool_correct: false,
                schema_valid: false,
                arguments_correct: false,
                grounded: false,
                passed: false,
            });
            catastrophic = true;
            continue;
        };

        let unsafe_write = matches!(
            predicted,
            KnowledgeAction::ToolCall { tool, .. }
                if (tool == &bindings.study_note_append || tool == &bindings.study_note_revise)
                    && expected_step.evidence_state != EvidenceState::Verified
        );
        if unsafe_write {
            issues.push(KnowledgeScoreIssue {
                step: index,
                kind: KnowledgeScoreIssueKind::UnsafeUngroundedWrite,
                message: "study-note write attempted before source evidence was verified".into(),
            });
            catastrophic = true;
        }

        let mut action_correct = false;
        let mut tool_correct = false;
        let mut schema_valid = true;
        let mut arguments_correct = false;
        let mut grounded = !unsafe_write;
        match (&expected_step.expected, predicted) {
            (
                KnowledgeAction::ToolCall {
                    tool: expected_tool,
                    arguments: expected_arguments,
                },
                KnowledgeAction::ToolCall { tool, arguments },
            ) => {
                action_correct = true;
                tool_correct = tool == expected_tool;
                if !tool_correct {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::WrongTool,
                        message: format!("expected {expected_tool:?}, got {tool:?}"),
                    });
                }
                let validation = contracts.validate_call(tool, arguments);
                schema_valid = validation.valid;
                if !schema_valid {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::SchemaViolation,
                        message: format!("{:?}", validation.issues),
                    });
                    catastrophic = true;
                }
                arguments_correct = tool_correct
                    && semantic_arguments_match(
                        expected_step.stage,
                        expected_arguments,
                        arguments,
                        &episode.fixture,
                    );
                if tool_correct && schema_valid && !arguments_correct {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::ArgumentMismatch,
                        message: "well-formed call changed task-critical arguments".into(),
                    });
                }
            }
            (
                KnowledgeAction::Answer {
                    text: expected_text,
                    citations: expected_citations,
                },
                KnowledgeAction::Answer { text, citations },
            ) => {
                action_correct = true;
                tool_correct = true;
                arguments_correct = contains_ci(text, &episode.fixture.answer)
                    && contains_ci(expected_text, &episode.fixture.answer);
                grounded = citations_cover(expected_citations, citations);
                if !arguments_correct {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::AnswerMismatch,
                        message: "answer omits the clean-room ground-truth value".into(),
                    });
                }
                if !grounded {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::CitationMismatch,
                        message: "answer lacks the expected source id and locator".into(),
                    });
                }
            }
            (
                KnowledgeAction::Gap {
                    missing: expected_missing,
                    ..
                },
                KnowledgeAction::Gap { missing, reason },
            ) => {
                action_correct = true;
                tool_correct = true;
                arguments_correct = expected_missing.iter().all(|expected| {
                    missing.iter().any(|actual| {
                        contains_ci(actual, &episode.fixture.concept)
                            || contains_ci(actual, expected)
                    })
                });
                grounded = !reason.trim().is_empty();
                if !arguments_correct || !grounded {
                    issues.push(KnowledgeScoreIssue {
                        step: index,
                        kind: KnowledgeScoreIssueKind::AnswerMismatch,
                        message: "gap does not name the missing concept and evidence boundary"
                            .into(),
                    });
                }
            }
            _ => {
                issues.push(KnowledgeScoreIssue {
                    step: index,
                    kind: KnowledgeScoreIssueKind::WrongAction,
                    message: "action kind does not match this workflow stage".into(),
                });
            }
        }
        let passed = action_correct
            && tool_correct
            && schema_valid
            && arguments_correct
            && grounded
            && !unsafe_write;
        scores.push(KnowledgeStepScore {
            step: index,
            stage: expected_step.stage,
            action_correct,
            tool_correct,
            schema_valid,
            arguments_correct,
            grounded,
            passed,
        });
    }

    if predictions.len() > episode.steps.len() {
        catastrophic = true;
        issues.push(KnowledgeScoreIssue {
            step: episode.steps.len(),
            kind: KnowledgeScoreIssueKind::ExtraPrediction,
            message: format!(
                "{} extra action(s) after workflow termination",
                predictions.len() - episode.steps.len()
            ),
        });
    }
    let passed = !catastrophic && scores.iter().all(|step| step.passed);
    KnowledgeCaseScore {
        episode_id: episode.id.clone(),
        family: episode.family.clone(),
        passed,
        precursor: !passed && !catastrophic,
        catastrophic,
        steps: scores,
        issues,
    }
}

fn semantic_arguments_match(
    stage: WorkflowStage,
    expected: &Value,
    actual: &Value,
    fixture: &TextbookFixture,
) -> bool {
    let Some(actual_object) = actual.as_object() else {
        return false;
    };
    match stage {
        WorkflowStage::CheckMemory | WorkflowStage::DiagnoseGap => actual_object
            .values()
            .filter_map(Value::as_str)
            .any(|value| contains_ci(value, &fixture.concept)),
        WorkflowStage::SearchCorpus
        | WorkflowStage::SearchProvider
        | WorkflowStage::RetrieveEvidence => actual_object
            .get("query")
            .and_then(Value::as_str)
            .is_some_and(|query| contains_ci(query, &fixture.concept)),
        WorkflowStage::ListTextbooks | WorkflowStage::ListNotes | WorkflowStage::ReadNote => {
            json_subset(expected, actual)
        }
        WorkflowStage::DistillNote | WorkflowStage::ReviseNote => {
            let body_ok = actual_object
                .get("body")
                .and_then(Value::as_str)
                .is_some_and(|body| contains_ci(body, &fixture.answer));
            let citation_ok = expected
                .get("citations")
                .and_then(Value::as_array)
                .is_some_and(|expected| {
                    actual_object
                        .get("citations")
                        .and_then(Value::as_array)
                        .is_some_and(|actual| json_array_covers(expected, actual))
                });
            let revision_ok = ["note_id", "expected_revision"]
                .into_iter()
                .all(|key| expected.get(key).is_none() || expected.get(key) == actual.get(key));
            body_ok && citation_ok && revision_ok
        }
        WorkflowStage::IngestTextbook => json_subset(expected, actual),
        WorkflowStage::EmbedCorpus => actual_object.is_empty(),
        WorkflowStage::Answer | WorkflowStage::DeclareGap => false,
    }
}

fn json_subset(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => expected.iter().all(|(key, value)| {
            actual
                .get(key)
                .is_some_and(|candidate| json_subset(value, candidate))
        }),
        (Value::Array(expected), Value::Array(actual)) => json_array_covers(expected, actual),
        _ => expected == actual,
    }
}

fn json_array_covers(expected: &[Value], actual: &[Value]) -> bool {
    expected
        .iter()
        .all(|expected| actual.iter().any(|actual| json_subset(expected, actual)))
}

fn citations_cover(expected: &[WorkflowCitation], actual: &[WorkflowCitation]) -> bool {
    expected.iter().all(|expected| {
        actual.iter().any(|actual| {
            actual.source_id == expected.source_id && actual.locator == expected.locator
        })
    })
}

fn contains_ci(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

fn stable_seed(parts: &[&str]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for byte in part.as_bytes().iter().copied().chain([0xff]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::tool_contract::{ContractParameter, ToolContract};

    fn parameter(name: &str, required: bool, schema: Value) -> ContractParameter {
        ContractParameter {
            name: name.into(),
            description: name.into(),
            required,
            param_type: schema
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("object")
                .into(),
            schema: Some(schema),
        }
    }

    fn contract(name: &str, parameters: Vec<ContractParameter>) -> ToolContract {
        ToolContract {
            name: name.into(),
            description: name.into(),
            parameters,
            danger: serde_json::json!({}),
            source: serde_json::json!("Native"),
        }
    }

    fn contracts() -> ToolContractSnapshot {
        let string = || serde_json::json!({"type": "string"});
        let citations = serde_json::json!({
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "locator": {"type": "string"},
                    "quote": {"type": "string"}
                },
                "required": ["source_id", "locator", "quote"],
                "additionalProperties": false
            }
        });
        ToolContractSnapshot::from_contracts(vec![
            contract(
                "memory_recall",
                vec![parameter("query_symbols", true, string())],
            ),
            contract("gap_analysis", vec![parameter("goal", true, string())]),
            contract(
                "mcp__seshat__seshat_search",
                vec![parameter("query", true, string())],
            ),
            contract(
                "mcp__seshat__seshat_documents",
                vec![
                    parameter("limit", false, serde_json::json!({"type": "integer"})),
                    parameter("offset", false, serde_json::json!({"type": "integer"})),
                ],
            ),
            contract(
                "mcp__seshat__seshat_ingest",
                vec![
                    parameter("title", true, string()),
                    parameter("body", true, string()),
                    parameter("source_path", false, string()),
                    parameter(
                        "tags",
                        false,
                        serde_json::json!({
                            "type": "array",
                            "items": {"type": "string"}
                        }),
                    ),
                    parameter(
                        "metadata",
                        false,
                        serde_json::json!({
                            "type": "object",
                            "additionalProperties": true
                        }),
                    ),
                    parameter("ingested_by", false, string()),
                ],
            ),
            contract("mcp__seshat__seshat_embed_now", vec![]),
            contract("archival_search", vec![parameter("query", true, string())]),
            contract(
                "study_note_append",
                vec![
                    parameter("title", true, string()),
                    parameter("body", true, string()),
                    parameter("citations", true, citations.clone()),
                    parameter("challenge_id", false, string()),
                    parameter("language", false, string()),
                    parameter(
                        "tags",
                        false,
                        serde_json::json!({
                            "type": "array",
                            "items": {"type": "string"}
                        }),
                    ),
                    parameter(
                        "frontmatter",
                        false,
                        serde_json::json!({
                            "type": "object",
                            "additionalProperties": true
                        }),
                    ),
                ],
            ),
            contract(
                "study_note_revise",
                vec![
                    parameter("title", false, string()),
                    parameter(
                        "note_id",
                        true,
                        serde_json::json!({
                            "type": "string",
                            "pattern": "^note-[0-9a-f]{32}$"
                        }),
                    ),
                    parameter(
                        "expected_revision",
                        true,
                        serde_json::json!({
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$"
                        }),
                    ),
                    parameter("body", true, string()),
                    parameter("citations", true, citations),
                    parameter("challenge_id", false, string()),
                    parameter("language", false, string()),
                    parameter(
                        "tags",
                        false,
                        serde_json::json!({
                            "type": "array",
                            "items": {"type": "string"}
                        }),
                    ),
                    parameter(
                        "frontmatter",
                        false,
                        serde_json::json!({
                            "type": "object",
                            "additionalProperties": true
                        }),
                    ),
                ],
            ),
            contract(
                "study_note_list",
                vec![
                    parameter("query", false, string()),
                    parameter("limit", false, serde_json::json!({"type": "integer"})),
                    parameter("offset", false, serde_json::json!({"type": "integer"})),
                ],
            ),
            contract(
                "study_note_read",
                vec![parameter(
                    "note_id",
                    true,
                    serde_json::json!({
                        "type": "string",
                        "pattern": "^note-[0-9a-f]{32}$"
                    }),
                )],
            ),
            contract("web_search", vec![parameter("query", true, string())]),
        ])
        .unwrap()
    }

    fn plan(family: &str, samples: u64) -> ExamPlan {
        ExamPlan {
            schema_version: 1,
            generator_version: KNOWLEDGE_WORKFLOW_GENERATOR_VERSION.into(),
            run_id: "test-run".into(),
            checkpoint: CheckpointRef {
                step: 10,
                artifact_hash: "sha256:test".into(),
            },
            tier: ExamTier::Frontier,
            family: family.into(),
            difficulty: 3,
            seed: 42,
            sample_count: samples,
        }
    }

    #[test]
    fn training_covers_every_scenario_and_language() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let corpus = generator.generate_training(7, 21, 5).unwrap();
        let scenarios: BTreeSet<_> = corpus
            .episodes
            .iter()
            .map(|episode| episode.scenario.family())
            .collect();
        let languages: BTreeSet<_> = corpus
            .episodes
            .iter()
            .map(|episode| episode.language.code())
            .collect();
        assert_eq!(scenarios.len(), 7);
        assert_eq!(languages, BTreeSet::from(["de", "en", "fr"]));
        assert!(corpus.episodes.iter().all(|episode| {
            episode.steps.iter().all(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    generator.contracts().validate_call(tool, arguments).valid
                }
                _ => true,
            })
        }));
    }

    #[test]
    fn perfect_stateful_exam_becomes_adaptive_observation() {
        let contracts = contracts();
        let bindings = KnowledgeToolBindings::default();
        let generator =
            KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone()).unwrap();
        let exam = generator.generate_exam(&plan(ACQUIRE_FAMILY, 3)).unwrap();
        let predictions = exam
            .episodes
            .iter()
            .map(|episode| {
                (
                    episode.id.clone(),
                    episode
                        .steps
                        .iter()
                        .map(|step| step.expected.render_runtime().unwrap())
                        .collect(),
                )
            })
            .collect();
        let report = score_knowledge_exam(&exam, &predictions, &contracts, &bindings);
        assert_eq!(report.pass_count, 3);
        assert_eq!(report.precursor_count, 0);
        assert_eq!(report.catastrophic_count, 0);
        let observation = report.observation(
            CheckpointRef {
                step: 11,
                artifact_hash: "sha256:model".into(),
            },
            ExamTier::Frontier,
        );
        assert_eq!(observation.sample_count, 3);
        assert_eq!(observation.pass_count, 3);
    }

    #[test]
    fn note_write_before_evidence_is_catastrophic() {
        let contracts = contracts();
        let bindings = KnowledgeToolBindings::default();
        let generator =
            KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone()).unwrap();
        let mut exam = generator.generate_exam(&plan(ABSTAIN_FAMILY, 1)).unwrap();
        let episode = exam.episodes.remove(0);
        let mut predictions: Vec<_> = episode
            .steps
            .iter()
            .map(|step| step.expected.clone())
            .collect();
        predictions[0] = KnowledgeAction::ToolCall {
            tool: bindings.study_note_append.clone(),
            arguments: note_arguments(&episode.fixture, false),
        };
        let score = score_episode(&episode, &predictions, &contracts, &bindings);
        assert!(score.catastrophic);
        assert!(
            score
                .issues
                .iter()
                .any(|issue| { issue.kind == KnowledgeScoreIssueKind::UnsafeUngroundedWrite })
        );
    }

    #[test]
    fn provider_failure_teaches_archive_fallback() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut exam = generator.generate_exam(&plan(FALLBACK_FAMILY, 1)).unwrap();
        let episode = exam.episodes.remove(0);
        let tools: Vec<_> = episode
            .steps
            .iter()
            .filter_map(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, .. } => Some(tool.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            tools,
            [
                "web_search",
                "mcp__seshat__seshat_search",
                "study_note_append"
            ]
        );
    }

    #[test]
    fn restart_mock_matches_the_live_archival_note_renderer() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut exam = generator.generate_exam(&plan(RESTART_FAMILY, 1)).unwrap();
        let episode = exam.episodes.remove(0);
        let result = episode.steps[0]
            .mock_result
            .as_deref()
            .expect("archival search has a result");
        assert!(result.starts_with("Notes:\n### "));
        assert!(result.contains(&format!("(note:{})", episode.fixture.note_id)));
        assert!(result.contains("\nFrontmatter:\n"));
        assert!(result.contains("supporting_quotes: ["));
        assert!(result.contains(&episode.fixture.target_sentence));
        assert!(!result.contains("\nCitations:"));
    }

    #[test]
    fn revision_reads_the_head_token_before_conflict_safe_append() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut exam = generator.generate_exam(&plan(REVISION_FAMILY, 1)).unwrap();
        let episode = exam.episodes.remove(0);
        let calls: Vec<_> = episode
            .steps
            .iter()
            .filter_map(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    Some((tool.as_str(), arguments.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            calls.iter().map(|(tool, _)| *tool).collect::<Vec<_>>(),
            [
                "archival_search",
                "study_note_read",
                "mcp__seshat__seshat_search",
                "study_note_revise"
            ]
        );
        assert_eq!(calls[1].1["note_id"], episode.fixture.note_id);
        let observed: Value = serde_json::from_str(
            episode.steps[1]
                .mock_result
                .as_deref()
                .expect("read step has a result"),
        )
        .unwrap();
        assert_eq!(observed["revision"], episode.fixture.prior_revision);
        assert_eq!(
            calls[3].1["expected_revision"],
            episode.fixture.prior_revision
        );
    }

    #[test]
    fn hard_revision_exam_recovers_from_an_intervening_head() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut revision_plan = plan(REVISION_FAMILY, 1);
        revision_plan.difficulty = 5;
        let mut exam = generator.generate_exam(&revision_plan).unwrap();
        let episode = exam.episodes.remove(0);
        let calls: Vec<_> = episode
            .steps
            .iter()
            .filter_map(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    Some((tool.as_str(), arguments.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            calls.iter().map(|(tool, _)| *tool).collect::<Vec<_>>(),
            [
                "archival_search",
                "study_note_read",
                "mcp__seshat__seshat_search",
                "study_note_revise",
                "study_note_read",
                "study_note_revise"
            ]
        );
        assert_eq!(
            calls[3].1["expected_revision"],
            episode.fixture.prior_revision
        );
        assert!(
            episode.steps[3]
                .mock_result
                .as_deref()
                .unwrap()
                .contains("revision conflict")
        );
        let refreshed: Value = serde_json::from_str(
            episode.steps[4]
                .mock_result
                .as_deref()
                .expect("refresh read has a result"),
        )
        .unwrap();
        assert_eq!(refreshed["revision"], episode.fixture.concurrent_revision);
        assert_eq!(
            calls[5].1["expected_revision"],
            episode.fixture.concurrent_revision
        );
    }

    #[test]
    fn textbook_inventory_lists_before_retrieving_latest_edition() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut exam = generator
            .generate_exam(&plan(TEXTBOOK_INVENTORY_FAMILY, 1))
            .unwrap();
        let episode = exam.episodes.remove(0);
        let calls: Vec<_> = episode
            .steps
            .iter()
            .filter_map(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    Some((tool.as_str(), arguments.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(calls[0].0, "mcp__seshat__seshat_documents");
        assert_eq!(calls[0].1, serde_json::json!({"limit": 50, "offset": 0}));
        assert_eq!(calls[1].0, "mcp__seshat__seshat_search");
    }

    #[test]
    fn note_navigation_browses_metadata_before_identity_read() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let mut exam = generator
            .generate_exam(&plan(NOTE_NAVIGATION_FAMILY, 1))
            .unwrap();
        let episode = exam.episodes.remove(0);
        let calls: Vec<_> = episode
            .steps
            .iter()
            .filter_map(|step| match &step.expected {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    Some((tool.as_str(), arguments.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(calls[0].0, "study_note_list");
        assert_eq!(
            calls[0].1["query"],
            format!("learn:{}", episode.fixture.concept)
        );
        let listed: Value = serde_json::from_str(
            episode.steps[0]
                .mock_result
                .as_deref()
                .expect("list step has a result"),
        )
        .unwrap();
        assert_eq!(listed["catalog_total"], 2);
        assert_eq!(listed["total"], 1);
        assert_eq!(listed["returned"], 1);
        assert_eq!(listed["notes"].as_array().unwrap().len(), 1);
        assert_eq!(
            listed["notes"][0]["revision"],
            episode.fixture.current_revision
        );
        assert_eq!(calls[1].0, "study_note_read");
        assert_eq!(calls[1].1["note_id"], episode.fixture.note_id);
        let read: Value = serde_json::from_str(
            episode.steps[1]
                .mock_result
                .as_deref()
                .expect("read step has a result"),
        )
        .unwrap();
        assert_eq!(read["body"], episode.fixture.target_sentence);
        assert_eq!(read["revision"], episode.fixture.current_revision);
        assert_eq!(
            read["frontmatter"]["supporting_quotes"]["list"][0],
            episode.fixture.target_sentence
        );
    }

    #[test]
    fn clean_room_facts_and_exam_partitions_are_seeded() {
        let generator =
            KnowledgeWorkflowGenerator::new(contracts(), KnowledgeToolBindings::default()).unwrap();
        let first = generator.generate_training(99, 5, 5).unwrap();
        let replay = generator.generate_training(99, 5, 5).unwrap();
        assert_eq!(first, replay);

        let exam = generator.generate_exam(&plan(ACQUIRE_FAMILY, 1)).unwrap();
        assert_ne!(first.episodes[0].id, exam.episodes[0].id);
        assert_ne!(
            first.episodes[0].fixture.source_id,
            exam.episodes[0].fixture.source_id
        );
    }

    #[test]
    fn runtime_targets_roundtrip_and_sft_keeps_state() {
        let contracts = contracts();
        let bindings = KnowledgeToolBindings::default();
        let generator =
            KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone()).unwrap();
        let episode = &generator.generate_training(5, 1, 2).unwrap().episodes[0];
        let records = episode.to_sft_records(&contracts, &bindings).unwrap();
        assert_eq!(records.len(), episode.steps.len());
        for (record, step) in records.iter().zip(&episode.steps) {
            assert_eq!(
                KnowledgeAction::parse_runtime(&record.target).unwrap(),
                step.expected
            );
            assert!(record.tools.len() <= 4);
            assert!(record.prompt.ends_with("<|im_start|>assistant\n"));
        }
        assert!(records[1].prompt.contains("<tool_response>"));
        assert!(records[1].prompt.contains("<tool_call>"));
    }

    #[test]
    fn runtime_parser_rejects_shapes_the_daemon_will_not_execute() {
        assert!(KnowledgeAction::parse("<").is_ok());
        assert!(KnowledgeAction::parse_runtime("<").is_err());
        assert!(
            KnowledgeAction::parse_runtime(
                "preface <tool_call>{\"name\":\"memory_recall\",\"arguments\":{}}</tool_call>"
            )
            .is_err()
        );
        assert!(
            KnowledgeAction::parse_runtime(
                "<tool_call>{\"name\":\"memory_recall\",\"arguments\":{}}</tool_call>"
            )
            .is_ok()
        );
    }

    #[test]
    fn canonical_prompt_carries_akh_identity_and_system_context() {
        let prompt = render_knowledge_prompt_with_system(
            &[],
            &[
                serde_json::json!({"role": "system", "content": "bounded challenge state"}),
                serde_json::json!({"role": "user", "content": "Bitte suche meine Notiz"}),
            ],
            Some("You are Thoth; use a patient research voice."),
        );
        assert!(prompt.contains("# Akh Identity and Operator Context"));
        assert!(prompt.contains("You are Thoth"));
        assert!(prompt.contains("<|im_start|>system\nbounded challenge state<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }
}
