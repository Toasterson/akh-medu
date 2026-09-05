//! `grug-data` — Phase 51 (Grug-Brain BabyLLM) data generator + eval scorer.
//!
//! Pure Rust, no machine-learning dependency. Two jobs:
//!
//! 1. **Generate** deterministic, seeded JSON tool-calling training examples
//!    with owned ground truth (see [`Generator`], [`ToolCatalog`],
//!    [`Dataset`]).
//! 2. **Score** a model's raw output string against an example's ground
//!    truth, gated level-by-level, plus an anti-collapse diversity alarm
//!    (see [`eval::score_output`], [`eval::EvalReport`],
//!    [`eval::DiversityReport`]).
//!
//! # Quick start
//!
//! ```
//! use grug_data::{Dataset, eval};
//!
//! let dataset = Dataset::new(42).expect("built-in catalog splits cleanly");
//! let train = dataset.train_examples(100).unwrap();
//! let holdout = dataset.holdout_examples(20).unwrap();
//!
//! // Hand a model `train[i].user_request` + `train[i].tools`, get back a
//! // raw string, score it:
//! let raw = train[0].expected_call_as_json_string();
//! let outcome = eval::score_output(&raw, &train[0]);
//! assert!(outcome.arg_correct());
//! ```

pub mod adaptive_exam;
pub mod catalog;
mod catalog_tools;
pub mod constrain;
pub mod dataset;
pub mod error;
pub mod eval;
pub mod format;
pub mod foundation;
pub mod generator;
pub mod ground;
pub mod knowledge_workflow;
pub mod organ;
pub mod phrasing;
pub mod procedural;
pub mod reason;
pub mod repair;
pub mod retriever;
pub mod schema;
pub mod semantic;
pub mod similarity;
pub mod tool_contract;
pub mod tool_curriculum;
pub mod vision_audit;
pub mod vision_text;

pub use adaptive_exam::{
    AdaptiveExamController, AdaptiveExamError, CapabilitySpec, CheckpointRef, ControllerConfig,
    ControllerDecision, ExamObservation, ExamPlan, ExamTier, SiblingSeparationEvidence,
    SiblingSeparationPolicy, TrainingSiblingPlan, WarningLevel, benjamini_hochberg,
    character_ngram_jaccard, derive_training_sibling, perceptual_hash_hamming,
    validate_sibling_separation, wilson_interval,
};
pub use catalog::ToolCatalog;
pub use constrain::{GrammarState, ToolCallGrammar};
pub use dataset::{AxisReport, Dataset, EvalAxis};
pub use error::{GrugError, GrugResult};
pub use eval::{
    AbstentionOutcome, AbstentionReport, ArgMismatch, DiversityReport, EvalOutcome, EvalReport,
    SchemaViolation, score_abstention_fmt,
};
pub use format::{ParsedOutput, TargetFormat, parse_call, parse_output, render_call, render_tools};
pub use foundation::{
    FOUNDATION_CORPUS_GENERATOR_VERSION, FOUNDATION_CORPUS_MANIFEST_SCHEMA_VERSION,
    FOUNDATION_RECORD_SCHEMA_VERSION, FoundationCorpusError, FoundationCorpusManifest,
    FoundationDecontaminationSummary, FoundationRecord, FoundationShard, FoundationSourceSummary,
    FoundationSplit, FoundationSplitManifest, safe_relative_path as safe_foundation_relative_path,
    update_foundation_fingerprint,
};
pub use generator::{DistractorConfig, Generator, PhrasingMode};
pub use ground::{GroundStats, ground_values};
pub use knowledge_workflow::{
    ABSTAIN_FAMILY, ACQUIRE_FAMILY, EvidenceState, FALLBACK_FAMILY,
    KNOWLEDGE_WORKFLOW_GENERATOR_VERSION, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeAction,
    KnowledgeCaseScore, KnowledgeEpisode, KnowledgeEvaluationMode, KnowledgeExamEvidence,
    KnowledgeExamReport, KnowledgeExamSet, KnowledgeScenario, KnowledgeScoreIssue,
    KnowledgeScoreIssueKind, KnowledgeSftRecord, KnowledgeStep, KnowledgeStepScore,
    KnowledgeToolBindings, KnowledgeTrainingCorpus, KnowledgeWorkflowError,
    KnowledgeWorkflowGenerator, NOTE_NAVIGATION_FAMILY, RESTART_FAMILY, REVISION_FAMILY,
    TEXTBOOK_INVENTORY_FAMILY, TextbookFixture, WorkflowCitation, WorkflowLanguage, WorkflowStage,
    knowledge_capability_specs, render_knowledge_prompt, render_knowledge_prompt_with_system,
    score_episode as score_knowledge_episode, score_knowledge_exam,
    score_raw_episode as score_raw_knowledge_episode,
};
pub use organ::{GapRecoveryReport, GapResponse, respond_to_gap};
pub use phrasing::{PhrasingBank, PhrasingRegistry, PhrasingTemplate, TemplatePart};
pub use procedural::{
    DEFAULT_TEMPLATES_PER_TOOL, build_catalog as build_procedural_catalog, build_catalog_signal,
};
pub use reason::{
    AbstentionStats as ReasonAbstentionStats, DepthStats as ReasonDepthStats, EntityId, EntityType,
    Fact as ReasonFact, ReasonConfig, ReasonDataset, ReasonDiversityReport, ReasonError,
    ReasonEvalSummary, ReasonExample, RelationKind, ReplayReport, TraceArm, score_reason_outputs,
};
pub use repair::{RepairStats, repair_call};
pub use retriever::{
    EmbedRetriever, Retriever, SemanticRetriever, embed_recall_at_k, retrieval_recall_at_k,
    semantic_recall_at_k,
};
// Phase 51 P4 slice 1 — the High-Grug-v2 deductive corpus arms.
pub use reason::register::{Op as GrugOp, Slot as GrugSlot, Sort as GrugSort, Truth as GrugTruth};
pub use reason::rule::corpus::{
    ExampleKind, GoalSpec, LogicArm, PropConn, PropLiteral, RuleCorpusConfig, RuleDataset,
    RuleExample,
};
pub use reason::rule::replay::{
    ComponentSummary, RuleDepthStats, RuleEvalSummary, RuleReplayReport, replay_rule,
    score_rule_outputs,
};
pub use reason::rule::world::{Predicate, Rule, RuleWorld, RuleWorldConfig};
pub use schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};
pub use semantic::{SignalMode, describe_tool, glossed_phrase};
pub use similarity::{levenshtein, normalized_similarity, normalized_similarity_ci};
pub use tool_contract::{
    ArgumentValidation, ChangedToolContract, ContractParameter, SchemaIssue, SchemaIssueKind,
    TOOL_CONTRACT_DELTA_SCHEMA_VERSION, TOOL_CONTRACT_SCHEMA_VERSION, ToolContract,
    ToolContractDelta, ToolContractError, ToolContractSnapshot,
};
pub use tool_curriculum::{
    TOOL_CONTRACT_DELTA_CURRICULUM_GENERATOR_VERSION, TOOL_CONTRACT_EXAM_GENERATOR_VERSION,
    TOOL_CONTRACT_EXAM_SCHEMA_VERSION, TOOL_CONTRACT_SIBLING_GENERATOR_VERSION,
    TOOL_CONTRACT_SIBLING_SCHEMA_VERSION, TOOL_CURRICULUM_GENERATOR_VERSION, ToolContractCaseScore,
    ToolContractDeltaCurriculumGenerator, ToolContractEvaluationMode, ToolContractExamCase,
    ToolContractExamEvidence, ToolContractExamReport, ToolContractExamSet,
    ToolContractFamilyReport, ToolContractTrainingSiblingPlan, ToolCurriculumError,
    ToolCurriculumGenerator, derive_tool_contract_training_sibling, score_tool_contract_exam,
    tool_contract_capability_specs,
};
pub use vision_audit::{
    VISION_AUDIT_GENERATOR_VERSION, VisionAuditGenerator, VisionAuditGeneratorError,
};
pub use vision_text::{
    MetricInterval, RenderedVisionTextCase, VisionArtifactProvenance, VisionCorpusDomain,
    VisionCorruption, VisionLanguage, VisionLayout, VisionSeparationReport, VisionSliceScore,
    VisionTextError, VisionTextGenerator, VisionTextManifest, VisionTextPrediction,
    VisionTextScoreSummary, audit_vision_sibling_separation, letterbox_rgb, load_manifest_rgb,
    read_png_normalized_rgb, read_rgb_png, render_glyph_probe, score_vision_text,
    vision_perceptual_hash, vision_pixel_hash, write_png,
};
