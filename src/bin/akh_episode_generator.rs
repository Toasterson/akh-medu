//! Phase 51 P6.3 — engine-in-the-loop episode generator.
//!
//! Builds knowledge- and tool-contract-family training episodes by driving
//! akh-medu's own [`Engine`] (rule engine, event calculus, causal manager)
//! rather than templating strings: for each record the *target* is derived
//! by actually executing the relevant `Tool` against a grounded graph state
//! (symbolic completion, E1), gated by an event-calculus `holds_at`
//! projection (temporal validity, E2), and scored by a real
//! `CausalManager::predict_effects` -> `verify_prediction` round trip
//! (causal step reward, E3). See:
//!
//! - `docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md`
//!   (Decision 3, the P6.3 row, Experiments E1-E4)
//! - `docs/ai/decisions/106-local-burn-metal-lane-and-engine-in-the-loop-training.md`
//! - `docs/ai/plans/2026-09-12-akh-R0-episode-contract.md` (the three
//!   reference worlds the oracle family mirrors)
//! - `examples/akh_symbolic_episode_bridge.rs`,
//!   `examples/akh_temporal_incident_bridge.rs`,
//!   `examples/akh_causal_episode_bridge.rs` (the recovered bridges this
//!   generator's engine plumbing follows)
//!
//! Natural-language prompt phrasing lives entirely in
//! `training/episodes/templates/*.toml` (data files) — this source file
//! never contains an authored word or phrase used for prompt text, per the
//! repo's language-agnostic hard rule (CLAUDE.md / MEMORY.md). Rust code
//! only substitutes `{placeholder}` tokens into template rows.
//!
//! Output: `training/episodes/p6-3-20260913/<family>.jsonl` plus a
//! `manifest.json` (per-family counts, generator git hash, template file
//! hashes, seed, decontamination summary).
//!
//! Run: `cargo run --release --bin akh_episode_generator -- --help`
//! Test: `cargo test --bin akh_episode_generator`

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write as _};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use akh_medu::agent::tasks::{self, TaskPredicates};
use akh_medu::agent::tool_manifest::ToolParamSchema;
use akh_medu::agent::tools::memory_write::MemoryWriteTool;
use akh_medu::agent::tools::reactor_manage::ReactorManageTool;
use akh_medu::agent::tools::structured_task::{StructuredTaskCreateTool, StructuredTaskUpdateTool};
use akh_medu::agent::{
    ActionSchema, AgentError, AgentPredicates, CausalEffect, CausalError, CausalManager,
    EffectKind, Event, EventCalculusEngine, EventCalculusError, PatternElement, Tool, ToolInput,
    ToolManifest,
};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::error::AkhError;
use akh_medu::graph::Triple;
use akh_medu::provenance::{DerivationKind, ProvenanceRecord};
use akh_medu::symbol::SymbolId;
use akh_medu::vsa::Dimension;
use grug_data::{
    KnowledgeAction, KnowledgeSftRecord, KnowledgeWorkflowError, REVISION_FAMILY,
    ToolContractError, ToolContractSnapshot, ToolCurriculumError, WorkflowCitation,
    WorkflowLanguage, WorkflowStage, render_request, sample_offered_tools,
};

// ═══════════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
enum GeneratorError {
    #[error("failed to read template file {path}")]
    #[diagnostic(
        code(akh::episode_gen::template_io),
        help("verify the path exists under training/episodes/templates/")
    )]
    TemplateIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse template file {path}")]
    #[diagnostic(
        code(akh::episode_gen::template_parse),
        help("check the TOML syntax in the named file")
    )]
    TemplateParse {
        path: PathBuf,
        #[source]
        source: Box<toml::de::Error>,
    },

    #[error("no template row for family={family:?} kind={kind:?} language={language:?}")]
    #[diagnostic(
        code(akh::episode_gen::template_missing),
        help("add a [[template]] row to the matching training/episodes/templates/*.toml file")
    )]
    TemplateMissing {
        family: String,
        kind: String,
        language: String,
    },

    #[error(
        "template for family={family:?} kind={kind:?} still has unsubstituted {{{placeholder}}}"
    )]
    #[diagnostic(
        code(akh::episode_gen::template_placeholder),
        help(
            "every {{...}} in the template text needs a matching substitution key from the caller"
        )
    )]
    TemplateUnsubstituted {
        family: String,
        kind: String,
        placeholder: String,
    },

    #[error(
        "generated target for {id} is not runtime-valid (parse_runtime/render_runtime round trip failed)"
    )]
    #[diagnostic(
        code(akh::episode_gen::non_canonical_target),
        help(
            "the completed (treatment) target must be exactly what akh-medu's runtime decoder accepts"
        )
    )]
    NonCanonicalTarget { id: String },

    /// Reserved for a future strict mode that fails the whole run on any
    /// sentinel collision rather than silently dropping the record (today's
    /// behavior, tracked in `DecontaminationSummary`).
    #[allow(dead_code)]
    #[error("record {id} collides with a held-out sentinel ({reason})")]
    #[diagnostic(
        code(akh::episode_gen::sentinel_collision),
        help("regenerate with a different --seed/index; sentinels must never enter training data")
    )]
    SentinelCollision { id: String, reason: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Agent(#[from] AgentError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] AkhError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Causal(#[from] CausalError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Temporal(#[from] EventCalculusError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Task(#[from] tasks::TaskError),

    #[error("grug-data knowledge-workflow error: {0}")]
    #[diagnostic(
        code(akh::episode_gen::knowledge_workflow),
        help("KnowledgeAction::parse_runtime/render_runtime failed on a generated target")
    )]
    KnowledgeWorkflow(#[from] KnowledgeWorkflowError),

    #[error("failed to read frozen tool-contract snapshot {path}")]
    #[diagnostic(
        code(akh::episode_gen::contracts_io),
        help(
            "verify --contracts points at a real contracts.json, e.g. \
             runs/checkpoints/local-curricula-v3-8k/sft/contracts.json"
        )
    )]
    ContractsIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("grug-data tool-contract error: {0}")]
    #[diagnostic(
        code(akh::episode_gen::tool_contract),
        help(
            "the --contracts file failed ToolContractSnapshot::verify_integrity — it may be \
             hand-edited or truncated; regenerate it from the source akh"
        )
    )]
    ToolContract(#[from] ToolContractError),

    #[error("grug-data tool-curriculum error: {0}")]
    #[diagnostic(
        code(akh::episode_gen::tool_curriculum),
        help(
            "sample_offered_tools/render_request rejected an input — check the target tool is \
             present in --contracts and the offered-tool budget (4) fits its size"
        )
    )]
    ToolCurriculum(#[from] ToolCurriculumError),

    #[error("json serialization failed")]
    #[diagnostic(code(akh::episode_gen::json))]
    Json(#[from] serde_json::Error),

    #[error("io error at {path}")]
    #[diagnostic(
        code(akh::episode_gen::io),
        help("check filesystem permissions/disk space for the output directory")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

type GenResult<T> = std::result::Result<T, GeneratorError>;

// ═══════════════════════════════════════════════════════════════════════
// Domain types
// ═══════════════════════════════════════════════════════════════════════

/// One generated-episode family. The four tool-contract families are the
/// P5 release-gate blockers; the three oracle families regenerate the R0
/// reference worlds through the same engine-driven path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Family {
    MemoryWrite,
    StructuredTaskCreate,
    StructuredTaskUpdate,
    ReactorManage,
    OracleAcquire,
    OracleRevise,
    OracleRepair,
    /// P6.5/E4: sourced correction retracting an owned edge
    /// (`Engine::retract_owned_rule_snapshot`) vs. an append-only control.
    KnowledgeRevise,
}

impl Family {
    /// `KnowledgeSftRecord`-style dotted family tag, matching the
    /// `tool_contract.<name>` convention `grug-data`'s tool curriculum
    /// already uses (`crates/grug-data/src/tool_curriculum.rs`).
    fn tag(self) -> &'static str {
        match self {
            Self::MemoryWrite => "tool_contract.memory_write",
            Self::StructuredTaskCreate => "tool_contract.structured_task_create",
            Self::StructuredTaskUpdate => "tool_contract.structured_task_update",
            Self::ReactorManage => "tool_contract.reactor_manage",
            Self::OracleAcquire => "knowledge.oracle_acquire",
            Self::OracleRevise => "knowledge.oracle_revise",
            Self::OracleRepair => "knowledge.oracle_repair",
            // Real grug-data family name (`crates/grug-data/src/
            // knowledge_workflow.rs`), not a locally invented tag — E4's
            // records are training data for that existing revision family.
            Self::KnowledgeRevise => REVISION_FAMILY,
        }
    }

    /// Output JSONL file stem under the run directory. The three oracle
    /// sub-families share one file (they are the 24-episode oracle set).
    fn file_stem(self) -> &'static str {
        match self {
            Self::MemoryWrite => "memory_write",
            Self::StructuredTaskCreate => "structured_task_create",
            Self::StructuredTaskUpdate => "structured_task_update",
            Self::ReactorManage => "reactor_manage",
            Self::OracleAcquire | Self::OracleRevise | Self::OracleRepair => "oracle",
            Self::KnowledgeRevise => "knowledge_revise",
        }
    }
}

/// Newtype for a generated episode's stable identity string, so a bare
/// `String` can't be passed where an id is expected by accident. Not
/// threaded through `EpisodeRecord.id` itself (that field must stay a plain
/// `String` to match `KnowledgeSftRecord`'s shape for the steering
/// sidecar), but kept as the strongly-typed id contract for any future
/// caller (e.g. `DecontaminationSummary`/label lookups) that indexes
/// episodes by id rather than by raw string.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct EpisodeId(String);

impl std::fmt::Display for EpisodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Global generation seed. Newtype so `Seed` and `--count` (both `u64`-ish)
/// cannot be swapped at a call site by accident.
#[derive(Debug, Clone, Copy)]
struct Seed(u64);

/// One training/eval record. Field names mirror
/// `grug_data::knowledge_workflow::KnowledgeSftRecord` (the schema
/// `train_knowledge` reads) so a consumer can lift `prompt`/`target`
/// directly; `stage`/`language` are open strings here rather than that
/// crate's `WorkflowStage`/`WorkflowLanguage` enums because this generator
/// needs stages and languages (Italian, Spanish) the current enums don't
/// carry yet (`grug-data` is not modified by this change — see the P6.3
/// RESULT doc's "known gaps"). `grug-data` has no `deny_unknown_fields` on
/// that struct, so the extra `labels` field here is silently ignored by any
/// reader that only wants the base SFT shape; the compatibility tests below
/// additionally verify the English rows parse byte-for-byte as a real
/// `KnowledgeSftRecord`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpisodeRecord {
    schema_version: u32,
    id: String,
    episode_id: String,
    step: usize,
    stage: String,
    family: String,
    difficulty: u8,
    language: String,
    contract_hash: String,
    tools: Vec<Value>,
    conversation: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system_context: Option<String>,
    prompt: String,
    target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    labels: Option<EpisodeLabels>,
}

/// P6.3 experiment labels (E1-E3). Every field is optional so a partially
/// applicable family (e.g. an oracle episode with no multi-step causal
/// story) doesn't need placeholder values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct EpisodeLabels {
    /// E1 control: the naive/uncompleted proposal a model might emit before
    /// grounding against the graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e1_raw_target: Option<String>,
    /// Whether `e1_raw_target` itself round-trips through the runtime
    /// decoder (`KnowledgeAction::parse_runtime`). `false` means the raw
    /// control is envelope-invalid, not just value-wrong.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e1_raw_target_runtime_valid: Option<bool>,
    /// Whether completion changed the raw proposal's text at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e1_completed_changed_from_raw: Option<bool>,
    /// E2 treatment: event-calculus `holds_at` projection at the episode's
    /// decision time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e2_temporal_validity: Option<bool>,
    /// E2 control: the raw event list (name/timestamp/initiates), undigested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e2_raw_events: Option<Vec<TemporalEventRaw>>,
    /// E3 treatment: `CausalManager::verify_prediction` for the *completed*
    /// target's effects, after the tool actually ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e3_causal_reward_completed: Option<f64>,
    /// E3 control: the same verification for the *raw* target's effects,
    /// which were never actually applied to the graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e3_causal_reward_raw: Option<f64>,
    /// P6.5/E2: which `--e2-prompt` mode produced this record
    /// (`none`/`raw-events`/`projection`) — carried per-record because a
    /// single manifest-level value can't distinguish records once corpora
    /// from different modes are concatenated for training.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e2_prompt_mode: Option<String>,
    /// P6.5/E4: `"treatment"` (sourced correction through
    /// `Engine::retract_owned_rule_snapshot`) or `"control"`
    /// (append-only, no retraction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e4_role: Option<String>,
    /// P6.5/E4: whether the stale owned edge was actually retracted from
    /// the graph before the corrected target was executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e4_retraction_applied: Option<bool>,
    /// P6.5/E4: whether the stale edge is still present in the graph after
    /// the episode's action ran (`true` for the append-only control — the
    /// unresolved contradiction that is the point of the control).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e4_stale_edge_still_present: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemporalEventRaw {
    name: String,
    timestamp: u64,
    initiates: bool,
}

// ═══════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Parser)]
#[command(
    name = "akh_episode_generator",
    about = "Phase 51 P6.3 engine-in-the-loop episode generator"
)]
struct Args {
    /// Target sibling count per blocker family (memory_write,
    /// structured_task_create, structured_task_update, reactor_manage).
    #[arg(long, default_value_t = 2000)]
    count: usize,

    /// Number of seeds per oracle scenario (3 scenarios x this = total
    /// oracle episodes; default 8 x 3 = 24).
    #[arg(long, default_value_t = 8)]
    oracle_seeds: usize,

    /// Global deterministic seed.
    #[arg(long, default_value_t = 20_260_913)]
    seed: u64,

    /// Output directory for the JSONL + manifest.
    #[arg(long, default_value = "training/episodes/p6-3-20260913")]
    out_dir: PathBuf,

    /// Directory holding the `*.toml` prompt-phrasing template files.
    #[arg(long, default_value = "training/episodes/templates")]
    templates_dir: PathBuf,

    /// Optional held-out sentinel JSONL file(s) to decontaminate against
    /// (exact-text + 13-word-shingle, mirroring
    /// `crates/grug-data/examples/prepare_foundation.rs`). Repeatable.
    /// None are checked into this worktree (they live on the training
    /// volume only) — omit this flag to generate with zero sentinels
    /// loaded; the manifest records that honestly.
    #[arg(long = "sentinel")]
    sentinel_paths: Vec<PathBuf>,

    /// Optional baseline JSONL file(s) whose texts are NOT held out: any
    /// 13-word shingle or exact text that also occurs in a baseline file
    /// (tool-contract descriptions, the base SFT corpus) is removed from the
    /// sentinel index before decontamination. Without this, boilerplate the
    /// sentinels share with every training record (e.g. a tool's own
    /// description) makes every generated record collide. Repeatable.
    #[arg(long = "sentinel-baseline")]
    sentinel_baseline_paths: Vec<PathBuf>,

    /// P6.5/E1 control: which target the `<family>/train.sft.jsonl`
    /// steering sidecar carries. `completed` (default) is the P6.3
    /// behavior — the engine-completed, schema-valid, value-exact target.
    /// `raw-valid` swaps in `labels.e1_raw_target` and drops any record
    /// whose raw target does not itself round-trip through
    /// `KnowledgeAction::parse_runtime` (schema-invalid raw targets are not
    /// a fair E1 control — the control must be schema-valid-but-wrong, so
    /// the only difference from the treatment is symbolic completion, not
    /// valid-vs-garbage).
    #[arg(long = "e1-targets", value_enum, default_value_t = E1TargetsMode::Completed)]
    e1_targets: E1TargetsMode,

    /// P6.5/E2 treatment: what the user-turn prompt says about temporal
    /// validity for `reactor_manage` (and `structured_task_update`).
    /// `none` (default) is the P6.3 behavior — the prompt never mentions
    /// the grant/revoke events. `raw-events` lists the raw event
    /// name/timestamp pairs (the E2 control corpus). `projection` states
    /// the event-calculus `holds_at` current-truth projection at decision
    /// time (the E2 treatment). In both non-`none` modes the target
    /// depends on validity: a revoked/closed precondition produces a
    /// `KnowledgeAction::Gap` refusal instead of the normal tool call.
    #[arg(long = "e2-prompt", value_enum, default_value_t = E2PromptMode::None)]
    e2_prompt: E2PromptMode,

    /// P6.5/E4 (stretch): also generate the revision-curriculum family —
    /// paired episodes where a sourced correction retracts an owned edge
    /// through `Engine::retract_owned_rule_snapshot` (treatment) vs. an
    /// append-only sibling that never retracts (control). Off by default
    /// since it is a stretch experiment layered on top of E1/E2.
    #[arg(long = "e4-revision", default_value_t = false)]
    e4_revision: bool,

    /// Number of revision pairs (treatment + control each) to generate
    /// when `--e4-revision` is set.
    #[arg(long = "e4-count", default_value_t = 200)]
    e4_count: usize,

    /// Frozen tool-contract snapshot (`contracts.json`) matching the base
    /// SFT corpus and release-gate exam this run's records will be mixed
    /// alongside — e.g.
    /// `runs/checkpoints/local-curricula-v3-8k/sft/contracts.json`.
    /// Required to emit the `<family>/train.sft.jsonl` steering sidecar
    /// (P6.3b): `train_knowledge` only mixes supplemental records "under
    /// the same tool contract", so a sidecar built from this generator's
    /// own four-tool ad hoc snapshot is silently out of distribution and
    /// gets rejected. Omit this flag to skip sidecar emission entirely —
    /// the primary `<family>.jsonl` files are written either way.
    #[arg(long = "contracts")]
    contracts: Option<PathBuf>,
}

/// P6.5/E1 control mode — see `Args::e1_targets`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum E1TargetsMode {
    Completed,
    RawValid,
}

impl std::fmt::Display for E1TargetsMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Completed => "completed",
            Self::RawValid => "raw-valid",
        })
    }
}

/// P6.5/E2 prompt mode — see `Args::e2_prompt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum E2PromptMode {
    None,
    RawEvents,
    Projection,
}

impl std::fmt::Display for E2PromptMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::None => "none",
            Self::RawEvents => "raw-events",
            Self::Projection => "projection",
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Templates
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize)]
struct TemplateFile {
    #[allow(dead_code)]
    schema_version: u32,
    #[serde(default)]
    template: Vec<TemplateRow>,
    #[serde(default)]
    stress: Option<StressPool>,
}

#[derive(Debug, Deserialize, Clone)]
struct TemplateRow {
    family: String,
    kind: String,
    language: String,
    text: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct StressPool {
    #[serde(default)]
    unicode_chars: Vec<String>,
    #[serde(default)]
    json_tags: Vec<String>,
}

struct TemplateBook {
    rows: Vec<TemplateRow>,
    stress: StressPool,
    /// path (relative to `templates_dir`) -> `blake3:<hex>`.
    file_hashes: BTreeMap<String, String>,
}

const TEMPLATE_FILES: &[&str] = &[
    "memory_write.toml",
    "structured_task.toml",
    "reactor_manage.toml",
    "oracle.toml",
    "knowledge_revise.toml",
];

impl TemplateBook {
    fn load(dir: &Path) -> GenResult<Self> {
        let mut rows = Vec::new();
        let mut stress = StressPool::default();
        let mut file_hashes = BTreeMap::new();
        for name in TEMPLATE_FILES {
            let path = dir.join(name);
            let bytes = fs::read(&path).map_err(|source| GeneratorError::TemplateIo {
                path: path.clone(),
                source,
            })?;
            file_hashes.insert(
                name.to_string(),
                format!("blake3:{}", blake3::hash(&bytes).to_hex()),
            );
            let text = String::from_utf8_lossy(&bytes).into_owned();
            let parsed: TemplateFile =
                toml::from_str(&text).map_err(|source| GeneratorError::TemplateParse {
                    path: path.clone(),
                    source: Box::new(source),
                })?;
            rows.extend(parsed.template);
            if let Some(s) = parsed.stress {
                stress = s;
            }
        }
        Ok(Self {
            rows,
            stress,
            file_hashes,
        })
    }

    fn render(
        &self,
        family: &str,
        kind: &str,
        language: &str,
        vars: &[(&str, &str)],
    ) -> GenResult<String> {
        let row = self
            .rows
            .iter()
            .find(|r| r.family == family && r.kind == kind && r.language == language)
            .ok_or_else(|| GeneratorError::TemplateMissing {
                family: family.into(),
                kind: kind.into(),
                language: language.into(),
            })?;
        // Validate every `{key}` placeholder *in the original template
        // text* against the supplied vars before substituting — checking
        // the post-substitution text instead would false-positive on any
        // substituted value that itself contains braces (e.g. the nested
        // JSON copy-fidelity values `generate_memory_write` renders).
        let mut cursor = 0usize;
        while let Some(rel_start) = row.text[cursor..].find('{') {
            let start = cursor + rel_start;
            let Some(rel_end) = row.text[start..].find('}') else {
                break;
            };
            let end = start + rel_end;
            let key = &row.text[start + 1..end];
            if !vars.iter().any(|(k, _)| *k == key) {
                return Err(GeneratorError::TemplateUnsubstituted {
                    family: family.into(),
                    kind: kind.into(),
                    placeholder: format!("{{{key}}}"),
                });
            }
            cursor = end + 1;
        }
        let mut text = row.text.clone();
        for (key, value) in vars {
            text = text.replace(&format!("{{{key}}}"), value);
        }
        Ok(text)
    }

    /// Deterministically pick a unicode stress character by index.
    fn unicode_char(&self, index: usize) -> &str {
        if self.stress.unicode_chars.is_empty() {
            "?"
        } else {
            &self.stress.unicode_chars[index % self.stress.unicode_chars.len()]
        }
    }

    fn json_tag(&self, index: usize) -> &str {
        if self.stress.json_tags.is_empty() {
            "tag"
        } else {
            &self.stress.json_tags[index % self.stress.json_tags.len()]
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Decontamination (mirrors crates/grug-data/examples/prepare_foundation.rs)
// ═══════════════════════════════════════════════════════════════════════

const SENTINEL_WORD_NGRAM_SIZE: usize = 13;

fn normalized_words(text: &str) -> Vec<String> {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn normalized_text_hash(text: &str) -> [u8; 32] {
    let normalized = normalized_words(text).join(" ");
    *blake3::hash(normalized.as_bytes()).as_bytes()
}

fn word_ngram_hashes(text: &str, size: usize) -> Vec<[u8; 32]> {
    let words = normalized_words(text);
    if words.len() < size {
        return Vec::new();
    }
    words
        .windows(size)
        .map(|w| *blake3::hash(w.join(" ").as_bytes()).as_bytes())
        .collect()
}

#[derive(Debug, Default, Serialize)]
struct DecontaminationSummary {
    method: String,
    word_ngram_size: usize,
    sentinel_files: usize,
    sentinel_texts_indexed: usize,
    baseline_files: usize,
    baseline_texts_indexed: usize,
    sentinel_ngrams_removed_by_baseline: usize,
    sentinel_exact_removed_by_baseline: usize,
    records_checked: u64,
    records_dropped_exact: u64,
    records_dropped_ngram: u64,
}

struct SentinelIndex {
    exact: HashSet<[u8; 32]>,
    ngram: HashSet<[u8; 32]>,
    files: usize,
    texts: usize,
    /// Baseline files consulted (see [`SentinelIndex::subtract_baseline`]).
    baseline_files: usize,
    baseline_texts: usize,
    ngrams_removed_by_baseline: usize,
    exact_removed_by_baseline: usize,
}

impl SentinelIndex {
    fn empty() -> Self {
        Self {
            exact: HashSet::new(),
            ngram: HashSet::new(),
            files: 0,
            texts: 0,
            baseline_files: 0,
            baseline_texts: 0,
            ngrams_removed_by_baseline: 0,
            exact_removed_by_baseline: 0,
        }
    }

    /// Removes from this index every exact-text and 13-word-shingle hash that
    /// also occurs in `baseline_paths`. Baseline text is, by definition, not
    /// held out (tool descriptions repeated in every contract prompt, the base
    /// SFT corpus the model already trains on), so a collision with it is not
    /// sentinel leakage. Only the genuinely sentinel-private shingles survive.
    fn subtract_baseline(mut self, baseline_paths: &[PathBuf]) -> GenResult<Self> {
        if baseline_paths.is_empty() {
            return Ok(self);
        }
        let baseline = Self::load(baseline_paths)?;
        let before_ngram = self.ngram.len();
        let before_exact = self.exact.len();
        self.ngram.retain(|hash| !baseline.ngram.contains(hash));
        self.exact.retain(|hash| !baseline.exact.contains(hash));
        self.ngrams_removed_by_baseline = before_ngram - self.ngram.len();
        self.exact_removed_by_baseline = before_exact - self.exact.len();
        self.baseline_files = baseline.files;
        self.baseline_texts = baseline.texts;
        Ok(self)
    }

    fn load(paths: &[PathBuf]) -> GenResult<Self> {
        let mut index = Self::empty();
        for path in paths {
            let file = File::open(path).map_err(|source| GeneratorError::TemplateIo {
                path: path.clone(),
                source,
            })?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|source| GeneratorError::TemplateIo {
                    path: path.clone(),
                    source,
                })?;
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                for key in ["prompt", "target", "user_request", "request", "text"] {
                    if let Some(text) = value.get(key).and_then(Value::as_str) {
                        index.index_text(text);
                    }
                }
                if let Some(call) = value.get("expected_call")
                    && let Ok(rendered) = serde_json::to_string(call)
                {
                    index.index_text(&rendered);
                }
            }
            index.files += 1;
        }
        Ok(index)
    }

    fn index_text(&mut self, text: &str) {
        self.exact.insert(normalized_text_hash(text));
        self.ngram
            .extend(word_ngram_hashes(text, SENTINEL_WORD_NGRAM_SIZE));
        self.texts += 1;
    }

    /// Returns `Some(reason)` if `text` collides with an indexed sentinel.
    fn collision(&self, text: &str) -> Option<&'static str> {
        if self.exact.contains(&normalized_text_hash(text)) {
            return Some("exact");
        }
        if word_ngram_hashes(text, SENTINEL_WORD_NGRAM_SIZE)
            .iter()
            .any(|h| self.ngram.contains(h))
        {
            return Some("13-word-shingle");
        }
        None
    }

    /// Checks every text-bearing field of a record; `None` means clean.
    fn record_collision(&self, record: &EpisodeRecord) -> Option<&'static str> {
        self.collision(&record.prompt)
            .or_else(|| self.collision(&record.target))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Scratch engine directory (std-only RAII temp dir — `tempfile` is a
// dev-dependency, not visible to `src/bin/*.rs` targets)
// ═══════════════════════════════════════════════════════════════════════

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new(label: &str) -> GenResult<Self> {
        let path = std::env::temp_dir().join(format!(
            "akh-episode-gen-{}-{}-{}",
            std::process::id(),
            label,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&path).map_err(|source| GeneratorError::Io {
            path: path.clone(),
            source,
        })?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn build_engine(dir: &Path) -> GenResult<Engine> {
    Ok(Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        data_dir: Some(dir.to_path_buf()),
        ..EngineConfig::default()
    })?)
}

// ═══════════════════════════════════════════════════════════════════════
// Tool-definition rendering (OpenAI function-call shape)
// ═══════════════════════════════════════════════════════════════════════

fn openai_tool_def(manifest: &ToolManifest) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for p in &manifest.parameters {
        let schema = param_schema_json(p);
        properties.insert(p.name.clone(), schema);
        if p.required {
            required.push(Value::String(p.name.clone()));
        }
    }
    json!({
        "type": "function",
        "function": {
            "name": manifest.name,
            "description": manifest.description,
            "parameters": {
                "type": "object",
                "properties": Value::Object(properties),
                "required": required,
            }
        }
    })
}

fn param_schema_json(p: &ToolParamSchema) -> Value {
    if let Some(schema) = &p.schema {
        let mut schema = schema.clone();
        if let Some(obj) = schema.as_object_mut() {
            obj.entry("description")
                .or_insert_with(|| Value::String(p.description.clone()));
        }
        schema
    } else {
        json!({"type": p.param_type, "description": p.description})
    }
}

fn tool_call_target(id: &str, tool: &str, arguments: Value) -> GenResult<String> {
    let action = KnowledgeAction::ToolCall {
        tool: tool.to_string(),
        arguments,
    };
    let rendered = action.render_runtime()?;
    let round_trip = KnowledgeAction::parse_runtime(&rendered).and_then(|a| a.render_runtime());
    match round_trip {
        Ok(ref r) if *r == rendered => {}
        _ => return Err(GeneratorError::NonCanonicalTarget { id: id.into() }),
    }
    Ok(rendered)
}

/// Build a deliberately envelope-broken raw target: free text leaking
/// outside the `<tool_call>` tags, exactly the documented `reactor_manage`
/// failure mode (`release-v3-failure-analysis.md`'s French case). Never
/// executed against the engine — text only, for the E1 control field.
fn malformed_envelope_target(tool: &str, arguments: &Value) -> String {
    let call = serde_json::to_string(&json!({"name": tool, "arguments": arguments}))
        .unwrap_or_else(|_| "{}".into());
    format!("Done.\n<tool_call>\n{call}\n</tool_call>")
}

// ═══════════════════════════════════════════════════════════════════════
// Labelling helpers (E2 temporal, E3 causal)
// ═══════════════════════════════════════════════════════════════════════

/// Record a grant (and optional revoke) event on a fresh fluent, then
/// project `holds_at` at `decision_time`. Returns the projection plus the
/// raw event list (E2 control).
fn temporal_label(
    ec: &mut EventCalculusEngine,
    engine: &Engine,
    fluent_label: &str,
    grant_time: u64,
    revoke_time: Option<u64>,
    decision_time: u64,
) -> GenResult<(bool, Vec<TemporalEventRaw>)> {
    let fluent = engine.resolve_or_create_entity(fluent_label)?;
    let grant_id = engine.resolve_or_create_entity(&format!("{fluent_label}:grant"))?;
    ec.record_event(
        Event {
            symbol_id: grant_id,
            name: format!("{fluent_label}:grant"),
            timestamp: grant_time,
            initiates: vec![fluent],
            terminates: vec![],
        },
        engine,
    )?;
    let mut raw = vec![TemporalEventRaw {
        name: "grant".into(),
        timestamp: grant_time,
        initiates: true,
    }];
    if let Some(revoke_time) = revoke_time {
        let revoke_id = engine.resolve_or_create_entity(&format!("{fluent_label}:revoke"))?;
        ec.record_event(
            Event {
                symbol_id: revoke_id,
                name: format!("{fluent_label}:revoke"),
                timestamp: revoke_time,
                initiates: vec![],
                terminates: vec![fluent],
            },
            engine,
        )?;
        raw.push(TemporalEventRaw {
            name: "revoke".into(),
            timestamp: revoke_time,
            initiates: false,
        });
    }
    Ok((ec.holds_at(fluent, decision_time), raw))
}

/// Register a single-effect causal schema under a unique name and verify it
/// against the *current* engine state. `effect` is a concrete `(s, p, o)`
/// triple this action is claimed to assert.
fn causal_label(
    causal: &mut CausalManager,
    engine: &Engine,
    unique_action_name: &str,
    effect: (SymbolId, SymbolId, SymbolId),
) -> GenResult<f64> {
    let action_id = engine.resolve_or_create_entity(&format!("action:{unique_action_name}"))?;
    let schema = ActionSchema {
        action_id,
        name: unique_action_name.to_string(),
        description: "P6.3 generated action schema".into(),
        parameters: vec![],
        preconditions: vec![],
        effects: vec![CausalEffect {
            kind: EffectKind::Assert,
            subject: PatternElement::Concrete(effect.0),
            predicate: PatternElement::Concrete(effect.1),
            object: PatternElement::Concrete(effect.2),
            confidence: 1.0,
        }],
        success_rate: 0.0,
        execution_count: 0,
    };
    causal.register_action_schema(schema);
    let transition = causal.predict_effects(unique_action_name, engine)?;
    Ok(if causal.verify_prediction(&transition, engine) {
        1.0
    } else {
        0.0
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Family: memory_write
// ═══════════════════════════════════════════════════════════════════════

enum ValueKind {
    Hex,
    NestedJson,
    Unicode,
}

fn value_kind(i: usize) -> ValueKind {
    match i % 3 {
        0 => ValueKind::Hex,
        1 => ValueKind::NestedJson,
        _ => ValueKind::Unicode,
    }
}

fn stress_value(book: &TemplateBook, kind: &ValueKind, seed: u64, i: usize) -> String {
    match kind {
        ValueKind::Hex => {
            let hash = blake3::hash(format!("hex-seed:{seed}:{i}").as_bytes());
            hash.to_hex()[..32].to_string()
        }
        ValueKind::NestedJson => {
            let a = (seed as usize + i) % 256;
            let b = i % 256;
            let port = 8000 + (i % 1000);
            json!({
                "ip": format!("10.{a}.{b}.7"),
                "port": port,
                "tags": [book.json_tag(i), book.json_tag(i + 1)],
            })
            .to_string()
        }
        ValueKind::Unicode => {
            let a = book.unicode_char(i);
            let b = book.unicode_char(i + 1);
            let c = book.unicode_char(i + 2);
            format!("value-{seed}-{i}-{a}{b}{c}-#{i}")
        }
    }
}

/// A corrupted echo of `value`, standing in for what an under-trained model
/// emits: truncated and missing the tail (drops nested structure / hex
/// suffix / unicode tail alike). Never a hand-authored phrase — a
/// structural transform of the ground-truth value.
fn corrupt_copy(value: &str) -> String {
    let half = (value.chars().count() / 2).max(1);
    let truncated: String = value.chars().take(half).collect();
    truncated
}

#[allow(clippy::too_many_arguments)]
fn generate_memory_write(
    engine: &Engine,
    agent_preds: &AgentPredicates,
    causal: &mut CausalManager,
    ec: &mut EventCalculusEngine,
    book: &TemplateBook,
    manifest: &ToolManifest,
    seed: Seed,
    count: usize,
) -> GenResult<Vec<EpisodeRecord>> {
    let tool = MemoryWriteTool::new(agent_preds.clone());
    let tools = vec![openai_tool_def(manifest)];
    let contract_hash = format!(
        "blake3:{}",
        blake3::hash(manifest.description.as_bytes()).to_hex()
    );
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        let seed_v = seed.0;
        let doc_label = format!("doc:mw:{seed_v}:{i}");
        let predicate = "has-field";
        let value = stress_value(book, &value_kind(i), seed_v, i);

        // Ground the source fact directly in the graph — this is the
        // "source of truth" the model's fact must copy verbatim.
        let doc_id = engine.resolve_or_create_entity(&doc_label)?;
        let predicate_id = engine.resolve_or_create_relation(predicate)?;
        let value_id = engine.resolve_or_create_entity(&format!("literal:{value}"))?;
        let mut source_triples = vec![Triple::new(doc_id, predicate_id, value_id)];
        let mut prov =
            ProvenanceRecord::new(doc_id, DerivationKind::Extracted).with_confidence(1.0);
        engine.add_triples_with_provenance(&mut source_triples, &mut prov)?;

        let id = format!("p6-3-memory_write-{seed_v}-{i:05}");
        let excerpt = corrupt_copy(&value); // shown as the "source excerpt" is the ground truth; the corrupted copy models the failure mode below
        let request = book.render(
            "memory_write",
            "record_field",
            "en",
            &[
                ("doc_label", &doc_label),
                ("source_excerpt", &value),
                ("predicate", predicate),
            ],
        )?;

        // RAW (E1 control): value copied from a truncated echo, not the
        // graph-verified literal — models the documented copy-fidelity
        // failure exactly (schema-valid envelope, wrong value).
        let raw_args = json!({
            "summary": format!("Recorded {doc_label} {predicate}"),
            "facts": [{"subject": doc_label, "predicate": predicate, "object": excerpt}],
        });
        let raw_target = tool_call_target(&format!("{id}-raw"), "memory_write", raw_args)?;
        let raw_runtime_valid = KnowledgeAction::parse_runtime(&raw_target).is_ok();

        // COMPLETED (E1 treatment): value read back from the graph
        // (`engine.resolve_label`), then the tool is actually executed and
        // must succeed — symbolic completion, not a template guess.
        let ground_truth = engine.resolve_label(value_id);
        let ground_truth_value = ground_truth
            .strip_prefix("literal:")
            .unwrap_or(&ground_truth)
            .to_string();
        let completed_args = json!({
            "summary": format!("Recorded {doc_label} {predicate}"),
            "facts": [{"subject": doc_label, "predicate": predicate, "object": ground_truth_value}],
            "tags": ["copy-fidelity"],
            "source": doc_label,
        });
        let completed_target = tool_call_target(
            &format!("{id}-completed"),
            "memory_write",
            completed_args.clone(),
        )?;
        let output = tool.execute(
            engine,
            ToolInput::new()
                .with_value("summary", completed_args["summary"].clone())
                .with_value("facts", completed_args["facts"].clone())
                .with_value("tags", completed_args["tags"].clone())
                .with_value("source", completed_args["source"].clone()),
        )?;
        if !output.success {
            return Err(GeneratorError::Agent(AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: format!("completed target execution failed for {id}"),
            }));
        }

        // E2: is the source still "current" at decision time? 1-in-4
        // episodes model a superseded source (write should be considered
        // stale), the rest a still-current one.
        let fluent_label = format!("source-current:{doc_label}");
        let (temporal_validity, raw_events) = if i % 4 == 3 {
            temporal_label(ec, engine, &fluent_label, 0, Some(50), 100)?
        } else {
            temporal_label(ec, engine, &fluent_label, 0, None, 10)?
        };

        // E3: causal reward for the completed vs. raw fact triple.
        let object_completed =
            engine.resolve_or_create_entity(&format!("literal:{ground_truth_value}"))?;
        let object_raw = engine.resolve_or_create_entity(&format!("literal:{excerpt}"))?;
        let reward_completed = causal_label(
            causal,
            engine,
            &format!("{id}:completed"),
            (doc_id, predicate_id, object_completed),
        )?;
        let reward_raw = causal_label(
            causal,
            engine,
            &format!("{id}:raw"),
            (doc_id, predicate_id, object_raw),
        )?;

        out.push(EpisodeRecord {
            schema_version: 1,
            id: id.clone(),
            episode_id: id.clone(),
            step: 0,
            stage: "answer".into(),
            family: Family::MemoryWrite.tag().into(),
            difficulty: 1,
            language: "english".into(),
            contract_hash: contract_hash.clone(),
            tools: tools.clone(),
            conversation: vec![json!({"role": "user", "content": request})],
            system_context: None,
            prompt: akh_prompt(&tools, &[json!({"role": "user", "content": request})]),
            target: completed_target,
            labels: Some(EpisodeLabels {
                e1_raw_target: Some(raw_target),
                e1_raw_target_runtime_valid: Some(raw_runtime_valid),
                e1_completed_changed_from_raw: Some(excerpt != ground_truth_value),
                e2_temporal_validity: Some(temporal_validity),
                e2_raw_events: Some(raw_events),
                e3_causal_reward_completed: Some(reward_completed),
                e3_causal_reward_raw: Some(reward_raw),
                ..Default::default()
            }),
        });
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════
// Family: structured_task_create / structured_task_update
// ═══════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
fn generate_structured_task(
    engine: &Engine,
    task_preds: &TaskPredicates,
    agent_preds: &AgentPredicates,
    causal: &mut CausalManager,
    ec: &mut EventCalculusEngine,
    book: &TemplateBook,
    create_manifest: &ToolManifest,
    update_manifest: &ToolManifest,
    seed: Seed,
    count: usize,
    e2_prompt_mode: E2PromptMode,
) -> GenResult<(Vec<EpisodeRecord>, Vec<EpisodeRecord>)> {
    let create_tool = StructuredTaskCreateTool::new(task_preds.clone(), agent_preds.clone());
    let update_tool = StructuredTaskUpdateTool::new(task_preds.clone());
    // Both tools offered together in both directions — the discriminator is
    // the grounded engine state, not tool availability.
    let tools = vec![
        openai_tool_def(create_manifest),
        openai_tool_def(update_manifest),
    ];
    let contract_hash = format!(
        "blake3:{}",
        blake3::hash(
            format!(
                "{}{}",
                create_manifest.description, update_manifest.description
            )
            .as_bytes()
        )
        .to_hex()
    );

    let mut creates = Vec::with_capacity(count);
    let mut updates = Vec::with_capacity(count);

    for i in 0..count {
        let seed_v = seed.0;

        // --- create sibling: no task with this title exists yet ---------
        let create_title = format!("Task-create-{seed_v}-{i}");
        let create_due = format!("2027-01-{:02}", 1 + (i % 27));
        let create_id = format!("p6-3-structured_task_create-{seed_v}-{i:05}");
        let create_request = book.render(
            "structured_task_create",
            "create_due",
            "en",
            &[("title", &create_title), ("due", &create_due)],
        )?;
        let raw_create_args = json!({"title": create_title});
        let raw_create_target = tool_call_target(
            &format!("{create_id}-raw"),
            "structured_task_create",
            raw_create_args.clone(),
        )?;
        let completed_create_args = json!({"title": create_title, "due": create_due});
        let completed_create_target = tool_call_target(
            &format!("{create_id}-completed"),
            "structured_task_create",
            completed_create_args.clone(),
        )?;
        let create_output = create_tool.execute(
            engine,
            ToolInput::new()
                .with_value("title", completed_create_args["title"].clone())
                .with_value("due", completed_create_args["due"].clone()),
        )?;
        if !create_output.success || create_output.symbols_involved.is_empty() {
            return Err(GeneratorError::Agent(AgentError::ToolExecution {
                tool_name: "structured_task_create".into(),
                message: format!("completed create failed for {create_id}"),
            }));
        }
        let created_task_id = create_output.symbols_involved[0];

        let fluent_label = format!("task-open:{create_title}");
        let (create_validity, create_events) =
            temporal_label(ec, engine, &fluent_label, 0, None, 5)?;
        let title_lit = engine.resolve_or_create_entity(&format!("literal:{create_title}"))?;
        let reward_completed = causal_label(
            causal,
            engine,
            &format!("{create_id}:completed"),
            (created_task_id, task_preds.has_title, title_lit),
        )?;
        // Raw control claims the same title triple was asserted by *itself*
        // with no due date; check the due-date effect specifically instead,
        // which the raw proposal never establishes.
        let due_sym = engine.resolve_or_create_entity(&format!("date:{create_due}"))?;
        let phantom_task = engine.resolve_or_create_entity(&format!("phantom-task:{create_id}"))?;
        let reward_raw = causal_label(
            causal,
            engine,
            &format!("{create_id}:raw"),
            (phantom_task, task_preds.has_due, due_sym),
        )?;

        creates.push(EpisodeRecord {
            schema_version: 1,
            id: create_id.clone(),
            episode_id: create_id.clone(),
            step: 0,
            stage: "answer".into(),
            family: Family::StructuredTaskCreate.tag().into(),
            difficulty: 1,
            language: "english".into(),
            contract_hash: contract_hash.clone(),
            tools: tools.clone(),
            conversation: vec![json!({"role": "user", "content": create_request})],
            system_context: None,
            prompt: akh_prompt(
                &tools,
                &[json!({"role": "user", "content": create_request})],
            ),
            target: completed_create_target,
            labels: Some(EpisodeLabels {
                e1_raw_target: Some(raw_create_target.clone()),
                e1_raw_target_runtime_valid: Some(
                    KnowledgeAction::parse_runtime(&raw_create_target).is_ok(),
                ),
                e1_completed_changed_from_raw: Some(raw_create_args != completed_create_args),
                e2_temporal_validity: Some(create_validity),
                e2_raw_events: Some(create_events),
                e3_causal_reward_completed: Some(reward_completed),
                e3_causal_reward_raw: Some(reward_raw),
                ..Default::default()
            }),
        });

        // --- update sibling: task already exists (grounded precondition) ---
        let update_title = format!("Task-update-{seed_v}-{i}");
        let update_create = create_tool.execute(
            engine,
            ToolInput::new().with_value("title", json!(update_title)),
        )?;
        let existing_task_id = update_create.symbols_involved[0];
        let numeric_id = existing_task_id.get().to_string();
        let new_priority = if i % 2 == 0 { "high" } else { "low" };
        let update_id = format!("p6-3-structured_task_update-{seed_v}-{i:05}");

        // Decide + project validity *before* the prompt/target, same
        // rationale as `generate_reactor_manage`'s E2 gating: in
        // `raw-events`/`projection` mode the prompt actually carries the
        // evidence, so the target must agree with it.
        let is_closed = i % 5 == 4;
        let grant_time: u64 = 0;
        let revoke_time = if is_closed { Some(3) } else { None };
        let decision_time = if is_closed { 10 } else { 5 };
        let fluent_label = format!("task-open:{update_title}");
        let (update_validity, update_events) = temporal_label(
            ec,
            engine,
            &fluent_label,
            grant_time,
            revoke_time,
            decision_time,
        )?;

        let update_request = match e2_prompt_mode {
            E2PromptMode::None => book.render(
                "structured_task_update",
                "update_priority",
                "en",
                &[
                    ("title", &update_title),
                    ("id", &numeric_id),
                    ("priority", new_priority),
                ],
            )?,
            E2PromptMode::RawEvents if is_closed => book.render(
                "structured_task_update",
                "update_priority_raw_events_closed",
                "en",
                &[
                    ("title", &update_title),
                    ("id", &numeric_id),
                    ("priority", new_priority),
                    ("grant_time", &grant_time.to_string()),
                    ("revoke_time", &revoke_time.unwrap().to_string()),
                ],
            )?,
            E2PromptMode::RawEvents => book.render(
                "structured_task_update",
                "update_priority_raw_events_open",
                "en",
                &[
                    ("title", &update_title),
                    ("id", &numeric_id),
                    ("priority", new_priority),
                    ("grant_time", &grant_time.to_string()),
                ],
            )?,
            E2PromptMode::Projection if is_closed => book.render(
                "structured_task_update",
                "update_priority_projection_closed",
                "en",
                &[
                    ("title", &update_title),
                    ("id", &numeric_id),
                    ("priority", new_priority),
                ],
            )?,
            E2PromptMode::Projection => book.render(
                "structured_task_update",
                "update_priority_projection_open",
                "en",
                &[
                    ("title", &update_title),
                    ("id", &numeric_id),
                    ("priority", new_priority),
                ],
            )?,
        };
        // RAW: the naive proposal before graph lookup, carrying a
        // placeholder id instead of the real numeric one. Orthogonal to E2
        // — always the E1 control regardless of mode.
        let raw_update_args = json!({"id": "<task-id>", "priority": new_priority});
        let raw_update_target = tool_call_target(
            &format!("{update_id}-raw"),
            "structured_task_update",
            raw_update_args.clone(),
        )?;

        let gate_on_validity = e2_prompt_mode != E2PromptMode::None;
        let stage;
        let completed_update_target;
        if gate_on_validity && !update_validity {
            stage = "declare_gap";
            completed_update_target = KnowledgeAction::Gap {
                missing: vec!["task-open".into()],
                reason: format!(
                    "task {numeric_id} was closed before decision time; not updating its priority"
                ),
            }
            .render_runtime()?;
            // The priority-change effect must NOT have been applied — skip
            // `update_tool.execute` entirely so the causal check below is a
            // genuine negative signal, not a post-hoc label on an action
            // that actually ran.
        } else {
            stage = "answer";
            // COMPLETED: symbolic completion resolves the placeholder to the
            // real numeric id from the graph, then the tool actually runs.
            let completed_update_args = json!({"id": numeric_id, "priority": new_priority});
            completed_update_target = tool_call_target(
                &format!("{update_id}-completed"),
                "structured_task_update",
                completed_update_args.clone(),
            )?;
            let update_output = update_tool.execute(
                engine,
                ToolInput::new()
                    .with_value("id", completed_update_args["id"].clone())
                    .with_value("priority", completed_update_args["priority"].clone()),
            )?;
            if !update_output.success {
                return Err(GeneratorError::Agent(AgentError::ToolExecution {
                    tool_name: "structured_task_update".into(),
                    message: format!("completed update failed for {update_id}"),
                }));
            }
        }

        let priority_sym =
            engine.resolve_or_create_entity(&format!("task:priority:{new_priority}"))?;
        let update_reward_completed = causal_label(
            causal,
            engine,
            &format!("{update_id}:completed"),
            (existing_task_id, task_preds.has_priority, priority_sym),
        )?;
        let phantom_update =
            engine.resolve_or_create_entity(&format!("phantom-task:{update_id}"))?;
        let update_reward_raw = causal_label(
            causal,
            engine,
            &format!("{update_id}:raw"),
            (phantom_update, task_preds.has_priority, priority_sym),
        )?;

        updates.push(EpisodeRecord {
            schema_version: 1,
            id: update_id.clone(),
            episode_id: update_id.clone(),
            step: 0,
            stage: stage.into(),
            family: Family::StructuredTaskUpdate.tag().into(),
            difficulty: 1,
            language: "english".into(),
            contract_hash: contract_hash.clone(),
            tools: tools.clone(),
            conversation: vec![json!({"role": "user", "content": update_request})],
            system_context: None,
            prompt: akh_prompt(
                &tools,
                &[json!({"role": "user", "content": update_request})],
            ),
            target: completed_update_target,
            labels: Some(EpisodeLabels {
                e1_raw_target: Some(raw_update_target.clone()),
                e1_raw_target_runtime_valid: Some(
                    KnowledgeAction::parse_runtime(&raw_update_target).is_ok(),
                ),
                e1_completed_changed_from_raw: Some(true),
                e2_temporal_validity: Some(update_validity),
                e2_raw_events: Some(update_events),
                e2_prompt_mode: Some(e2_prompt_mode.to_string()),
                e3_causal_reward_completed: Some(update_reward_completed),
                e3_causal_reward_raw: Some(update_reward_raw),
                ..Default::default()
            }),
        });
    }
    Ok((creates, updates))
}

// ═══════════════════════════════════════════════════════════════════════
// Family: reactor_manage (multilingual)
// ═══════════════════════════════════════════════════════════════════════

const REACTOR_LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("de", "german"),
    ("fr", "french"),
    ("it", "italian"),
    ("es", "spanish"),
];

#[allow(clippy::too_many_arguments)]
fn generate_reactor_manage(
    engine: &Engine,
    causal: &mut CausalManager,
    ec: &mut EventCalculusEngine,
    book: &TemplateBook,
    manifest: &ToolManifest,
    seed: Seed,
    count: usize,
    e2_prompt_mode: E2PromptMode,
) -> GenResult<Vec<EpisodeRecord>> {
    let tools = vec![openai_tool_def(manifest)];
    let contract_hash = format!(
        "blake3:{}",
        blake3::hash(manifest.description.as_bytes()).to_hex()
    );
    let mut out = Vec::with_capacity(count);
    let reactor_added = engine.resolve_or_create_entity("reactor:added")?;
    let reactor_added_pred = engine.resolve_or_create_relation("reactor:has-state")?;

    for i in 0..count {
        let seed_v = seed.0;
        let (lang_code, lang_name) = REACTOR_LANGUAGES[i % REACTOR_LANGUAGES.len()];
        let name = format!("schedule-{seed_v}-{i}");
        let every_seconds = 60 + (i % 3600);
        let id = format!("p6-3-reactor_manage-{seed_v}-{i:05}");

        // Decide + project temporal validity *before* rendering the prompt
        // or the target: in `raw-events`/`projection` mode both the prompt
        // text and the target depend on this outcome (E2 gates the target,
        // not just the label, once the prompt actually carries the
        // evidence). In `none` mode this is unchanged from P6.3 — a label
        // only, the target is always `add`.
        let is_revoked = i % 6 == 5;
        let grant_time: u64 = 0;
        let revoke_time = if is_revoked { Some(20) } else { None };
        let decision_time = if is_revoked { 30 } else { 5 };
        let fluent_label = format!("authorized:{name}");
        let (temporal_validity, raw_events) = temporal_label(
            ec,
            engine,
            &fluent_label,
            grant_time,
            revoke_time,
            decision_time,
        )?;

        let request = match e2_prompt_mode {
            E2PromptMode::None => book.render(
                "reactor_manage",
                "add",
                lang_code,
                &[
                    ("name", &name),
                    ("every_seconds", &every_seconds.to_string()),
                ],
            )?,
            E2PromptMode::RawEvents if is_revoked => book.render(
                "reactor_manage",
                "add_raw_events_revoked",
                lang_code,
                &[
                    ("name", &name),
                    ("every_seconds", &every_seconds.to_string()),
                    ("grant_time", &grant_time.to_string()),
                    ("revoke_time", &revoke_time.unwrap().to_string()),
                ],
            )?,
            E2PromptMode::RawEvents => book.render(
                "reactor_manage",
                "add_raw_events_granted",
                lang_code,
                &[
                    ("name", &name),
                    ("every_seconds", &every_seconds.to_string()),
                    ("grant_time", &grant_time.to_string()),
                ],
            )?,
            E2PromptMode::Projection if is_revoked => book.render(
                "reactor_manage",
                "add_projection_revoked",
                lang_code,
                &[
                    ("name", &name),
                    ("every_seconds", &every_seconds.to_string()),
                ],
            )?,
            E2PromptMode::Projection => book.render(
                "reactor_manage",
                "add_projection_granted",
                lang_code,
                &[
                    ("name", &name),
                    ("every_seconds", &every_seconds.to_string()),
                ],
            )?,
        };

        let schedule_value = json!({
            "id": name,
            "name": name,
            "rule": {"kind": "interval", "every_seconds": every_seconds, "anchor": 0},
            "action": {"kind": "llm_turn", "prompt": format!("review {name}")},
        });
        let completed_args = json!({"operation": "add", "schedule": schedule_value});

        // RAW: the envelope failure mode actually observed
        // (`release-v3-failure-analysis.md`) — free text outside the tags.
        // The envelope shape (tool name + argument nesting) offered is
        // identical to every other language row; only `request` differs.
        // Orthogonal to E2: this is always the E1 control regardless of mode.
        let raw_target = malformed_envelope_target("reactor_manage", &completed_args);
        let raw_runtime_valid = KnowledgeAction::parse_runtime(&raw_target).is_ok();

        // In `none` mode the target is always `add`, unchanged from P6.3.
        // In `raw-events`/`projection` mode a revoked authorization means the
        // correct action is a refusal, not `add` — the prompt now actually
        // says so, so the target must agree.
        let gate_on_validity = e2_prompt_mode != E2PromptMode::None;
        let stage;
        let completed_target;
        let schedule_entity =
            engine.resolve_or_create_entity(&format!("reactor-schedule:{name}"))?;

        if gate_on_validity && !temporal_validity {
            stage = "declare_gap";
            completed_target = KnowledgeAction::Gap {
                missing: vec!["reactor-authorization".into()],
                reason: format!(
                    "authorization for schedule {name} was revoked before decision time; not adding it"
                ),
            }
            .render_runtime()?;
            // The add effect must NOT have been asserted — the schedule
            // entity exists (resolved above for causal bookkeeping) but the
            // `reactor:has-state` mirror triple was never written, so
            // verify_prediction below genuinely returns false.
        } else {
            stage = "answer";
            completed_target = tool_call_target(
                &format!("{id}-completed"),
                "reactor_manage",
                completed_args.clone(),
            )?;
            let output = ReactorManageTool.execute(
                engine,
                ToolInput::new()
                    .with_param("operation", "add")
                    .with_value("schedule", schedule_value.clone()),
            )?;
            if !output.success {
                return Err(GeneratorError::Agent(AgentError::ToolExecution {
                    tool_name: "reactor_manage".into(),
                    message: format!("completed add failed for {id}"),
                }));
            }
            // Mirror the reactor's own (non-KG) meta-store mutation into the
            // graph so CausalManager has something to verify against — the
            // reactor's durable store is a separate KV namespace from the KG.
            let mut mirror = vec![Triple::new(
                schedule_entity,
                reactor_added_pred,
                reactor_added,
            )];
            let mut prov = ProvenanceRecord::new(schedule_entity, DerivationKind::Extracted)
                .with_confidence(1.0);
            engine.add_triples_with_provenance(&mut mirror, &mut prov)?;
        }

        let reward_completed = causal_label(
            causal,
            engine,
            &format!("{id}:completed"),
            (schedule_entity, reactor_added_pred, reactor_added),
        )?;
        let phantom = engine.resolve_or_create_entity(&format!("phantom-reactor:{id}"))?;
        let reward_raw = causal_label(
            causal,
            engine,
            &format!("{id}:raw"),
            (phantom, reactor_added_pred, reactor_added),
        )?;

        out.push(EpisodeRecord {
            schema_version: 1,
            id: id.clone(),
            episode_id: id.clone(),
            step: 0,
            stage: stage.into(),
            family: Family::ReactorManage.tag().into(),
            difficulty: 1,
            language: lang_name.into(),
            contract_hash: contract_hash.clone(),
            tools: tools.clone(),
            conversation: vec![json!({"role": "user", "content": request})],
            system_context: None,
            prompt: akh_prompt(&tools, &[json!({"role": "user", "content": request})]),
            target: completed_target,
            labels: Some(EpisodeLabels {
                e1_raw_target: Some(raw_target),
                e1_raw_target_runtime_valid: Some(raw_runtime_valid),
                e1_completed_changed_from_raw: Some(true),
                e2_temporal_validity: Some(temporal_validity),
                e2_raw_events: Some(raw_events),
                e2_prompt_mode: Some(e2_prompt_mode.to_string()),
                e3_causal_reward_completed: Some(reward_completed),
                e3_causal_reward_raw: Some(reward_raw),
                ..Default::default()
            }),
        });
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════
// Oracle family — regenerates the three R0 reference worlds through the
// engine (docs/ai/plans/2026-09-12-akh-R0-episode-contract.md).
// ═══════════════════════════════════════════════════════════════════════

fn generate_oracle_acquire(book: &TemplateBook, seed: Seed, i: usize) -> GenResult<EpisodeRecord> {
    let scratch = ScratchDir::new(&format!("oracle-acquire-{}-{i}", seed.0))?;
    let command = format!("cmd-{}-{i}", seed.0);
    let syntax = format!("syntax-{}-{i} --flag={i}", seed.0);
    let id = format!("p6-3-oracle_acquire-{}-{i:05}", seed.0);

    // Phase 1: acquire and persist, then drop the engine.
    {
        let engine = build_engine(scratch.path())?;
        let preds = AgentPredicates::init(&engine)?;
        let tool = MemoryWriteTool::new(preds);
        let out = tool.execute(
            &engine,
            ToolInput::new()
                .with_value("summary", json!(format!("Learned syntax for {command}")))
                .with_value(
                    "facts",
                    json!([{"subject": command, "predicate": "has-syntax", "object": syntax}]),
                )
                .with_value("source", json!("help-text")),
        )?;
        if !out.success {
            return Err(GeneratorError::Agent(AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: format!("oracle_acquire write failed for {id}"),
            }));
        }
        engine.persist()?;
    }

    // Phase 2: reopen (simulated restart) and recall.
    let engine = build_engine(scratch.path())?;
    let cmd_sym = engine.resolve_symbol(&command)?;
    let has_syntax = engine.resolve_or_create_relation("has-syntax")?;
    let recalled = engine
        .triples_from(cmd_sym)
        .into_iter()
        .find(|t| t.predicate == has_syntax)
        .map(|t| engine.resolve_label(t.object));
    let recalled_syntax = recalled.ok_or_else(|| {
        GeneratorError::Agent(AgentError::ToolExecution {
            tool_name: "memory_recall".into(),
            message: format!("oracle_acquire recall failed for {id} after restart"),
        })
    })?;
    let recalled_ok = recalled_syntax == syntax;

    let mut causal = CausalManager::new(&engine)?;
    let mut ec = EventCalculusEngine::new(&engine)?;
    let value_id = engine.resolve_or_create_entity(&syntax)?;
    let reward_completed = causal_label(
        &mut causal,
        &engine,
        &format!("{id}:completed"),
        (cmd_sym, has_syntax, value_id),
    )?;
    let (temporal_validity, raw_events) = temporal_label(
        &mut ec,
        &engine,
        &format!("learned:{command}"),
        0,
        None,
        100,
    )?;

    let request = book.render(
        "oracle_acquire",
        "acquire",
        "en",
        &[("command", &command), ("syntax", &syntax)],
    )?;
    let raw_target = KnowledgeAction::Answer {
        text: "I don't recall the syntax.".into(),
        citations: vec![],
    }
    .render_runtime()?;
    let completed_target = KnowledgeAction::Answer {
        text: format!("The syntax is: {recalled_syntax}"),
        citations: vec![WorkflowCitation {
            source_id: command.clone(),
            locator: "has-syntax".into(),
            quote: Some(recalled_syntax.clone()),
        }],
    }
    .render_runtime()?;

    Ok(EpisodeRecord {
        schema_version: 1,
        id: id.clone(),
        episode_id: id.clone(),
        step: 0,
        stage: "answer".into(),
        family: Family::OracleAcquire.tag().into(),
        difficulty: 2,
        language: "english".into(),
        contract_hash: "blake3:oracle-acquire".into(),
        tools: vec![],
        conversation: vec![json!({"role": "user", "content": request})],
        system_context: None,
        prompt: akh_prompt(&[], &[json!({"role": "user", "content": request})]),
        target: completed_target,
        labels: Some(EpisodeLabels {
            e1_raw_target: Some(raw_target),
            e1_raw_target_runtime_valid: Some(true),
            e1_completed_changed_from_raw: Some(recalled_ok),
            e2_temporal_validity: Some(temporal_validity),
            e2_raw_events: Some(raw_events),
            e3_causal_reward_completed: Some(reward_completed),
            e3_causal_reward_raw: None,
            ..Default::default()
        }),
    })
}

fn generate_oracle_revise(
    engine: &Engine,
    causal: &mut CausalManager,
    ec: &mut EventCalculusEngine,
    book: &TemplateBook,
    seed: Seed,
    i: usize,
) -> GenResult<EpisodeRecord> {
    let job = format!("job-{}-{i}", seed.0);
    let old_worker = format!("worker-a-{}-{i}", seed.0);
    let new_worker = format!("worker-b-{}-{i}", seed.0);
    let id = format!("p6-3-oracle_revise-{}-{i:05}", seed.0);

    let job_id = engine.resolve_or_create_entity(&job)?;
    let old_worker_id = engine.resolve_or_create_entity(&old_worker)?;
    let new_worker_id = engine.resolve_or_create_entity(&new_worker)?;
    let assigned_to = engine.resolve_or_create_relation("assigned-to")?;
    let current_worker = engine.resolve_or_create_relation("current-worker")?;

    let mut initial = vec![
        Triple::new(job_id, assigned_to, old_worker_id),
        Triple::new(job_id, current_worker, old_worker_id),
    ];
    let mut prov = ProvenanceRecord::new(job_id, DerivationKind::Extracted).with_confidence(1.0);
    engine.add_triples_with_provenance(&mut initial, &mut prov)?;

    let request = book.render(
        "oracle_revise",
        "revise",
        "en",
        &[
            ("job", &job),
            ("old_worker", &old_worker),
            ("new_worker", &new_worker),
        ],
    )?;

    // RAW (E1 control): stale — proposes recording the *old* worker as
    // current, i.e. failing to apply the correction.
    let raw_args = json!({
        "summary": format!("Route for {job}"),
        "facts": [{"subject": job, "predicate": "current-worker", "object": old_worker}],
    });
    let raw_target = tool_call_target(&format!("{id}-raw"), "memory_write", raw_args)?;

    // COMPLETED: retract the stale current-worker edge, assert the new one,
    // and keep both assigned-to facts as history.
    engine.remove_triple(job_id, current_worker, old_worker_id)?;
    let mut correction = vec![
        Triple::new(job_id, current_worker, new_worker_id),
        Triple::new(job_id, assigned_to, new_worker_id),
    ];
    let mut prov2 = ProvenanceRecord::new(job_id, DerivationKind::Extracted).with_confidence(1.0);
    engine.add_triples_with_provenance(&mut correction, &mut prov2)?;
    let completed_args = json!({
        "summary": format!("Route for {job}"),
        "facts": [{"subject": job, "predicate": "current-worker", "object": new_worker}],
        "tags": ["route-correction"],
    });
    let completed_target =
        tool_call_target(&format!("{id}-completed"), "memory_write", completed_args)?;

    let history_present = engine
        .triples_from(job_id)
        .iter()
        .any(|t| t.predicate == assigned_to && t.object == old_worker_id);
    let current_is_new = engine
        .triples_from(job_id)
        .iter()
        .any(|t| t.predicate == current_worker && t.object == new_worker_id);

    let (temporal_validity, raw_events) =
        temporal_label(ec, engine, &format!("route-current:{job}"), 0, None, 5)?;
    let reward_completed = causal_label(
        causal,
        engine,
        &format!("{id}:completed"),
        (job_id, current_worker, new_worker_id),
    )?;
    let reward_raw = causal_label(
        causal,
        engine,
        &format!("{id}:raw"),
        (job_id, current_worker, old_worker_id),
    )?;

    Ok(EpisodeRecord {
        schema_version: 1,
        id: id.clone(),
        episode_id: id.clone(),
        step: 0,
        stage: "answer".into(),
        family: Family::OracleRevise.tag().into(),
        difficulty: 3,
        language: "english".into(),
        contract_hash: "blake3:oracle-revise".into(),
        tools: vec![],
        conversation: vec![json!({"role": "user", "content": request})],
        system_context: None,
        prompt: akh_prompt(&[], &[json!({"role": "user", "content": request})]),
        target: completed_target,
        labels: Some(EpisodeLabels {
            e1_raw_target: Some(raw_target),
            e1_raw_target_runtime_valid: Some(true),
            e1_completed_changed_from_raw: Some(history_present && current_is_new),
            e2_temporal_validity: Some(temporal_validity),
            e2_raw_events: Some(raw_events),
            e3_causal_reward_completed: Some(reward_completed),
            e3_causal_reward_raw: Some(reward_raw),
            ..Default::default()
        }),
    })
}

fn generate_oracle_repair(
    engine: &Engine,
    causal: &mut CausalManager,
    ec: &mut EventCalculusEngine,
    book: &TemplateBook,
    seed: Seed,
    i: usize,
) -> GenResult<EpisodeRecord> {
    let target_system = format!("system-{}-{i}", seed.0);
    let id = format!("p6-3-oracle_repair-{}-{i:05}", seed.0);
    let request = book.render(
        "oracle_repair",
        "repair",
        "en",
        &[("target", &target_system)],
    )?;

    // Grant then revoke, decision at t=20 (post-revoke): the correct
    // behavior is refusal.
    let fluent_label = format!("repair-permission:{}:{i}", seed.0);
    let (temporal_validity, raw_events) =
        temporal_label(ec, engine, &fluent_label, 0, Some(15), 20)?;

    // RAW (E1 control): proceeds with the repair despite revoked authority
    // — the exact interaction-repair failure mode this world tests for.
    let repair_target_sym = engine.resolve_or_create_entity(&target_system)?;
    let repaired_pred = engine.resolve_or_create_relation("repaired-by")?;
    let operator = engine.resolve_or_create_entity("operator")?;
    let raw_args = json!({
        "summary": format!("Repaired {target_system}"),
        "facts": [{"subject": target_system, "predicate": "repaired-by", "object": "operator"}],
    });
    let raw_target = tool_call_target(&format!("{id}-raw"), "memory_write", raw_args)?;

    // COMPLETED (E1 treatment): a Gap declination — the model must not act
    // while `holds_at` is false.
    let completed_target = KnowledgeAction::Gap {
        missing: vec!["repair-authorization".into()],
        reason: format!("authorization for {target_system} was revoked; awaiting re-authorization"),
    }
    .render_runtime()?;

    // E3: the would-be repair effect was never applied — verify_prediction
    // must come back false, a genuine negative causal signal.
    let reward_completed = causal_label(
        causal,
        engine,
        &format!("{id}:completed"),
        (repair_target_sym, repaired_pred, operator),
    )?;

    Ok(EpisodeRecord {
        schema_version: 1,
        id: id.clone(),
        episode_id: id.clone(),
        step: 0,
        stage: "declare_gap".into(),
        family: Family::OracleRepair.tag().into(),
        difficulty: 3,
        language: "english".into(),
        contract_hash: "blake3:oracle-repair".into(),
        tools: vec![],
        conversation: vec![json!({"role": "user", "content": request})],
        system_context: None,
        prompt: akh_prompt(&[], &[json!({"role": "user", "content": request})]),
        target: completed_target,
        labels: Some(EpisodeLabels {
            e1_raw_target: Some(raw_target),
            e1_raw_target_runtime_valid: Some(true),
            e1_completed_changed_from_raw: Some(true),
            e2_temporal_validity: Some(temporal_validity),
            e2_raw_events: Some(raw_events),
            e3_causal_reward_completed: Some(reward_completed),
            e3_causal_reward_raw: None,
            ..Default::default()
        }),
    })
}

fn generate_oracle(book: &TemplateBook, seed: Seed, seeds: usize) -> GenResult<Vec<EpisodeRecord>> {
    let mut out = Vec::with_capacity(seeds * 3);
    for i in 0..seeds {
        out.push(generate_oracle_acquire(book, seed, i)?);
    }
    {
        let scratch = ScratchDir::new(&format!("oracle-revise-{}", seed.0))?;
        let engine = build_engine(scratch.path())?;
        let mut causal = CausalManager::new(&engine)?;
        let mut ec = EventCalculusEngine::new(&engine)?;
        for i in 0..seeds {
            out.push(generate_oracle_revise(
                &engine,
                &mut causal,
                &mut ec,
                book,
                seed,
                i,
            )?);
        }
    }
    {
        let scratch = ScratchDir::new(&format!("oracle-repair-{}", seed.0))?;
        let engine = build_engine(scratch.path())?;
        let mut causal = CausalManager::new(&engine)?;
        let mut ec = EventCalculusEngine::new(&engine)?;
        for i in 0..seeds {
            out.push(generate_oracle_repair(
                &engine,
                &mut causal,
                &mut ec,
                book,
                seed,
                i,
            )?);
        }
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════
// Family: knowledge_revise (P6.5/E4 revision curriculum, stretch)
//
// Treatment: a sourced correction retracts the stale owned edge through the
// engine's real revision path (`Engine::retract_owned_rule_snapshot`,
// recovered P6.0, exercised by `tests/exact_owned_revision.rs`), then the
// corrected fact is asserted and `memory_write` actually runs against the
// revised graph. Control: the append-only sibling never retracts — the new
// fact is asserted alongside the stale one, leaving both present (the
// unresolved contradiction that is the point of the control). Distinct from
// `oracle_revise` (P6.3), which uses a plain `remove_triple`, not the
// exact-owned-edge path this experiment measures.
// ═══════════════════════════════════════════════════════════════════════

/// Shared setup for one revision-pair index: a fresh isolated engine with
/// the stale `(job, current-worker, old_worker)` fact asserted and its exact
/// live `Triple` (subject/predicate/object *and* metadata) resolved, which
/// is what `retract_owned_rule_snapshot` requires to match.
fn revision_pair_setup(
    label: &str,
    seed: Seed,
    i: usize,
) -> GenResult<(ScratchDir, Engine, String, String, String, Triple)> {
    let scratch = ScratchDir::new(&format!("{label}-{}-{i}", seed.0))?;
    let engine = build_engine(scratch.path())?;
    let job = format!("p6-5-revise-job-{}-{i}", seed.0);
    let old_worker = format!("p6-5-revise-worker-a-{}-{i}", seed.0);
    let new_worker = format!("p6-5-revise-worker-b-{}-{i}", seed.0);

    let job_id = engine.resolve_or_create_entity(&job)?;
    let old_worker_id = engine.resolve_or_create_entity(&old_worker)?;
    let current_worker = engine.resolve_or_create_relation("current-worker")?;
    engine.add_triple(&Triple::new(job_id, current_worker, old_worker_id))?;
    let stale = engine
        .all_triples()
        .into_iter()
        .find(|t| t.subject == job_id && t.predicate == current_worker && t.object == old_worker_id)
        .ok_or_else(|| {
            GeneratorError::Agent(AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: format!("stale fact not found before revision for {label}-{i}"),
            })
        })?;
    Ok((scratch, engine, job, old_worker, new_worker, stale))
}

/// E4 treatment: sourced correction through `retract_owned_rule_snapshot`.
fn generate_revision_treatment(
    book: &TemplateBook,
    seed: Seed,
    i: usize,
) -> GenResult<EpisodeRecord> {
    let (_scratch, engine, job, old_worker, new_worker, stale) =
        revision_pair_setup("revise-treatment", seed, i)?;
    let id = format!("p6-5-knowledge_revise-treatment-{}-{i:05}", seed.0);

    // The sourced correction: retract the exact owned edge through the
    // engine's real revision path, not a bare `remove_triple`.
    engine.retract_owned_rule_snapshot(&[stale], &[])?;

    let job_id = engine.resolve_symbol(&job)?;
    let old_worker_id = engine.resolve_symbol(&old_worker)?;
    let current_worker = engine.resolve_or_create_relation("current-worker")?;

    let agent_preds = AgentPredicates::init(&engine)?;
    let mut causal = CausalManager::new(&engine)?;
    let tool = MemoryWriteTool::new(agent_preds.clone());
    let manifest = tool.manifest();
    let tools = vec![openai_tool_def(&manifest)];
    let contract_hash = format!(
        "blake3:{}",
        blake3::hash(manifest.description.as_bytes()).to_hex()
    );

    let request = book.render(
        "knowledge_revise",
        "revise",
        "en",
        &[
            ("job", &job),
            ("old_worker", &old_worker),
            ("new_worker", &new_worker),
        ],
    )?;

    let args = json!({
        "summary": format!("Route correction for {job}"),
        "facts": [{"subject": job, "predicate": "current-worker", "object": new_worker}],
        "tags": ["route-correction", "revision"],
        "source": job,
    });
    let target = tool_call_target(&format!("{id}-completed"), "memory_write", args.clone())?;
    let output = tool.execute(
        &engine,
        ToolInput::new()
            .with_value("summary", args["summary"].clone())
            .with_value("facts", args["facts"].clone())
            .with_value("tags", args["tags"].clone())
            .with_value("source", args["source"].clone()),
    )?;
    if !output.success {
        return Err(GeneratorError::Agent(AgentError::ToolExecution {
            tool_name: "memory_write".into(),
            message: format!("revision-treatment write failed for {id}"),
        }));
    }

    let retraction_applied = !engine.has_triple(job_id, current_worker, old_worker_id);
    let reward = causal_label(
        &mut causal,
        &engine,
        &format!("{id}:completed"),
        (job_id, current_worker, engine.resolve_symbol(&new_worker)?),
    )?;

    Ok(EpisodeRecord {
        schema_version: 1,
        id: id.clone(),
        episode_id: id.clone(),
        step: 0,
        stage: "answer".into(),
        family: Family::KnowledgeRevise.tag().into(),
        difficulty: 3,
        language: "english".into(),
        contract_hash,
        tools: tools.clone(),
        conversation: vec![json!({"role": "user", "content": request})],
        system_context: None,
        prompt: akh_prompt(&tools, &[json!({"role": "user", "content": request})]),
        target,
        labels: Some(EpisodeLabels {
            e4_role: Some("treatment".into()),
            e4_retraction_applied: Some(retraction_applied),
            e4_stale_edge_still_present: Some(!retraction_applied),
            e3_causal_reward_completed: Some(reward),
            ..Default::default()
        }),
    })
}

/// E4 control: append-only sibling, the stale edge is never retracted.
fn generate_revision_control(
    book: &TemplateBook,
    seed: Seed,
    i: usize,
) -> GenResult<EpisodeRecord> {
    let (_scratch, engine, job, old_worker, new_worker, _stale) =
        revision_pair_setup("revise-control", seed, i)?;
    let id = format!("p6-5-knowledge_revise-control-{}-{i:05}", seed.0);

    let job_id = engine.resolve_symbol(&job)?;
    let old_worker_id = engine.resolve_symbol(&old_worker)?;
    let current_worker = engine.resolve_or_create_relation("current-worker")?;

    let agent_preds = AgentPredicates::init(&engine)?;
    let mut causal = CausalManager::new(&engine)?;
    let tool = MemoryWriteTool::new(agent_preds.clone());
    let manifest = tool.manifest();
    let tools = vec![openai_tool_def(&manifest)];
    let contract_hash = format!(
        "blake3:{}",
        blake3::hash(manifest.description.as_bytes()).to_hex()
    );

    let request = book.render(
        "knowledge_revise",
        "revise",
        "en",
        &[
            ("job", &job),
            ("old_worker", &old_worker),
            ("new_worker", &new_worker),
        ],
    )?;

    // No `tags`/`source` and no prior retraction — the naive append-only
    // proposal a model might emit without engaging the revision path.
    let args = json!({
        "summary": format!("Route correction for {job}"),
        "facts": [{"subject": job, "predicate": "current-worker", "object": new_worker}],
    });
    let target = tool_call_target(&format!("{id}-completed"), "memory_write", args.clone())?;
    let output = tool.execute(
        &engine,
        ToolInput::new()
            .with_value("summary", args["summary"].clone())
            .with_value("facts", args["facts"].clone()),
    )?;
    if !output.success {
        return Err(GeneratorError::Agent(AgentError::ToolExecution {
            tool_name: "memory_write".into(),
            message: format!("revision-control write failed for {id}"),
        }));
    }

    let stale_still_present = engine.has_triple(job_id, current_worker, old_worker_id);
    let reward = causal_label(
        &mut causal,
        &engine,
        &format!("{id}:completed"),
        (job_id, current_worker, engine.resolve_symbol(&new_worker)?),
    )?;

    Ok(EpisodeRecord {
        schema_version: 1,
        id: id.clone(),
        episode_id: id.clone(),
        step: 0,
        stage: "answer".into(),
        family: Family::KnowledgeRevise.tag().into(),
        difficulty: 3,
        language: "english".into(),
        contract_hash,
        tools: tools.clone(),
        conversation: vec![json!({"role": "user", "content": request})],
        system_context: None,
        prompt: akh_prompt(&tools, &[json!({"role": "user", "content": request})]),
        target,
        labels: Some(EpisodeLabels {
            e4_role: Some("control".into()),
            e4_retraction_applied: Some(false),
            e4_stale_edge_still_present: Some(stale_still_present),
            e3_causal_reward_completed: Some(reward),
            ..Default::default()
        }),
    })
}

fn generate_revision_pairs(
    book: &TemplateBook,
    seed: Seed,
    count: usize,
) -> GenResult<Vec<EpisodeRecord>> {
    let mut out = Vec::with_capacity(count * 2);
    for i in 0..count {
        out.push(generate_revision_treatment(book, seed, i)?);
        out.push(generate_revision_control(book, seed, i)?);
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════
// Prompt rendering (local, mirrors grug_data::render_knowledge_prompt's
// OpenAI-tool-call/<tool_call> convention without depending on its exact
// system-prompt text, which is that crate's data to own)
// ═══════════════════════════════════════════════════════════════════════

fn akh_prompt(tools: &[Value], conversation: &[Value]) -> String {
    let tools_json = serde_json::to_string(tools).unwrap_or_else(|_| "[]".into());
    let mut prompt = format!(
        "<|im_start|>system\nTools available:\n{tools_json}\n\nCall a tool with <tool_call>{{\"name\":...,\"arguments\":{{...}}}}</tool_call>, or answer directly.<|im_end|>\n"
    );
    for message in conversation {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ═══════════════════════════════════════════════════════════════════════
// Manifest
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: u32,
    generator: &'static str,
    generator_git_hash: String,
    seed: u64,
    template_file_hashes: BTreeMap<String, String>,
    per_family_counts: BTreeMap<String, u64>,
    decontamination: DecontaminationSummary,
    /// P6.5/E1: `--e1-targets` mode this run used (`completed`/`raw-valid`).
    e1_targets_mode: String,
    /// P6.5/E1: total records dropped from the *sidecar*
    /// (`<family>/train.sft.jsonl`) across all four tool-contract families
    /// because their raw target was not runtime-valid (`raw-valid` mode
    /// only; always 0 in `completed` mode). The primary `<family>.jsonl`
    /// files are never affected by this — only the steering sidecar is.
    e1_raw_invalid_dropped_from_sidecar: u64,
    /// P6.5/E2: `--e2-prompt` mode this run used (`none`/`raw-events`/
    /// `projection`).
    e2_prompt_mode: String,
    /// P6.5/E4: whether the revision-curriculum family was generated this
    /// run (`--e4-revision`).
    e4_revision_enabled: bool,
    /// P6.3b: `--contracts` path, if supplied. `None` means no steering
    /// sidecar was written this run (the primary `<family>.jsonl` files are
    /// unaffected either way).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_contracts_path: Option<String>,
    /// P6.3b: the frozen snapshot's own `manifest_hash` — the value every
    /// tool-contract-family primary record's `contract_hash` and every
    /// sidecar `KnowledgeSftRecord`'s `contract_hash` now carries, in place
    /// of this generator's ad hoc four-tool `contract_hash` (below).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_contract_hash: Option<String>,
    /// This generator's own four-tool `ToolContractSnapshot::from_serializable`
    /// hash (documented at its call site) — always computed, kept for
    /// debugging even when `frozen_contract_hash` supersedes it on every
    /// written record.
    ad_hoc_contract_hash: String,
    /// P6.3b: total records dropped from the sidecar across all four
    /// tool-contract families because their target failed
    /// `ToolContractSnapshot::validate_call` against the frozen contract (or
    /// named a tool absent from it). Always 0 when `frozen_contracts_path`
    /// is `None`.
    sidecar_schema_invalid_dropped: u64,
    /// P6.3b: per-family record count actually written to
    /// `<family>/train.sft.jsonl`. Empty when `frozen_contracts_path` is
    /// `None` (no sidecar was written this run).
    sidecar_per_family_counts: BTreeMap<String, u64>,
    /// P6.5/E4: the `knowledge_revise_treatment/` sidecar's write stats
    /// (records with `labels.e4_retraction_applied == true`). `None` unless
    /// both `--e4-revision` and `--contracts` were given this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e4_sidecar_treatment: Option<E4SidecarStats>,
    /// P6.5/E4: the `knowledge_revise_control/` sidecar's write stats (the
    /// append-only, never-retracted siblings). `None` unless both
    /// `--e4-revision` and `--contracts` were given this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    e4_sidecar_control: Option<E4SidecarStats>,
}

/// P6.5/E4: one `knowledge_revise` arm's sidecar write stats, recorded in
/// the top-level manifest so a downstream consumer doesn't have to open
/// `knowledge_revise_treatment/manifest.json` / `knowledge_revise_control/
/// manifest.json` separately to learn whether the sidecar was written, how
/// many records it kept, and how many were dropped as schema-invalid
/// against the frozen `--contracts` snapshot.
#[derive(Debug, Serialize)]
struct E4SidecarStats {
    /// Directory under `--out-dir` holding this arm's `train.sft.jsonl`,
    /// `manifest.json`, and `separation.json`.
    dir: String,
    /// Records written to `<dir>/train.sft.jsonl`.
    kept: u64,
    /// Records dropped because their target failed
    /// `ToolContractSnapshot::validate_call` against the frozen contract.
    /// Always 0 in practice for E4 (both arms always target `memory_write`
    /// with arguments the frozen contract already accepts), kept for parity
    /// with the four tool-contract families' sidecar accounting.
    schema_invalid_dropped: u64,
}

impl E4SidecarStats {
    fn new(dir: &str, stats: &SidecarWriteStats) -> Self {
        Self {
            dir: dir.to_string(),
            kept: stats.kept as u64,
            schema_invalid_dropped: stats.schema_invalid_dropped,
        }
    }
}

fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

// ═══════════════════════════════════════════════════════════════════════
// Orchestration
// ═══════════════════════════════════════════════════════════════════════

fn write_family_jsonl(
    out_dir: &Path,
    stem: &str,
    records: Vec<EpisodeRecord>,
    sentinels: &SentinelIndex,
    decon: &mut DecontaminationSummary,
) -> GenResult<(u64, Vec<EpisodeRecord>)> {
    let path = out_dir.join(format!("{stem}.jsonl"));
    let append = path.exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|source| GeneratorError::Io {
            path: path.clone(),
            source,
        })?;
    let mut writer = BufWriter::new(file);
    let mut kept = 0u64;
    let mut survivors = Vec::with_capacity(records.len());
    for record in records {
        decon.records_checked += 1;
        if let Some(reason) = sentinels.record_collision(&record) {
            if reason == "exact" {
                decon.records_dropped_exact += 1;
            } else {
                decon.records_dropped_ngram += 1;
            }
            continue;
        }
        let line = serde_json::to_string(&record)?;
        writeln!(writer, "{line}").map_err(|source| GeneratorError::Io {
            path: path.clone(),
            source,
        })?;
        kept += 1;
        survivors.push(record);
    }
    writer.flush().map_err(|source| GeneratorError::Io {
        path: path.clone(),
        source,
    })?;
    let _ = append;
    Ok((kept, survivors))
}

// ═══════════════════════════════════════════════════════════════════════
// train_knowledge steering-sibling sidecar
//
// Additional per-family artifacts shaped exactly like the pod's adaptive
// loop `training-sibling/{manifest.json,separation.json,train.sft.jsonl}`
// (see e.g. `runs/checkpoints/.../windows/window-0002-attempt-0001/
// training-sibling/` on the training Mac), so a plain
// `train_knowledge --train-jsonl <family>/train.sft.jsonl` (or
// `--eval-jsonl`) load works with zero conversion.
//
// Boundary, stated once here rather than at every call site: the *full*
// `--steering-sibling-manifest` ingestion mode
// (`crates/grug-train/examples/train_knowledge.rs::verify_steering_inputs`)
// additionally requires a live `--steering-decision`
// (`AdaptiveExamController` state) and `--steering-frontier-exam`
// (`ToolContractExamSet`/`KnowledgeExamSet`) and *regenerates* the sibling
// records from `ToolCurriculumGenerator`/`KnowledgeWorkflowGenerator` to
// cross-check them. Those two artifacts only exist mid-adaptive-loop on a
// training pod; this generator is a genuinely different, engine-grounded
// generation path (ADR 106 decision 3) and does not fabricate them —
// doing so would either be inert (fields that only pass a shallow
// `starts_with("blake3:")` check) or actively wrong (claiming a frontier
// exam that was never run). What *is* real and immediately usable without
// any of those four steering flags: `train.sft.jsonl` is exactly
// `Vec<KnowledgeSftRecord>`, and `contract_hash`/`sft_hash`/
// `separation_hash` are computed the same way (`ToolContractSnapshot`,
// BLAKE3-of-file-bytes) grug-data's own tooling computes them.
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize)]
struct SteeringManifest {
    schema_version: u32,
    complete: bool,
    contract_hash: String,
    difficulty: u8,
    family: String,
    generator_version: String,
    kind: String,
    plan_hash: String,
    records: u64,
    separation_hash: String,
    sft_hash: String,
    source_exam_file_hash: String,
    source_exam_fingerprint: String,
    tool: String,
    /// P6.5/E1: which `--e1-targets` mode produced `train.sft.jsonl`'s
    /// `target` field for this family (`completed`/`raw-valid`). Additive —
    /// `train_knowledge.rs`'s `TrainingSiblingManifest` reader has no
    /// `deny_unknown_fields`, so this is silently ignored by that consumer.
    #[serde(default)]
    e1_targets_mode: String,
    /// P6.5/E1: records dropped from this family's sidecar because
    /// `labels.e1_raw_target_runtime_valid` was `false` (`raw-valid` mode
    /// only; always 0 in `completed` mode).
    #[serde(default)]
    e1_raw_invalid_dropped: u64,
    /// P6.3b: records dropped from this family's sidecar because their
    /// target's tool/arguments did not pass
    /// `ToolContractSnapshot::validate_call` against the frozen `--contracts`
    /// snapshot (or the tool was altogether absent from it). Always 0 when
    /// the frozen contract genuinely matches the runtime the primary record
    /// was generated against.
    #[serde(default)]
    schema_invalid_dropped: u64,
}

/// Outcome of writing one family's `<family>/{train.sft.jsonl,manifest.json,
/// separation.json}` steering sidecar.
struct SidecarWriteStats {
    kept: usize,
    /// See `SteeringManifest::e1_raw_invalid_dropped`.
    e1_raw_invalid_dropped: u64,
    /// See `SteeringManifest::schema_invalid_dropped`.
    schema_invalid_dropped: u64,
}

#[derive(Debug, Serialize)]
struct SeparationJson {
    held_out_axes: Vec<String>,
    max_transcript_ngram_jaccard: f64,
    min_image_hash_hamming_distance: Option<u32>,
    exact_user_request_overlap: u64,
    exact_prompt_target_overlap: u64,
    exam_family_cases: u64,
    sibling_records: u64,
}

fn workflow_stage(stage: &str) -> Option<WorkflowStage> {
    match stage {
        "answer" => Some(WorkflowStage::Answer),
        "declare_gap" => Some(WorkflowStage::DeclareGap),
        _ => None,
    }
}

fn workflow_language(language: &str) -> Option<WorkflowLanguage> {
    match language {
        "english" => Some(WorkflowLanguage::English),
        "german" => Some(WorkflowLanguage::German),
        "french" => Some(WorkflowLanguage::French),
        // Italian/Spanish are not representable by grug-data's current
        // 3-variant WorkflowLanguage enum. Extending it is out of scope
        // here (wide blast radius across grug-data's existing generators
        // and tests, which assume `WorkflowLanguage::all()` has exactly 3
        // members) — tracked as a known gap in the P6.3 RESULT doc. These
        // records still ship in the primary `<family>.jsonl` (this
        // generator's own schema); they are simply excluded from the
        // `KnowledgeSftRecord`-typed steering sidecar.
        _ => None,
    }
}

/// Deterministic per-record seed for the sidecar's own offered-tool
/// sampling and phrasing-variant selection (`sample_offered_tools`,
/// `render_request`). Derived from the record's stable `id` (itself a pure
/// function of `--seed`/index), never from a shared/mutable RNG, so
/// re-running the generator with the same `--seed` reproduces byte-identical
/// sidecar files — the same determinism guarantee P6.3's primary files
/// already have (see the RESULT doc's "regenerating ... gave byte-identical
/// jsonl").
fn sidecar_seed(id: &str) -> u64 {
    let hash = blake3::hash(id.as_bytes());
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().expect("8 bytes"))
}

/// Outcome of rendering one primary record's steering-sidecar counterpart.
/// A strongly-typed alternative to a bare `Option`/error so callers can
/// report *why* a record didn't make the sidecar, not just that it didn't.
enum SidecarBuildOutcome {
    Built(Box<KnowledgeSftRecord>),
    /// No `WorkflowLanguage` counterpart for this record's language
    /// (Italian/Spanish — see `workflow_language`).
    UnrepresentableLanguage,
    /// The record's target does not round-trip to a schema-valid call
    /// against the frozen contract (`ToolContractSnapshot::validate_call`
    /// rejected it, or the tool is altogether absent from the frozen
    /// snapshot).
    SchemaInvalid,
}

/// Render one primary `EpisodeRecord` into the base tool-contract corpus's
/// own distribution: prompt via `render_knowledge_prompt_with_system`, four
/// offered tools (`target_tool` + seeded distractors) via
/// `grug_data::sample_offered_tools` against the *frozen* snapshot, and a
/// user turn built from grug-data's own tool-curriculum phrase table
/// (`grug_data::render_request`) over the exact arguments recovered from
/// `target_text` — never this generator's own ad hoc prompt/tool list. See
/// the module doc on `write_steering_sidecar` for the full distribution-
/// mismatch rationale (P6.3b).
///
/// `target_tool` is the family's canonical tool name (needed for the
/// `declare_gap` branch, where the target itself carries no tool name to
/// recover). `target_text` is the record's completed target or, in
/// `--e1-targets raw-valid` mode, `labels.e1_raw_target` — the caller
/// decides which.
fn build_sidecar_record(
    frozen: &ToolContractSnapshot,
    record: &EpisodeRecord,
    target_tool: &str,
    target_text: &str,
) -> GenResult<SidecarBuildOutcome> {
    let Some(language) = workflow_language(&record.language) else {
        return Ok(SidecarBuildOutcome::UnrepresentableLanguage);
    };
    let Some(stage) = workflow_stage(&record.stage) else {
        return Ok(SidecarBuildOutcome::SchemaInvalid);
    };
    let seed = sidecar_seed(&record.id);
    // The request the model sees is ALWAYS rendered from the engine-completed
    // episode (`record.target`): that is the situation's ground truth. The
    // emitted target may differ from it only under `--e1-targets raw-valid`,
    // where the control arm keeps the identical prompt and swaps in the
    // schema-valid-but-not-value-exact raw proposal — otherwise the two E1
    // arms would train on different episodes, not different targets.
    let completed = KnowledgeAction::parse_runtime(&record.target)?;
    let action = KnowledgeAction::parse_runtime(target_text)?;
    if let KnowledgeAction::ToolCall { tool, arguments } = &action
        && !frozen.validate_call(tool, arguments).valid
    {
        return Ok(SidecarBuildOutcome::SchemaInvalid);
    }
    let (tools, conversation) = match &completed {
        KnowledgeAction::ToolCall { tool, arguments } => {
            let Some(contract) = frozen.tool(tool) else {
                return Ok(SidecarBuildOutcome::SchemaInvalid);
            };
            if !frozen.validate_call(tool, arguments).valid {
                return Ok(SidecarBuildOutcome::SchemaInvalid);
            }
            let request = render_request(language, &contract.description, arguments, seed as usize);
            let tools = sample_offered_tools(frozen, tool, 4, seed)?;
            (tools, vec![json!({"role": "user", "content": request})])
        }
        // A refusal has no arguments to render as a curriculum request;
        // `render_runtime`'s single Gap format (already used to build
        // `target_text`) is already the base corpus's only Gap rendering —
        // reused, not reimplemented. The offered-tool set still centers the
        // tool that was declined, matching what the model was actually
        // asked to (not) call.
        KnowledgeAction::Gap { .. } => {
            if frozen.tool(target_tool).is_none() {
                return Ok(SidecarBuildOutcome::SchemaInvalid);
            }
            let tools = sample_offered_tools(frozen, target_tool, 4, seed)?;
            (tools, record.conversation.clone())
        }
        KnowledgeAction::Answer { .. } => return Ok(SidecarBuildOutcome::SchemaInvalid),
    };
    let prompt = grug_data::render_knowledge_prompt_with_system(&tools, &conversation, None);
    Ok(SidecarBuildOutcome::Built(Box::new(KnowledgeSftRecord {
        schema_version: record.schema_version,
        id: record.id.clone(),
        episode_id: record.episode_id.clone(),
        step: record.step,
        stage,
        family: record.family.clone(),
        difficulty: record.difficulty,
        language,
        contract_hash: frozen.manifest_hash.clone(),
        tools,
        conversation,
        system_context: None,
        prompt,
        target: target_text.to_string(),
        weight: None,
    })))
}

fn file_blake3(path: &Path) -> GenResult<String> {
    let bytes = fs::read(path).map_err(|source| GeneratorError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(format!("blake3:{}", blake3::hash(&bytes).to_hex()))
}

/// Write `<out_dir>/<dir_name>/{train.sft.jsonl,manifest.json,separation.json}`
/// for one tool-contract family from its already-decontaminated records, in
/// the base tool-contract corpus's own distribution: every record is
/// rebuilt by `build_sidecar_record` against `frozen` (the `--contracts`
/// snapshot) rather than carried over from the primary `EpisodeRecord`'s
/// engine-only prompt/tools — see that function's doc comment for why.
///
/// `dir_name` and `tool` are separate parameters (rather than one, as
/// before P6.5/E4) because the `knowledge_revise` treatment/control
/// sidecars live in their own directories (`knowledge_revise_treatment/`,
/// `knowledge_revise_control/`) while still building against the real
/// `memory_write` contract (`tool`, passed to `build_sidecar_record` as
/// `target_tool` and recorded in the written `SteeringManifest.tool`). The
/// four tool-contract families pass the same string for both, unchanged.
fn write_steering_sidecar(
    out_dir: &Path,
    family_tag: &str,
    dir_name: &str,
    tool: &str,
    frozen: &ToolContractSnapshot,
    records: &[EpisodeRecord],
    e1_targets_mode: E1TargetsMode,
) -> GenResult<SidecarWriteStats> {
    let dir = out_dir.join(dir_name);
    fs::create_dir_all(&dir).map_err(|source| GeneratorError::Io {
        path: dir.clone(),
        source,
    })?;

    let mut e1_raw_invalid_dropped = 0u64;
    let mut schema_invalid_dropped = 0u64;
    let mut sft: Vec<KnowledgeSftRecord> = Vec::with_capacity(records.len());
    for record in records {
        let target_text = match e1_targets_mode {
            E1TargetsMode::Completed => record.target.clone(),
            E1TargetsMode::RawValid => {
                let Some(labels) = record.labels.as_ref() else {
                    continue;
                };
                let Some(raw_target) = labels.e1_raw_target.as_ref() else {
                    continue;
                };
                if labels.e1_raw_target_runtime_valid != Some(true) {
                    // Schema-invalid raw target: not a fair E1 control (the
                    // point is schema-valid-but-value-wrong vs.
                    // schema-valid-and-value-exact), so this record is
                    // dropped from the sidecar rather than swapped in.
                    e1_raw_invalid_dropped += 1;
                    continue;
                }
                raw_target.clone()
            }
        };
        match build_sidecar_record(frozen, record, tool, &target_text)? {
            SidecarBuildOutcome::Built(sft_record) => sft.push(*sft_record),
            SidecarBuildOutcome::UnrepresentableLanguage => {}
            SidecarBuildOutcome::SchemaInvalid => schema_invalid_dropped += 1,
        }
    }
    let sft_path = dir.join("train.sft.jsonl");
    {
        let file = File::create(&sft_path).map_err(|source| GeneratorError::Io {
            path: sft_path.clone(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        for record in &sft {
            let line = serde_json::to_string(record)?;
            writeln!(writer, "{line}").map_err(|source| GeneratorError::Io {
                path: sft_path.clone(),
                source,
            })?;
        }
        writer.flush().map_err(|source| GeneratorError::Io {
            path: sft_path.clone(),
            source,
        })?;
    }
    let sft_hash = file_blake3(&sft_path)?;

    let mut seen_prompts = HashSet::new();
    let mut seen_pairs = HashSet::new();
    let mut exact_user_request_overlap = 0u64;
    let mut exact_prompt_target_overlap = 0u64;
    for record in &sft {
        if !seen_prompts.insert(record.prompt.clone()) {
            exact_user_request_overlap += 1;
        }
        if !seen_pairs.insert((record.prompt.clone(), record.target.clone())) {
            exact_prompt_target_overlap += 1;
        }
    }
    let separation = SeparationJson {
        held_out_axes: vec!["engine_state_index".into(), "seed_namespace".into()],
        // No frontier exam exists in this generation path to compute a
        // real n-gram Jaccard distance against (see the boundary note
        // above); 0.0 here means "not computed against an exam", not "zero
        // measured overlap" — `exact_*_overlap` below are the real,
        // measured numbers.
        max_transcript_ngram_jaccard: 0.0,
        min_image_hash_hamming_distance: None,
        exact_user_request_overlap,
        exact_prompt_target_overlap,
        exam_family_cases: 0,
        sibling_records: sft.len() as u64,
    };
    let separation_path = dir.join("separation.json");
    fs::write(&separation_path, serde_json::to_string_pretty(&separation)?).map_err(|source| {
        GeneratorError::Io {
            path: separation_path.clone(),
            source,
        }
    })?;
    let separation_hash = file_blake3(&separation_path)?;

    let plan_hash = format!(
        "blake3:{}",
        blake3::hash(
            format!("akh-episode-generator-engine-v1:{family_tag}:{}", sft.len()).as_bytes()
        )
        .to_hex()
    );
    let manifest = SteeringManifest {
        schema_version: 1,
        complete: true,
        contract_hash: frozen.manifest_hash.clone(),
        difficulty: 1,
        family: family_tag.to_string(),
        generator_version: "akh-episode-generator-engine-v1".into(),
        kind: "engine_in_the_loop_training_sibling".into(),
        plan_hash,
        records: sft.len() as u64,
        separation_hash,
        sft_hash,
        // No live frontier exam backs this generator (see the boundary
        // note above); this hash is self-referential (over the tool name +
        // family), not a real exam file's digest. It satisfies the shallow
        // `starts_with("blake3:")` shape check but will correctly fail the
        // deeper `verify_steering_inputs` cross-check against an actual
        // `--steering-frontier-exam`, which is the intended behavior.
        source_exam_file_hash: format!(
            "blake3:{}",
            blake3::hash(format!("no-frontier-exam:{tool}").as_bytes()).to_hex()
        ),
        source_exam_fingerprint: format!("engine-generated-no-frontier-exam:{tool}"),
        tool: tool.to_string(),
        e1_targets_mode: e1_targets_mode.to_string(),
        e1_raw_invalid_dropped,
        schema_invalid_dropped,
    };
    let manifest_path = dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?).map_err(|source| {
        GeneratorError::Io {
            path: manifest_path,
            source,
        }
    })?;

    Ok(SidecarWriteStats {
        kept: sft.len(),
        e1_raw_invalid_dropped,
        schema_invalid_dropped,
    })
}

/// Load and verify a frozen `ToolContractSnapshot` (`contracts.json`) from
/// disk. `verify_integrity` recomputes `manifest_hash` from the tool list —
/// a hand-edited or truncated snapshot fails closed here rather than
/// silently mislabeling every sidecar record with a stale hash.
fn load_frozen_contracts(path: &Path) -> GenResult<ToolContractSnapshot> {
    let bytes = fs::read(path).map_err(|source| GeneratorError::ContractsIo {
        path: path.to_path_buf(),
        source,
    })?;
    let snapshot: ToolContractSnapshot = serde_json::from_slice(&bytes)?;
    snapshot.verify_integrity()?;
    Ok(snapshot)
}

fn run(args: &Args) -> GenResult<()> {
    let book = TemplateBook::load(&args.templates_dir)?;
    let sentinels = SentinelIndex::load(&args.sentinel_paths)?
        .subtract_baseline(&args.sentinel_baseline_paths)?;
    fs::create_dir_all(&args.out_dir).map_err(|source| GeneratorError::Io {
        path: args.out_dir.clone(),
        source,
    })?;

    let scratch = ScratchDir::new("main")?;
    let engine = build_engine(scratch.path())?;
    let agent_preds = AgentPredicates::init(&engine)?;
    let task_preds = TaskPredicates::init(&engine)?;
    let mut causal = CausalManager::new(&engine)?;
    let mut ec = EventCalculusEngine::new(&engine)?;

    let memory_write_manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();
    let create_manifest =
        StructuredTaskCreateTool::new(task_preds.clone(), agent_preds.clone()).manifest();
    let update_manifest = StructuredTaskUpdateTool::new(task_preds.clone()).manifest();
    let reactor_manifest = ReactorManageTool.manifest();

    let seed = Seed(args.seed);

    let memory_write_records = generate_memory_write(
        &engine,
        &agent_preds,
        &mut causal,
        &mut ec,
        &book,
        &memory_write_manifest,
        seed,
        args.count,
    )?;
    let (create_records, update_records) = generate_structured_task(
        &engine,
        &task_preds,
        &agent_preds,
        &mut causal,
        &mut ec,
        &book,
        &create_manifest,
        &update_manifest,
        seed,
        args.count,
        args.e2_prompt,
    )?;
    let reactor_records = generate_reactor_manage(
        &engine,
        &mut causal,
        &mut ec,
        &book,
        &reactor_manifest,
        seed,
        args.count,
        args.e2_prompt,
    )?;
    let oracle_records = generate_oracle(&book, seed, args.oracle_seeds)?;
    let revision_records = if args.e4_revision {
        generate_revision_pairs(&book, seed, args.e4_count)?
    } else {
        Vec::new()
    };

    engine.persist()?;

    let mut decon = DecontaminationSummary {
        method: format!("nfkc_lower_whitespace_exact+word_{SENTINEL_WORD_NGRAM_SIZE}gram"),
        word_ngram_size: SENTINEL_WORD_NGRAM_SIZE,
        sentinel_files: sentinels.files,
        sentinel_texts_indexed: sentinels.texts,
        baseline_files: sentinels.baseline_files,
        baseline_texts_indexed: sentinels.baseline_texts,
        sentinel_ngrams_removed_by_baseline: sentinels.ngrams_removed_by_baseline,
        sentinel_exact_removed_by_baseline: sentinels.exact_removed_by_baseline,
        ..Default::default()
    };

    // Real (not fabricated) tool-contract snapshot over this generator's
    // four tools — `ToolContractSnapshot::from_serializable` is grug-data's
    // own bridge for snapshotting runtime manifests without a build-time
    // dependency the other way (`crates/grug-data/src/tool_contract.rs`).
    // `contract_hash` below is genuinely computed this way; it will not
    // equal a specific pod run's `contracts.json` hash, which snapshots the
    // akh's *complete* tool catalog rather than just these four.
    let contract_snapshot = ToolContractSnapshot::from_serializable(&[
        memory_write_manifest.clone(),
        create_manifest.clone(),
        update_manifest.clone(),
        reactor_manifest.clone(),
    ])
    .map_err(|error| {
        GeneratorError::Agent(AgentError::ToolExecution {
            tool_name: "tool_contract_snapshot".into(),
            message: error.to_string(),
        })
    })?;
    let contract_hash = contract_snapshot.manifest_hash.clone();

    // P6.3b: the frozen tool-contract snapshot the base SFT corpus and
    // release-gate exam were built against. When present, every primary
    // tool-contract-family record's `contract_hash` is overlaid with this
    // (not the ad hoc four-tool `contract_hash` above), and the steering
    // sidecar is rebuilt in that snapshot's own distribution (see
    // `build_sidecar_record`). Without it, sidecar emission is skipped
    // entirely rather than emitting an out-of-distribution sidecar
    // `train_knowledge` would reject anyway.
    let frozen_contracts = args
        .contracts
        .as_ref()
        .map(|path| load_frozen_contracts(path))
        .transpose()?;
    let mut memory_write_records = memory_write_records;
    let mut create_records = create_records;
    let mut update_records = update_records;
    let mut reactor_records = reactor_records;
    if let Some(frozen) = &frozen_contracts {
        for records in [
            &mut memory_write_records,
            &mut create_records,
            &mut update_records,
            &mut reactor_records,
        ] {
            for record in records.iter_mut() {
                record.contract_hash = frozen.manifest_hash.clone();
            }
        }
    }

    let mut per_family = BTreeMap::new();
    let mut sidecar_per_family = BTreeMap::new();
    let mut e1_raw_invalid_dropped_from_sidecar = 0u64;
    let mut sidecar_schema_invalid_dropped = 0u64;
    let (kept, survivors) = write_family_jsonl(
        &args.out_dir,
        Family::MemoryWrite.file_stem(),
        memory_write_records,
        &sentinels,
        &mut decon,
    )?;
    if let Some(frozen) = &frozen_contracts {
        let stats = write_steering_sidecar(
            &args.out_dir,
            Family::MemoryWrite.tag(),
            "memory_write",
            "memory_write",
            frozen,
            &survivors,
            args.e1_targets,
        )?;
        e1_raw_invalid_dropped_from_sidecar += stats.e1_raw_invalid_dropped;
        sidecar_schema_invalid_dropped += stats.schema_invalid_dropped;
        sidecar_per_family.insert("memory_write".to_string(), stats.kept as u64);
    }
    per_family.insert(Family::MemoryWrite.file_stem().to_string(), kept);

    let (kept, survivors) = write_family_jsonl(
        &args.out_dir,
        Family::StructuredTaskCreate.file_stem(),
        create_records,
        &sentinels,
        &mut decon,
    )?;
    if let Some(frozen) = &frozen_contracts {
        let stats = write_steering_sidecar(
            &args.out_dir,
            Family::StructuredTaskCreate.tag(),
            "structured_task_create",
            "structured_task_create",
            frozen,
            &survivors,
            args.e1_targets,
        )?;
        e1_raw_invalid_dropped_from_sidecar += stats.e1_raw_invalid_dropped;
        sidecar_schema_invalid_dropped += stats.schema_invalid_dropped;
        sidecar_per_family.insert("structured_task_create".to_string(), stats.kept as u64);
    }
    per_family.insert(Family::StructuredTaskCreate.file_stem().to_string(), kept);

    let (kept, survivors) = write_family_jsonl(
        &args.out_dir,
        Family::StructuredTaskUpdate.file_stem(),
        update_records,
        &sentinels,
        &mut decon,
    )?;
    if let Some(frozen) = &frozen_contracts {
        let stats = write_steering_sidecar(
            &args.out_dir,
            Family::StructuredTaskUpdate.tag(),
            "structured_task_update",
            "structured_task_update",
            frozen,
            &survivors,
            args.e1_targets,
        )?;
        e1_raw_invalid_dropped_from_sidecar += stats.e1_raw_invalid_dropped;
        sidecar_schema_invalid_dropped += stats.schema_invalid_dropped;
        sidecar_per_family.insert("structured_task_update".to_string(), stats.kept as u64);
    }
    per_family.insert(Family::StructuredTaskUpdate.file_stem().to_string(), kept);

    let (kept, survivors) = write_family_jsonl(
        &args.out_dir,
        Family::ReactorManage.file_stem(),
        reactor_records,
        &sentinels,
        &mut decon,
    )?;
    if let Some(frozen) = &frozen_contracts {
        let stats = write_steering_sidecar(
            &args.out_dir,
            Family::ReactorManage.tag(),
            "reactor_manage",
            "reactor_manage",
            frozen,
            &survivors,
            args.e1_targets,
        )?;
        e1_raw_invalid_dropped_from_sidecar += stats.e1_raw_invalid_dropped;
        sidecar_schema_invalid_dropped += stats.schema_invalid_dropped;
        sidecar_per_family.insert("reactor_manage".to_string(), stats.kept as u64);
    }
    per_family.insert(Family::ReactorManage.file_stem().to_string(), kept);

    let (kept, _survivors) = write_family_jsonl(
        &args.out_dir,
        "oracle",
        oracle_records,
        &sentinels,
        &mut decon,
    )?;
    per_family.insert("oracle".to_string(), kept);

    let mut e4_sidecar_treatment = None;
    let mut e4_sidecar_control = None;
    if args.e4_revision {
        let (kept, survivors) = write_family_jsonl(
            &args.out_dir,
            Family::KnowledgeRevise.file_stem(),
            revision_records,
            &sentinels,
            &mut decon,
        )?;
        per_family.insert(Family::KnowledgeRevise.file_stem().to_string(), kept);

        // P6.5/E4: the trainer-facing sidecars, mirroring the four
        // tool-contract families' `<tool>/{train.sft.jsonl,manifest.json,
        // separation.json}` shape so `train_knowledge` can load this
        // curriculum too. Two directories, not one, because treatment
        // (`e4_role == "treatment"`) and control (`"control"`) are
        // deliberately different arms of the experiment — see
        // `build_sidecar_record`'s doc comment on why the request is always
        // rendered from the record's own completed target. Skipped (both
        // stay `None`) when `--contracts` was not given, exactly like the
        // four existing sidecars.
        if let Some(frozen) = &frozen_contracts {
            let (treatment_survivors, control_survivors): (Vec<_>, Vec<_>) =
                survivors.into_iter().partition(|record| {
                    record
                        .labels
                        .as_ref()
                        .and_then(|labels| labels.e4_role.as_deref())
                        == Some("treatment")
                });

            let treatment_stats = write_steering_sidecar(
                &args.out_dir,
                Family::KnowledgeRevise.tag(),
                "knowledge_revise_treatment",
                "memory_write",
                frozen,
                &treatment_survivors,
                E1TargetsMode::Completed,
            )?;
            e4_sidecar_treatment = Some(E4SidecarStats::new(
                "knowledge_revise_treatment",
                &treatment_stats,
            ));

            let control_stats = write_steering_sidecar(
                &args.out_dir,
                Family::KnowledgeRevise.tag(),
                "knowledge_revise_control",
                "memory_write",
                frozen,
                &control_survivors,
                E1TargetsMode::Completed,
            )?;
            e4_sidecar_control = Some(E4SidecarStats::new(
                "knowledge_revise_control",
                &control_stats,
            ));
        }
    }

    let manifest = Manifest {
        schema_version: 1,
        generator: "akh_episode_generator/p6.3",
        generator_git_hash: git_hash(),
        seed: args.seed,
        template_file_hashes: book.file_hashes.clone(),
        per_family_counts: per_family,
        decontamination: decon,
        e1_targets_mode: args.e1_targets.to_string(),
        e1_raw_invalid_dropped_from_sidecar,
        e2_prompt_mode: args.e2_prompt.to_string(),
        e4_revision_enabled: args.e4_revision,
        frozen_contracts_path: args
            .contracts
            .as_ref()
            .map(|path| path.display().to_string()),
        frozen_contract_hash: frozen_contracts.as_ref().map(|f| f.manifest_hash.clone()),
        ad_hoc_contract_hash: contract_hash,
        sidecar_schema_invalid_dropped,
        sidecar_per_family_counts: sidecar_per_family,
        e4_sidecar_treatment,
        e4_sidecar_control,
    };
    let manifest_path = args.out_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?).map_err(|source| {
        GeneratorError::Io {
            path: manifest_path,
            source,
        }
    })?;

    Ok(())
}

fn main() -> miette::Result<()> {
    let args = Args::parse();
    run(&args)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use akh_medu::agent::tasks::TaskListFilter;

    fn test_book() -> TemplateBook {
        TemplateBook::load(Path::new("training/episodes/templates")).expect("templates load")
    }

    fn fresh_engine() -> (ScratchDir, Engine) {
        let scratch = ScratchDir::new("test").expect("scratch dir");
        let engine = build_engine(scratch.path()).expect("engine");
        (scratch, engine)
    }

    /// A frozen 4-tool `ToolContractSnapshot` standing in for the real
    /// pod's `contracts.json` (53 tools, unreachable on this machine — see
    /// the P6.3b RESULT doc). Exactly the boundary case
    /// `offered_tool_count == contracts.tools.len() == 4`: every offered-tool
    /// set is this whole snapshot, shuffled, which is enough to exercise
    /// `sample_offered_tools`/`validate_call` without needing the pod's file.
    fn test_frozen_contracts(engine: &Engine) -> ToolContractSnapshot {
        let agent_preds = AgentPredicates::init(engine).unwrap();
        let task_preds = TaskPredicates::init(engine).unwrap();
        let mw = MemoryWriteTool::new(agent_preds.clone()).manifest();
        let create =
            StructuredTaskCreateTool::new(task_preds.clone(), agent_preds.clone()).manifest();
        let update = StructuredTaskUpdateTool::new(task_preds.clone()).manifest();
        let reactor = ReactorManageTool.manifest();
        ToolContractSnapshot::from_serializable(&[mw, create, update, reactor]).unwrap()
    }

    #[test]
    fn memory_write_completion_is_value_exact_and_labelled() {
        let (_scratch, engine) = fresh_engine();
        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();

        let records = generate_memory_write(
            &engine,
            &agent_preds,
            &mut causal,
            &mut ec,
            &book,
            &manifest,
            Seed(1),
            6,
        )
        .unwrap();
        assert_eq!(records.len(), 6);
        for record in &records {
            let labels = record.labels.as_ref().expect("labels present");
            // The completed target must actually differ from the corrupted
            // raw proposal (symbolic completion did real work).
            assert_eq!(labels.e1_completed_changed_from_raw, Some(true));
            // The completed target always parses as a valid runtime call.
            assert!(KnowledgeAction::parse_runtime(&record.target).is_ok());
            // The completed causal reward must be 1.0 — the action really
            // ran and the KG really holds the asserted triple.
            assert_eq!(labels.e3_causal_reward_completed, Some(1.0));
        }
        // At least one episode (i % 3 == 1) used the nested-JSON stress kind.
        assert!(records[1].target.contains("\\\"ip\\\""));
    }

    #[test]
    fn structured_task_create_vs_update_is_grounded() {
        let (_scratch, engine) = fresh_engine();
        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let task_preds = TaskPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let create_manifest =
            StructuredTaskCreateTool::new(task_preds.clone(), agent_preds.clone()).manifest();
        let update_manifest = StructuredTaskUpdateTool::new(task_preds.clone()).manifest();

        let (creates, updates) = generate_structured_task(
            &engine,
            &task_preds,
            &agent_preds,
            &mut causal,
            &mut ec,
            &book,
            &create_manifest,
            &update_manifest,
            Seed(7),
            4,
            E2PromptMode::None,
        )
        .unwrap();
        assert_eq!(creates.len(), 4);
        assert_eq!(updates.len(), 4);
        for record in &creates {
            assert!(record.target.contains("structured_task_create"));
        }
        for record in &updates {
            assert!(record.target.contains("structured_task_update"));
            // The completed update's id is a real numeric id resolvable in
            // the graph, never the raw placeholder.
            let labels = record.labels.as_ref().unwrap();
            assert_eq!(labels.e1_raw_target_runtime_valid, Some(true));
            assert!(!record.target.contains("<task-id>"));
        }

        // Independently verify against the real store: every update's title
        // actually exists as an open task before the update ran.
        let open = tasks::list_tasks(&engine, &task_preds, TaskListFilter::All).unwrap();
        assert!(
            open.len() >= 8,
            "4 create + 4 update-precondition tasks expected, got {}",
            open.len()
        );
    }

    #[test]
    fn reactor_manage_envelope_is_identical_across_languages() {
        let (_scratch, engine) = fresh_engine();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let manifest = ReactorManageTool.manifest();

        let records = generate_reactor_manage(
            &engine,
            &mut causal,
            &mut ec,
            &book,
            &manifest,
            Seed(3),
            5,
            E2PromptMode::None,
        )
        .unwrap();
        assert_eq!(records.len(), 5);
        let languages: HashSet<_> = records.iter().map(|r| r.language.clone()).collect();
        assert_eq!(
            languages.len(),
            5,
            "expected all 5 languages exercised across 5 records"
        );
        // Every completed target uses the identical envelope: same tool
        // name, same argument keys, regardless of language.
        for record in &records {
            let action = KnowledgeAction::parse_runtime(&record.target).expect("runtime-valid");
            match action {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    assert_eq!(tool, "reactor_manage");
                    assert_eq!(
                        arguments.get("operation").and_then(Value::as_str),
                        Some("add")
                    );
                    assert!(arguments.get("schedule").is_some());
                }
                other => panic!("expected a tool call, got {other:?}"),
            }
            // The raw control models the documented malformed-envelope
            // failure: it must NOT parse as a valid runtime call.
            let labels = record.labels.as_ref().unwrap();
            assert_eq!(labels.e1_raw_target_runtime_valid, Some(false));
        }
    }

    #[test]
    fn oracle_scenarios_produce_engine_verified_labels() {
        let book = test_book();
        let records = generate_oracle(&book, Seed(9), 2).unwrap();
        assert_eq!(records.len(), 6);
        for record in &records {
            let labels = record.labels.as_ref().unwrap();
            assert!(labels.e2_temporal_validity.is_some());
            assert!(labels.e3_causal_reward_completed.is_some());
        }
        // The repair scenario's completed target is a refusal (Gap), never
        // a tool call, because the permission was revoked before decision
        // time — a genuine E2-gated behavior change, not a coin flip.
        let repair = records
            .iter()
            .find(|r| r.family == Family::OracleRepair.tag())
            .unwrap();
        assert!(!repair.target.contains("<tool_call>"));
        assert_eq!(
            repair.labels.as_ref().unwrap().e2_temporal_validity,
            Some(false)
        );
    }

    #[test]
    fn sentinel_exact_text_is_excluded() {
        let dir = ScratchDir::new("sentinel-exact").unwrap();
        let sentinel_path = dir.path().join("sentinel.jsonl");
        fs::write(
            &sentinel_path,
            "{\"prompt\": \"the quick brown fox\", \"target\": \"jumps\"}\n",
        )
        .unwrap();
        let index = SentinelIndex::load(&[sentinel_path]).unwrap();
        assert_eq!(index.collision("The Quick Brown Fox"), Some("exact"));
        assert_eq!(index.collision("totally unrelated text"), None);
    }

    #[test]
    fn baseline_shingles_are_subtracted_from_the_sentinel_index() {
        let dir = tempfile::tempdir().unwrap();
        let sentinel_path = dir.path().join("sentinel.jsonl");
        let baseline_path = dir.path().join("baseline.jsonl");
        let boilerplate = "one two three four five six seven eight nine ten eleven twelve thirteen";
        let private = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu";
        std::fs::write(
            &sentinel_path,
            format!("{{\"text\":\"{boilerplate} {private}\"}}\n"),
        )
        .unwrap();
        std::fs::write(&baseline_path, format!("{{\"text\":\"{boilerplate}\"}}\n")).unwrap();
        let index = SentinelIndex::load(std::slice::from_ref(&sentinel_path))
            .unwrap()
            .subtract_baseline(std::slice::from_ref(&baseline_path))
            .unwrap();
        assert_eq!(index.ngrams_removed_by_baseline, 1);
        assert!(
            index.collision(boilerplate).is_none(),
            "shared boilerplate must not collide"
        );
        assert_eq!(index.collision(private), Some("13-word-shingle"));
    }

    #[test]
    fn sentinel_thirteen_word_shingle_is_excluded() {
        let dir = ScratchDir::new("sentinel-ngram").unwrap();
        let sentinel_path = dir.path().join("sentinel.jsonl");
        let long_sentence = (0..20)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        fs::write(
            &sentinel_path,
            format!("{{\"prompt\": \"{long_sentence}\", \"target\": \"x\"}}\n"),
        )
        .unwrap();
        let index = SentinelIndex::load(&[sentinel_path]).unwrap();
        // A 13-word window entirely contained in the sentinel's 20-word
        // prompt must collide even though the surrounding text differs.
        let overlapping = (2..16)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let probe = format!("totally different prefix {overlapping} totally different suffix");
        assert_eq!(index.collision(&probe), Some("13-word-shingle"));
        // Fewer than 13 shared words must not collide.
        let too_short = (0..5)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(index.collision(&too_short), None);
    }

    #[test]
    fn clean_record_passes_decontamination() {
        let index = SentinelIndex::empty();
        assert_eq!(
            index.record_collision(&EpisodeRecord {
                schema_version: 1,
                id: "x".into(),
                episode_id: "x".into(),
                step: 0,
                stage: "answer".into(),
                family: "tool_contract.memory_write".into(),
                difficulty: 1,
                language: "english".into(),
                contract_hash: "blake3:x".into(),
                tools: vec![],
                conversation: vec![],
                system_context: None,
                prompt: "anything at all".into(),
                target: "<tool_call>\n{\"name\":\"memory_write\",\"arguments\":{}}\n</tool_call>"
                    .into(),
                labels: None,
            }),
            None
        );
    }

    #[test]
    fn template_placeholder_substitution_round_trips() {
        let book = test_book();
        let rendered = book
            .render(
                "memory_write",
                "record_field",
                "en",
                &[
                    ("doc_label", "doc:x"),
                    ("source_excerpt", "hello"),
                    ("predicate", "has-field"),
                ],
            )
            .unwrap();
        assert!(!rendered.contains('{'));
        assert!(rendered.contains("doc:x"));
    }

    #[test]
    fn causal_label_negative_control_returns_zero() {
        let (_scratch, engine) = fresh_engine();
        let mut causal = CausalManager::new(&engine).unwrap();
        let s = engine
            .resolve_or_create_entity("never-written-subject")
            .unwrap();
        let p = engine
            .resolve_or_create_relation("never-written-predicate")
            .unwrap();
        let o = engine
            .resolve_or_create_entity("never-written-object")
            .unwrap();
        // Nothing was ever asserted for (s, p, o); verify_prediction must
        // come back false, proving the mechanism can genuinely fail.
        let reward = causal_label(&mut causal, &engine, "negative-control", (s, p, o)).unwrap();
        assert_eq!(reward, 0.0);
    }

    #[test]
    fn steering_sidecar_records_are_real_knowledge_sft_records() {
        let (_scratch, engine) = fresh_engine();
        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();
        let records = generate_memory_write(
            &engine,
            &agent_preds,
            &mut causal,
            &mut ec,
            &book,
            &manifest,
            Seed(4),
            3,
        )
        .unwrap();

        let frozen = test_frozen_contracts(&engine);
        let sft =
            match build_sidecar_record(&frozen, &records[0], "memory_write", &records[0].target)
                .unwrap()
            {
                SidecarBuildOutcome::Built(record) => *record,
                _ => panic!("expected a built sidecar record, got a drop outcome instead"),
            };
        // Round-trips through the exact grug_data type used by
        // `train_knowledge`'s plain --train-jsonl loader.
        let line = serde_json::to_string(&sft).unwrap();
        let reparsed: KnowledgeSftRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(reparsed.family, "tool_contract.memory_write");
        assert_eq!(reparsed.contract_hash, frozen.manifest_hash);
        assert!(matches!(reparsed.language, WorkflowLanguage::English));
        assert!(matches!(reparsed.stage, WorkflowStage::Answer));
        // Four offered tools, target present, every entry taken verbatim
        // from the frozen contract's own `openai_definition()`.
        assert_eq!(reparsed.tools.len(), 4);
        let offered_names: HashSet<_> = reparsed
            .tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(offered_names.contains("memory_write"));
        for tool in &reparsed.tools {
            let name = tool["function"]["name"].as_str().unwrap();
            assert_eq!(*tool, frozen.tool(name).unwrap().openai_definition());
        }
    }

    #[test]
    fn italian_and_spanish_are_excluded_from_the_sft_sidecar_not_silently_corrupted() {
        assert_eq!(workflow_language("italian"), None);
        assert_eq!(workflow_language("spanish"), None);
        assert_eq!(
            workflow_language("english"),
            Some(WorkflowLanguage::English)
        );
    }

    #[test]
    fn write_steering_sidecar_produces_the_pods_artifact_shape() {
        let dir = ScratchDir::new("sidecar-shape").unwrap();
        let (_scratch, engine) = fresh_engine();
        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();
        let records = generate_memory_write(
            &engine,
            &agent_preds,
            &mut causal,
            &mut ec,
            &book,
            &manifest,
            Seed(5),
            4,
        )
        .unwrap();

        let frozen = test_frozen_contracts(&engine);
        let stats = write_steering_sidecar(
            dir.path(),
            "tool_contract.memory_write",
            "memory_write",
            "memory_write",
            &frozen,
            &records,
            E1TargetsMode::Completed,
        )
        .unwrap();
        assert_eq!(stats.kept, 4);
        assert_eq!(stats.e1_raw_invalid_dropped, 0);
        assert_eq!(stats.schema_invalid_dropped, 0);
        let family_dir = dir.path().join("memory_write");
        assert!(family_dir.join("train.sft.jsonl").exists());
        assert!(family_dir.join("manifest.json").exists());
        assert!(family_dir.join("separation.json").exists());

        let manifest_json: SteeringManifest =
            serde_json::from_str(&fs::read_to_string(family_dir.join("manifest.json")).unwrap())
                .unwrap();
        assert!(manifest_json.complete);
        assert_eq!(manifest_json.records, 4);
        assert_eq!(manifest_json.contract_hash, frozen.manifest_hash);
        assert!(manifest_json.sft_hash.starts_with("blake3:"));
        assert!(manifest_json.separation_hash.starts_with("blake3:"));

        // The manifest's declared sft_hash must equal the real file hash —
        // exactly what `verify_steering_inputs` checks.
        let actual_hash = file_blake3(&family_dir.join("train.sft.jsonl")).unwrap();
        assert_eq!(manifest_json.sft_hash, actual_hash);
    }

    #[test]
    fn e1_raw_valid_mode_drops_invalid_and_swaps_valid_targets() {
        // memory_write's raw target is always runtime-valid (schema-valid,
        // value-wrong): kept, with the sidecar target swapped to the raw one.
        let (_scratch, engine) = fresh_engine();
        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();
        let mw_manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();
        let mw_records = generate_memory_write(
            &engine,
            &agent_preds,
            &mut causal,
            &mut ec,
            &book,
            &mw_manifest,
            Seed(11),
            3,
        )
        .unwrap();
        for r in &mw_records {
            assert_eq!(
                r.labels.as_ref().unwrap().e1_raw_target_runtime_valid,
                Some(true)
            );
        }
        let frozen = test_frozen_contracts(&engine);
        let dir = ScratchDir::new("e1-raw-valid-mw").unwrap();
        let stats = write_steering_sidecar(
            dir.path(),
            "tool_contract.memory_write",
            "memory_write",
            "memory_write",
            &frozen,
            &mw_records,
            E1TargetsMode::RawValid,
        )
        .unwrap();
        assert_eq!(stats.kept, mw_records.len());
        assert_eq!(stats.e1_raw_invalid_dropped, 0);
        assert_eq!(stats.schema_invalid_dropped, 0);
        let sft_lines: Vec<KnowledgeSftRecord> =
            fs::read_to_string(dir.path().join("memory_write/train.sft.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
        for (sft, record) in sft_lines.iter().zip(mw_records.iter()) {
            let raw = record
                .labels
                .as_ref()
                .unwrap()
                .e1_raw_target
                .clone()
                .unwrap();
            assert_eq!(
                sft.target, raw,
                "sidecar target must be swapped to the raw target"
            );
            assert_ne!(
                sft.target, record.target,
                "raw-valid mode must not carry the completed target"
            );
        }

        // reactor_manage's raw target is a deliberately malformed envelope:
        // always dropped, regardless of count.
        let mut causal2 = CausalManager::new(&engine).unwrap();
        let mut ec2 = EventCalculusEngine::new(&engine).unwrap();
        let reactor_manifest = ReactorManageTool.manifest();
        let reactor_records = generate_reactor_manage(
            &engine,
            &mut causal2,
            &mut ec2,
            &book,
            &reactor_manifest,
            Seed(12),
            4,
            E2PromptMode::None,
        )
        .unwrap();
        for r in &reactor_records {
            assert_eq!(
                r.labels.as_ref().unwrap().e1_raw_target_runtime_valid,
                Some(false)
            );
        }
        let dir2 = ScratchDir::new("e1-raw-valid-reactor").unwrap();
        let stats2 = write_steering_sidecar(
            dir2.path(),
            "tool_contract.reactor_manage",
            "reactor_manage",
            "reactor_manage",
            &frozen,
            &reactor_records,
            E1TargetsMode::RawValid,
        )
        .unwrap();
        assert_eq!(
            stats2.kept, 0,
            "every reactor_manage raw target is schema-invalid"
        );
        assert_eq!(stats2.e1_raw_invalid_dropped, reactor_records.len() as u64);
        assert_eq!(stats2.schema_invalid_dropped, 0);
    }

    #[test]
    fn e2_raw_events_and_projection_prompts_differ_and_gate_the_target() {
        let book = test_book();
        let manifest = ReactorManageTool.manifest();

        let run_mode = |seed: u64, mode: E2PromptMode| {
            let (_scratch, engine) = fresh_engine();
            let mut causal = CausalManager::new(&engine).unwrap();
            let mut ec = EventCalculusEngine::new(&engine).unwrap();
            generate_reactor_manage(
                &engine,
                &mut causal,
                &mut ec,
                &book,
                &manifest,
                Seed(seed),
                6,
                mode,
            )
            .unwrap()
        };

        let none_records = run_mode(30, E2PromptMode::None);
        let raw_events_records = run_mode(30, E2PromptMode::RawEvents);
        let projection_records = run_mode(30, E2PromptMode::Projection);

        // Index 5 is the revoked case (`i % 6 == 5`) in every mode — the
        // engine-computed validity itself must not depend on the prompt mode.
        assert_eq!(
            none_records[5]
                .labels
                .as_ref()
                .unwrap()
                .e2_temporal_validity,
            Some(false)
        );
        assert_eq!(
            raw_events_records[5]
                .labels
                .as_ref()
                .unwrap()
                .e2_temporal_validity,
            Some(false)
        );
        assert_eq!(
            projection_records[5]
                .labels
                .as_ref()
                .unwrap()
                .e2_temporal_validity,
            Some(false)
        );

        // The prompt text itself differs across the three modes for the
        // same underlying episode.
        let none_prompt = &none_records[5].conversation[0]["content"];
        let raw_prompt = &raw_events_records[5].conversation[0]["content"];
        let proj_prompt = &projection_records[5].conversation[0]["content"];
        assert_ne!(
            none_prompt, raw_prompt,
            "none vs raw-events prompt must differ"
        );
        assert_ne!(
            none_prompt, proj_prompt,
            "none vs projection prompt must differ"
        );
        assert_ne!(
            raw_prompt, proj_prompt,
            "raw-events vs projection prompt must differ"
        );

        // `none` mode: target is always `add`, unaffected by validity — the
        // P6.3 behavior, unchanged.
        assert!(matches!(
            KnowledgeAction::parse_runtime(&none_records[5].target).unwrap(),
            KnowledgeAction::ToolCall { .. }
        ));

        // `raw-events`/`projection` modes: revoked ⇒ a refusal, never `add`.
        for (records, mode) in [
            (&raw_events_records, E2PromptMode::RawEvents),
            (&projection_records, E2PromptMode::Projection),
        ] {
            let revoked_action = KnowledgeAction::parse_runtime(&records[5].target).unwrap();
            assert!(
                matches!(revoked_action, KnowledgeAction::Gap { .. }),
                "revoked episode in {mode} mode must not target `add`, got {revoked_action:?}"
            );
            assert_eq!(
                records[5]
                    .labels
                    .as_ref()
                    .unwrap()
                    .e2_prompt_mode
                    .as_deref(),
                Some(mode.to_string()).as_deref()
            );
        }

        // Valid (i=0) ⇒ `add` target in every mode.
        for records in [&none_records, &raw_events_records, &projection_records] {
            match KnowledgeAction::parse_runtime(&records[0].target).unwrap() {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    assert_eq!(tool, "reactor_manage");
                    assert_eq!(
                        arguments.get("operation").and_then(Value::as_str),
                        Some("add")
                    );
                }
                other => panic!("expected an `add` tool call for the valid episode, got {other:?}"),
            }
        }
    }

    #[test]
    fn e4_revised_graph_target_differs_from_append_only_control() {
        let book = test_book();
        let treatment = generate_revision_treatment(&book, Seed(40), 0).unwrap();
        let control = generate_revision_control(&book, Seed(40), 0).unwrap();

        assert_ne!(
            treatment.target, control.target,
            "the revised-graph target must differ from the append-only control"
        );
        let t_labels = treatment.labels.as_ref().unwrap();
        let c_labels = control.labels.as_ref().unwrap();
        assert_eq!(t_labels.e4_role.as_deref(), Some("treatment"));
        assert_eq!(t_labels.e4_retraction_applied, Some(true));
        assert_eq!(t_labels.e4_stale_edge_still_present, Some(false));
        assert_eq!(c_labels.e4_role.as_deref(), Some("control"));
        assert_eq!(c_labels.e4_retraction_applied, Some(false));
        assert_eq!(
            c_labels.e4_stale_edge_still_present,
            Some(true),
            "the append-only control must leave the stale edge unresolved"
        );
    }

    #[test]
    fn generate_revision_pairs_produces_treatment_and_control_siblings() {
        let book = test_book();
        let records = generate_revision_pairs(&book, Seed(41), 3).unwrap();
        assert_eq!(records.len(), 6);
        let treatments = records
            .iter()
            .filter(|r| r.labels.as_ref().unwrap().e4_role.as_deref() == Some("treatment"))
            .count();
        let controls = records
            .iter()
            .filter(|r| r.labels.as_ref().unwrap().e4_role.as_deref() == Some("control"))
            .count();
        assert_eq!(treatments, 3);
        assert_eq!(controls, 3);
        for r in &records {
            assert_eq!(r.family, Family::KnowledgeRevise.tag());
        }
    }

    /// P6.5/E4 sidecars: `knowledge_revise_treatment/` and
    /// `knowledge_revise_control/` are the trainer-facing counterparts of
    /// `knowledge_revise.jsonl`, mirroring the four tool-contract families'
    /// `<dir>/{train.sft.jsonl,manifest.json,separation.json}` shape so
    /// `train_knowledge` can mix this curriculum in too. Also the
    /// acceptance test for the "do treatment/control prompts differ"
    /// question the task asked to settle empirically: the *primary*
    /// `EpisodeRecord.prompt` (book-templated from job/worker names only)
    /// is identical across a pair, but the *sidecar's* rendered request
    /// (`build_sidecar_record` → `grug_data::render_request`, which lists
    /// every top-level argument of the record's own completed target) is
    /// NOT — the retraction changes the completed `memory_write` arguments
    /// themselves (treatment carries `tags`/`source`, the append-only
    /// control does not), so the two arms' sidecar prompts genuinely
    /// describe different situations, not just different targets. This is
    /// unlike E1, where the sidecar deliberately keeps the *same* prompt
    /// across arms and swaps only the target.
    #[test]
    fn e4_sidecars_are_written_for_treatment_and_control() {
        let book = test_book();
        let (_scratch, engine) = fresh_engine();
        let frozen = test_frozen_contracts(&engine);

        let records = generate_revision_pairs(&book, Seed(42), 5).unwrap();
        assert_eq!(records.len(), 10);

        // Primary-record prompt identity within a pair (same index i):
        // book.render only substitutes job/old_worker/new_worker names,
        // which `revision_pair_setup` derives from `(seed, i)` alone (not
        // the treatment/control label), so the rendered request — and
        // therefore the whole primary `prompt` — is byte-identical across
        // the pair. Only `target` (and the labels) differ.
        for pair in records.chunks(2) {
            let (treatment, control) = (&pair[0], &pair[1]);
            assert_eq!(
                treatment.labels.as_ref().unwrap().e4_role.as_deref(),
                Some("treatment")
            );
            assert_eq!(
                control.labels.as_ref().unwrap().e4_role.as_deref(),
                Some("control")
            );
            assert_eq!(
                treatment.prompt, control.prompt,
                "primary record prompt must be identical across a treatment/control pair"
            );
            assert_ne!(
                treatment.target, control.target,
                "targets must differ: retraction+tags/source vs. append-only"
            );
        }

        let (treatment_records, control_records): (Vec<EpisodeRecord>, Vec<EpisodeRecord>) =
            records.into_iter().partition(|r| {
                r.labels.as_ref().and_then(|l| l.e4_role.as_deref()) == Some("treatment")
            });
        assert_eq!(treatment_records.len(), 5);
        assert_eq!(control_records.len(), 5);

        let dir = ScratchDir::new("e4-sidecars").unwrap();
        let treatment_stats = write_steering_sidecar(
            dir.path(),
            Family::KnowledgeRevise.tag(),
            "knowledge_revise_treatment",
            "memory_write",
            &frozen,
            &treatment_records,
            E1TargetsMode::Completed,
        )
        .unwrap();
        let control_stats = write_steering_sidecar(
            dir.path(),
            Family::KnowledgeRevise.tag(),
            "knowledge_revise_control",
            "memory_write",
            &frozen,
            &control_records,
            E1TargetsMode::Completed,
        )
        .unwrap();

        assert_eq!(treatment_stats.kept, 5);
        assert_eq!(control_stats.kept, 5);
        assert_eq!(treatment_stats.e1_raw_invalid_dropped, 0);
        assert_eq!(control_stats.e1_raw_invalid_dropped, 0);
        assert_eq!(treatment_stats.schema_invalid_dropped, 0);
        assert_eq!(control_stats.schema_invalid_dropped, 0);

        for dir_name in ["knowledge_revise_treatment", "knowledge_revise_control"] {
            let family_dir = dir.path().join(dir_name);
            assert!(family_dir.join("train.sft.jsonl").exists());
            assert!(family_dir.join("manifest.json").exists());
            assert!(family_dir.join("separation.json").exists());
            let manifest_json: SteeringManifest = serde_json::from_str(
                &fs::read_to_string(family_dir.join("manifest.json")).unwrap(),
            )
            .unwrap();
            assert!(manifest_json.complete);
            assert_eq!(manifest_json.records, 5);
            assert_eq!(manifest_json.family, Family::KnowledgeRevise.tag());
            assert_eq!(manifest_json.tool, "memory_write");
            assert_eq!(manifest_json.contract_hash, frozen.manifest_hash);
        }

        let load_sft = |dir_name: &str| -> Vec<KnowledgeSftRecord> {
            fs::read_to_string(dir.path().join(dir_name).join("train.sft.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let treatment_sft = load_sft("knowledge_revise_treatment");
        let control_sft = load_sft("knowledge_revise_control");
        assert_eq!(treatment_sft.len(), 5);
        assert_eq!(control_sft.len(), 5);

        // Every sidecar target must validate against the frozen contract.
        for sft in treatment_sft.iter().chain(control_sft.iter()) {
            match KnowledgeAction::parse_runtime(&sft.target).unwrap() {
                KnowledgeAction::ToolCall { tool, arguments } => {
                    assert_eq!(tool, "memory_write");
                    assert!(
                        frozen.validate_call(&tool, &arguments).valid,
                        "sidecar target must validate against the frozen contract"
                    );
                }
                other => panic!("expected a memory_write tool call, got {other:?}"),
            }
        }

        // The prompt-identity finding, now checked against the actual
        // written sidecars (not just reasoned about): treatment/control
        // sidecar prompts differ at every paired index.
        for (t, c) in treatment_sft.iter().zip(control_sft.iter()) {
            assert_ne!(
                t.prompt, c.prompt,
                "E4 sidecar prompts differ by design (see the test's doc comment)"
            );
        }

        // Determinism: regenerating with the same seed must be byte-identical.
        let records2 = generate_revision_pairs(&book, Seed(42), 5).unwrap();
        let (treatment_records2, control_records2): (Vec<EpisodeRecord>, Vec<EpisodeRecord>) =
            records2.into_iter().partition(|r| {
                r.labels.as_ref().and_then(|l| l.e4_role.as_deref()) == Some("treatment")
            });
        let dir3 = ScratchDir::new("e4-sidecars-repeat").unwrap();
        write_steering_sidecar(
            dir3.path(),
            Family::KnowledgeRevise.tag(),
            "knowledge_revise_treatment",
            "memory_write",
            &frozen,
            &treatment_records2,
            E1TargetsMode::Completed,
        )
        .unwrap();
        write_steering_sidecar(
            dir3.path(),
            Family::KnowledgeRevise.tag(),
            "knowledge_revise_control",
            "memory_write",
            &frozen,
            &control_records2,
            E1TargetsMode::Completed,
        )
        .unwrap();
        for dir_name in ["knowledge_revise_treatment", "knowledge_revise_control"] {
            let a = fs::read_to_string(dir.path().join(dir_name).join("train.sft.jsonl")).unwrap();
            let b = fs::read_to_string(dir3.path().join(dir_name).join("train.sft.jsonl")).unwrap();
            assert_eq!(
                a, b,
                "{dir_name}/train.sft.jsonl must be byte-identical across runs with the same seed"
            );
        }
    }

    /// P6.3b acceptance test: for each of the four tool-contract families,
    /// build one primary record, render its sidecar counterpart against a
    /// frozen contract loaded from disk (exercising the real `--contracts`
    /// path via `load_frozen_contracts`), and assert structural equality
    /// against a *real* base-corpus record for the same family/frozen
    /// snapshot generated by grug-data's own `ToolCurriculumGenerator`.
    ///
    /// No pod-produced `tools.train.sft.jsonl`/`contracts.json`/
    /// `all-tools.json` is reachable on this machine (checked before writing
    /// this generator — see the P6.3b RESULT doc's "reused vs replicated"
    /// section), so the real `ToolCurriculumGenerator` run against the same
    /// frozen snapshot is the authoritative in-repo stand-in for "one real
    /// base-corpus record of the same family": it exercises the actual base
    /// corpus's code path, not a hand-authored fixture that could silently
    /// drift from it.
    #[test]
    fn sidecar_matches_base_corpus_distribution_per_family() {
        use grug_data::ToolCurriculumGenerator;

        let (_scratch, engine) = fresh_engine();
        let frozen = test_frozen_contracts(&engine);

        // Round-trip the frozen snapshot through disk via the same loader
        // `--contracts` uses, so `contract_hash` below is genuinely "read
        // from the file", not the in-memory value we just built.
        let contracts_dir = ScratchDir::new("contracts-fixture").unwrap();
        let contracts_path = contracts_dir.path().join("contracts.json");
        fs::write(&contracts_path, serde_json::to_string(&frozen).unwrap()).unwrap();
        let loaded = load_frozen_contracts(&contracts_path).unwrap();
        assert_eq!(loaded.manifest_hash, frozen.manifest_hash);

        let base_gen = ToolCurriculumGenerator::new(loaded.clone(), 20_260_913, 4).unwrap();
        let base_records = base_gen.generate("p6-3b-acceptance", 1).unwrap();

        let agent_preds = AgentPredicates::init(&engine).unwrap();
        let task_preds = TaskPredicates::init(&engine).unwrap();
        let mut causal = CausalManager::new(&engine).unwrap();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();
        let book = test_book();

        let sidecars: Vec<(&str, KnowledgeSftRecord)> = {
            let mut out = Vec::new();

            let mw_manifest = MemoryWriteTool::new(agent_preds.clone()).manifest();
            let mw_records = generate_memory_write(
                &engine,
                &agent_preds,
                &mut causal,
                &mut ec,
                &book,
                &mw_manifest,
                Seed(101),
                1,
            )
            .unwrap();
            out.push((
                "memory_write",
                match build_sidecar_record(
                    &loaded,
                    &mw_records[0],
                    "memory_write",
                    &mw_records[0].target,
                )
                .unwrap()
                {
                    SidecarBuildOutcome::Built(r) => *r,
                    _ => panic!("memory_write sidecar record must build"),
                },
            ));

            let create_manifest =
                StructuredTaskCreateTool::new(task_preds.clone(), agent_preds.clone()).manifest();
            let update_manifest = StructuredTaskUpdateTool::new(task_preds.clone()).manifest();
            let (create_records, update_records) = generate_structured_task(
                &engine,
                &task_preds,
                &agent_preds,
                &mut causal,
                &mut ec,
                &book,
                &create_manifest,
                &update_manifest,
                Seed(102),
                1,
                E2PromptMode::None,
            )
            .unwrap();
            out.push((
                "structured_task_create",
                match build_sidecar_record(
                    &loaded,
                    &create_records[0],
                    "structured_task_create",
                    &create_records[0].target,
                )
                .unwrap()
                {
                    SidecarBuildOutcome::Built(r) => *r,
                    _ => panic!("structured_task_create sidecar record must build"),
                },
            ));
            out.push((
                "structured_task_update",
                match build_sidecar_record(
                    &loaded,
                    &update_records[0],
                    "structured_task_update",
                    &update_records[0].target,
                )
                .unwrap()
                {
                    SidecarBuildOutcome::Built(r) => *r,
                    _ => panic!("structured_task_update sidecar record must build"),
                },
            ));

            let reactor_manifest = ReactorManageTool.manifest();
            let reactor_records = generate_reactor_manage(
                &engine,
                &mut causal,
                &mut ec,
                &book,
                &reactor_manifest,
                Seed(103),
                1,
                E2PromptMode::None,
            )
            .unwrap();
            out.push((
                "reactor_manage",
                match build_sidecar_record(
                    &loaded,
                    &reactor_records[0],
                    "reactor_manage",
                    &reactor_records[0].target,
                )
                .unwrap()
                {
                    SidecarBuildOutcome::Built(r) => *r,
                    _ => panic!("reactor_manage sidecar record must build"),
                },
            ));

            out
        };

        const TOOLS_MARKER: &str = "# Tools\n";
        for (tool, sidecar) in &sidecars {
            let base = base_records
                .iter()
                .find(|r| r.family == format!("tool_contract.{tool}"))
                .unwrap_or_else(|| panic!("base corpus record for family {tool}"));

            // 1. Identical system-prompt preamble up to "# Tools\n" — the
            // fixed `render_knowledge_prompt_with_system` text both records
            // share byte-for-byte.
            let base_end = base.prompt.find(TOOLS_MARKER).unwrap() + TOOLS_MARKER.len();
            let sidecar_end = sidecar.prompt.find(TOOLS_MARKER).unwrap() + TOOLS_MARKER.len();
            assert_eq!(
                &base.prompt[..base_end],
                &sidecar.prompt[..sidecar_end],
                "system prompt preamble must match for {tool}"
            );

            // 2. Four offered tools, target present, every entry equal to
            // the frozen contract's own openai_definition().
            assert_eq!(sidecar.tools.len(), 4, "{tool} must offer exactly 4 tools");
            let offered: HashSet<_> = sidecar
                .tools
                .iter()
                .filter_map(|t| t["function"]["name"].as_str())
                .collect();
            assert!(
                offered.contains(tool),
                "{tool} must be among its own offered tools"
            );
            for entry in &sidecar.tools {
                let name = entry["function"]["name"].as_str().unwrap();
                assert_eq!(
                    *entry,
                    loaded.tool(name).unwrap().openai_definition(),
                    "offered tool {name} must be exactly the frozen contract's definition"
                );
            }

            // 3. contract_hash equals the hash read from the frozen
            // contracts.json file (never a hardcoded literal).
            assert_eq!(sidecar.contract_hash, loaded.manifest_hash);

            // 4. The target validates against the frozen contract.
            match KnowledgeAction::parse_runtime(&sidecar.target).unwrap() {
                KnowledgeAction::ToolCall {
                    tool: called,
                    arguments,
                } => {
                    assert!(
                        loaded.validate_call(&called, &arguments).valid,
                        "{tool}'s sidecar target must validate against the frozen contract"
                    );

                    // 5. Identical user-turn skeleton: recomputing
                    // `render_request` over the frozen contract's own
                    // description and the target's own arguments must
                    // reproduce the sidecar's user content exactly — proof
                    // it really is grug-data's phrase table, not a
                    // reimplementation, for whichever of the 3 phrasings
                    // this record's seed selected.
                    let language = sidecar.language;
                    let seed = sidecar_seed(&sidecar.id);
                    let description = &loaded.tool(&called).unwrap().description;
                    let expected = render_request(language, description, &arguments, seed as usize);
                    let actual = sidecar.conversation[0]["content"].as_str().unwrap();
                    assert_eq!(
                        actual, expected,
                        "{tool}'s user turn must be exactly grug-data's own render_request output"
                    );
                }
                other => panic!("{tool}: expected a tool-call target, got {other:?}"),
            }
        }
    }
}
