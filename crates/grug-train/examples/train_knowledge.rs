//! Train a checkpointable BabyLLM on Akh-Medu's stateful knowledge workflow.
//!
//! The input is the runtime-native SFT JSONL emitted by
//! `grug-data/examples/knowledge_workflow.rs`. The trainer refuses mixed or
//! tampered tool-contract identities, keeps the held-out stream disjoint, and
//! publishes an atomic `candidate-model.json`. A separate exam gate promotes
//! a passing candidate to the `active-model.json` that Akh-Medu can load.
//!
//! ```text
//! cargo run -p grug-train --example train_knowledge --release -- \
//!   --contracts runs/knowledge/contracts.json \
//!   --train-jsonl runs/knowledge/train.sft.jsonl \
//!   --eval-jsonl runs/knowledge/eval.sft.jsonl \
//!   --artifacts-dir runs/knowledge/model-small \
//!   --model small --epochs 8
//! ```

use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use grug_data::adaptive_exam::ADAPTIVE_EXAM_SCHEMA_VERSION;
use grug_data::{
    AdaptiveExamController, ControllerDecision, ExamPlan, ExamTier,
    KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeAction, KnowledgeExamSet, KnowledgeSftRecord,
    KnowledgeToolBindings, KnowledgeWorkflowGenerator, SiblingSeparationEvidence,
    SiblingSeparationPolicy, ToolContractExamSet, ToolContractSnapshot, WarningLevel,
    derive_tool_contract_training_sibling, derive_training_sibling,
    render_knowledge_prompt_with_system,
};
use grug_train::knowledge_train::{
    KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION, KnowledgeAdaptationLineage, KnowledgeModelArtifact,
    KnowledgeModelShape, KnowledgeModelSize, KnowledgeSteeringLineage, fingerprint_artifact_file,
    fingerprint_knowledge_records, generate_knowledge_outputs, score_knowledge_outputs,
};
use grug_train::train::{init_training, to_eval_model, train_step_guarded};
use grug_train::{
    AttentionBackend, BpeTokenizer, FoundationLineage, GenerateConfig, PositionalEncoding,
    RemediationLineage, RemediationSupplement, VerifiedFoundationBundle, VerifiedKnowledgeBundle,
    build_knowledge_batch, checkpoint, read_sibling_generator_metadata,
    validate_knowledge_steering_share, validate_remediation_share, verify_active_knowledge_bundle,
    verify_candidate_knowledge_bundle, verify_foundation_bundle,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "train_knowledge")]
struct Args {
    /// Exact live-tool snapshot used to generate both SFT streams.
    #[arg(long)]
    contracts: PathBuf,

    /// Runtime-native training records, one KnowledgeSftRecord per line.
    #[arg(long)]
    train_jsonl: PathBuf,

    /// Additional canonical SFT streams (for example Dojo dialogue records)
    /// mixed into training under the same tokenizer and tool contract.
    #[arg(long)]
    supplemental_jsonl: Vec<PathBuf>,

    /// Frozen, disjoint KnowledgeSftRecord stream for teacher-forced steering.
    #[arg(long)]
    eval_jsonl: Option<PathBuf>,

    /// Held-out companions for `--supplemental-jsonl`. Episode and record
    /// identities are checked against the complete training mixture.
    #[arg(long)]
    supplemental_eval_jsonl: Vec<PathBuf>,

    /// Run directory for tokenizer, copied contracts, logs and checkpoints.
    #[arg(long)]
    artifacts_dir: PathBuf,

    /// Capacity rung: `small` (~30M at a 3k vocab) or `350m`.
    #[arg(long, default_value = "small")]
    model: KnowledgeModelSize,

    #[arg(long, default_value_t = 8)]
    epochs: usize,

    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Base Adam learning rate. Defaults to 3e-4 for small, 1e-4 for 350m.
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Linear warm-up steps; zero disables warm-up.
    #[arg(long, default_value_t = 250)]
    warmup_steps: usize,

    #[arg(long, default_value_t = 3_000)]
    bpe_vocab_size: usize,

    #[arg(long, default_value_t = 250)]
    eval_every: usize,

    #[arg(long, default_value_t = 32)]
    eval_samples: usize,

    #[arg(long, default_value_t = 384)]
    eval_max_new_tokens: usize,

    #[arg(long, default_value_t = 500)]
    ckpt_every: usize,

    /// Skip an update before a loss spike can poison Adam state.
    #[arg(long, default_value_t = 20.0)]
    loss_cap: f32,

    /// Skip an update when its pre-clip whole-model gradient norm exceeds this.
    #[arg(long, default_value_t = 100.0)]
    grad_norm_cap: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Resume model and optimizer state from a checkpoint directory.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Permit a resumed run's `learning_rate` and `warmup_steps` to differ
    /// from the resumed run.json (ADR 104 D5: ordinary adaptive-loop windows
    /// deliberately train at a reduced learning rate). Every other resume
    /// invariant -- corpus/contract hashes, model shape, batch size,
    /// optimizer type, evaluation schedule -- stays strict regardless of
    /// this flag. Off by default; the change is recorded in
    /// `run-history/schedule-change-step_<step>.json`.
    #[arg(long, default_value_t = false)]
    allow_schedule_change: bool,

    /// Verified causal-language foundation bundle. Its tokenizer and model
    /// shape become immutable inputs to this SFT run; Adam starts fresh.
    #[arg(long)]
    foundation_bundle: Option<PathBuf>,

    /// Exam-certified active BabyLLM bundle to adapt to a new immutable
    /// corpus/tool snapshot. Weights are loaded through the serving verifier,
    /// the tokenizer/foundation lineage are preserved, and Adam starts fresh.
    #[arg(long)]
    adapt_from_bundle: Option<PathBuf>,

    /// Unpromoted candidate whose optimizer/model state begins a new bounded
    /// segment after an adaptive-controller `steer` decision.
    #[arg(long)]
    steer_from_bundle: Option<PathBuf>,

    /// Uncertified candidate bundle whose weights *and* retained Adam state
    /// continue on `--train-jsonl` (base) plus one or more
    /// `--supplemental-jsonl` engine-generated siblings (P6.4 local
    /// remediation). Unlike `--adapt-from-bundle` (fresh Adam, certified
    /// parent only) or `--steer-from-bundle` (requires a live
    /// adaptive-controller decision + frontier exam), this continues an
    /// uncertified candidate's own segment on a corpus that changed without
    /// controller authorization -- e.g. a base curriculum plus siblings from
    /// `akh_episode_generator`. Requires at least one `--supplemental-jsonl`
    /// file; see `--remediation-max-share-ppm`.
    #[arg(long)]
    remediate_from_bundle: Option<PathBuf>,

    /// Ceiling on how much of a remediation segment's mixture
    /// `--supplemental-jsonl` may occupy, in parts per million (default
    /// 50_000 = 5%, matching the adaptive loop's steering ceiling). Only
    /// meaningful with `--remediate-from-bundle`.
    #[arg(long, default_value_t = grug_train::DEFAULT_REMEDIATION_MAX_SHARE_PPM)]
    remediation_max_share_ppm: u64,

    /// Complete `adaptive_exam_controller step` output authorizing the segment.
    #[arg(long)]
    steering_decision: Option<PathBuf>,

    /// Materialized frontier exam selected by the controller.
    #[arg(long)]
    steering_frontier_exam: Option<PathBuf>,

    /// Completed separation-validated sibling `manifest.json`.
    #[arg(long)]
    steering_sibling_manifest: Option<PathBuf>,

    /// Exact sibling SFT file hashed by the sibling manifest. It is
    /// automatically included in the training mixture.
    #[arg(long)]
    steering_sibling_jsonl: Option<PathBuf>,

    /// Allow `--foundation-bundle` to warm-start a longer-context SFT run
    /// from a shorter Rope foundation (ADR 104 D3, e.g. an
    /// `akh_tool_small_rope4k`-shaped 4096 foundation feeding an
    /// `akh_tool_small_8k`-shaped 8192 SFT run). Only `max_seq_len` (which
    /// must strictly grow) and, under `PositionalEncoding::Rope`, `scale`
    /// may differ between the foundation and this run's `--model` shape --
    /// every other architectural field must still match exactly. Off by
    /// default: a foundation/SFT shape mismatch stays a hard error unless
    /// this is explicitly set. See
    /// `KnowledgeModelShape::context_extension_scale`.
    #[arg(long, default_value_t = false)]
    extend_context: bool,

    /// Validate contracts, canonical prompts, split isolation, tokenizer and
    /// sequence budgets without allocating or updating model weights.
    #[arg(long)]
    validate_only: bool,
}

#[derive(Debug)]
struct CorpusInventory {
    contract_hash: String,
    fingerprint: String,
    families: BTreeSet<String>,
    languages: BTreeSet<String>,
    stages: BTreeSet<String>,
    record_ids: HashSet<String>,
    episode_ids: HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RunDescriptor {
    schema_version: u32,
    seed: u64,
    model_size: KnowledgeModelSize,
    shape: KnowledgeModelShape,
    parameter_count: usize,
    contract_hash: String,
    training_corpus_hash: String,
    eval_corpus_hash: Option<String>,
    training_records: usize,
    eval_records: usize,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    warmup_steps: usize,
    eval_every: usize,
    eval_samples: usize,
    eval_max_new_tokens: usize,
    ckpt_every: usize,
    loss_cap: f32,
    grad_norm_cap: f32,
    tokenizer_vocab_size: usize,
    train_jsonl: PathBuf,
    supplemental_jsonl: Vec<PathBuf>,
    eval_jsonl: Option<PathBuf>,
    supplemental_eval_jsonl: Vec<PathBuf>,
    foundation_bundle: Option<PathBuf>,
    foundation: Option<FoundationLineage>,
    adapt_from_bundle: Option<PathBuf>,
    adapted_from: Option<KnowledgeAdaptationLineage>,
    steer_from_bundle: Option<PathBuf>,
    steering_decision: Option<PathBuf>,
    steering_frontier_exam: Option<PathBuf>,
    steering_sibling_manifest: Option<PathBuf>,
    steering_sibling_jsonl: Option<PathBuf>,
    steered_from: Option<KnowledgeSteeringLineage>,
    #[serde(default)]
    remediate_from_bundle: Option<PathBuf>,
    #[serde(default)]
    remediated_from: Option<RemediationLineage>,
    start_step: usize,
    /// Every `--allow-schedule-change` resume applied to this bundle's
    /// lineage, oldest first. Absent from run.json files written before
    /// this field existed, so old bundles still load strictly.
    #[serde(default)]
    schedule_changes: Vec<ScheduleChange>,
}

/// One sanctioned `--allow-schedule-change` resume: the learning-rate and/or
/// warmup-step schedule changed while every other resume invariant (corpus,
/// contract, model shape, batch size, optimizer, eval schedule) held. Mirrors
/// the `run-history/schedule-change-step_<step>.json` record so the change
/// is visible both from the live `run.json` and from the immutable history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ScheduleChange {
    resumed_at_step: usize,
    epochs: usize,
    from_learning_rate: f64,
    to_learning_rate: f64,
    from_warmup_steps: usize,
    to_warmup_steps: usize,
}

#[derive(Deserialize)]
struct ControllerStepArtifact {
    controller: AdaptiveExamController,
    decisions: Vec<ControllerDecision>,
    next_plan: ExamPlan,
}

#[derive(Deserialize)]
struct TrainingSiblingManifest {
    complete: bool,
    contract_hash: String,
    family: String,
    #[serde(default)]
    episodes: Option<usize>,
    records: usize,
    source_exam_fingerprint: String,
    source_exam_file_hash: String,
    separation_hash: String,
    sft_hash: String,
}

#[derive(Serialize)]
struct TrainLogRow {
    step: usize,
    epoch: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f64,
    update_skipped: bool,
    step_ms: f64,
}

#[derive(Serialize)]
struct EvalLogRow<'a> {
    schema_version: u32,
    step: usize,
    contract_hash: &'a str,
    teacher_forced: &'a grug_train::KnowledgeTeacherForcedEval,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let device = {
        let device = grug_train::gpu_device();
        println!("train_knowledge: CubeCL GPU backend, device={device:?}");
        device
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let device = {
        println!("train_knowledge: CPU backend (burn-ndarray)");
        Default::default()
    };

    run::<ActiveBackend>(args, device)
}

fn run<B: AutodiffBackend + AttentionBackend>(
    args: Args,
    device: B::Device,
) -> Result<(), Box<dyn Error>>
where
    B::InnerBackend: AttentionBackend,
{
    std::fs::create_dir_all(&args.artifacts_dir)?;

    for (label, parent) in [
        ("--adapt-from-bundle", args.adapt_from_bundle.as_ref()),
        ("--steer-from-bundle", args.steer_from_bundle.as_ref()),
        (
            "--remediate-from-bundle",
            args.remediate_from_bundle.as_ref(),
        ),
    ] {
        if let Some(parent) = parent
            && parent.canonicalize()? == args.artifacts_dir.canonicalize()?
        {
            return Err(invalid_data(format!(
                "--artifacts-dir must differ from {label}"
            )));
        }
    }

    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&args.contracts)?))?;
    contracts.verify_integrity()?;

    let mut train_records = load_records(&args.train_jsonl)?;
    let base_only_training_records = train_records.len();
    let mut supplemental_file_records: Vec<(PathBuf, Vec<KnowledgeSftRecord>)> = Vec::new();
    for path in &args.supplemental_jsonl {
        let records = load_records(path)?;
        supplemental_file_records.push((path.clone(), records.clone()));
        train_records.extend(records);
    }
    let supplemental_total_records: usize = supplemental_file_records
        .iter()
        .map(|(_, records)| records.len())
        .sum();
    let remediation_share_ppm = if args.remediate_from_bundle.is_some() {
        Some(validate_remediation_share(
            base_only_training_records,
            supplemental_total_records,
            args.remediation_max_share_ppm,
        )?)
    } else {
        None
    };
    let steering_records = args
        .steering_sibling_jsonl
        .as_deref()
        .map(load_records)
        .transpose()?
        .unwrap_or_default();
    let base_training_records = train_records.len();
    let steering_share_ppm = if steering_records.is_empty() {
        None
    } else {
        Some(
            validate_knowledge_steering_share(base_training_records, steering_records.len())
                .map_err(invalid_data)?,
        )
    };
    train_records.extend(steering_records.iter().cloned());
    let train_inventory = validate_records("training", &train_records, &contracts)?;
    let mut eval_records = args
        .eval_jsonl
        .as_deref()
        .map(load_records)
        .transpose()?
        .unwrap_or_default();
    for path in &args.supplemental_eval_jsonl {
        eval_records.extend(load_records(path)?);
    }
    let eval_inventory = if eval_records.is_empty() {
        None
    } else {
        Some(validate_records("evaluation", &eval_records, &contracts)?)
    };
    if let Some(eval) = &eval_inventory {
        reject_holdout_leakage(&train_inventory, eval)?;
    }

    println!(
        "train_knowledge: records train={} eval={} contract={}",
        train_records.len(),
        eval_records.len(),
        train_inventory.contract_hash
    );
    println!(
        "train_knowledge: families={:?} languages={:?} stages={:?}",
        train_inventory.families, train_inventory.languages, train_inventory.stages
    );

    let foundation = args
        .foundation_bundle
        .as_deref()
        .map(verify_foundation_bundle)
        .transpose()?;
    let adaptation = args
        .adapt_from_bundle
        .as_deref()
        .map(verify_active_knowledge_bundle)
        .transpose()?;
    let steering_parent = args
        .steer_from_bundle
        .as_deref()
        .map(verify_candidate_knowledge_bundle)
        .transpose()?;
    let remediation_parent = args
        .remediate_from_bundle
        .as_deref()
        .map(verify_candidate_knowledge_bundle)
        .transpose()?;
    if let Some(parent) = &remediation_parent {
        if parent.artifact().certification.is_some() {
            return Err(invalid_data(
                "--remediate-from-bundle must be an uncertified candidate; a certified bundle \
                 belongs behind --adapt-from-bundle instead",
            ));
        }
        if args
            .remediate_from_bundle
            .as_deref()
            .expect("remediation_parent implies remediate_from_bundle")
            .join("active-model.json")
            .exists()
        {
            return Err(invalid_data(
                "--remediate-from-bundle cannot be a bundle that also carries an \
                 active-model.json",
            ));
        }
        if parent.artifact().checkpoint_optimizer_hash.is_none() {
            return Err(invalid_data(
                "--remediate-from-bundle candidate does not carry a content-addressed Adam \
                 state",
            ));
        }
    }

    let installed_contracts_path = args.artifacts_dir.join("contracts.json");
    if args.resume.is_some() {
        let installed: ToolContractSnapshot = serde_json::from_reader(BufReader::new(
            File::open(&installed_contracts_path).map_err(|error| {
                invalid_data(format!(
                    "resume requires original contracts {}: {error}",
                    installed_contracts_path.display()
                ))
            })?,
        ))?;
        installed.verify_integrity()?;
        if installed != contracts {
            return Err(invalid_data(
                "resume contract input differs from the installed immutable snapshot",
            ));
        }
    } else {
        write_json_pretty_atomic(&installed_contracts_path, &contracts)?;
    }
    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    if let Some(foundation) = &foundation {
        install_verified_tokenizer(
            foundation.tokenizer_path(),
            &foundation.artifact().tokenizer_hash,
            &tokenizer_path,
            "foundation",
        )?;
    } else if let Some(parent) = &adaptation {
        install_verified_tokenizer(
            parent.tokenizer_path(),
            &parent.artifact().tokenizer_hash,
            &tokenizer_path,
            "adaptation parent",
        )?;
    } else if let Some(parent) = &steering_parent {
        install_verified_tokenizer(
            parent.tokenizer_path(),
            &parent.artifact().tokenizer_hash,
            &tokenizer_path,
            "steering parent",
        )?;
    } else if let Some(parent) = &remediation_parent {
        install_verified_tokenizer(
            parent.tokenizer_path(),
            &parent.artifact().tokenizer_hash,
            &tokenizer_path,
            "remediation parent",
        )?;
    }
    let tokenizer = load_or_train_tokenizer(&args, &train_records, &tokenizer_path)?;
    let tokenizer_hash = fingerprint_artifact_file(&tokenizer_path)?;
    let mut model_config = args.model.config(tokenizer.vocab_size());
    let mut shape = KnowledgeModelShape::from_config(&model_config);
    // ADR 104 D3 (`--extend-context`): a foundation whose shape differs from
    // this run's target shape *only* in `max_seq_len`/Rope `scale` is not a
    // hard error -- it is the intended "train short, fine-tune long"
    // fallback (pretrain a Rope foundation at a shorter context, warm-start
    // the tool SFT at a longer one). `context_extension_scale` enforces that
    // every other architectural field still matches exactly.
    let mut context_extended_from: Option<usize> = None;
    let mut context_extension_scale: Option<f64> = None;
    if let Some(foundation) = &foundation {
        if foundation.artifact().tokenizer_hash != tokenizer_hash {
            return Err(invalid_data(
                "foundation bundle tokenizer does not match this knowledge run",
            ));
        }
        if foundation.artifact().model_size == args.model && foundation.artifact().shape == shape {
            // Exact match: unchanged pre-ADR-104-D3 behavior.
        } else if args.extend_context {
            let effective_scale = foundation
                .artifact()
                .shape
                .context_extension_scale(&shape)?;
            if let PositionalEncoding::Rope { theta, .. } = shape.positional_encoding {
                shape.positional_encoding = PositionalEncoding::Rope {
                    theta,
                    scale: effective_scale,
                };
            }
            context_extended_from = Some(foundation.artifact().shape.max_seq_len);
            context_extension_scale = Some(effective_scale);
            println!(
                "train_knowledge: context-extending foundation max_seq_len {} -> {} (rope scale {effective_scale})",
                foundation.artifact().shape.max_seq_len,
                shape.max_seq_len
            );
        } else {
            return Err(invalid_data(
                "foundation bundle model shape does not match this knowledge run (pass \
                 --extend-context to warm-start a longer-context SFT run from a shorter Rope \
                 foundation)",
            ));
        }
    }
    // Both ordinary resumes and steered segments retain the verified
    // candidate's effective Rope scale and context-extension lineage. The
    // canonical --model preset describes the architecture, not the scale
    // established by an earlier --extend-context foundation warm start.
    let resumed_candidate = args
        .resume
        .as_ref()
        .map(|_| verify_candidate_knowledge_bundle(&args.artifacts_dir))
        .transpose()?;
    if let Some(parent) = resumed_candidate
        .as_ref()
        .or(steering_parent.as_ref())
        .or(remediation_parent.as_ref())
    {
        shape = continued_candidate_shape(&shape, &parent.artifact().shape)?;
        context_extended_from = context_extended_from.or(parent.artifact().context_extended_from);
        model_config = shape.config();
    }
    if let Some(parent) = &adaptation
        && (parent.artifact().model_size != args.model
            || parent.artifact().shape != shape
            || parent.artifact().tokenizer_hash != tokenizer_hash)
    {
        return Err(invalid_data(
            "adaptation parent model shape/tokenizer does not match this knowledge run",
        ));
    }
    if let Some(parent) = &steering_parent
        && (parent.artifact().model_size != args.model
            || parent.artifact().shape != shape
            || parent.artifact().tokenizer_hash != tokenizer_hash
            || parent.artifact().contract_hash != contracts.manifest_hash
            || parent.artifact().certification.is_some())
    {
        return Err(invalid_data(
            "steering parent must be an uncertified candidate with the same model, tokenizer, and tool contract",
        ));
    }
    if let Some(parent) = &remediation_parent
        && (parent.artifact().model_size != args.model
            || parent.artifact().shape != shape
            || parent.artifact().tokenizer_hash != tokenizer_hash
            || parent.artifact().contract_hash != contracts.manifest_hash)
    {
        return Err(invalid_data(
            "remediation parent must have the same model, tokenizer, and tool contract as this \
             knowledge run",
        ));
    }
    if args.eval_max_new_tokens + 3 >= model_config.max_seq_len {
        return Err(invalid_data(format!(
            "--eval-max-new-tokens={} leaves no useful prompt budget in context {}",
            args.eval_max_new_tokens, model_config.max_seq_len
        )));
    }
    report_sequence_lengths(&tokenizer, &train_records, model_config.max_seq_len)?;
    let direct_steering_lineage = steering_parent
        .as_ref()
        .map(|parent| {
            verify_steering_inputs(
                &args,
                parent,
                &contracts,
                &steering_records,
                base_training_records,
                steering_share_ppm.expect("steering records have a validated share"),
            )
        })
        .transpose()?;
    if args.validate_only {
        println!(
            "train_knowledge: validation complete; corpus={} eval={} tokenizer={} shape={:?}",
            train_inventory.fingerprint,
            eval_inventory
                .as_ref()
                .map(|inventory| inventory.fingerprint.as_str())
                .unwrap_or("none"),
            tokenizer_hash,
            shape
        );
        return Ok(());
    }

    B::seed(&device, args.seed);
    let (mut model, mut optimizer) = init_training::<B>(&model_config, &device);
    let parameter_count = model.num_params();
    let learning_rate = args.learning_rate.unwrap_or(match args.model {
        KnowledgeModelSize::Small
        | KnowledgeModelSize::Small8k
        | KnowledgeModelSize::SmallRope4k => 3e-4,
        KnowledgeModelSize::Model350m | KnowledgeModelSize::Model350m8k => 1e-4,
    });
    println!(
        "train_knowledge: model={} params={} vocab={} context={} lr={learning_rate}",
        args.model,
        parameter_count,
        tokenizer.vocab_size(),
        model_config.max_seq_len
    );
    let direct_remediation_lineage = remediation_parent
        .as_ref()
        .map(|parent| {
            println!(
                "train_knowledge: remediating candidate step {} model={} with retained Adam \
                 state, effective lr={learning_rate} warmup_steps={}",
                parent.artifact().checkpoint_step,
                parent.artifact().checkpoint_model_hash,
                args.warmup_steps
            );
            build_remediation_lineage(
                &args,
                parent,
                &supplemental_file_records,
                base_only_training_records,
                supplemental_total_records,
                remediation_share_ppm.expect("remediation records have a validated share"),
                learning_rate,
            )
        })
        .transpose()?;

    let (resume_artifact, resume_schedule_changes) = match verify_resume_manifest(
        &args,
        &train_inventory,
        eval_inventory.as_ref(),
        train_records.len(),
        eval_records.len(),
        &shape,
        parameter_count,
        tokenizer.vocab_size(),
        learning_rate,
    )? {
        Some((artifact, schedule_changes)) => (Some(artifact), schedule_changes),
        None => (None, Vec::new()),
    };
    let foundation_lineage = foundation
        .as_ref()
        .map(VerifiedFoundationBundle::lineage)
        .or_else(|| {
            adaptation
                .as_ref()
                .and_then(|parent| parent.artifact().foundation.clone())
        })
        .or_else(|| {
            steering_parent
                .as_ref()
                .and_then(|parent| parent.artifact().foundation.clone())
        })
        .or_else(|| {
            resume_artifact
                .as_ref()
                .and_then(|artifact| artifact.foundation.clone())
        });
    let adaptation_lineage = adaptation
        .as_ref()
        .map(VerifiedKnowledgeBundle::adaptation_lineage)
        .transpose()?
        .or_else(|| {
            resume_artifact
                .as_ref()
                .and_then(|artifact| artifact.adapted_from.clone())
        });
    let steering_lineage = direct_steering_lineage.or_else(|| {
        resume_artifact
            .as_ref()
            .and_then(|artifact| artifact.steered_from.clone())
    });
    let remediation_lineage = direct_remediation_lineage.or_else(|| {
        resume_artifact
            .as_ref()
            .and_then(|artifact| artifact.remediated_from.clone())
    });
    let mut start_step = 0usize;
    if let Some(resume) = &args.resume {
        let (loaded_model, loaded_optimizer, step) =
            checkpoint::load_checkpoint(resume, model, optimizer, &device)?;
        model = loaded_model;
        optimizer = loaded_optimizer;
        start_step = step + 1;
        println!(
            "train_knowledge: resumed {} at step {start_step}",
            resume.display()
        );
    } else if let Some(parent) = &steering_parent {
        let (loaded_model, loaded_optimizer, step) =
            checkpoint::load_checkpoint(parent.checkpoint_dir(), model, optimizer, &device)?;
        if step != parent.artifact().checkpoint_step {
            return Err(invalid_data(format!(
                "steering parent checkpoint metadata says {step}, artifact says {}",
                parent.artifact().checkpoint_step
            )));
        }
        model = loaded_model;
        optimizer = loaded_optimizer;
        start_step = step
            .checked_add(1)
            .ok_or_else(|| invalid_data("steering checkpoint step overflow"))?;
        println!(
            "train_knowledge: steering from candidate step {step} model={} with retained Adam state",
            parent.artifact().checkpoint_model_hash
        );
    } else if let Some(foundation) = &foundation {
        let loaded = foundation.load_model_only::<B>(&device)?;
        model = match context_extension_scale {
            Some(effective_scale) => {
                // The foundation's own shape (unmodified) always loads
                // cleanly -- `load_model_only` is byte-for-byte the same
                // call as the non-extended path. Re-serving at the SFT
                // target's longer `max_seq_len` is then exactly
                // `TinyTransformer::with_serving_length`'s job: it only
                // rebuilds each layer's non-learned rotary-frequency table,
                // never a `Param` tensor (see
                // `transformer::tests::with_serving_length_leaves_every_learned_weight_bitwise_identical_under_ntk_rescale`).
                let extended =
                    loaded.with_serving_length(shape.max_seq_len, effective_scale, &device)?;
                println!(
                    "train_knowledge: warm-started foundation step {} model={} corpus={} context-extended {} -> {} (rope scale {effective_scale})",
                    foundation.artifact().checkpoint_step,
                    foundation.artifact().checkpoint_model_hash,
                    foundation.artifact().training_corpus_hash,
                    foundation.artifact().shape.max_seq_len,
                    shape.max_seq_len,
                );
                extended
            }
            None => {
                println!(
                    "train_knowledge: warm-started foundation step {} model={} corpus={}",
                    foundation.artifact().checkpoint_step,
                    foundation.artifact().checkpoint_model_hash,
                    foundation.artifact().training_corpus_hash
                );
                loaded
            }
        };
    } else if let Some(parent) = &adaptation {
        model = parent.load_model_only::<B>(&device)?;
        println!(
            "train_knowledge: adapting certified active step {} model={} contract={} with fresh Adam state",
            parent.artifact().checkpoint_step,
            parent.artifact().checkpoint_model_hash,
            parent.artifact().contract_hash
        );
    } else if let Some(parent) = &remediation_parent {
        let (loaded_model, loaded_optimizer, step) =
            checkpoint::load_checkpoint(parent.checkpoint_dir(), model, optimizer, &device)?;
        if step != parent.artifact().checkpoint_step {
            return Err(invalid_data(format!(
                "remediation parent checkpoint metadata says {step}, artifact says {}",
                parent.artifact().checkpoint_step
            )));
        }
        model = loaded_model;
        optimizer = loaded_optimizer;
        start_step = step
            .checked_add(1)
            .ok_or_else(|| invalid_data("remediation checkpoint step overflow"))?;
        println!(
            "train_knowledge: remediating from candidate step {step} model={} with retained \
             Adam state",
            parent.artifact().checkpoint_model_hash
        );
    }
    let segment_start_step = if steering_parent.is_some() || remediation_parent.is_some() {
        start_step
    } else if args.resume.is_some() {
        read_segment_start_step(&args.artifacts_dir.join("run.json"))?
    } else {
        0
    };

    let run = RunDescriptor {
        schema_version: 1,
        seed: args.seed,
        model_size: args.model,
        shape: shape.clone(),
        parameter_count,
        contract_hash: train_inventory.contract_hash.clone(),
        training_corpus_hash: train_inventory.fingerprint.clone(),
        eval_corpus_hash: eval_inventory
            .as_ref()
            .map(|value| value.fingerprint.clone()),
        training_records: train_records.len(),
        eval_records: eval_records.len(),
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate,
        warmup_steps: args.warmup_steps,
        eval_every: args.eval_every,
        eval_samples: args.eval_samples,
        eval_max_new_tokens: args.eval_max_new_tokens,
        ckpt_every: args.ckpt_every,
        loss_cap: args.loss_cap,
        grad_norm_cap: args.grad_norm_cap,
        tokenizer_vocab_size: tokenizer.vocab_size(),
        train_jsonl: args.train_jsonl.clone(),
        supplemental_jsonl: args.supplemental_jsonl.clone(),
        eval_jsonl: args.eval_jsonl.clone(),
        supplemental_eval_jsonl: args.supplemental_eval_jsonl.clone(),
        foundation_bundle: args.foundation_bundle.clone(),
        foundation: foundation_lineage.clone(),
        adapt_from_bundle: args.adapt_from_bundle.clone(),
        adapted_from: adaptation_lineage.clone(),
        steer_from_bundle: args.steer_from_bundle.clone(),
        steering_decision: args.steering_decision.clone(),
        steering_frontier_exam: args.steering_frontier_exam.clone(),
        steering_sibling_manifest: args.steering_sibling_manifest.clone(),
        steering_sibling_jsonl: args.steering_sibling_jsonl.clone(),
        steered_from: steering_lineage.clone(),
        remediate_from_bundle: args.remediate_from_bundle.clone(),
        remediated_from: remediation_lineage.clone(),
        start_step: segment_start_step,
        schedule_changes: resume_schedule_changes,
    };
    if let Some(artifact) = &resume_artifact {
        preserve_resume_run_descriptor(&args.artifacts_dir, artifact.checkpoint_step)?;
    }
    write_json_pretty_atomic(&args.artifacts_dir.join("run.json"), &run)?;

    let eval_subset = evenly_spaced(&eval_records, args.eval_samples);
    let steps_per_epoch = train_records.len().div_ceil(args.batch_size);
    let segment_steps = args
        .epochs
        .checked_mul(steps_per_epoch)
        .ok_or_else(|| invalid_data("training step count overflow"))?;
    let total_steps = segment_start_step
        .checked_add(segment_steps)
        .ok_or_else(|| invalid_data("training step count overflow"))?;
    if start_step >= total_steps {
        return Err(invalid_data(format!(
            "resume starts at step {start_step}, but this run only has {total_steps} steps; increase --epochs"
        )));
    }
    println!(
        "train_knowledge: segment_start={segment_start_step} end_exclusive={total_steps} ({} epochs x {steps_per_epoch}), eval_samples={}",
        args.epochs,
        eval_subset.len()
    );

    let train_log = args.artifacts_dir.join("train.jsonl");
    let eval_log = args.artifacts_dir.join("eval_teacher_forced.jsonl");
    let started = Instant::now();
    let mut global_step = segment_start_step;
    let mut skipped_updates = 0usize;
    let mut final_loss = f32::NAN;

    for epoch in 0..args.epochs {
        let mut order: Vec<usize> = (0..train_records.len()).collect();
        let mut rng = StdRng::seed_from_u64(args.seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9));
        order.shuffle(&mut rng);

        for indices in order.chunks(args.batch_size) {
            let step = global_step;
            global_step += 1;
            if step < start_step {
                continue;
            }
            let records: Vec<_> = indices
                .iter()
                .map(|index| train_records[*index].clone())
                .collect();
            let batch = build_knowledge_batch::<B, _>(
                &tokenizer,
                &records,
                model_config.max_seq_len,
                &device,
            )?;
            let effective_lr = warmup_learning_rate(learning_rate, args.warmup_steps, step);
            let step_started = Instant::now();
            let (next_model, loss, grad_norm) = train_step_guarded(
                model,
                &mut optimizer,
                &batch,
                effective_lr,
                args.loss_cap,
                args.grad_norm_cap,
            );
            model = next_model;
            final_loss = loss;
            let skipped = !loss.is_finite()
                || loss > args.loss_cap
                || !grad_norm.is_finite()
                || grad_norm > args.grad_norm_cap;
            skipped_updates += usize::from(skipped);
            let row = TrainLogRow {
                step,
                epoch,
                loss,
                grad_norm,
                learning_rate: effective_lr,
                update_skipped: skipped,
                step_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
            };
            append_json_line(&train_log, &row)?;
            if step % 10 == 0 || step + 1 == total_steps || skipped {
                println!(
                    "step {step:6} epoch {epoch:3}: loss={loss:.4} grad={grad_norm:.3} lr={effective_lr:.2e}{}",
                    if skipped { " SKIPPED" } else { "" }
                );
            }
            if !loss.is_finite() {
                return Err(invalid_data(format!(
                    "non-finite loss at step {step}; update was blocked and candidate-model.json was not advanced"
                )));
            }

            if !eval_subset.is_empty()
                && ((step + 1) % args.eval_every == 0 || step + 1 == total_steps)
            {
                evaluate_and_log(
                    &model,
                    &tokenizer,
                    &eval_subset,
                    &train_inventory.contract_hash,
                    step,
                    model_config.max_seq_len,
                    args.eval_max_new_tokens,
                    &device,
                    &args.artifacts_dir,
                    &eval_log,
                )?;
            }

            if (step + 1) % args.ckpt_every == 0 || step + 1 == total_steps {
                let relative_checkpoint = format!("checkpoints/step_{step:08}");
                let checkpoint_dir = args.artifacts_dir.join(&relative_checkpoint);
                checkpoint::save_checkpoint(&checkpoint_dir, step, &model, &optimizer)?;
                let artifact = KnowledgeModelArtifact {
                    schema_version: KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION,
                    model_size: args.model,
                    shape: shape.clone(),
                    parameter_count,
                    tokenizer_file: "tokenizer.json".into(),
                    tokenizer_hash: tokenizer_hash.clone(),
                    contracts_file: "contracts.json".into(),
                    checkpoint_dir: relative_checkpoint,
                    checkpoint_step: step,
                    checkpoint_model_hash: fingerprint_artifact_file(
                        checkpoint_dir.join("model.mpk"),
                    )?,
                    checkpoint_optimizer_hash: Some(fingerprint_artifact_file(
                        checkpoint_dir.join("optim.mpk"),
                    )?),
                    contract_hash: train_inventory.contract_hash.clone(),
                    training_corpus_hash: train_inventory.fingerprint.clone(),
                    training_records: train_records.len(),
                    eval_corpus_hash: eval_inventory
                        .as_ref()
                        .map(|value| value.fingerprint.clone()),
                    capability_families: train_inventory.families.clone(),
                    languages: train_inventory.languages.clone(),
                    stages: train_inventory.stages.clone(),
                    foundation: foundation_lineage.clone(),
                    context_extended_from,
                    adapted_from: adaptation_lineage.clone(),
                    steered_from: steering_lineage.clone(),
                    remediated_from: remediation_lineage.clone(),
                    certification: None,
                };
                artifact.verify().map_err(invalid_data)?;
                write_json_pretty_atomic(
                    &args.artifacts_dir.join("candidate-model.json"),
                    &artifact,
                )?;
                println!(
                    "  checkpoint + candidate model -> {}",
                    checkpoint_dir.display()
                );
            }
        }
    }

    println!(
        "train_knowledge: done steps={} elapsed={:.1}s final_loss={final_loss:.4} skipped_updates={skipped_updates}",
        total_steps,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.allow_schedule_change && args.resume.is_none() {
        return Err(invalid_data(
            "--allow-schedule-change only applies to a --resume run",
        ));
    }
    let steering_inputs = [
        args.steer_from_bundle.is_some(),
        args.steering_decision.is_some(),
        args.steering_frontier_exam.is_some(),
        args.steering_sibling_manifest.is_some(),
        args.steering_sibling_jsonl.is_some(),
    ];
    let steering_count = steering_inputs.into_iter().filter(|value| *value).count();
    if steering_count != 0 && steering_count != steering_inputs.len() {
        return Err(invalid_data(
            "candidate steering requires --steer-from-bundle, --steering-decision, --steering-frontier-exam, --steering-sibling-manifest, and --steering-sibling-jsonl together",
        ));
    }
    let initialization_sources = usize::from(args.resume.is_some())
        + usize::from(args.foundation_bundle.is_some())
        + usize::from(args.adapt_from_bundle.is_some())
        + usize::from(args.steer_from_bundle.is_some())
        + usize::from(args.remediate_from_bundle.is_some());
    if initialization_sources > 1 {
        return Err(invalid_data(
            "--resume, --foundation-bundle, --adapt-from-bundle, --steer-from-bundle, and \
             --remediate-from-bundle are mutually exclusive",
        ));
    }
    if args.remediate_from_bundle.is_some() && args.supplemental_jsonl.is_empty() {
        return Err(invalid_data(
            "--remediate-from-bundle requires at least one --supplemental-jsonl engine-generated \
             sibling corpus",
        ));
    }
    if args.remediate_from_bundle.is_none()
        && args.remediation_max_share_ppm != grug_train::DEFAULT_REMEDIATION_MAX_SHARE_PPM
    {
        return Err(invalid_data(
            "--remediation-max-share-ppm only applies to a --remediate-from-bundle run",
        ));
    }
    if args.remediate_from_bundle.is_some()
        && (args.remediation_max_share_ppm == 0 || args.remediation_max_share_ppm > 1_000_000)
    {
        return Err(invalid_data(format!(
            "--remediation-max-share-ppm must be in 1..=1_000_000, got {}",
            args.remediation_max_share_ppm
        )));
    }
    if args.remediate_from_bundle.is_some() && args.learning_rate.is_none() {
        return Err(invalid_data(
            "--remediate-from-bundle requires an explicit --learning-rate: a continued Adam \
             state should never silently pick up the small/350m default schedule",
        ));
    }
    if args.epochs == 0 || args.batch_size == 0 {
        return Err(invalid_data("--epochs and --batch-size must be positive"));
    }
    if args.bpe_vocab_size < 260 {
        return Err(invalid_data("--bpe-vocab-size must be at least 260"));
    }
    if args.eval_every == 0 || args.ckpt_every == 0 {
        return Err(invalid_data(
            "--eval-every and --ckpt-every must be positive",
        ));
    }
    if args.eval_samples == 0
        && (args.eval_jsonl.is_some() || !args.supplemental_eval_jsonl.is_empty())
    {
        return Err(invalid_data(
            "--eval-samples must be positive when --eval-jsonl is present",
        ));
    }
    if args.learning_rate.is_some_and(|value| value <= 0.0) {
        return Err(invalid_data("--learning-rate must be positive"));
    }
    if !args.loss_cap.is_finite()
        || args.loss_cap <= 0.0
        || !args.grad_norm_cap.is_finite()
        || args.grad_norm_cap <= 0.0
    {
        return Err(invalid_data(
            "--loss-cap and --grad-norm-cap must be finite and positive",
        ));
    }
    Ok(())
}

fn load_records(path: &Path) -> Result<Vec<KnowledgeSftRecord>, Box<dyn Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut records = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(&line).map_err(|error| {
            invalid_data(format!(
                "{}:{} is not a KnowledgeSftRecord: {error}",
                path.display(),
                line_index + 1
            ))
        })?;
        records.push(record);
    }
    if records.is_empty() {
        return Err(invalid_data(format!(
            "{} contains no SFT records",
            path.display()
        )));
    }
    Ok(records)
}

fn validate_records(
    label: &str,
    records: &[KnowledgeSftRecord],
    contracts: &ToolContractSnapshot,
) -> Result<CorpusInventory, Box<dyn Error>> {
    let mut families = BTreeSet::new();
    let mut languages = BTreeSet::new();
    let mut stages = BTreeSet::new();
    let mut record_ids = HashSet::new();
    let mut episode_ids = HashSet::new();

    for record in records {
        if record.schema_version != KNOWLEDGE_WORKFLOW_SCHEMA_VERSION {
            return Err(invalid_data(format!(
                "{label} record {} uses schema {}, expected {}",
                record.id, record.schema_version, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION
            )));
        }
        if record.contract_hash != contracts.manifest_hash {
            return Err(invalid_data(format!(
                "{label} record {} uses contract {}, loaded snapshot is {}",
                record.id, record.contract_hash, contracts.manifest_hash
            )));
        }
        if !record_ids.insert(record.id.clone()) {
            return Err(invalid_data(format!(
                "duplicate {label} record id {}",
                record.id
            )));
        }
        let rendered = render_knowledge_prompt_with_system(
            &record.tools,
            &record.conversation,
            record.system_context.as_deref(),
        );
        if rendered != record.prompt {
            return Err(invalid_data(format!(
                "{label} record {} prompt does not match the canonical runtime renderer",
                record.id
            )));
        }
        let expected = KnowledgeAction::parse_runtime(&record.target).map_err(|error| {
            invalid_data(format!(
                "{label} record {} has an invalid runtime target: {error}",
                record.id
            ))
        })?;
        if expected.render_runtime()?.trim() != record.target.trim() {
            return Err(invalid_data(format!(
                "{label} record {} target is parseable but non-canonical",
                record.id
            )));
        }
        if let KnowledgeAction::ToolCall { tool, .. } = &expected {
            let offered = record.tools.iter().any(|definition| {
                definition
                    .pointer("/function/name")
                    .and_then(serde_json::Value::as_str)
                    == Some(tool.as_str())
            });
            if !offered {
                return Err(invalid_data(format!(
                    "{label} record {} calls {tool:?}, which is not in its offered tools",
                    record.id
                )));
            }
        }
        families.insert(record.family.clone());
        languages.insert(json_label(record.language));
        stages.insert(json_label(record.stage));
        episode_ids.insert(record.episode_id.clone());
    }

    Ok(CorpusInventory {
        contract_hash: contracts.manifest_hash.clone(),
        fingerprint: fingerprint_knowledge_records(records)?,
        families,
        languages,
        stages,
        record_ids,
        episode_ids,
    })
}

fn reject_holdout_leakage(
    train: &CorpusInventory,
    eval: &CorpusInventory,
) -> Result<(), Box<dyn Error>> {
    if let Some(id) = train.record_ids.intersection(&eval.record_ids).next() {
        return Err(invalid_data(format!(
            "evaluation leakage: record {id:?} is also in training"
        )));
    }
    if let Some(id) = train.episode_ids.intersection(&eval.episode_ids).next() {
        return Err(invalid_data(format!(
            "evaluation leakage: episode {id:?} is also in training"
        )));
    }
    Ok(())
}

fn load_or_train_tokenizer(
    args: &Args,
    records: &[KnowledgeSftRecord],
    path: &Path,
) -> Result<BpeTokenizer, Box<dyn Error>> {
    if path.exists() {
        println!("train_knowledge: loading tokenizer {}", path.display());
        return Ok(BpeTokenizer::load(path)?);
    }
    if args.resume.is_some() {
        return Err(invalid_data(format!(
            "cannot resume without the original tokenizer at {}",
            path.display()
        )));
    }
    println!(
        "train_knowledge: training BPE tokenizer (ceiling {})",
        args.bpe_vocab_size
    );
    let corpus = records
        .iter()
        .flat_map(|record| [record.prompt.as_str(), record.target.as_str()]);
    let tokenizer = BpeTokenizer::train(corpus, args.bpe_vocab_size)?;
    tokenizer.save(path)?;
    Ok(tokenizer)
}

fn report_sequence_lengths(
    tokenizer: &BpeTokenizer,
    records: &[KnowledgeSftRecord],
    context: usize,
) -> Result<(), Box<dyn Error>> {
    let mut lengths: Vec<usize> = records
        .iter()
        .map(|record| {
            grug_train::encode_knowledge_record(tokenizer, record).map(|encoded| encoded.ids.len())
        })
        .collect::<Result<_, _>>()?;
    lengths.sort_unstable();
    let percentile = |numerator: usize| {
        lengths
            .get((lengths.len().saturating_sub(1) * numerator) / 100)
            .copied()
            .unwrap_or(0)
    };
    let over = lengths.iter().filter(|length| **length > context).count();
    println!(
        "train_knowledge: token lengths p50={} p95={} max={} context={} compacted={over}",
        percentile(50),
        percentile(95),
        lengths.last().copied().unwrap_or(0),
        context
    );
    Ok(())
}

fn verify_steering_inputs(
    args: &Args,
    parent: &VerifiedKnowledgeBundle,
    contracts: &ToolContractSnapshot,
    sibling_records: &[KnowledgeSftRecord],
    base_training_records: usize,
    sibling_share_ppm: u64,
) -> Result<KnowledgeSteeringLineage, Box<dyn Error>> {
    let decision_path = args
        .steering_decision
        .as_deref()
        .ok_or_else(|| invalid_data("missing steering decision"))?;
    let frontier_path = args
        .steering_frontier_exam
        .as_deref()
        .ok_or_else(|| invalid_data("missing steering frontier exam"))?;
    let manifest_path = args
        .steering_sibling_manifest
        .as_deref()
        .ok_or_else(|| invalid_data("missing steering sibling manifest"))?;
    let sibling_path = args
        .steering_sibling_jsonl
        .as_deref()
        .ok_or_else(|| invalid_data("missing steering sibling SFT"))?;

    let step: ControllerStepArtifact =
        serde_json::from_reader(BufReader::new(File::open(decision_path)?))?;
    if step.controller.schema_version != ADAPTIVE_EXAM_SCHEMA_VERSION
        || step.next_plan.schema_version != ADAPTIVE_EXAM_SCHEMA_VERSION
        || step.next_plan.tier != ExamTier::Frontier
    {
        return Err(invalid_data(
            "steering requires a supported controller and a frontier next_plan",
        ));
    }
    let parent_step = u64::try_from(parent.artifact().checkpoint_step)
        .map_err(|_| invalid_data("parent checkpoint step does not fit u64"))?;
    if step.next_plan.checkpoint.step != parent_step
        || step.next_plan.checkpoint.artifact_hash != parent.artifact().checkpoint_model_hash
    {
        return Err(invalid_data(
            "controller frontier plan is bound to a different parent checkpoint",
        ));
    }
    let replayed_plan = step.controller.next_plan(
        step.next_plan.tier,
        step.next_plan.checkpoint.clone(),
        step.next_plan.sample_count,
    )?;
    if replayed_plan != step.next_plan {
        return Err(invalid_data(
            "controller state does not reproduce its declared next frontier plan",
        ));
    }
    if step
        .decisions
        .iter()
        .any(|decision| decision.warning == WarningLevel::Stop)
    {
        return Err(invalid_data(
            "controller emitted stop; a new training segment requires operator-authored recovery, not automatic steering",
        ));
    }
    let selected: Vec<_> = step
        .decisions
        .iter()
        .filter(|decision| {
            decision.family == step.next_plan.family && decision.tier == ExamTier::Frontier
        })
        .collect();
    if selected.len() != 1
        || selected[0].warning != WarningLevel::Steer
        || !selected[0].statistically_eligible
    {
        return Err(invalid_data(
            "selected frontier family lacks one statistically eligible steer decision",
        ));
    }

    let sibling: TrainingSiblingManifest =
        serde_json::from_reader(BufReader::new(File::open(manifest_path)?))?;
    if !sibling.complete
        || sibling.contract_hash != contracts.manifest_hash
        || sibling.family != step.next_plan.family
        || sibling.records != sibling_records.len()
        || sibling.source_exam_fingerprint.trim().is_empty()
        || !sibling.source_exam_file_hash.starts_with("blake3:")
        || !sibling.separation_hash.starts_with("blake3:")
        || !sibling.sft_hash.starts_with("blake3:")
    {
        return Err(invalid_data(
            "training sibling manifest is incomplete or disagrees with the controller/current contract",
        ));
    }
    let sibling_root = manifest_path
        .parent()
        .ok_or_else(|| invalid_data("sibling manifest has no parent directory"))?;
    if sibling_root.join("train.sft.jsonl").canonicalize()? != sibling_path.canonicalize()? {
        return Err(invalid_data(
            "--steering-sibling-jsonl is not the train.sft.jsonl beside its manifest",
        ));
    }
    let actual_sft_hash = fingerprint_artifact_file(sibling_path)?;
    if actual_sft_hash != sibling.sft_hash {
        return Err(invalid_data(format!(
            "steering sibling SFT hash is {actual_sft_hash}, manifest declares {}",
            sibling.sft_hash
        )));
    }
    let separation_path = sibling_root.join("separation.json");
    let actual_separation_hash = fingerprint_artifact_file(&separation_path)?;
    if actual_separation_hash != sibling.separation_hash {
        return Err(invalid_data(
            "steering sibling separation evidence hash differs from its manifest",
        ));
    }
    let separation: SiblingSeparationEvidence =
        serde_json::from_reader(BufReader::new(File::open(&separation_path)?))?;
    let actual_frontier_hash = fingerprint_artifact_file(frontier_path)?;
    if actual_frontier_hash != sibling.source_exam_file_hash {
        return Err(invalid_data(
            "steering frontier exam hash differs from the sibling manifest",
        ));
    }

    let regenerated_records = if step.next_plan.family.starts_with("knowledge.") {
        let exam: KnowledgeExamSet =
            serde_json::from_reader(BufReader::new(File::open(frontier_path)?))?;
        if exam.contract_hash != contracts.manifest_hash
            || exam.plan != step.next_plan
            || exam.exam_fingerprint != exam.plan.fingerprint()
            || exam.exam_fingerprint != sibling.source_exam_fingerprint
        {
            return Err(invalid_data(
                "knowledge frontier exam does not match the selected controller plan/sibling",
            ));
        }
        let bindings = KnowledgeToolBindings::default();
        let generator = KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone())?;
        if generator.generate_exam(&exam.plan)? != exam {
            return Err(invalid_data(
                "knowledge frontier exam does not replay from its declared plan",
            ));
        }
        let episode_count = sibling
            .episodes
            .ok_or_else(|| invalid_data("knowledge sibling manifest has no episode count"))?;
        let sibling_plan = derive_training_sibling(
            &exam.plan,
            u64::try_from(episode_count)
                .map_err(|_| invalid_data("knowledge sibling episode count does not fit u64"))?,
            SiblingSeparationPolicy {
                require_image_distance: false,
                ..SiblingSeparationPolicy::default()
            },
        )?;
        sibling_plan.validate_separation(&separation)?;
        let corpus = generator.generate_training_sibling(&sibling_plan)?;
        let mut records = Vec::new();
        for episode in corpus.episodes {
            records.extend(episode.to_sft_records(contracts, &bindings)?);
        }
        records
    } else if step.next_plan.family.starts_with("tool_contract.") {
        let exam: ToolContractExamSet =
            serde_json::from_reader(BufReader::new(File::open(frontier_path)?))?;
        exam.verify(contracts)?;
        let selected_tool = step
            .next_plan
            .family
            .strip_prefix("tool_contract.")
            .ok_or_else(|| invalid_data("selected tool family has no tool_contract. prefix"))?;
        let expected_scope = BTreeSet::from([selected_tool.to_string()]);
        if !exam.target_tools.is_empty() && exam.target_tools != expected_scope {
            return Err(invalid_data(format!(
                "tool frontier target scope {:?} does not match selected family {:?}",
                exam.target_tools, step.next_plan.family
            )));
        }
        if exam.run_id != step.next_plan.run_id
            || exam.checkpoint != step.next_plan.checkpoint
            || exam.tier != step.next_plan.tier
            || exam.difficulty != step.next_plan.difficulty
            || exam.seed != step.next_plan.seed
            || exam.samples_per_tool != step.next_plan.sample_count
            || exam.exam_fingerprint != sibling.source_exam_fingerprint
        {
            return Err(invalid_data(
                "tool frontier exam does not match the selected controller plan/sibling",
            ));
        }
        let sibling_plan = derive_tool_contract_training_sibling(
            &exam,
            contracts,
            &step.next_plan.family,
            u64::try_from(sibling.records)
                .map_err(|_| invalid_data("tool sibling record count does not fit u64"))?,
        )?;
        sibling_plan.validate_separation(&separation)?;
        sibling_plan.generate(contracts)?
    } else {
        return Err(invalid_data(format!(
            "unsupported adaptive training family {:?}",
            step.next_plan.family
        )));
    };
    if regenerated_records != sibling_records {
        return Err(invalid_data(
            "steering sibling SFT does not replay from its frontier/plan",
        ));
    }
    if sibling_records
        .iter()
        .any(|record| record.family != step.next_plan.family)
    {
        return Err(invalid_data(
            "steering sibling contains records outside the selected family",
        ));
    }

    let foundation =
        parent.artifact().foundation.as_ref().ok_or_else(|| {
            invalid_data("steering parent has no verified causal-foundation lineage")
        })?;
    let lineage = KnowledgeSteeringLineage {
        artifact_hash: parent.artifact_hash().into(),
        checkpoint_model_hash: parent.artifact().checkpoint_model_hash.clone(),
        checkpoint_step: parent.artifact().checkpoint_step,
        contract_hash: parent.artifact().contract_hash.clone(),
        training_corpus_hash: parent.artifact().training_corpus_hash.clone(),
        tokenizer_hash: parent.artifact().tokenizer_hash.clone(),
        model_size: parent.artifact().model_size,
        shape: parent.artifact().shape.clone(),
        parameter_count: parent.artifact().parameter_count,
        foundation_artifact_hash: foundation.artifact_hash.clone(),
        controller_step_hash: fingerprint_artifact_file(decision_path)?,
        frontier_plan_fingerprint: step.next_plan.fingerprint(),
        family: step.next_plan.family,
        source_exam_fingerprint: sibling.source_exam_fingerprint,
        sibling_manifest_hash: fingerprint_artifact_file(manifest_path)?,
        sibling_sft_hash: sibling.sft_hash,
        base_training_records,
        sibling_training_records: sibling_records.len(),
        sibling_share_ppm,
    };
    lineage.verify().map_err(invalid_data)?;
    Ok(lineage)
}

/// Build the P6.4 remediation lineage for `--remediate-from-bundle`: the
/// parent's checkpoint identity plus one content-addressed entry per
/// `--supplemental-jsonl` file, each carrying whatever `generator_git_hash`/
/// `seed` a sibling `manifest.json` next to it happens to record (see
/// [`read_sibling_generator_metadata`]).
#[allow(clippy::too_many_arguments)]
fn build_remediation_lineage(
    args: &Args,
    parent: &VerifiedKnowledgeBundle,
    supplemental_file_records: &[(PathBuf, Vec<KnowledgeSftRecord>)],
    base_training_records: usize,
    supplemental_training_records: usize,
    supplemental_share_ppm: u64,
    learning_rate: f64,
) -> Result<RemediationLineage, Box<dyn Error>> {
    let parent_bundle_path = args
        .remediate_from_bundle
        .as_deref()
        .ok_or_else(|| invalid_data("remediation lineage requires --remediate-from-bundle"))?;
    let parent_checkpoint_optimizer_hash = parent
        .artifact()
        .checkpoint_optimizer_hash
        .clone()
        .ok_or_else(|| invalid_data("remediation parent has no content-addressed Adam state"))?;
    let mut supplemental = Vec::with_capacity(supplemental_file_records.len());
    for (path, records) in supplemental_file_records {
        let sft_hash = fingerprint_artifact_file(path)?;
        let (generator_git_hash, seed) = read_sibling_generator_metadata(path);
        supplemental.push(RemediationSupplement {
            path: path.display().to_string(),
            sft_hash,
            records: records.len(),
            generator_git_hash,
            seed,
        });
    }
    let lineage = RemediationLineage {
        parent_bundle: parent_bundle_path.display().to_string(),
        parent_checkpoint_step: parent.artifact().checkpoint_step,
        parent_checkpoint_model_hash: parent.artifact().checkpoint_model_hash.clone(),
        parent_checkpoint_optimizer_hash,
        contract_hash: parent.artifact().contract_hash.clone(),
        tokenizer_hash: parent.artifact().tokenizer_hash.clone(),
        supplemental,
        base_training_records,
        supplemental_training_records,
        supplemental_share_ppm,
        max_share_ppm: args.remediation_max_share_ppm,
        learning_rate,
        warmup_steps: args.warmup_steps,
        epochs: args.epochs,
    };
    lineage.verify().map_err(invalid_data)?;
    Ok(lineage)
}

fn read_segment_start_step(path: &Path) -> Result<usize, Box<dyn Error>> {
    let value: serde_json::Value =
        serde_json::from_reader(BufReader::new(File::open(path).map_err(|error| {
            invalid_data(format!(
                "resume requires original run descriptor {}: {error}",
                path.display()
            ))
        })?))?;
    value
        .get("start_step")
        .and_then(serde_json::Value::as_u64)
        .map(usize::try_from)
        .transpose()
        .map_err(|_| invalid_data("run start_step does not fit usize"))
        .map(|value| value.unwrap_or(0))
}

/// A verified `--resume` artifact paired with the accumulated
/// `--allow-schedule-change` history to carry forward into the new run.json.
type ResumeVerification = (KnowledgeModelArtifact, Vec<ScheduleChange>);

/// Whether a resumed run.json differs from this invocation along the one
/// sanctioned escape hatch (`learning_rate`/`warmup_steps`, ADR 104 D5) and/or
/// along every other invariant that must never change on resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResumeComparison {
    schedule_changed: bool,
    other_mismatch: bool,
}

impl ResumeComparison {
    /// Whether `verify_resume_manifest` must reject this resume outright.
    /// `other_mismatch` is fatal unconditionally; `schedule_changed` is
    /// fatal unless the caller explicitly opted in via
    /// `--allow-schedule-change` (ADR 104 D5).
    fn rejects(self, allow_schedule_change: bool) -> bool {
        self.other_mismatch || (self.schedule_changed && !allow_schedule_change)
    }
}

/// Exact, stable text of the resume-invariant violation error so the
/// production error path and its tests cannot drift apart.
const RESUME_SCHEDULE_INVARIANT_ERROR: &str = "resume arguments do not preserve the original corpus, model, optimizer, or evaluation schedule";

/// Pure comparison between a resumed bundle's stored `run.json` and this
/// invocation's corpus/model/schedule inputs. Kept free of I/O so the
/// resume-invariant policy (which fields may move under
/// `--allow-schedule-change` and which never can) is unit-testable without a
/// real checkpoint or candidate-model.json.
#[allow(clippy::too_many_arguments)]
fn compare_resume_run(
    stored: &RunDescriptor,
    args: &Args,
    inventory: &CorpusInventory,
    eval_hash: Option<&str>,
    shape: &KnowledgeModelShape,
    parameter_count: usize,
    training_records: usize,
    eval_records: usize,
    tokenizer_vocab_size: usize,
    learning_rate: f64,
) -> ResumeComparison {
    // `learning_rate`/`warmup_steps` are the one sanctioned schedule escape
    // hatch (ADR 104 D5: ordinary adaptive-loop windows deliberately train
    // at a reduced learning rate). Every other invariant below stays strict
    // no matter what `--allow-schedule-change` says.
    let schedule_changed =
        stored.learning_rate != learning_rate || stored.warmup_steps != args.warmup_steps;
    let other_mismatch = stored.schema_version != 1
        || stored.seed != args.seed
        || stored.model_size != args.model
        || stored.shape != *shape
        || stored.parameter_count != parameter_count
        || stored.contract_hash != inventory.contract_hash
        || stored.training_corpus_hash != inventory.fingerprint
        || stored.eval_corpus_hash.as_deref() != eval_hash
        || stored.training_records != training_records
        || stored.eval_records != eval_records
        || args.epochs < stored.epochs
        || stored.batch_size != args.batch_size
        || stored.eval_every != args.eval_every
        || stored.eval_samples != args.eval_samples
        || stored.eval_max_new_tokens != args.eval_max_new_tokens
        || stored.ckpt_every != args.ckpt_every
        || stored.loss_cap != args.loss_cap
        || stored.grad_norm_cap != args.grad_norm_cap
        || stored.tokenizer_vocab_size != tokenizer_vocab_size;
    ResumeComparison {
        schedule_changed,
        other_mismatch,
    }
}

fn verify_resume_manifest(
    args: &Args,
    inventory: &CorpusInventory,
    eval_inventory: Option<&CorpusInventory>,
    training_records: usize,
    eval_records: usize,
    shape: &KnowledgeModelShape,
    parameter_count: usize,
    tokenizer_vocab_size: usize,
    learning_rate: f64,
) -> Result<Option<ResumeVerification>, Box<dyn Error>> {
    let Some(resume) = args.resume.as_deref() else {
        return Ok(None);
    };
    let verified = verify_candidate_knowledge_bundle(&args.artifacts_dir)?;
    let artifact = verified.artifact().clone();
    if artifact.certification.is_some() || args.artifacts_dir.join("active-model.json").exists() {
        return Err(invalid_data(
            "a certified or activated knowledge bundle cannot be resumed in place",
        ));
    }
    if artifact.checkpoint_optimizer_hash.is_none() {
        return Err(invalid_data(
            "resume candidate does not carry a content-addressed Adam state",
        ));
    }
    if resume.canonicalize()? != verified.checkpoint_dir().canonicalize()? {
        return Err(invalid_data(format!(
            "--resume {} is not the exact checkpoint referenced by candidate-model.json",
            resume.display()
        )));
    }
    if artifact.contract_hash != inventory.contract_hash
        || artifact.training_corpus_hash != inventory.fingerprint
        || artifact.model_size != args.model
        || &artifact.shape != shape
        || artifact.shape.vocab_size != tokenizer_vocab_size
    {
        return Err(invalid_data(format!(
            "resume manifest does not match this contract/corpus/model: {artifact:?}"
        )));
    }
    let run_path = args.artifacts_dir.join("run.json");
    let stored: RunDescriptor =
        serde_json::from_reader(BufReader::new(File::open(&run_path).map_err(|error| {
            invalid_data(format!(
                "resume requires original run descriptor {}: {error}",
                run_path.display()
            ))
        })?))?;
    let eval_hash = eval_inventory.map(|value| value.fingerprint.as_str());
    let comparison = compare_resume_run(
        &stored,
        args,
        inventory,
        eval_hash,
        shape,
        parameter_count,
        training_records,
        eval_records,
        tokenizer_vocab_size,
        learning_rate,
    );
    if comparison.rejects(args.allow_schedule_change) {
        return Err(invalid_data(RESUME_SCHEDULE_INVARIANT_ERROR));
    }
    let mut schedule_changes = stored.schedule_changes.clone();
    if comparison.schedule_changed {
        let change = ScheduleChange {
            resumed_at_step: artifact.checkpoint_step,
            epochs: args.epochs,
            from_learning_rate: stored.learning_rate,
            to_learning_rate: learning_rate,
            from_warmup_steps: stored.warmup_steps,
            to_warmup_steps: args.warmup_steps,
        };
        record_schedule_change(&args.artifacts_dir, artifact.checkpoint_step, &change)?;
        schedule_changes.push(change);
    }
    Ok(Some((artifact, schedule_changes)))
}

/// Append an immutable `run-history/schedule-change-step_<step>.json` record
/// for a `--allow-schedule-change` resume, mirroring
/// `preserve_resume_run_descriptor`'s before-resume snapshot. Idempotent: a
/// second resume from the same checkpoint step leaves the first record in
/// place rather than overwriting it.
fn record_schedule_change(
    artifacts_dir: &Path,
    checkpoint_step: usize,
    change: &ScheduleChange,
) -> Result<(), Box<dyn Error>> {
    let target = artifacts_dir
        .join("run-history")
        .join(format!("schedule-change-step_{checkpoint_step:08}.json"));
    if target.exists() {
        return Ok(());
    }
    write_json_pretty_atomic(&target, change)
}

fn preserve_resume_run_descriptor(
    artifacts_dir: &Path,
    checkpoint_step: usize,
) -> Result<(), Box<dyn Error>> {
    let source = artifacts_dir.join("run.json");
    let bytes = std::fs::read(&source)?;
    let history_dir = artifacts_dir.join("run-history");
    std::fs::create_dir_all(&history_dir)?;
    let target = history_dir.join(format!("before-resume-step_{checkpoint_step:08}.json"));
    if target.exists() {
        return Ok(());
    }
    let temporary = target.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, target)?;
    Ok(())
}

fn install_verified_tokenizer(
    source: &Path,
    expected: &str,
    target: &Path,
    source_label: &str,
) -> Result<(), Box<dyn Error>> {
    if target.exists() {
        let actual = fingerprint_artifact_file(target)?;
        if actual != expected {
            return Err(invalid_data(format!(
                "existing tokenizer {} is {actual}, {source_label} requires {expected}",
                target.display(),
            )));
        }
        return Ok(());
    }
    let temporary = target.with_extension(format!("tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(invalid_data(format!(
            "stale temporary tokenizer exists: {}",
            temporary.display()
        )));
    }
    std::fs::copy(source, &temporary)?;
    let actual = fingerprint_artifact_file(&temporary)?;
    if actual != expected {
        let _ = std::fs::remove_file(&temporary);
        return Err(invalid_data(format!(
            "copied {source_label} tokenizer is {actual}, expected {expected}"
        )));
    }
    std::fs::rename(temporary, target)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_and_log<B: AutodiffBackend + AttentionBackend>(
    model: &grug_train::TinyTransformer<B>,
    tokenizer: &BpeTokenizer,
    records: &[KnowledgeSftRecord],
    contract_hash: &str,
    step: usize,
    max_seq_len: usize,
    max_new_tokens: usize,
    device: &B::Device,
    artifacts_dir: &Path,
    eval_log: &Path,
) -> Result<(), Box<dyn Error>>
where
    B::InnerBackend: AttentionBackend,
{
    let eval_model = to_eval_model(model);
    let outputs = generate_knowledge_outputs(
        &eval_model,
        tokenizer,
        records,
        GenerateConfig {
            max_new_tokens,
            max_seq_len,
        },
        device,
    );
    let score = score_knowledge_outputs(records, &outputs);
    println!(
        "  eval@{step}: parse={:.3} action_exact={:.3} wire_exact={:.3} n={}",
        score.parse_rate, score.action_exact_rate, score.target_exact_rate, score.samples
    );
    append_json_line(
        eval_log,
        &EvalLogRow {
            schema_version: 1,
            step,
            contract_hash,
            teacher_forced: &score,
        },
    )?;

    let predictions_dir = artifacts_dir.join("eval_predictions");
    std::fs::create_dir_all(&predictions_dir)?;
    let prediction_path = predictions_dir.join(format!("step_{step:08}.jsonl"));
    let mut writer = BufWriter::new(File::create(prediction_path)?);
    for (record, output) in records.iter().zip(outputs) {
        let expected = KnowledgeAction::parse_runtime(&record.target).ok();
        let predicted = KnowledgeAction::parse_runtime(&output).ok();
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "record_id": record.id,
                "episode_id": record.episode_id,
                "step": record.step,
                "family": record.family,
                "stage": record.stage,
                "expected": record.target,
                "output": output,
                "parse_ok": predicted.is_some(),
                "action_exact": predicted.is_some() && predicted == expected,
            }),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn evenly_spaced(records: &[KnowledgeSftRecord], limit: usize) -> Vec<KnowledgeSftRecord> {
    if records.len() <= limit {
        return records.to_vec();
    }
    (0..limit)
        .map(|index| records[index * records.len() / limit].clone())
        .collect()
}

fn warmup_learning_rate(base: f64, warmup_steps: usize, step: usize) -> f64 {
    if warmup_steps == 0 {
        base
    } else {
        base * ((step + 1).min(warmup_steps) as f64 / warmup_steps as f64)
    }
}

fn json_label<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

fn append_json_line<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_json_pretty_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    let bytes = serde_json::to_vec_pretty(value)?;
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

/// Preserve the already verified candidate's effective context without
/// accepting structural drift or applying the extension a second time.
fn continued_candidate_shape(
    canonical: &KnowledgeModelShape,
    candidate: &KnowledgeModelShape,
) -> Result<KnowledgeModelShape, Box<dyn Error>> {
    if !candidate.matches_canonical_shape(canonical) {
        return Err(invalid_data(format!(
            "candidate checkpoint shape {candidate:?} is not structurally compatible with model shape {canonical:?}"
        )));
    }
    Ok(candidate.clone())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

#[cfg(test)]
mod schedule_change_tests {
    use super::*;

    /// A tiny, arbitrary but internally consistent shape -- only used for
    /// equality comparisons in these tests, never fed to a real model.
    fn sample_shape() -> KnowledgeModelShape {
        KnowledgeModelShape {
            vocab_size: 64,
            d_model: 8,
            n_heads: 1,
            n_layers: 1,
            d_ff: 16,
            max_seq_len: 32,
            positional_encoding: PositionalEncoding::Learned,
            attention_chunk: None,
        }
    }

    fn sample_inventory() -> CorpusInventory {
        CorpusInventory {
            contract_hash: "blake3:contract".into(),
            fingerprint: "blake3:corpus".into(),
            families: BTreeSet::new(),
            languages: BTreeSet::new(),
            stages: BTreeSet::new(),
            record_ids: HashSet::new(),
            episode_ids: HashSet::new(),
        }
    }

    /// Minimal `Args` covering only the fields `compare_resume_run` reads.
    /// `--allow-schedule-change` is left off here; individual tests set it
    /// through `Args::try_parse_from` when it matters.
    fn sample_args(extra: &[&str]) -> Args {
        let mut argv = vec![
            "train_knowledge",
            "--contracts",
            "contracts.json",
            "--train-jsonl",
            "train.jsonl",
            "--artifacts-dir",
            "artifacts",
        ];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("fixed test argv must parse")
    }

    /// The stored `run.json` matching `sample_args(&[])` (every field at its
    /// clap default) / `sample_shape()` / `sample_inventory()` exactly, with
    /// `learning_rate = 1e-4`. Resuming with these exact values back through
    /// `compare_resume_run` must be a clean no-op (neither the schedule nor
    /// anything else has moved).
    fn sample_run() -> RunDescriptor {
        RunDescriptor {
            schema_version: 1,
            seed: 42,
            model_size: KnowledgeModelSize::Small,
            shape: sample_shape(),
            parameter_count: 1_234,
            contract_hash: "blake3:contract".into(),
            training_corpus_hash: "blake3:corpus".into(),
            eval_corpus_hash: None,
            training_records: 100,
            eval_records: 10,
            epochs: 8,
            batch_size: 8,
            learning_rate: 1e-4,
            warmup_steps: 250,
            eval_every: 250,
            eval_samples: 32,
            eval_max_new_tokens: 384,
            ckpt_every: 500,
            loss_cap: 20.0,
            grad_norm_cap: 100.0,
            tokenizer_vocab_size: 64,
            train_jsonl: PathBuf::from("train.jsonl"),
            supplemental_jsonl: Vec::new(),
            eval_jsonl: None,
            supplemental_eval_jsonl: Vec::new(),
            foundation_bundle: None,
            foundation: None,
            adapt_from_bundle: None,
            adapted_from: None,
            steer_from_bundle: None,
            steering_decision: None,
            steering_frontier_exam: None,
            steering_sibling_manifest: None,
            steering_sibling_jsonl: None,
            steered_from: None,
            remediate_from_bundle: None,
            remediated_from: None,
            start_step: 0,
            schedule_changes: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compare(stored: &RunDescriptor, args: &Args, learning_rate: f64) -> ResumeComparison {
        compare_resume_run(
            stored,
            args,
            &sample_inventory(),
            None,
            &sample_shape(),
            stored.parameter_count,
            stored.training_records,
            stored.eval_records,
            stored.tokenizer_vocab_size,
            learning_rate,
        )
    }

    /// An unchanged resume (identical schedule and everything else) is
    /// always a no-op, regardless of the flag.
    #[test]
    fn identical_resume_is_never_a_schedule_change_or_mismatch() {
        let stored = sample_run();
        let args = sample_args(&[]);
        let comparison = compare(&stored, &args, stored.learning_rate);
        assert!(!comparison.schedule_changed);
        assert!(!comparison.other_mismatch);
        assert!(!comparison.rejects(false));
        assert!(!comparison.rejects(true));
    }

    /// (a) A different learning rate with no `--allow-schedule-change`
    /// rejects with the exact existing invariant error.
    #[test]
    fn different_learning_rate_without_flag_rejects_with_existing_error() {
        let stored = sample_run();
        let args = sample_args(&[]);
        assert!(!args.allow_schedule_change);
        let comparison = compare(&stored, &args, stored.learning_rate * 0.2);
        assert!(comparison.schedule_changed);
        assert!(!comparison.other_mismatch);
        assert!(comparison.rejects(args.allow_schedule_change));

        let error = invalid_data(RESUME_SCHEDULE_INVARIANT_ERROR);
        assert_eq!(error.to_string(), RESUME_SCHEDULE_INVARIANT_ERROR);
    }

    /// (b) With `--allow-schedule-change`, a differing learning rate and/or
    /// warmup is accepted, and `record_schedule_change` writes a
    /// `run-history/schedule-change-step_<step>.json` record carrying the
    /// exact from/to values.
    #[test]
    fn allow_schedule_change_accepts_and_records_from_to_values() {
        let stored = sample_run();
        let args = sample_args(&["--allow-schedule-change", "--warmup-steps", "3"]);
        assert!(args.allow_schedule_change);
        let new_learning_rate = stored.learning_rate * 0.2;
        let comparison = compare(&stored, &args, new_learning_rate);
        assert!(comparison.schedule_changed);
        assert!(!comparison.other_mismatch);
        assert!(!comparison.rejects(args.allow_schedule_change));

        let change = ScheduleChange {
            resumed_at_step: 500,
            epochs: args.epochs,
            from_learning_rate: stored.learning_rate,
            to_learning_rate: new_learning_rate,
            from_warmup_steps: stored.warmup_steps,
            to_warmup_steps: args.warmup_steps,
        };
        let temp = tempfile::tempdir().expect("tempdir");
        record_schedule_change(temp.path(), 500, &change).expect("record schedule change");
        let recorded_path = temp
            .path()
            .join("run-history")
            .join("schedule-change-step_00000500.json");
        let recorded: ScheduleChange = serde_json::from_reader(BufReader::new(
            File::open(&recorded_path).expect("schedule-change record exists"),
        ))
        .expect("schedule-change record parses");
        assert_eq!(recorded, change);
        assert_eq!(recorded.from_learning_rate, stored.learning_rate);
        assert_eq!(recorded.to_learning_rate, new_learning_rate);
        assert_eq!(recorded.from_warmup_steps, stored.warmup_steps);
        assert_eq!(recorded.to_warmup_steps, 3);

        // Idempotent: a second resume from the same checkpoint step leaves
        // the first record in place rather than overwriting it.
        let other_change = ScheduleChange {
            to_learning_rate: 9.9,
            ..change.clone()
        };
        record_schedule_change(temp.path(), 500, &other_change).expect("second call is a no-op");
        let still_first: ScheduleChange = serde_json::from_reader(BufReader::new(
            File::open(&recorded_path).expect("schedule-change record still exists"),
        ))
        .expect("schedule-change record still parses");
        assert_eq!(still_first, change);
    }

    /// (c) Even with `--allow-schedule-change`, a changed batch size still
    /// fails: only `learning_rate`/`warmup_steps` are sanctioned escape
    /// hatches.
    #[test]
    fn allow_schedule_change_does_not_waive_batch_size_invariant() {
        let stored = sample_run();
        let args = sample_args(&["--allow-schedule-change", "--batch-size", "999"]);
        let comparison = compare(&stored, &args, stored.learning_rate);
        assert!(!comparison.schedule_changed);
        assert!(comparison.other_mismatch);
        assert!(comparison.rejects(args.allow_schedule_change));
    }

    /// (c) Even with `--allow-schedule-change`, a changed corpus hash still
    /// fails.
    #[test]
    fn allow_schedule_change_does_not_waive_corpus_hash_invariant() {
        let stored = sample_run();
        let args = sample_args(&["--allow-schedule-change"]);
        let mut inventory = sample_inventory();
        inventory.fingerprint = "blake3:different-corpus".into();
        let comparison = compare_resume_run(
            &stored,
            &args,
            &inventory,
            None,
            &sample_shape(),
            stored.parameter_count,
            stored.training_records,
            stored.eval_records,
            stored.tokenizer_vocab_size,
            stored.learning_rate,
        );
        assert!(!comparison.schedule_changed);
        assert!(comparison.other_mismatch);
        assert!(comparison.rejects(args.allow_schedule_change));
    }

    /// `RunDescriptor::schedule_changes` defaults to empty when absent, so a
    /// run.json written before this field existed still deserializes.
    #[test]
    fn schedule_changes_defaults_when_absent_from_legacy_run_json() {
        let mut legacy = serde_json::to_value(sample_run()).expect("serialize sample run");
        legacy
            .as_object_mut()
            .expect("run descriptor is a JSON object")
            .remove("schedule_changes");
        let parsed: RunDescriptor =
            serde_json::from_value(legacy).expect("legacy run.json without schedule_changes");
        assert!(parsed.schedule_changes.is_empty());
    }
}

#[cfg(test)]
mod continuation_shape_tests {
    use super::*;

    #[test]
    fn continued_8k_candidate_keeps_its_extended_rope_scale() {
        let canonical = KnowledgeModelShape::from_config(&KnowledgeModelSize::Small8k.config(8000));
        let mut candidate = canonical.clone();
        candidate.positional_encoding = PositionalEncoding::Rope {
            theta: 10000.0,
            scale: 2.0,
        };
        let continued = continued_candidate_shape(&canonical, &candidate).unwrap();
        assert_eq!(continued, candidate);
        assert_eq!(
            continued.config().positional_encoding,
            candidate.positional_encoding
        );
        assert_eq!(
            continued_candidate_shape(&canonical, &continued).unwrap(),
            candidate
        );
    }

    #[test]
    fn continuation_rejects_structural_drift() {
        let canonical = KnowledgeModelShape::from_config(&KnowledgeModelSize::Small8k.config(8000));
        let mut wrong_length = canonical.clone();
        wrong_length.max_seq_len = 4096;
        assert!(continued_candidate_shape(&canonical, &wrong_length).is_err());
        let mut wrong_theta = canonical.clone();
        wrong_theta.positional_encoding = PositionalEncoding::Rope {
            theta: 20000.0,
            scale: 2.0,
        };
        assert!(continued_candidate_shape(&canonical, &wrong_theta).is_err());
        let mut wrong_width = canonical.clone();
        wrong_width.d_model = 256;
        assert!(continued_candidate_shape(&canonical, &wrong_width).is_err());
    }
}

/// End-to-end CPU coverage for `--remediate-from-bundle` (P6.4): trains a
/// tiny real candidate, then remediates it on the same base corpus plus one
/// engine-generated-shaped supplemental sibling, and finally resumes the
/// remediated segment -- proving the fifth start mode threads weights *and*
/// Adam state through `run()`, records a `remediation` lineage block, and
/// does not break `--resume` of its own output. CPU (`burn-ndarray`) only;
/// never exercises a GPU backend.
#[cfg(test)]
mod remediation_integration_tests {
    use super::*;
    use grug_data::{ToolContractSnapshot, WorkflowLanguage, WorkflowStage};

    fn sample_contracts() -> ToolContractSnapshot {
        ToolContractSnapshot::from_contracts(Vec::new()).expect("empty contract snapshot builds")
    }

    fn sample_record(
        id: &str,
        episode_id: &str,
        contract_hash: &str,
        question: &str,
        answer: &str,
    ) -> KnowledgeSftRecord {
        let tools: Vec<serde_json::Value> = Vec::new();
        let conversation = vec![serde_json::json!({"role": "user", "content": question})];
        let prompt = render_knowledge_prompt_with_system(&tools, &conversation, None);
        let target = KnowledgeAction::Answer {
            text: answer.into(),
            citations: Vec::new(),
        }
        .render_runtime()
        .expect("render_runtime never fails for plain ascii text");
        KnowledgeSftRecord {
            schema_version: grug_data::KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            id: id.into(),
            episode_id: episode_id.into(),
            step: 0,
            stage: WorkflowStage::Answer,
            family: "remediation_test".into(),
            difficulty: 1,
            language: WorkflowLanguage::English,
            contract_hash: contract_hash.into(),
            tools,
            conversation,
            system_context: None,
            prompt,
            target,
        }
    }

    fn write_jsonl(path: &Path, records: &[KnowledgeSftRecord]) {
        let mut body = String::new();
        for record in records {
            body.push_str(&record.to_jsonl().expect("KnowledgeSftRecord serializes"));
            body.push('\n');
        }
        std::fs::write(path, body).expect("write jsonl fixture");
    }

    fn load_candidate(artifacts_dir: &Path) -> KnowledgeModelArtifact {
        serde_json::from_reader(
            File::open(artifacts_dir.join("candidate-model.json"))
                .expect("candidate-model.json exists"),
        )
        .expect("candidate-model.json parses")
    }

    #[test]
    fn remediate_from_bundle_continues_adam_and_records_lineage_and_then_resumes() {
        let workdir = tempfile::tempdir().expect("tempdir");
        let contracts = sample_contracts();
        let contracts_path = workdir.path().join("contracts.json");
        std::fs::write(
            &contracts_path,
            serde_json::to_vec_pretty(&contracts).expect("serialize contracts"),
        )
        .expect("write contracts.json");

        let base_records: Vec<KnowledgeSftRecord> = (0..4)
            .map(|i| {
                sample_record(
                    &format!("base-{i}"),
                    &format!("base-episode-{i}"),
                    &contracts.manifest_hash,
                    &format!("What is {i} plus {i}?"),
                    &format!("the sum is {}", i + i),
                )
            })
            .collect();
        let base_path = workdir.path().join("base.sft.jsonl");
        write_jsonl(&base_path, &base_records);

        // --- Phase 1: train an ordinary, uncertified parent candidate. ---
        let parent_dir = workdir.path().join("parent");
        let parent_args = Args::try_parse_from([
            "train_knowledge",
            "--contracts",
            contracts_path.to_str().unwrap(),
            "--train-jsonl",
            base_path.to_str().unwrap(),
            "--artifacts-dir",
            parent_dir.to_str().unwrap(),
            "--model",
            "small",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--ckpt-every",
            "1",
            "--eval-every",
            "999999",
            "--bpe-vocab-size",
            "260",
            "--learning-rate",
            "1e-3",
        ])
        .expect("parent argv parses");
        validate_args(&parent_args).expect("parent args are valid");
        run::<grug_train::CpuAutodiffBackend>(parent_args, Default::default())
            .expect("parent run succeeds");

        let parent_candidate = load_candidate(&parent_dir);
        assert!(parent_candidate.checkpoint_optimizer_hash.is_some());
        assert!(parent_candidate.certification.is_none());

        // --- Phase 2: remediate the parent on base + one engine-shaped sibling. ---
        let supplemental_records: Vec<KnowledgeSftRecord> = (0..2)
            .map(|i| {
                sample_record(
                    &format!("sibling-{i}"),
                    &format!("sibling-episode-{i}"),
                    &contracts.manifest_hash,
                    &format!("What is the sibling fact {i}?"),
                    &format!("sibling-answer-{i}"),
                )
            })
            .collect();
        let supplemental_dir = workdir.path().join("sibling");
        std::fs::create_dir_all(&supplemental_dir).expect("mkdir sibling");
        let supplemental_path = supplemental_dir.join("train.sft.jsonl");
        write_jsonl(&supplemental_path, &supplemental_records);
        // A P6.3-generator-shaped sidecar `manifest.json` next to the
        // supplemental file, carrying `generator_git_hash`/`seed`.
        std::fs::write(
            supplemental_dir.join("manifest.json"),
            serde_json::json!({
                "schema_version": 1,
                "generator": "akh-episode-generator-engine-v1",
                "generator_git_hash": "deadbeefcafefeed0000000000000000000000",
                "seed": 20260913,
            })
            .to_string(),
        )
        .expect("write sibling manifest.json");

        let remediation_dir = workdir.path().join("remediated");
        let remediation_args = Args::try_parse_from([
            "train_knowledge",
            "--contracts",
            contracts_path.to_str().unwrap(),
            "--train-jsonl",
            base_path.to_str().unwrap(),
            "--supplemental-jsonl",
            supplemental_path.to_str().unwrap(),
            "--artifacts-dir",
            remediation_dir.to_str().unwrap(),
            "--model",
            "small",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--ckpt-every",
            "1",
            "--eval-every",
            "999999",
            "--bpe-vocab-size",
            "260",
            "--learning-rate",
            "5e-4",
            "--remediate-from-bundle",
            parent_dir.to_str().unwrap(),
            // This tiny fixture's supplemental share (2 of 6 records, 1 in 3)
            // is far above the real default ceiling (5%) -- raise it here so
            // the test exercises the mechanism itself, not the boundary (the
            // boundary is covered by `remediation::tests` in the lib crate).
            "--remediation-max-share-ppm",
            "500000",
        ])
        .expect("remediation argv parses");
        validate_args(&remediation_args).expect("remediation args are valid");
        run::<grug_train::CpuAutodiffBackend>(remediation_args, Default::default())
            .expect("remediation run succeeds");

        let remediated_candidate = load_candidate(&remediation_dir);
        let lineage = remediated_candidate
            .remediated_from
            .clone()
            .expect("candidate-model.json must carry a remediation block");
        assert_eq!(
            lineage.parent_checkpoint_step,
            parent_candidate.checkpoint_step
        );
        assert_eq!(
            lineage.parent_checkpoint_model_hash,
            parent_candidate.checkpoint_model_hash
        );
        assert_eq!(
            lineage.parent_checkpoint_optimizer_hash,
            parent_candidate
                .checkpoint_optimizer_hash
                .clone()
                .expect("parent carries an optimizer hash")
        );
        assert_eq!(lineage.supplemental.len(), 1);
        assert_eq!(lineage.supplemental[0].records, 2);
        assert_eq!(lineage.base_training_records, 4);
        assert_eq!(lineage.supplemental_training_records, 2);
        assert_eq!(lineage.supplemental_share_ppm, 333_333);
        assert_eq!(lineage.max_share_ppm, 500_000);
        assert_eq!(
            lineage.supplemental[0].generator_git_hash.as_deref(),
            Some("deadbeefcafefeed0000000000000000000000")
        );
        assert_eq!(lineage.supplemental[0].seed, Some(20260913));
        assert!(
            remediated_candidate.checkpoint_step > parent_candidate.checkpoint_step,
            "remediation must continue the step counter past the parent's"
        );

        // --- Phase 3: --resume of the remediation segment must keep working. ---
        let resume_checkpoint = remediation_dir.join(&remediated_candidate.checkpoint_dir);
        let resume_args = Args::try_parse_from([
            "train_knowledge",
            "--contracts",
            contracts_path.to_str().unwrap(),
            "--train-jsonl",
            base_path.to_str().unwrap(),
            "--supplemental-jsonl",
            supplemental_path.to_str().unwrap(),
            "--artifacts-dir",
            remediation_dir.to_str().unwrap(),
            "--model",
            "small",
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--ckpt-every",
            "1",
            "--eval-every",
            "999999",
            "--bpe-vocab-size",
            "260",
            "--learning-rate",
            "5e-4",
            "--resume",
            resume_checkpoint.to_str().unwrap(),
        ])
        .expect("resume argv parses");
        validate_args(&resume_args).expect("resume args are valid");
        run::<grug_train::CpuAutodiffBackend>(resume_args, Default::default())
            .expect("resume of a remediation run succeeds");

        let resumed_candidate = load_candidate(&remediation_dir);
        assert!(resumed_candidate.checkpoint_step > remediated_candidate.checkpoint_step);
        let resumed_lineage = resumed_candidate
            .remediated_from
            .expect("resumed candidate must still carry the remediation block");
        assert_eq!(resumed_lineage, lineage);
    }
}
