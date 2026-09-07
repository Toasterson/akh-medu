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
    BpeTokenizer, FoundationLineage, GenerateConfig, VerifiedFoundationBundle,
    VerifiedKnowledgeBundle, build_knowledge_batch, checkpoint, validate_knowledge_steering_share,
    verify_active_knowledge_bundle, verify_candidate_knowledge_bundle, verify_foundation_bundle,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[cfg(not(any(feature = "gpu", feature = "cuda")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda"))]
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
    start_step: usize,
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

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let device = {
        let device = grug_train::gpu_device();
        println!("train_knowledge: CubeCL GPU backend, device={device:?}");
        device
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let device = {
        println!("train_knowledge: CPU backend (burn-ndarray)");
        Default::default()
    };

    run::<ActiveBackend>(args, device)
}

fn run<B: AutodiffBackend>(args: Args, device: B::Device) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&args.artifacts_dir)?;

    for (label, parent) in [
        ("--adapt-from-bundle", args.adapt_from_bundle.as_ref()),
        ("--steer-from-bundle", args.steer_from_bundle.as_ref()),
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
    for path in &args.supplemental_jsonl {
        train_records.extend(load_records(path)?);
    }
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
    }
    let tokenizer = load_or_train_tokenizer(&args, &train_records, &tokenizer_path)?;
    let tokenizer_hash = fingerprint_artifact_file(&tokenizer_path)?;
    let model_config = args.model.config(tokenizer.vocab_size());
    let shape = KnowledgeModelShape::from_config(&model_config);
    if let Some(foundation) = &foundation {
        if foundation.artifact().model_size != args.model
            || foundation.artifact().shape != shape
            || foundation.artifact().tokenizer_hash != tokenizer_hash
        {
            return Err(invalid_data(
                "foundation bundle model shape/tokenizer does not match this knowledge run",
            ));
        }
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
        KnowledgeModelSize::Small | KnowledgeModelSize::Small8k => 3e-4,
        KnowledgeModelSize::Model350m | KnowledgeModelSize::Model350m8k => 1e-4,
    });
    println!(
        "train_knowledge: model={} params={} vocab={} context={} lr={learning_rate}",
        args.model,
        parameter_count,
        tokenizer.vocab_size(),
        model_config.max_seq_len
    );

    let resume_artifact = verify_resume_manifest(
        &args,
        &train_inventory,
        eval_inventory.as_ref(),
        train_records.len(),
        eval_records.len(),
        &shape,
        parameter_count,
        tokenizer.vocab_size(),
        learning_rate,
    )?;
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
        model = foundation.load_model_only::<B>(&device)?;
        println!(
            "train_knowledge: warm-started foundation step {} model={} corpus={}",
            foundation.artifact().checkpoint_step,
            foundation.artifact().checkpoint_model_hash,
            foundation.artifact().training_corpus_hash
        );
    } else if let Some(parent) = &adaptation {
        model = parent.load_model_only::<B>(&device)?;
        println!(
            "train_knowledge: adapting certified active step {} model={} contract={} with fresh Adam state",
            parent.artifact().checkpoint_step,
            parent.artifact().checkpoint_model_hash,
            parent.artifact().contract_hash
        );
    }
    let segment_start_step = if steering_parent.is_some() {
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
        start_step: segment_start_step,
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
                    adapted_from: adaptation_lineage.clone(),
                    steered_from: steering_lineage.clone(),
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
        + usize::from(args.steer_from_bundle.is_some());
    if initialization_sources > 1 {
        return Err(invalid_data(
            "--resume, --foundation-bundle, --adapt-from-bundle, and --steer-from-bundle are mutually exclusive",
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
) -> Result<Option<KnowledgeModelArtifact>, Box<dyn Error>> {
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
    if stored.schema_version != 1
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
        || stored.learning_rate != learning_rate
        || stored.warmup_steps != args.warmup_steps
        || stored.eval_every != args.eval_every
        || stored.eval_samples != args.eval_samples
        || stored.eval_max_new_tokens != args.eval_max_new_tokens
        || stored.ckpt_every != args.ckpt_every
        || stored.loss_cap != args.loss_cap
        || stored.grad_norm_cap != args.grad_norm_cap
        || stored.tokenizer_vocab_size != tokenizer_vocab_size
    {
        return Err(invalid_data(
            "resume arguments do not preserve the original corpus, model, optimizer, or evaluation schedule",
        ));
    }
    Ok(Some(artifact))
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
fn evaluate_and_log<B: AutodiffBackend>(
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
) -> Result<(), Box<dyn Error>> {
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

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
