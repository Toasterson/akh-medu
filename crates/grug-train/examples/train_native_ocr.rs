//! Warm-start the compact native OCR stem from a verified BabyLLM text/tool
//! checkpoint while interleaving held-out text-retention updates.
//!
//! This command always writes `multimodal-candidate.json`. A separate frozen
//! audit + network-denied replay gate is required to create an active pointer.

use burn::module::Module;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use clap::Parser;
use grug_data::{
    KnowledgeAction, KnowledgeSftRecord, ToolContractSnapshot, VisionCorpusDomain,
    VisionTextManifest, VisionTextPrediction, audit_vision_sibling_separation, letterbox_rgb,
    load_manifest_rgb, score_vision_text, vision_perceptual_hash, vision_pixel_hash,
};
use grug_train::train::to_eval_model;
use grug_train::{
    AttentionBackend, BpeTokenizer, CompactConvVisionStemConfig, CompactVisionShape,
    CompactVisionTransformer, CompactVisionTransformerConfig, GenerateConfig,
    KnowledgeModelArtifact, MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION, MultimodalModelArtifact,
    VisionCorpusLineage, build_knowledge_batch, build_vision_ocr_batch, fingerprint_artifact_file,
    fingerprint_knowledge_records, generate_compact_vision_ocr, generate_knowledge_outputs,
    load_model_only, rgb_batch_to_nchw, score_knowledge_outputs, to_eval_compact_vision_model,
    train_compact_text_step_guarded, train_compact_vision_step_guarded,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

#[cfg(feature = "cuda")]
const BACKEND_NAME: &str = "cuda";
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
const BACKEND_NAME: &str = "rocm";
#[cfg(all(feature = "metal", not(feature = "gpu"), not(feature = "cuda")))]
const BACKEND_NAME: &str = "metal";
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
const BACKEND_NAME: &str = "cpu-ndarray";

#[derive(Parser, Debug)]
#[command(name = "train_native_ocr")]
struct Args {
    /// Directory containing the verified BabyLLM text artifact and checkpoint.
    #[arg(long)]
    text_bundle: PathBuf,

    /// Normally `active-model.json`. Candidate warm starts require the explicit
    /// unsafe-for-promotion flag below.
    #[arg(long, default_value = "active-model.json")]
    text_artifact: String,

    #[arg(long)]
    allow_uncertified_text_candidate: bool,

    #[arg(long)]
    vision_train_manifest: PathBuf,

    #[arg(long)]
    vision_audit_manifest: PathBuf,

    /// JSON emitted by `validate_vision_corpus` for the exact audit manifest.
    #[arg(long)]
    audit_validation: PathBuf,

    #[arg(long, required = true)]
    retention_train_jsonl: Vec<PathBuf>,

    #[arg(long, required = true)]
    retention_eval_jsonl: Vec<PathBuf>,

    #[arg(long)]
    artifacts_dir: PathBuf,

    #[arg(long, default_value_t = 480)]
    canvas_width: usize,

    #[arg(long, default_value_t = 192)]
    canvas_height: usize,

    #[arg(long, default_value_t = 64)]
    vision_hidden_channels: usize,

    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Stem-only visual updates while the complete text trunk is `no_grad`.
    #[arg(long, default_value_t = 500)]
    stem_warmup_updates: usize,

    #[arg(long, default_value_t = 3)]
    joint_epochs: usize,

    /// Perform one text retention update after this many joint visual updates.
    #[arg(long, default_value_t = 1)]
    retention_every: usize,

    #[arg(long, default_value_t = 3e-4)]
    stem_learning_rate: f64,

    #[arg(long, default_value_t = 2e-5)]
    joint_learning_rate: f64,

    #[arg(long, default_value_t = 2e-6)]
    minimum_learning_rate: f64,

    #[arg(long, default_value_t = 20.0)]
    loss_cap: f32,

    #[arg(long, default_value_t = 100.0)]
    grad_norm_cap: f32,

    #[arg(long, default_value_t = 250)]
    checkpoint_every: usize,

    /// Frozen audit semantic groups evaluated after training. Every selected
    /// group includes all seven rendering variants.
    #[arg(long, default_value_t = 15)]
    audit_eval_semantic_items: usize,

    #[arg(long, default_value_t = 60)]
    retention_eval_samples: usize,

    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 16_000_000)]
    maximum_image_pixels: u64,

    #[arg(long, default_value_t = 120_400)]
    seed: u64,

    /// Validate all lineage, corpus, context and retention boundaries without
    /// allocating model weights.
    #[arg(long)]
    validate_only: bool,

    /// Resume an incomplete periodic checkpoint inside `--artifacts-dir`.
    /// Model, Adam state, run descriptor and update counters are all verified.
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Debug)]
struct VerifiedTextBase {
    artifact: KnowledgeModelArtifact,
    artifact_hash: String,
    tokenizer_path: PathBuf,
    contracts_path: PathBuf,
    checkpoint_dir: PathBuf,
}

#[derive(Debug)]
struct VisionCase {
    manifest: VisionTextManifest,
    normalized_rgb: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct RunDescriptor<'a> {
    schema_version: u32,
    backend: &'static str,
    seed: u64,
    base_text_artifact_hash: &'a str,
    base_text_checkpoint_hash: &'a str,
    base_text_checkpoint_step: usize,
    base_text_certified: bool,
    canvas_width: usize,
    canvas_height: usize,
    visual_tokens: usize,
    text_context_tokens: usize,
    train_corpus: &'a VisionCorpusLineage,
    audit_corpus: &'a VisionCorpusLineage,
    audit_validation_hash: &'a str,
    retention_corpus_hash: &'a str,
    retention_records: usize,
    retention_eval_corpus_hash: &'a str,
    retention_eval_records: usize,
    batch_size: usize,
    stem_warmup_updates: usize,
    joint_epochs: usize,
    retention_every: usize,
    stem_learning_rate: f64,
    joint_learning_rate: f64,
    minimum_learning_rate: f64,
    loss_cap: f32,
    grad_norm_cap: f32,
    checkpoint_every: usize,
    audit_eval_semantic_items: usize,
    retention_eval_samples: usize,
    max_new_tokens: usize,
    maximum_image_pixels: u64,
}

#[derive(Debug, Serialize)]
struct UpdateLog<'a> {
    update: usize,
    stage: &'a str,
    epoch: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f64,
    skipped: bool,
    elapsed_ms: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CheckpointMetadata {
    schema_version: u32,
    architecture: String,
    backend: String,
    stage: String,
    update: usize,
    stem_updates: usize,
    joint_vision_updates: usize,
    joint_text_updates: usize,
    skipped_updates: usize,
    parameter_count: usize,
    complete: bool,
    model_hash: String,
    optimizer_hash: String,
}

#[derive(Debug, Serialize)]
struct CompletionReport<'a> {
    kind: &'static str,
    candidate_only: bool,
    active_model_written: bool,
    backend: &'static str,
    updates: usize,
    stem_updates: usize,
    joint_vision_updates: usize,
    joint_text_updates: usize,
    skipped_updates: usize,
    parameter_count: usize,
    training_seconds: f64,
    base_text_score: &'a grug_train::KnowledgeTeacherForcedEval,
    final_text_score: &'a grug_train::KnowledgeTeacherForcedEval,
    text_action_exact_regression: f64,
    audit_cases_scored: usize,
    audit_score: &'a grug_data::VisionTextScoreSummary,
    candidate_artifact: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;
    let base = verify_text_base(&args)?;
    let tokenizer = BpeTokenizer::load(&base.tokenizer_path)?;
    if tokenizer.vocab_size() != base.artifact.shape.vocab_size {
        return Err(invalid_data(
            "text tokenizer vocabulary does not match artifact shape",
        ));
    }
    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&base.contracts_path)?))?;
    contracts.verify_integrity()?;
    if contracts.manifest_hash != base.artifact.contract_hash {
        return Err(invalid_data(
            "contract snapshot identity differs from text artifact",
        ));
    }

    let training_manifests = read_vision_manifest(&args.vision_train_manifest)?;
    let audit_manifests = read_vision_manifest(&args.vision_audit_manifest)?;
    validate_manifest_domains(&training_manifests, VisionCorpusDomain::TrainingSibling)?;
    validate_manifest_domains(&audit_manifests, VisionCorpusDomain::Audit)?;
    let separation = audit_vision_sibling_separation(
        &audit_manifests,
        &training_manifests,
        grug_data::SiblingSeparationPolicy::default(),
    );
    if !separation.admitted
        || !separation
            .evidence
            .held_out_axes
            .iter()
            .any(|axis| axis == "renderer")
        || !separation
            .evidence
            .held_out_axes
            .iter()
            .any(|axis| axis == "content_bank")
    {
        return Err(invalid_data(format!(
            "audit/training separation is insufficient: {separation:?}"
        )));
    }
    let validation = validate_audit_evidence(
        &args.audit_validation,
        &args.vision_audit_manifest,
        audit_manifests.len(),
    )?;
    let audit_validation_hash = fingerprint_artifact_file(&args.audit_validation)?;
    let held_out_axes: BTreeSet<String> = separation.evidence.held_out_axes.into_iter().collect();
    let training_lineage = vision_lineage(&args.vision_train_manifest, &training_manifests, None)?;
    let audit_lineage = vision_lineage(
        &args.vision_audit_manifest,
        &audit_manifests,
        Some(held_out_axes),
    )?;
    if validation
        .pointer("/minimum_observed_cell_cases")
        .and_then(Value::as_u64)
        != Some(audit_lineage.minimum_language_layout_corruption_cases as u64)
    {
        return Err(invalid_data(
            "audit validation matrix floor differs from recomputed manifest floor",
        ));
    }
    training_lineage.verify(false).map_err(invalid_data)?;
    audit_lineage.verify(true).map_err(invalid_data)?;

    let retention_train = load_retention_records(&args.retention_train_jsonl)?;
    let retention_eval = load_retention_records(&args.retention_eval_jsonl)?;
    validate_retention_records(&retention_train, &contracts)?;
    validate_retention_records(&retention_eval, &contracts)?;
    reject_retention_leakage(&retention_train, &retention_eval)?;
    let retention_hash = fingerprint_knowledge_records(&retention_train)?;
    let retention_eval_hash = fingerprint_knowledge_records(&retention_eval)?;
    let vision_shape = CompactVisionShape {
        input_channels: 3,
        hidden_channels: args.vision_hidden_channels,
        output_stride: CompactConvVisionStemConfig::OUTPUT_STRIDE,
        max_grid_rows: args.canvas_height.div_ceil(16),
        max_grid_columns: args.canvas_width.div_ceil(16),
        canvas_width: args.canvas_width,
        canvas_height: args.canvas_height,
    };
    vision_shape
        .verify(&base.artifact.shape)
        .map_err(invalid_data)?;
    let text_context_tokens = base
        .artifact
        .shape
        .max_seq_len
        .saturating_sub(vision_shape.visual_tokens());
    validate_ocr_token_budget(&tokenizer, &training_manifests, text_context_tokens)?;
    validate_ocr_token_budget(&tokenizer, &audit_manifests, text_context_tokens)?;

    let run = RunDescriptor {
        schema_version: 1,
        backend: BACKEND_NAME,
        seed: args.seed,
        base_text_artifact_hash: &base.artifact_hash,
        base_text_checkpoint_hash: &base.artifact.checkpoint_model_hash,
        base_text_checkpoint_step: base.artifact.checkpoint_step,
        base_text_certified: base.artifact.certification.is_some(),
        canvas_width: args.canvas_width,
        canvas_height: args.canvas_height,
        visual_tokens: vision_shape.visual_tokens(),
        text_context_tokens,
        train_corpus: &training_lineage,
        audit_corpus: &audit_lineage,
        audit_validation_hash: &audit_validation_hash,
        retention_corpus_hash: &retention_hash,
        retention_records: retention_train.len(),
        retention_eval_corpus_hash: &retention_eval_hash,
        retention_eval_records: retention_eval.len(),
        batch_size: args.batch_size,
        stem_warmup_updates: args.stem_warmup_updates,
        joint_epochs: args.joint_epochs,
        retention_every: args.retention_every,
        stem_learning_rate: args.stem_learning_rate,
        joint_learning_rate: args.joint_learning_rate,
        minimum_learning_rate: args.minimum_learning_rate,
        loss_cap: args.loss_cap,
        grad_norm_cap: args.grad_norm_cap,
        checkpoint_every: args.checkpoint_every,
        audit_eval_semantic_items: args.audit_eval_semantic_items,
        retention_eval_samples: args.retention_eval_samples,
        max_new_tokens: args.max_new_tokens,
        maximum_image_pixels: args.maximum_image_pixels,
    };
    let run_value = serde_json::to_value(&run)?;
    let resume_metadata = args
        .resume
        .as_deref()
        .map(|checkpoint| verify_resume_inputs(&args, checkpoint, &run_value, &base))
        .transpose()?;
    if args.validate_only {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &run)?;
        println!();
        return Ok(());
    }
    if resume_metadata.is_none() {
        if args.artifacts_dir.exists() && args.artifacts_dir.read_dir()?.next().is_some() {
            return Err(invalid_data(format!(
                "artifacts directory {} is not empty; multimodal runs are append-only",
                args.artifacts_dir.display()
            )));
        }
        std::fs::create_dir_all(&args.artifacts_dir)?;
        write_json_atomic(&args.artifacts_dir.join("run.json"), &run)?;
        copy_verified_file(
            &base.tokenizer_path,
            &args.artifacts_dir.join("tokenizer.json"),
            &base.artifact.tokenizer_hash,
        )?;
        copy_file_atomic(
            &base.contracts_path,
            &args.artifacts_dir.join("contracts.json"),
        )?;
    }

    eprintln!(
        "loading and normalizing {} training images",
        training_manifests.len()
    );
    let training_cases = load_cases(&args.vision_train_manifest, training_manifests, &args)?;
    eprintln!(
        "loading and normalizing {} frozen audit images",
        audit_manifests.len()
    );
    let audit_cases = load_cases(&args.vision_audit_manifest, audit_manifests, &args)?;

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let device = grug_train::gpu_device();
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let device = Default::default();
    type B = ActiveBackend;
    B::seed(&device, args.seed);
    let text_config = base.artifact.shape.config();
    let (text_model, loaded_step) = load_model_only(
        &base.checkpoint_dir,
        text_config.init::<B>(&device),
        &device,
    )?;
    if loaded_step != base.artifact.checkpoint_step {
        return Err(invalid_data(format!(
            "loaded base step {loaded_step} differs from artifact {}",
            base.artifact.checkpoint_step
        )));
    }
    let eval_records = evenly_spaced(&retention_eval, args.retention_eval_samples);
    let base_eval_model = to_eval_model(&text_model);
    let base_outputs = generate_knowledge_outputs(
        &base_eval_model,
        &tokenizer,
        &eval_records,
        GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            max_seq_len: base.artifact.shape.max_seq_len,
        },
        &device,
    );
    let base_text_score = score_knowledge_outputs(&eval_records, &base_outputs);
    write_or_verify_json(
        &args.artifacts_dir.join("text-retention-before.json"),
        &base_text_score,
    )?;

    let config = CompactVisionTransformerConfig::new(
        text_config,
        CompactConvVisionStemConfig::new(
            3,
            args.vision_hidden_channels,
            base.artifact.shape.d_model,
            vision_shape.max_grid_rows,
            vision_shape.max_grid_columns,
        ),
    );
    let resume_joint = resume_metadata
        .as_ref()
        .is_some_and(|metadata| matches!(metadata.stage.as_str(), "joint" | "complete_candidate"));
    let shell = if resume_joint {
        config.init::<B>(&device)
    } else {
        config.init_with_transformer(text_model.no_grad(), &device)
    };
    let fresh_optimizer = compact_optimizer::<B>();
    let (mut model, mut optimizer) =
        if let (Some(path), Some(metadata)) = (args.resume.as_deref(), resume_metadata.as_ref()) {
            load_multimodal_checkpoint(path, shell, fresh_optimizer, &device, metadata)?
        } else {
            (shell, fresh_optimizer)
        };
    let parameter_count = model.num_params();
    if resume_metadata
        .as_ref()
        .is_some_and(|metadata| metadata.parameter_count != parameter_count)
    {
        return Err(invalid_data(
            "resume checkpoint parameter count differs from the reconstructed model",
        ));
    }
    let update_log_path = resume_metadata.as_ref().map_or_else(
        || args.artifacts_dir.join("updates.jsonl"),
        |metadata| {
            args.artifacts_dir
                .join(format!("updates.resume-from-{:08}.jsonl", metadata.update))
        },
    );
    let training_started = Instant::now();
    let mut update = resume_metadata
        .as_ref()
        .map(|metadata| metadata.update)
        .unwrap_or(0);
    let mut stem_updates = resume_metadata
        .as_ref()
        .map(|metadata| metadata.stem_updates)
        .unwrap_or(0);
    let mut joint_vision_updates = resume_metadata
        .as_ref()
        .map(|metadata| metadata.joint_vision_updates)
        .unwrap_or(0);
    let mut joint_text_updates = resume_metadata
        .as_ref()
        .map(|metadata| metadata.joint_text_updates)
        .unwrap_or(0);
    let mut skipped_updates = resume_metadata
        .as_ref()
        .map(|metadata| metadata.skipped_updates)
        .unwrap_or(0);
    let mut train_order: Vec<usize> = (0..training_cases.len()).collect();
    train_order.shuffle(&mut StdRng::seed_from_u64(
        args.seed ^ 0x7374_656d_2d6f_7264,
    ));
    let total_joint_vision = args.joint_epochs * training_cases.len().div_ceil(args.batch_size);
    validate_checkpoint_counters(
        resume_metadata.as_ref(),
        args.stem_warmup_updates,
        total_joint_vision,
        args.retention_every,
    )?;

    if !resume_joint {
        for warmup in stem_updates..args.stem_warmup_updates {
            let indices = cyclic_batch_indices(&train_order, warmup, args.batch_size);
            let (images, batch) = build_visual_batch::<B>(
                &training_cases,
                &indices,
                &tokenizer,
                text_context_tokens,
                args.canvas_width,
                args.canvas_height,
                &device,
            )?;
            let step_started = Instant::now();
            let (next, loss, grad_norm) = train_compact_vision_step_guarded(
                model,
                &mut optimizer,
                images,
                &batch,
                args.stem_learning_rate,
                args.loss_cap,
                args.grad_norm_cap,
            );
            model = next;
            update += 1;
            stem_updates += 1;
            let skipped = update_was_skipped(loss, grad_norm, &args);
            skipped_updates += usize::from(skipped);
            log_update(
                &update_log_path,
                &UpdateLog {
                    update,
                    stage: "stem_warmup_text_frozen",
                    epoch: 0,
                    loss,
                    grad_norm,
                    learning_rate: args.stem_learning_rate,
                    skipped,
                    elapsed_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
                },
            )?;
            maybe_checkpoint(
                &args,
                &model,
                &optimizer,
                "stem_warmup_text_frozen",
                update,
                stem_updates,
                joint_vision_updates,
                joint_text_updates,
                skipped_updates,
                parameter_count,
                false,
            )?;
        }

        // Reinitialize the module shell with grad-enabled parameters and load
        // the warmed record. Adam intentionally restarts at this boundary.
        model = config.init::<B>(&device).load_record(model.into_record());
        optimizer = compact_optimizer::<B>();
    }
    let mut retention_order: Vec<usize> = (0..retention_train.len()).collect();
    retention_order.shuffle(&mut StdRng::seed_from_u64(
        args.seed ^ 0x7265_7465_6e74_696f,
    ));
    let batches_per_epoch = training_cases.len().div_ceil(args.batch_size);

    for epoch in 0..args.joint_epochs {
        let mut train_order: Vec<usize> = (0..training_cases.len()).collect();
        train_order.shuffle(&mut StdRng::seed_from_u64(
            args.seed ^ 0x6a6f_696e_742d_6570 ^ epoch as u64,
        ));
        for (batch_index, indices) in train_order.chunks(args.batch_size).enumerate() {
            let schedule_index = epoch * batches_per_epoch + batch_index;
            if schedule_index < joint_vision_updates {
                continue;
            }
            if schedule_index != joint_vision_updates {
                return Err(invalid_data(
                    "resume counters do not identify a contiguous joint-training prefix",
                ));
            }
            let learning_rate = cosine_learning_rate(
                schedule_index,
                total_joint_vision,
                args.joint_learning_rate,
                args.minimum_learning_rate,
            );
            let (images, batch) = build_visual_batch::<B>(
                &training_cases,
                indices,
                &tokenizer,
                text_context_tokens,
                args.canvas_width,
                args.canvas_height,
                &device,
            )?;
            let step_started = Instant::now();
            let (next, loss, grad_norm) = train_compact_vision_step_guarded(
                model,
                &mut optimizer,
                images,
                &batch,
                learning_rate,
                args.loss_cap,
                args.grad_norm_cap,
            );
            model = next;
            update += 1;
            joint_vision_updates += 1;
            let skipped = update_was_skipped(loss, grad_norm, &args);
            skipped_updates += usize::from(skipped);
            log_update(
                &update_log_path,
                &UpdateLog {
                    update,
                    stage: "joint_vision",
                    epoch: epoch + 1,
                    loss,
                    grad_norm,
                    learning_rate,
                    skipped,
                    elapsed_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
                },
            )?;

            if joint_vision_updates % args.retention_every == 0 {
                let retention_indices =
                    cyclic_batch_indices(&retention_order, joint_text_updates, args.batch_size);
                let records: Vec<_> = retention_indices
                    .iter()
                    .map(|index| retention_train[*index].clone())
                    .collect();
                let batch = build_knowledge_batch::<B, _>(
                    &tokenizer,
                    &records,
                    base.artifact.shape.max_seq_len,
                    &device,
                )?;
                let step_started = Instant::now();
                let (next, loss, grad_norm) = train_compact_text_step_guarded(
                    model,
                    &mut optimizer,
                    &batch,
                    learning_rate,
                    args.loss_cap,
                    args.grad_norm_cap,
                );
                model = next;
                update += 1;
                joint_text_updates += 1;
                let skipped = update_was_skipped(loss, grad_norm, &args);
                skipped_updates += usize::from(skipped);
                log_update(
                    &update_log_path,
                    &UpdateLog {
                        update,
                        stage: "joint_text_retention",
                        epoch: epoch + 1,
                        loss,
                        grad_norm,
                        learning_rate,
                        skipped,
                        elapsed_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
                    },
                )?;
            }
            maybe_checkpoint(
                &args,
                &model,
                &optimizer,
                "joint",
                update,
                stem_updates,
                joint_vision_updates,
                joint_text_updates,
                skipped_updates,
                parameter_count,
                false,
            )?;
        }
    }

    let checkpoint_dir = if resume_metadata
        .as_ref()
        .is_some_and(|metadata| metadata.stage == "complete_candidate")
    {
        args.resume
            .clone()
            .expect("complete resume metadata has a checkpoint path")
    } else {
        save_checkpoint(
            &args,
            &model,
            &optimizer,
            "complete_candidate",
            update,
            stem_updates,
            joint_vision_updates,
            joint_text_updates,
            skipped_updates,
            parameter_count,
            true,
        )?
    };
    let eval_model = to_eval_compact_vision_model(&model);
    let final_text_outputs = generate_knowledge_outputs(
        eval_model.transformer(),
        &tokenizer,
        &eval_records,
        GenerateConfig {
            max_new_tokens: args.max_new_tokens,
            max_seq_len: base.artifact.shape.max_seq_len,
        },
        &device,
    );
    let final_text_score = score_knowledge_outputs(&eval_records, &final_text_outputs);
    write_or_verify_json(
        &args.artifacts_dir.join("text-retention-after.json"),
        &final_text_score,
    )?;

    let selected_audit = select_semantic_groups(&audit_cases, args.audit_eval_semantic_items);
    let mut predictions = Vec::with_capacity(selected_audit.len());
    for case in &selected_audit {
        let image = rgb_batch_to_nchw::<<B as AutodiffBackend>::InnerBackend>(
            std::slice::from_ref(&case.normalized_rgb),
            args.canvas_width,
            args.canvas_height,
            &device,
        )?;
        predictions.push(VisionTextPrediction {
            case_id: case.manifest.case_id.clone(),
            text: generate_compact_vision_ocr(
                &eval_model,
                image,
                &tokenizer,
                args.max_new_tokens,
                base.artifact.shape.max_seq_len,
                &device,
            ),
            confidence: None,
        });
    }
    let audit_manifests: Vec<_> = selected_audit
        .iter()
        .map(|case| case.manifest.clone())
        .collect();
    let audit_score = score_vision_text(&audit_manifests, &predictions)?;
    write_or_verify_json(&args.artifacts_dir.join("audit-score.json"), &audit_score)?;
    write_or_verify_json_lines(
        &args.artifacts_dir.join("audit-predictions.jsonl"),
        &predictions,
    )?;

    let checkpoint_relative = checkpoint_dir
        .strip_prefix(&args.artifacts_dir)?
        .to_path_buf();
    let candidate = MultimodalModelArtifact {
        schema_version: MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION,
        model_size: base.artifact.model_size,
        text_shape: base.artifact.shape.clone(),
        vision_shape,
        parameter_count,
        tokenizer_file: "tokenizer.json".into(),
        tokenizer_hash: base.artifact.tokenizer_hash.clone(),
        contracts_file: "contracts.json".into(),
        contract_hash: base.artifact.contract_hash.clone(),
        checkpoint_dir: checkpoint_relative.to_string_lossy().into_owned(),
        checkpoint_step: update,
        checkpoint_model_hash: fingerprint_artifact_file(&checkpoint_dir.join("model.mpk"))?,
        checkpoint_optimizer_hash: Some(fingerprint_artifact_file(
            &checkpoint_dir.join("optim.mpk"),
        )?),
        base_text_artifact_hash: base.artifact_hash,
        base_text_checkpoint_step: base.artifact.checkpoint_step,
        base_text_checkpoint_hash: base.artifact.checkpoint_model_hash,
        base_text_certified: base.artifact.certification.is_some(),
        vision_training_corpus: training_lineage,
        audit_corpus: audit_lineage,
        retention_corpus_hash: retention_hash,
        retention_records: retention_train.len(),
        retention_eval_corpus_hash: retention_eval_hash,
        retention_eval_records: retention_eval.len(),
        stem_warmup_updates: stem_updates,
        joint_vision_updates,
        joint_text_updates,
        skipped_updates,
        certification: None,
    };
    candidate.verify().map_err(invalid_data)?;
    let candidate_path = args.artifacts_dir.join("multimodal-candidate.json");
    write_or_verify_json(&candidate_path, &candidate)?;
    let regression =
        (base_text_score.action_exact_rate - final_text_score.action_exact_rate).max(0.0);
    let report = CompletionReport {
        kind: "native_ocr_shared_text_decoder_training",
        candidate_only: true,
        active_model_written: false,
        backend: BACKEND_NAME,
        updates: update,
        stem_updates,
        joint_vision_updates,
        joint_text_updates,
        skipped_updates,
        parameter_count,
        training_seconds: training_started.elapsed().as_secs_f64(),
        base_text_score: &base_text_score,
        final_text_score: &final_text_score,
        text_action_exact_regression: regression,
        audit_cases_scored: selected_audit.len(),
        audit_score: &audit_score,
        candidate_artifact: candidate_path.display().to_string(),
    };
    write_json_atomic(&args.artifacts_dir.join("completion.json"), &report)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    Ok(())
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.batch_size == 0
        || args.joint_epochs == 0
        || args.retention_every == 0
        || args.retention_eval_samples == 0
        || args.audit_eval_semantic_items == 0
        || args.canvas_width == 0
        || args.canvas_height == 0
        || args.vision_hidden_channels == 0
        || args.checkpoint_every == 0
    {
        return Err(invalid_data(
            "counts, dimensions and checkpoint cadence must be positive",
        ));
    }
    if !(args.minimum_learning_rate > 0.0
        && args.minimum_learning_rate <= args.joint_learning_rate
        && args.joint_learning_rate <= args.stem_learning_rate)
    {
        return Err(invalid_data(
            "learning rates must satisfy 0 < minimum <= joint <= stem",
        ));
    }
    if !args.loss_cap.is_finite()
        || args.loss_cap <= 0.0
        || !args.grad_norm_cap.is_finite()
        || args.grad_norm_cap <= 0.0
    {
        return Err(invalid_data(
            "loss/gradient caps must be finite and positive",
        ));
    }
    if Path::new(&args.text_artifact).components().count() != 1 {
        return Err(invalid_data(
            "--text-artifact must be one filename inside the bundle",
        ));
    }
    Ok(())
}

fn verify_text_base(args: &Args) -> Result<VerifiedTextBase, Box<dyn Error>> {
    let manifest_path = resolve_child(&args.text_bundle, &args.text_artifact)?;
    let artifact: KnowledgeModelArtifact =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    artifact.verify().map_err(invalid_data)?;
    let certified = artifact.certification.is_some();
    if !certified && !args.allow_uncertified_text_candidate {
        return Err(invalid_data(
            "text base is not certified; pass an active model or explicitly mark this as a non-promotable experiment",
        ));
    }
    let tokenizer_path = resolve_child(&args.text_bundle, &artifact.tokenizer_file)?;
    verify_hash(&tokenizer_path, &artifact.tokenizer_hash)?;
    let contracts_path = resolve_child(&args.text_bundle, &artifact.contracts_file)?;
    let checkpoint_dir = resolve_child(&args.text_bundle, &artifact.checkpoint_dir)?;
    verify_hash(
        &checkpoint_dir.join("model.mpk"),
        &artifact.checkpoint_model_hash,
    )?;
    Ok(VerifiedTextBase {
        artifact,
        artifact_hash: fingerprint_artifact_file(&manifest_path)?,
        tokenizer_path,
        contracts_path,
        checkpoint_dir,
    })
}

fn read_vision_manifest(path: &Path) -> Result<Vec<VisionTextManifest>, Box<dyn Error>> {
    let mut cases = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        cases.push(
            serde_json::from_str(&line).map_err(|error| {
                invalid_data(format!("{}:{}: {error}", path.display(), index + 1))
            })?,
        );
    }
    if cases.is_empty() {
        return Err(invalid_data(format!(
            "{} contains no cases",
            path.display()
        )));
    }
    Ok(cases)
}

fn validate_manifest_domains(
    cases: &[VisionTextManifest],
    domain: VisionCorpusDomain,
) -> Result<(), Box<dyn Error>> {
    let mut ids = HashSet::new();
    for case in cases {
        if case.domain != domain {
            return Err(invalid_data(format!(
                "case {} is {:?}, expected {:?}",
                case.case_id, case.domain, domain
            )));
        }
        if !ids.insert(&case.case_id) {
            return Err(invalid_data(format!(
                "duplicate vision case {}",
                case.case_id
            )));
        }
        let provenance = case
            .provenance
            .as_ref()
            .ok_or_else(|| invalid_data(format!("case {} lacks provenance", case.case_id)))?;
        if !provenance.source_hash.starts_with("blake3:")
            || case
                .pixel_hash_blake3
                .as_deref()
                .is_none_or(|hash| !hash.starts_with("blake3:"))
        {
            return Err(invalid_data(format!(
                "case {} lacks cryptographic source/pixel hashes",
                case.case_id
            )));
        }
    }
    Ok(())
}

fn validate_audit_evidence(
    path: &Path,
    manifest_path: &Path,
    cases: usize,
) -> Result<Value, Box<dyn Error>> {
    let evidence: Value = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let manifest_hash = fingerprint_artifact_file(manifest_path)?;
    if evidence.get("valid").and_then(Value::as_bool) != Some(true)
        || evidence.get("manifest_hash").and_then(Value::as_str) != Some(manifest_hash.as_str())
        || evidence.get("cases").and_then(Value::as_u64) != Some(cases as u64)
        || evidence
            .pointer("/separation_from_training/admitted")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(invalid_data(
            "audit validation is not green or is bound to a different manifest",
        ));
    }
    Ok(evidence)
}

fn vision_lineage(
    path: &Path,
    cases: &[VisionTextManifest],
    held_out_axes: Option<BTreeSet<String>>,
) -> Result<VisionCorpusLineage, Box<dyn Error>> {
    let mut semantics = BTreeSet::new();
    let mut generators = BTreeSet::new();
    let mut renderers = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut languages = BTreeSet::new();
    let mut layouts = BTreeSet::new();
    let mut corruptions = BTreeSet::new();
    let mut cells = BTreeMap::<String, usize>::new();
    for case in cases {
        semantics.insert(case.semantic_id.clone());
        generators.insert(case.generator_version.clone());
        renderers.insert(case.renderer_id.clone());
        if let Some(provenance) = &case.provenance {
            sources.insert(provenance.source_hash.clone());
        }
        let language = language_label(case);
        let layout = json_label(case.layout);
        let corruption = json_label(case.corruption);
        languages.insert(language.clone());
        layouts.insert(layout.clone());
        corruptions.insert(corruption.clone());
        *cells
            .entry(format!("{language}/{layout}/{corruption}"))
            .or_default() += 1;
    }
    let minimum = ["en", "de", "fr"]
        .into_iter()
        .flat_map(|language| {
            ["line", "paragraph", "receipt", "terminal", "api_error"]
                .into_iter()
                .flat_map(move |layout| {
                    [
                        "clean",
                        "low_contrast",
                        "salt_pepper",
                        "box_blur",
                        "occlusion",
                        "uneven_lighting",
                        "horizontal_shear",
                    ]
                    .into_iter()
                    .map(move |corruption| format!("{language}/{layout}/{corruption}"))
                })
        })
        .map(|key| cells.get(&key).copied().unwrap_or(0))
        .min()
        .unwrap_or(0);
    Ok(VisionCorpusLineage {
        manifest_hash: fingerprint_artifact_file(path)?,
        cases: cases.len(),
        semantic_items: semantics.len(),
        minimum_language_layout_corruption_cases: minimum,
        generator_versions: generators,
        renderer_ids: renderers,
        source_hashes: sources,
        languages,
        layouts,
        corruptions,
        held_out_axes: held_out_axes.unwrap_or_default(),
    })
}

fn load_cases(
    manifest_path: &Path,
    manifests: Vec<VisionTextManifest>,
    args: &Args,
) -> Result<Vec<VisionCase>, Box<dyn Error>> {
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    manifests
        .into_iter()
        .map(|manifest| {
            let (width, height, rgb) =
                load_manifest_rgb(root, &manifest, args.maximum_image_pixels)?;
            if width != manifest.width || height != manifest.height || manifest.channels != 3 {
                return Err(invalid_data(format!(
                    "case {} decoded shape differs from manifest",
                    manifest.case_id
                )));
            }
            if vision_pixel_hash(&rgb) != manifest.pixel_hash_fnv1a64
                || manifest.pixel_hash_blake3.as_deref()
                    != Some(format!("blake3:{}", blake3::hash(&rgb).to_hex()).as_str())
                || vision_perceptual_hash(&rgb, width, height) != manifest.perceptual_hash
            {
                return Err(invalid_data(format!(
                    "case {} decoded pixels fail manifest hashes",
                    manifest.case_id
                )));
            }
            let normalized_rgb = letterbox_rgb(
                &rgb,
                width,
                height,
                args.canvas_width as u32,
                args.canvas_height as u32,
                [248, 248, 248],
            )?;
            Ok(VisionCase {
                manifest,
                normalized_rgb,
            })
        })
        .collect()
}

fn load_retention_records(paths: &[PathBuf]) -> Result<Vec<KnowledgeSftRecord>, Box<dyn Error>> {
    let mut records = Vec::new();
    for path in paths {
        for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(&line).map_err(|error| {
                invalid_data(format!("{}:{}: {error}", path.display(), index + 1))
            })?);
        }
    }
    if records.is_empty() {
        return Err(invalid_data("retention corpus contains no records"));
    }
    Ok(records)
}

fn validate_retention_records(
    records: &[KnowledgeSftRecord],
    contracts: &ToolContractSnapshot,
) -> Result<(), Box<dyn Error>> {
    let mut ids = HashSet::new();
    for record in records {
        if record.contract_hash != contracts.manifest_hash || !ids.insert(&record.id) {
            return Err(invalid_data(format!(
                "retention record {} has duplicate identity or wrong contract",
                record.id
            )));
        }
        KnowledgeAction::parse_runtime(&record.target)
            .map_err(|error| invalid_data(error.to_string()))?;
    }
    Ok(())
}

fn reject_retention_leakage(
    train: &[KnowledgeSftRecord],
    eval: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let train_records: HashSet<_> = train.iter().map(|record| &record.id).collect();
    let train_episodes: HashSet<_> = train.iter().map(|record| &record.episode_id).collect();
    if let Some(overlap) = eval.iter().find(|record| {
        train_records.contains(&record.id) || train_episodes.contains(&record.episode_id)
    }) {
        return Err(invalid_data(format!(
            "retention eval record/episode {} overlaps training",
            overlap.id
        )));
    }
    Ok(())
}

fn validate_ocr_token_budget(
    tokenizer: &BpeTokenizer,
    cases: &[VisionTextManifest],
    text_context: usize,
) -> Result<(), Box<dyn Error>> {
    let prompt_tokens = tokenizer.encode(grug_train::OCR_PROMPT)?.len();
    for case in cases {
        let required = 1 + prompt_tokens + 1 + tokenizer.encode(&case.transcript)?.len() + 1;
        if required > text_context {
            return Err(invalid_data(format!(
                "case {} needs {required} text positions but visual prefix leaves {text_context}",
                case.case_id
            )));
        }
    }
    Ok(())
}

fn build_visual_batch<B: AutodiffBackend>(
    cases: &[VisionCase],
    indices: &[usize],
    tokenizer: &BpeTokenizer,
    text_context: usize,
    width: usize,
    height: usize,
    device: &B::Device,
) -> Result<(burn::tensor::Tensor<B, 4>, grug_train::train::Batch<B>), Box<dyn Error>> {
    let images: Vec<_> = indices
        .iter()
        .map(|index| cases[*index].normalized_rgb.clone())
        .collect();
    let transcripts: Vec<_> = indices
        .iter()
        .map(|index| cases[*index].manifest.transcript.clone())
        .collect();
    Ok((
        rgb_batch_to_nchw::<B>(&images, width, height, device)?,
        build_vision_ocr_batch::<B, _>(tokenizer, &transcripts, text_context, device)?,
    ))
}

fn cyclic_batch_indices(order: &[usize], batch: usize, batch_size: usize) -> Vec<usize> {
    (0..batch_size)
        .map(|offset| order[(batch * batch_size + offset) % order.len()])
        .collect()
}

fn select_semantic_groups(cases: &[VisionCase], groups: usize) -> Vec<&VisionCase> {
    let mut selected = BTreeSet::new();
    for case in cases {
        if selected.len() >= groups && !selected.contains(&case.manifest.semantic_id) {
            continue;
        }
        selected.insert(case.manifest.semantic_id.clone());
    }
    cases
        .iter()
        .filter(|case| selected.contains(&case.manifest.semantic_id))
        .collect()
}

fn evenly_spaced(records: &[KnowledgeSftRecord], limit: usize) -> Vec<KnowledgeSftRecord> {
    if records.len() <= limit {
        return records.to_vec();
    }
    (0..limit)
        .map(|index| records[index * records.len() / limit].clone())
        .collect()
}

fn compact_optimizer<B: AutodiffBackend + AttentionBackend>()
-> OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>
where
    B::InnerBackend: AttentionBackend,
{
    AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init::<B, CompactVisionTransformer<B>>()
}

fn cosine_learning_rate(step: usize, steps: usize, peak: f64, minimum: f64) -> f64 {
    if steps <= 1 {
        return minimum;
    }
    let progress = step.min(steps - 1) as f64 / (steps - 1) as f64;
    minimum + 0.5 * (peak - minimum) * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn update_was_skipped(loss: f32, grad_norm: f32, args: &Args) -> bool {
    !loss.is_finite()
        || loss > args.loss_cap
        || !grad_norm.is_finite()
        || grad_norm > args.grad_norm_cap
}

fn log_update(path: &Path, row: &UpdateLog<'_>) -> Result<(), Box<dyn Error>> {
    let mut writer = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut writer, row)?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_checkpoint<B: AutodiffBackend + AttentionBackend>(
    args: &Args,
    model: &CompactVisionTransformer<B>,
    optimizer: &OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    stage: &str,
    update: usize,
    stem_updates: usize,
    joint_vision_updates: usize,
    joint_text_updates: usize,
    skipped_updates: usize,
    parameter_count: usize,
    complete: bool,
) -> Result<(), Box<dyn Error>>
where
    B::InnerBackend: AttentionBackend,
{
    if !complete && update % args.checkpoint_every != 0 {
        return Ok(());
    }
    save_checkpoint(
        args,
        model,
        optimizer,
        stage,
        update,
        stem_updates,
        joint_vision_updates,
        joint_text_updates,
        skipped_updates,
        parameter_count,
        complete,
    )?;
    Ok(())
}

fn verify_resume_inputs(
    args: &Args,
    checkpoint: &Path,
    expected_run: &Value,
    base: &VerifiedTextBase,
) -> Result<CheckpointMetadata, Box<dyn Error>> {
    let artifacts = args.artifacts_dir.canonicalize().map_err(|error| {
        invalid_data(format!(
            "resume artifacts directory {} is unavailable: {error}",
            args.artifacts_dir.display()
        ))
    })?;
    let checkpoints = artifacts.join("checkpoints").canonicalize()?;
    let checkpoint = checkpoint.canonicalize()?;
    if checkpoint.parent() != Some(checkpoints.as_path()) {
        return Err(invalid_data(
            "--resume must name one direct checkpoint child inside --artifacts-dir",
        ));
    }
    if ["multimodal-active.json", "completion.json"]
        .iter()
        .any(|name| artifacts.join(name).exists())
    {
        return Err(invalid_data(
            "a completed or activated multimodal run cannot be resumed",
        ));
    }
    let stored_run: Value =
        serde_json::from_reader(BufReader::new(File::open(artifacts.join("run.json"))?))?;
    if &stored_run != expected_run {
        return Err(invalid_data(
            "resume arguments do not exactly reproduce the original run descriptor",
        ));
    }
    verify_hash(
        &artifacts.join("tokenizer.json"),
        &base.artifact.tokenizer_hash,
    )?;
    verify_hash(
        &artifacts.join("contracts.json"),
        &fingerprint_artifact_file(&base.contracts_path)?,
    )?;
    let metadata: CheckpointMetadata =
        serde_json::from_reader(BufReader::new(File::open(checkpoint.join("meta.json"))?))?;
    let expected_directory_name = if metadata.complete {
        format!("candidate_update_{:08}", metadata.update)
    } else {
        format!("update_{:08}", metadata.update)
    };
    if metadata.schema_version != 1
        || metadata.architecture != "compact_native_ocr_shared_text_decoder_v1"
        || metadata.backend != BACKEND_NAME
        || metadata.complete != (metadata.stage == "complete_candidate")
        || !matches!(
            metadata.stage.as_str(),
            "stem_warmup_text_frozen" | "joint" | "complete_candidate"
        )
        || metadata.model_hash != fingerprint_artifact_file(checkpoint.join("model.mpk"))?
        || metadata.optimizer_hash != fingerprint_artifact_file(checkpoint.join("optim.mpk"))?
        || checkpoint.file_name().and_then(|name| name.to_str())
            != Some(expected_directory_name.as_str())
    {
        return Err(invalid_data(
            "resume checkpoint metadata, hashes, backend, or directory identity is invalid",
        ));
    }
    if artifacts.join("multimodal-candidate.json").exists() && !metadata.complete {
        return Err(invalid_data(
            "a written multimodal candidate can only be recovered from its complete checkpoint",
        ));
    }
    Ok(metadata)
}

fn validate_checkpoint_counters(
    metadata: Option<&CheckpointMetadata>,
    stem_target: usize,
    joint_vision_target: usize,
    retention_every: usize,
) -> Result<(), Box<dyn Error>> {
    let Some(metadata) = metadata else {
        return Ok(());
    };
    let counted_updates = metadata
        .stem_updates
        .checked_add(metadata.joint_vision_updates)
        .and_then(|value| value.checked_add(metadata.joint_text_updates));
    if counted_updates != Some(metadata.update) || metadata.skipped_updates > metadata.update {
        return Err(invalid_data(
            "resume checkpoint update counters are internally inconsistent",
        ));
    }
    match metadata.stage.as_str() {
        "stem_warmup_text_frozen"
            if metadata.stem_updates <= stem_target
                && metadata.joint_vision_updates == 0
                && metadata.joint_text_updates == 0 => {}
        "joint"
            if metadata.stem_updates == stem_target
                && metadata.joint_vision_updates <= joint_vision_target
                && metadata.joint_text_updates
                    == metadata.joint_vision_updates / retention_every => {}
        "complete_candidate"
            if metadata.stem_updates == stem_target
                && metadata.joint_vision_updates == joint_vision_target
                && metadata.joint_text_updates
                    == metadata.joint_vision_updates / retention_every => {}
        _ => {
            return Err(invalid_data(
                "resume checkpoint stage counters do not match this training schedule",
            ));
        }
    }
    Ok(())
}

fn load_multimodal_checkpoint<B: AutodiffBackend + AttentionBackend>(
    directory: &Path,
    model: CompactVisionTransformer<B>,
    optimizer: OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    device: &B::Device,
    metadata: &CheckpointMetadata,
) -> Result<
    (
        CompactVisionTransformer<B>,
        OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    ),
    Box<dyn Error>,
>
where
    B::InnerBackend: AttentionBackend,
{
    verify_hash(&directory.join("model.mpk"), &metadata.model_hash)?;
    verify_hash(&directory.join("optim.mpk"), &metadata.optimizer_hash)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = model.load_file(directory.join("model"), &recorder, device)?;
    let optimizer_record = recorder.load(directory.join("optim"), device)?;
    Ok((model, optimizer.load_record(optimizer_record)))
}

#[allow(clippy::too_many_arguments)]
fn save_checkpoint<B: AutodiffBackend + AttentionBackend>(
    args: &Args,
    model: &CompactVisionTransformer<B>,
    optimizer: &OptimizerAdaptor<Adam, CompactVisionTransformer<B>, B>,
    stage: &str,
    update: usize,
    stem_updates: usize,
    joint_vision_updates: usize,
    joint_text_updates: usize,
    skipped_updates: usize,
    parameter_count: usize,
    complete: bool,
) -> Result<PathBuf, Box<dyn Error>>
where
    B::InnerBackend: AttentionBackend,
{
    let directory = args.artifacts_dir.join("checkpoints").join(if complete {
        format!("candidate_update_{update:08}")
    } else {
        format!("update_{update:08}")
    });
    let parent = directory
        .parent()
        .ok_or_else(|| invalid_data("checkpoint has no parent directory"))?;
    std::fs::create_dir_all(parent)?;
    let directory_name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data("checkpoint directory name is not valid UTF-8"))?;
    let temporary = parent.join(format!(".{directory_name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(invalid_data(format!(
            "temporary checkpoint path already exists: {}",
            temporary.display()
        )));
    }
    std::fs::create_dir(&temporary)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(temporary.join("model"), &recorder)?;
    recorder.record(optimizer.to_record(), temporary.join("optim"))?;
    let model_hash = fingerprint_artifact_file(temporary.join("model.mpk"))?;
    let optimizer_hash = fingerprint_artifact_file(temporary.join("optim.mpk"))?;
    write_json_atomic(
        &temporary.join("meta.json"),
        &CheckpointMetadata {
            schema_version: 1,
            architecture: "compact_native_ocr_shared_text_decoder_v1".into(),
            backend: BACKEND_NAME.into(),
            stage: stage.into(),
            update,
            stem_updates,
            joint_vision_updates,
            joint_text_updates,
            skipped_updates,
            parameter_count,
            complete,
            model_hash,
            optimizer_hash,
        },
    )?;
    std::fs::rename(&temporary, &directory)?;
    Ok(directory)
}

fn resolve_child(bundle: &Path, child: impl AsRef<Path>) -> Result<PathBuf, Box<dyn Error>> {
    let child = child.as_ref();
    if child.as_os_str().is_empty()
        || child.is_absolute()
        || !child
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(invalid_data(format!(
            "unsafe bundle path {}",
            child.display()
        )));
    }
    Ok(bundle.join(child))
}

fn verify_hash(path: &Path, expected: &str) -> Result<(), Box<dyn Error>> {
    let actual = fingerprint_artifact_file(path)?;
    if actual != expected {
        return Err(invalid_data(format!(
            "hash mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )));
    }
    Ok(())
}

fn copy_verified_file(source: &Path, target: &Path, expected: &str) -> Result<(), Box<dyn Error>> {
    copy_file_atomic(source, target)?;
    verify_hash(target, expected)
}

fn copy_file_atomic(source: &Path, target: &Path) -> Result<(), Box<dyn Error>> {
    let temporary = target.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::copy(source, &temporary)?;
    std::fs::rename(temporary, target)?;
    Ok(())
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn write_or_verify_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return write_json_atomic(path, value);
    }
    let actual: Value = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let expected = serde_json::to_value(value)?;
    if actual != expected {
        return Err(invalid_data(format!(
            "existing resumable output {} differs from the recomputed value",
            path.display()
        )));
    }
    Ok(())
}

fn write_json_lines<T: Serialize>(path: &Path, values: &[T]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_or_verify_json_lines<T: Serialize>(
    path: &Path,
    values: &[T],
) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return write_json_lines(path, values);
    }
    let mut actual = Vec::new();
    for (index, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            return Err(invalid_data(format!(
                "existing JSONL output {} contains a blank line at {}",
                path.display(),
                index + 1
            )));
        }
        actual.push(serde_json::from_str::<Value>(&line).map_err(|error| {
            invalid_data(format!(
                "existing JSONL output {} has invalid row {}: {error}",
                path.display(),
                index + 1
            ))
        })?);
    }
    let expected: Vec<Value> = values
        .iter()
        .map(serde_json::to_value)
        .collect::<Result<_, _>>()?;
    if actual != expected {
        return Err(invalid_data(format!(
            "existing resumable output {} differs from the recomputed rows",
            path.display()
        )));
    }
    Ok(())
}

fn language_label(case: &VisionTextManifest) -> String {
    match json_label(case.language).as_str() {
        "english" => "en".into(),
        "german" => "de".into(),
        "french" => "fr".into(),
        other => other.into(),
    }
}

fn json_label<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_metadata(
        stage: &str,
        stem_updates: usize,
        joint_vision_updates: usize,
        joint_text_updates: usize,
    ) -> CheckpointMetadata {
        CheckpointMetadata {
            schema_version: 1,
            architecture: "compact_native_ocr_shared_text_decoder_v1".into(),
            backend: BACKEND_NAME.into(),
            stage: stage.into(),
            update: stem_updates + joint_vision_updates + joint_text_updates,
            stem_updates,
            joint_vision_updates,
            joint_text_updates,
            skipped_updates: 0,
            parameter_count: 42,
            complete: stage == "complete_candidate",
            model_hash: "blake3:model".into(),
            optimizer_hash: "blake3:optimizer".into(),
        }
    }

    #[test]
    fn cosine_schedule_reaches_declared_endpoints() {
        assert_eq!(cosine_learning_rate(0, 10, 1e-3, 1e-5), 1e-3);
        assert!((cosine_learning_rate(9, 10, 1e-3, 1e-5) - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn cyclic_batches_are_total_at_short_tail() {
        assert_eq!(cyclic_batch_indices(&[3, 5, 8], 1, 4), vec![5, 8, 3, 5]);
    }

    #[test]
    fn resume_counters_accept_each_exact_training_stage() {
        let stem = checkpoint_metadata("stem_warmup_text_frozen", 40, 0, 0);
        validate_checkpoint_counters(Some(&stem), 100, 30, 2).unwrap();

        let joint = checkpoint_metadata("joint", 100, 18, 9);
        validate_checkpoint_counters(Some(&joint), 100, 30, 2).unwrap();

        let complete = checkpoint_metadata("complete_candidate", 100, 30, 15);
        validate_checkpoint_counters(Some(&complete), 100, 30, 2).unwrap();
    }

    #[test]
    fn resume_counters_refuse_inconsistent_or_wrong_schedule_state() {
        let mut bad_total = checkpoint_metadata("joint", 100, 18, 9);
        bad_total.update += 1;
        assert!(validate_checkpoint_counters(Some(&bad_total), 100, 30, 2).is_err());

        let wrong_retention = checkpoint_metadata("joint", 100, 18, 8);
        assert!(validate_checkpoint_counters(Some(&wrong_retention), 100, 30, 2).is_err());

        let incomplete_stem = checkpoint_metadata("joint", 99, 18, 9);
        assert!(validate_checkpoint_counters(Some(&incomplete_stem), 100, 30, 2).is_err());

        let early_complete = checkpoint_metadata("complete_candidate", 100, 29, 14);
        assert!(validate_checkpoint_counters(Some(&early_complete), 100, 30, 2).is_err());
    }

    #[test]
    fn resumable_outputs_are_append_only_and_exact() {
        let temp = tempfile::tempdir().unwrap();
        let json = temp.path().join("score.json");
        write_or_verify_json(&json, &serde_json::json!({"score": 1})).unwrap();
        write_or_verify_json(&json, &serde_json::json!({"score": 1})).unwrap();
        assert!(write_or_verify_json(&json, &serde_json::json!({"score": 2})).is_err());

        let jsonl = temp.path().join("predictions.jsonl");
        let rows = [serde_json::json!({"id": 1}), serde_json::json!({"id": 2})];
        write_or_verify_json_lines(&jsonl, &rows).unwrap();
        write_or_verify_json_lines(&jsonl, &rows).unwrap();
        assert!(write_or_verify_json_lines(&jsonl, &[serde_json::json!({"id": 3})]).is_err());
    }
}
