//! Causal-language pretraining for the compact tool-native BabyLLM.
//!
//! This stage consumes an immutable `FoundationCorpusManifest`, trains the
//! definitive byte-level BPE, packs content-addressed u32 token streams, and
//! writes a resumable `foundation-model.json`. `train_knowledge` can then
//! warm-start from this exact checkpoint without resetting the language trunk.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use grug_data::{
    FoundationCorpusManifest, FoundationRecord, FoundationSourceSummary, FoundationSplit,
    FoundationSplitManifest, update_foundation_fingerprint,
};
use grug_train::foundation::{
    FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION, FOUNDATION_MODEL_MANIFEST, FoundationModelArtifact,
    build_causal_batch, causal_batch_loss, foundation_training_languages,
    foundation_training_sources, verify_foundation_bundle,
};
use grug_train::knowledge_train::{
    KnowledgeModelShape, KnowledgeModelSize, fingerprint_artifact_file,
};
use grug_train::train::{init_training, to_eval_model, train_step_guarded};
use grug_train::{BpeTokenizer, checkpoint};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
use grug_train::CpuAutodiffBackend as ActiveBackend;
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
use grug_train::GpuAutodiffBackend as ActiveBackend;

const TOKEN_CACHE_SCHEMA_VERSION: u32 = 1;

/// `--bpe-pretokenizer` choice for a freshly-trained tokenizer (ADR 104 D2).
/// Combined with `--bpe-min-hex-run` into a `grug_train::bpe::PreTokenization`
/// at the `BpeTokenizer::train_with_pretokenization` call site; kept as its
/// own CLI-facing type (rather than exposing `PreTokenization` directly to
/// clap) since the two flags map onto one enum variant's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BpePreTokenizerArg {
    ByteLevel,
    HexAware,
}

impl BpePreTokenizerArg {
    fn into_pretokenization(self, min_hex_run: usize) -> grug_train::bpe::PreTokenization {
        match self {
            BpePreTokenizerArg::ByteLevel => grug_train::bpe::PreTokenization::ByteLevel,
            BpePreTokenizerArg::HexAware => {
                grug_train::bpe::PreTokenization::HexAwareByteLevel { min_hex_run }
            }
        }
    }
}

impl fmt::Display for BpePreTokenizerArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            BpePreTokenizerArg::ByteLevel => "byte-level",
            BpePreTokenizerArg::HexAware => "hex-aware",
        })
    }
}

impl FromStr for BpePreTokenizerArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "byte-level" | "byte_level" | "bytelevel" => Ok(Self::ByteLevel),
            "hex-aware" | "hex_aware" | "hexaware" => Ok(Self::HexAware),
            other => Err(format!(
                "unknown --bpe-pretokenizer {other:?}; expected byte-level or hex-aware"
            )),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "train_foundation")]
struct Args {
    /// Prepared `foundation-corpus.json`.
    #[arg(long)]
    manifest: PathBuf,

    /// Self-contained run directory for tokenizer, token cache and checkpoints.
    #[arg(long)]
    artifacts_dir: PathBuf,

    #[arg(long, default_value = "small")]
    model: KnowledgeModelSize,

    /// Optimizer updates. The packed stream cycles deterministically if this
    /// exceeds one pass; `training_tokens_seen` makes repeats explicit.
    #[arg(long, default_value_t = 10_000)]
    steps: usize,

    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Defaults to 3e-4 for small and 5e-5 for 350m.
    #[arg(long)]
    learning_rate: Option<f64>,

    #[arg(long, default_value_t = 500)]
    warmup_steps: usize,

    #[arg(long, default_value_t = 0.1)]
    lr_min_ratio: f64,

    #[arg(long, default_value_t = 8_000)]
    bpe_vocab_size: usize,

    /// Pre-tokenization scheme for a freshly-trained BPE tokenizer (ADR 104
    /// D2). Ignored when `<artifacts-dir>/tokenizer.json` already exists --
    /// an existing tokenizer's own saved JSON always governs. `hex-aware`
    /// isolates long hex-character runs (see `--bpe-min-hex-run`) and
    /// always splits decimal digits individually, so the model copies
    /// identifiers/content hashes one character per token instead of
    /// merging them inconsistently; `byte-level` keeps the original P1a
    /// scheme for comparison runs.
    #[arg(long, default_value_t = BpePreTokenizerArg::HexAware)]
    bpe_pretokenizer: BpePreTokenizerArg,

    /// Minimum contiguous hex-character run length that `--bpe-pretokenizer
    /// hex-aware` isolates into one token per character. Ignored for
    /// `--bpe-pretokenizer byte-level`.
    #[arg(long, default_value_t = grug_train::bpe::DEFAULT_MIN_HEX_RUN)]
    bpe_min_hex_run: usize,

    /// Deterministic reservoir used to train BPE without retaining a huge
    /// web corpus in memory.
    #[arg(long, default_value_t = 100_000)]
    tokenizer_sample_records: usize,

    #[arg(long, default_value_t = 250)]
    eval_every: usize,

    #[arg(long, default_value_t = 8)]
    eval_batches: usize,

    #[arg(long, default_value_t = 500)]
    ckpt_every: usize,

    #[arg(long, default_value_t = 20.0)]
    loss_cap: f32,

    #[arg(long, default_value_t = 100.0)]
    grad_norm_cap: f32,

    #[arg(long, default_value_t = 500)]
    max_skipped_updates: usize,

    #[arg(long, default_value_t = 51_004)]
    seed: u64,

    /// Exact checkpoint directory named by the existing foundation manifest.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Packed training/eval context in tokens for this run. Defaults to the
    /// model shape's `max_seq_len`; must not exceed it. Lets an 8k-shape
    /// (`small_8k`, RoPE) train its early stages at 2048/4096 on a
    /// memory-bound local GPU (ADR 106, Phase 51 P6) without changing the
    /// checkpointed shape -- the shape, not this flag, governs inference.
    #[arg(long)]
    context: Option<usize>,

    /// Verify/decontamination lineage, train BPE, and materialize the packed
    /// token cache without allocating model weights.
    #[arg(long)]
    validate_only: bool,
}

#[derive(Debug)]
struct CorpusAudit {
    tokenizer_sample: Vec<String>,
}

#[derive(Debug)]
struct SplitAudit {
    ids: HashSet<[u8; 32]>,
    languages: BTreeMap<String, u64>,
    sources: BTreeMap<(String, String, String, String), u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TokenFile {
    file: String,
    file_hash: String,
    tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TokenCacheManifest {
    schema_version: u32,
    tokenizer_hash: String,
    /// Audit trail alongside `tokenizer_hash` for *why* two tokenizers with
    /// different hashes differ (ADR 104 D2): `"byte-level"` or
    /// `"hex-aware"`. Not itself load-bearing for cache validity --
    /// `tokenizer_hash` already changes if the pre-tokenizer changes -- so
    /// old caches predating this field default to `"byte-level"`, which is
    /// what they were actually built with.
    #[serde(default = "default_bpe_pretokenizer_label")]
    bpe_pretokenizer: String,
    /// Paired with `bpe_pretokenizer`; meaningless (and defaulted to 0) for
    /// `"byte-level"` caches.
    #[serde(default)]
    bpe_min_hex_run: usize,
    training_corpus_hash: String,
    evaluation_corpus_hash: String,
    training: TokenFile,
    evaluation: TokenFile,
}

fn default_bpe_pretokenizer_label() -> String {
    "byte-level".to_string()
}

#[derive(Serialize)]
struct RunDescriptor<'a> {
    schema_version: u32,
    manifest: &'a Path,
    manifest_hash: &'a str,
    model_size: KnowledgeModelSize,
    shape: &'a KnowledgeModelShape,
    parameter_count: usize,
    tokenizer_hash: &'a str,
    bpe_pretokenizer: String,
    bpe_min_hex_run: usize,
    token_cache_hash: &'a str,
    steps: usize,
    batch_size: usize,
    learning_rate: f64,
    warmup_steps: usize,
    lr_min_ratio: f64,
    loss_cap: f32,
    grad_norm_cap: f32,
    seed: u64,
}

#[derive(Serialize)]
struct TrainLogRow {
    step: usize,
    loss: f32,
    grad_norm: f32,
    learning_rate: f64,
    update_skipped: bool,
    tokens_seen: u64,
    step_ms: f64,
}

#[derive(Serialize)]
struct EvalLogRow {
    step: usize,
    loss: f32,
    perplexity: f64,
    batches: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    validate_args(&args)?;

    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let device = {
        let device = grug_train::gpu_device();
        println!("train_foundation: CubeCL GPU backend, device={device:?}");
        device
    };
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let device = {
        println!("train_foundation: CPU backend (burn-ndarray)");
        Default::default()
    };
    run::<ActiveBackend>(args, device)
}

fn run<B: AutodiffBackend>(args: Args, device: B::Device) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&args.artifacts_dir)?;
    let manifest: FoundationCorpusManifest =
        serde_json::from_reader(BufReader::new(File::open(&args.manifest)?))?;
    manifest.verify()?;
    let manifest_hash = fingerprint_artifact_file(&args.manifest)?;
    println!(
        "train_foundation: auditing train={} eval={} languages={:?}",
        manifest.training.records, manifest.evaluation.records, manifest.languages
    );
    let audit = audit_corpus(
        &args.manifest,
        &manifest,
        args.tokenizer_sample_records,
        args.seed,
    )?;

    let bundled_manifest = args.artifacts_dir.join("foundation-corpus.json");
    copy_verified(&args.manifest, &bundled_manifest, &manifest_hash)?;
    let tokenizer_path = args.artifacts_dir.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        BpeTokenizer::load(&tokenizer_path)?
    } else {
        if args.resume.is_some() {
            return Err(invalid_data(format!(
                "resume requires the original tokenizer at {}",
                tokenizer_path.display()
            )));
        }
        let pretokenization = args
            .bpe_pretokenizer
            .into_pretokenization(args.bpe_min_hex_run);
        println!(
            "train_foundation: training BPE ceiling={} pretokenizer={pretokenization} from {}-record reservoir",
            args.bpe_vocab_size,
            audit.tokenizer_sample.len()
        );
        let tokenizer = BpeTokenizer::train_with_pretokenization(
            audit.tokenizer_sample,
            args.bpe_vocab_size,
            pretokenization,
        )?;
        tokenizer.save(&tokenizer_path)?;
        tokenizer
    };
    let tokenizer_hash = fingerprint_artifact_file(&tokenizer_path)?;
    let token_cache = load_or_build_token_cache(
        &args.artifacts_dir,
        &args.manifest,
        &manifest,
        &tokenizer,
        &tokenizer_hash,
        &args.bpe_pretokenizer.to_string(),
        args.bpe_min_hex_run,
    )?;
    let token_cache_path = args.artifacts_dir.join("token-cache.json");
    let token_cache_hash = fingerprint_artifact_file(&token_cache_path)?;
    let shape = KnowledgeModelShape::from_config(&args.model.config(tokenizer.vocab_size()));
    let context = match args.context {
        None => shape.max_seq_len,
        Some(context) if context >= 2 && context <= shape.max_seq_len => context,
        Some(context) => {
            return Err(invalid_data(format!(
                "--context {context} must be in 2..={} for model {}",
                shape.max_seq_len, args.model
            )));
        }
    };
    println!(
        "train_foundation: tokenizer={} vocab={} bpe_pretokenizer={} bpe_min_hex_run={} \
         tokens train={} eval={} context={}",
        tokenizer_hash,
        tokenizer.vocab_size(),
        args.bpe_pretokenizer,
        args.bpe_min_hex_run,
        token_cache.training.tokens,
        token_cache.evaluation.tokens,
        context
    );
    if token_cache.training.tokens <= context as u64 + 1
        || token_cache.evaluation.tokens <= context as u64 + 1
    {
        return Err(invalid_data(format!(
            "packed corpus is too small for context {} (train={}, eval={})",
            context, token_cache.training.tokens, token_cache.evaluation.tokens
        )));
    }
    if args.validate_only {
        println!(
            "train_foundation: validation complete manifest={} token_cache={} shape={shape:?}",
            manifest_hash, token_cache_hash
        );
        return Ok(());
    }

    if args.resume.is_none() && args.artifacts_dir.join("run.json").exists() {
        return Err(invalid_data(
            "run.json already exists; pass --resume for an existing run or use a fresh artifacts directory",
        ));
    }
    B::seed(&device, args.seed);
    let config = shape.config();
    let (mut model, mut optimizer) = init_training::<B>(&config, &device);
    let parameter_count = model.num_params();
    let learning_rate = args.learning_rate.unwrap_or(match args.model {
        KnowledgeModelSize::Small
        | KnowledgeModelSize::Small8k
        | KnowledgeModelSize::SmallRope4k => 3e-4,
        KnowledgeModelSize::Model350m | KnowledgeModelSize::Model350m8k => 5e-5,
    });

    let mut start_step = 0usize;
    if let Some(resume) = &args.resume {
        let verified = verify_foundation_bundle(&args.artifacts_dir)?;
        if verified.artifact().training_corpus_hash != manifest.training.fingerprint
            || verified.artifact().evaluation_corpus_hash != manifest.evaluation.fingerprint
            || verified.artifact().tokenizer_hash != tokenizer_hash
            || verified.artifact().shape != shape
        {
            return Err(invalid_data(
                "existing foundation artifact does not match this corpus/tokenizer/model",
            ));
        }
        let declared = args.artifacts_dir.join(&verified.artifact().checkpoint_dir);
        if resume.canonicalize()? != declared.canonicalize()? {
            return Err(invalid_data(format!(
                "--resume {} is not the checkpoint bound by {}",
                resume.display(),
                FOUNDATION_MODEL_MANIFEST
            )));
        }
        let (loaded_model, loaded_optimizer, step) =
            checkpoint::load_checkpoint(resume, model, optimizer, &device)?;
        model = loaded_model;
        optimizer = loaded_optimizer;
        start_step = step + 1;
    }
    if start_step >= args.steps {
        return Err(invalid_data(format!(
            "resume starts at step {start_step}, but --steps is {}",
            args.steps
        )));
    }

    write_json_pretty_atomic(
        &args.artifacts_dir.join("run.json"),
        &RunDescriptor {
            schema_version: 1,
            manifest: &args.manifest,
            manifest_hash: &manifest_hash,
            model_size: args.model,
            shape: &shape,
            parameter_count,
            tokenizer_hash: &tokenizer_hash,
            bpe_pretokenizer: args.bpe_pretokenizer.to_string(),
            bpe_min_hex_run: args.bpe_min_hex_run,
            token_cache_hash: &token_cache_hash,
            steps: args.steps,
            batch_size: args.batch_size,
            learning_rate,
            warmup_steps: args.warmup_steps,
            lr_min_ratio: args.lr_min_ratio,
            loss_cap: args.loss_cap,
            grad_norm_cap: args.grad_norm_cap,
            seed: args.seed,
        },
    )?;

    println!(
        "train_foundation: model={} params={} steps={} batch={} lr={learning_rate:.2e}",
        args.model, parameter_count, args.steps, args.batch_size
    );
    let train_path = resolve_artifact_child(&args.artifacts_dir, &token_cache.training.file)?;
    let eval_path = resolve_artifact_child(&args.artifacts_dir, &token_cache.evaluation.file)?;
    let mut train_file = File::open(train_path)?;
    let mut eval_file = File::open(eval_path)?;
    let train_log = args.artifacts_dir.join("train.jsonl");
    let eval_log = args.artifacts_dir.join("eval_causal.jsonl");
    let started = Instant::now();
    let tokens_per_step = (args.batch_size * context) as u64;
    let mut loss_ema = f32::NAN;
    let mut grad_ema = f32::NAN;
    let mut skipped = 0usize;
    let mut final_loss = f32::NAN;

    for step in start_step..args.steps {
        let sequences = read_step_sequences(
            &mut train_file,
            token_cache.training.tokens,
            step,
            args.batch_size,
            context,
        )?;
        let batch = build_causal_batch::<B>(&sequences, tokenizer.pad_id(), context, &device)?;
        let lr = learning_rate_at_step(
            step,
            args.steps,
            learning_rate,
            args.warmup_steps,
            args.lr_min_ratio,
        );
        let dynamic_loss_cap = if step > 100 && loss_ema.is_finite() {
            args.loss_cap.min((loss_ema * 4.0).max(loss_ema + 2.0))
        } else {
            args.loss_cap
        };
        let dynamic_grad_cap = if step > 100 && grad_ema.is_finite() {
            args.grad_norm_cap.min((grad_ema * 5.0).max(grad_ema + 1.0))
        } else {
            args.grad_norm_cap
        };
        let step_started = Instant::now();
        let (next, loss, grad_norm) = train_step_guarded(
            model,
            &mut optimizer,
            &batch,
            lr,
            dynamic_loss_cap,
            dynamic_grad_cap,
        );
        model = next;
        let update_skipped = !loss.is_finite()
            || loss > dynamic_loss_cap
            || !grad_norm.is_finite()
            || grad_norm > dynamic_grad_cap;
        if update_skipped {
            skipped += 1;
            if skipped > args.max_skipped_updates {
                return Err(invalid_data(format!(
                    "aborting after {skipped} guarded updates; no newer foundation manifest was written"
                )));
            }
        } else {
            final_loss = loss;
            loss_ema = ema(loss_ema, loss);
            grad_ema = ema(grad_ema, grad_norm);
        }
        let tokens_seen = (step as u64 + 1) * tokens_per_step;
        append_json_line(
            &train_log,
            &TrainLogRow {
                step,
                loss,
                grad_norm,
                learning_rate: lr,
                update_skipped,
                tokens_seen,
                step_ms: step_started.elapsed().as_secs_f64() * 1_000.0,
            },
        )?;
        if step % 10 == 0 || step + 1 == args.steps || update_skipped {
            println!(
                "step {step:6}: loss={loss:.4} grad={grad_norm:.3} lr={lr:.2e} tokens={tokens_seen}{}",
                if update_skipped { " SKIPPED" } else { "" }
            );
        }

        if (step + 1) % args.eval_every == 0 || step + 1 == args.steps {
            let loss = evaluate_causal(
                &model,
                &tokenizer,
                &mut eval_file,
                token_cache.evaluation.tokens,
                args.eval_batches,
                args.batch_size,
                context,
                &device,
            )?;
            let perplexity = f64::from(loss).min(80.0).exp();
            println!("  eval@{step}: loss={loss:.4} perplexity={perplexity:.2}");
            append_json_line(
                &eval_log,
                &EvalLogRow {
                    step,
                    loss,
                    perplexity,
                    batches: args.eval_batches,
                },
            )?;
        }

        if (step + 1) % args.ckpt_every == 0 || step + 1 == args.steps {
            let relative_checkpoint = format!("checkpoints/step_{step:08}");
            let checkpoint_dir = args.artifacts_dir.join(&relative_checkpoint);
            checkpoint::save_checkpoint(&checkpoint_dir, step, &model, &optimizer)?;
            let artifact = FoundationModelArtifact {
                schema_version: FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION,
                model_size: args.model,
                shape: shape.clone(),
                parameter_count,
                tokenizer_file: "tokenizer.json".into(),
                tokenizer_hash: tokenizer_hash.clone(),
                corpus_manifest_file: "foundation-corpus.json".into(),
                corpus_manifest_hash: manifest_hash.clone(),
                training_corpus_hash: manifest.training.fingerprint.clone(),
                evaluation_corpus_hash: manifest.evaluation.fingerprint.clone(),
                training_records: manifest.training.records,
                evaluation_records: manifest.evaluation.records,
                corpus_tokens: token_cache.training.tokens,
                evaluation_tokens: token_cache.evaluation.tokens,
                training_tokens_seen: (step as u64 + 1) * tokens_per_step,
                languages: foundation_training_languages(&manifest),
                sources: foundation_training_sources(&manifest),
                checkpoint_dir: relative_checkpoint,
                checkpoint_step: step,
                checkpoint_model_hash: fingerprint_artifact_file(checkpoint_dir.join("model.mpk"))?,
                checkpoint_optimizer_hash: Some(fingerprint_artifact_file(
                    checkpoint_dir.join("optim.mpk"),
                )?),
            };
            artifact.verify().map_err(invalid_data)?;
            write_json_pretty_atomic(
                &args.artifacts_dir.join(FOUNDATION_MODEL_MANIFEST),
                &artifact,
            )?;
            println!(
                "  checkpoint + foundation model -> {}",
                checkpoint_dir.display()
            );
        }
    }
    println!(
        "train_foundation: done elapsed={:.1}s final_loss={final_loss:.4} skipped={skipped}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn validate_args(args: &Args) -> Result<(), Box<dyn Error>> {
    if args.steps == 0
        || args.batch_size == 0
        || args.tokenizer_sample_records == 0
        || args.eval_every == 0
        || args.eval_batches == 0
        || args.ckpt_every == 0
    {
        return Err(invalid_data("count arguments must be positive"));
    }
    if args.bpe_vocab_size < 260 {
        return Err(invalid_data("--bpe-vocab-size must be at least 260"));
    }
    if args.bpe_min_hex_run == 0 {
        return Err(invalid_data("--bpe-min-hex-run must be at least 1"));
    }
    if args.learning_rate.is_some_and(|value| value <= 0.0)
        || !(0.0..=1.0).contains(&args.lr_min_ratio)
    {
        return Err(invalid_data(
            "learning rate must be positive and --lr-min-ratio in 0..=1",
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
    Ok(())
}

fn audit_corpus(
    manifest_path: &Path,
    manifest: &FoundationCorpusManifest,
    sample_size: usize,
    seed: u64,
) -> Result<CorpusAudit, Box<dyn Error>> {
    let root = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let eval = audit_split(
        root,
        FoundationSplit::Evaluation,
        &manifest.evaluation,
        None,
    )?;
    let mut sampler = ReservoirSampler::new(sample_size, seed);
    let train = audit_split(
        root,
        FoundationSplit::Training,
        &manifest.training,
        Some(&mut sampler),
    )?;
    if let Some(id) = train.ids.intersection(&eval.ids).next() {
        return Err(invalid_data(format!(
            "foundation train/evaluation id leakage detected (id hash {})",
            hex_hash(id)
        )));
    }

    let mut languages = train.languages;
    for (language, count) in eval.languages {
        *languages.entry(language).or_default() += count;
    }
    if languages != manifest.languages {
        return Err(invalid_data(format!(
            "foundation language counts differ from manifest: actual={languages:?} declared={:?}",
            manifest.languages
        )));
    }
    let declared_sources: BTreeMap<_, _> = manifest
        .sources
        .iter()
        .map(|source| (source_key(source), source.records))
        .collect();
    let mut actual_sources: BTreeMap<_, _> = train
        .sources
        .into_iter()
        .map(|(key, count)| ((FoundationSplit::Training, key), count))
        .collect();
    actual_sources.extend(
        eval.sources
            .into_iter()
            .map(|(key, count)| ((FoundationSplit::Evaluation, key), count)),
    );
    if actual_sources != declared_sources {
        return Err(invalid_data(
            "foundation source/revision/license/language counts differ from manifest",
        ));
    }
    Ok(CorpusAudit {
        tokenizer_sample: sampler.finish(),
    })
}

fn audit_split(
    root: &Path,
    split: FoundationSplit,
    declared: &FoundationSplitManifest,
    mut sampler: Option<&mut ReservoirSampler>,
) -> Result<SplitAudit, Box<dyn Error>> {
    let mut records = 0u64;
    let mut jsonl_bytes = 0u64;
    let mut fingerprint = blake3::Hasher::new();
    let mut ids = HashSet::new();
    let mut languages = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for shard in &declared.shards {
        let path = resolve_manifest_child(root, &shard.file)?;
        let mut reader = BufReader::new(File::open(&path)?);
        let mut file_hash = blake3::Hasher::new();
        let mut shard_records = 0u64;
        let mut shard_bytes = 0u64;
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            file_hash.update(&line);
            shard_bytes += read as u64;
            let payload = line.strip_suffix(b"\n").unwrap_or(&line);
            let payload = payload.strip_suffix(b"\r").unwrap_or(payload);
            if payload.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let record: FoundationRecord = serde_json::from_slice(payload).map_err(|error| {
                invalid_data(format!(
                    "{} record {} is invalid JSON: {error}",
                    path.display(),
                    shard_records + 1
                ))
            })?;
            record.validate()?;
            let id_hash = *blake3::hash(record.id.as_bytes()).as_bytes();
            if !ids.insert(id_hash) {
                return Err(invalid_data(format!(
                    "duplicate {} id {:?}",
                    split.as_str(),
                    record.id
                )));
            }
            update_foundation_fingerprint(&mut fingerprint, &record)?;
            *languages.entry(record.language.clone()).or_default() += 1;
            *sources
                .entry((
                    record.source.clone(),
                    record.source_revision.clone(),
                    record.license.clone(),
                    record.language.clone(),
                ))
                .or_default() += 1;
            if let Some(sampler) = sampler.as_deref_mut() {
                sampler.observe(record.text);
            }
            records += 1;
            shard_records += 1;
        }
        let actual_hash = format!("blake3:{}", file_hash.finalize().to_hex());
        if actual_hash != shard.file_hash
            || shard_records != shard.records
            || shard_bytes != shard.jsonl_bytes
        {
            return Err(invalid_data(format!(
                "shard {} differs from manifest: hash={actual_hash} records={shard_records} bytes={shard_bytes}",
                path.display()
            )));
        }
        jsonl_bytes += shard_bytes;
    }
    let fingerprint = format!("blake3:{}", fingerprint.finalize().to_hex());
    if records != declared.records
        || jsonl_bytes != declared.jsonl_bytes
        || fingerprint != declared.fingerprint
    {
        return Err(invalid_data(format!(
            "{} split differs from manifest: fingerprint={fingerprint} records={records} bytes={jsonl_bytes}",
            split.as_str()
        )));
    }
    Ok(SplitAudit {
        ids,
        languages,
        sources,
    })
}

struct ReservoirSampler {
    values: Vec<String>,
    capacity: usize,
    seen: u64,
    rng: StdRng,
}

impl ReservoirSampler {
    fn new(capacity: usize, seed: u64) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity,
            seen: 0,
            rng: StdRng::seed_from_u64(seed ^ 0xb9e3_7712_104f_a11e),
        }
    }

    fn observe(&mut self, value: String) {
        self.seen += 1;
        if self.values.len() < self.capacity {
            self.values.push(value);
            return;
        }
        let index = self.rng.gen_range(0..self.seen);
        if index < self.capacity as u64 {
            self.values[index as usize] = value;
        }
    }

    fn finish(self) -> Vec<String> {
        self.values
    }
}

fn load_or_build_token_cache(
    artifacts_dir: &Path,
    source_manifest_path: &Path,
    manifest: &FoundationCorpusManifest,
    tokenizer: &BpeTokenizer,
    tokenizer_hash: &str,
    bpe_pretokenizer: &str,
    bpe_min_hex_run: usize,
) -> Result<TokenCacheManifest, Box<dyn Error>> {
    let cache_path = artifacts_dir.join("token-cache.json");
    if cache_path.exists() {
        let cache: TokenCacheManifest =
            serde_json::from_reader(BufReader::new(File::open(&cache_path)?))?;
        verify_token_cache(artifacts_dir, manifest, tokenizer_hash, &cache)?;
        return Ok(cache);
    }
    let token_dir = artifacts_dir.join("tokens");
    std::fs::create_dir_all(&token_dir)?;
    reject_partial_token_cache(&token_dir)?;
    let root = source_manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let training = tokenize_split(
        root,
        &manifest.training,
        tokenizer,
        &token_dir.join("training.u32le.tmp"),
        "tokens/training.u32le",
    )?;
    let evaluation = tokenize_split(
        root,
        &manifest.evaluation,
        tokenizer,
        &token_dir.join("evaluation.u32le.tmp"),
        "tokens/evaluation.u32le",
    )?;
    let cache = TokenCacheManifest {
        schema_version: TOKEN_CACHE_SCHEMA_VERSION,
        tokenizer_hash: tokenizer_hash.into(),
        bpe_pretokenizer: bpe_pretokenizer.into(),
        bpe_min_hex_run,
        training_corpus_hash: manifest.training.fingerprint.clone(),
        evaluation_corpus_hash: manifest.evaluation.fingerprint.clone(),
        training,
        evaluation,
    };
    write_json_pretty_atomic(&cache_path, &cache)?;
    verify_token_cache(artifacts_dir, manifest, tokenizer_hash, &cache)?;
    Ok(cache)
}

fn reject_partial_token_cache(token_dir: &Path) -> Result<(), Box<dyn Error>> {
    let partials = [
        token_dir.join("training.u32le"),
        token_dir.join("training.u32le.tmp"),
        token_dir.join("evaluation.u32le"),
        token_dir.join("evaluation.u32le.tmp"),
    ]
    .into_iter()
    .filter(|path| path.exists())
    .map(|path| path.display().to_string())
    .collect::<Vec<_>>();
    if !partials.is_empty() {
        return Err(invalid_data(format!(
            "partial token cache exists without token-cache.json; quarantine or remove only these incomplete run artifacts: {}",
            partials.join(", ")
        )));
    }
    Ok(())
}

fn tokenize_split(
    root: &Path,
    split: &FoundationSplitManifest,
    tokenizer: &BpeTokenizer,
    temporary: &Path,
    relative_final: &str,
) -> Result<TokenFile, Box<dyn Error>> {
    let final_path = temporary
        .parent()
        .ok_or("temporary token cache has no parent")?
        .join(
            Path::new(relative_final)
                .file_name()
                .ok_or("token cache final path has no filename")?,
        );
    if temporary.exists() || final_path.exists() {
        return Err(invalid_data(format!(
            "token cache destination is not empty: temporary={} final={}",
            temporary.display(),
            final_path.display()
        )));
    }
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temporary)?,
    );
    let mut hasher = blake3::Hasher::new();
    let mut tokens = 0u64;
    for shard in &split.shards {
        let path = resolve_manifest_child(root, &shard.file)?;
        let reader = BufReader::new(File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: FoundationRecord = serde_json::from_str(&line)?;
            write_token(tokenizer.bos_id(), &mut output, &mut hasher)?;
            tokens += 1;
            for token in tokenizer.encode(&record.text)? {
                write_token(token, &mut output, &mut hasher)?;
                tokens += 1;
            }
            write_token(tokenizer.eos_id(), &mut output, &mut hasher)?;
            tokens += 1;
        }
    }
    output.flush()?;
    let file = output.into_inner()?;
    file.sync_all()?;
    std::fs::rename(temporary, final_path)?;
    Ok(TokenFile {
        file: relative_final.into(),
        file_hash: format!("blake3:{}", hasher.finalize().to_hex()),
        tokens,
    })
}

fn write_token(
    token: u32,
    writer: &mut impl Write,
    hasher: &mut blake3::Hasher,
) -> Result<(), std::io::Error> {
    let bytes = token.to_le_bytes();
    writer.write_all(&bytes)?;
    hasher.update(&bytes);
    Ok(())
}

fn verify_token_cache(
    root: &Path,
    corpus: &FoundationCorpusManifest,
    tokenizer_hash: &str,
    cache: &TokenCacheManifest,
) -> Result<(), Box<dyn Error>> {
    if cache.schema_version != TOKEN_CACHE_SCHEMA_VERSION
        || cache.tokenizer_hash != tokenizer_hash
        || cache.training_corpus_hash != corpus.training.fingerprint
        || cache.evaluation_corpus_hash != corpus.evaluation.fingerprint
    {
        return Err(invalid_data(
            "token cache identity does not match corpus/tokenizer",
        ));
    }
    for token_file in [&cache.training, &cache.evaluation] {
        if token_file.tokens < 2 || !safe_relative_path(&token_file.file) {
            return Err(invalid_data("invalid token cache file metadata"));
        }
        let path = resolve_artifact_child(root, &token_file.file)?;
        let metadata = path.metadata()?;
        if metadata.len() != token_file.tokens * 4 {
            return Err(invalid_data(format!(
                "token cache {} length {} is not {} u32 tokens",
                path.display(),
                metadata.len(),
                token_file.tokens
            )));
        }
        let actual = fingerprint_artifact_file(&path)?;
        if actual != token_file.file_hash {
            return Err(invalid_data(format!(
                "token cache {} hash mismatch: expected {}, got {actual}",
                path.display(),
                token_file.file_hash
            )));
        }
    }
    Ok(())
}

fn read_step_sequences(
    file: &mut File,
    total_tokens: u64,
    step: usize,
    batch_size: usize,
    context: usize,
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let span = (batch_size * context) as u64;
    let start = (step as u64).wrapping_mul(span) % total_tokens;
    let tokens = read_tokens_circular(file, total_tokens, start, span as usize + 1)?;
    Ok((0..batch_size)
        .map(|index| {
            let offset = index * context;
            tokens[offset..offset + context + 1].to_vec()
        })
        .collect())
}

fn read_tokens_circular(
    file: &mut File,
    total_tokens: u64,
    start: u64,
    count: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut result = Vec::with_capacity(count);
    let mut cursor = start % total_tokens;
    while result.len() < count {
        let available = (total_tokens - cursor) as usize;
        let take = available.min(count - result.len());
        file.seek(SeekFrom::Start(cursor * 4))?;
        let mut bytes = vec![0u8; take * 4];
        file.read_exact(&mut bytes)?;
        result.extend(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes"))),
        );
        cursor = 0;
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_causal<B: AutodiffBackend>(
    model: &grug_train::TinyTransformer<B>,
    tokenizer: &BpeTokenizer,
    file: &mut File,
    total_tokens: u64,
    batches: usize,
    batch_size: usize,
    context: usize,
    device: &B::Device,
) -> Result<f32, Box<dyn Error>> {
    // Evaluation must not retain a backward graph. On the production A100,
    // evaluating through `B` nearly doubled allocator pressure compared with
    // the training step. Project once to the inner inference backend and keep
    // the checkpointed autodiff model untouched.
    let eval_model = to_eval_model(model);
    let mut sum = 0.0f64;
    for index in 0..batches {
        let virtual_step = index
            * ((total_tokens as usize / (batch_size * context)).max(1) / batches.max(1)).max(1);
        let sequences = read_step_sequences(file, total_tokens, virtual_step, batch_size, context)?;
        let batch =
            build_causal_batch::<B::InnerBackend>(&sequences, tokenizer.pad_id(), context, device)?;
        sum += f64::from(causal_batch_loss(&eval_model, &batch));
    }
    Ok((sum / batches as f64) as f32)
}

fn learning_rate_at_step(
    step: usize,
    total_steps: usize,
    peak: f64,
    warmup: usize,
    minimum_ratio: f64,
) -> f64 {
    if warmup > 0 && step < warmup {
        return peak * (step + 1) as f64 / warmup as f64;
    }
    let minimum = peak * minimum_ratio;
    let decay_steps = total_steps.saturating_sub(warmup).max(1);
    let progress = (step.saturating_sub(warmup) as f64 / decay_steps as f64).min(1.0);
    minimum + 0.5 * (peak - minimum) * (1.0 + (std::f64::consts::PI * progress).cos())
}

fn ema(previous: f32, value: f32) -> f32 {
    if previous.is_finite() {
        previous * 0.98 + value * 0.02
    } else {
        value
    }
}

fn source_key(
    source: &FoundationSourceSummary,
) -> (FoundationSplit, (String, String, String, String)) {
    (
        source.split,
        (
            source.source.clone(),
            source.source_revision.clone(),
            source.license.clone(),
            source.language.clone(),
        ),
    )
}

fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn copy_verified(source: &Path, target: &Path, expected_hash: &str) -> Result<(), Box<dyn Error>> {
    if target.exists() {
        let actual = fingerprint_artifact_file(target)?;
        if actual != expected_hash {
            return Err(invalid_data(format!(
                "existing {} has hash {actual}, expected {expected_hash}",
                target.display()
            )));
        }
        return Ok(());
    }
    let temporary = target.with_extension(format!("tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(invalid_data(format!(
            "stale temporary file exists: {}",
            temporary.display()
        )));
    }
    std::fs::copy(source, &temporary)?;
    let actual = fingerprint_artifact_file(&temporary)?;
    if actual != expected_hash {
        let _ = std::fs::remove_file(&temporary);
        return Err(invalid_data("copied manifest hash changed"));
    }
    std::fs::rename(temporary, target)?;
    Ok(())
}

fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn resolve_manifest_child(root: &Path, relative: &str) -> Result<PathBuf, Box<dyn Error>> {
    if !safe_relative_path(relative) {
        return Err(invalid_data(format!(
            "unsafe corpus shard path {relative:?}"
        )));
    }
    let canonical_root = root.canonicalize()?;
    let canonical = root.join(relative).canonicalize()?;
    if !canonical.starts_with(&canonical_root) {
        return Err(invalid_data(format!(
            "corpus shard escapes manifest directory: {relative:?}"
        )));
    }
    Ok(canonical)
}

fn resolve_artifact_child(root: &Path, relative: &str) -> Result<PathBuf, Box<dyn Error>> {
    if !safe_relative_path(relative) {
        return Err(invalid_data(format!("unsafe artifact path {relative:?}")));
    }
    let canonical_root = root.canonicalize()?;
    let canonical = root.join(relative).canonicalize()?;
    if !canonical.starts_with(&canonical_root) {
        return Err(invalid_data(format!(
            "artifact path escapes bundle: {relative:?}"
        )));
    }
    Ok(canonical)
}

fn write_json_pretty_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn append_json_line(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        message.into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_token_cache_is_rejected_before_more_bytes_are_written() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("training.u32le"), [1, 2, 3]).unwrap();
        let error = reject_partial_token_cache(directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("training.u32le"));
        assert!(!directory.path().join("evaluation.u32le.tmp").exists());
    }

    #[test]
    fn circular_reader_wraps_without_dropping_tokens() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokens.bin");
        let mut output = File::create(&path).unwrap();
        for token in [10u32, 20, 30] {
            output.write_all(&token.to_le_bytes()).unwrap();
        }
        drop(output);
        let mut input = File::open(path).unwrap();
        assert_eq!(
            read_tokens_circular(&mut input, 3, 2, 5).unwrap(),
            vec![30, 10, 20, 30, 10]
        );
    }

    #[test]
    fn cosine_schedule_warms_and_decays() {
        assert_eq!(learning_rate_at_step(0, 100, 1.0, 10, 0.1), 0.1);
        assert_eq!(learning_rate_at_step(9, 100, 1.0, 10, 0.1), 1.0);
        assert!(learning_rate_at_step(99, 100, 1.0, 10, 0.1) < 0.11);
    }
}
