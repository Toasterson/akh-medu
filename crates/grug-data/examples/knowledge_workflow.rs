//! Generate and score stateful textbook/notes curricula against captured tools.
//!
//! Training corpus:
//!
//! cargo run -p grug-data --example knowledge_workflow -- training \
//!   --contracts contracts.json --seed 42 --episodes 500 --sft
//!
//! Prepare one non-overwriting run bundle (train/eval SFT plus all sentinels):
//!
//! cargo run -p grug-data --example knowledge_workflow -- prepare \
//!   --contracts contracts.json --out runs/knowledge/corpus-v2 \
//!   --run-id corpus-v2 --train-seed 42 --eval-seed 4242 \
//!   --sentinel-seed 9001 --train-episodes 5000 --eval-episodes 500
//!
//! Quarantined frontier exam:
//!
//! cargo run -p grug-data --example knowledge_workflow -- exam \
//!   --contracts contracts.json --family knowledge.acquire_textbook \
//!   --tier frontier --run-id run-1 --seed 91 --difficulty 2 --samples 200 \
//!   --checkpoint-step 100 --checkpoint-hash sha256:model
//!
//! `prepare`, `training`, `exam` and `sibling` all accept an optional
//! `--length-buckets` flag (comma-separated target token counts, default
//! 2048,4096,8192,12288) that scopes which distance buckets the
//! `knowledge.length_generalization` family (ADR 104 D5(b)) probes. Pass a
//! narrower list, e.g. `--length-buckets 2048,4096`, to probe only the
//! buckets a smaller-context checkpoint can hold:
//!
//! cargo run -p grug-data --example knowledge_workflow -- exam \
//!   --contracts contracts.json --family knowledge.length_generalization \
//!   --tier frontier --run-id run-1 --seed 91 --difficulty 1 --samples 8 \
//!   --checkpoint-step 100 --checkpoint-hash sha256:model \
//!   --length-buckets 2048,4096
//!
//! `training` and `prepare` also accept a repeatable `--family-weight
//! family=weight` flag (measured: v3's equal per-family allocation let
//! `knowledge.length_generalization`'s padded, multi-step episodes crowd out
//! 31% of SFT records while `knowledge.provider_fallback` got a sliver,
//! plausibly biasing the trained model toward free-text answers). Weight
//! must be a finite number greater than zero; the family id must be one of
//! the eight `knowledge.*` families. Unweighted families keep weight `1.0`,
//! and the flag may repeat to set several families at once:
//!
//! cargo run -p grug-data --example knowledge_workflow -- training \
//!   --contracts contracts.json --seed 42 --episodes 7350 --sft \
//!   --family-weight knowledge.length_generalization=0.25
//!
//! cargo run -p grug-data --example knowledge_workflow -- prepare \
//!   --contracts contracts.json --out runs/knowledge/corpus-v4 \
//!   --run-id corpus-v4 --train-seed 42 --eval-seed 4242 \
//!   --sentinel-seed 9001 --train-episodes 7350 --eval-episodes 500 \
//!   --family-weight knowledge.length_generalization=0.25 \
//!   --family-weight knowledge.provider_fallback=2.0
//!
//! Leaving `--family-weight` unset reproduces today's allocation exactly
//! (every family at its implicit default weight of `1.0`); setting any
//! non-uniform weight changes which scenario lands on which episode index
//! for every family, not only the reweighted one, so opted-in runs will not
//! byte-for-byte match an unweighted run at the same seed. `prepare` records
//! the effective weights it used under `family_weights` in `curriculum.json`.
//!
//! Separately namespaced training sibling from a materialized frontier exam:
//!
//! cargo run -p grug-data --example knowledge_workflow -- sibling \
//!   --contracts contracts.json --exam frontier.json \
//!   --out runs/knowledge/sibling-1 --samples 500
//!
//! Score predictions represented as a JSON object from episode id to an array
//! of raw per-step model outputs:
//!
//! cargo run -p grug-data --example knowledge_workflow -- score \
//!   --contracts contracts.json --exam exam.json --predictions predictions.json \
//!   --checkpoint-step 101 --checkpoint-hash sha256:candidate

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use grug_data::adaptive_exam::ADAPTIVE_EXAM_SCHEMA_VERSION;
use grug_data::{
    CheckpointRef, ExamPlan, ExamTier, KNOWLEDGE_WORKFLOW_GENERATOR_VERSION,
    KNOWLEDGE_WORKFLOW_SCHEMA_VERSION, KnowledgeEpisode, KnowledgeEvaluationMode,
    KnowledgeExamEvidence, KnowledgeExamSet, KnowledgeToolBindings, KnowledgeTrainingCorpus,
    KnowledgeWorkflowGenerator, SiblingSeparationEvidence, SiblingSeparationPolicy,
    ToolContractSnapshot, WorkflowLanguage, WorkflowStage, character_ngram_jaccard,
    derive_training_sibling, knowledge_capability_specs, score_knowledge_exam,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments
        .next()
        .ok_or("expected prepare, training, exam, sibling, or score")?;
    let arguments: Vec<String> = arguments.collect();
    match command.as_str() {
        "prepare" => prepare(&arguments),
        "training" => training(&arguments),
        "exam" => exam(&arguments),
        "sibling" => sibling(&arguments),
        "score" => score(&arguments),
        _ => Err(format!(
            "unknown command {command:?}; expected prepare, training, exam, sibling, or score"
        )
        .into()),
    }
}

fn prepare(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let contracts = load_contracts(required(arguments, "--contracts")?)?;
    let output = Path::new(required(arguments, "--out")?);
    let run_id = required(arguments, "--run-id")?;
    let train_seed: u64 = required(arguments, "--train-seed")?.parse()?;
    let eval_seed: u64 = required(arguments, "--eval-seed")?.parse()?;
    let sentinel_seed: u64 = required(arguments, "--sentinel-seed")?.parse()?;
    let train_episodes: usize = required(arguments, "--train-episodes")?.parse()?;
    let eval_episodes: usize = required(arguments, "--eval-episodes")?.parse()?;
    let max_difficulty: u8 = optional(arguments, "--max-difficulty")
        .unwrap_or("5")
        .parse()?;
    let sentinel_difficulty: u8 = optional(arguments, "--sentinel-difficulty")
        .unwrap_or("1")
        .parse()?;
    let sentinel_samples: u64 = optional(arguments, "--sentinel-samples")
        .unwrap_or("100")
        .parse()?;
    if train_seed == eval_seed
        || train_seed == sentinel_seed
        || eval_seed == sentinel_seed
        || train_episodes == 0
        || eval_episodes == 0
        || sentinel_samples == 0
    {
        return Err(
            "seeds must be pairwise distinct and corpus/exam sizes must be positive".into(),
        );
    }

    // A prepared directory is an immutable quarantine boundary. Refusing an
    // existing path prevents a rerun from silently replacing sentinels.
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(output)?;
    std::fs::create_dir(output.join("sentinels"))?;

    let bindings = KnowledgeToolBindings::default();
    let generator = KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone())?;
    let generator = apply_length_buckets(generator, arguments)?;
    let generator = apply_family_weights(generator, arguments)?;
    let train = generator.generate_training(train_seed, train_episodes, max_difficulty)?;
    let evaluation = generator.generate_training(eval_seed, eval_episodes, max_difficulty)?;
    verify_training_coverage(&train)?;
    reject_episode_overlap(&train.episodes, &evaluation.episodes, "train", "evaluation")?;

    write_json_pretty(&output.join("contracts.json"), &contracts)?;
    write_json_pretty(&output.join("train.episodes.json"), &train)?;
    write_json_pretty(&output.join("eval.episodes.json"), &evaluation)?;
    let train_records = write_sft(
        &output.join("train.sft.jsonl"),
        &train.episodes,
        &contracts,
        &bindings,
    )?;
    let eval_records = write_sft(
        &output.join("eval.sft.jsonl"),
        &evaluation.episodes,
        &contracts,
        &bindings,
    )?;

    let mut sentinels = Vec::new();
    let mut sentinel_ids = BTreeSet::new();
    let mut sentinel_sources = BTreeSet::new();
    for (index, capability) in knowledge_capability_specs().into_iter().enumerate() {
        let plan = ExamPlan {
            schema_version: ADAPTIVE_EXAM_SCHEMA_VERSION,
            generator_version: KNOWLEDGE_WORKFLOW_GENERATOR_VERSION.into(),
            run_id: run_id.into(),
            // Sentinel identity deliberately excludes this placeholder. The
            // evaluator binds each result to the actual candidate checkpoint.
            checkpoint: CheckpointRef {
                step: 0,
                artifact_hash: "unbound:immutable-sentinel".into(),
            },
            tier: ExamTier::Sentinel,
            family: capability.id,
            difficulty: sentinel_difficulty,
            seed: sentinel_seed.wrapping_add(index as u64),
            sample_count: sentinel_samples,
        };
        let exam = generator.generate_exam(&plan)?;
        reject_episode_overlap(&train.episodes, &exam.episodes, "train", "sentinel")?;
        reject_episode_overlap(
            &evaluation.episodes,
            &exam.episodes,
            "evaluation",
            "sentinel",
        )?;
        for episode in &exam.episodes {
            if !sentinel_ids.insert(episode.id.clone())
                || !sentinel_sources.insert(episode.fixture.source_id.clone())
            {
                return Err(format!(
                    "sentinel collision across capability families at {}",
                    episode.id
                )
                .into());
            }
        }
        let filename = format!("{}.json", plan.family.replace('.', "-"));
        let relative = format!("sentinels/{filename}");
        let path = output.join(&relative);
        write_json_pretty(&path, &exam)?;
        sentinels.push(serde_json::json!({
            "family": plan.family,
            "path": relative,
            "exam_fingerprint": exam.exam_fingerprint,
            "file_hash": fingerprint_file(&path)?,
            "samples": plan.sample_count,
            "difficulty": plan.difficulty
        }));
    }

    let manifest = serde_json::json!({
        "schema_version": KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
        "generator_version": KNOWLEDGE_WORKFLOW_GENERATOR_VERSION,
        "run_id": run_id,
        "contract_hash": contracts.manifest_hash,
        "contracts": {"path": "contracts.json", "file_hash": fingerprint_file(&output.join("contracts.json"))?},
        "family_weights": generator.family_weights(),
        "training": {
            "seed": train_seed,
            "episodes": train.episodes.len(),
            "records": train_records,
            "episodes_hash": fingerprint_file(&output.join("train.episodes.json"))?,
            "sft_hash": fingerprint_file(&output.join("train.sft.jsonl"))?
        },
        "evaluation": {
            "seed": eval_seed,
            "episodes": evaluation.episodes.len(),
            "records": eval_records,
            "episodes_hash": fingerprint_file(&output.join("eval.episodes.json"))?,
            "sft_hash": fingerprint_file(&output.join("eval.sft.jsonl"))?
        },
        "sentinel_seed": sentinel_seed,
        "sentinels": sentinels
    });
    write_json_pretty(&output.join("curriculum.json"), &manifest)?;
    println!(
        "prepared {} train records, {} eval records and {} immutable sentinel families in {}",
        train_records,
        eval_records,
        knowledge_capability_specs().len(),
        output.display()
    );
    Ok(())
}

fn training(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let contracts = load_contracts(required(arguments, "--contracts")?)?;
    let bindings = KnowledgeToolBindings::default();
    let generator = KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone())?;
    let generator = apply_length_buckets(generator, arguments)?;
    let generator = apply_family_weights(generator, arguments)?;
    let seed = required(arguments, "--seed")?.parse()?;
    let episode_count = required(arguments, "--episodes")?.parse()?;
    let max_difficulty = optional(arguments, "--max-difficulty")
        .unwrap_or("5")
        .parse()?;
    let corpus = generator.generate_training(seed, episode_count, max_difficulty)?;
    if has_flag(arguments, "--sft") {
        for episode in corpus.episodes {
            for record in episode.to_sft_records(&contracts, &bindings)? {
                println!("{}", record.to_jsonl()?);
            }
        }
    } else if has_flag(arguments, "--jsonl") {
        for episode in corpus.episodes {
            println!("{}", episode.to_jsonl()?);
        }
    } else {
        serde_json::to_writer_pretty(std::io::stdout().lock(), &corpus)?;
        println!();
    }
    Ok(())
}

fn exam(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let contracts = load_contracts(required(arguments, "--contracts")?)?;
    let generator = KnowledgeWorkflowGenerator::new(contracts, KnowledgeToolBindings::default())?;
    let generator = apply_length_buckets(generator, arguments)?;
    let plan = ExamPlan {
        schema_version: ADAPTIVE_EXAM_SCHEMA_VERSION,
        generator_version: optional(arguments, "--plan-generator-version")
            .unwrap_or(KNOWLEDGE_WORKFLOW_GENERATOR_VERSION)
            .into(),
        run_id: required(arguments, "--run-id")?.into(),
        checkpoint: CheckpointRef {
            step: required(arguments, "--checkpoint-step")?.parse()?,
            artifact_hash: required(arguments, "--checkpoint-hash")?.into(),
        },
        tier: parse_tier(required(arguments, "--tier")?)?,
        family: required(arguments, "--family")?.into(),
        difficulty: required(arguments, "--difficulty")?.parse()?,
        seed: required(arguments, "--seed")?.parse()?,
        sample_count: required(arguments, "--samples")?.parse()?,
    };
    let exam = generator.generate_exam(&plan)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &exam)?;
    println!();
    Ok(())
}

/// Turn one controller-selected frontier exam into a separately namespaced,
/// similarity-gated training sibling. The exact exam is verified and remains
/// outside the emitted directory; only its fingerprint is carried forward.
fn sibling(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let contracts = load_contracts(required(arguments, "--contracts")?)?;
    let exam_path = Path::new(required(arguments, "--exam")?);
    let output = Path::new(required(arguments, "--out")?);
    let samples: u64 = required(arguments, "--samples")?.parse()?;
    if samples == 0 {
        return Err("--samples must be positive".into());
    }
    if output.exists() {
        return Err(format!(
            "output {} already exists; adaptive siblings are immutable",
            output.display()
        )
        .into());
    }

    let exam: KnowledgeExamSet = serde_json::from_reader(File::open(exam_path)?)?;
    if exam.contract_hash != contracts.manifest_hash
        || exam.plan.tier != ExamTier::Frontier
        || exam.exam_fingerprint != exam.plan.fingerprint()
    {
        return Err(
            "training siblings require an intact frontier exam under the loaded contract".into(),
        );
    }
    let bindings = KnowledgeToolBindings::default();
    let generator = KnowledgeWorkflowGenerator::new(contracts.clone(), bindings.clone())?;
    // Must match the buckets the source exam was generated with (see
    // `--length-buckets`); otherwise a `knowledge.length_generalization`
    // exam cannot replay identically below.
    let generator = apply_length_buckets(generator, arguments)?;
    let regenerated = generator.generate_exam(&exam.plan)?;
    if regenerated != exam {
        return Err("frontier exam does not replay from its declared plan".into());
    }

    let policy = SiblingSeparationPolicy {
        require_image_distance: false,
        ..SiblingSeparationPolicy::default()
    };
    let plan = derive_training_sibling(&exam.plan, samples, policy)?;
    let corpus = generator.generate_training_sibling(&plan)?;
    reject_episode_overlap(
        &exam.episodes,
        &corpus.episodes,
        "frontier",
        "training-sibling",
    )?;
    if corpus
        .episodes
        .iter()
        .any(|episode| episode.family != plan.family)
    {
        return Err("training sibling contains an unexpected capability family".into());
    }
    let evidence = SiblingSeparationEvidence {
        held_out_axes: vec!["seed_namespace".into(), "source_fixture".into()],
        max_transcript_ngram_jaccard: maximum_episode_similarity(&exam.episodes, &corpus.episodes),
        min_image_hash_hamming_distance: None,
    };
    plan.validate_separation(&evidence)?;

    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(output)?;
    let plan_path = output.join("training-sibling-plan.json");
    let evidence_path = output.join("separation.json");
    let episodes_path = output.join("train.episodes.json");
    let sft_path = output.join("train.sft.jsonl");
    write_json_pretty(&plan_path, &plan)?;
    write_json_pretty(&evidence_path, &evidence)?;
    write_json_pretty(&episodes_path, &corpus)?;
    let records = write_sft(&sft_path, &corpus.episodes, &contracts, &bindings)?;
    let manifest = serde_json::json!({
        "schema_version": KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
        "complete": true,
        "kind": "adaptive_knowledge_training_sibling",
        "generator_version": KNOWLEDGE_WORKFLOW_GENERATOR_VERSION,
        "contract_hash": contracts.manifest_hash,
        "family": plan.family,
        "difficulty": plan.difficulty,
        "episodes": corpus.episodes.len(),
        "records": records,
        "source_exam_fingerprint": exam.exam_fingerprint,
        "source_exam_file_hash": fingerprint_file(exam_path)?,
        "plan_hash": fingerprint_file(&plan_path)?,
        "separation_hash": fingerprint_file(&evidence_path)?,
        "episodes_hash": fingerprint_file(&episodes_path)?,
        "sft_hash": fingerprint_file(&sft_path)?,
    });
    // Completion marker is intentionally last. A partial directory cannot be
    // admitted to a later training segment.
    write_json_pretty(&output.join("manifest.json"), &manifest)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &manifest)?;
    println!();
    Ok(())
}

fn maximum_episode_similarity(left: &[KnowledgeEpisode], right: &[KnowledgeEpisode]) -> f64 {
    // Only the first step is actually placed in the user conversation by
    // `to_sft_records`; later `step.input` values are generator annotations,
    // not model-visible exam text. Including those fixed workflow annotations
    // would measure implementation boilerplate rather than data leakage.
    left.iter()
        .filter_map(|episode| episode.steps.first())
        .flat_map(|left| {
            right
                .iter()
                .filter_map(|episode| episode.steps.first())
                .map(move |right| character_ngram_jaccard(&left.input, &right.input, 3))
        })
        .fold(0.0, f64::max)
}

fn score(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let contracts = load_contracts(required(arguments, "--contracts")?)?;
    let exam: KnowledgeExamSet =
        serde_json::from_reader(File::open(required(arguments, "--exam")?)?)?;
    if exam.contract_hash != contracts.manifest_hash {
        return Err(format!(
            "exam uses contract {}, but scorer loaded {}",
            exam.contract_hash, contracts.manifest_hash
        )
        .into());
    }
    let predictions: BTreeMap<String, Vec<String>> =
        serde_json::from_reader(File::open(required(arguments, "--predictions")?)?)?;
    let report = score_knowledge_exam(
        &exam,
        &predictions,
        &contracts,
        &KnowledgeToolBindings::default(),
    );
    let observation = report.observation(
        CheckpointRef {
            step: required(arguments, "--checkpoint-step")?.parse()?,
            artifact_hash: required(arguments, "--checkpoint-hash")?.into(),
        },
        exam.plan.tier,
    );
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &KnowledgeExamEvidence {
            schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
            evaluation_mode: KnowledgeEvaluationMode::ExternalPredictions,
            report,
            observation,
        },
    )?;
    println!();
    Ok(())
}

fn verify_training_coverage(corpus: &KnowledgeTrainingCorpus) -> Result<(), Box<dyn Error>> {
    let expected_families: BTreeSet<_> = knowledge_capability_specs()
        .into_iter()
        .map(|capability| capability.id)
        .collect();
    let expected_languages: BTreeSet<_> = WorkflowLanguage::all()
        .into_iter()
        .map(|language| language.as_str())
        .collect();
    let expected_stages: BTreeSet<_> = WorkflowStage::all()
        .into_iter()
        .map(|stage| stage.as_str())
        .collect();
    let families: BTreeSet<_> = corpus
        .episodes
        .iter()
        .map(|episode| episode.family.clone())
        .collect();
    let languages: BTreeSet<_> = corpus
        .episodes
        .iter()
        .map(|episode| episode.language.as_str())
        .collect();
    let stages: BTreeSet<_> = corpus
        .episodes
        .iter()
        .flat_map(|episode| episode.steps.iter().map(|step| step.stage.as_str()))
        .collect();
    if families != expected_families || languages != expected_languages || stages != expected_stages
    {
        return Err(format!(
            "training corpus lacks canonical coverage: families={families:?}, languages={languages:?}, stages={stages:?}"
        )
        .into());
    }
    Ok(())
}

fn reject_episode_overlap(
    left: &[KnowledgeEpisode],
    right: &[KnowledgeEpisode],
    left_label: &str,
    right_label: &str,
) -> Result<(), Box<dyn Error>> {
    let ids: BTreeSet<_> = left.iter().map(|episode| episode.id.as_str()).collect();
    let sources: BTreeSet<_> = left
        .iter()
        .map(|episode| episode.fixture.source_id.as_str())
        .collect();
    if let Some(episode) = right.iter().find(|episode| {
        ids.contains(episode.id.as_str()) || sources.contains(episode.fixture.source_id.as_str())
    }) {
        return Err(format!(
            "{left_label}/{right_label} quarantine collision at episode {} source {}",
            episode.id, episode.fixture.source_id
        )
        .into());
    }
    Ok(())
}

fn write_sft(
    path: &Path,
    episodes: &[KnowledgeEpisode],
    contracts: &ToolContractSnapshot,
    bindings: &KnowledgeToolBindings,
) -> Result<usize, Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    let mut count = 0;
    for episode in episodes {
        for record in episode.to_sft_records(contracts, bindings)? {
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
            count += 1;
        }
    }
    writer.flush()?;
    Ok(count)
}

fn write_json_pretty<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn fingerprint_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn load_contracts(path: &str) -> Result<ToolContractSnapshot, Box<dyn Error>> {
    let contracts: ToolContractSnapshot = if path == "-" {
        serde_json::from_reader(std::io::stdin().lock())?
    } else {
        serde_json::from_reader(File::open(path)?)?
    };
    contracts.verify_integrity()?;
    Ok(contracts)
}

/// Parse `--length-buckets 2048,4096,...` for `knowledge.length_generalization`
/// (ADR 104 D5(b)). `None` leaves the generator's default buckets in place.
fn parse_length_buckets(arguments: &[String]) -> Result<Option<Vec<u32>>, Box<dyn Error>> {
    let Some(raw) = optional(arguments, "--length-buckets") else {
        return Ok(None);
    };
    let buckets = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .map_err(|error| format!("invalid --length-buckets entry {part:?}: {error}"))
        })
        .collect::<Result<Vec<u32>, String>>()?;
    Ok(Some(buckets))
}

fn apply_length_buckets(
    generator: KnowledgeWorkflowGenerator,
    arguments: &[String],
) -> Result<KnowledgeWorkflowGenerator, Box<dyn Error>> {
    match parse_length_buckets(arguments)? {
        Some(buckets) => Ok(generator.with_length_buckets(buckets)?),
        None => Ok(generator),
    }
}

/// Parse every repeated `--family-weight family=weight` occurrence for
/// `training`/`prepare` allocation. Missing entirely leaves the generator's
/// default weights (every family `1.0`) untouched, reproducing today's
/// allocation exactly.
fn parse_family_weights(arguments: &[String]) -> Result<BTreeMap<&str, f64>, Box<dyn Error>> {
    let mut weights = BTreeMap::new();
    for raw in all_values(arguments, "--family-weight") {
        let (family, weight) = raw
            .split_once('=')
            .ok_or_else(|| format!("invalid --family-weight {raw:?}: expected family=weight"))?;
        let weight: f64 = weight
            .parse()
            .map_err(|error| format!("invalid --family-weight {raw:?}: {error}"))?;
        weights.insert(family, weight);
    }
    Ok(weights)
}

fn apply_family_weights(
    generator: KnowledgeWorkflowGenerator,
    arguments: &[String],
) -> Result<KnowledgeWorkflowGenerator, Box<dyn Error>> {
    let weights = parse_family_weights(arguments)?;
    if weights.is_empty() {
        Ok(generator)
    } else {
        Ok(generator.with_family_weights(weights)?)
    }
}

fn required<'a>(arguments: &'a [String], flag: &str) -> Result<&'a str, Box<dyn Error>> {
    optional(arguments, flag).ok_or_else(|| format!("{flag} is required").into())
}

fn optional<'a>(arguments: &'a [String], flag: &str) -> Option<&'a str> {
    let index = arguments.iter().position(|value| value == flag)?;
    arguments.get(index + 1).map(String::as_str)
}

/// Every value following a (possibly repeated) occurrence of `flag`, in
/// argument order. Used for flags like `--family-weight` that may be passed
/// more than once, unlike [`optional`], which only ever returns the first.
fn all_values<'a>(arguments: &'a [String], flag: &str) -> Vec<&'a str> {
    arguments
        .iter()
        .zip(arguments.iter().skip(1))
        .filter_map(|(name, value)| (name == flag).then_some(value.as_str()))
        .collect()
}

fn has_flag(arguments: &[String], flag: &str) -> bool {
    arguments.iter().any(|value| value == flag)
}

fn parse_tier(value: &str) -> Result<ExamTier, Box<dyn Error>> {
    match value {
        "sentinel" => Ok(ExamTier::Sentinel),
        "rolling-probe" => Ok(ExamTier::RollingProbe),
        "frontier" => Ok(ExamTier::Frontier),
        "audit" => Ok(ExamTier::Audit),
        _ => Err(format!("unknown tier {value:?}").into()),
    }
}
