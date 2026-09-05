//! Materialize a safe old-to-new tool-contract adaptation curriculum.
//!
//! Every current tool receives replay examples, added/model-visible changed
//! tools receive extra positives, and removed or schema-incompatible legacy
//! requests receive explicit gap targets. Train/eval seeds are disjoint and
//! the complete bundle is immutable and content-addressed.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use grug_data::{
    KnowledgeSftRecord, TOOL_CONTRACT_DELTA_CURRICULUM_GENERATOR_VERSION, ToolContractDelta,
    ToolContractDeltaCurriculumGenerator, ToolContractSnapshot,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
struct GenerationCounts {
    replay_examples_per_current_tool: usize,
    target_examples_per_added_or_changed_tool: usize,
    legacy_attempts_per_removed_or_changed_tool: usize,
}

#[derive(Debug)]
struct Args {
    previous_contracts: PathBuf,
    contracts: PathBuf,
    out: PathBuf,
    run_id: String,
    train_seed: u64,
    eval_seed: u64,
    train: GenerationCounts,
    evaluation: GenerationCounts,
    offered_tools: usize,
}

#[derive(Serialize)]
struct ArtifactRef {
    file: String,
    file_hash: String,
}

#[derive(Serialize)]
struct ContractRef {
    file: String,
    file_hash: String,
    contract_hash: String,
}

#[derive(Serialize)]
struct StreamSummary {
    seed: u64,
    generation: GenerationCounts,
    records: usize,
    families: BTreeMap<String, usize>,
    artifact: ArtifactRef,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    generator_version: &'static str,
    run_id: String,
    previous_contracts: ContractRef,
    current_contracts: ContractRef,
    delta: ArtifactRef,
    delta_hash: String,
    offered_tools: usize,
    training: StreamSummary,
    evaluation: StreamSummary,
    exact_record_id_overlap: usize,
    exact_prompt_target_overlap: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() {
        return Err(format!(
            "{} already exists; contract-delta curricula are immutable",
            args.out.display()
        )
        .into());
    }
    let previous = read_snapshot(&args.previous_contracts)?;
    let current = read_snapshot(&args.contracts)?;
    let delta = ToolContractDelta::between(&previous, &current)?;
    if !delta.model_adaptation_required() {
        return Err(format!(
            "contract delta {} has no model-visible addition/change/removal; do not update weights",
            delta.delta_hash
        )
        .into());
    }

    let training = ToolContractDeltaCurriculumGenerator::new(
        previous.clone(),
        current.clone(),
        args.train_seed,
        args.offered_tools,
    )?
    .generate(
        &format!("{}-train", args.run_id),
        args.train.replay_examples_per_current_tool,
        args.train.target_examples_per_added_or_changed_tool,
        args.train.legacy_attempts_per_removed_or_changed_tool,
    )?;
    let evaluation = ToolContractDeltaCurriculumGenerator::new(
        previous.clone(),
        current.clone(),
        args.eval_seed,
        args.offered_tools,
    )?
    .generate(
        &format!("{}-eval", args.run_id),
        args.evaluation.replay_examples_per_current_tool,
        args.evaluation.target_examples_per_added_or_changed_tool,
        args.evaluation.legacy_attempts_per_removed_or_changed_tool,
    )?;
    reject_overlap(&training, &evaluation)?;

    let parent = args
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let name = args
        .out
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("--out must have a UTF-8 directory name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(format!("stale temporary path exists: {}", temporary.display()).into());
    }
    std::fs::create_dir(&temporary)?;
    let result = write_bundle(
        &temporary,
        &args,
        &previous,
        &current,
        &delta,
        &training,
        &evaluation,
    );
    match result {
        Ok(()) => {
            std::fs::rename(temporary, &args.out)?;
            println!(
                "prepared contract delta {}: added={} changed={} removed={} train={} eval={} -> {}",
                delta.delta_hash,
                delta.added.len(),
                delta.changed.len(),
                delta.removed.len(),
                training.len(),
                evaluation.len(),
                args.out.display()
            );
            Ok(())
        }
        Err(error) => {
            let _ = std::fs::remove_dir_all(temporary);
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_bundle(
    output: &Path,
    args: &Args,
    previous: &ToolContractSnapshot,
    current: &ToolContractSnapshot,
    delta: &ToolContractDelta,
    training: &[KnowledgeSftRecord],
    evaluation: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let previous_path = output.join("previous-contracts.json");
    let current_path = output.join("contracts.json");
    let delta_path = output.join("contract-delta.json");
    let train_path = output.join("train.sft.jsonl");
    let eval_path = output.join("eval.sft.jsonl");
    write_json_pretty(&previous_path, previous)?;
    write_json_pretty(&current_path, current)?;
    write_json_pretty(&delta_path, delta)?;
    write_jsonl(&train_path, training)?;
    write_jsonl(&eval_path, evaluation)?;

    let manifest = Manifest {
        schema_version: 1,
        generator_version: TOOL_CONTRACT_DELTA_CURRICULUM_GENERATOR_VERSION,
        run_id: args.run_id.clone(),
        previous_contracts: ContractRef {
            file: "previous-contracts.json".into(),
            file_hash: fingerprint_file(&previous_path)?,
            contract_hash: previous.manifest_hash.clone(),
        },
        current_contracts: ContractRef {
            file: "contracts.json".into(),
            file_hash: fingerprint_file(&current_path)?,
            contract_hash: current.manifest_hash.clone(),
        },
        delta: ArtifactRef {
            file: "contract-delta.json".into(),
            file_hash: fingerprint_file(&delta_path)?,
        },
        delta_hash: delta.delta_hash.clone(),
        offered_tools: args.offered_tools,
        training: stream_summary(args.train_seed, args.train, training, &train_path)?,
        evaluation: stream_summary(args.eval_seed, args.evaluation, evaluation, &eval_path)?,
        exact_record_id_overlap: 0,
        exact_prompt_target_overlap: 0,
    };
    // Manifest is the commit marker and is deliberately written last.
    write_json_pretty(&output.join("manifest.json"), &manifest)?;
    Ok(())
}

fn stream_summary(
    seed: u64,
    generation: GenerationCounts,
    records: &[KnowledgeSftRecord],
    path: &Path,
) -> Result<StreamSummary, Box<dyn Error>> {
    let mut families = BTreeMap::new();
    for record in records {
        *families.entry(record.family.clone()).or_insert(0) += 1;
    }
    Ok(StreamSummary {
        seed,
        generation,
        records: records.len(),
        families,
        artifact: ArtifactRef {
            file: path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or("stream filename is not UTF-8")?
                .into(),
            file_hash: fingerprint_file(path)?,
        },
    })
}

fn read_snapshot(path: &Path) -> Result<ToolContractSnapshot, Box<dyn Error>> {
    let snapshot: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(path)?))?;
    snapshot.verify_integrity()?;
    Ok(snapshot)
}

fn write_json_pretty(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn write_jsonl(path: &Path, records: &[KnowledgeSftRecord]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    writer.into_inner()?.sync_all()?;
    Ok(())
}

fn reject_overlap(
    training: &[KnowledgeSftRecord],
    evaluation: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let training_ids: HashSet<_> = training.iter().map(|record| record.id.as_str()).collect();
    if let Some(id) = evaluation
        .iter()
        .map(|record| record.id.as_str())
        .find(|id| training_ids.contains(id))
    {
        return Err(format!("train/evaluation record identity overlap: {id}").into());
    }
    let training_content: HashSet<_> = training
        .iter()
        .map(|record| content_identity(&record.prompt, &record.target))
        .collect();
    if let Some(record) = evaluation
        .iter()
        .find(|record| training_content.contains(&content_identity(&record.prompt, &record.target)))
    {
        return Err(format!(
            "train/evaluation exact prompt+target overlap for {}",
            record.id
        )
        .into());
    }
    Ok(())
}

fn content_identity(prompt: &str, target: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in [prompt.as_bytes(), target.as_bytes()] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    *hasher.finalize().as_bytes()
}

fn fingerprint_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut previous_contracts = None;
    let mut contracts = None;
    let mut out = None;
    let mut run_id = None;
    let mut train_seed = 51_501_u64;
    let mut eval_seed = 51_502_u64;
    let mut train = GenerationCounts {
        replay_examples_per_current_tool: 8,
        target_examples_per_added_or_changed_tool: 48,
        legacy_attempts_per_removed_or_changed_tool: 24,
    };
    let mut evaluation = GenerationCounts {
        replay_examples_per_current_tool: 12,
        target_examples_per_added_or_changed_tool: 12,
        legacy_attempts_per_removed_or_changed_tool: 12,
    };
    let mut offered_tools = 4_usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match argument.as_str() {
            "--previous-contracts" => {
                previous_contracts = Some(PathBuf::from(value(&mut arguments, &argument)?))
            }
            "--contracts" => contracts = Some(PathBuf::from(value(&mut arguments, &argument)?)),
            "--out" => out = Some(PathBuf::from(value(&mut arguments, &argument)?)),
            "--run-id" => run_id = Some(value(&mut arguments, &argument)?),
            "--train-seed" => train_seed = value(&mut arguments, &argument)?.parse()?,
            "--eval-seed" => eval_seed = value(&mut arguments, &argument)?.parse()?,
            "--replay-train-per-tool" => {
                train.replay_examples_per_current_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--target-train-per-tool" => {
                train.target_examples_per_added_or_changed_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--legacy-train-attempts-per-tool" => {
                train.legacy_attempts_per_removed_or_changed_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--replay-eval-per-tool" => {
                evaluation.replay_examples_per_current_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--target-eval-per-tool" => {
                evaluation.target_examples_per_added_or_changed_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--legacy-eval-attempts-per-tool" => {
                evaluation.legacy_attempts_per_removed_or_changed_tool =
                    value(&mut arguments, &argument)?.parse()?
            }
            "--offered-tools" => offered_tools = value(&mut arguments, &argument)?.parse()?,
            "--help" | "-h" => {
                println!(
                    "prepare_tool_contract_delta --previous-contracts FILE --contracts FILE --out DIR --run-id ID [--train-seed 51501] [--eval-seed 51502] [--replay-train-per-tool 8] [--target-train-per-tool 48] [--legacy-train-attempts-per-tool 24] [--replay-eval-per-tool 12] [--target-eval-per-tool 12] [--legacy-eval-attempts-per-tool 12] [--offered-tools 4]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if train_seed == eval_seed
        || train.replay_examples_per_current_tool == 0
        || evaluation.replay_examples_per_current_tool == 0
        || offered_tools == 0
    {
        return Err(
            "train/eval seeds must differ; replay and offered-tool counts must be positive".into(),
        );
    }
    let run_id = run_id.ok_or("--run-id is required")?;
    if run_id.trim().is_empty() {
        return Err("--run-id must not be blank".into());
    }
    Ok(Args {
        previous_contracts: previous_contracts.ok_or("--previous-contracts is required")?,
        contracts: contracts.ok_or("--contracts is required")?,
        out: out.ok_or("--out is required")?,
        run_id,
        train_seed,
        eval_seed,
        train,
        evaluation,
        offered_tools,
    })
}
