//! Materialize disjoint SFT streams over every captured Akh/MCP tool.
//!
//! The output uses `KnowledgeSftRecord`, so the knowledge trainer, evaluator,
//! and daemon all see the same prompt/action ABI. Values and distractor order
//! differ between train and evaluation; exact prompt/target overlap is refused.

use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use grug_data::{
    KnowledgeSftRecord, TOOL_CURRICULUM_GENERATOR_VERSION, ToolContractSnapshot,
    ToolCurriculumGenerator,
};
use serde::Serialize;

#[derive(Debug)]
struct Args {
    contracts: PathBuf,
    out: PathBuf,
    run_id: String,
    train_seed: u64,
    eval_seed: u64,
    train_examples_per_tool: usize,
    eval_examples_per_tool: usize,
    offered_tools: usize,
}

#[derive(Serialize)]
struct StreamSummary {
    seed: u64,
    examples_per_tool: usize,
    records: usize,
    file: String,
    file_hash: String,
}

#[derive(Serialize)]
struct Manifest {
    schema_version: u32,
    generator_version: &'static str,
    run_id: String,
    contract_hash: String,
    offered_tools: usize,
    tools: BTreeSet<String>,
    languages: BTreeSet<String>,
    training: StreamSummary,
    evaluation: StreamSummary,
    exact_prompt_target_overlap: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() {
        return Err(format!(
            "{} already exists; tool curricula are immutable",
            args.out.display()
        )
        .into());
    }
    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&args.contracts)?))?;
    contracts.verify_integrity()?;
    if args.offered_tools > contracts.tools.len() {
        return Err(format!(
            "--offered-tools {} exceeds contract size {}",
            args.offered_tools,
            contracts.tools.len()
        )
        .into());
    }

    let training =
        ToolCurriculumGenerator::new(contracts.clone(), args.train_seed, args.offered_tools)?
            .generate(
                &format!("{}-train", args.run_id),
                args.train_examples_per_tool,
            )?;
    let evaluation =
        ToolCurriculumGenerator::new(contracts.clone(), args.eval_seed, args.offered_tools)?
            .generate(
                &format!("{}-eval", args.run_id),
                args.eval_examples_per_tool,
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
    let result = write_bundle(&temporary, &args, &contracts, &training, &evaluation);
    match result {
        Ok(()) => {
            std::fs::rename(temporary, &args.out)?;
            println!(
                "prepared tool curriculum: tools={} train={} eval={} offered={} -> {}",
                contracts.tools.len(),
                training.len(),
                evaluation.len(),
                args.offered_tools,
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

fn write_bundle(
    output: &Path,
    args: &Args,
    contracts: &ToolContractSnapshot,
    training: &[KnowledgeSftRecord],
    evaluation: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let train_path = output.join("train.sft.jsonl");
    let eval_path = output.join("eval.sft.jsonl");
    write_jsonl(&train_path, training)?;
    write_jsonl(&eval_path, evaluation)?;
    let languages = training
        .iter()
        .chain(evaluation)
        .map(|record| json_label(record.language))
        .collect();
    let manifest = Manifest {
        schema_version: 1,
        generator_version: TOOL_CURRICULUM_GENERATOR_VERSION,
        run_id: args.run_id.clone(),
        contract_hash: contracts.manifest_hash.clone(),
        offered_tools: args.offered_tools,
        tools: contracts
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect(),
        languages,
        training: StreamSummary {
            seed: args.train_seed,
            examples_per_tool: args.train_examples_per_tool,
            records: training.len(),
            file: "train.sft.jsonl".into(),
            file_hash: fingerprint_file(&train_path)?,
        },
        evaluation: StreamSummary {
            seed: args.eval_seed,
            examples_per_tool: args.eval_examples_per_tool,
            records: evaluation.len(),
            file: "eval.sft.jsonl".into(),
            file_hash: fingerprint_file(&eval_path)?,
        },
        exact_prompt_target_overlap: 0,
    };
    let manifest_path = output.join("manifest.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)?;
    serde_json::to_writer_pretty(&mut file, &manifest)?;
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
    let file = writer.into_inner()?;
    file.sync_all()?;
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
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn json_label<T: Serialize>(value: T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(String::from))
        .unwrap_or_else(|| "unknown".into())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut contracts = None;
    let mut out = None;
    let mut run_id = None;
    let mut train_seed = 51_041u64;
    let mut eval_seed = 51_042u64;
    let mut train_examples_per_tool = 48usize;
    let mut eval_examples_per_tool = 12usize;
    let mut offered_tools = 4usize;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--contracts" => {
                contracts = Some(PathBuf::from(
                    arguments.next().ok_or("--contracts requires FILE")?,
                ))
            }
            "--out" => out = Some(PathBuf::from(arguments.next().ok_or("--out requires DIR")?)),
            "--run-id" => run_id = Some(arguments.next().ok_or("--run-id requires ID")?),
            "--train-seed" => {
                train_seed = arguments.next().ok_or("--train-seed requires N")?.parse()?
            }
            "--eval-seed" => {
                eval_seed = arguments.next().ok_or("--eval-seed requires N")?.parse()?
            }
            "--train-examples-per-tool" => {
                train_examples_per_tool = arguments
                    .next()
                    .ok_or("--train-examples-per-tool requires N")?
                    .parse()?
            }
            "--eval-examples-per-tool" => {
                eval_examples_per_tool = arguments
                    .next()
                    .ok_or("--eval-examples-per-tool requires N")?
                    .parse()?
            }
            "--offered-tools" => {
                offered_tools = arguments
                    .next()
                    .ok_or("--offered-tools requires N")?
                    .parse()?
            }
            "--help" | "-h" => {
                println!(
                    "prepare_tool_curriculum --contracts FILE --out DIR --run-id ID [--train-seed 51041] [--eval-seed 51042] [--train-examples-per-tool 48] [--eval-examples-per-tool 12] [--offered-tools 4]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if train_seed == eval_seed
        || train_examples_per_tool == 0
        || eval_examples_per_tool == 0
        || offered_tools == 0
    {
        return Err("train/eval seeds must differ and counts must be positive".into());
    }
    let run_id = run_id.ok_or("--run-id is required")?;
    if run_id.trim().is_empty() {
        return Err("--run-id must not be blank".into());
    }
    Ok(Args {
        contracts: contracts.ok_or("--contracts is required")?,
        out: out.ok_or("--out is required")?,
        run_id,
        train_seed,
        eval_seed,
        train_examples_per_tool,
        eval_examples_per_tool,
        offered_tools,
    })
}
