//! Materialize a quarantined synthetic exam over every captured Akh tool.

use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};

use grug_data::{CheckpointRef, ExamTier, ToolContractSnapshot, ToolCurriculumGenerator};

#[derive(Debug)]
struct Args {
    contracts: PathBuf,
    out: PathBuf,
    run_id: String,
    checkpoint_step: u64,
    checkpoint_hash: String,
    tier: ExamTier,
    difficulty: u8,
    seed: u64,
    samples_per_tool: usize,
    offered_tools: usize,
    family: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() {
        return Err(format!(
            "{} already exists; quarantined exams are immutable",
            args.out.display()
        )
        .into());
    }
    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&args.contracts)?))?;
    let generator = ToolCurriculumGenerator::new(contracts.clone(), args.seed, args.offered_tools)?;
    let checkpoint = CheckpointRef {
        step: args.checkpoint_step,
        artifact_hash: args.checkpoint_hash,
    };
    let exam = if let Some(family) = &args.family {
        let tool = family
            .strip_prefix("tool_contract.")
            .ok_or_else(|| format!("--family {family:?} must use the tool_contract.NAME form"))?;
        generator.generate_exam_for_tools(
            &args.run_id,
            checkpoint,
            args.tier,
            args.difficulty,
            args.samples_per_tool,
            &BTreeSet::from([tool.to_string()]),
        )?
    } else {
        generator.generate_exam(
            &args.run_id,
            checkpoint,
            args.tier,
            args.difficulty,
            args.samples_per_tool,
        )?
    };
    exam.verify(&contracts)?;
    write_json_pretty_new(&args.out, &exam)?;
    println!(
        "generated tool exam: catalog_tools={} target_tools={} samples_per_tool={} cases={} tier={:?} difficulty={} fingerprint={} -> {}",
        contracts.tools.len(),
        if exam.target_tools.is_empty() {
            contracts.tools.len()
        } else {
            exam.target_tools.len()
        },
        exam.samples_per_tool,
        exam.cases.len(),
        exam.tier,
        exam.difficulty,
        exam.exam_fingerprint,
        args.out.display()
    );
    Ok(())
}

fn write_json_pretty_new<T: serde::Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut contracts = None;
    let mut out = None;
    let mut run_id = None;
    let mut checkpoint_step = None;
    let mut checkpoint_hash = None;
    let mut tier = ExamTier::RollingProbe;
    let mut difficulty = 3u8;
    let mut seed = 51_043u64;
    let mut samples_per_tool = 24usize;
    let mut offered_tools = 4usize;
    let mut family = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--contracts" => {
                contracts = Some(PathBuf::from(
                    arguments.next().ok_or("--contracts requires FILE")?,
                ))
            }
            "--out" => {
                out = Some(PathBuf::from(
                    arguments.next().ok_or("--out requires FILE")?,
                ))
            }
            "--run-id" => run_id = Some(arguments.next().ok_or("--run-id requires ID")?),
            "--checkpoint-step" => {
                checkpoint_step = Some(
                    arguments
                        .next()
                        .ok_or("--checkpoint-step requires N")?
                        .parse()?,
                )
            }
            "--checkpoint-hash" => {
                checkpoint_hash = Some(arguments.next().ok_or("--checkpoint-hash requires HASH")?)
            }
            "--tier" => tier = parse_tier(&arguments.next().ok_or("--tier requires TIER")?)?,
            "--difficulty" => {
                difficulty = arguments.next().ok_or("--difficulty requires N")?.parse()?
            }
            "--seed" => seed = arguments.next().ok_or("--seed requires N")?.parse()?,
            "--samples-per-tool" => {
                samples_per_tool = arguments
                    .next()
                    .ok_or("--samples-per-tool requires N")?
                    .parse()?
            }
            "--offered-tools" => {
                offered_tools = arguments
                    .next()
                    .ok_or("--offered-tools requires N")?
                    .parse()?
            }
            "--family" => {
                family = Some(
                    arguments
                        .next()
                        .ok_or("--family requires tool_contract.NAME")?,
                )
            }
            "--help" | "-h" => {
                println!(
                    "generate_tool_contract_exam --contracts FILE --out FILE --run-id ID --checkpoint-step N --checkpoint-hash blake3:HASH [--tier rolling-probe] [--difficulty 3] [--seed 51043] [--samples-per-tool 24] [--offered-tools 4] [--family tool_contract.NAME]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let run_id = run_id.ok_or("--run-id is required")?;
    let checkpoint_hash = checkpoint_hash.ok_or("--checkpoint-hash is required")?;
    let checkpoint_valid = if tier == ExamTier::Sentinel {
        checkpoint_step == Some(0) && checkpoint_hash == "unbound:immutable-sentinel"
    } else {
        checkpoint_hash.starts_with("blake3:")
    };
    if run_id.trim().is_empty()
        || !checkpoint_valid
        || !(1..=3).contains(&difficulty)
        || samples_per_tool == 0
        || offered_tools == 0
    {
        return Err(
            "run id/hash/difficulty/sample/tool arguments are invalid; a sentinel requires --checkpoint-step 0 --checkpoint-hash unbound:immutable-sentinel"
                .into(),
        );
    }
    Ok(Args {
        contracts: contracts.ok_or("--contracts is required")?,
        out: out.ok_or("--out is required")?,
        run_id,
        checkpoint_step: checkpoint_step.ok_or("--checkpoint-step is required")?,
        checkpoint_hash,
        tier,
        difficulty,
        seed,
        samples_per_tool,
        offered_tools,
        family,
    })
}

fn parse_tier(value: &str) -> Result<ExamTier, Box<dyn Error>> {
    match value {
        "sentinel" => Ok(ExamTier::Sentinel),
        "rolling-probe" => Ok(ExamTier::RollingProbe),
        "frontier" => Ok(ExamTier::Frontier),
        "audit" => Ok(ExamTier::Audit),
        _ => Err(format!(
            "unknown --tier {value:?}; expected sentinel, rolling-probe, frontier, or audit"
        )
        .into()),
    }
}
