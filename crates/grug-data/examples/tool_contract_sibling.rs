//! Derive a separated targeted SFT sibling from a verified tool frontier exam.

use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use grug_data::{
    KnowledgeSftRecord, SiblingSeparationEvidence, ToolContractExamSet, ToolContractSnapshot,
    character_ngram_jaccard, derive_tool_contract_training_sibling, render_knowledge_prompt,
};
use serde::Serialize;

#[derive(Debug)]
struct Args {
    contracts: PathBuf,
    exam: PathBuf,
    family: String,
    out: PathBuf,
    samples: u64,
}

#[derive(Debug, Serialize)]
struct SeparationReport {
    held_out_axes: Vec<String>,
    max_transcript_ngram_jaccard: f64,
    min_image_hash_hamming_distance: Option<u32>,
    exact_user_request_overlap: usize,
    exact_prompt_target_overlap: usize,
    exam_family_cases: usize,
    sibling_records: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.out.exists() {
        return Err(format!(
            "{} already exists; adaptive siblings are immutable",
            args.out.display()
        )
        .into());
    }
    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&args.contracts)?))?;
    contracts.verify_integrity()?;
    let exam: ToolContractExamSet =
        serde_json::from_reader(BufReader::new(File::open(&args.exam)?))?;
    let plan =
        derive_tool_contract_training_sibling(&exam, &contracts, &args.family, args.samples)?;
    plan.verify_against_exam(&exam, &contracts)?;
    let records = plan.generate(&contracts)?;
    let separation = separation_report(&exam, &records, &plan.tool)?;
    plan.validate_separation(&SiblingSeparationEvidence {
        held_out_axes: separation.held_out_axes.clone(),
        max_transcript_ngram_jaccard: separation.max_transcript_ngram_jaccard,
        min_image_hash_hamming_distance: separation.min_image_hash_hamming_distance,
    })?;
    if separation.exact_user_request_overlap != 0 || separation.exact_prompt_target_overlap != 0 {
        return Err("tool frontier and training sibling have exact content overlap".into());
    }

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
    let result = write_bundle(&temporary, &args, &plan, &separation, &records);
    match result {
        Ok(()) => {
            std::fs::rename(temporary, &args.out)?;
            println!(
                "prepared tool sibling: family={} records={} max_jaccard={:.6} source={} -> {}",
                plan.family,
                records.len(),
                separation.max_transcript_ngram_jaccard,
                plan.source_exam_fingerprint,
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
    plan: &grug_data::ToolContractTrainingSiblingPlan,
    separation: &SeparationReport,
    records: &[KnowledgeSftRecord],
) -> Result<(), Box<dyn Error>> {
    let plan_path = output.join("training-sibling-plan.json");
    let separation_path = output.join("separation.json");
    let sft_path = output.join("train.sft.jsonl");
    write_json_pretty(&plan_path, plan)?;
    write_json_pretty(&separation_path, separation)?;
    write_jsonl(&sft_path, records)?;
    let manifest = serde_json::json!({
        "schema_version": 1,
        "complete": true,
        "kind": "adaptive_tool_contract_training_sibling",
        "generator_version": plan.generator_version,
        "contract_hash": plan.contract_hash,
        "family": plan.family,
        "tool": plan.tool,
        "difficulty": plan.difficulty,
        "records": records.len(),
        "source_exam_fingerprint": plan.source_exam_fingerprint,
        "source_exam_file_hash": fingerprint_file(&args.exam)?,
        "plan_hash": fingerprint_file(&plan_path)?,
        "separation_hash": fingerprint_file(&separation_path)?,
        "sft_hash": fingerprint_file(&sft_path)?,
    });
    // Completion marker is intentionally last.
    write_json_pretty(&output.join("manifest.json"), &manifest)?;
    Ok(())
}

fn separation_report(
    exam: &ToolContractExamSet,
    records: &[KnowledgeSftRecord],
    tool: &str,
) -> Result<SeparationReport, Box<dyn Error>> {
    let exam_cases: Vec<_> = exam.cases.iter().filter(|case| case.tool == tool).collect();
    let exam_requests: Vec<_> = exam_cases
        .iter()
        .map(|case| user_content(&case.conversation))
        .collect::<Result<_, _>>()?;
    let sibling_requests: Vec<_> = records
        .iter()
        .map(|record| user_content(&record.conversation))
        .collect::<Result<_, _>>()?;
    let max_transcript_ngram_jaccard = exam_requests
        .iter()
        .flat_map(|left| {
            sibling_requests
                .iter()
                .map(move |right| character_ngram_jaccard(left, right, 3))
        })
        .fold(0.0, f64::max);
    let exam_request_set: HashSet<_> = exam_requests.iter().copied().collect();
    let exact_user_request_overlap = sibling_requests
        .iter()
        .filter(|request| exam_request_set.contains(**request))
        .count();

    let exam_content: BTreeSet<_> = exam_cases
        .iter()
        .map(|case| {
            Ok(content_identity(
                &render_knowledge_prompt(&case.tools, &case.conversation),
                &case.expected.render_runtime()?,
            ))
        })
        .collect::<Result<_, grug_data::KnowledgeWorkflowError>>()?;
    let exact_prompt_target_overlap = records
        .iter()
        .filter(|record| exam_content.contains(&content_identity(&record.prompt, &record.target)))
        .count();

    Ok(SeparationReport {
        held_out_axes: vec!["prompt_bank".into(), "seed_namespace".into()],
        max_transcript_ngram_jaccard,
        min_image_hash_hamming_distance: None,
        exact_user_request_overlap,
        exact_prompt_target_overlap,
        exam_family_cases: exam_cases.len(),
        sibling_records: records.len(),
    })
}

fn user_content(conversation: &[serde_json::Value]) -> Result<&str, Box<dyn Error>> {
    conversation
        .first()
        .filter(|message| message.get("role").and_then(serde_json::Value::as_str) == Some("user"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "conversation has no first user text message".into())
}

fn content_identity(prompt: &str, target: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for value in [prompt.as_bytes(), target.as_bytes()] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    *hasher.finalize().as_bytes()
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
    let mut contracts = None;
    let mut exam = None;
    let mut family = None;
    let mut out = None;
    let mut samples = 500_u64;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--contracts" => {
                contracts = Some(PathBuf::from(
                    arguments.next().ok_or("--contracts requires FILE")?,
                ))
            }
            "--exam" => {
                exam = Some(PathBuf::from(
                    arguments.next().ok_or("--exam requires FILE")?,
                ))
            }
            "--family" => family = Some(arguments.next().ok_or("--family requires ID")?),
            "--out" => out = Some(PathBuf::from(arguments.next().ok_or("--out requires DIR")?)),
            "--samples" => samples = arguments.next().ok_or("--samples requires N")?.parse()?,
            "--help" | "-h" => {
                println!(
                    "tool_contract_sibling --contracts FILE --exam FRONTIER.json --family tool_contract.NAME --out DIR [--samples 500]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if samples == 0 {
        return Err("--samples must be positive".into());
    }
    Ok(Args {
        contracts: contracts.ok_or("--contracts is required")?,
        exam: exam.ok_or("--exam is required")?,
        family: family.ok_or("--family is required")?,
        out: out.ok_or("--out is required")?,
        samples,
    })
}
