//! Convert canonical Akh runtime SFT into causal pretraining fluency text.
//!
//! This does not replace target-masked knowledge/tool SFT. It gives the
//! definitive tokenizer and language trunk early exposure to ChatML, tool
//! schemas, strict action envelopes, and persona dialogue without importing
//! held-out examples into the foundation stream.

use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use grug_data::{
    FOUNDATION_RECORD_SCHEMA_VERSION, FoundationRecord, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
    KnowledgeAction, KnowledgeSftRecord, ToolContractSnapshot, render_knowledge_prompt_with_system,
};

#[derive(Debug)]
struct Args {
    contracts: PathBuf,
    input: PathBuf,
    output: PathBuf,
    source: String,
    source_revision: String,
    license: String,
    max_records: Option<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.output.exists() {
        return Err(format!(
            "{} already exists; conversion outputs are immutable",
            args.output.display()
        )
        .into());
    }
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let contracts: ToolContractSnapshot =
        serde_json::from_reader(BufReader::new(File::open(&args.contracts)?))?;
    contracts.verify_integrity()?;
    let reader = BufReader::new(File::open(&args.input)?);
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    let mut written = 0usize;
    let mut fingerprint = blake3::Hasher::new();
    for (line_index, line) in reader.lines().enumerate() {
        if args.max_records.is_some_and(|limit| written >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: KnowledgeSftRecord = serde_json::from_str(&line).map_err(|error| {
            format!(
                "{}:{} is not a KnowledgeSftRecord: {error}",
                args.input.display(),
                line_index + 1
            )
        })?;
        validate_sft(&record, &contracts)?;
        let mut text = record.prompt.clone();
        text.push_str(&record.target);
        text.push_str("<|im_end|>\n");
        let identity = serde_json::to_vec(&serde_json::json!({
            "source": args.source,
            "source_revision": args.source_revision,
            "record_id": record.id,
            "text": text,
        }))?;
        let foundation = FoundationRecord {
            schema_version: FOUNDATION_RECORD_SCHEMA_VERSION,
            id: format!("akh-sft-{}", blake3::hash(&identity).to_hex()),
            source: args.source.clone(),
            source_revision: args.source_revision.clone(),
            license: args.license.clone(),
            // Foundation acquisition uses stable BCP-47/ISO language codes.
            // Keep the SFT bridge on the same vocabulary so corpus manifests
            // cannot report `en` and `english` as two separate languages.
            language: record.language.code().into(),
            text,
        };
        foundation.validate()?;
        let encoded = serde_json::to_vec(&foundation)?;
        writer.write_all(&encoded)?;
        writer.write_all(b"\n")?;
        fingerprint.update(&(encoded.len() as u64).to_le_bytes());
        fingerprint.update(&encoded);
        written += 1;
    }
    writer.flush()?;
    let file = writer.into_inner()?;
    file.sync_all()?;
    if written == 0 {
        let _ = std::fs::remove_file(&args.output);
        return Err("input produced no foundation records".into());
    }
    println!(
        "converted {written} canonical SFT records -> {} fingerprint=blake3:{}",
        args.output.display(),
        fingerprint.finalize().to_hex()
    );
    Ok(())
}

fn validate_sft(
    record: &KnowledgeSftRecord,
    contracts: &ToolContractSnapshot,
) -> Result<(), Box<dyn Error>> {
    if record.schema_version != KNOWLEDGE_WORKFLOW_SCHEMA_VERSION
        || record.contract_hash != contracts.manifest_hash
    {
        return Err(format!(
            "record {} has incompatible schema/contract identity",
            record.id
        )
        .into());
    }
    let rendered = render_knowledge_prompt_with_system(
        &record.tools,
        &record.conversation,
        record.system_context.as_deref(),
    );
    if rendered != record.prompt {
        return Err(format!("record {} has a non-canonical prompt", record.id).into());
    }
    let action = KnowledgeAction::parse_runtime(&record.target)?;
    if action.render_runtime()?.trim() != record.target.trim() {
        return Err(format!("record {} has a non-canonical target", record.id).into());
    }
    if let KnowledgeAction::ToolCall { tool, arguments } = action {
        let offered = record.tools.iter().any(|definition| {
            definition
                .pointer("/function/name")
                .and_then(serde_json::Value::as_str)
                == Some(tool.as_str())
        });
        if !offered {
            return Err(format!("record {} calls an unoffered tool {tool}", record.id).into());
        }
        let validation = contracts.validate_call(&tool, &arguments);
        if !validation.valid {
            return Err(format!(
                "record {} calls {tool} with invalid arguments: {:?}",
                record.id, validation.issues
            )
            .into());
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut contracts = None;
    let mut input = None;
    let mut output = None;
    let mut source = None;
    let mut source_revision = None;
    let mut license = None;
    let mut max_records = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--contracts" => {
                contracts = Some(PathBuf::from(
                    arguments.next().ok_or("--contracts requires FILE")?,
                ))
            }
            "--input" => {
                input = Some(PathBuf::from(
                    arguments.next().ok_or("--input requires FILE")?,
                ))
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires FILE")?,
                ))
            }
            "--source" => source = Some(arguments.next().ok_or("--source requires NAME")?),
            "--source-revision" => {
                source_revision = Some(arguments.next().ok_or("--source-revision requires REV")?)
            }
            "--license" => license = Some(arguments.next().ok_or("--license requires SPDX")?),
            "--max-records" => {
                max_records = Some(
                    arguments
                        .next()
                        .ok_or("--max-records requires N")?
                        .parse()?,
                )
            }
            "--help" | "-h" => {
                println!(
                    "convert_sft_foundation --contracts FILE --input FILE --output FILE --source NAME --source-revision REV --license SPDX [--max-records N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let args = Args {
        contracts: contracts.ok_or("--contracts is required")?,
        input: input.ok_or("--input is required")?,
        output: output.ok_or("--output is required")?,
        source: source.ok_or("--source is required")?,
        source_revision: source_revision.ok_or("--source-revision is required")?,
        license: license.ok_or("--license is required")?,
        max_records,
    };
    for (field, value) in [
        ("source", args.source.as_str()),
        ("source-revision", args.source_revision.as_str()),
        ("license", args.license.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("--{field} must not be blank").into());
        }
    }
    Ok(args)
}
