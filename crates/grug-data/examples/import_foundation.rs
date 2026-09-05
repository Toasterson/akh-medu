//! Normalize local text or provider-exported JSONL into `FoundationRecord`.
//!
//! Acquisition stays outside the model trainer. This small adapter accepts
//! public Hugging Face exports, offline mirrors, or ordinary text files and
//! stamps every record with explicit source revision/licence/language lineage.

use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use grug_data::{FOUNDATION_RECORD_SCHEMA_VERSION, FoundationRecord};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputFormat {
    Jsonl,
    Text,
}

#[derive(Debug)]
struct Args {
    input: Vec<PathBuf>,
    output: PathBuf,
    format: InputFormat,
    text_pointer: String,
    id_pointer: Option<String>,
    source: String,
    source_revision: String,
    license: String,
    language: String,
    min_chars: usize,
    max_records: Option<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    if args.output.exists() {
        return Err(format!(
            "{} already exists; import outputs are immutable",
            args.output.display()
        )
        .into());
    }
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    let mut written = 0usize;
    let mut dropped_short = 0usize;
    for path in &args.input {
        match args.format {
            InputFormat::Jsonl => {
                let reader = BufReader::new(File::open(path)?);
                for (line_index, line) in reader.lines().enumerate() {
                    if args.max_records.is_some_and(|limit| written >= limit) {
                        break;
                    }
                    let line = line?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let value: Value = serde_json::from_str(&line).map_err(|error| {
                        format!("{}:{} is not JSON: {error}", path.display(), line_index + 1)
                    })?;
                    let text = value
                        .pointer(&args.text_pointer)
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            format!(
                                "{}:{} has no string at --text-pointer {}",
                                path.display(),
                                line_index + 1,
                                args.text_pointer
                            )
                        })?;
                    if text.chars().count() < args.min_chars {
                        dropped_short += 1;
                        continue;
                    }
                    let upstream_id = args
                        .id_pointer
                        .as_deref()
                        .and_then(|pointer| value.pointer(pointer))
                        .and_then(json_scalar_id)
                        .unwrap_or_else(|| format!("line-{}", line_index + 1));
                    let id = stable_id(&args, path, &upstream_id, text);
                    write_record(&mut output, &args, id, text.to_string())?;
                    written += 1;
                }
            }
            InputFormat::Text => {
                if args.max_records.is_some_and(|limit| written >= limit) {
                    break;
                }
                let mut text = String::new();
                File::open(path)?.read_to_string(&mut text)?;
                if text.chars().count() < args.min_chars {
                    dropped_short += 1;
                    continue;
                }
                let upstream_id = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("document");
                let id = stable_id(&args, path, upstream_id, &text);
                write_record(&mut output, &args, id, text)?;
                written += 1;
            }
        }
    }
    output.flush()?;
    let file = output.into_inner()?;
    file.sync_all()?;
    if written == 0 {
        let _ = std::fs::remove_file(&args.output);
        return Err("no records survived import".into());
    }
    println!(
        "imported {written} foundation records (dropped_short={dropped_short}) -> {}",
        args.output.display()
    );
    Ok(())
}

fn write_record(
    writer: &mut impl Write,
    args: &Args,
    id: String,
    text: String,
) -> Result<(), Box<dyn Error>> {
    let record = FoundationRecord {
        schema_version: FOUNDATION_RECORD_SCHEMA_VERSION,
        id,
        source: args.source.clone(),
        source_revision: args.source_revision.clone(),
        license: args.license.clone(),
        language: args.language.clone(),
        text,
    };
    record.validate()?;
    serde_json::to_writer(&mut *writer, &record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn stable_id(args: &Args, path: &std::path::Path, upstream_id: &str, text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    for component in [
        args.source.as_bytes(),
        args.source_revision.as_bytes(),
        path.as_os_str().as_encoded_bytes(),
        upstream_id.as_bytes(),
        text.as_bytes(),
    ] {
        hasher.update(&(component.len() as u64).to_le_bytes());
        hasher.update(component);
    }
    format!("{}-{}", args.source, hasher.finalize().to_hex())
}

fn json_scalar_id(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut input = Vec::new();
    let mut output = None;
    let mut format = None;
    let mut text_pointer = "/text".to_string();
    let mut id_pointer = None;
    let mut source = None;
    let mut source_revision = None;
    let mut license = None;
    let mut language = None;
    let mut min_chars = 200usize;
    let mut max_records = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--input" => input.push(PathBuf::from(
                arguments.next().ok_or("--input requires FILE")?,
            )),
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires FILE")?,
                ))
            }
            "--format" => {
                format = Some(
                    match arguments
                        .next()
                        .ok_or("--format requires jsonl|text")?
                        .as_str()
                    {
                        "jsonl" => InputFormat::Jsonl,
                        "text" => InputFormat::Text,
                        other => return Err(format!("unknown input format {other:?}").into()),
                    },
                )
            }
            "--text-pointer" => {
                text_pointer = arguments.next().ok_or("--text-pointer requires POINTER")?
            }
            "--id-pointer" => {
                id_pointer = Some(arguments.next().ok_or("--id-pointer requires POINTER")?)
            }
            "--source" => source = Some(arguments.next().ok_or("--source requires NAME")?),
            "--source-revision" => {
                source_revision = Some(arguments.next().ok_or("--source-revision requires REV")?)
            }
            "--license" => license = Some(arguments.next().ok_or("--license requires SPDX")?),
            "--language" => language = Some(arguments.next().ok_or("--language requires TAG")?),
            "--min-chars" => {
                min_chars = arguments.next().ok_or("--min-chars requires N")?.parse()?
            }
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
                    "import_foundation --input FILE... --output FILE --format jsonl|text --source NAME --source-revision REV --license SPDX --language TAG [--text-pointer /text] [--id-pointer /id] [--min-chars 200] [--max-records N]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if input.is_empty() {
        return Err("at least one --input is required".into());
    }
    let args = Args {
        input,
        output: output.ok_or("--output is required")?,
        format: format.ok_or("--format is required")?,
        text_pointer,
        id_pointer,
        source: source.ok_or("--source is required")?,
        source_revision: source_revision.ok_or("--source-revision is required")?,
        license: license.ok_or("--license is required")?,
        language: language.ok_or("--language is required")?,
        min_chars,
        max_records,
    };
    for (field, value) in [
        ("source", args.source.as_str()),
        ("source-revision", args.source_revision.as_str()),
        ("license", args.license.as_str()),
        ("language", args.language.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("--{field} must not be blank").into());
        }
    }
    if !args.text_pointer.starts_with('/') {
        return Err("--text-pointer must be a JSON Pointer beginning with /".into());
    }
    if args
        .id_pointer
        .as_ref()
        .is_some_and(|pointer| !pointer.starts_with('/'))
    {
        return Err("--id-pointer must be a JSON Pointer beginning with /".into());
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_ids_are_stable_strings() {
        assert_eq!(
            json_scalar_id(&serde_json::json!(42)).as_deref(),
            Some("42")
        );
        assert!(json_scalar_id(&serde_json::json!({"x": 1})).is_none());
    }
}
