//! Score OCR predictions against a generated vision manifest.
//!
//! Predictions are JSONL objects with `case_id`, `text`, and optional
//! `confidence`. The summary includes controller-ready pass/precursor counts.

use grug_data::{VisionTextManifest, VisionTextPrediction, score_vision_text};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let (manifest_path, prediction_path) = parse_args()?;
    let cases: Vec<VisionTextManifest> = read_jsonl(&manifest_path)?;
    let predictions: Vec<VisionTextPrediction> = read_jsonl(&prediction_path)?;
    let summary = score_vision_text(&cases, &predictions)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &summary)?;
    println!();
    Ok(())
}

fn parse_args() -> Result<(String, String), Box<dyn Error>> {
    let mut manifest = None;
    let mut predictions = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(value),
            "--predictions" => predictions = Some(value),
            _ => return Err(format!("unknown flag {flag:?}").into()),
        }
    }
    Ok((
        manifest.ok_or("--manifest is required")?,
        predictions.ok_or("--predictions is required")?,
    ))
}

fn read_jsonl<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<Vec<T>, Box<dyn Error>> {
    let file = BufReader::new(File::open(path)?);
    file.lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(
                serde_json::from_str(&line)
                    .map_err(|error| format!("line {}: {error}", index + 1).into()),
            ),
            Err(error) => Some(Err(error.into())),
        })
        .collect()
}
