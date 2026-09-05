//! Reject a candidate vision training manifest that is too close to an exam.

use grug_data::{SiblingSeparationPolicy, VisionTextManifest, audit_vision_sibling_separation};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let exam_path = required(&args, "--exam")?;
    let training_path = required(&args, "--training")?;
    let report = audit_vision_sibling_separation(
        &read_manifest(exam_path)?,
        &read_manifest(training_path)?,
        SiblingSeparationPolicy::default(),
    );
    serde_json::to_writer_pretty(std::io::stdout().lock(), &report)?;
    println!();
    if !report.admitted {
        return Err(report
            .rejection
            .unwrap_or_else(|| "sibling separation rejected".into())
            .into());
    }
    Ok(())
}

fn required<'a>(args: &'a [String], flag: &str) -> Result<&'a str, Box<dyn Error>> {
    let index = args
        .iter()
        .position(|value| value == flag)
        .ok_or_else(|| format!("{flag} is required"))?;
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn read_manifest(path: &str) -> Result<Vec<VisionTextManifest>, Box<dyn Error>> {
    BufReader::new(File::open(path)?)
        .lines()
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
