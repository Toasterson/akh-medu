//! Stateless CLI boundary for the serializable adaptive-exam controller.
//!
//! Initialize controller state:
//!
//! ```text
//! cargo run -p grug-data --example adaptive_exam_controller -- init \
//!   --run-id run-001 --generator-version vision-text-v1 --seed 42 \
//!   --capabilities vision.text.line:0:5:2,reason.chain.depth:0:6:3 \
//!   --min-samples-per-window 24 --enable-precursor-steering
//! ```
//!
//! Apply JSONL observations and emit the updated state plus next plan:
//!
//! ```text
//! cargo run -p grug-data --example adaptive_exam_controller -- step \
//!   --state state.json --observations observations.jsonl --tier frontier \
//!   --checkpoint-step 100 --checkpoint-hash sha256:... --samples 200
//! ```

use grug_data::{
    AdaptiveExamController, CapabilitySpec, CheckpointRef, ControllerConfig, ControllerDecision,
    ExamObservation, ExamPlan, ExamTier, ToolContractSnapshot, tool_contract_capability_specs,
};
use serde::Serialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Serialize)]
struct StepOutput {
    controller: AdaptiveExamController,
    decisions: Vec<ControllerDecision>,
    next_plan: ExamPlan,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("init") => initialize(args.collect()),
        Some("step") => step(args.collect()),
        Some(command) => Err(format!("unknown command {command:?}; expected init or step").into()),
        None => Err("expected init or step".into()),
    }
}

fn initialize(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    validate_flags(
        &args,
        &[
            "--run-id",
            "--generator-version",
            "--seed",
            "--capabilities",
            "--contracts",
            "--min-samples-per-window",
            "--precursor-limit",
            "--catastrophic-failure-limit",
            "--steer-windows",
            "--stop-windows",
        ],
        &["--enable-precursor-steering"],
    )?;
    let run_id = required(&args, "--run-id")?;
    let generator_version = required(&args, "--generator-version")?;
    let seed: u64 = required(&args, "--seed")?.parse()?;
    let mut capabilities = optional(&args, "--capabilities")
        .map(|values| {
            values
                .split(',')
                .map(|value| {
                    let fields: Vec<_> = value.split(':').collect();
                    if fields.len() != 4 {
                        return Err(format!("capability {value:?} must be id:min:max:initial"));
                    }
                    Ok(CapabilitySpec::new(
                        fields[0],
                        fields[1]
                            .parse()
                            .map_err(|error| format!("{value}: {error}"))?,
                        fields[2]
                            .parse()
                            .map_err(|error| format!("{value}: {error}"))?,
                    )
                    .with_initial_difficulty(
                        fields[3]
                            .parse()
                            .map_err(|error| format!("{value}: {error}"))?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    if let Some(path) = optional(&args, "--contracts") {
        let contracts: ToolContractSnapshot =
            serde_json::from_reader(BufReader::new(File::open(path)?))?;
        capabilities.extend(tool_contract_capability_specs(&contracts)?);
    }
    if capabilities.is_empty() {
        return Err("at least one --capabilities list or --contracts snapshot is required".into());
    }
    let mut ids = BTreeSet::new();
    if let Some(duplicate) = capabilities
        .iter()
        .map(|capability| capability.id.as_str())
        .find(|id| !ids.insert(*id))
    {
        return Err(format!("capability {duplicate:?} was registered more than once").into());
    }
    let mut config = ControllerConfig::default();
    if let Some(value) = optional(&args, "--min-samples-per-window") {
        config.min_samples_per_window = value.parse()?;
    }
    if let Some(value) = optional(&args, "--precursor-limit") {
        config.precursor_limit = value.parse()?;
    }
    if let Some(value) = optional(&args, "--catastrophic-failure-limit") {
        config.catastrophic_failure_limit = value.parse()?;
    }
    if let Some(value) = optional(&args, "--steer-windows") {
        config.steer_windows = value.parse()?;
    }
    if let Some(value) = optional(&args, "--stop-windows") {
        config.stop_windows = value.parse()?;
    }
    if has_flag(&args, "--enable-precursor-steering") {
        config.precursor_steering_enabled = true;
    }
    let controller =
        AdaptiveExamController::new(generator_version, run_id, seed, config, capabilities)?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &controller)?;
    println!();
    Ok(())
}

fn step(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    validate_flags(
        &args,
        &[
            "--state",
            "--observations",
            "--tier",
            "--checkpoint-step",
            "--checkpoint-hash",
            "--samples",
        ],
        &[],
    )?;
    let state_path = required(&args, "--state")?;
    let observations_path = required(&args, "--observations")?;
    let tier = parse_tier(required(&args, "--tier")?)?;
    let checkpoint = CheckpointRef {
        step: required(&args, "--checkpoint-step")?.parse()?,
        artifact_hash: required(&args, "--checkpoint-hash")?.into(),
    };
    let samples = required(&args, "--samples")?.parse()?;
    let mut controller: AdaptiveExamController = serde_json::from_reader(File::open(state_path)?)?;
    let mut decisions = Vec::new();
    for (index, line) in BufReader::new(File::open(observations_path)?)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let observation: ExamObservation = serde_json::from_str(&line)
            .map_err(|error| format!("observation line {}: {error}", index + 1))?;
        decisions.push(controller.observe(observation)?);
    }
    let next_plan = controller.next_plan(tier, checkpoint, samples)?;
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &StepOutput {
            controller,
            decisions,
            next_plan,
        },
    )?;
    println!();
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

fn optional<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|value| value == flag)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|value| value == flag)
}

fn validate_flags(
    args: &[String],
    value_flags: &[&str],
    boolean_flags: &[&str],
) -> Result<(), Box<dyn Error>> {
    let mut seen = BTreeSet::new();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if !seen.insert(flag) {
            return Err(format!("{flag} was supplied more than once").into());
        }
        if boolean_flags.contains(&flag) {
            index += 1;
            continue;
        }
        if !value_flags.contains(&flag) {
            return Err(format!("unknown argument {flag:?}").into());
        }
        if args
            .get(index + 1)
            .is_none_or(|value| value.starts_with("--"))
        {
            return Err(format!("{flag} requires a value").into());
        }
        index += 2;
    }
    Ok(())
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
