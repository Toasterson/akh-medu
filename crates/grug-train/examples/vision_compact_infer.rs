//! Offline CPU replay for the compact native OCR checkpoint.
//!
//! This deliberately has no provider or network client. It rebuilds the
//! compatibility tuple, loads `model.mpk`, renders deterministic local cases,
//! and measures greedy pixel-to-text inference through the shared decoder.

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use grug_data::{VisionCorpusDomain, VisionTextGenerator, VisionTextPrediction, score_vision_text};
use grug_train::{
    ByteTokenizer, CompactConvVisionStemConfig, CompactVisionTransformerConfig, CpuBackend,
    TinyTransformerConfig, generate_compact_vision_ocr, rgb_batch_to_nchw,
};
use serde::Serialize;
use serde_json::Value;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Serialize)]
struct FailureExample {
    case_id: String,
    expected: String,
    predicted: String,
}

#[derive(Debug, Serialize)]
struct InferenceReport {
    kind: &'static str,
    backend: &'static str,
    network_required: bool,
    checkpoint: String,
    cases: usize,
    parameters: usize,
    max_visual_tokens: usize,
    visual_fraction_of_context: f64,
    inference_seconds: f64,
    seconds_per_case: f64,
    exact_rate: f64,
    character_error_rate: f64,
    word_error_rate: f64,
    blank_rate: f64,
    unique_semantic_prediction_rate: f64,
    cross_semantic_collapse_rate: f64,
    failure_examples: Vec<FailureExample>,
}

fn main() -> Result<(), Box<dyn Error>> {
    const CORPUS_SEED: u64 = 51_100;
    const DIFFICULTY: u8 = 1;
    const MODEL_LIMIT: usize = 512;

    let checkpoint = required_argument::<String>("--checkpoint").map(PathBuf::from)?;
    let case_count = argument("--cases")?.unwrap_or(3usize);
    if case_count == 0 {
        return Err("--cases must be positive".into());
    }
    validate_checkpoint_metadata(&checkpoint)?;

    let generator = VisionTextGenerator;
    let mut rendered = Vec::with_capacity(case_count);
    for index in 0..case_count {
        let spec = generator.spec(
            CORPUS_SEED,
            (index * 15) as u64,
            VisionCorpusDomain::TrainingSibling,
            DIFFICULTY,
        )?;
        rendered.push(generator.render(&spec, 0, format!("memory://{index}"))?);
    }
    let width = rendered[0].manifest.width as usize;
    let height = rendered[0].manifest.height as usize;
    if rendered.iter().any(|case| {
        case.manifest.width as usize != width || case.manifest.height as usize != height
    }) {
        return Err("clean-line replay requires equal image geometry".into());
    }
    let max_visual_tokens = height.div_ceil(16) * width.div_ceil(16);
    let generation_limit = rendered
        .iter()
        .map(|case| case.manifest.transcript.len())
        .max()
        .unwrap_or(0)
        .saturating_add(8);

    let device = Default::default();
    let tokenizer = ByteTokenizer::new();
    let config = CompactVisionTransformerConfig::new(
        TinyTransformerConfig::new(tokenizer.vocab_size(), 192, 6, 6, 768, MODEL_LIMIT),
        CompactConvVisionStemConfig::new(3, 24, 192, 4, 24),
    );
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = config.init::<CpuBackend>(&device).load_file(
        checkpoint.join("model"),
        &recorder,
        &device,
    )?;
    let parameters = model.num_params();

    let started = Instant::now();
    let predictions: Vec<VisionTextPrediction> = rendered
        .iter()
        .map(|case| {
            let image = rgb_batch_to_nchw::<CpuBackend>(
                std::slice::from_ref(&case.rgb),
                width,
                height,
                &device,
            )?;
            Ok(VisionTextPrediction {
                case_id: case.manifest.case_id.clone(),
                text: generate_compact_vision_ocr(
                    &model,
                    image,
                    &tokenizer,
                    generation_limit,
                    MODEL_LIMIT,
                    &device,
                ),
                confidence: None,
            })
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    let inference_seconds = started.elapsed().as_secs_f64();
    let manifests: Vec<_> = rendered.iter().map(|case| case.manifest.clone()).collect();
    let score = score_vision_text(&manifests, &predictions)?;
    let failure_examples = manifests
        .iter()
        .zip(&predictions)
        .filter(|(manifest, prediction)| manifest.transcript != prediction.text)
        .take(20)
        .map(|(manifest, prediction)| FailureExample {
            case_id: manifest.case_id.clone(),
            expected: manifest.transcript.clone(),
            predicted: prediction.text.clone(),
        })
        .collect();

    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &InferenceReport {
            kind: "compact_shared_decoder_local_offline_replay",
            backend: "cpu-ndarray",
            network_required: false,
            checkpoint: checkpoint.display().to_string(),
            cases: case_count,
            parameters,
            max_visual_tokens,
            visual_fraction_of_context: max_visual_tokens as f64 / MODEL_LIMIT as f64,
            inference_seconds,
            seconds_per_case: inference_seconds / case_count as f64,
            exact_rate: score.overall.exact_rate,
            character_error_rate: score.overall.character_error_rate,
            word_error_rate: score.overall.word_error_rate,
            blank_rate: score.overall.blank_rate,
            unique_semantic_prediction_rate: score.unique_semantic_prediction_rate,
            cross_semantic_collapse_rate: score.cross_semantic_collapse_rate,
            failure_examples,
        },
    )?;
    println!();
    Ok(())
}

fn validate_checkpoint_metadata(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
    let metadata: Value = serde_json::from_slice(&std::fs::read(checkpoint.join("meta.json"))?)?;
    let expected = [
        ("architecture", Value::from("compact_shared_decoder_v1")),
        ("parameters", Value::from(3_310_340u64)),
    ];
    for (field, expected) in expected {
        if metadata.get(field) != Some(&expected) {
            return Err(format!(
                "checkpoint metadata {field:?} is {:?}; expected {expected:?}",
                metadata.get(field)
            )
            .into());
        }
    }
    if metadata.pointer("/transformer/max_sequence") != Some(&Value::from(512u64))
        || metadata.pointer("/vision/output_stride") != Some(&Value::from(16u64))
    {
        return Err("checkpoint transformer/stem compatibility tuple does not match V1".into());
    }
    Ok(())
}

fn required_argument<T: std::str::FromStr>(flag: &str) -> Result<T, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    argument(flag)?.ok_or_else(|| format!("{flag} is required").into())
}

fn argument<T: std::str::FromStr>(flag: &str) -> Result<Option<T>, Box<dyn Error>>
where
    T::Err: Error + 'static,
{
    let args: Vec<String> = std::env::args().collect();
    let Some(index) = args.iter().position(|value| value == flag) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    Ok(Some(value.parse()?))
}
