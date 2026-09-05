//! V1 clean-line OCR gate for the compact native vision stem.
//!
//! The corpus is a training-domain sibling, never an exam lane. This rung
//! deliberately asks whether a 2--10M shared decoder can overfit at least 128
//! clean rendered lines before any larger or more varied run is attempted.

use burn::module::Module;
use burn::optim::Optimizer;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use grug_data::{VisionCorpusDomain, VisionTextGenerator, VisionTextPrediction, score_vision_text};
use grug_train::train::Batch;
use grug_train::{
    ByteTokenizer, CompactConvVisionStemConfig, CompactVisionTransformerConfig,
    TinyTransformerConfig, build_vision_ocr_batch, generate_compact_vision_ocr,
    init_compact_vision_training, rgb_batch_to_nchw, to_eval_compact_vision_model,
    train_compact_vision_step,
};
use serde::Serialize;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(any(feature = "gpu", feature = "cuda"))]
type TrainingBackend = grug_train::GpuAutodiffBackend;
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
type TrainingBackend = grug_train::CpuAutodiffBackend;

#[cfg(feature = "cuda")]
const BACKEND_NAME: &str = "cuda";
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
const BACKEND_NAME: &str = "rocm";
#[cfg(not(any(feature = "gpu", feature = "cuda")))]
const BACKEND_NAME: &str = "cpu-ndarray";

struct PreparedBatch<B: AutodiffBackend> {
    images: burn::tensor::Tensor<B, 4>,
    text: Batch<B>,
}

#[derive(Debug, Serialize)]
struct FailureExample {
    case_id: String,
    expected: String,
    predicted: String,
}

#[derive(Debug, Serialize)]
struct GateReport {
    kind: &'static str,
    backend: &'static str,
    corpus_domain: &'static str,
    corpus_seed: u64,
    difficulty: u8,
    corruption: &'static str,
    cases: usize,
    batch_size: usize,
    steps: usize,
    checkpoint_every: usize,
    progress_checkpoints_saved: usize,
    learning_rate_schedule: &'static str,
    peak_learning_rate: f64,
    minimum_learning_rate: f64,
    parameters: usize,
    max_visual_tokens: usize,
    visual_fraction_of_context: f64,
    max_transcript_bytes: usize,
    training_seconds: f64,
    inference_seconds: f64,
    initial_loss: f32,
    final_loss: f32,
    loss_decreased: bool,
    exact_rate: f64,
    character_error_rate: f64,
    word_error_rate: f64,
    blank_rate: f64,
    metamorphic_consistency: f64,
    unique_semantic_prediction_rate: f64,
    cross_semantic_collapse_rate: f64,
    required_cases: usize,
    required_exact_rate: f64,
    maximum_collapse_rate: f64,
    v1_gate_passed: bool,
    checkpoint_saved: bool,
    failure_examples: Vec<FailureExample>,
}

#[derive(Debug, Serialize)]
struct CheckpointMetadata {
    schema_version: u32,
    architecture: &'static str,
    backend: &'static str,
    steps: usize,
    parameters: usize,
    corpus_seed: u64,
    cases: usize,
    batch_size: usize,
    checkpoint_every: usize,
    training_complete: bool,
    learning_rate_schedule: &'static str,
    peak_learning_rate: f64,
    minimum_learning_rate: f64,
    transformer: TransformerMetadata,
    vision: VisionMetadata,
    exact_rate: f64,
    character_error_rate: f64,
    cross_semantic_collapse_rate: f64,
    v1_gate_passed: bool,
}

#[derive(Debug, Serialize)]
struct ProgressCheckpointMetadata {
    schema_version: u32,
    architecture: &'static str,
    backend: &'static str,
    step: usize,
    total_steps: usize,
    parameters: usize,
    corpus_seed: u64,
    cases: usize,
    batch_size: usize,
    checkpoint_every: usize,
    training_complete: bool,
    learning_rate_schedule: &'static str,
    peak_learning_rate: f64,
    minimum_learning_rate: f64,
    current_learning_rate: f64,
    current_loss: f32,
}

#[derive(Debug, Serialize)]
struct TransformerMetadata {
    vocab_size: usize,
    d_model: usize,
    heads: usize,
    layers: usize,
    feed_forward: usize,
    max_sequence: usize,
}

#[derive(Debug, Serialize)]
struct VisionMetadata {
    input_channels: usize,
    hidden_channels: usize,
    output_stride: usize,
    max_grid_rows: usize,
    max_grid_columns: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    const CORPUS_SEED: u64 = 51_100;
    const DIFFICULTY: u8 = 1;
    const TEXT_LIMIT: usize = 144;
    const MODEL_LIMIT: usize = 512;
    const REQUIRED_CASES: usize = 128;
    const REQUIRED_EXACT: f64 = 0.98;
    const MAXIMUM_COLLAPSE: f64 = 0.01;

    let steps = argument("--steps")?.unwrap_or(4_000usize);
    let case_count = argument("--cases")?.unwrap_or(REQUIRED_CASES);
    let batch_size = argument("--batch-size")?.unwrap_or(32usize);
    let checkpoint_every = argument("--checkpoint-every")?.unwrap_or(500usize);
    let peak_learning_rate = argument("--learning-rate")?.unwrap_or(1e-3f64);
    let minimum_learning_rate = argument("--min-learning-rate")?.unwrap_or(1e-5f64);
    let artifacts_dir = argument::<String>("--artifacts-dir")?.map(PathBuf::from);
    if steps == 0 || case_count == 0 || batch_size == 0 {
        return Err("--steps, --cases and --batch-size must be positive".into());
    }
    if !(minimum_learning_rate > 0.0 && minimum_learning_rate <= peak_learning_rate) {
        return Err("learning rates must satisfy 0 < minimum <= peak".into());
    }

    let generator = VisionTextGenerator;
    let mut rendered = Vec::with_capacity(case_count);
    for index in 0..case_count {
        // Multiples of 15 select the English clean-line family while retaining
        // distinct item markers and domain-separated content selection.
        let item_index = (index * 15) as u64;
        let spec = generator.spec(
            CORPUS_SEED,
            item_index,
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
        return Err("clean-line gate requires equal image geometry".into());
    }
    let max_visual_tokens = height.div_ceil(16) * width.div_ceil(16);
    let max_transcript_bytes = rendered
        .iter()
        .map(|case| case.manifest.transcript.len())
        .max()
        .unwrap_or(0);

    #[cfg(any(feature = "gpu", feature = "cuda"))]
    let device = grug_train::gpu_device();
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    let device = Default::default();
    type B = TrainingBackend;
    B::seed(&device, CORPUS_SEED);
    let tokenizer = ByteTokenizer::new();
    let prepared: Vec<PreparedBatch<B>> = rendered
        .chunks(batch_size)
        .map(|chunk| {
            let images: Vec<Vec<u8>> = chunk.iter().map(|case| case.rgb.clone()).collect();
            let transcripts: Vec<String> = chunk
                .iter()
                .map(|case| case.manifest.transcript.clone())
                .collect();
            Ok(PreparedBatch {
                images: rgb_batch_to_nchw::<B>(&images, width, height, &device)?,
                text: build_vision_ocr_batch::<B, _>(
                    &tokenizer,
                    &transcripts,
                    TEXT_LIMIT,
                    &device,
                )?,
            })
        })
        .collect::<Result<_, Box<dyn Error>>>()?;

    let config = CompactVisionTransformerConfig::new(
        TinyTransformerConfig::new(tokenizer.vocab_size(), 192, 6, 6, 768, MODEL_LIMIT),
        CompactConvVisionStemConfig::new(3, 24, 192, 4, 24),
    );
    let (mut model, mut optimizer) = init_compact_vision_training::<B>(&config, &device);
    let parameters = model.num_params();
    if !(2_000_000..=10_000_000).contains(&parameters) {
        return Err(format!("V1 model has {parameters} parameters; expected 2--10M").into());
    }

    let training_started = Instant::now();
    let mut initial_loss = None;
    let mut final_loss = f32::NAN;
    let mut progress_checkpoints_saved = 0usize;
    for step in 0..steps {
        let batch = &prepared[step % prepared.len()];
        let learning_rate =
            cosine_learning_rate(step, steps, peak_learning_rate, minimum_learning_rate);
        let (next, loss) = train_compact_vision_step(
            model,
            &mut optimizer,
            batch.images.clone(),
            &batch.text,
            learning_rate,
        );
        model = next;
        if !loss.is_finite() {
            return Err(format!("non-finite compact OCR loss at step {}", step + 1).into());
        }
        initial_loss.get_or_insert(loss);
        final_loss = loss;
        if step == 0 || (step + 1) % 50 == 0 || step + 1 == steps {
            eprintln!(
                "step={} loss={loss:.6} lr={learning_rate:.8} elapsed={:.1}s",
                step + 1,
                training_started.elapsed().as_secs_f64()
            );
        }
        if let Some(root) = artifacts_dir.as_ref()
            && checkpoint_every > 0
            && (step + 1) % checkpoint_every == 0
            && step + 1 < steps
        {
            let checkpoint_dir = root.join("progress").join(format!("step-{:06}", step + 1));
            std::fs::create_dir_all(&checkpoint_dir)?;
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            model
                .clone()
                .save_file(checkpoint_dir.join("model"), &recorder)?;
            recorder.record(optimizer.to_record(), checkpoint_dir.join("optim"))?;
            let metadata = ProgressCheckpointMetadata {
                schema_version: 1,
                architecture: "compact_shared_decoder_v1",
                backend: BACKEND_NAME,
                step: step + 1,
                total_steps: steps,
                parameters,
                corpus_seed: CORPUS_SEED,
                cases: case_count,
                batch_size,
                checkpoint_every,
                training_complete: false,
                learning_rate_schedule: "cosine",
                peak_learning_rate,
                minimum_learning_rate,
                current_learning_rate: learning_rate,
                current_loss: loss,
            };
            // Metadata is written last and marks a complete resumable snapshot;
            // an interrupted directory without it is never treated as valid.
            std::fs::write(
                checkpoint_dir.join("meta.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
            progress_checkpoints_saved += 1;
            eprintln!("checkpoint={}", checkpoint_dir.display());
        }
    }
    let training_seconds = training_started.elapsed().as_secs_f64();
    let initial_loss = initial_loss.unwrap_or(final_loss);
    if !final_loss.is_finite() || final_loss >= initial_loss {
        return Err(format!(
            "compact OCR loss did not decrease: {initial_loss:.6} -> {final_loss:.6}"
        )
        .into());
    }

    let eval_model = to_eval_compact_vision_model(&model);
    let generation_limit = max_transcript_bytes.saturating_add(8);
    let inference_started = Instant::now();
    let predictions: Vec<VisionTextPrediction> = rendered
        .iter()
        .map(|case| {
            let image = rgb_batch_to_nchw::<<B as AutodiffBackend>::InnerBackend>(
                std::slice::from_ref(&case.rgb),
                width,
                height,
                &device,
            )?;
            Ok(VisionTextPrediction {
                case_id: case.manifest.case_id.clone(),
                text: generate_compact_vision_ocr(
                    &eval_model,
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
    let inference_seconds = inference_started.elapsed().as_secs_f64();
    let manifests: Vec<_> = rendered.iter().map(|case| case.manifest.clone()).collect();
    let score = score_vision_text(&manifests, &predictions)?;
    let failure_examples: Vec<_> = manifests
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
    let v1_gate_passed = case_count >= REQUIRED_CASES
        && score.overall.exact_rate >= REQUIRED_EXACT
        && score.cross_semantic_collapse_rate <= MAXIMUM_COLLAPSE;
    let checkpoint_saved = if let Some(dir) = artifacts_dir {
        std::fs::create_dir_all(&dir)?;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.clone().save_file(dir.join("model"), &recorder)?;
        recorder.record(optimizer.to_record(), dir.join("optim"))?;
        let metadata = CheckpointMetadata {
            schema_version: 1,
            architecture: "compact_shared_decoder_v1",
            backend: BACKEND_NAME,
            steps,
            parameters,
            corpus_seed: CORPUS_SEED,
            cases: case_count,
            batch_size,
            checkpoint_every,
            training_complete: true,
            learning_rate_schedule: "cosine",
            peak_learning_rate,
            minimum_learning_rate,
            transformer: TransformerMetadata {
                vocab_size: tokenizer.vocab_size(),
                d_model: 192,
                heads: 6,
                layers: 6,
                feed_forward: 768,
                max_sequence: MODEL_LIMIT,
            },
            vision: VisionMetadata {
                input_channels: 3,
                hidden_channels: 24,
                output_stride: CompactConvVisionStemConfig::OUTPUT_STRIDE,
                max_grid_rows: 4,
                max_grid_columns: 24,
            },
            exact_rate: score.overall.exact_rate,
            character_error_rate: score.overall.character_error_rate,
            cross_semantic_collapse_rate: score.cross_semantic_collapse_rate,
            v1_gate_passed,
        };
        std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&metadata)?)?;
        true
    } else {
        false
    };

    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &GateReport {
            kind: "compact_shared_decoder_clean_line_overfit",
            backend: BACKEND_NAME,
            corpus_domain: "training_sibling",
            corpus_seed: CORPUS_SEED,
            difficulty: DIFFICULTY,
            corruption: "clean",
            cases: case_count,
            batch_size,
            steps,
            checkpoint_every,
            progress_checkpoints_saved,
            learning_rate_schedule: "cosine",
            peak_learning_rate,
            minimum_learning_rate,
            parameters,
            max_visual_tokens,
            visual_fraction_of_context: max_visual_tokens as f64 / MODEL_LIMIT as f64,
            max_transcript_bytes,
            training_seconds,
            inference_seconds,
            initial_loss,
            final_loss,
            loss_decreased: final_loss < initial_loss,
            exact_rate: score.overall.exact_rate,
            character_error_rate: score.overall.character_error_rate,
            word_error_rate: score.overall.word_error_rate,
            blank_rate: score.overall.blank_rate,
            metamorphic_consistency: score.metamorphic_consistency,
            unique_semantic_prediction_rate: score.unique_semantic_prediction_rate,
            cross_semantic_collapse_rate: score.cross_semantic_collapse_rate,
            required_cases: REQUIRED_CASES,
            required_exact_rate: REQUIRED_EXACT,
            maximum_collapse_rate: MAXIMUM_COLLAPSE,
            v1_gate_passed,
            checkpoint_saved,
            failure_examples,
        },
    )?;
    println!();
    Ok(())
}

fn cosine_learning_rate(step: usize, total_steps: usize, peak: f64, minimum: f64) -> f64 {
    if total_steps <= 1 {
        return minimum;
    }
    let progress = step as f64 / (total_steps - 1) as f64;
    let cosine = (std::f64::consts::PI * progress).cos();
    minimum + 0.5 * (peak - minimum) * (1.0 + cosine)
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

#[cfg(test)]
mod tests {
    use super::cosine_learning_rate;

    #[test]
    fn cosine_schedule_reaches_both_endpoints_and_is_monotonic() {
        let rates: Vec<_> = (0..10)
            .map(|step| cosine_learning_rate(step, 10, 1e-3, 1e-5))
            .collect();

        assert!((rates[0] - 1e-3).abs() < f64::EPSILON);
        assert!((rates[9] - 1e-5).abs() < f64::EPSILON);
        assert!(rates.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn one_step_schedule_uses_the_minimum_rate() {
        assert_eq!(cosine_learning_rate(0, 1, 1e-3, 1e-5), 1e-5);
    }
}
