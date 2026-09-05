//! End-to-end CPU smoke: render -> patch -> shared decoder -> train -> decode.
//!
//! This is a wiring/optimization proof, not the V1 128-case promotion gate.

use burn::tensor::backend::Backend;
use grug_data::{VisionCorpusDomain, VisionTextGenerator, VisionTextPrediction, score_vision_text};
use grug_train::{
    ByteTokenizer, CpuAutodiffBackend, NativeVisionTransformerConfig, TinyTransformerConfig,
    VisionPatchStemConfig, build_vision_ocr_batch, generate_vision_ocr, init_vision_training,
    patchify_rgb, stack_patch_grids, to_eval_vision_model, train_vision_step,
};
use serde::Serialize;
use std::error::Error;

#[derive(Serialize)]
struct SmokeReport {
    cases: usize,
    steps: usize,
    parameters: usize,
    initial_loss: f32,
    final_loss: f32,
    loss_decreased: bool,
    exact_rate: f64,
    character_error_rate: f64,
    unique_semantic_prediction_rate: f64,
    cross_semantic_collapse_rate: f64,
    v1_gate_passed: bool,
    predictions: Vec<VisionTextPrediction>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let steps = argument("--steps")?.unwrap_or(100usize);
    let case_count = argument("--cases")?.unwrap_or(3usize);
    let learning_rate = argument("--learning-rate")?.unwrap_or(1e-3f64);
    if steps == 0 || case_count == 0 {
        return Err("--steps and --cases must be positive".into());
    }

    let generator = VisionTextGenerator;
    let mut rendered = Vec::with_capacity(case_count);
    for index in 0..case_count {
        // Every 15th item is an English single-line layout. Keeping dimensions
        // equal lets this smoke isolate model wiring from dynamic tiling.
        let item_index = (index * 15) as u64;
        let spec = generator.spec(51, item_index, VisionCorpusDomain::TrainingSibling, 1)?;
        rendered.push(generator.render(&spec, 0, format!("memory://{index}"))?);
    }
    let grids: Vec<_> = rendered
        .iter()
        .map(|case| {
            patchify_rgb(
                &case.rgb,
                case.manifest.width as usize,
                case.manifest.height as usize,
                16,
            )
        })
        .collect::<Result<_, _>>()?;
    let rows = grids[0].rows;
    let columns = grids[0].columns;
    let transcripts: Vec<String> = rendered
        .iter()
        .map(|case| case.manifest.transcript.clone())
        .collect();

    let device = Default::default();
    CpuAutodiffBackend::seed(&device, 51);
    let patches = stack_patch_grids::<CpuAutodiffBackend>(&grids, &device)?;
    let tokenizer = ByteTokenizer::new();
    let text_batch =
        build_vision_ocr_batch::<CpuAutodiffBackend, _>(&tokenizer, &transcripts, 144, &device)?;
    let config = NativeVisionTransformerConfig::new(
        TinyTransformerConfig::new(tokenizer.vocab_size(), 64, 4, 2, 256, 256),
        VisionPatchStemConfig::new(3, 16, 64, 4, 24),
    );
    let (mut model, mut optimizer) = init_vision_training::<CpuAutodiffBackend>(&config, &device);
    let parameters = model.num_params();
    let mut initial_loss = None;
    let mut final_loss = f32::NAN;
    for step in 0..steps {
        let (next, loss) = train_vision_step(
            model,
            &mut optimizer,
            patches.clone(),
            rows,
            columns,
            &text_batch,
            learning_rate,
        );
        model = next;
        initial_loss.get_or_insert(loss);
        final_loss = loss;
        if step % 25 == 0 || step + 1 == steps {
            eprintln!("step={} loss={loss:.6}", step + 1);
        }
    }
    let initial_loss = initial_loss.unwrap_or(final_loss);
    if !final_loss.is_finite() || final_loss >= initial_loss {
        return Err(format!(
            "vision smoke loss did not decrease: {initial_loss:.6} -> {final_loss:.6}"
        )
        .into());
    }

    let model = to_eval_vision_model(&model);
    let predictions: Vec<VisionTextPrediction> = grids
        .iter()
        .zip(&rendered)
        .map(|(grid, case)| VisionTextPrediction {
            case_id: case.manifest.case_id.clone(),
            text: generate_vision_ocr(
                &model,
                grid.clone().into_tensor(&device),
                grid.rows,
                grid.columns,
                &tokenizer,
                96,
                256,
                &device,
            ),
            confidence: None,
        })
        .collect();
    let manifests: Vec<_> = rendered.into_iter().map(|case| case.manifest).collect();
    let score = score_vision_text(&manifests, &predictions)?;
    let v1_gate_passed =
        score.overall.exact_rate >= 0.98 && score.cross_semantic_collapse_rate <= 0.01;
    if !v1_gate_passed {
        eprintln!(
            "capability gate remains red: exact={:.3} cer={:.3} collapse={:.3}",
            score.overall.exact_rate,
            score.overall.character_error_rate,
            score.cross_semantic_collapse_rate,
        );
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &SmokeReport {
            cases: case_count,
            steps,
            parameters,
            initial_loss,
            final_loss,
            loss_decreased: true,
            exact_rate: score.overall.exact_rate,
            character_error_rate: score.overall.character_error_rate,
            unique_semantic_prediction_rate: score.unique_semantic_prediction_rate,
            cross_semantic_collapse_rate: score.cross_semantic_collapse_rate,
            v1_gate_passed,
            predictions,
        },
    )?;
    println!();
    Ok(())
}

fn argument<T: std::str::FromStr>(flag: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T::Err: std::error::Error + 'static,
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
