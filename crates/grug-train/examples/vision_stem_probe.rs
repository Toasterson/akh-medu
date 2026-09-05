//! Stem-only glyph-resolution probe required before decoder-scale OCR runs.
//!
//! A trainable patch projection, factorized positions and LayerNorm feed a
//! single linear classifier. There is no language-model decoder, so failures
//! isolate visual representation/readout from autoregressive optimization.

use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use grug_data::render_glyph_probe;
use grug_train::{
    CpuAutodiffBackend, PatchGrid, VisionPatchStem, VisionPatchStemConfig, patchify_rgb,
    stack_patch_grids,
};
use serde::Serialize;
use std::error::Error;

const CANVAS: usize = 48;
const GLYPHS: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

#[derive(Module, Debug)]
struct StemReadout<B: Backend> {
    stem: VisionPatchStem<B>,
    classifier: Linear<B>,
    feature_width: usize,
}

impl<B: Backend> StemReadout<B> {
    fn forward(&self, patches: Tensor<B, 3>, rows: usize, columns: usize) -> Tensor<B, 2> {
        let batch = patches.dims()[0];
        let encoded = self
            .stem
            .forward(patches, rows, columns)
            .reshape([batch, self.feature_width]);
        self.classifier.forward(encoded)
    }
}

#[derive(Debug, Serialize)]
struct ProbeResult {
    glyph_pixels: usize,
    patch_size: usize,
    visual_tokens: usize,
    parameters: usize,
    steps: usize,
    initial_loss: f32,
    final_loss: f32,
    training_position_accuracy: f64,
    held_out_position_accuracy: f64,
    gate_passed: bool,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    kind: &'static str,
    renderer: &'static str,
    canvas: [usize; 2],
    classes: usize,
    held_out_gate: f64,
    results: Vec<ProbeResult>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let steps = argument("--steps")?.unwrap_or(120usize);
    let learning_rate = argument("--learning-rate")?.unwrap_or(3e-3f64);
    let glyph_scale_filter = argument::<usize>("--glyph-scale")?;
    let patch_size_filter = argument::<usize>("--patch-size")?;
    if steps == 0 {
        return Err("--steps must be positive".into());
    }
    if glyph_scale_filter.is_some_and(|scale| ![1, 2].contains(&scale)) {
        return Err("--glyph-scale must be 1 or 2".into());
    }
    if patch_size_filter.is_some_and(|size| ![8, 16, 24, 32, 48].contains(&size)) {
        return Err("--patch-size must be one of 8,16,24,32,48".into());
    }
    let mut results = Vec::new();
    for glyph_scale in [1usize, 2] {
        if glyph_scale_filter.is_some_and(|filter| filter != glyph_scale) {
            continue;
        }
        for patch_size in [8usize, 16, 24, 32, 48] {
            if patch_size_filter.is_some_and(|filter| filter != patch_size) {
                continue;
            }
            eprintln!("probe glyph={}px patch={}px", glyph_scale * 8, patch_size);
            results.push(run_probe(glyph_scale, patch_size, steps, learning_rate)?);
        }
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &ProbeReport {
            kind: "stem_plus_linear_glyph_classification",
            renderer: grug_data::vision_text::VISION_TEXT_RENDERER_ID,
            canvas: [CANVAS, CANVAS],
            classes: GLYPHS.chars().count(),
            held_out_gate: 0.95,
            results,
        },
    )?;
    println!();
    Ok(())
}

fn run_probe(
    glyph_scale: usize,
    patch_size: usize,
    steps: usize,
    learning_rate: f64,
) -> Result<ProbeResult, Box<dyn Error>> {
    type B = CpuAutodiffBackend;
    let device = Default::default();
    B::seed(&device, (glyph_scale as u64) * 10_000 + patch_size as u64);
    let characters: Vec<char> = GLYPHS.chars().collect();
    let train_offsets = training_offsets(glyph_scale);
    let held_out_offsets = held_out_offsets(glyph_scale);
    let train_sets: Vec<Vec<PatchGrid>> = train_offsets
        .iter()
        .map(|&(x, y)| grids(&characters, glyph_scale, patch_size, x, y))
        .collect::<Result<_, _>>()?;
    let train_all: Vec<PatchGrid> = train_sets.iter().flatten().cloned().collect();
    let held_out_all: Vec<PatchGrid> = held_out_offsets
        .iter()
        .map(|&(x, y)| grids(&characters, glyph_scale, patch_size, x, y))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let rows = train_all[0].rows;
    let columns = train_all[0].columns;
    let d_model = 32usize;
    let feature_width = rows * columns * d_model;
    let model = StemReadout {
        stem: VisionPatchStemConfig::new(3, patch_size, d_model, rows, columns).init(&device),
        classifier: LinearConfig::new(feature_width, characters.len()).init(&device),
        feature_width,
    };
    let parameters = model.num_params();
    let mut model = model;
    let mut optimizer: OptimizerAdaptor<Adam, StemReadout<B>, B> = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let targets = class_targets::<B>(characters.len(), &device);
    let mut initial_loss = None;
    let mut final_loss = f32::NAN;
    for step in 0..steps {
        let patches = stack_patch_grids::<B>(&train_sets[step % train_sets.len()], &device)?;
        let logits = model.forward(patches, rows, columns);
        let loss = loss_fn.forward(logits, targets.clone());
        let value: f32 = loss.clone().into_scalar().elem();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, gradients);
        initial_loss.get_or_insert(value);
        final_loss = value;
    }
    let training_position_accuracy = accuracy(
        model.forward(stack_patch_grids::<B>(&train_all, &device)?, rows, columns),
        characters.len(),
        train_offsets.len(),
    )?;
    let held_out_position_accuracy = accuracy(
        model.forward(
            stack_patch_grids::<B>(&held_out_all, &device)?,
            rows,
            columns,
        ),
        characters.len(),
        held_out_offsets.len(),
    )?;
    Ok(ProbeResult {
        glyph_pixels: glyph_scale * 8,
        patch_size,
        visual_tokens: rows * columns,
        parameters,
        steps,
        initial_loss: initial_loss.unwrap_or(final_loss),
        final_loss,
        training_position_accuracy,
        held_out_position_accuracy,
        gate_passed: held_out_position_accuracy >= 0.95,
    })
}

fn grids(
    characters: &[char],
    glyph_scale: usize,
    patch_size: usize,
    offset_x: usize,
    offset_y: usize,
) -> Result<Vec<PatchGrid>, Box<dyn Error>> {
    characters
        .iter()
        .map(|&character| {
            let rgb =
                render_glyph_probe(character, glyph_scale, offset_x, offset_y, CANVAS, CANVAS)?;
            Ok(patchify_rgb(&rgb, CANVAS, CANVAS, patch_size)?)
        })
        .collect()
}

fn training_offsets(scale: usize) -> [(usize, usize); 4] {
    let centre = (CANVAS - 8 * scale) / 2;
    [
        (centre - 2, centre),
        (centre + 2, centre),
        (centre, centre - 2),
        (centre, centre + 2),
    ]
}

fn held_out_offsets(scale: usize) -> [(usize, usize); 2] {
    let centre = (CANVAS - 8 * scale) / 2;
    [(centre, centre), (centre + 3, centre - 1)]
}

fn class_targets<B: Backend>(classes: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::from_data(
        TensorData::new((0..classes as i64).collect(), [classes]),
        device,
    )
}

fn accuracy<B: Backend>(
    logits: Tensor<B, 2>,
    classes: usize,
    repetitions: usize,
) -> Result<f64, Box<dyn Error>> {
    let values: Vec<f32> = logits.into_data().to_vec()?;
    let mut correct = 0usize;
    for (index, row) in values.chunks_exact(classes).enumerate() {
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(class, _)| class)
            .ok_or("empty classifier row")?;
        if predicted == index % classes {
            correct += 1;
        }
    }
    Ok(correct as f64 / (classes * repetitions) as f64)
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
