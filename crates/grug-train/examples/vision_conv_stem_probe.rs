//! Compare the compact convolutional fallback against the failed raw-patch
//! held-out-position glyph gate, without involving a language-model decoder.

use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use grug_data::render_glyph_probe;
use grug_train::{CompactConvVisionStem, CompactConvVisionStemConfig, rgb_batch_to_nchw};
use serde::Serialize;
use std::error::Error;
use std::time::Instant;

const CANVAS: usize = 48;
const GLYPHS: &str = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
type ProbeBackend = grug_train::GpuAutodiffBackend;
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
type ProbeBackend = grug_train::CpuAutodiffBackend;

#[cfg(feature = "cuda")]
const BACKEND_NAME: &str = "cuda";
#[cfg(all(feature = "gpu", not(feature = "cuda")))]
const BACKEND_NAME: &str = "rocm";
#[cfg(all(feature = "metal", not(feature = "gpu"), not(feature = "cuda")))]
const BACKEND_NAME: &str = "metal";
#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
const BACKEND_NAME: &str = "cpu-ndarray";

#[derive(Module, Debug)]
struct ConvStemReadout<B: Backend> {
    stem: CompactConvVisionStem<B>,
    classifier: Linear<B>,
    d_model: usize,
}

impl<B: Backend> ConvStemReadout<B> {
    fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch = images.dims()[0];
        let pooled = self
            .stem
            .forward(images)
            .mean_dim(1)
            .reshape([batch, self.d_model]);
        self.classifier.forward(pooled)
    }
}

#[derive(Debug, Serialize)]
struct ProbeResult {
    glyph_pixels: usize,
    output_stride: usize,
    visual_tokens: usize,
    parameters: usize,
    steps: usize,
    elapsed_seconds: f64,
    initial_loss: f32,
    final_loss: f32,
    training_position_accuracy: f64,
    held_out_position_accuracy: f64,
    gate_passed: bool,
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    kind: &'static str,
    backend: &'static str,
    renderer: &'static str,
    canvas: [usize; 2],
    classes: usize,
    training_positions_per_class: usize,
    held_out_positions_per_class: usize,
    held_out_gate: f64,
    results: Vec<ProbeResult>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let steps = argument("--steps")?.unwrap_or(240usize);
    let learning_rate = argument("--learning-rate")?.unwrap_or(3e-3f64);
    let glyph_scale_filter = argument::<usize>("--glyph-scale")?;
    if steps == 0 {
        return Err("--steps must be positive".into());
    }
    if glyph_scale_filter.is_some_and(|scale| ![1, 2].contains(&scale)) {
        return Err("--glyph-scale must be 1 or 2".into());
    }
    let mut results = Vec::new();
    for glyph_scale in [1usize, 2] {
        if glyph_scale_filter.is_some_and(|filter| filter != glyph_scale) {
            continue;
        }
        eprintln!("compact-conv probe glyph={}px", glyph_scale * 8);
        results.push(run_probe(glyph_scale, steps, learning_rate)?);
    }
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &ProbeReport {
            kind: "compact_conv_stem_plus_global_linear_glyph_classification",
            backend: BACKEND_NAME,
            renderer: grug_data::vision_text::VISION_TEXT_RENDERER_ID,
            canvas: [CANVAS, CANVAS],
            classes: GLYPHS.chars().count(),
            training_positions_per_class: training_offsets(1).len(),
            held_out_positions_per_class: held_out_offsets(1).len(),
            held_out_gate: 0.95,
            results,
        },
    )?;
    println!();
    Ok(())
}

fn run_probe(
    glyph_scale: usize,
    steps: usize,
    learning_rate: f64,
) -> Result<ProbeResult, Box<dyn Error>> {
    type B = ProbeBackend;
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
    let device = grug_train::gpu_device();
    #[cfg(not(any(feature = "gpu", feature = "cuda", feature = "metal")))]
    let device = Default::default();
    B::seed(&device, 90_000 + glyph_scale as u64);
    let characters: Vec<char> = GLYPHS.chars().collect();
    let train_offsets = training_offsets(glyph_scale);
    let held_out_offsets = held_out_offsets(glyph_scale);
    if train_offsets
        .iter()
        .any(|offset| held_out_offsets.contains(offset))
    {
        return Err("training and held-out translation sets overlap".into());
    }
    let train_sets: Vec<Vec<Vec<u8>>> = train_offsets
        .iter()
        .map(|&(x, y)| images(&characters, glyph_scale, x, y))
        .collect::<Result<_, _>>()?;
    let train_all: Vec<Vec<u8>> = train_sets.iter().flatten().cloned().collect();
    let held_out_all: Vec<Vec<u8>> = held_out_offsets
        .iter()
        .map(|&(x, y)| images(&characters, glyph_scale, x, y))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let d_model = 32usize;
    let model = ConvStemReadout {
        stem: CompactConvVisionStemConfig::new(3, 8, d_model, 3, 3).init(&device),
        classifier: LinearConfig::new(d_model, characters.len()).init(&device),
        d_model,
    };
    let parameters = model.num_params();
    let mut model = model;
    let mut optimizer: OptimizerAdaptor<Adam, ConvStemReadout<B>, B> = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let targets = class_targets::<B>(characters.len(), &device);
    let mut initial_loss = None;
    let mut final_loss = f32::NAN;
    let started = Instant::now();
    for step in 0..steps {
        let images = rgb_batch_to_nchw::<B>(
            &train_sets[step % train_sets.len()],
            CANVAS,
            CANVAS,
            &device,
        )?;
        let loss = loss_fn.forward(model.forward(images), targets.clone());
        let value: f32 = loss.clone().into_scalar().elem();
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, gradients);
        initial_loss.get_or_insert(value);
        final_loss = value;
        if step == 0 || (step + 1) % 10 == 0 || step + 1 == steps {
            eprintln!(
                "step={} loss={value:.6} elapsed={:.1}s",
                step + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let training_position_accuracy = accuracy(
        model.forward(rgb_batch_to_nchw::<B>(&train_all, CANVAS, CANVAS, &device)?),
        characters.len(),
        train_offsets.len(),
    )?;
    let held_out_position_accuracy = accuracy(
        model.forward(rgb_batch_to_nchw::<B>(
            &held_out_all,
            CANVAS,
            CANVAS,
            &device,
        )?),
        characters.len(),
        held_out_offsets.len(),
    )?;
    Ok(ProbeResult {
        glyph_pixels: glyph_scale * 8,
        output_stride: CompactConvVisionStemConfig::OUTPUT_STRIDE,
        visual_tokens: 9,
        parameters,
        steps,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        initial_loss: initial_loss.unwrap_or(final_loss),
        final_loss,
        training_position_accuracy,
        held_out_position_accuracy,
        gate_passed: held_out_position_accuracy >= 0.95,
    })
}

fn images(
    characters: &[char],
    glyph_scale: usize,
    offset_x: usize,
    offset_y: usize,
) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    characters
        .iter()
        .map(|&character| {
            Ok(render_glyph_probe(
                character,
                glyph_scale,
                offset_x,
                offset_y,
                CANVAS,
                CANVAS,
            )?)
        })
        .collect()
}

fn training_offsets(scale: usize) -> Vec<(usize, usize)> {
    let centre = (CANVAS - 8 * scale) / 2;
    let held_out = held_out_offsets(scale);
    (-6isize..=6)
        .flat_map(|dy| (-6isize..=6).map(move |dx| (dx, dy)))
        .map(|(dx, dy)| {
            (
                centre.checked_add_signed(dx).expect("x offset in canvas"),
                centre.checked_add_signed(dy).expect("y offset in canvas"),
            )
        })
        .filter(|offset| !held_out.contains(offset))
        .collect()
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
