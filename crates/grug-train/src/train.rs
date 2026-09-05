//! Teacher-forced next-token training loop for [`TinyTransformer`].
//!
//! Each [`grug_data::schema::Example`] is formatted into one flat byte
//! sequence:
//!
//! ```text
//! BOS <tools+request bytes> SEP <expected-call-JSON bytes> EOS
//! ```
//!
//! Cross-entropy is computed over every position, but the loss is only
//! *counted* for positions predicting a target token (i.e. everything from
//! the first target byte through the trailing `EOS`) -- prompt-token
//! predictions are masked out. This is standard instruction-tuning framing:
//! the model is trained to *generate* the call, not to reproduce the prompt
//! it was given.

use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::ElementConversion;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::TargetFormat;
use grug_data::schema::Example;

use crate::pointer::{PointerTransformer, PointerTransformerConfig};
use crate::tokenizer::Vocab;
use crate::transformer::{TinyTransformer, TinyTransformerConfig};

/// Render the tools offered in `example` plus the user request as the
/// prompt half of a training sequence.
///
/// Format is deliberately simple (not a "real" chat template): a JSON array
/// of tool schemas, a newline, then the request text. The model only needs
/// this to be *consistent* between training and generation, which
/// [`format_prompt`] and [`crate::generate::generate`] both call.
pub fn format_prompt(example: &Example) -> String {
    format_prompt_fmt(example, TargetFormat::Json)
}

/// [`format_prompt`] under an explicit [`TargetFormat`] (Phase 51 P2): renders
/// the offered tools via `grug_data::render_tools` so the *prompt-side* schema
/// switches format alongside the target, not just the call.
pub fn format_prompt_fmt(example: &Example, fmt: TargetFormat) -> String {
    let tools = grug_data::render_tools(&example.tools, fmt);
    format!("{tools}\n{}", example.user_request)
}

/// One tokenized training example: the full `BOS..EOS` id sequence, plus the
/// index of the first token whose *prediction* should be counted in the
/// loss (i.e. the position right after `SEP`).
pub struct Encoded {
    pub ids: Vec<u32>,
    /// Index into `ids` of the first target token. Predicting `ids[i]` uses
    /// `ids[..i]` as context, so the loss-mask start for the *targets*
    /// tensor (which is `ids[1..]`, predicted from `ids[..len-1]`) is
    /// `target_start - 1`.
    pub target_start: usize,
}

/// Tokenize `example` into the `BOS <prompt> SEP <target> EOS` framing.
pub fn encode_example<V: Vocab>(
    tokenizer: &V,
    example: &Example,
) -> crate::error::GrugResult<Encoded> {
    encode_example_fmt(tokenizer, example, TargetFormat::Json)
}

/// [`encode_example`] under an explicit [`TargetFormat`] (Phase 51 P2): both
/// the prompt-side schema and the target call are rendered in `fmt`.
pub fn encode_example_fmt<V: Vocab>(
    tokenizer: &V,
    example: &Example,
    fmt: TargetFormat,
) -> crate::error::GrugResult<Encoded> {
    let prompt = format_prompt_fmt(example, fmt);
    let target = grug_data::render_call(&example.expected_call, fmt);

    let mut ids = Vec::with_capacity(prompt.len() + target.len() + 3);
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(&prompt)?);
    ids.push(tokenizer.sep_id());
    let target_start = ids.len();
    ids.extend(tokenizer.encode(&target)?);
    ids.push(tokenizer.eos_id());

    Ok(Encoded { ids, target_start })
}

/// Hyperparameters for [`train_step`] / the training loop in `examples/spike0.rs`.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub learning_rate: f64,
    pub max_seq_len: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-3,
            max_seq_len: 256,
        }
    }
}

/// A batch of tokenized examples, padded to a common length and ready for
/// [`TinyTransformer::forward`].
///
/// `input_ids` and `targets` are both `[batch_size, seq_len - 1]`: standard
/// next-token framing where `input_ids[:, i]` predicts `targets[:, i] ==
/// input_ids[:, i+1]`. `loss_mask` is `true` where that position's target is
/// a real target-half token (i.e. should count toward the loss) and `false`
/// for prompt tokens and padding.
pub struct Batch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub loss_mask: Vec<bool>,
    /// Phase 51 P1a-4 Step B2: `true` where the target token at this position
    /// also appears in that example's **prompt** region (i.e. it is copyable
    /// from the offered schema / request). Used by the copy-supervision loss
    /// (`train_step_ptr_supervised`) to push `p_gen` toward *copy* at copyable
    /// positions, which is what forces the pointer path to actually train (the
    /// gate otherwise collapses to "always generate" — see Step B findings).
    pub copy_mask: Vec<bool>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Build a training batch from a slice of examples, truncating/padding every
/// sequence to the batch's own max length (capped at `max_seq_len`).
pub fn build_batch<B: Backend, V: Vocab>(
    tokenizer: &V,
    examples: &[Example],
    max_seq_len: usize,
    device: &B::Device,
) -> crate::error::GrugResult<Batch<B>> {
    build_batch_fmt(tokenizer, examples, max_seq_len, device, TargetFormat::Json)
}

/// [`build_batch`] under an explicit [`TargetFormat`] (Phase 51 P2).
pub fn build_batch_fmt<B: Backend, V: Vocab>(
    tokenizer: &V,
    examples: &[Example],
    max_seq_len: usize,
    device: &B::Device,
    fmt: TargetFormat,
) -> crate::error::GrugResult<Batch<B>> {
    let pad_id = tokenizer.pad_id();
    let encoded: Vec<Encoded> = examples
        .iter()
        .map(|ex| encode_example_fmt(tokenizer, ex, fmt))
        .collect::<crate::error::GrugResult<Vec<_>>>()?;

    let longest = encoded
        .iter()
        .map(|e| e.ids.len())
        .max()
        .unwrap_or(1)
        .min(max_seq_len);
    let batch_size = encoded.len();
    // Full sequences are `longest`; input/target pairs are `longest - 1`.
    let seq_len = longest.saturating_sub(1).max(1);

    let mut input_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    let mut target_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    let mut loss_mask: Vec<bool> = Vec::with_capacity(batch_size * seq_len);
    let mut copy_mask: Vec<bool> = Vec::with_capacity(batch_size * seq_len);

    for enc in &encoded {
        let mut ids = enc.ids.clone();
        ids.truncate(longest);
        while ids.len() < longest {
            ids.push(pad_id);
        }

        // The prompt region (everything strictly before the first target
        // token) is the copy source: a target token is "copyable" iff it also
        // appears here. Built once per example.
        let prompt_end = enc.target_start.min(longest);
        let prompt_set: std::collections::HashSet<u32> =
            ids[0..prompt_end].iter().copied().collect();

        for i in 0..seq_len {
            input_flat.push(ids[i] as i64);
            target_flat.push(ids[i + 1] as i64);
            // Target position i+1 counts toward the loss iff it's at or
            // past target_start (i.e. it's part of the expected-call JSON
            // or the trailing EOS) and isn't PAD.
            let is_target_token = i + 1 >= enc.target_start && ids[i + 1] != pad_id;
            loss_mask.push(is_target_token);
            // Copyable iff the target token appears in this example's prompt.
            copy_mask.push(prompt_set.contains(&ids[i + 1]));
        }
    }

    let input_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [batch_size, seq_len]), device);
    let targets =
        Tensor::<B, 2, Int>::from_data(TensorData::new(target_flat, [batch_size, seq_len]), device);

    Ok(Batch {
        input_ids,
        targets,
        loss_mask,
        copy_mask,
        batch_size,
        seq_len,
    })
}

/// Run one forward + backward + optimizer-step training iteration.
///
/// Returns the scalar loss (as `f32`, already detached) for logging.
pub fn train_step<B: AutodiffBackend>(
    model: TinyTransformer<B>,
    optim: &mut OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
    batch: &Batch<B>,
    lr: f64,
) -> (TinyTransformer<B>, f32) {
    let (model, loss, _grad_norm) =
        train_step_guarded(model, optim, batch, lr, f32::INFINITY, f32::INFINITY);
    (model, loss)
}

/// Forward-only masked loss through the ordinary text path. Keeping this
/// generic over [`Backend`] lets expensive evaluations run without retaining
/// an autodiff graph.
pub fn text_batch_loss<B: Backend>(model: &TinyTransformer<B>, batch: &Batch<B>) -> f32 {
    let device = batch.input_ids.device();
    let mask_flags: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|&enabled| i64::from(enabled))
        .collect();
    let mask = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask_flags, [batch.batch_size * batch.seq_len]),
        &device,
    );
    let loss = masked_cross_entropy(
        model.forward(batch.input_ids.clone()),
        batch.targets.clone(),
        &mask,
    );
    loss.into_scalar().elem()
}

/// [`train_step`] with a loss cap and a grad-norm cap: when the batch loss is
/// non-finite OR above `loss_cap`, OR the pre-clip global grad norm is
/// non-finite OR above `grad_norm_cap`, the optimizer step is skipped and the
/// model returned untouched. Returns `(model, loss, grad_norm)`; `grad_norm`
/// is `f32::NAN` when the loss gate skipped before backward() ran (grad norm
/// was never computed).
///
/// Why the loss cap exists (p350m runs 1+2, 2026-09-01): on a task the model
/// has already floored (~0.5 loss by step 600), warmup keeps raising the LR
/// while Adam's second moment decays toward zero, so the effective step size
/// grows until one harder batch spikes the loss (0.5 -> 7.4), a gradient
/// overflows, and Norm-clipping turns the inf norm into NaN weights in a
/// single update. Skipping spike batches breaks that chain before weight
/// damage, which clipping alone demonstrably did not.
///
/// Why the grad-norm cap also exists: `init_training`'s `Norm(1.0)` clipping
/// operates per parameter tensor, not globally, so it can still divide an
/// individual tensor's inf norm into NaN weights even when the *loss* looks
/// fine that step. The literature on loss spikes (and the run-1/run-2
/// blowups) shows the pre-clip grad-norm spike foreshocks the loss spike by a
/// few steps -- gating on the norm directly, before it ever reaches
/// `optim.step`, is a strictly earlier and cheaper signal than waiting for
/// loss to move.
pub fn train_step_guarded<B: AutodiffBackend>(
    model: TinyTransformer<B>,
    optim: &mut OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
    batch: &Batch<B>,
    lr: f64,
    loss_cap: f32,
    grad_norm_cap: f32,
) -> (TinyTransformer<B>, f32, f32) {
    let device = batch.input_ids.device();
    let mask_flags: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|&m| if m { 1 } else { 0 })
        .collect();
    let mask_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask_flags, [batch.batch_size * batch.seq_len]),
        &device,
    );

    let logits = model.forward(batch.input_ids.clone()); // [batch, seq, vocab]
    let masked_loss = masked_cross_entropy(logits, batch.targets.clone(), &mask_tensor);

    // Read the loss BEFORE stepping: a single non-finite batch must not reach
    // the optimizer, or the Adam moments are poisoned and every later step is
    // NaN (observed: p350m step 880, healthy ~0.5 loss -> sudden NaN). The
    // caller decides whether to skip or abort.
    let loss_value: f32 = masked_loss.clone().into_scalar().elem();
    if !loss_value.is_finite() || loss_value > loss_cap {
        return (model, loss_value, f32::NAN);
    }

    let grads = masked_loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);

    // Pre-clip global L2 grad norm -- see the doc comment above for why this
    // is gated separately from the loss.
    let grad_norm = global_grad_norm(&grads, &model);
    if !grad_norm.is_finite() || grad_norm > grad_norm_cap {
        return (model, loss_value, grad_norm);
    }

    let model = optim.step(lr, model, grads);

    (model, loss_value, grad_norm)
}

/// Global (whole-model) pre-clip L2 grad norm: `sqrt(sum of squares)` over
/// every registered gradient tensor in `grads`, restricted to `model`'s
/// parameters. Walks the model via [`ModuleVisitor`] (the same mechanism
/// [`GradientsParams::from_grads`] uses internally) so it works for any
/// module shape without hardcoding layer names.
pub(crate) fn global_grad_norm<B: AutodiffBackend, M: Module<B>>(
    grads: &GradientsParams,
    model: &M,
) -> f32 {
    struct NormVisitor<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f32,
        _backend: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> ModuleVisitor<B> for NormVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) else {
                return;
            };
            let sq_sum: f32 = grad.powf_scalar(2.0).sum().into_scalar().elem();
            self.sum_sq += sq_sum;
        }
    }

    let mut visitor = NormVisitor::<B> {
        grads,
        sum_sq: 0.0,
        _backend: std::marker::PhantomData,
    };
    model.visit(&mut visitor);
    visitor.sum_sq.sqrt()
}

/// Cross-entropy over `logits` ([batch, seq, vocab]) vs `targets` ([batch,
/// seq]), averaged only over positions where `mask` ([batch*seq], 1/0) is
/// nonzero.
///
/// Implemented via `log_softmax` + `gather` rather than
/// `burn::nn::loss::CrossEntropyLoss` because that type's `pad_tokens`
/// option masks by *target token value*, not by *sequence position* -- we
/// need to mask out prompt positions regardless of what byte value happens
/// to sit there.
pub(crate) fn masked_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    mask: &Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    use burn::tensor::activation::log_softmax;

    let [b, s, v] = logits.dims();
    let logits_flat = logits.reshape([b * s, v]);
    let targets_flat = targets.reshape([b * s, 1]);

    let log_probs = log_softmax(logits_flat, 1); // [b*s, v]
    let picked = log_probs.gather(1, targets_flat).reshape([b * s]); // [b*s]

    let mask_f = mask.clone().float();
    let denom = mask_f.clone().sum();
    let masked_sum = (picked * mask_f).sum();

    // Guard against an all-masked-out batch (denom == 0) producing NaN;
    // this should not happen in practice (every example has a non-empty
    // target), but keep the loop from silently going NaN if it ever does.
    let denom_scalar: f32 = denom.clone().into_scalar().elem();
    if denom_scalar == 0.0 {
        return masked_sum.neg();
    }

    (masked_sum.neg()) / denom
}

/// Build a fresh model + Adam optimizer for training, per [`TrainConfig`].
pub fn init_training<B: AutodiffBackend>(
    model_config: &TinyTransformerConfig,
    device: &B::Device,
) -> (
    TinyTransformer<B>,
    OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
) {
    let model = model_config.init::<B>(device);
    // Gradient-norm clipping: without it the harder/longer hard-distractor task
    // hits a large-gradient batch and diverges to NaN around step ~100 (the
    // easier 2-distractor data happened not to). Norm-clip at 1.0 is the standard
    // stabilizer and is cheap. (Phase 51 P1a-3 NaN fix.)
    let optim = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init::<B, TinyTransformer<B>>();
    (model, optim)
}

/// Project a training-mode model to its inference-mode (`valid()`)
/// projection, for generation/eval. Thin wrapper so callers don't need to
/// import `AutodiffModule` themselves.
pub fn to_eval_model<B: AutodiffBackend>(
    model: &TinyTransformer<B>,
) -> TinyTransformer<B::InnerBackend> {
    model.valid()
}

// ----------------------------------------------------------------------------
// Phase 51 P1a-4 Step B: pointer-generator training (see `crate::pointer`).
// ----------------------------------------------------------------------------

/// Build a fresh [`PointerTransformer`] + Adam optimizer (same grad-norm
/// clipping as [`init_training`]).
pub fn init_training_ptr<B: AutodiffBackend>(
    config: &PointerTransformerConfig,
    device: &B::Device,
) -> (
    PointerTransformer<B>,
    OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
) {
    let model = config.init::<B>(device);
    let optim = AdamConfig::new()
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(1.0)))
        .init::<B, PointerTransformer<B>>();
    (model, optim)
}

/// One forward + backward + optimizer step for the pointer-generator model.
/// Identical framing to [`train_step`], but the model emits a *mixed*
/// distribution (`PointerTransformer::log_probs`) so the loss is an NLL over
/// those log-probs ([`masked_nll`]) rather than a cross-entropy over logits.
pub fn train_step_ptr<B: AutodiffBackend>(
    model: PointerTransformer<B>,
    optim: &mut OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
    batch: &Batch<B>,
    lr: f64,
) -> (PointerTransformer<B>, f32) {
    let device = batch.input_ids.device();
    let mask_flags: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|&m| if m { 1 } else { 0 })
        .collect();
    let mask_tensor = Tensor::<B, 1, Int>::from_data(
        TensorData::new(mask_flags, [batch.batch_size * batch.seq_len]),
        &device,
    );

    let log_probs = model.log_probs(batch.input_ids.clone()); // [batch, seq, vocab]
    let masked_loss = masked_nll(log_probs, batch.targets.clone(), &mask_tensor);

    let grads = masked_loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(lr, model, grads);

    let loss_value: f32 = masked_loss.into_scalar().elem();
    (model, loss_value)
}

/// Negative log-likelihood over a model that already emits **log-probabilities**
/// `[batch, seq, vocab]` (the pointer-generator mixed distribution), averaged
/// only over positions where `mask` is nonzero. The logits counterpart is
/// [`masked_cross_entropy`]; this one skips the `log_softmax` because the input
/// is already normalised log-probs.
fn masked_nll<B: AutodiffBackend>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    mask: &Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let [b, s, v] = log_probs.dims();
    let log_probs_flat = log_probs.reshape([b * s, v]);
    let targets_flat = targets.reshape([b * s, 1]);

    let picked = log_probs_flat.gather(1, targets_flat).reshape([b * s]); // [b*s]

    let mask_f = mask.clone().float();
    let denom = mask_f.clone().sum();
    let masked_sum = (picked * mask_f).sum();

    let denom_scalar: f32 = denom.clone().into_scalar().elem();
    if denom_scalar == 0.0 {
        return masked_sum.neg();
    }
    (masked_sum.neg()) / denom
}

/// Project a training-mode [`PointerTransformer`] to its inference-mode
/// projection, for generation/eval.
pub fn to_eval_model_ptr<B: AutodiffBackend>(
    model: &PointerTransformer<B>,
) -> PointerTransformer<B::InnerBackend> {
    model.valid()
}

/// Copy-supervised pointer training step (Phase 51 P1a-4 Step B2).
///
/// Step B's pointer-generator failed because marginal-likelihood training gave
/// `p_gen` no reason to copy — it collapsed to "always generate" (measured mean
/// `p_gen` ≈ 0.96), leaving the copy pathway untrained. This step adds an
/// auxiliary **BCE** that supervises `p_gen` directly: at positions whose
/// target token is *copyable* (present in the prompt — `batch.copy_mask`) it
/// pushes `p_gen` toward 0 (copy); elsewhere toward 1 (generate). With `p_gen`
/// forced low at copyable positions, the NLL term then trains the copy
/// distribution to actually place mass on the right source token. Total loss is
/// `NLL + copy_lambda · BCE`.
///
/// `copy_lambda = 0.0` reproduces [`train_step_ptr`] exactly (BCE contributes
/// nothing), so this is a strict generalisation.
pub fn train_step_ptr_supervised<B: AutodiffBackend>(
    model: PointerTransformer<B>,
    optim: &mut OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
    batch: &Batch<B>,
    lr: f64,
    copy_lambda: f64,
) -> (PointerTransformer<B>, f32) {
    let device = batch.input_ids.device();
    let n = batch.batch_size * batch.seq_len;

    let mask_int: Vec<i64> = batch
        .loss_mask
        .iter()
        .map(|&m| if m { 1 } else { 0 })
        .collect();
    let mask_int_t = Tensor::<B, 1, Int>::from_data(TensorData::new(mask_int, [n]), &device);

    let parts = model.forward_parts(batch.input_ids.clone());
    let nll = masked_nll(parts.log_probs, batch.targets.clone(), &mask_int_t);

    // Auxiliary BCE on p_gen. Target = P(generate): 1 where the token is NOT
    // copyable, 0 (copy) where it is. Averaged over loss-mask positions only.
    let loss_flags: Vec<f32> = batch
        .loss_mask
        .iter()
        .map(|&m| if m { 1.0 } else { 0.0 })
        .collect();
    let loss_mask_f = Tensor::<B, 1>::from_data(TensorData::new(loss_flags, [n]), &device);
    let gen_target: Vec<f32> = batch
        .copy_mask
        .iter()
        .map(|&c| if c { 0.0 } else { 1.0 })
        .collect();
    let gen_target_t = Tensor::<B, 1>::from_data(TensorData::new(gen_target, [n]), &device);

    let eps = 1e-6;
    let p = parts.p_gen.reshape([n]).clamp(eps, 1.0 - eps); // [n]
    let one_minus_t = gen_target_t.clone().neg().add_scalar(1.0);
    let one_minus_p = p.clone().neg().add_scalar(1.0);
    let bce = (gen_target_t * p.log() + one_minus_t * one_minus_p.log()).neg(); // [n]
    let denom = loss_mask_f.clone().sum();
    let bce_loss = (bce * loss_mask_f).sum() / denom;

    let total = nll + bce_loss.mul_scalar(copy_lambda);

    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(lr, model, grads);

    let loss_value: f32 = total.into_scalar().elem();
    (model, loss_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::tokenizer::ByteTokenizer;
    use grug_data::Dataset;

    fn tiny_model_config() -> TinyTransformerConfig {
        TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64)
    }

    /// Phase 51 P1a-4 Step B: a pointer-generator training step must run
    /// forward+backward (through the scatter-based copy distribution) and
    /// actually *learn* — several teacher-forced steps on a fixed tiny batch
    /// should drive the loss down, which is only possible if gradients flow
    /// through the copy path and optimizer update. This is the end-to-end
    /// autodiff check the forward-only `pointer::tests` cannot make.
    #[test]
    fn ptr_train_step_learns_a_fixed_batch() {
        use crate::pointer::PointerTransformerConfig;
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(3).unwrap();
        let examples = dataset.train_examples(4).unwrap();

        // Byte tokenizer -> vocab 260; generous ctx so targets aren't truncated.
        let cfg = PointerTransformerConfig::new(TinyTransformerConfig::new(
            crate::tokenizer::VOCAB_SIZE,
            32,
            2,
            2,
            64,
            512,
        ));
        let (mut model, mut optim) = init_training_ptr::<CpuAutodiffBackend>(&cfg, &device);
        let batch = build_batch::<CpuAutodiffBackend, _>(&tok, &examples, 512, &device).unwrap();

        let mut first = f32::NAN;
        let mut last = f32::NAN;
        for step in 0..8 {
            let (next, loss) = train_step_ptr(model, &mut optim, &batch, 3e-3);
            model = next;
            assert!(
                loss.is_finite(),
                "loss went non-finite at step {step}: {loss}"
            );
            if step == 0 {
                first = loss;
            }
            last = loss;
        }
        assert!(
            last < first,
            "pointer training did not reduce loss on a fixed batch: first={first}, last={last}"
        );
    }

    #[test]
    fn encode_example_places_target_start_after_sep() {
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(1).unwrap();
        let examples = dataset.train_examples(1).unwrap();
        let enc = encode_example(&tok, &examples[0]).unwrap();

        assert_eq!(enc.ids[enc.target_start - 1], crate::tokenizer::SEP);
        assert_eq!(*enc.ids.last().unwrap(), crate::tokenizer::EOS);
    }

    #[test]
    fn build_batch_produces_expected_shapes() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(2).unwrap();
        let examples = dataset.train_examples(4).unwrap();

        let batch = build_batch::<CpuAutodiffBackend, _>(&tok, &examples, 64, &device).unwrap();
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.input_ids.dims(), [4, batch.seq_len]);
        assert_eq!(batch.targets.dims(), [4, batch.seq_len]);
        assert_eq!(batch.loss_mask.len(), 4 * batch.seq_len);
        // At least one position per example should count toward the loss.
        assert!(batch.loss_mask.iter().any(|&m| m));
    }

    /// The core Spike-0 gate: a short training run (few dozen steps, tiny
    /// model + tiny batch) measurably reduces the masked cross-entropy loss.
    /// Kept fast (<< 30s) by using a small model and few examples.
    #[test]
    fn training_loop_decreases_loss() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(3).unwrap();
        let examples = dataset.train_examples(4).unwrap();

        let (mut model, mut optim) =
            init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let batch = build_batch::<CpuAutodiffBackend, _>(&tok, &examples, 64, &device).unwrap();

        let lr = 1e-2;
        let mut losses = Vec::new();
        for _ in 0..30 {
            let (next_model, loss) = train_step(model, &mut optim, &batch, lr);
            model = next_model;
            losses.push(loss);
        }

        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        assert!(
            last < first,
            "expected loss to decrease over {} steps: first={first}, last={last}, all={losses:?}",
            losses.len()
        );
    }

    /// P1a: the same training loop, batching, and masked-loss machinery
    /// must work unchanged with [`crate::bpe::BpeTokenizer`] substituted for
    /// [`ByteTokenizer`] -- this is the entire point of the `Vocab` seam.
    #[test]
    fn training_loop_decreases_loss_with_bpe_tokenizer() {
        let device = Default::default();
        let dataset = Dataset::new(5).unwrap();
        let train_examples = dataset.train_examples(16).unwrap();

        let corpus: Vec<String> = train_examples
            .iter()
            .flat_map(|ex| vec![format_prompt(ex), ex.expected_call_as_json_string()])
            .collect();
        let tok = crate::bpe::BpeTokenizer::train(corpus, 512).unwrap();

        let model_config = TinyTransformerConfig::new(tok.vocab_size(), 16, 2, 1, 32, 128);
        let (mut model, mut optim) = init_training::<CpuAutodiffBackend>(&model_config, &device);
        let batch =
            build_batch::<CpuAutodiffBackend, _>(&tok, &train_examples[..4], 128, &device).unwrap();

        let lr = 1e-2;
        let mut losses = Vec::new();
        for _ in 0..30 {
            let (next_model, loss) = train_step(model, &mut optim, &batch, lr);
            model = next_model;
            losses.push(loss);
        }

        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        assert!(
            last < first,
            "expected loss to decrease over {} steps with BPE tokenizer: first={first}, last={last}, all={losses:?}",
            losses.len()
        );
    }
}
