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

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::schema::Example;

use crate::tokenizer::{ByteTokenizer, EOS, PAD, SEP};
use crate::transformer::{TinyTransformer, TinyTransformerConfig};

/// Render the tools offered in `example` plus the user request as the
/// prompt half of a training sequence.
///
/// Format is deliberately simple (not a "real" chat template): a JSON array
/// of tool schemas, a newline, then the request text. The model only needs
/// this to be *consistent* between training and generation, which
/// [`format_prompt`] and [`crate::generate::generate`] both call.
pub fn format_prompt(example: &Example) -> String {
    let tools_json = serde_json::to_string(&example.tools).unwrap_or_default();
    format!("{tools_json}\n{}", example.user_request)
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
pub fn encode_example(tokenizer: &ByteTokenizer, example: &Example) -> Encoded {
    let prompt = format_prompt(example);
    let target = example.expected_call_as_json_string();

    let mut ids = Vec::with_capacity(prompt.len() + target.len() + 3);
    ids.push(crate::tokenizer::BOS);
    ids.extend(tokenizer.encode(&prompt));
    ids.push(SEP);
    let target_start = ids.len();
    ids.extend(tokenizer.encode(&target));
    ids.push(EOS);

    Encoded { ids, target_start }
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
pub struct Batch<B: AutodiffBackend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub loss_mask: Vec<bool>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Build a training batch from a slice of examples, truncating/padding every
/// sequence to the batch's own max length (capped at `max_seq_len`).
pub fn build_batch<B: AutodiffBackend>(
    tokenizer: &ByteTokenizer,
    examples: &[Example],
    max_seq_len: usize,
    device: &B::Device,
) -> Batch<B> {
    let encoded: Vec<Encoded> = examples.iter().map(|ex| encode_example(tokenizer, ex)).collect();

    let longest = encoded.iter().map(|e| e.ids.len()).max().unwrap_or(1).min(max_seq_len);
    let batch_size = encoded.len();
    // Full sequences are `longest`; input/target pairs are `longest - 1`.
    let seq_len = longest.saturating_sub(1).max(1);

    let mut input_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    let mut target_flat: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
    let mut loss_mask: Vec<bool> = Vec::with_capacity(batch_size * seq_len);

    for enc in &encoded {
        let mut ids = enc.ids.clone();
        ids.truncate(longest);
        while ids.len() < longest {
            ids.push(PAD);
        }

        for i in 0..seq_len {
            input_flat.push(ids[i] as i64);
            target_flat.push(ids[i + 1] as i64);
            // Target position i+1 counts toward the loss iff it's at or
            // past target_start (i.e. it's part of the expected-call JSON
            // or the trailing EOS) and isn't PAD.
            let is_target_token = i + 1 >= enc.target_start && ids[i + 1] != PAD;
            loss_mask.push(is_target_token);
        }
    }

    let input_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_flat, [batch_size, seq_len]),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(target_flat, [batch_size, seq_len]),
        device,
    );

    Batch {
        input_ids,
        targets,
        loss_mask,
        batch_size,
        seq_len,
    }
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
    let device = batch.input_ids.device();
    let mask_flags: Vec<i64> = batch.loss_mask.iter().map(|&m| if m { 1 } else { 0 }).collect();
    let mask_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(mask_flags, [batch.batch_size * batch.seq_len]), &device);

    let logits = model.forward(batch.input_ids.clone()); // [batch, seq, vocab]
    let masked_loss = masked_cross_entropy(logits, batch.targets.clone(), &mask_tensor);

    let grads = masked_loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optim.step(lr, model, grads);

    let loss_value: f32 = masked_loss.into_scalar().elem();
    (model, loss_value)
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
fn masked_cross_entropy<B: AutodiffBackend>(
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
) -> (TinyTransformer<B>, OptimizerAdaptor<Adam, TinyTransformer<B>, B>) {
    let model = model_config.init::<B>(device);
    let optim = AdamConfig::new().init::<B, TinyTransformer<B>>();
    (model, optim)
}

/// Project a training-mode model to its inference-mode (`valid()`)
/// projection, for generation/eval. Thin wrapper so callers don't need to
/// import `AutodiffModule` themselves.
pub fn to_eval_model<B: AutodiffBackend>(model: &TinyTransformer<B>) -> TinyTransformer<B::InnerBackend> {
    model.valid()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use grug_data::Dataset;

    fn tiny_model_config() -> TinyTransformerConfig {
        TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64)
    }

    #[test]
    fn encode_example_places_target_start_after_sep() {
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(1).unwrap();
        let examples = dataset.train_examples(1).unwrap();
        let enc = encode_example(&tok, &examples[0]);

        assert_eq!(enc.ids[enc.target_start - 1], SEP);
        assert_eq!(*enc.ids.last().unwrap(), EOS);
    }

    #[test]
    fn build_batch_produces_expected_shapes() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = Dataset::new(2).unwrap();
        let examples = dataset.train_examples(4).unwrap();

        let batch = build_batch::<CpuAutodiffBackend>(&tok, &examples, 64, &device);
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
        let batch = build_batch::<CpuAutodiffBackend>(&tok, &examples, 64, &device);

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
}
