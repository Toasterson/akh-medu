//! Training/eval plumbing for the Phase 51 P3-R reasoning corpus
//! (`grug_data::reason::ReasonExample`).
//!
//! `crate::train::train_step` / `init_training` / `to_eval_model` /
//! `crate::checkpoint::*` / `crate::generate::generate` are all reused
//! *unchanged* here -- they're generic over `Batch<B>` and a tokenizer +
//! prompt string, not over any particular `Example` type. This module
//! supplies the one piece that genuinely differs from the tool-calling
//! corpus: turning a [`ReasonExample`]'s already-rendered `prompt`/`target`
//! strings into an [`Encoded`]/[`Batch`] -- there is no tool-schema
//! rendering step on this side, since a reasoning example carries its own
//! plain-text prompt (fact lines + a `goal` line) and target (a High-Grug v1
//! trace or an answer-only line) directly.

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::reason::ReasonExample;

use crate::error::GrugResult;
use crate::generate::{GenerateConfig, generate};
use crate::tokenizer::Vocab;
use crate::train::{Batch, Encoded};
use crate::transformer::TinyTransformer;

/// The prompt half of the training sequence for a [`ReasonExample`] --
/// exactly `example.prompt`. Named to mirror `crate::train::format_prompt`'s
/// role for the tool-calling corpus, even though here it's a plain
/// pass-through (no schema to render).
pub fn format_reason_prompt(example: &ReasonExample) -> &str {
    &example.prompt
}

/// Tokenize one [`ReasonExample`] into the same `BOS <prompt> SEP <target>
/// EOS` framing [`crate::train::encode_example_fmt`] uses for the
/// tool-calling corpus, reading straight from `example.prompt`/
/// `example.target` instead of rendering a tool schema + call.
pub fn encode_reason_example<V: Vocab>(
    tokenizer: &V,
    example: &ReasonExample,
) -> GrugResult<Encoded> {
    let mut ids = Vec::with_capacity(example.prompt.len() + example.target.len() + 3);
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(&example.prompt)?);
    ids.push(tokenizer.sep_id());
    let target_start = ids.len();
    ids.extend(tokenizer.encode(&example.target)?);
    ids.push(tokenizer.eos_id());
    Ok(Encoded { ids, target_start })
}

/// [`crate::train::build_batch_fmt`]'s counterpart for [`ReasonExample`]s:
/// identical truncate/pad/loss-mask discipline (see that function's docs),
/// built via [`encode_reason_example`]. `copy_mask` is always `false` here --
/// the Phase 51 P3-R reasoning run does not use copy-supervised training
/// (`crate::train::train_step_ptr_supervised`), only plain `train_step`.
pub fn build_reason_batch<B: AutodiffBackend, V: Vocab>(
    tokenizer: &V,
    examples: &[ReasonExample],
    max_seq_len: usize,
    device: &B::Device,
) -> GrugResult<Batch<B>> {
    let pad_id = tokenizer.pad_id();
    let encoded: Vec<Encoded> = examples
        .iter()
        .map(|ex| encode_reason_example(tokenizer, ex))
        .collect::<GrugResult<Vec<_>>>()?;

    let longest = encoded
        .iter()
        .map(|e| e.ids.len())
        .max()
        .unwrap_or(1)
        .min(max_seq_len);
    let batch_size = encoded.len();
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

        for i in 0..seq_len {
            input_flat.push(ids[i] as i64);
            target_flat.push(ids[i + 1] as i64);
            // Target position i+1 counts toward the loss iff it's at or past
            // target_start (i.e. part of the trace/answer text or the
            // trailing EOS) and isn't PAD -- same rule as the tool-calling
            // batch builder.
            let is_target_token = i + 1 >= enc.target_start && ids[i + 1] != pad_id;
            loss_mask.push(is_target_token);
            copy_mask.push(false);
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

/// Greedy-generate on every [`ReasonExample`] in `examples`, returning the raw
/// output strings in the same order -- feed straight to
/// `grug_data::reason::score_reason_outputs(examples, &outputs)`.
pub fn generate_reason_outputs<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    examples: &[ReasonExample],
    gen_config: GenerateConfig,
    device: &B::Device,
) -> Vec<String> {
    examples
        .iter()
        .map(|ex| generate(model, tokenizer, &ex.prompt, gen_config, device))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::tokenizer::ByteTokenizer;
    use crate::train::{init_training, to_eval_model, train_step};
    use crate::transformer::TinyTransformerConfig;
    use grug_data::reason::{ReasonConfig, ReasonDataset, TraceArm};

    fn tiny_dataset() -> ReasonDataset {
        ReasonDataset::new(ReasonConfig {
            seed: 9,
            entities_per_type: 20,
            distractor_factor: 1,
            train_depths: vec![2],
            holdout_subject_frac: 0.25,
        })
        .unwrap()
    }

    #[test]
    fn encode_reason_example_places_target_start_after_sep() {
        let tok = ByteTokenizer::new();
        let dataset = tiny_dataset();
        let examples = dataset.train_examples(1, TraceArm::Trace).unwrap();
        let enc = encode_reason_example(&tok, &examples[0]).unwrap();
        assert_eq!(enc.ids[enc.target_start - 1], crate::tokenizer::SEP);
        assert_eq!(*enc.ids.last().unwrap(), crate::tokenizer::EOS);
    }

    #[test]
    fn build_reason_batch_produces_expected_shapes() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = tiny_dataset();
        let examples = dataset.train_examples(4, TraceArm::Trace).unwrap();

        let batch =
            build_reason_batch::<CpuAutodiffBackend, _>(&tok, &examples, 512, &device).unwrap();
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.input_ids.dims(), [4, batch.seq_len]);
        assert!(
            batch.loss_mask.iter().any(|&m| m),
            "at least one position should count toward the loss"
        );
    }

    /// End-to-end proof this batch/training plumbing actually trains: a
    /// handful of `train_step`s on a fixed reasoning-corpus batch must
    /// measurably reduce the masked loss, exactly like
    /// `crate::train::tests::training_loop_decreases_loss`.
    #[test]
    fn training_loop_decreases_loss_on_reasoning_batch() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let dataset = tiny_dataset();
        let examples = dataset.train_examples(4, TraceArm::Trace).unwrap();

        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 256);
        let (mut model, mut optim) = init_training::<CpuAutodiffBackend>(&model_config, &device);
        let batch =
            build_reason_batch::<CpuAutodiffBackend, _>(&tok, &examples, 256, &device).unwrap();

        let mut losses = Vec::new();
        for _ in 0..30 {
            let (next_model, loss) = train_step(model, &mut optim, &batch, 1e-2);
            model = next_model;
            losses.push(loss);
        }
        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        assert!(
            last < first,
            "expected loss to decrease: first={first}, last={last}, all={losses:?}"
        );

        // The eval-mode projection + greedy generation path also runs
        // end-to-end without panicking (correctness of an undertrained tiny
        // model's output is not asserted here).
        let eval_model = to_eval_model(&model);
        let gen_config = GenerateConfig {
            max_new_tokens: 16,
            max_seq_len: 256,
        };
        let outputs = generate_reason_outputs(&eval_model, &tok, &examples, gen_config, &device);
        assert_eq!(outputs.len(), examples.len());
    }
}
