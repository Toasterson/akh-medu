//! Training/eval plumbing for the Phase 51 **P4** deductive corpus arms
//! (`grug_data::RuleExample` — the propositional + rule-based High-Grug-v2
//! arms).
//!
//! Identical in spirit to [`crate::reason_train`]: `crate::train::*`,
//! `crate::checkpoint::*`, and `crate::generate::generate` are reused
//! unchanged (they're generic over `Batch<B>` + a tokenizer + a prompt
//! string). A [`RuleExample`] already carries its own rendered
//! `prompt`/`target`, so — as with the P3-R corpus — the only corpus-specific
//! piece is turning those two strings into an [`Encoded`]/[`Batch`] with the
//! shared `BOS <prompt> SEP <target> EOS` framing and target-only loss mask.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};

use grug_data::RuleExample;

use crate::error::GrugResult;
use crate::generate::{GenerateConfig, generate};
use crate::tokenizer::Vocab;
use crate::train::{Batch, Encoded};
use crate::transformer::{AttentionBackend, TinyTransformer};

/// The prompt half of the training sequence for a [`RuleExample`] — exactly
/// `example.prompt` (fact/rule lines + a `goal` line). A plain pass-through,
/// mirroring [`crate::reason_train::format_reason_prompt`].
pub fn format_rule_prompt(example: &RuleExample) -> &str {
    &example.prompt
}

/// Tokenize one [`RuleExample`] into the shared `BOS <prompt> SEP <target>
/// EOS` framing, reading straight from `example.prompt`/`example.target`.
pub fn encode_rule_example<V: Vocab>(tokenizer: &V, example: &RuleExample) -> GrugResult<Encoded> {
    let mut ids = Vec::with_capacity(example.prompt.len() + example.target.len() + 3);
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(&example.prompt)?);
    ids.push(tokenizer.sep_id());
    let target_start = ids.len();
    ids.extend(tokenizer.encode(&example.target)?);
    ids.push(tokenizer.eos_id());
    Ok(Encoded { ids, target_start })
}

/// [`crate::reason_train::build_reason_batch`]'s counterpart for
/// [`RuleExample`]s: identical truncate/pad/target-only-loss-mask discipline,
/// no copy supervision.
pub fn build_rule_batch<B: AutodiffBackend, V: Vocab>(
    tokenizer: &V,
    examples: &[RuleExample],
    max_seq_len: usize,
    device: &B::Device,
) -> GrugResult<Batch<B>> {
    let pad_id = tokenizer.pad_id();
    let encoded: Vec<Encoded> = examples
        .iter()
        .map(|ex| encode_rule_example(tokenizer, ex))
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

/// Greedy-generate on every [`RuleExample`] in `examples`, returning the raw
/// output strings in order — feed straight to
/// `grug_data::score_rule_outputs(examples, &outputs)`.
pub fn generate_rule_outputs<B: AttentionBackend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    examples: &[RuleExample],
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
    use grug_data::{RuleCorpusConfig, RuleDataset, RuleWorldConfig, TraceArm};

    fn tiny_dataset() -> RuleDataset {
        RuleDataset::new(RuleCorpusConfig {
            world: RuleWorldConfig {
                seed: 9,
                preds_per_cell: 6,
                max_level: 4,
                guard_prob: 0.5,
            },
            distractor_facts: 2,
            distractor_rules: 3,
            yes_frac: 0.5,
            prop_literals: (2, 3),
        })
        .unwrap()
    }

    #[test]
    fn encode_rule_example_places_target_start_after_sep() {
        let tok = ByteTokenizer::new();
        let ds = tiny_dataset();
        let examples = ds.rule_examples(1, &[3], TraceArm::Trace, true);
        let enc = encode_rule_example(&tok, &examples[0]).unwrap();
        assert_eq!(enc.ids[enc.target_start - 1], crate::tokenizer::SEP);
        assert_eq!(*enc.ids.last().unwrap(), crate::tokenizer::EOS);
    }

    #[test]
    fn training_loop_decreases_loss_on_rule_batch() {
        let device = Default::default();
        let tok = ByteTokenizer::new();
        let ds = tiny_dataset();
        let examples = ds.rule_examples(4, &[2, 3], TraceArm::Trace, true);

        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 512);
        let (mut model, mut optim) = init_training::<CpuAutodiffBackend>(&model_config, &device);
        let batch =
            build_rule_batch::<CpuAutodiffBackend, _>(&tok, &examples, 512, &device).unwrap();

        let mut losses = Vec::new();
        for _ in 0..30 {
            let (next_model, loss) = train_step(model, &mut optim, &batch, 1e-2);
            model = next_model;
            losses.push(loss);
        }
        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "expected loss to decrease: {losses:?}"
        );

        let eval_model = to_eval_model(&model);
        let gen_config = GenerateConfig {
            max_new_tokens: 16,
            max_seq_len: 512,
        };
        let outputs = generate_rule_outputs(&eval_model, &tok, &examples, gen_config, &device);
        assert_eq!(outputs.len(), examples.len());
    }
}
