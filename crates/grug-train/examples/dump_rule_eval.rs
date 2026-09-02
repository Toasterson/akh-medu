//! Phase 51 P4 "reasoning exam" harness — step 1: dump the deterministic
//! rule-corpus holdout prompts so an out-of-process model (the MLX character
//! model / its Qwen3-0.6B base, via `training/character/reasoning_exam.py`)
//! can be tested against the exact same deductive-rule task the 350M Burn
//! model is scored on.
//!
//! Regenerates the two held-out streams `train_rule.rs` builds — in-dist
//! (`--train-depths`, default `2,3,4`) and length-generalization
//! (`--eval-depths`, default `5,6`) — with the same `RuleCorpusConfig` (seed
//! 42, preds_per_cell 6, max_level 6, guard_prob 0.5, distractor_facts 4,
//! distractor_rules 6, yes_frac 0.5, arm trace, typed_slots true) and writes
//! one JSON object per line:
//!
//! ```text
//! {"id": "in_dist_000", "axis": "in_dist", "prompt": "fact ... \ngoal ..."}
//! ```
//!
//! `prompt` is exactly `format_rule_prompt(example)` — the same text the
//! 350M model sees at eval time.
//!
//! NOTE on `--n`: `RuleDataset::rule_examples_eval` seeds its RNG once per
//! call and consumes it strictly sequentially (see `rule_stream` in
//! `crates/grug-data/src/reason/rule/corpus.rs`), so calling with `--n 64`
//! directly yields byte-identical examples to the first 64 of a
//! `train_rule.rs` run's `--holdout-examples 256` stream (or any N >= 64).
//! That's the "same first-64 of each holdout stream" this tool dumps.
//!
//! v2 additions (reasoning-exam v2 — separate grammar failure from reasoning
//! failure): every row now also carries `"expected"` (`example.gold_answer`'s
//! surface, i.e. `"yes"`/`"no"`/`"gap"` — mirrors what `score_rule_jsonl`
//! regenerates from the same example, here just for convenience/debugging)
//! and, when `--few-shot K` is `> 0`, an `"exemplars"` array of `K` complete
//! worked examples (`{"prompt", "target"}`, target including the final
//! `call answer | value ...` line) for a caller to prepend as in-context
//! demonstrations.
//!
//! Exemplars are drawn from the **training** stream
//! (`RuleDataset::rule_examples`, salt `RULE_SEED_SALT`) — never
//! `rule_examples_eval` (salt `RULE_EVAL_SEED_SALT`), so they cannot overlap
//! the holdout rows this tool dumps: the two streams are seeded from
//! different salts XORed into the same base seed and drawn independently, so
//! there is no shared RNG state to collide even before considering the
//! (astronomically unlikely) content collision case. One fixed exemplar set
//! is drawn per axis (over that axis's own depths — `--train-depths` for
//! `in_dist`, `--eval-depths` for `len_gen`) and reused verbatim across every
//! row of that axis, so a few-shot prompt for a given axis is stable and
//! reviewable.
//!
//! ```text
//! cargo run -p grug-train --example dump_rule_eval -- \
//!     --out /tmp/rule_eval.jsonl --n 64
//!
//! cargo run -p grug-train --example dump_rule_eval -- \
//!     --out /tmp/rule_eval_fewshot3.jsonl --n 64 --few-shot 3
//! ```

use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;
use grug_data::{RuleCorpusConfig, RuleDataset, RuleWorldConfig, TraceArm};
use grug_train::rule_train::format_rule_prompt;

#[derive(Parser, Debug)]
#[command(name = "dump_rule_eval")]
struct Args {
    /// Output JSONL path.
    #[arg(long)]
    out: PathBuf,

    /// Examples per axis (in_dist and len_gen each get this many).
    #[arg(long, default_value_t = 64)]
    n: usize,

    // --- world / corpus knobs (train_rule.rs defaults) ---
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 6)]
    preds_per_cell: u32,
    #[arg(long, default_value_t = 6)]
    max_level: u32,
    #[arg(long, default_value_t = 0.5)]
    guard_prob: f64,
    #[arg(long, default_value_t = 4)]
    distractor_facts: usize,
    #[arg(long, default_value_t = 6)]
    distractor_rules: usize,
    #[arg(long, default_value_t = 0.5)]
    yes_frac: f64,

    #[arg(long, default_value = "2,3,4")]
    train_depths: String,
    #[arg(long, default_value = "5,6")]
    eval_depths: String,

    /// Number of few-shot worked examples to emit per row (0 = none, v1
    /// shape). Drawn from the TRAINING stream, one fixed set per axis (see
    /// module docs).
    #[arg(long, default_value_t = 0)]
    few_shot: usize,
}

fn parse_depths(s: &str, flag: &str) -> Vec<usize> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag}: not an integer: {t:?}"))
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    let train_depths = parse_depths(&args.train_depths, "--train-depths");
    let eval_depths = parse_depths(&args.eval_depths, "--eval-depths");

    let dataset = RuleDataset::new(RuleCorpusConfig {
        world: RuleWorldConfig {
            seed: args.seed,
            preds_per_cell: args.preds_per_cell,
            max_level: args.max_level,
            guard_prob: args.guard_prob,
        },
        distractor_facts: args.distractor_facts,
        distractor_rules: args.distractor_rules,
        yes_frac: args.yes_frac,
        prop_literals: (2, 4), // unused by the rule arm
    })
    .expect("RuleCorpusConfig builds a valid dataset");

    let in_dist = dataset.rule_examples_eval(args.n, &train_depths, TraceArm::Trace, true);
    let len_gen = dataset.rule_examples_eval(args.n, &eval_depths, TraceArm::Trace, true);

    // Few-shot exemplars: TRAINING-stream draws (never rule_examples_eval),
    // one fixed set per axis, over that axis's own depths.
    let fewshot_in_dist = if args.few_shot > 0 {
        dataset.rule_examples(args.few_shot, &train_depths, TraceArm::Trace, true)
    } else {
        Vec::new()
    };
    let fewshot_len_gen = if args.few_shot > 0 {
        dataset.rule_examples(args.few_shot, &eval_depths, TraceArm::Trace, true)
    } else {
        Vec::new()
    };

    let mut f = std::fs::File::create(&args.out).expect("can create --out file");
    let mut n_written = 0usize;
    for (axis, examples, exemplars) in [
        ("in_dist", &in_dist, &fewshot_in_dist),
        ("len_gen", &len_gen, &fewshot_len_gen),
    ] {
        let exemplar_json: Vec<serde_json::Value> = exemplars
            .iter()
            .map(|e| {
                serde_json::json!({
                    "prompt": format_rule_prompt(e),
                    "target": e.target,
                })
            })
            .collect();
        for (i, ex) in examples.iter().enumerate() {
            let row = serde_json::json!({
                "id": format!("{axis}_{i:03}"),
                "axis": axis,
                "prompt": format_rule_prompt(ex),
                "expected": ex.gold_answer.surface(),
                "exemplars": exemplar_json,
            });
            writeln!(f, "{row}").expect("write jsonl row");
            n_written += 1;
        }
    }
    println!(
        "dump_rule_eval: wrote {n_written} rows ({} in_dist + {} len_gen, n={}, few_shot={}) -> {}",
        in_dist.len(),
        len_gen.len(),
        args.n,
        args.few_shot,
        args.out.display()
    );
}
