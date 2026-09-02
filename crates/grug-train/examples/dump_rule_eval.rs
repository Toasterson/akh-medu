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
//! ```text
//! cargo run -p grug-train --example dump_rule_eval -- \
//!     --out /tmp/rule_eval.jsonl --n 64
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

    let mut f = std::fs::File::create(&args.out).expect("can create --out file");
    let mut n_written = 0usize;
    for (axis, examples) in [("in_dist", &in_dist), ("len_gen", &len_gen)] {
        for (i, ex) in examples.iter().enumerate() {
            let row = serde_json::json!({
                "id": format!("{axis}_{i:03}"),
                "axis": axis,
                "prompt": format_rule_prompt(ex),
            });
            writeln!(f, "{row}").expect("write jsonl row");
            n_written += 1;
        }
    }
    println!(
        "dump_rule_eval: wrote {n_written} rows ({} in_dist + {} len_gen, n={}) -> {}",
        in_dist.len(),
        len_gen.len(),
        args.n,
        args.out.display()
    );
}
