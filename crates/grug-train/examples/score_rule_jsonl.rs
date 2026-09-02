//! Phase 51 P4 "reasoning exam" harness — step 2: score a model's raw
//! completions (from `training/character/reasoning_exam.py`, run against the
//! prompts `dump_rule_eval.rs` dumped) with the real `score_rule_outputs`
//! replayer, printing the same `select/bind/fire/chain` summary shape
//! `train_rule.rs`/`eval_rule.rs` print.
//!
//! Regenerates the *same* in-dist / len-gen holdout examples
//! `dump_rule_eval.rs` dumped (pass matching `--n`/corpus flags — defaults
//! match), and matches them to `{"id", "axis", "output"}` rows read from
//! `--outputs` by `id` (`"{axis}_{i:03}"`).
//!
//! ```text
//! cargo run -p grug-train --example score_rule_jsonl -- \
//!     --outputs /tmp/rule_eval_outputs.jsonl --n 64 --label base-qwen3-0.6b
//! ```

use std::collections::HashMap;
use std::io::BufRead as _;
use std::path::PathBuf;

use clap::Parser;
use grug_data::{
    RuleCorpusConfig, RuleDataset, RuleEvalSummary, RuleExample, RuleWorldConfig, TraceArm,
    score_rule_outputs,
};

#[derive(Parser, Debug)]
#[command(name = "score_rule_jsonl")]
struct Args {
    /// JSONL of `{"id", "axis", "output"}` rows (model completions).
    #[arg(long)]
    outputs: PathBuf,

    /// Must match the `--n` used to dump the prompts.
    #[arg(long, default_value_t = 64)]
    n: usize,

    // --- world / corpus knobs (must match dump_rule_eval.rs) ---
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

    /// Optional: also write a JSON summary (same shape as eval_rule.json).
    #[arg(long)]
    summary_out: Option<PathBuf>,
    /// Free-text label recorded in the JSON summary (e.g. "base",
    /// "white-belt-v0", "checkpoint-0000400").
    #[arg(long, default_value = "")]
    label: String,
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

    let file = std::fs::File::open(&args.outputs)
        .unwrap_or_else(|e| panic!("can't open --outputs {}: {e}", args.outputs.display()));
    let reader = std::io::BufReader::new(file);
    let mut by_id: HashMap<String, String> = HashMap::new();
    for line in reader.lines() {
        let line = line.expect("readable line");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("invalid JSON line {line:?}: {e}"));
        let id = v
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or_else(|| panic!("row missing string \"id\": {line:?}"))
            .to_string();
        let output = v.get("output").and_then(|x| x.as_str()).unwrap_or("").to_string();
        by_id.insert(id, output);
    }
    println!(
        "score_rule_jsonl: loaded {} output rows from {}",
        by_id.len(),
        args.outputs.display()
    );

    let mut missing = 0usize;
    let mut score_axis = |axis: &str, examples: &[RuleExample]| -> RuleEvalSummary {
        let outputs: Vec<String> = (0..examples.len())
            .map(|i| {
                let id = format!("{axis}_{i:03}");
                match by_id.get(&id) {
                    Some(s) => s.clone(),
                    None => {
                        missing += 1;
                        String::new()
                    }
                }
            })
            .collect();
        score_rule_outputs(examples, &outputs)
    };

    println!(
        "\n=== Phase 51 reasoning exam -- label={:?} outputs={} ===",
        args.label,
        args.outputs.display()
    );

    let in_summary = score_axis("in_dist", &in_dist);
    print_summary_table("in-distribution", &in_summary);

    let len_summary = score_axis("len_gen", &len_gen);
    print_summary_table("length-generalization", &len_summary);

    if missing > 0 {
        eprintln!(
            "score_rule_jsonl: WARNING {missing} regenerated holdout ids had no matching output row (scored as empty string, i.e. answer/exec/valid=0 for those)"
        );
    }

    if let Some(out) = &args.summary_out {
        let summary_json = serde_json::json!({
            "label": args.label,
            "outputs_file": args.outputs.display().to_string(),
            "n_per_axis": args.n,
            "missing_outputs": missing,
            "in_distribution": summary_to_json(&in_summary),
            "length_generalization": summary_to_json(&len_summary),
        });
        std::fs::write(out, serde_json::to_string_pretty(&summary_json).unwrap())
            .unwrap_or_else(|e| panic!("can't write --summary-out {}: {e}", out.display()));
        println!("score_rule_jsonl: wrote {}", out.display());
    }
}

fn print_summary_table(label: &str, s: &RuleEvalSummary) {
    let c = &s.components;
    println!(
        "\n{label:<24}: n={} ans={:.3} exec={:.3} valid={:.3}",
        s.n, s.answer_accuracy, s.executability_rate, s.trace_valid_rate
    );
    println!(
        "  components           : select={:.3} bind={:.3} fire={:.3} chain={:.3} (fires={}, chain_expected={})",
        c.select_rate, c.bind_rate, c.fire_rate, c.chain_rate, c.fire_total, c.chain_expected_total
    );
    let mut depths: Vec<&usize> = s.per_depth.keys().collect();
    depths.sort();
    for depth in depths {
        let d = &s.per_depth[depth];
        println!(
            "    depth {depth}: n={:<4} ans={:.3} exec={:.3}",
            d.n, d.answer_accuracy, d.executability_rate
        );
    }
}

fn summary_to_json(s: &RuleEvalSummary) -> serde_json::Value {
    let per_depth: serde_json::Map<String, serde_json::Value> = s
        .per_depth
        .iter()
        .map(|(depth, st)| {
            (
                depth.to_string(),
                serde_json::json!({
                    "n": st.n,
                    "answer_accuracy": st.answer_accuracy,
                    "executability_rate": st.executability_rate,
                }),
            )
        })
        .collect();
    serde_json::json!({
        "n": s.n,
        "answer_accuracy": s.answer_accuracy,
        "executability_rate": s.executability_rate,
        "trace_valid_rate": s.trace_valid_rate,
        "components": {
            "select_rate": s.components.select_rate,
            "bind_rate": s.components.bind_rate,
            "fire_rate": s.components.fire_rate,
            "chain_rate": s.components.chain_rate,
            "fire_total": s.components.fire_total,
            "chain_expected_total": s.components.chain_expected_total,
        },
        "per_depth": per_depth,
    })
}
