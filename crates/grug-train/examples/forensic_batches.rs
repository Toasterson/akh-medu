//! Throwaway forensic tool for the Phase 51 P4 350M loss/gradient "storm"
//! investigation: regenerate the exact deterministic training stream (same
//! corpus config + same in-process BPE training as `train_rule.rs`) and
//! characterize the examples that fall inside the observed epoch-periodic
//! trigger stream-index windows.
//!
//! CPU-only, default features (no gpu/cuda) — this only needs the corpus +
//! tokenizer + encoder, no model forward pass.
//!
//! ```text
//! cargo run -p grug-train --example forensic_batches --release
//! ```

use grug_data::{GrugTruth, RuleCorpusConfig, RuleDataset, RuleExample, RuleWorldConfig, TraceArm};
use grug_train::rule_train::{encode_rule_example, format_rule_prompt};
use grug_train::BpeTokenizer;

/// Same MAX_SEQ_LEN as `training/configs/350m.env` (`MAX_SEQ_LEN=1024`), the
/// context window `build_rule_batch` truncates to at train time.
const MAX_SEQ_LEN: usize = 1024;

const BATCH_SIZE: usize = 16;

fn main() {
    // --- 1. regenerate the exact 40000-example train stream, EXACTLY as
    // train_rule.rs does (RuleCorpusConfig / RuleDataset::rule_examples). ---
    let dataset = RuleDataset::new(RuleCorpusConfig {
        world: RuleWorldConfig {
            seed: 42,
            preds_per_cell: 6,
            max_level: 6,
            guard_prob: 0.5,
        },
        distractor_facts: 4,
        distractor_rules: 6,
        yes_frac: 0.5,
        prop_literals: (2, 4), // unused by the rule arm, train_rule.rs default
    })
    .expect("RuleCorpusConfig builds a valid dataset");

    let train_depths = vec![2usize, 3, 4];
    let arm = TraceArm::Trace; // --arm trace
    let typed_slots = true;
    let train_examples: usize = 40_000;

    println!("forensic_batches: generating {train_examples} rule examples (seed=42, depths={train_depths:?}, arm=trace, typed_slots=true)...");
    let stream: Vec<RuleExample> = dataset.rule_examples(train_examples, &train_depths, arm, typed_slots);
    assert_eq!(stream.len(), train_examples);

    // --- 2. train the BPE in-process, EXACTLY as train_rule.rs's
    // load_or_train_tokenizer (take(60_000), prompt+target per example, vocab
    // 3000). Since stream.len() == 40000 < 60000, this is just all of them. ---
    println!("forensic_batches: training BPE tokenizer (vocab_size=3000) from the corpus...");
    let corpus: Vec<String> = stream
        .iter()
        .take(60_000)
        .flat_map(|ex| vec![format_rule_prompt(ex).to_string(), ex.target.clone()])
        .collect();
    let tokenizer = BpeTokenizer::train(corpus, 3000).expect("BPE training");
    println!(
        "forensic_batches: BPE vocab_size={} (requested 3000)",
        tokenizer.vocab_size()
    );

    // --- 3. encode every example, recording forensic stats. ---
    struct Stat {
        idx: usize,
        total_tokens: usize,
        target_start: usize,
        target_tokens: usize,
        depth: usize,
        yes: bool,
        prompt_chars: usize,
        target_chars: usize,
    }

    let mut stats: Vec<Stat> = Vec::with_capacity(stream.len());
    for (idx, ex) in stream.iter().enumerate() {
        let enc = encode_rule_example(&tokenizer, ex).expect("encoding should not fail");
        let total_tokens = enc.ids.len();
        let target_start = enc.target_start;
        let target_tokens = total_tokens.saturating_sub(target_start);
        stats.push(Stat {
            idx,
            total_tokens,
            target_start,
            target_tokens,
            depth: ex.depth,
            yes: ex.gold_answer == GrugTruth::Yes,
            prompt_chars: ex.prompt.chars().count(),
            target_chars: ex.target.chars().count(),
        });
    }

    // Classification helpers, matching build_rule_batch's truncate-to-longest
    // semantics *when the batch's own longest example already exceeds
    // MAX_SEQ_LEN* (the `.min(max_seq_len)` in build_rule_batch). We report
    // the per-example truncation outcome as if MAX_SEQ_LEN were the batch's
    // effective cap (a conservative/representative view — the real cap is
    // min(longest_in_batch, MAX_SEQ_LEN), so any example this flags is a
    // genuine risk whenever it shares a batch with something at least as
    // long, which is exactly the trigger-window batches we're inspecting).
    let is_over_ctx = |s: &Stat| s.total_tokens > MAX_SEQ_LEN;
    // empty loss mask: input/target arrays are ids[0..longest-1] /
    // ids[1..longest]; a token at position i+1 counts as target-loss iff
    // i+1 >= target_start AND ids[i+1] != pad. With truncation to
    // MAX_SEQ_LEN, if target_start >= MAX_SEQ_LEN - 1 (i.e. >= seq_len,
    // since seq_len = longest-1 <= MAX_SEQ_LEN-1), there is no i in
    // [0, seq_len) with i+1 >= target_start, so the mask is entirely empty.
    let is_empty_mask = |s: &Stat| s.target_start >= MAX_SEQ_LEN.saturating_sub(1);
    // partial target cut: over context AND not fully empty-masked, i.e. the
    // target starts before the cap but the full target (target_start +
    // target_tokens) extends past it -- truncation lops off the tail of the
    // target/EOS, corrupting (not eliminating) supervision.
    let is_target_cut = |s: &Stat| {
        is_over_ctx(s) && !is_empty_mask(s) && (s.target_start + s.target_tokens) > MAX_SEQ_LEN
    };

    // --- 4. trigger windows (stream indices, batch_size=16). ---
    let mut windows: Vec<(&str, std::ops::Range<usize>)> = vec![
        ("steps 874-882", 874 * BATCH_SIZE..883 * BATCH_SIZE),
        ("steps 1220-1231", 1220 * BATCH_SIZE..1232 * BATCH_SIZE),
        ("steps 2084-2095", 2084 * BATCH_SIZE..2096 * BATCH_SIZE),
    ];
    let scattered_steps = [234usize, 260, 601, 1108, 1148, 1361, 2206, 2306];
    for step in scattered_steps {
        // leak a short-lived label string; fine for a throwaway forensic tool
        let label: &'static str = Box::leak(format!("step {step}").into_boxed_str());
        windows.push((label, step * BATCH_SIZE..(step + 1) * BATCH_SIZE));
    }

    // Stream length is 40000 but stream indices in train_rule.rs actually
    // wrap via `% train_examples.len()`; since train_examples == stream.len()
    // == 40000 here, step*16..step*16+16 for step up to 2499 (40000/16) never
    // wraps, so plain slicing is correct and matches the real run exactly.
    let window_stats: Vec<(&str, Vec<&Stat>)> = windows
        .iter()
        .map(|(label, range)| {
            let sl = stats[range.start.min(stats.len())..range.end.min(stats.len())]
                .iter()
                .collect::<Vec<_>>();
            (*label, sl)
        })
        .collect();

    // Random control sample of 200 (deterministic pseudo-random pick via a
    // fixed stride so it's reproducible without pulling in `rand`).
    let control: Vec<&Stat> = {
        let stride = stats.len() / 200;
        (0..200).map(|i| &stats[(i * stride) % stats.len()]).collect()
    };

    // --- 5. reporting helpers. ---
    fn pct(count: usize, total: usize) -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * count as f64 / total as f64
        }
    }

    fn summarize(
        label: &str,
        items: &[&Stat],
        is_over_ctx: impl Fn(&Stat) -> bool,
        is_empty_mask: impl Fn(&Stat) -> bool,
        is_target_cut: impl Fn(&Stat) -> bool,
    ) {
        let n = items.len();
        if n == 0 {
            println!("{label}: n=0 (out of stream range)");
            return;
        }
        let mut lens: Vec<usize> = items.iter().map(|s| s.total_tokens).collect();
        lens.sort_unstable();
        let mean = lens.iter().sum::<usize>() as f64 / n as f64;
        let p95 = lens[((n as f64 * 0.95) as usize).min(n - 1)];
        let max = *lens.last().unwrap();
        let over = items.iter().filter(|s| is_over_ctx(s)).count();
        let empty = items.iter().filter(|s| is_empty_mask(s)).count();
        let cut = items.iter().filter(|s| is_target_cut(s)).count();

        let mut depth_counts = std::collections::BTreeMap::new();
        for s in items {
            *depth_counts.entry(s.depth).or_insert(0usize) += 1;
        }
        let n_yes = items.iter().filter(|s| s.yes).count();

        println!(
            "{label:<20} n={n:5} tok(mean/p95/max)={mean:7.1}/{p95:5}/{max:5}  over_ctx={over:4} ({:5.1}%)  empty_mask={empty:4} ({:5.1}%)  target_cut={cut:4} ({:5.1}%)  yes={n_yes:4}/{n}  depths={depth_counts:?}",
            pct(over, n),
            pct(empty, n),
            pct(cut, n),
        );
    }

    println!();
    println!("=== Forensic report: Phase 51 P4 350M loss/gradient storm ===");
    println!("MAX_SEQ_LEN (train cap) = {MAX_SEQ_LEN}, batch_size = {BATCH_SIZE}, stream_len = {}", stats.len());
    println!();

    let all_refs: Vec<&Stat> = stats.iter().collect();
    println!("--- (a) Full 40000-example stream ---");
    summarize("FULL STREAM", &all_refs, is_over_ctx, is_empty_mask, is_target_cut);
    println!();

    println!("--- (b) Trigger ranges (epoch-periodic non-finite-gradient windows) ---");
    for (label, items) in &window_stats {
        summarize(label, items, is_over_ctx, is_empty_mask, is_target_cut);
    }
    println!();

    println!("--- (c) Random control sample (n=200, stride-sampled) ---");
    summarize("CONTROL (n=200)", &control, is_over_ctx, is_empty_mask, is_target_cut);
    println!();

    // --- 6. compact comparison table (trigger ranges vs global). ---
    println!("--- Compact comparison table: trigger ranges vs global ---");
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>10} {:>30}",
        "range", "%over_ctx", "%target_cut", "%empty_mask", "max_tok", "depth_mix (count)"
    );
    let print_row = |label: &str, items: &[&Stat]| {
        let n = items.len();
        if n == 0 {
            println!("{label:<20} {:>10}", "n=0");
            return;
        }
        let over = items.iter().filter(|s| is_over_ctx(s)).count();
        let empty = items.iter().filter(|s| is_empty_mask(s)).count();
        let cut = items.iter().filter(|s| is_target_cut(s)).count();
        let max = items.iter().map(|s| s.total_tokens).max().unwrap_or(0);
        let mut depth_counts = std::collections::BTreeMap::new();
        for s in items {
            *depth_counts.entry(s.depth).or_insert(0usize) += 1;
        }
        println!(
            "{label:<20} {:>9.1}% {:>11.1}% {:>11.1}% {:>10} {:>30}",
            pct(over, n),
            pct(cut, n),
            pct(empty, n),
            max,
            format!("{depth_counts:?}")
        );
    };
    print_row("FULL (global)", &all_refs);
    print_row("CONTROL (n=200)", &control);
    for (label, items) in &window_stats {
        print_row(label, items);
    }
    println!();

    // --- 7. verdict line. ---
    let global_over = pct(all_refs.iter().filter(|s| is_over_ctx(s)).count(), all_refs.len());
    let trigger_total: Vec<&Stat> = window_stats.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let trigger_over = pct(trigger_total.iter().filter(|s| is_over_ctx(s)).count(), trigger_total.len().max(1));
    let trigger_cut = pct(trigger_total.iter().filter(|s| is_target_cut(s)).count(), trigger_total.len().max(1));
    let trigger_empty = pct(trigger_total.iter().filter(|s| is_empty_mask(s)).count(), trigger_total.len().max(1));
    println!(
        "VERDICT-INPUTS: global_over_ctx%={global_over:.2} trigger_over_ctx%={trigger_over:.2} trigger_target_cut%={trigger_cut:.2} trigger_empty_mask%={trigger_empty:.2} (n_trigger_examples={})",
        trigger_total.len()
    );
}
