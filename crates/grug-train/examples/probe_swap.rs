//! Phase 51 GWT workspace-emergence probe, Stage S: **single-swap-read-by-many**
//! (`docs/ai/research/2026-07-09-global-workspace-language-models.md` §7.2) --
//! the JSON-era proxy for Anthropic's France->China broadcast test, run on our
//! own tool-calling corpus.
//!
//! The question: if we patch the hidden state at a shared argument value's
//! *prompt* positions at layer `ell`, does that patch propagate correctly into
//! the generated call -- evidence of a broadcast-like locus ("write once, read
//! by many") -- or does it only affect lexically-adjacent computation?
//!
//! # Family construction (see module docs on `find_verbatim_string_args` and
//! `build_families` for the exact mining procedure)
//!
//! From the **training split** (in-distribution), this mines `--families`
//! groups sharing one substitutable string argument value `A`, each with up to
//! `--family-k` different (tool, phrasing) framings that happen to embed the
//! same fuzzed value verbatim in their request text -- the "read by many" set.
//! A different value `B` (also drawn from the corpus's own observed values, so
//! it is guaranteed to be a legal same-type substitute) is spliced in at the
//! prompt positions where `A`'s tokens differ from `B`'s.
//!
//! Inference only. Loads a trained checkpoint the same way `eval_repair.rs`
//! does (`Dataset::new_sized_with_distractors`, the plain non-registry P1a
//! reconstruction -- Stage S was specified against this reconstruction, not
//! the registry one).
//!
//! ```text
//! source scripts/train-env.sh
//! cargo run -p grug-train --example probe_swap --release -- \
//!     --artifacts-dir /home/toasty/ws/agi/akh-medu/runs/p2-grug --format grug --sweep
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use clap::Parser;
use serde::Serialize;

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

use grug_data::schema::{Example, ParamType, ToolCall, ToolSchema};
use grug_data::{Dataset, DistractorConfig, TargetFormat};
use grug_train::generate::{GenerateConfig, generate};
use grug_train::tokenizer::Vocab;
use grug_train::train::format_prompt_fmt;
use grug_train::transformer::{TinyTransformer, TinyTransformerConfig};
use grug_train::{AttentionBackend, BpeTokenizer, checkpoint};

// Inference-only probe: a plain (non-autodiff) backend, same rationale as
// `probe_lens.rs`.
#[cfg(not(feature = "gpu"))]
use grug_train::CpuBackend as ActiveBackend;
#[cfg(feature = "gpu")]
use grug_train::GpuBackend as ActiveBackend;

#[derive(Parser, Debug)]
#[command(name = "probe_swap")]
struct Args {
    #[arg(long)]
    artifacts_dir: PathBuf,
    /// Checkpoint subdirectory under `<artifacts-dir>/checkpoints/`; defaults
    /// to the lexicographically last (latest-step) entry.
    #[arg(long)]
    ckpt: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1000)]
    n_tools: usize,
    #[arg(long, default_value_t = 0.15)]
    holdout_frac: f64,
    #[arg(long, default_value_t = 2)]
    distractors: usize,
    #[arg(long, default_value = "json")]
    format: String,
    /// Target number of A-value families to build.
    #[arg(long, default_value_t = 24)]
    families: usize,
    /// Target "read by many" width K per family (>= 3 is the real broadcast
    /// test; families where the corpus can't supply that many distinct
    /// framings fall back to K < 3 and are flagged `fallback_k1` in the
    /// output).
    #[arg(long, default_value_t = 4)]
    family_k: usize,
    /// Patch a single layer instead of sweeping every layer.
    #[arg(long)]
    layer: Option<usize>,
    /// Sweep every layer (0..=n_layers); wins over `--layer` if both given.
    /// Also the default when `--layer` is omitted.
    #[arg(long, default_value_t = false)]
    sweep: bool,
    #[arg(long, default_value_t = 96)]
    max_new_tokens: usize,
    /// How many training examples to draw when mining substitution families.
    #[arg(long, default_value_t = 6000)]
    pool_size: usize,
    #[arg(long)]
    out: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Family mining
// ---------------------------------------------------------------------------

/// A single "read by many" family member: one (tool, phrasing) framing whose
/// request embeds the family's shared value `A` verbatim, plus everything
/// needed to run patched generation without re-deriving it per layer/step.
struct FamilyMember {
    tool_name: String,
    param_name: String,
    /// The `A`-sequence prompt string (tools schema + request), for the
    /// unpatched `control_A` measurement.
    prompt_a: String,
    /// `BOS <prompt_a tokens> SEP`.
    ids_a_prompt: Vec<u32>,
    /// `BOS <prompt_b tokens> SEP`, same length as `ids_a_prompt` by
    /// construction (mining rejects any pair that doesn't tokenize to equal
    /// length).
    ids_b_prompt: Vec<u32>,
    /// Inclusive `[lo, hi]` token-index span (into both prompt id arrays
    /// above) where `A`'s and `B`'s encodings differ -- the splice target.
    value_span: (usize, usize),
    offered_tools: Vec<ToolSchema>,
    gold_call: ToolCall,
}

struct Family {
    value_a: String,
    value_b: String,
    members: Vec<FamilyMember>,
    /// The corpus could not supply >= 3 distinct framings sharing `value_a`
    /// (different tools, or repeated framings of the same tool) -- this
    /// family degraded to the K=1 fallback the spec allows.
    fallback_k1: bool,
}

/// Every `(param_name, value)` pair in `example`'s gold call where the param
/// is declared `String` on the offered schema and the fuzzed value appears
/// verbatim in `user_request`. `grug_data::phrasing::TemplatePart::Arg`
/// documents this as a *structural* guarantee of the procedural generator's
/// pattern-based templates (see `crates/grug-data/src/phrasing.rs`), so this
/// is a cheap substring check, not a heuristic -- abstention examples (whose
/// gold is the gap sentinel, no real schema) are excluded up front.
fn find_verbatim_string_args(example: &Example) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if example.is_abstention() {
        return out;
    }
    let Some(schema) = example.expected_schema() else {
        return out;
    };
    for (k, v) in example.expected_call.arguments_map() {
        let Some(s) = v.as_str() else { continue };
        if s.is_empty() {
            continue;
        }
        let is_string_param = matches!(schema.param(&k).map(|p| &p.ty), Some(ParamType::String));
        if is_string_param && example.user_request.contains(s) {
            out.push((k, s.to_string()));
        }
    }
    out
}

/// `BOS <encode(prompt)> SEP` -- the framing every generation path in this
/// crate (`crate::generate`, `crate::train::encode_example_fmt`) uses.
fn encode_prompt_ids<V: Vocab>(tokenizer: &V, prompt: &str) -> grug_train::GrugResult<Vec<u32>> {
    let mut ids = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt)?);
    ids.push(tokenizer.sep_id());
    Ok(ids)
}

/// Mine `target_families` value-swap families out of `pool` (drawn from the
/// **training** split, in-distribution per the Stage-S spec).
///
/// For each candidate shared value `A` (ranked by how many distinct occurrences
/// it has in the pool, richest first -- the values most likely to yield a real
/// K>=3 "read by many" set), a distinct value `B` is chosen from the pool's own
/// observed value set (guaranteeing `B` is a legal same-type substitute, since
/// it is itself something the generator actually produced for a `String`
/// param), and up to `family_k` occurrences of `A` are validated: `request_a`
/// with `A` replaced by `B` must tokenize to the **same length** as `request_a`
/// itself (the spec's "resample otherwise" contract) -- occurrences that don't
/// satisfy this are dropped rather than resampled from scratch, since the pool
/// already offers plenty of alternate occurrences of the same `A`. Occurrences
/// are preferred by *distinct tool name* first, so a family's members are
/// genuinely different downstream computations rather than the same tool asked
/// K different ways -- falling back to repeated tools only if the pool can't
/// supply enough distinct ones.
fn build_families<V: Vocab>(
    tokenizer: &V,
    pool: &[Example],
    fmt: TargetFormat,
    target_families: usize,
    family_k: usize,
    max_seq_len: usize,
    max_new_tokens: usize,
) -> Vec<Family> {
    struct Occurrence {
        example_idx: usize,
        param: String,
    }

    let mut occurrences: Vec<Occurrence> = Vec::new();
    let mut by_value: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, ex) in pool.iter().enumerate() {
        for (param, value) in find_verbatim_string_args(ex) {
            let occ_idx = occurrences.len();
            by_value.entry(value).or_default().push(occ_idx);
            occurrences.push(Occurrence {
                example_idx: i,
                param,
            });
        }
    }

    let mut distinct_values: Vec<String> = by_value.keys().cloned().collect();
    distinct_values.sort_by(|a, b| {
        by_value[b]
            .len()
            .cmp(&by_value[a].len())
            .then_with(|| a.cmp(b))
    });

    if distinct_values.len() < 2 {
        // Not enough distinct string values in the whole pool to substitute
        // anything -- nothing to build.
        return Vec::new();
    }

    let room_for_span = max_seq_len.saturating_sub(max_new_tokens);
    let mut families: Vec<Family> = Vec::new();

    for a_val in &distinct_values {
        if families.len() >= target_families {
            break;
        }
        let occ_indices = &by_value[a_val];

        // Prefer distinct tool names first, then fill with repeats.
        let mut chosen: Vec<usize> = Vec::new();
        let mut seen_tools: HashSet<&str> = HashSet::new();
        for &oi in occ_indices {
            if chosen.len() >= family_k {
                break;
            }
            let ex = &pool[occurrences[oi].example_idx];
            if seen_tools.insert(ex.expected_call.name.as_str()) {
                chosen.push(oi);
            }
        }
        if chosen.len() < family_k {
            for &oi in occ_indices {
                if chosen.len() >= family_k {
                    break;
                }
                if !chosen.contains(&oi) {
                    chosen.push(oi);
                }
            }
        }

        // A fixed single B candidate is not enough: the equal-token-length
        // contract below depends on how A and B happen to tokenize, so a
        // particular (A, B) pairing can reject *every* member (observed on the
        // real p1a2/p2-grug corpora: the top-ranked A's first candidate B
        // tokenized to a different length and mining produced zero families).
        // Try B candidates in ranked order until one yields members, bounded
        // so a pathological value bank can't make this quadratic-expensive.
        const MAX_B_ATTEMPTS: usize = 32;
        let mut family: Option<(String, Vec<FamilyMember>)> = None;
        for b_val in distinct_values
            .iter()
            .filter(|v| *v != a_val)
            .take(MAX_B_ATTEMPTS)
        {
            let mut members = Vec::new();
            for &oi in &chosen {
                let occ = &occurrences[oi];
                let ex = &pool[occ.example_idx];

                let request_b = ex.user_request.replace(a_val.as_str(), b_val.as_str());
                if request_b == ex.user_request {
                    continue;
                }
                let mut ex_b = ex.clone();
                ex_b.user_request = request_b;

                let prompt_a = format_prompt_fmt(ex, fmt);
                let prompt_b = format_prompt_fmt(&ex_b, fmt);

                let (Ok(ids_a), Ok(ids_b)) = (
                    encode_prompt_ids(tokenizer, &prompt_a),
                    encode_prompt_ids(tokenizer, &prompt_b),
                ) else {
                    continue;
                };
                // Spec contract: equal tokenized length, difference confined to a
                // contiguous span -- computed below as [first_diff, last_diff],
                // which is contiguous by construction. Reject (rather than
                // resample in-place -- the pool has other occurrences of the same
                // A) on length mismatch or a prompt too long to leave headroom for
                // generation.
                if ids_a.len() != ids_b.len() || ids_a.len() > room_for_span {
                    continue;
                }
                let mut lo = None;
                let mut hi = None;
                for (idx, (x, y)) in ids_a.iter().zip(ids_b.iter()).enumerate() {
                    if x != y {
                        lo.get_or_insert(idx);
                        hi = Some(idx);
                    }
                }
                let (Some(lo), Some(hi)) = (lo, hi) else {
                    continue;
                };

                members.push(FamilyMember {
                    tool_name: ex.expected_call.name.clone(),
                    param_name: occ.param.clone(),
                    prompt_a,
                    ids_a_prompt: ids_a,
                    ids_b_prompt: ids_b,
                    value_span: (lo, hi),
                    offered_tools: ex.tools.clone(),
                    gold_call: ex.expected_call.clone(),
                });
            }

            if !members.is_empty() {
                family = Some((b_val.clone(), members));
                break;
            }
        }

        let Some((value_b, members)) = family else {
            continue;
        };
        let fallback_k1 = members.len() < 3;
        families.push(Family {
            value_a: a_val.clone(),
            value_b,
            members,
            fallback_k1,
        });
    }

    families
}

// ---------------------------------------------------------------------------
// Patched generation
// ---------------------------------------------------------------------------

fn argmax<B: Backend>(logits: Tensor<B, 1>) -> u32 {
    let idx: Tensor<B, 1, Int> = logits.argmax(0);
    let value: i64 = idx.into_scalar().elem();
    value as u32
}

/// Splice `h_b`'s rows at `span` (inclusive token-index range, PROMPT
/// positions only) into `h_a`, leaving everything else untouched.
fn splice_hidden<B: Backend>(
    h_a: Tensor<B, 3>,
    h_b: &Tensor<B, 3>,
    span: (usize, usize),
) -> Tensor<B, 3> {
    let d_model = h_a.dims()[2];
    let (lo, hi) = span;
    let values = h_b.clone().slice([0..1, lo..hi + 1, 0..d_model]);
    h_a.slice_assign([0..1, lo..hi + 1, 0..d_model], values)
}

/// Greedy patched generation (Stage S core op): at every decode step, run
/// `hidden_layers` on both the live `A`-sequence and a `B`-donor sequence that
/// shares the exact same generated suffix so far (recomputed fresh each step,
/// per the spec -- there is no KV cache to reuse here, matching every other
/// generation path in this crate), splice `B`'s rows into `A`'s hidden state at
/// `layer` over `member.value_span` (prompt positions only -- the generated
/// suffix is never patched), then resume the trunk from that spliced state via
/// `hidden_from` and take the next-token argmax from the last position --
/// exactly [`grug_train::generate::generate`]'s unpatched decode rule.
fn generate_patched<B: Backend + AttentionBackend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    member: &FamilyMember,
    layer: usize,
    max_new_tokens: usize,
    max_seq_len: usize,
    device: &B::Device,
) -> String {
    let eos_id = tokenizer.eos_id();
    let mut generated: Vec<u32> = Vec::new();

    for _ in 0..max_new_tokens {
        let mut seq_a = member.ids_a_prompt.clone();
        seq_a.extend(generated.iter().copied());
        let mut seq_b = member.ids_b_prompt.clone();
        seq_b.extend(generated.iter().copied());

        if seq_a.len() > max_seq_len {
            break;
        }
        let seq_len = seq_a.len();

        let to_input = |ids: &[u32]| -> Tensor<B, 2, Int> {
            let flat: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
            Tensor::<B, 2, Int>::from_data(TensorData::new(flat, [1, ids.len()]), device)
        };

        let layers_a = model.hidden_layers(to_input(&seq_a));
        let layers_b = model.hidden_layers(to_input(&seq_b));
        let layer_idx = layer.min(layers_a.len() - 1);
        let h_a = layers_a[layer_idx].clone();
        let h_b = &layers_b[layer_idx];

        let patched = splice_hidden(h_a, h_b, member.value_span);
        let hidden = model.hidden_from(patched, layer_idx);
        let logits = model.vocab_logits(hidden); // [1, seq_len, vocab]
        let vocab_size = logits.dims()[2];
        let last = logits
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        let next_id = argmax(last);
        if next_id == eos_id {
            break;
        }
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

fn value_matches(call: &ToolCall, param: &str, expected: &str) -> bool {
    call.arguments.get(param).and_then(|v| v.as_str()) == Some(expected)
}

/// "Otherwise equals the gold call": same tool name, and every argument other
/// than `patched_param` matches ground truth exactly (no drift, no stray
/// extras) -- `patched_param` itself is excluded since it is *expected* to
/// have changed under a successful patch.
fn call_intact(call: &ToolCall, gold: &ToolCall, patched_param: &str) -> bool {
    if call.name != gold.name {
        return false;
    }
    let got = call.arguments_map();
    let want = gold.arguments_map();
    for (k, wv) in &want {
        if k == patched_param {
            continue;
        }
        match got.get(k) {
            Some(gv) if gv == wv => {}
            _ => return false,
        }
    }
    for k in got.keys() {
        if k != patched_param && !want.contains_key(k) {
            return false;
        }
    }
    true
}

fn ratio(n: u64, d: u64) -> f64 {
    if d == 0 { 0.0 } else { n as f64 / d as f64 }
}

#[derive(Debug, Clone, Serialize)]
struct LayerResult {
    layer: usize,
    flip_rate: f64,
    intact_rate: f64,
    flip_and_intact_rate: f64,
    /// Fraction of scored families (K >= 1) where *every* member flipped at
    /// this layer -- the literal "read by many" test. Only informative for
    /// families with K >= 3; families that fell back to K < 3 still
    /// contribute (trivially: K=1 "all flipped" == "the one member flipped"),
    /// which biases this number upward when many families fell back -- see
    /// `families_fallback_k1` in the JSON output before over-reading this.
    read_by_many: f64,
    n_members: u64,
    n_families_scored: u64,
}

fn print_family_summary(families: &[Family]) {
    let fallback = families.iter().filter(|f| f.fallback_k1).count();
    println!(
        "\n--- {} families mined ({} fell back to K<3) ---",
        families.len(),
        fallback
    );
    for f in families {
        let tools: Vec<&str> = f.members.iter().map(|m| m.tool_name.as_str()).collect();
        println!(
            "  {:>10} -> {:<10} K={} tools={:?}{}",
            f.value_a,
            f.value_b,
            f.members.len(),
            tools,
            if f.fallback_k1 {
                "  [fallback K<3]"
            } else {
                ""
            }
        );
    }
}

fn print_layer_table(results: &[LayerResult]) {
    println!(
        "\n{:>6} | {:>9} | {:>10} | {:>16} | {:>13} | {:>9}",
        "layer", "flip", "intact", "flip_and_intact", "read_by_many", "n"
    );
    for r in results {
        println!(
            "{:>6} | {:>9.3} | {:>10.3} | {:>16.3} | {:>13.3} | {:>9}",
            r.layer,
            r.flip_rate,
            r.intact_rate,
            r.flip_and_intact_rate,
            r.read_by_many,
            r.n_members
        );
    }
}

/// Resolve the checkpoint directory: `--ckpt` if given, else the
/// lexicographically last `checkpoints/<name>` subdirectory.
fn resolve_checkpoint_dir(artifacts_dir: &Path, ckpt: Option<&str>) -> PathBuf {
    let checkpoints_dir = artifacts_dir.join("checkpoints");
    if let Some(name) = ckpt {
        return checkpoints_dir.join(name);
    }
    let mut entries: Vec<String> = std::fs::read_dir(&checkpoints_dir)
        .unwrap_or_else(|e| panic!("reading checkpoints dir {}: {e}", checkpoints_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    entries.sort();
    let last = entries.last().unwrap_or_else(|| {
        panic!(
            "no checkpoint subdirectories found under {}",
            checkpoints_dir.display()
        )
    });
    checkpoints_dir.join(last)
}

fn main() {
    let args = Args::parse();
    let fmt = TargetFormat::parse_label(&args.format).expect("--format json|grug");

    #[cfg(feature = "gpu")]
    let device = {
        let d = burn::backend::rocm::RocmDevice::default();
        println!("probe_swap: GPU backend, device: {d:?}");
        d
    };
    #[cfg(not(feature = "gpu"))]
    let device = {
        println!("probe_swap: CPU backend");
        Default::default()
    };

    let tokenizer =
        BpeTokenizer::load(args.artifacts_dir.join("tokenizer.json")).expect("tokenizer loads");
    let model_config = TinyTransformerConfig::p1a(tokenizer.vocab_size());
    let max_seq_len = model_config.max_seq_len;
    let n_layers = model_config.n_layers;

    let ckpt_dir = resolve_checkpoint_dir(&args.artifacts_dir, args.ckpt.as_deref());
    let model = model_config.init::<ActiveBackend>(&device);
    let (model, step) =
        checkpoint::load_model_only(&ckpt_dir, model, &device).expect("checkpoint loads");
    println!(
        "probe_swap: FORMAT={fmt:?} checkpoint {} (step {step})",
        ckpt_dir.display()
    );

    let distractors = DistractorConfig::easy_with_count(args.distractors);
    // eval_repair.rs's reconstruction: plain P1a dataset, training split.
    let dataset = Dataset::new_sized_with_distractors(
        args.seed,
        args.n_tools,
        args.holdout_frac,
        distractors,
    )
    .expect("dataset splits cleanly");
    let pool = dataset
        .train_examples(args.pool_size)
        .expect("training pool draws");

    let families = build_families(
        &tokenizer,
        &pool,
        fmt,
        args.families,
        args.family_k,
        max_seq_len,
        args.max_new_tokens,
    );
    print_family_summary(&families);

    if families.is_empty() {
        eprintln!("probe_swap: mined zero families -- nothing to score. Try a larger --pool-size.");
        return;
    }

    let layers_to_run: Vec<usize> = match (args.layer, args.sweep) {
        (Some(_), true) => (0..=n_layers).collect(),
        (Some(l), false) => vec![l],
        (None, _) => (0..=n_layers).collect(),
    };

    // control_A: unpatched generation on the A-sequence, measured once per
    // family (its first member), independent of layer.
    let gen_config = GenerateConfig {
        max_new_tokens: args.max_new_tokens,
        max_seq_len,
    };
    let mut control_a_hits = 0u64;
    let mut control_a_total = 0u64;
    for family in &families {
        if let Some(m0) = family.members.first() {
            let raw = generate(&model, &tokenizer, &m0.prompt_a, gen_config, &device);
            control_a_total += 1;
            if let Some(call) = grug_data::parse_call(&raw, fmt, &m0.offered_tools)
                && value_matches(&call, &m0.param_name, &family.value_a)
            {
                control_a_hits += 1;
            }
        }
    }
    let control_a_rate = ratio(control_a_hits, control_a_total);

    let mut layer_results: Vec<LayerResult> = Vec::new();
    for &layer in &layers_to_run {
        let mut n_members = 0u64;
        let mut flip_hits = 0u64;
        let mut intact_hits = 0u64;
        let mut flip_and_intact_hits = 0u64;
        let mut families_all_flip = 0u64;
        let mut families_eligible = 0u64;

        for family in &families {
            let mut member_flip_flags: Vec<bool> = Vec::with_capacity(family.members.len());
            for member in &family.members {
                let raw = generate_patched(
                    &model,
                    &tokenizer,
                    member,
                    layer,
                    args.max_new_tokens,
                    max_seq_len,
                    &device,
                );
                let parsed = grug_data::parse_call(&raw, fmt, &member.offered_tools);
                let flipped = parsed
                    .as_ref()
                    .is_some_and(|c| value_matches(c, &member.param_name, &family.value_b));
                let intact = parsed
                    .as_ref()
                    .is_some_and(|c| call_intact(c, &member.gold_call, &member.param_name));

                n_members += 1;
                if flipped {
                    flip_hits += 1;
                }
                if intact {
                    intact_hits += 1;
                }
                if flipped && intact {
                    flip_and_intact_hits += 1;
                }
                member_flip_flags.push(flipped);
            }
            if !member_flip_flags.is_empty() {
                families_eligible += 1;
                if member_flip_flags.iter().all(|&f| f) {
                    families_all_flip += 1;
                }
            }
        }

        layer_results.push(LayerResult {
            layer,
            flip_rate: ratio(flip_hits, n_members),
            intact_rate: ratio(intact_hits, n_members),
            flip_and_intact_rate: ratio(flip_and_intact_hits, n_members),
            read_by_many: ratio(families_all_flip, families_eligible),
            n_members,
            n_families_scored: families_eligible,
        });
    }

    println!("\n=== Phase 51 GWT probe -- Stage S (single-swap-read-by-many) ===");
    println!(
        "control_A (unpatched generation still emits A): {:.3} (n={})",
        control_a_rate, control_a_total
    );
    print_layer_table(&layer_results);

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.artifacts_dir.join("probe_swap.json"));
    let families_json: Vec<serde_json::Value> = families
        .iter()
        .map(|f| {
            serde_json::json!({
                "value_a": f.value_a,
                "value_b": f.value_b,
                "k": f.members.len(),
                "fallback_k1": f.fallback_k1,
                "members": f.members.iter().map(|m| serde_json::json!({
                    "tool": m.tool_name,
                    "param": m.param_name,
                    "value_span": [m.value_span.0, m.value_span.1],
                })).collect::<Vec<_>>(),
            })
        })
        .collect();

    let summary = serde_json::json!({
        "checkpoint_step": step,
        "checkpoint_dir": ckpt_dir.display().to_string(),
        "format": args.format,
        "seed": args.seed,
        "n_tools": args.n_tools,
        "holdout_frac": args.holdout_frac,
        "distractors": args.distractors,
        "pool_size": args.pool_size,
        "families_requested": args.families,
        "family_k_target": args.family_k,
        "families_built": families.len(),
        "families_fallback_k1": families.iter().filter(|f| f.fallback_k1).count(),
        "max_new_tokens": args.max_new_tokens,
        "layer_arg": args.layer,
        "sweep": args.sweep,
        "n_layers": n_layers,
        "control_a_rate": control_a_rate,
        "control_a_n": control_a_total,
        "families": families_json,
        "layers": layer_results,
    });
    match std::fs::write(&out_path, serde_json::to_string_pretty(&summary).unwrap()) {
        Ok(()) => println!("\nprobe_swap: wrote {}", out_path.display()),
        Err(e) => eprintln!("probe_swap: write failed: {e}"),
    }
}
