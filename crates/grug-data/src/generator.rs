//! Deterministic, seeded generation of tool-calling training examples.
//!
//! The generator owns ground truth: for every example it produces, the
//! `expected_call` is synthesized *first* (valid by construction against the
//! chosen tool's schema), and the natural-language `user_request` is a
//! template that references those same fuzzed values. There is no parsing or
//! inference involved in labeling — the label is the source of truth.
//!
//! Phrasing selection (train vs. holdout template half) is controlled by
//! [`PhrasingMode`], not by a separate code path — this is what lets
//! `Dataset` reuse the same `Generator` machinery for all three streams
//! (train, whole-tool holdout, held-out-phrasing).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};

use crate::catalog::ToolCatalog;
use crate::error::GrugResult;
use crate::schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};

/// Which half of each tool's phrasing bank a generator draws `user_request`
/// wordings from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhrasingMode {
    /// Only train-phrasing templates (the default for the training stream).
    Train,
    /// Only holdout-phrasing templates (axis B: held-out-phrasing eval).
    Holdout,
}

/// Controls how many distractor tools are offered alongside the target tool,
/// and how they are chosen.
///
/// The point of this knob is forcing in-context schema-reading: if
/// distractors are easy to rule out (unrelated name, unrelated param shape),
/// the model can pick the right tool from weak cues (e.g. a keyword match on
/// the tool name) without ever comparing param names against the schema. Hard
/// mode instead biases distractor selection towards tools that are *similar*
/// to the target — same name family and/or overlapping param shape — so the
/// only way to reliably pick (and correctly fill) the target is to actually
/// read each offered schema.
///
/// See [`DistractorConfig::hard`] (the default, and what P1a-2's follow-up
/// run should use) and [`DistractorConfig::easy`] (the old uniform-random
/// behavior, kept reachable for A/B comparison against the P1a-2 numbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistractorConfig {
    /// Total number of distractor tools offered alongside the target (the
    /// offered bundle is this many, plus 1 for the target itself).
    pub distractor_count: usize,
    /// When true, distractors are drawn from the tools most similar to the
    /// target (name-family + param-shape overlap) rather than uniformly at
    /// random.
    pub hard: bool,
}

impl DistractorConfig {
    /// The default, harder setting: 6 distractors (7 tools offered total),
    /// biased towards same-family/similar-shape tools.
    pub const fn hard() -> Self {
        Self {
            distractor_count: 6,
            hard: true,
        }
    }

    /// The original P1a-2 behavior: 2 uniformly-random distractors (3 tools
    /// offered total). Kept reachable for A/B comparison against the
    /// hard-distractor numbers.
    pub const fn easy() -> Self {
        Self {
            distractor_count: 2,
            hard: false,
        }
    }

    /// A custom distractor count using uniform-random selection.
    pub const fn easy_with_count(distractor_count: usize) -> Self {
        Self {
            distractor_count,
            hard: false,
        }
    }

    /// A custom distractor count using hard (similarity-biased) selection.
    pub const fn hard_with_count(distractor_count: usize) -> Self {
        Self {
            distractor_count,
            hard: true,
        }
    }
}

impl Default for DistractorConfig {
    /// Defaults to the harder setting — see the module-level rationale on
    /// [`DistractorConfig`].
    fn default() -> Self {
        Self::hard()
    }
}

/// A deterministic generator of tool-calling [`Example`]s.
///
/// Construct with [`Generator::new`] and a seed; the same seed always
/// produces the same example stream (see the `determinism` test). Internally
/// this wraps a single [`StdRng`] that is advanced example-by-example, so
/// calling `next_example` repeatedly on one `Generator` gives a reproducible
/// but non-repeating sequence — *not* the same as calling `new` again for
/// each example.
pub struct Generator {
    rng: StdRng,
    catalog: ToolCatalog,
    phrasing_mode: PhrasingMode,
    distractors: DistractorConfig,
    /// Precomputed per-tool similarity fingerprints, parallel to
    /// `catalog.tools()`, built once at construction rather than
    /// recomputed per example. See [`ToolFingerprint`] — this is what keeps
    /// hard-distractor selection affordable at thousands of tools × tens of
    /// thousands of examples (see the `procedural_generation_is_fast_at_moderate_volume`
    /// perf-gate test).
    fingerprints: Vec<ToolFingerprint>,
}

impl Generator {
    /// Build a generator over the built-in catalog, seeded deterministically,
    /// drawing train phrasings, using the default (hard) distractor config.
    pub fn new(seed: u64) -> Self {
        Self::with_catalog(seed, ToolCatalog::built_in())
    }

    /// Build a generator over a caller-supplied catalog (e.g. a train-only
    /// sub-catalog produced by [`ToolCatalog::split_at`]), drawing train
    /// phrasings, using the default (hard) distractor config.
    pub fn with_catalog(seed: u64, catalog: ToolCatalog) -> Self {
        Self::with_catalog_and_mode(seed, catalog, PhrasingMode::Train)
    }

    /// Build a generator over a caller-supplied catalog, drawing wordings
    /// from the given [`PhrasingMode`]. This is the constructor
    /// `Dataset::holdout_phrasing_examples` uses: same (train) tool catalog,
    /// but the *holdout* half of each tool's phrasing bank. Uses the default
    /// (hard) distractor config.
    pub fn with_catalog_and_mode(seed: u64, catalog: ToolCatalog, phrasing_mode: PhrasingMode) -> Self {
        Self::with_catalog_mode_and_distractors(seed, catalog, phrasing_mode, DistractorConfig::default())
    }

    /// Build a generator with full control over phrasing mode and distractor
    /// selection — the constructor `Dataset`'s sized/configurable builders use
    /// so the training bin can enable hard distractors + a chosen count for
    /// both the train stream and both eval axes.
    pub fn with_catalog_mode_and_distractors(
        seed: u64,
        catalog: ToolCatalog,
        phrasing_mode: PhrasingMode,
        distractors: DistractorConfig,
    ) -> Self {
        let fingerprints = catalog.tools().iter().map(ToolFingerprint::of).collect();
        Self {
            rng: StdRng::seed_from_u64(seed),
            catalog,
            phrasing_mode,
            distractors,
            fingerprints,
        }
    }

    /// Generate `n` examples, drawing tools uniformly from this generator's
    /// catalog.
    pub fn generate(&mut self, n: usize) -> GrugResult<Vec<Example>> {
        if self.catalog.is_empty() {
            return Err(crate::error::GrugError::EmptyCatalog);
        }
        (0..n).map(|_| self.try_next_example()).collect()
    }

    /// Generate one example, returning an error if this generator's catalog
    /// has no phrasing bank registered for the drawn tool (should not happen
    /// for catalogs built via `ToolCatalog::built_in` + `split_at`, but keeps
    /// the path total for caller-supplied catalogs).
    pub fn try_next_example(&mut self) -> GrugResult<Example> {
        let tools = self.catalog.tools();
        debug_assert!(!tools.is_empty(), "catalog must be non-empty");
        let tool = tools
            .choose(&mut self.rng)
            .expect("catalog non-empty (checked by caller)")
            .clone();

        let mut args = Map::new();
        for param in &tool.params {
            // Required params always get a value; optional params are
            // included about half the time so the eval scorer sees both
            // "omitted optional" (still valid) and "provided optional".
            let include = param.required || self.rng.gen_bool(0.5);
            if include {
                args.insert(param.name.clone(), fuzz_value(&mut self.rng, &param.ty));
            }
        }
        let arguments = Value::Object(args);
        let expected_call = ToolCall::new(tool.name.clone(), arguments);

        let bank = self.catalog.phrasings().get(&tool.name)?;
        let user_request = match self.phrasing_mode {
            PhrasingMode::Train => bank.render_train(&mut self.rng, &expected_call),
            PhrasingMode::Holdout => bank.render_holdout(&mut self.rng, &expected_call),
        };
        let mock_response = render_mock_response(&expected_call);

        // Offer a bundle of tools (the target tool plus distractors from the
        // same catalog) rather than just one, so the eval scorer's "tool name
        // is one of the offered tools" check and the model's tool-selection
        // task are both non-trivial. See `DistractorConfig` for why the
        // default biases distractor choice towards similar tools instead of
        // uniform-random ones.
        let offered = offer_bundle(
            &mut self.rng,
            tools,
            &self.fingerprints,
            &tool.name,
            &self.distractors,
        );

        Ok(Example {
            tools: offered,
            user_request,
            expected_call,
            mock_response,
        })
    }

    /// Generate one **abstention** example (Phase 51 P3c, ADR 089 mechanism 3):
    /// a request phrased for a target tool `T`, but an offered bundle that
    /// deliberately **excludes `T`** (only distractors), so no offered schema
    /// covers the request. The ground-truth target is [`ToolCall::gap`] — the
    /// model should emit the gap token, not a (necessarily wrong) call.
    ///
    /// The offered set is the same size as a normal example's (`distractor_count
    /// + 1` tools, none of them `T`), so abstention vs. call is *not* trivially
    ///   separable by counting offered tools — the model must actually read the
    ///   schemas and notice none match. Abstention is an in-distribution
    ///   classification the weights can learn (unlike open-world zero-shot), which
    ///   is exactly the claim P3c tests.
    ///
    /// Returns an error only if the drawn tool has no phrasing bank (same
    /// contract as [`Generator::try_next_example`]).
    pub fn try_next_abstention_example(&mut self) -> GrugResult<Example> {
        let tools = self.catalog.tools();
        debug_assert!(!tools.is_empty(), "catalog must be non-empty");
        let tool = tools
            .choose(&mut self.rng)
            .expect("catalog non-empty (checked by caller)")
            .clone();

        // Synthesize the *would-be* call only so the request phrasing can
        // reference concrete values (it reads like a genuine request for `T`);
        // the emitted ground truth is the gap sentinel, not this call.
        let mut args = Map::new();
        for param in &tool.params {
            let include = param.required || self.rng.gen_bool(0.5);
            if include {
                args.insert(param.name.clone(), fuzz_value(&mut self.rng, &param.ty));
            }
        }
        let would_be_call = ToolCall::new(tool.name.clone(), Value::Object(args));

        let bank = self.catalog.phrasings().get(&tool.name)?;
        let user_request = match self.phrasing_mode {
            PhrasingMode::Train => bank.render_train(&mut self.rng, &would_be_call),
            PhrasingMode::Holdout => bank.render_holdout(&mut self.rng, &would_be_call),
        };

        let offered = offer_distractors_only(
            &mut self.rng,
            tools,
            &self.fingerprints,
            &tool.name,
            &self.distractors,
        );
        debug_assert!(
            !offered.iter().any(|t| t.name == tool.name),
            "abstention example must not offer the target tool"
        );

        Ok(Example {
            tools: offered,
            user_request,
            expected_call: ToolCall::gap(),
            mock_response: String::new(),
        })
    }

    /// Generate `n` abstention examples (see [`Generator::try_next_abstention_example`]).
    pub fn generate_abstention(&mut self, n: usize) -> GrugResult<Vec<Example>> {
        if self.catalog.is_empty() {
            return Err(crate::error::GrugError::EmptyCatalog);
        }
        // A single-tool catalog cannot produce a distractors-only bundle.
        if self.catalog.tools().len() < 2 {
            return Err(crate::error::GrugError::EmptyCatalog);
        }
        (0..n).map(|_| self.try_next_abstention_example()).collect()
    }

    /// Generate one example. Panics if the catalog is empty or has no
    /// phrasing bank for the drawn tool — callers going through
    /// [`Generator::generate`] / [`Generator::try_next_example`] get a
    /// `GrugResult` instead; this method is exposed for callers (and
    /// doctests) that already know the built-in catalog is well-formed.
    pub fn next_example(&mut self) -> Example {
        self.try_next_example()
            .expect("built-in catalog has a phrasing bank for every tool")
    }

    /// Iterator form of example generation, for streaming large volumes
    /// (10k-50k+) without materializing a `Vec` up front. Each item is
    /// `GrugResult<Example>` so a missing phrasing bank surfaces per-item
    /// rather than aborting the whole stream construction.
    pub fn iter(&mut self) -> impl Iterator<Item = GrugResult<Example>> + '_ {
        std::iter::from_fn(move || Some(self.try_next_example()))
    }
}

/// Number of distinct [`ParamType`] "kinds" — the fixed-size histogram width
/// [`ToolFingerprint::kind_counts`] uses. Keeping this a small fixed array
/// (rather than a `HashMap`) is what makes [`similarity_score`] allocation-free.
const N_PARAM_KINDS: usize = 5;

/// Map a [`ParamType`] to a small dense index for the fixed-size kind-count
/// histogram. Order is arbitrary but must stay in sync with [`N_PARAM_KINDS`].
fn param_type_kind_index(ty: &ParamType) -> usize {
    match ty {
        ParamType::String => 0,
        ParamType::Integer => 1,
        ParamType::Number => 2,
        ParamType::Boolean => 3,
        ParamType::Enum(_) => 4,
    }
}

/// A precomputed similarity fingerprint for one [`ToolSchema`], built once
/// per catalog (see `Generator::fingerprints`) instead of recomputed for
/// every `(target, candidate)` pair the hard-distractor sort considers.
///
/// This is the perf fix for hard-mode distractor selection: at catalog sizes
/// in the hundreds-to-thousands, `hard_distractor_indices` runs an
/// `O(catalog_size log catalog_size)` sort *per example*, so anything the
/// comparator does must be cheap and allocation-free — no `HashSet`/`HashMap`
/// construction per comparison. Every field here is either a fixed-size
/// array or a small owned `Vec` built once.
#[derive(Debug, Clone)]
struct ToolFingerprint {
    /// The tool's `verb_noun` name split on `_`, sorted so two fingerprints'
    /// token overlap can be counted with a linear merge instead of hashing.
    name_tokens: Vec<String>,
    /// Count of params by [`param_type_kind_index`] — a dense
    /// "how many string/int/number/bool/enum params does this tool have"
    /// histogram.
    kind_counts: [u8; N_PARAM_KINDS],
    /// Param names, sorted, for a linear-merge overlap count (mirrors
    /// `name_tokens`'s approach).
    param_names: Vec<String>,
}

impl ToolFingerprint {
    fn of(tool: &ToolSchema) -> Self {
        let mut name_tokens: Vec<String> = tool.name.split('_').map(str::to_string).collect();
        name_tokens.sort_unstable();

        let mut kind_counts = [0u8; N_PARAM_KINDS];
        for p in &tool.params {
            kind_counts[param_type_kind_index(&p.ty)] =
                kind_counts[param_type_kind_index(&p.ty)].saturating_add(1);
        }

        let mut param_names: Vec<String> = tool.params.iter().map(|p| p.name.clone()).collect();
        param_names.sort_unstable();

        Self {
            name_tokens,
            kind_counts,
            param_names,
        }
    }
}

/// Count of shared elements between two already-sorted slices, via a linear
/// merge (no hashing/allocation) — the building block both
/// [`sorted_jaccard`] uses.
fn sorted_intersection_count<T: Ord>(a: &[T], b: &[T]) -> usize {
    let (mut i, mut j) = (0, 0);
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Jaccard similarity of two already-sorted slices (treated as sets — exact
/// duplicate elements within one slice, which shouldn't occur for either
/// name tokens or param names here, would inflate the union count slightly,
/// which is fine for a ranking heuristic). `0.0` if both empty.
fn sorted_jaccard<T: Ord>(a: &[T], b: &[T]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = sorted_intersection_count(a, b);
    let union = a.len() + b.len() - intersection;
    intersection as f64 / union.max(1) as f64
}

/// Multiset Jaccard over the two fixed-size kind-count histograms: standard
/// `sum(min(a_i, b_i)) / sum(max(a_i, b_i))` over the [`N_PARAM_KINDS`] bins.
fn kind_histogram_jaccard(a: &[u8; N_PARAM_KINDS], b: &[u8; N_PARAM_KINDS]) -> f64 {
    let mut min_sum = 0u32;
    let mut max_sum = 0u32;
    for k in 0..N_PARAM_KINDS {
        min_sum += a[k].min(b[k]) as u32;
        max_sum += a[k].max(b[k]) as u32;
    }
    if max_sum == 0 {
        0.0
    } else {
        min_sum as f64 / max_sum as f64
    }
}

/// How similar is `candidate` to `target` — the ranking key hard-mode
/// distractor selection sorts by (higher = more similar, i.e. a harder
/// distractor). Operates on precomputed [`ToolFingerprint`]s (no allocation),
/// so it is cheap enough to call `O(catalog_size)` times per example.
///
/// Combines two signals, weighted so name-family similarity (a strong,
/// cheap-to-fake surface cue) doesn't by itself dominate a candidate that
/// looks nothing like the target's actual schema:
///
/// - **Name-family overlap**: Jaccard similarity of the `verb_noun` token
///   sets (weight 1.0). Tools like `create_ticket`/`update_ticket` or
///   `fetch_ledger`/`archive_ledger` share a token and read as
///   "the same kind of thing" at a glance.
/// - **Param-shape overlap** (weight 1.0): multiset-Jaccard over each tool's
///   param-*type-kind* histogram — rewards "this tool has a similar mix of
///   string/int/enum/etc. params" — plus a bonus (weight 0.5) for literal
///   param *name* overlap (candidates that reuse an actual param name, e.g.
///   both tools taking a `target_id`, are especially hard to distinguish
///   from the request text alone since a copied value could look plausible
///   under either schema).
///
/// Returns a value in roughly `[0.0, 2.0]`; only relative order matters.
fn similarity_score(target: &ToolFingerprint, candidate: &ToolFingerprint) -> f64 {
    let name_sim = sorted_jaccard(&target.name_tokens, &candidate.name_tokens);
    let shape_sim = kind_histogram_jaccard(&target.kind_counts, &candidate.kind_counts);
    let name_overlap_bonus = sorted_jaccard(&target.param_names, &candidate.param_names);

    name_sim + shape_sim + 0.5 * name_overlap_bonus
}

/// Build the list of tools offered alongside `chosen_name`: the chosen tool
/// plus up to `config.distractor_count` distractors, order-shuffled.
///
/// Two selection strategies, both cheap per example since neither clones or
/// rescans the whole catalog more than once and hard-mode's scoring is
/// allocation-free (see [`ToolFingerprint`]):
///
/// - **Easy** (`config.hard == false`): distractor *indices* sampled
///   uniformly via `rand::seq::index::sample`, matching the original P1a-2
///   behavior. Kept reachable for A/B comparison.
/// - **Hard** (`config.hard == true`, the default): distractors are the tools
///   *most similar* to the target by [`similarity_score`] (name-family +
///   param-shape overlap), so ruling them out requires reading each offered
///   schema rather than pattern-matching on the request text. Ties are
///   broken by shuffling the candidate list before scoring, so which
///   same-score tool wins is still a deterministic function of the RNG
///   stream, not of catalog order.
fn offer_bundle(
    rng: &mut StdRng,
    tools: &[ToolSchema],
    fingerprints: &[ToolFingerprint],
    chosen_name: &str,
    config: &DistractorConfig,
) -> Vec<ToolSchema> {
    let chosen_idx = tools
        .iter()
        .position(|t| t.name == chosen_name)
        .expect("chosen tool is in this catalog");

    let n_distractors = config.distractor_count.min(tools.len().saturating_sub(1));
    let mut offered: Vec<ToolSchema> = if n_distractors == 0 {
        Vec::new()
    } else if config.hard {
        hard_distractor_indices(rng, fingerprints, chosen_idx, n_distractors)
            .into_iter()
            .map(|i| tools[i].clone())
            .collect()
    } else {
        // Sample distinct indices from the catalog excluding `chosen_idx` by
        // sampling from a size-(len-1) space and skipping over it, avoiding
        // an allocation of the full "all other tools" candidate list.
        let sample_space = tools.len() - 1;
        rand::seq::index::sample(rng, sample_space, n_distractors)
            .into_iter()
            .map(|i| if i >= chosen_idx { i + 1 } else { i })
            .map(|i| tools[i].clone())
            .collect()
    };
    offered.push(tools[chosen_idx].clone());
    offered.shuffle(rng);
    offered
}

/// Build a distractors-**only** offered bundle for an abstention example: the
/// `config.distractor_count + 1` tools chosen against `target_name` (so the set
/// is the same size as a normal example's target+distractors bundle), but with
/// `target_name` itself deliberately excluded. Same easy/hard selection as
/// [`offer_bundle`], order-shuffled.
///
/// Used by [`Generator::try_next_abstention_example`] — the request is phrased
/// for `target_name`, the offered set is everything-similar-but-not-it, so the
/// correct output is "no tool fits → gap".
fn offer_distractors_only(
    rng: &mut StdRng,
    tools: &[ToolSchema],
    fingerprints: &[ToolFingerprint],
    target_name: &str,
    config: &DistractorConfig,
) -> Vec<ToolSchema> {
    let target_idx = tools
        .iter()
        .position(|t| t.name == target_name)
        .expect("target tool is in this catalog");

    // Offered-set size matches a normal example (target + distractor_count),
    // but here every member is a distractor, capped at what the catalog allows.
    let n = (config.distractor_count + 1).min(tools.len().saturating_sub(1));
    let mut offered: Vec<ToolSchema> = if n == 0 {
        Vec::new()
    } else if config.hard {
        hard_distractor_indices(rng, fingerprints, target_idx, n)
            .into_iter()
            .map(|i| tools[i].clone())
            .collect()
    } else {
        let sample_space = tools.len() - 1;
        rand::seq::index::sample(rng, sample_space, n)
            .into_iter()
            .map(|i| if i >= target_idx { i + 1 } else { i })
            .map(|i| tools[i].clone())
            .collect()
    };
    offered.shuffle(rng);
    offered
}

/// Pick the `n_distractors` catalog indices (excluding `chosen_idx`) most
/// similar to `fingerprints[chosen_idx]` by [`similarity_score`], operating
/// purely on precomputed fingerprints (no `ToolSchema` cloning or
/// re-derivation).
///
/// Candidate order is shuffled *before* scoring/sorting so that ties (which
/// are common — many procedural tools share the same param-type histogram)
/// are broken by the RNG stream rather than by catalog insertion order; a
/// stable sort then preserves that shuffle for equal scores. This keeps
/// distractor selection a deterministic function of the seed while still
/// being unbiased by tool position in the catalog.
fn hard_distractor_indices(
    rng: &mut StdRng,
    fingerprints: &[ToolFingerprint],
    chosen_idx: usize,
    n_distractors: usize,
) -> Vec<usize> {
    let chosen = &fingerprints[chosen_idx];
    let mut candidates: Vec<usize> = (0..fingerprints.len()).filter(|&i| i != chosen_idx).collect();
    candidates.shuffle(rng);

    let cmp = |&a: &usize, &b: &usize| {
        let score_a = similarity_score(chosen, &fingerprints[a]);
        let score_b = similarity_score(chosen, &fingerprints[b]);
        // Higher similarity first; ties keep the pre-shuffle (RNG-ordered)
        // relative order, not catalog order.
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    // Top-k selection via `select_nth_unstable_by` (introselect, O(n)
    // average) instead of a full `O(n log n)` sort: this function runs once
    // per *generated example*, and `n_distractors` (k) is small (single
    // digits) while the catalog (n) can be in the thousands, so partitioning
    // around the k-th element and only sorting that small prefix is the
    // difference between an affordable per-example cost and a measured
    // multi-second-per-5k-examples blowup at n=2000 (see
    // `hard_distractor_selection_is_fast_at_thousands_of_tools`).
    if n_distractors < candidates.len() {
        candidates.select_nth_unstable_by(n_distractors - 1, cmp);
        candidates.truncate(n_distractors);
    }
    // The top-k prefix from `select_nth_unstable_by` is partitioned but not
    // internally ordered; sort just that (already tiny) prefix so caller
    // behavior (e.g. test assertions on exact picked order) stays
    // deterministic and rank-ordered.
    candidates.sort_by(cmp);
    candidates
}

/// Fuzz a single value respecting `ty`. Crate-visible so `crate::procedural`'s
/// tests can build representative `ToolCall`s using the exact same value
/// distribions the real generator uses, without duplicating this logic.
pub(crate) fn fuzz_value(rng: &mut StdRng, ty: &ParamType) -> Value {
    match ty {
        ParamType::String => Value::String(fuzz_string(rng)),
        ParamType::Integer => Value::from(rng.gen_range(1..=500) as i64),
        ParamType::Number => {
            let whole = rng.gen_range(1..=1000) as f64;
            let frac = (rng.gen_range(0..100) as f64) / 100.0;
            Value::from(whole + frac)
        }
        ParamType::Boolean => Value::Bool(rng.gen_bool(0.5)),
        ParamType::Enum(variants) => Value::String(
            variants
                .choose(rng)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    }
}

/// A small bank of plausible string fillers, picked generically so the
/// generated request text reads naturally, with a generic fallback for
/// anything else. Per-tool phrasing templates (see `catalog_tools.rs`) decide
/// *how* a value is embedded in the sentence; this only decides *what* the
/// fuzzed string value is.
fn fuzz_string(rng: &mut StdRng) -> String {
    const PLACES: &[&str] = &[
        "Zurich", "Tokyo", "Nairobi", "Lima", "Oslo", "Seoul", "Denver", "Cairo",
    ];
    const WORDS: &[&str] = &[
        "quarterly report",
        "the new proposal",
        "grandma's recipe",
        "the budget review",
        "our trip plans",
        "the release notes",
    ];
    if rng.gen_bool(0.5) {
        PLACES.choose(rng).unwrap().to_string()
    } else {
        WORDS.choose(rng).unwrap().to_string()
    }
}

/// Render a plausible mock tool response for `call`. Generic (rather than
/// per-tool-bespoke) now that the catalog is 40-60 tools — a fixed small
/// bank of response shapes keyed by the *type* of the tool's first
/// meaningful field is enough for a plausible "here's what the tool returned"
/// line; this is not scored by the eval harness, only shown to the model as
/// mock multi-turn context.
fn render_mock_response(call: &ToolCall) -> String {
    // Deterministic-but-varied: hash the tool name into a small bank of
    // generic plausible JSON response shapes.
    const RESPONSES: &[&str] = &[
        "{\"status\": \"ok\"}",
        "{\"status\": \"success\", \"id\": \"res-4471\"}",
        "{\"result\": \"done\"}",
        "{\"status\": \"completed\", \"details\": \"request handled\"}",
    ];
    let idx = call
        .name
        .bytes()
        .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
        % RESPONSES.len();
    RESPONSES[idx].to_string()
}

/// Ensure every required param actually got a value (defensive; exercised
/// implicitly by `next_example`, exposed for tests that want to assert
/// well-formedness of a generated call against its schema).
pub fn call_satisfies_required(schema: &ToolSchema, call: &ToolCall) -> bool {
    let args = call.arguments_map();
    schema
        .required_params()
        .all(|p: &ParamSpec| args.contains_key(&p.name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::build_catalog;

    /// A tiny hand-built catalog with an obvious "hard" pairing (two tools
    /// sharing a name token and an identical param shape) and an obvious
    /// "easy" one (nothing in common), so similarity-ranking tests don't
    /// depend on the procedural generator's randomness.
    fn toy_tools() -> Vec<ToolSchema> {
        vec![
            ToolSchema::new(
                "create_ticket",
                "d",
                vec![
                    ParamSpec::new("target_id", ParamType::String, true),
                    ParamSpec::new("priority", ParamType::Enum(vec!["low".into(), "high".into()]), false),
                ],
            ),
            // Shares the "ticket" name token AND the exact param shape with
            // `create_ticket` — the hardest possible distractor.
            ToolSchema::new(
                "update_ticket",
                "d",
                vec![
                    ParamSpec::new("target_id", ParamType::String, true),
                    ParamSpec::new("priority", ParamType::Enum(vec!["low".into(), "high".into()]), false),
                ],
            ),
            // Shares only the param shape (two params, string + enum) but no
            // name token.
            ToolSchema::new(
                "archive_widget",
                "d",
                vec![
                    ParamSpec::new("owner_name", ParamType::String, true),
                    ParamSpec::new("mode_level", ParamType::Enum(vec!["a".into(), "b".into()]), false),
                ],
            ),
            // Shares nothing: different name family, different param shape.
            ToolSchema::new(
                "get_weather",
                "d",
                vec![ParamSpec::new("days", ParamType::Integer, false)],
            ),
        ]
    }

    fn toy_fingerprints() -> Vec<ToolFingerprint> {
        toy_tools().iter().map(ToolFingerprint::of).collect()
    }

    #[test]
    fn hard_distractors_prefer_same_family_similar_shape_tools() {
        let fingerprints = toy_fingerprints();
        let mut rng = StdRng::seed_from_u64(1);
        // chosen = create_ticket (index 0); ask for exactly 1 distractor.
        let picked = hard_distractor_indices(&mut rng, &fingerprints, 0, 1);
        assert_eq!(
            picked,
            vec![1],
            "the same-family, same-shape tool (update_ticket) must be picked over \
             weaker-similarity candidates"
        );
    }

    #[test]
    fn hard_distractors_rank_shape_only_above_unrelated() {
        let fingerprints = toy_fingerprints();
        let mut rng = StdRng::seed_from_u64(2);
        // chosen = create_ticket (index 0); ask for 2 distractors — expect
        // update_ticket (family+shape) then archive_widget (shape only)
        // ahead of get_weather (unrelated).
        let picked = hard_distractor_indices(&mut rng, &fingerprints, 0, 2);
        assert_eq!(picked, vec![1, 2]);
    }

    #[test]
    fn easy_mode_is_not_similarity_biased() {
        // With easy mode and a catalog where the *only* other tool is
        // maximally similar, easy selection still has to include it (only
        // one candidate) — so instead assert easy mode doesn't panic and
        // respects the requested count; the "not biased" property is that
        // `offer_bundle` with `easy()` takes the uniform-random branch at
        // all (covered structurally: this test just exercises that path).
        let tools = toy_tools();
        let fingerprints = toy_fingerprints();
        let mut rng = StdRng::seed_from_u64(3);
        let config = DistractorConfig::easy_with_count(2);
        let offered = offer_bundle(&mut rng, &tools, &fingerprints, "create_ticket", &config);
        assert_eq!(offered.len(), 3);
        assert!(offered.iter().any(|t| t.name == "create_ticket"));
    }

    #[test]
    fn offered_tool_count_matches_config_and_caps_at_catalog_size() {
        let tools = toy_tools(); // 4 tools total
        let fingerprints = toy_fingerprints();
        let mut rng = StdRng::seed_from_u64(4);

        let small = DistractorConfig::hard_with_count(1);
        let offered = offer_bundle(&mut rng, &tools, &fingerprints, "create_ticket", &small);
        assert_eq!(offered.len(), 2, "1 distractor + target");

        let big = DistractorConfig::hard_with_count(100);
        let offered = offer_bundle(&mut rng, &tools, &fingerprints, "create_ticket", &big);
        assert_eq!(
            offered.len(),
            tools.len(),
            "distractor_count is capped at catalog_size - 1"
        );
    }

    #[test]
    fn target_is_always_among_offered_tools() {
        let tools = toy_tools();
        let fingerprints = toy_fingerprints();
        let mut rng = StdRng::seed_from_u64(5);
        for cfg in [
            DistractorConfig::hard_with_count(2),
            DistractorConfig::easy_with_count(2),
            DistractorConfig::hard_with_count(0),
        ] {
            for name in ["create_ticket", "update_ticket", "archive_widget", "get_weather"] {
                let offered = offer_bundle(&mut rng, &tools, &fingerprints, name, &cfg);
                assert!(
                    offered.iter().any(|t| t.name == name),
                    "target {name} missing from offered bundle under {cfg:?}"
                );
            }
        }
    }

    #[test]
    fn distractor_count_is_configurable_end_to_end_via_generator() {
        let catalog = build_catalog(11, 50);
        let mut generator = Generator::with_catalog_mode_and_distractors(
            21,
            catalog,
            PhrasingMode::Train,
            DistractorConfig::hard_with_count(7),
        );
        let examples = generator.generate(20).unwrap();
        for ex in &examples {
            assert_eq!(
                ex.tools.len(),
                8,
                "target + 7 hard distractors expected (catalog of 50 never caps this)"
            );
            assert!(ex.tools.iter().any(|t| t.name == ex.expected_call.name));
        }
    }

    #[test]
    fn abstention_example_excludes_target_and_targets_gap() {
        let catalog = build_catalog(7, 50);
        let mut generator = Generator::with_catalog_mode_and_distractors(
            13,
            catalog,
            PhrasingMode::Train,
            DistractorConfig::easy_with_count(2),
        );
        let examples = generator.generate_abstention(30).unwrap();
        assert_eq!(examples.len(), 30);
        for ex in &examples {
            assert!(ex.is_abstention(), "expected gap target");
            assert!(ex.expected_call.is_gap());
            // Offered set is the same size as a normal easy-2 example (3 tools)
            // and contains none named the gap sentinel...
            assert_eq!(ex.tools.len(), 3, "target + 2 distractors' worth, all distractors");
            assert!(!ex.tools.iter().any(|t| t.name == ToolCall::GAP_TOOL_NAME));
            // ...and, crucially, no offered tool's schema is *the* answer: the
            // request was phrased for a tool that is provably not in the bundle.
            // (We can't recover which tool from the text, but the generator
            // guarantees exclusion — asserted in try_next_abstention_example.)
            assert!(!ex.user_request.is_empty());
        }
    }

    #[test]
    fn hard_distractor_selection_is_deterministic_for_a_fixed_seed() {
        let catalog_a = build_catalog(99, 300);
        let catalog_b = build_catalog(99, 300);
        let mut gen_a = Generator::with_catalog_mode_and_distractors(
            555,
            catalog_a,
            PhrasingMode::Train,
            DistractorConfig::hard_with_count(5),
        );
        let mut gen_b = Generator::with_catalog_mode_and_distractors(
            555,
            catalog_b,
            PhrasingMode::Train,
            DistractorConfig::hard_with_count(5),
        );
        let batch_a = gen_a.generate(30).unwrap();
        let batch_b = gen_b.generate(30).unwrap();
        for (a, b) in batch_a.iter().zip(batch_b.iter()) {
            let names_a: Vec<&str> = a.tools.iter().map(|t| t.name.as_str()).collect();
            let names_b: Vec<&str> = b.tools.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(names_a, names_b, "same seed must give byte-identical offered bundles");
        }
    }

    #[test]
    fn similarity_score_is_symmetric_and_self_maximal() {
        let fingerprints = toy_fingerprints();
        for a in &fingerprints {
            for b in &fingerprints {
                assert!(
                    (similarity_score(a, b) - similarity_score(b, a)).abs() < 1e-9,
                    "similarity must be symmetric"
                );
            }
        }
        // A tool is at least as similar to itself as to anything else.
        for a in &fingerprints {
            let self_score = similarity_score(a, a);
            for b in &fingerprints {
                assert!(self_score >= similarity_score(a, b) - 1e-9);
            }
        }
    }

    #[test]
    fn hard_distractor_selection_completes_at_thousands_of_tools() {
        // Sanity check (not a tight perf guard -- see the `--release`-only
        // `hard_distractor_default_scales_to_thousands_of_tools_in_release`
        // integration test for that) that hard-mode distractor selection
        // over a 2000-tool catalog actually completes and produces
        // well-formed bundles, in an unoptimized debug build. A
        // pre-fingerprint version of this scorer allocated a
        // HashSet/HashMap per (target, candidate) comparison inside a full
        // `O(n log n)` sort, which made this configuration take *minutes* in
        // debug builds (empirically discovered running this crate's test
        // suite) -- the fingerprint + `select_nth_unstable_by` rewrite
        // brought debug-build cost down to low seconds, but debug-mode
        // timing is still too noisy/slow (~5x release) to gate tightly here.
        let catalog = build_catalog(17, 2000);
        let mut generator = Generator::with_catalog_mode_and_distractors(
            31,
            catalog,
            PhrasingMode::Train,
            DistractorConfig::hard_with_count(6),
        );
        let start = std::time::Instant::now();
        let examples = generator.generate(1_000).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(examples.len(), 1_000);
        for ex in &examples {
            assert_eq!(ex.tools.len(), 7);
        }
        assert!(
            elapsed.as_secs() < 20,
            "1k examples over a 2000-tool catalog with hard distractors took {elapsed:?} in a \
             debug build -- investigate if this creeps far past a few seconds"
        );
    }
}
