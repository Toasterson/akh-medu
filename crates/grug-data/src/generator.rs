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
}

impl Generator {
    /// Build a generator over the built-in catalog, seeded deterministically,
    /// drawing train phrasings.
    pub fn new(seed: u64) -> Self {
        Self::with_catalog(seed, ToolCatalog::built_in())
    }

    /// Build a generator over a caller-supplied catalog (e.g. a train-only
    /// sub-catalog produced by [`ToolCatalog::split_at`]), drawing train
    /// phrasings.
    pub fn with_catalog(seed: u64, catalog: ToolCatalog) -> Self {
        Self::with_catalog_and_mode(seed, catalog, PhrasingMode::Train)
    }

    /// Build a generator over a caller-supplied catalog, drawing wordings
    /// from the given [`PhrasingMode`]. This is the constructor
    /// `Dataset::holdout_phrasing_examples` uses: same (train) tool catalog,
    /// but the *holdout* half of each tool's phrasing bank.
    pub fn with_catalog_and_mode(seed: u64, catalog: ToolCatalog, phrasing_mode: PhrasingMode) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            catalog,
            phrasing_mode,
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

        // Offer a small bundle of tools (the target tool plus a few
        // distractors from the same catalog) rather than just one, so the
        // eval scorer's "tool name is one of the offered tools" check and
        // the model's tool-selection task are both non-trivial.
        let offered = offer_bundle(&mut self.rng, tools, &tool.name);

        Ok(Example {
            tools: offered,
            user_request,
            expected_call,
            mock_response,
        })
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

/// Build the list of tools offered alongside `chosen_name`: the chosen tool
/// plus up to 2 distractors, order-shuffled.
fn offer_bundle(rng: &mut StdRng, tools: &[ToolSchema], chosen_name: &str) -> Vec<ToolSchema> {
    let chosen = tools
        .iter()
        .find(|t| t.name == chosen_name)
        .expect("chosen tool is in this catalog")
        .clone();

    let mut distractors: Vec<ToolSchema> = tools
        .iter()
        .filter(|t| t.name != chosen_name)
        .cloned()
        .collect();
    distractors.shuffle(rng);
    distractors.truncate(2);

    let mut offered = vec![chosen];
    offered.extend(distractors);
    offered.shuffle(rng);
    offered
}

/// Fuzz a single value respecting `ty`.
fn fuzz_value(rng: &mut StdRng, ty: &ParamType) -> Value {
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
