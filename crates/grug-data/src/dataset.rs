//! The two orthogonal holdout axes as a single owned dataset object.
//!
//! This is the type the training-loop and eval-harness agents should reach
//! for first: [`Dataset::new`] builds the disjoint tool split once, then:
//!
//! - [`Dataset::train_examples`] draws from TRAIN tools × TRAIN phrasings
//!   only.
//! - [`Dataset::holdout_examples`] (axis A, whole-tool) draws from HOLDOUT
//!   tools (never trained) × any phrasing — tests in-context schema-reading
//!   on tools the model has never seen at all.
//! - [`Dataset::holdout_phrasing_examples`] (axis B, held-out-phrasing) draws
//!   from TRAIN tools × HOLDOUT phrasings — tests generalization to unseen
//!   wordings of *known* tools. Because `PhrasingBank` partitions its
//!   templates at construction (see `phrasing.rs`), a holdout-phrasing
//!   template is never reachable from the train stream's `PhrasingMode::Train`
//!   path, regardless of how many examples are drawn.
//!
//! All three streams use independent derived seeds so consuming one stream's
//! random draws leaks no information about another's.

use crate::catalog::{ToolCatalog, DEFAULT_HOLDOUT_COUNT};
use crate::error::GrugResult;
use crate::eval::{DiversityReport, EvalReport};
use crate::generator::{DistractorConfig, Generator, PhrasingMode};
use crate::schema::Example;

/// An owned train/holdout split plus the seed used to derive all three
/// streams.
///
/// Each stream uses a *different* derived seed (`seed`, `seed ^
/// HOLDOUT_SEED_SALT`, `seed ^ HOLDOUT_PHRASING_SEED_SALT`) so that, even
/// though they draw from different tool/phrasing selections, their random
/// draws are independent — a training run that consumes the train stream
/// cannot leak information about either holdout stream's random sequence.
pub struct Dataset {
    seed: u64,
    train_catalog: ToolCatalog,
    holdout_catalog: ToolCatalog,
    distractors: DistractorConfig,
}

/// XOR salt distinguishing the whole-tool holdout RNG stream (axis A) from
/// the train RNG stream for the same base seed. Arbitrary but fixed for
/// reproducibility.
const HOLDOUT_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Convert a whole-tool holdout *fraction* into a tool *count*, rounding up
/// and clamping to `[1, n_tools - 1]` so the split is never degenerate (see
/// `ToolCatalog::split_at`'s `DegenerateSplit` guard) even for small
/// `n_tools` or a `holdout_frac` of `0.0`/`1.0`.
fn holdout_count_from_frac(n_tools: usize, holdout_frac: f64) -> usize {
    let raw = (n_tools as f64 * holdout_frac).ceil() as usize;
    raw.clamp(1, n_tools.saturating_sub(1).max(1))
}

/// XOR salt distinguishing the held-out-phrasing RNG stream (axis B) from
/// both the train stream and the whole-tool holdout stream. Arbitrary but
/// fixed, and distinct from `HOLDOUT_SEED_SALT` so the two holdout axes never
/// share a random sequence even though axis B reuses the *train* tool
/// catalog.
const HOLDOUT_PHRASING_SEED_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;

impl Dataset {
    /// Build the default dataset: the built-in catalog, split with
    /// [`DEFAULT_HOLDOUT_COUNT`] tools reserved for whole-tool holdout, using
    /// the default (hard) distractor config.
    pub fn new(seed: u64) -> GrugResult<Self> {
        Self::with_catalog(seed, ToolCatalog::built_in(), DEFAULT_HOLDOUT_COUNT)
    }

    /// Build a dataset over a caller-supplied catalog and whole-tool holdout
    /// count, using the default (hard) distractor config.
    pub fn with_catalog(seed: u64, catalog: ToolCatalog, holdout_count: usize) -> GrugResult<Self> {
        Self::with_catalog_and_distractors(seed, catalog, holdout_count, DistractorConfig::default())
    }

    /// Build a dataset over a caller-supplied catalog, whole-tool holdout
    /// count, and explicit [`DistractorConfig`] — the full-control
    /// constructor every other `Dataset` builder delegates to.
    pub fn with_catalog_and_distractors(
        seed: u64,
        catalog: ToolCatalog,
        holdout_count: usize,
        distractors: DistractorConfig,
    ) -> GrugResult<Self> {
        let (train_catalog, holdout_catalog) = catalog.split_at(holdout_count)?;
        Ok(Self {
            seed,
            train_catalog,
            holdout_catalog,
            distractors,
        })
    }

    /// Build a dataset over the **procedural** catalog (see
    /// `crate::procedural`): `n_tools` deterministically-generated tools,
    /// reserving `holdout_frac` of them (rounded up, clamped to leave >= 1
    /// train tool) for the whole-tool holdout axis (axis A). This is the
    /// scale-up constructor for Phase 51 P1a-2 — use this instead of
    /// [`Dataset::new`] once the training run wants hundreds-to-thousands of
    /// tools rather than the hand-authored ~56. Uses the default (hard)
    /// distractor config; use [`Dataset::new_sized_with_distractors`] to
    /// override.
    ///
    /// # Example
    /// ```
    /// use grug_data::Dataset;
    /// // 1000 procedural tools, ~15% (150) reserved for whole-tool holdout.
    /// let dataset = Dataset::new_sized(42, 1000, 0.15).unwrap();
    /// assert_eq!(dataset.train_tools().len() + dataset.holdout_tools().len(), 1000);
    /// let examples = dataset.train_examples(100).unwrap();
    /// assert_eq!(examples.len(), 100);
    /// ```
    pub fn new_sized(seed: u64, n_tools: usize, holdout_frac: f64) -> GrugResult<Self> {
        Self::new_sized_with_distractors(seed, n_tools, holdout_frac, DistractorConfig::default())
    }

    /// Like [`Dataset::new_sized`], but with an explicit [`DistractorConfig`]
    /// — the knob the training bin should use to enable hard distractors (the
    /// default) with a chosen `distractor_count`, or fall back to
    /// [`DistractorConfig::easy`] for A/B comparison against the original
    /// P1a-2 numbers.
    ///
    /// # Example
    /// ```
    /// use grug_data::{Dataset, DistractorConfig};
    /// let dataset = Dataset::new_sized_with_distractors(
    ///     42, 1000, 0.15,
    ///     DistractorConfig::hard_with_count(7),
    /// ).unwrap();
    /// let examples = dataset.train_examples(10).unwrap();
    /// // Target + 7 hard distractors offered per example (catalog is large
    /// // enough that the cap never binds here).
    /// assert_eq!(examples[0].tools.len(), 8);
    /// ```
    pub fn new_sized_with_distractors(
        seed: u64,
        n_tools: usize,
        holdout_frac: f64,
        distractors: DistractorConfig,
    ) -> GrugResult<Self> {
        let catalog = ToolCatalog::procedural(seed, n_tools);
        let holdout_count = holdout_count_from_frac(n_tools, holdout_frac);
        Self::with_catalog_and_distractors(seed, catalog, holdout_count, distractors)
    }

    /// Like [`Dataset::new_sized`], but with an explicit
    /// templates-per-tool count instead of
    /// [`crate::procedural::DEFAULT_TEMPLATES_PER_TOOL`]. Uses the default
    /// (hard) distractor config.
    pub fn new_sized_with_templates(
        seed: u64,
        n_tools: usize,
        holdout_frac: f64,
        templates_per_tool: usize,
    ) -> GrugResult<Self> {
        let catalog = ToolCatalog::procedural_with_templates(seed, n_tools, templates_per_tool);
        let holdout_count = holdout_count_from_frac(n_tools, holdout_frac);
        Self::with_catalog_and_distractors(seed, catalog, holdout_count, DistractorConfig::default())
    }

    /// Tool schemas available for training (never includes a whole-tool
    /// holdout tool).
    pub fn train_tools(&self) -> &[crate::schema::ToolSchema] {
        self.train_catalog.tools()
    }

    /// Tool schemas reserved for the whole-tool holdout axis only.
    pub fn holdout_tools(&self) -> &[crate::schema::ToolSchema] {
        self.holdout_catalog.tools()
    }

    /// Generate `n` training examples: TRAIN tools × TRAIN phrasings only.
    pub fn train_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        generator.generate(n)
    }

    /// Streaming form of [`Dataset::train_examples`] for volumes (10k-50k+)
    /// where materializing a `Vec` up front isn't necessary — e.g. writing
    /// straight to a JSONL file. Uses the same derived seed, so consuming
    /// `n` items from this iterator yields exactly `train_examples(n)`.
    pub fn train_examples_iter(&self) -> impl Iterator<Item = GrugResult<Example>> + '_ {
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        std::iter::from_fn(move || Some(generator.try_next_example()))
    }

    /// Generate `n` **whole-tool holdout** (axis A) examples: HOLDOUT tools
    /// (never trained) × any phrasing, using an RNG stream independent of
    /// `train_examples`. Tests in-context schema-reading on tools the model
    /// has never seen. Uses the same [`DistractorConfig`] as the train
    /// stream, so axis-A eval poses the same discrimination task the model
    /// was trained on (just against unseen tools).
    pub fn holdout_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ HOLDOUT_SEED_SALT,
            self.holdout_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        generator.generate(n)
    }

    /// Generate `n` **held-out-phrasing** (axis B) examples: TRAIN tools ×
    /// HOLDOUT phrasings, using an RNG stream independent of both
    /// `train_examples` and `holdout_examples`. Tests generalization to
    /// unseen wordings of tools the model *has* been trained on — the
    /// honest-gate axis: a holdout-phrasing template can never have appeared
    /// in the training stream, because `PhrasingMode::Train` (used by
    /// `train_examples`) only ever draws from `PhrasingBank::train_templates`,
    /// disjoint by construction from `PhrasingBank::holdout_templates`. Uses
    /// the same [`DistractorConfig`] as the train stream.
    pub fn holdout_phrasing_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ HOLDOUT_PHRASING_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Holdout,
            self.distractors,
        );
        generator.generate(n)
    }
}

/// One eval axis's label, for [`AxisReport`] and any caller that wants to
/// tag which stream a report came from (e.g. a training loop's logged time
/// series).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalAxis {
    /// Held-out-phrasing (axis B): TRAIN tools × HOLDOUT phrasings.
    HeldOutPhrasing,
    /// Whole-tool holdout (axis A): HOLDOUT tools × any phrasing.
    WholeToolHoldout,
}

/// A labeled `(EvalReport, DiversityReport)` pair for one eval axis — the
/// convenience type a training loop logs per checkpoint, per axis.
#[derive(Debug, Clone, Copy)]
pub struct AxisReport {
    pub axis: EvalAxis,
    pub eval: EvalReport,
    pub diversity: DiversityReport,
}

impl Dataset {
    /// Score a batch of `(raw_output, example)` pairs and label the result
    /// with which axis it came from — a thin convenience so the training
    /// loop doesn't need to import `EvalReport`/`DiversityReport` separately
    /// per axis; it just calls this once per axis's batch.
    pub fn score_axis(axis: EvalAxis, pairs: &[(String, Example)]) -> AxisReport {
        AxisReport {
            axis,
            eval: EvalReport::from_outputs(pairs),
            diversity: DiversityReport::from_outputs(pairs),
        }
    }

    /// Run both holdout axes end-to-end given a model function: generate `n`
    /// examples per axis, call `model` on each `(user_request, tools)` to get
    /// a raw output string, and return both labeled reports. This is the
    /// single call the Wave-2 training loop needs per eval checkpoint beyond
    /// the training stream itself.
    pub fn eval_both_axes(
        &self,
        n_per_axis: usize,
        mut model: impl FnMut(&Example) -> String,
    ) -> GrugResult<(AxisReport, AxisReport)> {
        let holdout_phrasing = self.holdout_phrasing_examples(n_per_axis)?;
        let phrasing_pairs: Vec<(String, Example)> = holdout_phrasing
            .iter()
            .map(|ex| (model(ex), ex.clone()))
            .collect();

        let whole_tool = self.holdout_examples(n_per_axis)?;
        let whole_tool_pairs: Vec<(String, Example)> = whole_tool
            .iter()
            .map(|ex| (model(ex), ex.clone()))
            .collect();

        Ok((
            Self::score_axis(EvalAxis::HeldOutPhrasing, &phrasing_pairs),
            Self::score_axis(EvalAxis::WholeToolHoldout, &whole_tool_pairs),
        ))
    }
}
