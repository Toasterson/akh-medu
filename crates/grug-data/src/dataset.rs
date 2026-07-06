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
use crate::generator::{Generator, PhrasingMode};
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
}

/// XOR salt distinguishing the whole-tool holdout RNG stream (axis A) from
/// the train RNG stream for the same base seed. Arbitrary but fixed for
/// reproducibility.
const HOLDOUT_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// XOR salt distinguishing the held-out-phrasing RNG stream (axis B) from
/// both the train stream and the whole-tool holdout stream. Arbitrary but
/// fixed, and distinct from `HOLDOUT_SEED_SALT` so the two holdout axes never
/// share a random sequence even though axis B reuses the *train* tool
/// catalog.
const HOLDOUT_PHRASING_SEED_SALT: u64 = 0xC2B2_AE3D_27D4_EB4F;

impl Dataset {
    /// Build the default dataset: the built-in catalog, split with
    /// [`DEFAULT_HOLDOUT_COUNT`] tools reserved for whole-tool holdout.
    pub fn new(seed: u64) -> GrugResult<Self> {
        Self::with_catalog(seed, ToolCatalog::built_in(), DEFAULT_HOLDOUT_COUNT)
    }

    /// Build a dataset over a caller-supplied catalog and whole-tool holdout
    /// count.
    pub fn with_catalog(seed: u64, catalog: ToolCatalog, holdout_count: usize) -> GrugResult<Self> {
        let (train_catalog, holdout_catalog) = catalog.split_at(holdout_count)?;
        Ok(Self {
            seed,
            train_catalog,
            holdout_catalog,
        })
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
        let mut generator = Generator::with_catalog(self.seed, self.train_catalog.clone());
        generator.generate(n)
    }

    /// Streaming form of [`Dataset::train_examples`] for volumes (10k-50k+)
    /// where materializing a `Vec` up front isn't necessary — e.g. writing
    /// straight to a JSONL file. Uses the same derived seed, so consuming
    /// `n` items from this iterator yields exactly `train_examples(n)`.
    pub fn train_examples_iter(&self) -> impl Iterator<Item = GrugResult<Example>> + '_ {
        let mut generator = Generator::with_catalog(self.seed, self.train_catalog.clone());
        std::iter::from_fn(move || Some(generator.try_next_example()))
    }

    /// Generate `n` **whole-tool holdout** (axis A) examples: HOLDOUT tools
    /// (never trained) × any phrasing, using an RNG stream independent of
    /// `train_examples`. Tests in-context schema-reading on tools the model
    /// has never seen.
    pub fn holdout_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator =
            Generator::with_catalog(self.seed ^ HOLDOUT_SEED_SALT, self.holdout_catalog.clone());
        generator.generate(n)
    }

    /// Generate `n` **held-out-phrasing** (axis B) examples: TRAIN tools ×
    /// HOLDOUT phrasings, using an RNG stream independent of both
    /// `train_examples` and `holdout_examples`. Tests generalization to
    /// unseen wordings of tools the model *has* been trained on — the
    /// honest-gate axis: a holdout-phrasing template can never have appeared
    /// in the training stream, because `PhrasingMode::Train` (used by
    /// `train_examples`) only ever draws from `PhrasingBank::train_templates`,
    /// disjoint by construction from `PhrasingBank::holdout_templates`.
    pub fn holdout_phrasing_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog_and_mode(
            self.seed ^ HOLDOUT_PHRASING_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Holdout,
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
