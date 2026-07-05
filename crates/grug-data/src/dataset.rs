//! The whole-tool train/holdout split as a single owned dataset object.
//!
//! This is the type the training-loop and eval-harness agents should reach
//! for first: [`Dataset::new`] builds the disjoint tool split once, then
//! [`Dataset::train_examples`] / [`Dataset::holdout_examples`] each draw from
//! their own tool set — a holdout tool's name/description/params never reach
//! the training stream (see the `holdout_disjointness` test).

use crate::catalog::{ToolCatalog, DEFAULT_HOLDOUT_COUNT};
use crate::error::GrugResult;
use crate::generator::Generator;
use crate::schema::Example;

/// An owned train/holdout split plus the seed used to derive both streams.
///
/// Training and holdout generation use *different* derived seeds (`seed` and
/// `seed ^ HOLDOUT_SEED_SALT`) so that, even though they draw from disjoint
/// tool catalogs, their random draws are independent — a training run that
/// consumes the train stream cannot leak information about the holdout
/// stream's random sequence.
pub struct Dataset {
    seed: u64,
    train_catalog: ToolCatalog,
    holdout_catalog: ToolCatalog,
}

/// XOR salt distinguishing the holdout RNG stream from the train RNG stream
/// for the same base seed. Arbitrary but fixed for reproducibility.
const HOLDOUT_SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

impl Dataset {
    /// Build the default dataset: the built-in catalog, split with
    /// [`DEFAULT_HOLDOUT_COUNT`] tools reserved for holdout.
    pub fn new(seed: u64) -> GrugResult<Self> {
        Self::with_catalog(seed, ToolCatalog::built_in(), DEFAULT_HOLDOUT_COUNT)
    }

    /// Build a dataset over a caller-supplied catalog and holdout count.
    pub fn with_catalog(seed: u64, catalog: ToolCatalog, holdout_count: usize) -> GrugResult<Self> {
        let (train_catalog, holdout_catalog) = catalog.split_at(holdout_count)?;
        Ok(Self {
            seed,
            train_catalog,
            holdout_catalog,
        })
    }

    /// Tool schemas available for training (never includes a holdout tool).
    pub fn train_tools(&self) -> &[crate::schema::ToolSchema] {
        self.train_catalog.tools()
    }

    /// Tool schemas reserved for eval only.
    pub fn holdout_tools(&self) -> &[crate::schema::ToolSchema] {
        self.holdout_catalog.tools()
    }

    /// Generate `n` training examples drawn only from the training tool set.
    pub fn train_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog(self.seed, self.train_catalog.clone());
        generator.generate(n)
    }

    /// Generate `n` holdout examples drawn only from the holdout tool set,
    /// using an RNG stream independent of `train_examples`.
    pub fn holdout_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator =
            Generator::with_catalog(self.seed ^ HOLDOUT_SEED_SALT, self.holdout_catalog.clone());
        generator.generate(n)
    }
}
