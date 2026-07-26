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
    /// Which retrieval-signal tier this dataset's catalog was generated under
    /// (Phase 51 P3d). Determines what the registry retriever keys on:
    /// `Name` → the tool name (byte-identical to the P3a behavior);
    /// `Description`/`Paraphrase` → the tool **description**, because under
    /// those tiers the name is absent from requests and a name-keyed retriever
    /// would collapse (recall 0.002, P3d-1) and mislabel nearly every
    /// retrieval-scoped example as gap.
    signal_mode: crate::semantic::SignalMode,
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

/// XOR salt for the **abstention** RNG streams (Phase 51 P3c). Distinct from
/// every other salt so the gap examples' random draws are independent of the
/// call streams — a training run that mixes abstention in cannot leak the call
/// or holdout sequences, and the abstention holdout is reproducible on its own.
const ABSTENTION_SEED_SALT: u64 = 0x1D8E_3A0B_9C57_F26D;

/// A second salt distinguishing the abstention **holdout** stream (recall eval)
/// from the abstention **train-mix** stream, so the model is never evaluated on
/// an abstention example it could have trained on.
const ABSTENTION_HOLDOUT_SEED_SALT: u64 = 0x84A3_51CE_7B02_9F6E;

/// XOR salts for the **retrieval-scoped** streams (Phase 51 P3a, ADR 089
/// mechanism 1). Each is distinct from every other salt so retrieval train /
/// holdout / genuine-novelty / recall-probe draws are mutually independent.
const RETRIEVAL_TRAIN_SEED_SALT: u64 = 0x2545_F491_1A3B_C7D9;
const RETRIEVAL_HOLDOUT_SEED_SALT: u64 = 0xB7E1_5162_8AED_2A6B;
const RETRIEVAL_ABSTAIN_SEED_SALT: u64 = 0x6C62_72E1_5AF5_33A9;
const RETRIEVAL_RECALL_SEED_SALT: u64 = 0x3F84_D0C3_A190_7E5D;

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
            signal_mode: crate::semantic::SignalMode::Name,
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

    /// Like [`Dataset::new_sized_with_distractors`], but under an explicit
    /// [`SignalMode`](crate::semantic::SignalMode) (Phase 51 P3d). `Name`
    /// reproduces the current corpus; `Description`/`Paraphrase` build the
    /// glossed (name-free) corpus that P3d re-measures the stack on.
    pub fn new_sized_signal(
        seed: u64,
        n_tools: usize,
        holdout_frac: f64,
        distractors: DistractorConfig,
        mode: crate::semantic::SignalMode,
    ) -> GrugResult<Self> {
        let catalog = ToolCatalog::procedural_signal(seed, n_tools, mode);
        let holdout_count = holdout_count_from_frac(n_tools, holdout_frac);
        let mut ds = Self::with_catalog_and_distractors(seed, catalog, holdout_count, distractors)?;
        ds.signal_mode = mode;
        Ok(ds)
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

    /// Generate `n` training examples mixing concrete calls with **abstention**
    /// examples (Phase 51 P3c, ADR 089 mechanism 3): a fraction `abstain_frac`
    /// (clamped to `[0, 1]`) of the returned examples are gap-targets whose
    /// offered bundle excludes the tool the request needs. All draw from TRAIN
    /// tools × TRAIN phrasings (in-distribution), from RNG streams independent of
    /// every eval axis. Abstention examples are interleaved evenly
    /// (Bresenham-style) so every training batch sees a representative mix rather
    /// than a block of one kind.
    ///
    /// `abstain_frac == 0.0` reproduces [`Dataset::train_examples`] exactly (the
    /// call generator is advanced identically); `1.0` yields all-abstention.
    pub fn train_examples_with_abstention(&self, n: usize, abstain_frac: f64) -> GrugResult<Vec<Example>> {
        let frac = abstain_frac.clamp(0.0, 1.0);
        let mut call_gen = Generator::with_catalog_mode_and_distractors(
            self.seed,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        let mut abstain_gen = Generator::with_catalog_mode_and_distractors(
            self.seed ^ ABSTENTION_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        let mut out = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for _ in 0..n {
            acc += frac;
            if acc >= 1.0 {
                acc -= 1.0;
                out.push(abstain_gen.try_next_abstention_example()?);
            } else {
                out.push(call_gen.try_next_example()?);
            }
        }
        Ok(out)
    }

    /// Generate `n` **abstention holdout** examples (Phase 51 P3c): TRAIN tools ×
    /// HOLDOUT phrasings, offered bundle excludes the needed tool, gap target —
    /// the recall half of the abstention eval. Mirrors axis B (in-distribution
    /// tools, unseen wordings) so it measures whether the abstention *skill*
    /// generalizes, not whether a request string was memorized. Uses an RNG
    /// stream independent of the abstention train-mix and of both call axes.
    pub fn abstention_examples(&self, n: usize) -> GrugResult<Vec<Example>> {
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ ABSTENTION_HOLDOUT_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Holdout,
            self.distractors,
        );
        generator.generate_abstention(n)
    }

    /// The **registry**: the tools the model trains on and the retriever
    /// (mechanism 1) selects from — an alias for [`Dataset::train_tools`] under
    /// the retrieval-scoped framing, where the whole-tool holdout is repurposed
    /// as the *out-of-registry* (genuine-novelty) source rather than a target.
    pub fn registry(&self) -> &[crate::schema::ToolSchema] {
        self.train_catalog.tools()
    }

    /// Build a [`Retriever`](crate::retriever::Retriever) over this dataset's
    /// registry (mechanism 1). Indices it returns index [`Dataset::registry`].
    ///
    /// **Mode-aware (Phase 51 P3d):** under [`SignalMode::Name`] it keys on the
    /// tool *name* — byte-identical to the P3a behavior. Under
    /// `Description`/`Paraphrase` it keys on the tool **description**, because
    /// those corpora carry no name tokens in the request: a name-keyed
    /// retriever would collapse (recall 0.002, P3d-1) and mislabel nearly
    /// every retrieval-scoped example as gap. This mirrors the real organ,
    /// where the router matches the request against capability *descriptions*,
    /// not identifier strings.
    pub fn registry_retriever(&self) -> crate::retriever::Retriever {
        match self.signal_mode {
            crate::semantic::SignalMode::Name => crate::retriever::Retriever::new(self.registry()),
            crate::semantic::SignalMode::Description | crate::semantic::SignalMode::Paraphrase => {
                let descriptions: Vec<&str> =
                    self.registry().iter().map(|t| t.description.as_str()).collect();
                crate::retriever::Retriever::new_over(&descriptions)
            }
        }
    }

    /// Re-scope a base example to the retrieval-selected offered set (ADR 089
    /// mechanism 1): replace its offered tools with the retriever's top-`k` for
    /// the request, then **relabel** — if retrieval surfaced the target tool the
    /// label is the concrete call; if it did **not** (a retrieval miss, or a
    /// genuine out-of-registry request), the correct label becomes the gap
    /// token, because a tool that was never offered cannot be called. This is
    /// the training signal that couples mechanism 1 to mechanism 3: "call when
    /// the offered set covers the request, abstain when it doesn't."
    fn scope_to_retrieval(
        base: Example,
        retriever: &crate::retriever::Retriever,
        registry: &[crate::schema::ToolSchema],
        k: usize,
    ) -> Example {
        let offered: Vec<crate::schema::ToolSchema> = retriever
            .top_k(&base.user_request, k)
            .into_iter()
            .map(|i| registry[i].clone())
            .collect();
        let target_offered = offered.iter().any(|t| t.name == base.expected_call.name);
        let expected_call = if target_offered {
            base.expected_call
        } else {
            crate::schema::ToolCall::gap()
        };
        Example {
            tools: offered,
            expected_call,
            user_request: base.user_request,
            mock_response: base.mock_response,
        }
    }

    /// Generate `n` **retrieval-scoped training** examples (ADR 089 mechanism 1):
    /// a request for a registry tool, offered the retriever's top-`k` over the
    /// whole registry (TRAIN tools × TRAIN phrasings), labelled call-or-gap by
    /// [`Dataset::scope_to_retrieval`]. Most are calls (retrieval surfaces the
    /// target); retrieval misses become in-registry gaps. This is the realistic
    /// runtime offered set — request-keyed confusables, not random distractors.
    pub fn retrieval_train_examples(&self, k: usize, n: usize) -> GrugResult<Vec<Example>> {
        let retriever = self.registry_retriever();
        let registry = self.registry().to_vec();
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_TRAIN_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        (0..n)
            .map(|_| {
                generator
                    .try_next_example()
                    .map(|base| Self::scope_to_retrieval(base, &retriever, &registry, k))
            })
            .collect()
    }

    /// Generate `n` **retrieval-scoped holdout** examples (mechanism 1): same as
    /// [`Dataset::retrieval_train_examples`] but TRAIN tools × HOLDOUT phrasings
    /// and an independent RNG stream — the in-registry call/gap eval set (unseen
    /// wordings of registry tools).
    pub fn retrieval_holdout_examples(&self, k: usize, n: usize) -> GrugResult<Vec<Example>> {
        let retriever = self.registry_retriever();
        let registry = self.registry().to_vec();
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_HOLDOUT_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Holdout,
            self.distractors,
        );
        (0..n)
            .map(|_| {
                generator
                    .try_next_example()
                    .map(|base| Self::scope_to_retrieval(base, &retriever, &registry, k))
            })
            .collect()
    }

    /// Generate `n` **genuine-novelty abstention** examples under retrieval
    /// (mechanism 1 ∘ mechanism 3): a request for an **out-of-registry** tool
    /// (the whole-tool HOLDOUT catalog — never trained, never a valid target),
    /// offered the retriever's top-`k` over the **registry**. Since the true
    /// tool is not in the registry it can never be offered, so the label is
    /// always gap — but the offered set is the closest in-registry *confusables*
    /// (they share verb/noun tokens), making this a much harder abstention test
    /// than random distractors: the model must notice that none of the plausibly
    /// related tools actually match.
    pub fn retrieval_abstention_examples(&self, k: usize, n: usize) -> GrugResult<Vec<Example>> {
        if self.registry().len() < 2 {
            return Err(crate::error::GrugError::EmptyCatalog);
        }
        let retriever = self.registry_retriever();
        let registry = self.registry().to_vec();
        // Requests are drawn from the OUT-OF-REGISTRY (holdout) catalog.
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_ABSTAIN_SEED_SALT,
            self.holdout_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        (0..n)
            .map(|_| {
                generator.try_next_example().map(|base| {
                    let scoped = Self::scope_to_retrieval(base, &retriever, &registry, k);
                    debug_assert!(scoped.is_abstention(), "out-of-registry request must label as gap");
                    scoped
                })
            })
            .collect()
    }

    /// Generate `n` **retrieval-scoped training** examples mixing in-registry
    /// requests (call, or in-registry gap on a retrieval miss) with a fraction
    /// `novelty_frac` of **genuine-novelty** out-of-registry abstention examples
    /// (ADR 089 mechanism 1 ∘ 3). Interleaved Bresenham-style so every batch
    /// sees both. `novelty_frac == 0.0` is pure [`Dataset::retrieval_train_examples`].
    ///
    /// This is the corpus the retrieval-scoped model trains on: it learns to
    /// call when the retrieved offered set covers the request, and to gap both
    /// when retrieval missed an in-registry tool *and* when the request needs a
    /// capability the registry does not contain.
    pub fn retrieval_train_mix(&self, k: usize, n: usize, novelty_frac: f64) -> GrugResult<Vec<Example>> {
        let frac = novelty_frac.clamp(0.0, 1.0);
        let retriever = self.registry_retriever();
        let registry = self.registry().to_vec();
        let mut call_gen = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_TRAIN_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        let mut novelty_gen = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_ABSTAIN_SEED_SALT,
            self.holdout_catalog.clone(),
            PhrasingMode::Train,
            self.distractors,
        );
        let mut out = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for _ in 0..n {
            acc += frac;
            let base = if acc >= 1.0 {
                acc -= 1.0;
                novelty_gen.try_next_example()?
            } else {
                call_gen.try_next_example()?
            };
            out.push(Self::scope_to_retrieval(base, &retriever, &registry, k));
        }
        Ok(out)
    }

    /// `n` `(request, registry_target_index)` pairs for the model-independent
    /// **retrieval recall@k** probe (mechanism 1's premise check): registry
    /// tools × HOLDOUT phrasings, each paired with the target's index into
    /// [`Dataset::registry`]. Feed to
    /// [`retrieval_recall_at_k`](crate::retriever::retrieval_recall_at_k).
    pub fn retrieval_recall_pairs(&self, n: usize) -> GrugResult<Vec<(String, usize)>> {
        let registry = self.registry();
        let mut generator = Generator::with_catalog_mode_and_distractors(
            self.seed ^ RETRIEVAL_RECALL_SEED_SALT,
            self.train_catalog.clone(),
            PhrasingMode::Holdout,
            self.distractors,
        );
        (0..n)
            .map(|_| {
                generator.try_next_example().map(|base| {
                    let idx = registry
                        .iter()
                        .position(|t| t.name == base.expected_call.name)
                        .expect("target is a registry tool");
                    (base.user_request, idx)
                })
            })
            .collect()
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
        Self::score_axis_fmt(axis, pairs, crate::format::TargetFormat::Json)
    }

    /// Score one axis under an explicit [`TargetFormat`] (Phase 51 P2).
    pub fn score_axis_fmt(
        axis: EvalAxis,
        pairs: &[(String, Example)],
        fmt: crate::format::TargetFormat,
    ) -> AxisReport {
        AxisReport {
            axis,
            eval: EvalReport::from_outputs_fmt(pairs, fmt),
            diversity: DiversityReport::from_outputs_fmt(pairs, fmt),
        }
    }

    /// Score one axis with the KG-backed repair pass applied (Phase 51 P3b):
    /// the raw-vs-repaired lift is [`Dataset::score_axis_fmt`] vs this.
    pub fn score_axis_repaired_fmt(
        axis: EvalAxis,
        pairs: &[(String, Example)],
        fmt: crate::format::TargetFormat,
    ) -> AxisReport {
        AxisReport {
            axis,
            eval: EvalReport::from_outputs_repaired_fmt(pairs, fmt),
            diversity: DiversityReport::from_outputs_fmt(pairs, fmt),
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

#[cfg(test)]
mod retrieval_tests {
    use super::*;
    use crate::retriever::{retrieval_recall_at_k, Retriever};

    /// Retrieval-scoped training examples: offered set is ≤ k tools drawn from
    /// the registry, and the common case is a call whose target is offered.
    #[test]
    fn retrieval_train_examples_offer_top_k_and_mostly_call() {
        let dataset = Dataset::new_sized(42, 400, 0.15).unwrap();
        let k = 6;
        let examples = dataset.retrieval_train_examples(k, 200).unwrap();
        assert_eq!(examples.len(), 200);
        let registry_names: std::collections::HashSet<&str> =
            dataset.registry().iter().map(|t| t.name.as_str()).collect();
        let mut calls = 0;
        for ex in &examples {
            assert!(ex.tools.len() <= k, "offered set must be ≤ k");
            // Every offered tool is from the registry.
            for t in &ex.tools {
                assert!(registry_names.contains(t.name.as_str()));
            }
            if !ex.is_abstention() {
                calls += 1;
                // A call's target is necessarily among the offered tools.
                assert!(ex.tools.iter().any(|t| t.name == ex.expected_call.name));
            }
        }
        // Retrieval is good on this data, so the vast majority are calls.
        assert!(calls as f64 / examples.len() as f64 > 0.8, "expected mostly calls, got {calls}/200");
    }

    /// P3d-3: under D1 (description signal, name absent from requests) the
    /// mode-aware registry retriever keys on descriptions, so the retrieval
    /// streams stay call-dominant — the property the D1 retrain depends on.
    /// (A name-keyed retriever here would collapse to ~0 recall and label
    /// nearly everything gap; that this stays >0.8 calls is the wiring proof.)
    #[test]
    fn d1_retrieval_streams_stay_call_dominant_via_description_keyed_retriever() {
        let dataset = Dataset::new_sized_signal(
            42,
            400,
            0.15,
            DistractorConfig::easy_with_count(2),
            crate::semantic::SignalMode::Description,
        )
        .unwrap();
        let k = 8;
        // Model-independent premise first: recall@k over the D1 registry.
        let retriever = dataset.registry_retriever();
        let pairs = dataset.retrieval_recall_pairs(150).unwrap();
        let recall = retrieval_recall_at_k(&retriever, &pairs, k);
        assert!(recall > 0.9, "D1 description-keyed recall@{k} should be high, got {recall}");

        let examples = dataset.retrieval_train_examples(k, 200).unwrap();
        let calls = examples.iter().filter(|e| !e.is_abstention()).count();
        assert!(
            calls as f64 / examples.len() as f64 > 0.8,
            "D1 retrieval train stream should stay call-dominant, got {calls}/200"
        );
        // And the D1 *phrasing* genuinely carries no name tokens (spot-check).
        // Tokens that arrive via argument VALUES are exempt — a fuzzed string
        // value like "the release notes" legitimately contains "release" for a
        // `publish_release` tool; that is content, not a phrasing leak.
        for ex in examples.iter().take(20).filter(|e| !e.is_abstention()) {
            let req = ex.user_request.to_ascii_lowercase();
            let value_text = ex
                .expected_call
                .arguments_map()
                .values()
                .map(|v| v.to_string().to_ascii_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            for tok in ex.expected_call.name.split('_') {
                let tok = tok.to_ascii_lowercase();
                let in_values = value_text
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|w| w == tok);
                if in_values {
                    continue;
                }
                let leaked = req
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .any(|w| w == tok);
                assert!(!leaked, "D1 phrasing leaked name token {tok:?}: {:?}", ex.user_request);
            }
        }
    }

    /// Genuine-novelty abstention: an out-of-registry request, offered the
    /// registry's closest confusables, always labels as gap and never offers
    /// the true tool.
    #[test]
    fn retrieval_abstention_always_gaps_and_never_offers_target() {
        let dataset = Dataset::new_sized(7, 400, 0.15).unwrap();
        let out_of_reg: std::collections::HashSet<&str> =
            dataset.holdout_tools().iter().map(|t| t.name.as_str()).collect();
        let examples = dataset.retrieval_abstention_examples(6, 100).unwrap();
        assert_eq!(examples.len(), 100);
        for ex in &examples {
            assert!(ex.is_abstention(), "out-of-registry request must gap");
            // None of the offered tools is an out-of-registry tool.
            assert!(!ex.tools.iter().any(|t| out_of_reg.contains(t.name.as_str())));
        }
    }

    /// The mechanism-1 premise: retrieval recall@k over the registry is high, so
    /// the offered set almost always contains the target.
    #[test]
    fn retrieval_recall_is_high_over_registry() {
        let dataset = Dataset::new_sized(42, 1000, 0.15).unwrap();
        let retriever = Retriever::new(dataset.registry());
        let pairs = dataset.retrieval_recall_pairs(256).unwrap();
        let recall = retrieval_recall_at_k(&retriever, &pairs, 8);
        assert!(recall > 0.9, "recall@8 over the registry should be high, got {recall}");
    }
}
