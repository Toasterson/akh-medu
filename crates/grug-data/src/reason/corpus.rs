//! Procedural example generation: [`ReasonConfig`]/[`ReasonDataset`] build the
//! synthetic KG and generate [`ReasonExample`]s along the plan's three axes
//! (train, axis-B, axis-A) plus the abstention axis, deterministically from a
//! seed.
//!
//! Mirrors `crate::dataset`'s split discipline: every stream uses an
//! independently-salted RNG, and the TRAIN/HOLDOUT **subject** pools are
//! partitioned once at construction (index-based -- "last N are holdout",
//! mirroring `PhrasingBank`'s template split), not re-derived per call.

use std::collections::{HashMap, HashSet};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::GAP_VALUE;
use super::error::ReasonError;
use super::kg::{EntityId, EntityType, Fact, RelationKind, SyntheticKg, query_name, random_path};
use crate::error::GrugResult;

/// Which target format a generated example's `target` field is rendered in
/// -- the P3-R training-stream A/B knob (ADR 092's trace vs the answer-only
/// control).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceArm {
    /// The full High-Grug v1 trace: `step N | recall/derive/check ... | ...`
    /// lines, ending in `call answer | value ...`.
    Trace,
    /// The answer-only control: just the final `call answer | value ...`
    /// line, with no intermediate `step` lines at all.
    AnswerOnly,
}

impl TraceArm {
    /// Parse a short CLI label (`trace` / `answer`), case-insensitive.
    pub fn parse_label(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "trace" => Some(TraceArm::Trace),
            "answer" | "answer_only" | "answer-only" => Some(TraceArm::AnswerOnly),
            _ => None,
        }
    }
}

/// Config for [`ReasonDataset::new`]. `Default` matches the Phase 51 P3-R
/// plan: seed 42, a few hundred entities per type, 2x distractors, train
/// depths `[2, 3, 4]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasonConfig {
    pub seed: u64,
    pub entities_per_type: usize,
    /// Distractor facts generated per *revealed* gold fact (so a depth-`d`
    /// example gets `d * distractor_factor` distractors -- the plan's
    /// "default distractor count = 2x gold" is `distractor_factor == 2`).
    pub distractor_factor: usize,
    pub train_depths: Vec<usize>,
    /// Fraction of each entity type's index range reserved as the **holdout
    /// subject pool** (axis-B / abstention-eval never-trained subjects). The
    /// split is index-based ("last N are holdout"), not RNG-derived.
    pub holdout_subject_frac: f64,
}

impl Default for ReasonConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            entities_per_type: 300,
            distractor_factor: 2,
            train_depths: vec![2, 3, 4],
            holdout_subject_frac: 0.15,
        }
    }
}

/// One procedurally-generated multi-hop reasoning example: prompt text
/// (shuffled facts + a `goal` line) and a High-Grug v1 (or answer-only)
/// target trace, plus the structured ground truth
/// [`super::replay_trace`] checks any model output against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasonExample {
    pub prompt: String,
    pub target: String,
    /// The gold final answer's entity label, or the literal `"gap"` for an
    /// abstention example.
    pub gold_answer: String,
    /// The intended chain length: for an abstention example, this is the
    /// *full* (unrevealed) chain length the `goal` line's composed relation
    /// names, not the number of facts actually revealed in the prompt.
    pub depth: usize,
    pub is_abstention: bool,
    /// The full gold chain of facts from `subject` to the (possibly
    /// unreachable, for an abstention example) answer.
    pub gold_chain: Vec<Fact>,
    pub subject: EntityId,
    pub query_relation: String,
    /// Every fact actually rendered into `prompt` (gold + distractors,
    /// post-shuffle) -- what [`super::replay_trace`] checks `recall`/`check`
    /// statements against.
    pub prompt_facts: Vec<Fact>,
    /// For an abstention example, the 1-indexed hop at which the chain was
    /// broken (`gold_chain[broken_hop - 1]` is deliberately absent from
    /// `prompt_facts`). `None` for a normal (answerable) example.
    pub broken_hop: Option<usize>,
    pub arm: TraceArm,
}

/// The owned synthetic KG plus the train/holdout subject-pool split -- the
/// type callers reach for to generate [`ReasonExample`]s along the plan's
/// axes. See the module docs for the split discipline.
pub struct ReasonDataset {
    config: ReasonConfig,
    kg: SyntheticKg,
    /// Per entity type, the first index (inclusive) reserved for the holdout
    /// subject pool -- indices `< holdout_start[ty]` are TRAIN subjects,
    /// `>= holdout_start[ty]` are HOLDOUT subjects.
    holdout_start: HashMap<EntityType, u32>,
}

const TRAIN_SEED_SALT: u64 = 0x7C3A_1E9F_2B6D_44A1;
const AXIS_B_SEED_SALT: u64 = 0x4F1D_C8B2_9A73_5E60;
const AXIS_A_SEED_SALT: u64 = 0x2E9B_77AC_C1F0_3D48;
const ABSTENTION_SEED_SALT: u64 = 0x9D46_2FA0_1C7B_88E5;

const MIN_ENTITIES_PER_TYPE: usize = 4;

impl ReasonDataset {
    /// Build the KG and the train/holdout subject split. Errors if
    /// `config.train_depths` is empty or contains a zero depth, or if
    /// `config.entities_per_type` is too small to hold a non-empty pool on
    /// both sides of the split.
    pub fn new(config: ReasonConfig) -> GrugResult<Self> {
        if config.train_depths.is_empty() {
            return Err(ReasonError::EmptyTrainDepths.into());
        }
        if config.train_depths.contains(&0) {
            return Err(ReasonError::ZeroDepth.into());
        }
        if config.entities_per_type < MIN_ENTITIES_PER_TYPE {
            return Err(ReasonError::EntitiesPerTypeTooSmall {
                got: config.entities_per_type,
                min: MIN_ENTITIES_PER_TYPE,
            }
            .into());
        }

        let kg = SyntheticKg::generate(config.seed, config.entities_per_type);

        let holdout_count = ((config.entities_per_type as f64 * config.holdout_subject_frac).ceil()
            as u32)
            .clamp(1, config.entities_per_type as u32 - 1);
        let holdout_start_idx = config.entities_per_type as u32 - holdout_count;
        let holdout_start = EntityType::ALL
            .iter()
            .map(|&ty| (ty, holdout_start_idx))
            .collect();

        Ok(Self {
            config,
            kg,
            holdout_start,
        })
    }

    pub fn config(&self) -> &ReasonConfig {
        &self.config
    }

    fn holdout_start_for(&self, ty: EntityType) -> u32 {
        *self
            .holdout_start
            .get(&ty)
            .expect("every EntityType has a holdout_start entry, set for all EntityType::ALL at construction")
    }

    fn train_subject(&self, rng: &mut StdRng, ty: EntityType) -> EntityId {
        let bound = self.holdout_start_for(ty).max(1);
        EntityId {
            ty,
            index: rng.gen_range(0..bound),
        }
    }

    fn holdout_subject(&self, rng: &mut StdRng, ty: EntityType) -> EntityId {
        let start = self.holdout_start_for(ty);
        let total = self.config.entities_per_type as u32;
        let span = total.saturating_sub(start).max(1);
        EntityId {
            ty,
            index: start + rng.gen_range(0..span),
        }
    }

    fn pick_type(rng: &mut StdRng) -> EntityType {
        *EntityType::ALL
            .choose(rng)
            .expect("EntityType::ALL is non-empty")
    }

    /// `n` training examples: TRAIN depths (`config.train_depths`) x TRAIN
    /// subject pool, target rendered under `arm` (the training-stream A/B
    /// knob -- `train_reason`'s `--arm` flag selects this).
    pub fn train_examples(&self, n: usize, arm: TraceArm) -> GrugResult<Vec<ReasonExample>> {
        let mut rng = StdRng::seed_from_u64(self.config.seed ^ TRAIN_SEED_SALT);
        Ok((0..n)
            .map(|_| {
                let depth = *self
                    .config
                    .train_depths
                    .choose(&mut rng)
                    .expect("validated non-empty at construction");
                let ty = Self::pick_type(&mut rng);
                let subject = self.train_subject(&mut rng, ty);
                build_answer_example(
                    &mut rng,
                    &self.kg,
                    subject,
                    depth,
                    self.config.distractor_factor,
                    arm,
                )
            })
            .collect())
    }

    /// [`ReasonDataset::train_examples`] with an abstention mix: each example
    /// is a broken-chain gap example with probability `abstain_frac`
    /// (P3c's lesson, carried over: abstention is an in-distribution
    /// classification the weights can only learn if the training stream
    /// contains it — a stream without gap examples pins abstention
    /// precision/recall at 0 forever). Same seed salt as `train_examples`,
    /// so the two methods produce *different* streams for the same seed —
    /// callers pick one, they are not interleavable views of each other.
    pub fn train_examples_with_abstention(
        &self,
        n: usize,
        arm: TraceArm,
        abstain_frac: f64,
    ) -> GrugResult<Vec<ReasonExample>> {
        let mut rng = StdRng::seed_from_u64(self.config.seed ^ TRAIN_SEED_SALT);
        Ok((0..n)
            .map(|_| {
                let depth = *self
                    .config
                    .train_depths
                    .choose(&mut rng)
                    .expect("validated non-empty at construction");
                let ty = Self::pick_type(&mut rng);
                let subject = self.train_subject(&mut rng, ty);
                if rng.gen_bool(abstain_frac.clamp(0.0, 1.0)) {
                    build_abstention_example(
                        &mut rng,
                        &self.kg,
                        subject,
                        depth,
                        self.config.distractor_factor,
                        arm,
                    )
                } else {
                    build_answer_example(
                        &mut rng,
                        &self.kg,
                        subject,
                        depth,
                        self.config.distractor_factor,
                        arm,
                    )
                }
            })
            .collect())
    }

    /// Axis-B (in-distribution): TRAIN depths, HOLDOUT subject combinations.
    /// Always a full [`TraceArm::Trace`] target -- eval scores whatever the
    /// model emits via [`super::replay_trace`], not by string match against
    /// this field.
    pub fn axis_b_examples(&self, n: usize) -> GrugResult<Vec<ReasonExample>> {
        let mut rng = StdRng::seed_from_u64(self.config.seed ^ AXIS_B_SEED_SALT);
        Ok((0..n)
            .map(|_| {
                let depth = *self
                    .config
                    .train_depths
                    .choose(&mut rng)
                    .expect("validated non-empty at construction");
                let ty = Self::pick_type(&mut rng);
                let subject = self.holdout_subject(&mut rng, ty);
                build_answer_example(
                    &mut rng,
                    &self.kg,
                    subject,
                    depth,
                    self.config.distractor_factor,
                    TraceArm::Trace,
                )
            })
            .collect())
    }

    /// Axis-A (length generalization): caller-supplied `depths` (e.g.
    /// `&[5, 6]` -- never in `config.train_depths`), HOLDOUT subject pool.
    pub fn axis_a_examples(&self, n: usize, depths: &[usize]) -> GrugResult<Vec<ReasonExample>> {
        if depths.is_empty() {
            return Err(ReasonError::EmptyAxisADepths.into());
        }
        if depths.contains(&0) {
            return Err(ReasonError::ZeroDepth.into());
        }
        let mut rng = StdRng::seed_from_u64(self.config.seed ^ AXIS_A_SEED_SALT);
        Ok((0..n)
            .map(|_| {
                let depth = *depths.choose(&mut rng).expect("checked non-empty above");
                let ty = Self::pick_type(&mut rng);
                let subject = self.holdout_subject(&mut rng, ty);
                build_answer_example(
                    &mut rng,
                    &self.kg,
                    subject,
                    depth,
                    self.config.distractor_factor,
                    TraceArm::Trace,
                )
            })
            .collect())
    }

    /// Broken-chain abstention examples (ADR 092 D1's reused `gap`
    /// mechanism): TRAIN depths, TRAIN subject pool, chain deliberately
    /// broken at a random hop. Always a full [`TraceArm::Trace`] target.
    pub fn abstention_examples(&self, n: usize) -> GrugResult<Vec<ReasonExample>> {
        let mut rng = StdRng::seed_from_u64(self.config.seed ^ ABSTENTION_SEED_SALT);
        Ok((0..n)
            .map(|_| {
                let depth = *self
                    .config
                    .train_depths
                    .choose(&mut rng)
                    .expect("validated non-empty at construction");
                let ty = Self::pick_type(&mut rng);
                let subject = self.train_subject(&mut rng, ty);
                build_abstention_example(
                    &mut rng,
                    &self.kg,
                    subject,
                    depth,
                    self.config.distractor_factor,
                    TraceArm::Trace,
                )
            })
            .collect())
    }
}

/// Build a normal (answerable) example: a fresh random path of `depth`
/// relations from `subject`, distractors, a shuffled prompt, and a target
/// rendered under `arm`.
fn build_answer_example(
    rng: &mut StdRng,
    kg: &SyntheticKg,
    subject: EntityId,
    depth: usize,
    distractor_factor: usize,
    arm: TraceArm,
) -> ReasonExample {
    let path = random_path(rng, subject.ty, depth);
    let chain = kg
        .follow_chain(subject, &path)
        .expect("random_path only emits relations valid from the current type");
    let query = query_name(&path);
    let answer = chain
        .last()
        .expect("depth >= 1 (validated at construction), chain non-empty")
        .object;

    let distractors = build_distractors(rng, kg, &chain, distractor_factor, &[]);
    let mut prompt_facts = chain.clone();
    prompt_facts.extend(distractors);
    prompt_facts.shuffle(rng);

    let prompt = render_prompt(&prompt_facts, &query, subject);
    let target = match arm {
        TraceArm::Trace => render_trace_answer(subject, &chain, &query, answer),
        TraceArm::AnswerOnly => render_answer_only(&answer.to_string()),
    };

    ReasonExample {
        prompt,
        target,
        gold_answer: answer.to_string(),
        depth,
        is_abstention: false,
        gold_chain: chain,
        subject,
        query_relation: query,
        prompt_facts,
        broken_hop: None,
        arm,
    }
}

/// Build an abstention example: a fresh full-depth chain, then redact one
/// hop's fact from the prompt so the trace must stop at a `check ... | gap`
/// (or, under [`TraceArm::AnswerOnly`], the bare `call answer | value gap`).
fn build_abstention_example(
    rng: &mut StdRng,
    kg: &SyntheticKg,
    subject: EntityId,
    depth: usize,
    distractor_factor: usize,
    arm: TraceArm,
) -> ReasonExample {
    let path = random_path(rng, subject.ty, depth);
    let full_chain = kg
        .follow_chain(subject, &path)
        .expect("random_path only emits relations valid from the current type");
    // 1-indexed hop at which the chain breaks.
    let k = rng.gen_range(1..=depth);
    let revealed: Vec<Fact> = full_chain[0..k - 1].to_vec();
    let broken = full_chain[k - 1];
    let query = query_name(&path);

    let distractors = build_distractors(
        rng,
        kg,
        &revealed,
        distractor_factor,
        &[(broken.relation, broken.subject)],
    );
    let mut prompt_facts = revealed.clone();
    prompt_facts.extend(distractors);
    prompt_facts.shuffle(rng);

    let prompt = render_prompt(&prompt_facts, &query, subject);
    let target = match arm {
        TraceArm::Trace => {
            render_trace_abstention(subject, &revealed, broken.relation, broken.subject)
        }
        TraceArm::AnswerOnly => render_answer_only(GAP_VALUE),
    };

    ReasonExample {
        prompt,
        target,
        gold_answer: GAP_VALUE.to_string(),
        depth,
        is_abstention: true,
        gold_chain: full_chain,
        subject,
        query_relation: query,
        prompt_facts,
        broken_hop: Some(k),
        arm,
    }
}

/// Generate `factor` distractor facts per fact in `revealed`: same relation,
/// a different subject (never the excluded `(relation, subject)` pairs in
/// `extra_exclude` or already-used pairs), so the prompt contains genuinely
/// confusable-but-wrong facts of the relations that matter, rather than
/// giving the answer away or contradicting an abstention's `check`.
fn build_distractors(
    rng: &mut StdRng,
    kg: &SyntheticKg,
    revealed: &[Fact],
    factor: usize,
    extra_exclude: &[(RelationKind, EntityId)],
) -> Vec<Fact> {
    let mut seen: HashSet<(RelationKind, EntityId)> =
        revealed.iter().map(|f| (f.relation, f.subject)).collect();
    for &pair in extra_exclude {
        seen.insert(pair);
    }

    let mut out = Vec::with_capacity(revealed.len() * factor);
    let entities_per_type = kg.entities_per_type().max(1) as u32;
    for fact in revealed {
        let mut made = 0usize;
        let mut attempts = 0usize;
        let max_attempts = factor.saturating_mul(20).saturating_add(20);
        while made < factor && attempts < max_attempts {
            attempts += 1;
            let idx = rng.gen_range(0..entities_per_type);
            let candidate = EntityId {
                ty: fact.relation.source_type(),
                index: idx,
            };
            if seen.contains(&(fact.relation, candidate)) {
                continue;
            }
            if let Some(distractor) = kg.fact_for(fact.relation, candidate) {
                seen.insert((fact.relation, candidate));
                out.push(distractor);
                made += 1;
            }
        }
    }
    out
}

fn render_prompt(facts: &[Fact], query: &str, subject: EntityId) -> String {
    let mut lines: Vec<String> = facts.iter().map(Fact::to_line).collect();
    lines.push(format!("goal {query} {subject}"));
    lines.join("\n")
}

fn render_trace_answer(subject: EntityId, chain: &[Fact], query: &str, answer: EntityId) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(chain.len() + 2);
    let mut cur = subject;
    for (i, fact) in chain.iter().enumerate() {
        lines.push(format!(
            "step {} | recall {} {} | got {}",
            i + 1,
            fact.relation.word(),
            cur,
            fact.object
        ));
        cur = fact.object;
    }
    let by: Vec<String> = (1..=chain.len()).map(|i| i.to_string()).collect();
    lines.push(format!(
        "step {} | derive {query} {subject} {answer} | by {}",
        chain.len() + 1,
        by.join(" ")
    ));
    lines.push(format!("call answer | value {answer}"));
    lines.join("\n")
}

fn render_trace_abstention(
    subject: EntityId,
    revealed: &[Fact],
    check_relation: RelationKind,
    check_subject: EntityId,
) -> String {
    let mut lines: Vec<String> = Vec::with_capacity(revealed.len() + 2);
    let mut cur = subject;
    for (i, fact) in revealed.iter().enumerate() {
        lines.push(format!(
            "step {} | recall {} {} | got {}",
            i + 1,
            fact.relation.word(),
            cur,
            fact.object
        ));
        cur = fact.object;
    }
    lines.push(format!(
        "step {} | check {} {} | {GAP_VALUE}",
        revealed.len() + 1,
        check_relation.word(),
        check_subject
    ));
    lines.push(format!("call answer | value {GAP_VALUE}"));
    lines.join("\n")
}

fn render_answer_only(value: &str) -> String {
    format!("call answer | value {value}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ReasonConfig {
        ReasonConfig {
            seed: 42,
            entities_per_type: 40,
            distractor_factor: 2,
            train_depths: vec![2, 3, 4],
            holdout_subject_frac: 0.2,
        }
    }

    #[test]
    fn same_seed_reproduces_byte_identical_examples() {
        let a = ReasonDataset::new(tiny_config()).unwrap();
        let b = ReasonDataset::new(tiny_config()).unwrap();
        let ex_a = a.train_examples(30, TraceArm::Trace).unwrap();
        let ex_b = b.train_examples(30, TraceArm::Trace).unwrap();
        assert_eq!(ex_a, ex_b);

        let ab_a = a.axis_b_examples(10).unwrap();
        let ab_b = b.axis_b_examples(10).unwrap();
        assert_eq!(ab_a, ab_b);

        let aa_a = a.axis_a_examples(10, &[5, 6]).unwrap();
        let aa_b = b.axis_a_examples(10, &[5, 6]).unwrap();
        assert_eq!(aa_a, aa_b);

        let ast_a = a.abstention_examples(10).unwrap();
        let ast_b = b.abstention_examples(10).unwrap();
        assert_eq!(ast_a, ast_b);
    }

    #[test]
    fn gold_answer_traces_replay_perfectly() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let mut all = dataset.train_examples(20, TraceArm::Trace).unwrap();
        all.extend(dataset.axis_a_examples(10, &[5, 6]).unwrap());
        for ex in &all {
            let report = super::super::replay_trace(ex, &ex.target);
            assert_eq!(
                report.executability, 1.0,
                "gold trace should replay at 1.0 for {ex:?}"
            );
            assert!(
                report.answer_correct,
                "gold trace should give the correct answer for {ex:?}"
            );
            assert!(report.malformed_lines.is_empty());
        }
    }

    #[test]
    fn abstention_gold_replays_as_abstained() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let examples = dataset.abstention_examples(20).unwrap();
        for ex in &examples {
            assert!(ex.is_abstention);
            assert_eq!(ex.gold_answer, "gap");
            let report = super::super::replay_trace(ex, &ex.target);
            assert!(report.abstained, "expected abstained=true for {ex:?}");
            assert!(report.answer_correct);
            assert_eq!(report.executability, 1.0);
            assert!(report.gap_confirmed);
        }
    }

    #[test]
    fn axis_b_subjects_are_disjoint_from_train_subjects() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let train = dataset.train_examples(200, TraceArm::Trace).unwrap();
        let axis_b = dataset.axis_b_examples(100).unwrap();

        for ex in &train {
            let start = dataset.holdout_start_for(ex.subject.ty);
            assert!(
                ex.subject.index < start,
                "train subject {:?} should be below the holdout start {start}",
                ex.subject
            );
        }
        for ex in &axis_b {
            let start = dataset.holdout_start_for(ex.subject.ty);
            assert!(
                ex.subject.index >= start,
                "axis-B subject {:?} should be at/above the holdout start {start}",
                ex.subject
            );
        }

        let train_subjects: HashSet<EntityId> = train.iter().map(|e| e.subject).collect();
        let axis_b_subjects: HashSet<EntityId> = axis_b.iter().map(|e| e.subject).collect();
        assert!(train_subjects.is_disjoint(&axis_b_subjects));
    }

    #[test]
    fn axis_a_depths_are_never_in_train_depths() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let axis_a_depths = [5usize, 6usize];
        let examples = dataset.axis_a_examples(100, &axis_a_depths).unwrap();
        for ex in &examples {
            assert!(axis_a_depths.contains(&ex.depth));
            assert!(!dataset.config().train_depths.contains(&ex.depth));
        }
    }

    #[test]
    fn answer_only_targets_contain_no_step_lines() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let examples = dataset.train_examples(50, TraceArm::AnswerOnly).unwrap();
        for ex in &examples {
            assert!(
                !ex.target
                    .lines()
                    .any(|l| l.trim_start().starts_with("step")),
                "answer-only target should have no step lines: {:?}",
                ex.target
            );
            assert!(ex.target.starts_with("call answer | value "));
        }
    }

    #[test]
    fn train_with_abstention_mixes_gap_examples_under_both_arms() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        for arm in [TraceArm::Trace, TraceArm::AnswerOnly] {
            let examples = dataset
                .train_examples_with_abstention(400, arm, 0.2)
                .unwrap();
            let n_abstain = examples.iter().filter(|e| e.is_abstention).count();
            assert!(
                (40..=140).contains(&n_abstain),
                "expected ~20% abstention (arm {arm:?}), got {n_abstain}/400"
            );
            for ex in examples.iter().filter(|e| e.is_abstention) {
                assert_eq!(ex.gold_answer, GAP_VALUE);
                assert!(ex.target.contains(GAP_VALUE));
                if arm == TraceArm::AnswerOnly {
                    assert!(
                        !ex.target.contains("step"),
                        "answer-only abstention target must be bare: {:?}",
                        ex.target
                    );
                }
            }
        }
    }

    #[test]
    fn prompt_fact_count_is_gold_plus_distractors() {
        let config = tiny_config();
        let dataset = ReasonDataset::new(config.clone()).unwrap();
        let examples = dataset.train_examples(30, TraceArm::Trace).unwrap();
        for ex in &examples {
            let fact_lines = ex.prompt.lines().filter(|l| l.starts_with("fact ")).count();
            assert_eq!(
                fact_lines,
                ex.depth * (1 + config.distractor_factor),
                "example: {ex:?}"
            );
            assert_eq!(ex.prompt_facts.len(), fact_lines);
            assert!(ex.prompt.lines().last().unwrap().starts_with("goal "));
        }
    }

    #[test]
    fn depth_six_prompt_and_target_stay_well_under_length_budget() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        let examples = dataset.axis_a_examples(20, &[6]).unwrap();
        for ex in &examples {
            let total_len = ex.prompt.len() + ex.target.len();
            assert!(
                total_len < 3000,
                "depth-6 example unexpectedly long ({total_len} chars): {ex:?}"
            );
        }
    }

    #[test]
    fn axis_a_rejects_empty_depths() {
        let dataset = ReasonDataset::new(tiny_config()).unwrap();
        assert!(dataset.axis_a_examples(5, &[]).is_err());
    }

    #[test]
    fn new_rejects_empty_train_depths() {
        let mut config = tiny_config();
        config.train_depths = vec![];
        assert!(ReasonDataset::new(config).is_err());
    }

    #[test]
    fn new_rejects_zero_depth() {
        let mut config = tiny_config();
        config.train_depths = vec![0, 2];
        assert!(ReasonDataset::new(config).is_err());
    }

    #[test]
    fn new_rejects_too_few_entities() {
        let mut config = tiny_config();
        config.entities_per_type = 1;
        assert!(ReasonDataset::new(config).is_err());
    }

    /// Manual sequence-length probe under the plan's real default config
    /// (`ReasonConfig::default()`: entities_per_type=300, distractor_factor=2):
    /// median/max `prompt.len() + target.len()` (char count, a cheap proxy for
    /// token count) at depth 4 (the deepest trained depth) and depth 6 (the
    /// deepest axis-A depth) -- run with `--ignored --nocapture` to see the
    /// numbers reported in the task summary.
    #[test]
    #[ignore = "manual stats probe, not part of the fast test gate"]
    fn stats_probe_reason_lengths() {
        let dataset = ReasonDataset::new(ReasonConfig::default()).unwrap();
        for depth in [4usize, 6usize] {
            let examples = if depth <= 4 {
                dataset
                    .train_examples(500, TraceArm::Trace)
                    .unwrap()
                    .into_iter()
                    .filter(|e| e.depth == depth)
                    .collect()
            } else {
                dataset.axis_a_examples(500, &[depth]).unwrap()
            };
            let mut lens: Vec<usize> = examples
                .iter()
                .map(|e| e.prompt.len() + e.target.len())
                .collect();
            lens.sort_unstable();
            let n = lens.len();
            let median = lens.get(n / 2).copied().unwrap_or(0);
            let max = lens.last().copied().unwrap_or(0);
            let p95 = lens.get(n.saturating_mul(95) / 100).copied().unwrap_or(0);
            eprintln!("depth={depth} n={n} median_chars={median} p95_chars={p95} max_chars={max}");
        }
    }
}
