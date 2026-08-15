//! Phase 51 P3-RL A1 -- trace-quality criteria that [`super::replay_trace`]
//! structurally cannot express.
//!
//! Replay answers "is each statement *individually* valid against the prompt
//! facts". That is the right marking scheme (ADR 092 D6) and it is why this
//! corpus needs no LLM judge. But per-statement validity is blind to three
//! whole-trace properties, and each of them is a reward-hacking surface once
//! executability becomes an RL reward (ADR 096, plan §1.2/§1.4):
//!
//! - **Redundancy.** Re-establishing a fact the trace already holds is
//!   *individually valid every time*, so replay scores each repeat `1.0`.
//!   This is the shape of the self-correction looping that SRaR measured at
//!   48.1% of rollouts (arXiv 2605.17291) and cut to 26.5% with step-level
//!   credit. [`TraceQualityReport::redundancy_rate`] makes it a number.
//! - **Drift.** A statement can be perfectly valid and still be about
//!   entities the goal never asked for -- distractor facts are in
//!   `prompt_facts` precisely so the model *can* recall them.
//! - **Padding.** Executability is a *ratio*, so a policy can raise it by
//!   emitting extra valid-but-unnecessary `recall` lines.
//!   [`TraceQualityReport::wasted_valid_steps`] counts exactly that, and
//!   [`TraceQualityReport::hop_coverage`] normalises progress against
//!   `depth` (the required-hop denominator ADR 096 decision 4 mandates)
//!   rather than against emitted-statement count.
//!
//! # Why this is structural and not VSA
//!
//! ADR 096 decision 2 assigned these criteria to the VSA kernel as
//! *geometric* scorers. Building them exposed that framing as wrong **for the
//! lab corpus**, and the plan/ADR carry an amendment saying so. Two reasons:
//!
//! 1. `grug-data` declares "No ML dependency" and the workspace dependency
//!    direction is one-way -- the engine depends on `grug-data`
//!    (`Cargo.toml:176`), never the reverse. Reaching `src/vsa/` from here
//!    would invert that and drag the 113K-LOC engine (oxigraph, redb, hnsw,
//!    candle) into the isolated data crate, breaking the isolation ADR 087 §3
//!    established.
//! 2. More decisively: hypervector similarity is the tool you reach for when
//!    you have no ground truth and need *graded* resemblance. Here every
//!    entity is already a discrete symbol and [`ReasonExample::gold_chain`]
//!    *is* the ground truth. Exact set membership answers all three questions
//!    directly, with no dimension, no seed, and no approximation error. A
//!    hypervector cleanup step would only add a false-match rate to a
//!    comparison that is currently exact.
//!
//! VSA keeps its judging role where it was always the only option: in
//! production (Phase 41 / P6), where there is no gold chain and the loop has
//! nothing but a goal vector and a trace. That is `CompositeVerifier`'s
//! territory, in the engine crate, and nothing here replaces it.
//!
//! # Totality
//!
//! [`score_trace_quality`] never panics, on any input, matching
//! [`super::replay_trace`]'s contract -- it is fed raw model output, which at
//! out-of-depth axes is frequently truncated or malformed.

use std::collections::{HashMap, HashSet};

use super::corpus::ReasonExample;
use super::kg::{EntityId, RelationKind};
use super::replay::{ParsedLine, parse_line};

/// The identity a statement is deduplicated on.
///
/// Deliberately *not* the whole line: a `derive` that reaches the same
/// `(subject, object)` by a different `by`-chain is still the same claim
/// re-established, which is the looping behaviour we are trying to detect.
/// Keying on the rendered text instead would miss it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ClaimKey {
    /// `recall`/`check` both assert something about a `(relation, subject)`
    /// pair; a `check` after a `recall` of the same pair is contradictory as
    /// well as redundant, so they share a key space.
    Lookup(RelationKind, EntityId),
    /// `derive` asserts that `subject` reaches `object`.
    Reach(EntityId, EntityId),
}

/// Whole-trace quality measures, complementing [`super::ReplayReport`].
///
/// Every rate is `0.0` when its denominator is zero, never `NaN` -- these
/// feed reward arithmetic and an accidental `NaN` would silently poison a
/// whole rollout group's advantage.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceQualityReport {
    /// Statements that parsed into a recognised ADR-092 shape. The
    /// denominator for [`Self::redundancy_rate`] and [`Self::drift_rate`].
    pub steps_parsed: usize,
    /// Statements whose [`ClaimKey`] had already been asserted earlier in the
    /// same trace. The first assertion is never counted; only repeats are.
    pub redundant_steps: usize,
    /// `redundant_steps / steps_parsed`.
    pub redundancy_rate: f64,
    /// The largest number of times any single claim was asserted. `1` means
    /// no repetition at all; a large value is the signature of a trace stuck
    /// in a loop rather than one that merely backtracked once.
    pub max_claim_repeats: usize,
    /// Statements referring to at least one entity that appears nowhere in
    /// [`ReasonExample::gold_chain`] -- valid distractor recall, or invention.
    pub off_path_steps: usize,
    /// `off_path_steps / steps_parsed`.
    pub drift_rate: f64,
    /// Distinct gold hops the trace established, by a `recall` matching that
    /// hop's `(relation, subject, object)`. Capped at `depth`.
    pub gold_hops_hit: usize,
    /// `gold_hops_hit / depth` -- progress against *required* work, the
    /// padding-resistant denominator (ADR 096 decision 4). Unlike
    /// executability this cannot be raised by emitting more statements.
    pub hop_coverage: f64,
    /// Statements that replay would credit as valid but which are neither on
    /// the gold path nor a legitimate abstention `check`: the padding
    /// exploit, counted directly.
    pub wasted_valid_steps: usize,
}

impl TraceQualityReport {
    /// The report for a trace with no parsable statements.
    fn empty() -> Self {
        Self {
            steps_parsed: 0,
            redundant_steps: 0,
            redundancy_rate: 0.0,
            max_claim_repeats: 0,
            off_path_steps: 0,
            drift_rate: 0.0,
            gold_hops_hit: 0,
            hop_coverage: 0.0,
            wasted_valid_steps: 0,
        }
    }

    /// Whether the trace shows the looping signature: any claim asserted
    /// three or more times, or more than a quarter of statements redundant.
    ///
    /// A single repeat is ordinary backtracking and is not flagged; the
    /// threshold targets the "repeatedly rewrite and re-verify, often failing
    /// to terminate" behaviour, not any repetition at all.
    pub fn is_looping(&self) -> bool {
        self.max_claim_repeats >= 3 || self.redundancy_rate > 0.25
    }
}

/// `n / d` as an `f64`, or `0.0` when `d == 0`.
fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { n as f64 / d as f64 }
}

/// Score whole-trace quality for `output` against `example`.
///
/// Complements [`super::replay_trace`] rather than replacing it: replay says
/// whether each statement is *true*, this says whether the trace as a whole
/// is *non-redundant, on-goal and unpadded*. Both are needed to turn
/// executability into a reward that cannot be farmed.
///
/// Total: never panics regardless of how malformed `output` is.
pub fn score_trace_quality(example: &ReasonExample, output: &str) -> TraceQualityReport {
    let parsed: Vec<ParsedLine> = output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(parse_line)
        .filter(|p| !matches!(p, ParsedLine::Malformed | ParsedLine::CallAnswer { .. }))
        .collect();

    if parsed.is_empty() {
        return TraceQualityReport::empty();
    }

    // Entities and hops the goal actually requires. `gold_chain` covers all
    // `depth` hops even for an abstention example (where one is deliberately
    // withheld from `prompt_facts`), so it is the right on-path reference.
    let mut on_path: HashSet<EntityId> = HashSet::new();
    on_path.insert(example.subject);
    let mut gold_hops: HashSet<(RelationKind, EntityId, EntityId)> = HashSet::new();
    for fact in &example.gold_chain {
        on_path.insert(fact.subject);
        on_path.insert(fact.object);
        gold_hops.insert((fact.relation, fact.subject, fact.object));
    }

    let mut seen: HashMap<ClaimKey, usize> = HashMap::new();
    let mut redundant_steps = 0usize;
    let mut off_path_steps = 0usize;
    let mut hops_hit: HashSet<(RelationKind, EntityId, EntityId)> = HashSet::new();
    let mut wasted_valid_steps = 0usize;

    for line in &parsed {
        let (key, entities, on_gold) = match *line {
            ParsedLine::Recall {
                relation,
                subject,
                got,
                ..
            } => {
                let hop = (relation, subject, got);
                let is_gold = gold_hops.contains(&hop);
                if is_gold {
                    hops_hit.insert(hop);
                }
                (
                    ClaimKey::Lookup(relation, subject),
                    vec![subject, got],
                    is_gold,
                )
            }
            ParsedLine::Derive {
                subject, object, ..
            } => (
                ClaimKey::Reach(subject, object),
                vec![subject, object],
                on_path.contains(&subject) && on_path.contains(&object),
            ),
            ParsedLine::Check {
                relation, subject, ..
            } => (
                ClaimKey::Lookup(relation, subject),
                vec![subject],
                // A `check` is legitimate work only on an abstention example:
                // it is the mechanism that confirms the broken hop.
                example.is_abstention && on_path.contains(&subject),
            ),
            ParsedLine::CallAnswer { .. } | ParsedLine::Malformed => continue,
        };

        let count = seen.entry(key).or_insert(0);
        *count += 1;
        let is_repeat = *count > 1;
        if is_repeat {
            redundant_steps += 1;
        }

        if !entities.iter().all(|e| on_path.contains(e)) {
            off_path_steps += 1;
        }

        // Padding = a statement that does no required work. A repeat is
        // wasted even when it is on the gold path, because the hop it
        // establishes was already established.
        if !on_gold || is_repeat {
            wasted_valid_steps += 1;
        }
    }

    let steps_parsed = parsed.len();
    let gold_hops_hit = hops_hit.len().min(example.depth);
    let max_claim_repeats = seen.values().copied().max().unwrap_or(0);

    TraceQualityReport {
        steps_parsed,
        redundant_steps,
        redundancy_rate: ratio(redundant_steps, steps_parsed),
        max_claim_repeats,
        off_path_steps,
        drift_rate: ratio(off_path_steps, steps_parsed),
        gold_hops_hit,
        hop_coverage: ratio(gold_hops_hit, example.depth),
        wasted_valid_steps,
    }
}

/// Aggregate [`TraceQualityReport`]s over an evaluation axis.
///
/// Means are over *examples*, not statements, so one pathologically long
/// looping trace cannot dominate the axis average.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceQualitySummary {
    pub examples: usize,
    pub mean_redundancy_rate: f64,
    pub mean_drift_rate: f64,
    pub mean_hop_coverage: f64,
    /// Fraction of examples whose trace tripped
    /// [`TraceQualityReport::is_looping`] -- the number A1 exists to produce,
    /// directly comparable to SRaR's 48.1%.
    pub looping_fraction: f64,
    /// Mean padding per trace: the headroom a ratio-based executability
    /// reward would hand to a policy that learns to farm it.
    pub mean_wasted_valid_steps: f64,
}

/// Score a whole axis. `examples` and `outputs` are zipped; a length mismatch
/// scores only the common prefix rather than panicking.
pub fn summarize_trace_quality(
    examples: &[ReasonExample],
    outputs: &[String],
) -> TraceQualitySummary {
    let reports: Vec<TraceQualityReport> = examples
        .iter()
        .zip(outputs.iter())
        .map(|(e, o)| score_trace_quality(e, o))
        .collect();

    let n = reports.len();
    if n == 0 {
        return TraceQualitySummary {
            examples: 0,
            mean_redundancy_rate: 0.0,
            mean_drift_rate: 0.0,
            mean_hop_coverage: 0.0,
            looping_fraction: 0.0,
            mean_wasted_valid_steps: 0.0,
        };
    }

    let mean = |f: &dyn Fn(&TraceQualityReport) -> f64| -> f64 {
        reports.iter().map(f).sum::<f64>() / n as f64
    };

    TraceQualitySummary {
        examples: n,
        mean_redundancy_rate: mean(&|r| r.redundancy_rate),
        mean_drift_rate: mean(&|r| r.drift_rate),
        mean_hop_coverage: mean(&|r| r.hop_coverage),
        looping_fraction: mean(&|r| if r.is_looping() { 1.0 } else { 0.0 }),
        mean_wasted_valid_steps: mean(&|r| r.wasted_valid_steps as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::corpus::{ReasonConfig, ReasonDataset, TraceArm};

    fn example() -> ReasonExample {
        let ds = ReasonDataset::new(ReasonConfig::default()).expect("dataset");
        ds.train_examples(1, TraceArm::Trace).expect("examples").remove(0)
    }

    /// The gold trace is the definition of clean: nothing repeated, nothing
    /// off-path, every required hop hit, nothing wasted.
    #[test]
    fn gold_trace_is_clean() {
        let ex = example();
        let q = score_trace_quality(&ex, &ex.target);

        assert_eq!(q.redundant_steps, 0, "gold trace repeats nothing");
        assert_eq!(q.redundancy_rate, 0.0);
        assert_eq!(q.off_path_steps, 0, "gold trace stays on the gold chain");
        assert_eq!(q.gold_hops_hit, ex.depth, "gold trace hits every hop");
        assert_eq!(q.hop_coverage, 1.0);
        assert_eq!(q.wasted_valid_steps, 0);
        assert!(!q.is_looping());
    }

    /// Duplicating the gold trace's statements is the looping signature:
    /// replay still scores every line valid, quality does not.
    #[test]
    fn duplicated_statements_are_redundant_not_invalid() {
        let ex = example();
        let steps: Vec<&str> = ex
            .target
            .lines()
            .filter(|l| l.trim_start().starts_with("step "))
            .collect();
        assert!(!steps.is_empty(), "gold trace has step lines");

        // Each step emitted three times over -- the "repeatedly rewrite and
        // re-verify" shape.
        let looped: String = steps
            .iter()
            .flat_map(|s| std::iter::repeat_n(*s, 3))
            .collect::<Vec<_>>()
            .join("\n");

        let q = score_trace_quality(&ex, &looped);

        assert_eq!(q.steps_parsed, steps.len() * 3);
        assert_eq!(q.redundant_steps, steps.len() * 2, "two of every three");
        assert!((q.redundancy_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(q.max_claim_repeats, 3);
        assert!(q.is_looping(), "3x repetition must trip the detector");

        // The point of the whole module: replay is blind to this.
        let replay = super::super::replay_trace(&ex, &looped);
        assert_eq!(
            replay.executability, 1.0,
            "every repeated statement is individually valid -- \
             this is precisely what quality scoring adds"
        );

        // Coverage is unmoved by padding: the same hops, done three times.
        assert_eq!(q.hop_coverage, 1.0);
    }

    /// A single repeat is backtracking, not looping.
    #[test]
    fn one_repeat_is_not_looping() {
        let ex = example();
        let first = ex
            .target
            .lines()
            .find(|l| l.trim_start().starts_with("step "))
            .expect("a step line")
            .to_string();
        let with_repeat = format!("{}\n{}", ex.target.trim_end(), first);

        let q = score_trace_quality(&ex, &with_repeat);
        assert_eq!(q.redundant_steps, 1);
        assert_eq!(q.max_claim_repeats, 2);
        assert!(!q.is_looping(), "one repeat is ordinary backtracking");
    }

    /// Recalling a distractor is valid under replay but is drift, and is
    /// counted as padding -- the executability-farming exploit, measured.
    #[test]
    fn distractor_recall_is_drift_and_padding() {
        let ex = example();
        let gold: std::collections::HashSet<_> = ex
            .gold_chain
            .iter()
            .map(|f| (f.relation, f.subject, f.object))
            .collect();
        let Some(distractor) = ex
            .prompt_facts
            .iter()
            .find(|f| !gold.contains(&(f.relation, f.subject, f.object)))
        else {
            return; // no distractors configured; nothing to assert
        };

        let n = ex.target.lines().filter(|l| l.starts_with("step ")).count();
        // `RelationKind::parse` is the inverse of `word()`, not `code()` --
        // `code()` is only for composed query names.
        let padded = format!(
            "{}\nstep {} | recall {} {} | got {}",
            ex.target.trim_end(),
            n + 1,
            distractor.relation.word(),
            distractor.subject.label(),
            distractor.object.label(),
        );

        let q = score_trace_quality(&ex, &padded);
        assert_eq!(
            q.steps_parsed,
            n + 1,
            "the synthesized distractor line must actually parse"
        );
        assert!(q.wasted_valid_steps >= 1, "distractor recall is padding");
        assert_eq!(q.hop_coverage, 1.0, "padding cannot raise coverage");
        assert_eq!(q.redundant_steps, 0, "a distractor is not a repeat");
    }

    /// Total on adversarial input -- this is fed raw model output.
    #[test]
    fn total_on_garbage() {
        let ex = example();
        for junk in ["", "   ", "\n\n", "not a trace at all", "step | | |", "step x | y"] {
            let q = score_trace_quality(&ex, junk);
            assert!(q.redundancy_rate.is_finite(), "no NaN from {junk:?}");
            assert!(q.drift_rate.is_finite());
            assert!(q.hop_coverage.is_finite());
        }
    }

    /// A length mismatch scores the common prefix instead of panicking.
    #[test]
    fn summary_tolerates_length_mismatch() {
        let ex = example();
        let s = summarize_trace_quality(&[ex.clone(), ex.clone()], &[ex.target.clone()]);
        assert_eq!(s.examples, 1);
        assert!(s.mean_hop_coverage.is_finite());
    }

    #[test]
    fn empty_axis_summary_is_zero_not_nan() {
        let s = summarize_trace_quality(&[], &[]);
        assert_eq!(s.examples, 0);
        assert_eq!(s.looping_fraction, 0.0);
        assert!(s.mean_redundancy_rate.is_finite());
    }
}
