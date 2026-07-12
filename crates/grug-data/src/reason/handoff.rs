//! Phase 51 P6 -- the dual-process reasoning handoff (first slice).
//!
//! A trained System-1 "organ" emits ADR-092 traces and reasons near-perfectly
//! at trained depths, but cannot *render* traces beyond them (whole-trace
//! validity collapses to ~0 past the trained depth, even though a large
//! fraction of individual early steps still replay correctly). [`handoff`]
//! is the System-2 stand-in: it finds the longest prefix of the organ's
//! output that is *actually correct* (validates against the example's own
//! prompt facts, hop by hop, in the query's relation order -- not just
//! "replays some valid statements somewhere"), then deterministically
//! completes the remaining hops by looking the missing facts up directly in
//! [`ReasonExample::prompt_facts`], synthesizing canonical ADR-092 lines for
//! whatever the organ didn't (or couldn't) establish.
//!
//! # Recovering the relation sequence
//!
//! The plan asked whether [`ReasonExample::query_relation`] (the composed
//! `query_name` string, e.g. `"q_lvpf"`) needs to be inverted back into a
//! `Vec<RelationKind>`, or whether a new field should carry that sequence
//! explicitly. Neither is necessary: [`ReasonExample::gold_chain`] already
//! *is* the concrete realization of that path -- `gold_chain[k].relation` is
//! hop `k+1`'s relation, and `gold_chain[k].object` is the entity reached
//! after that hop, for `k` in `0..depth`. This holds even for an abstention
//! example: [`ReasonExample::gold_chain`] is documented as "the full gold
//! chain of facts from subject to the (possibly unreachable) answer", i.e.
//! it covers all `depth` hops regardless of which ones were actually
//! revealed into [`ReasonExample::prompt_facts`]. So `handoff` reads the
//! per-hop relation and expected entity straight off `gold_chain`, and never
//! parses `query_relation` -- it's only used verbatim (as the query label
//! text) when re-rendering a synthesized `derive` line, exactly as
//! `corpus::render_trace_answer` already does.

use std::collections::{HashMap, HashSet};

use super::GAP_VALUE;
use super::corpus::ReasonExample;
use super::kg::{EntityId, Fact, RelationKind};
use super::replay::{ParsedLine, parse_line, replay_trace};

/// The result of splicing a System-1 organ's raw trace with a deterministic
/// System-2 symbolic continuation. See [`handoff`].
#[derive(Debug, Clone, PartialEq)]
pub struct HandoffOutcome {
    /// Hops resolved by the neural prefix (the longest valid on-path prefix
    /// of `organ_output`; `0` if the organ established nothing).
    pub neural_hops: usize,
    /// Hops resolved by the symbolic continuation (`example.depth -
    /// neural_hops` unless the continuation hits a gap first, in which case
    /// the gap-confirming step itself counts as the last symbolic hop).
    pub symbolic_hops: usize,
    /// The final answer after continuation -- an entity label, or the
    /// literal `"gap"`.
    pub answer: String,
    /// Whether `answer` matches `example.gold_answer`.
    pub answer_correct: bool,
    /// `true` if the organ's own `call answer` was already correct *and* the
    /// full trace it sits in was fully valid -- used as-is, with no
    /// continuation spliced in. Keeps the handoff a no-op in-distribution.
    pub organ_answer_used: bool,
    /// The label of the last on-path entity the neural prefix actually
    /// reached -- `example.subject`'s label when `neural_hops == 0`,
    /// otherwise `example.gold_chain[neural_hops - 1].object`'s label. This
    /// is the production splice point: the P6 second slice hands the
    /// System-2 continuation exactly this entity as `thinking:seeks` (see
    /// `docs/ai/decisions/093-dual-process-trace-handoff.md`'s lab↔production
    /// mapping), so it is recorded here rather than recomputed downstream.
    pub splice_entity: String,
    /// The composed trace: neural-prefix hops + synthesized symbolic-suffix
    /// hops, renumbered `1..N` and re-rendered in canonical ADR-092 grammar,
    /// ending in `call answer | value <answer>`.
    pub composed_trace: String,
    /// `replay_trace(example, &composed_trace).executability` -- exposed so
    /// callers can assert/measure that the splice logic actually produces a
    /// fully-valid trace (expected ~1.0 whenever the splice logic is right).
    pub composed_executability: f64,
}

/// Splice `organ_output` (a System-1 organ's raw generated text for
/// `example`) with a deterministic symbolic continuation. Total: never
/// panics, regardless of how malformed `organ_output` is (empty string,
/// garbage text, a truncated trace, or a fully valid one all produce a
/// well-defined [`HandoffOutcome`]).
///
/// See the module docs for the four-step semantics (trivial-accept /
/// longest-valid-prefix / symbolic-continue / compose-and-replay).
pub fn handoff(example: &ReasonExample, organ_output: &str) -> HandoffOutcome {
    // Rule 1: trivial accept -- the organ's own trace is already fully valid
    // and its answer is correct. This is what keeps the handoff a no-op
    // in-distribution (axis-B): no need to touch a trace that's already
    // right.
    let organ_report = replay_trace(example, organ_output);
    if organ_report.answer_correct
        && organ_report.statements_total > 0
        && organ_report.statements_valid == organ_report.statements_total
        && organ_report.malformed_lines.is_empty()
    {
        return HandoffOutcome {
            neural_hops: example.depth,
            symbolic_hops: 0,
            answer: organ_report
                .answer
                .clone()
                .unwrap_or_else(|| example.gold_answer.clone()),
            answer_correct: true,
            organ_answer_used: true,
            splice_entity: splice_entity_label(example, example.depth),
            composed_trace: organ_output.trim().to_string(),
            composed_executability: organ_report.executability,
        };
    }

    // Rule 2: longest valid ON-PATH prefix -- how many leading hops of
    // `example.gold_chain` does the organ's raw output actually establish,
    // validated against `example.prompt_facts` exactly as `replay_trace`
    // would, and in the query's own relation order (an off-path-but-valid
    // recall of some other fact does not count).
    let neural_hops = longest_valid_prefix(example, organ_output).min(example.depth);

    // Rule 3 + 4: symbolic continuation from the entity the neural prefix
    // reached, then compose + re-replay.
    let mut lines: Vec<String> = Vec::with_capacity(example.depth + 2);
    let mut cur = example.subject;
    for hop in example.gold_chain.iter().take(neural_hops) {
        lines.push(format!(
            "step {} | recall {} {} | got {}",
            lines.len() + 1,
            hop.relation.word(),
            cur,
            hop.object,
        ));
        cur = hop.object;
    }

    let mut symbolic_hops = 0usize;
    let mut gap = false;
    for hop in example
        .gold_chain
        .iter()
        .skip(neural_hops)
        .take(example.depth - neural_hops)
    {
        let needed = Fact {
            relation: hop.relation,
            subject: cur,
            object: hop.object,
        };
        if example.prompt_facts.contains(&needed) {
            lines.push(format!(
                "step {} | recall {} {} | got {}",
                lines.len() + 1,
                hop.relation.word(),
                cur,
                hop.object,
            ));
            cur = hop.object;
            symbolic_hops += 1;
        } else {
            lines.push(format!(
                "step {} | check {} {} | {GAP_VALUE}",
                lines.len() + 1,
                hop.relation.word(),
                cur,
            ));
            symbolic_hops += 1;
            gap = true;
            break;
        }
    }

    let answer = if gap {
        GAP_VALUE.to_string()
    } else {
        cur.to_string()
    };

    if !gap {
        let by: Vec<String> = (1..=lines.len()).map(|i| i.to_string()).collect();
        lines.push(format!(
            "step {} | derive {} {} {} | by {}",
            lines.len() + 1,
            example.query_relation,
            example.subject,
            answer,
            by.join(" "),
        ));
    }
    lines.push(format!("call answer | value {answer}"));

    let composed_trace = lines.join("\n");
    let composed_report = replay_trace(example, &composed_trace);

    HandoffOutcome {
        neural_hops,
        symbolic_hops,
        answer,
        answer_correct: composed_report.answer_correct,
        organ_answer_used: false,
        splice_entity: splice_entity_label(example, neural_hops),
        composed_trace,
        composed_executability: composed_report.executability,
    }
}

/// The label of the last on-path entity established after `neural_hops` --
/// `example.subject` when `neural_hops == 0`, else
/// `example.gold_chain[neural_hops - 1].object`. Shared by both `handoff`
/// return sites (trivial-accept passes `example.depth`, the general path
/// passes the computed `neural_hops`) so the two can never disagree on what
/// "the splice point" means.
fn splice_entity_label(example: &ReasonExample, neural_hops: usize) -> String {
    if neural_hops == 0 {
        example.subject.to_string()
    } else {
        example.gold_chain[neural_hops - 1].object.to_string()
    }
}

/// How many leading hops of `example.gold_chain` does `organ_output`
/// validly establish, in order, starting from `example.subject`. Reuses
/// [`parse_line`] (the exact grammar [`super::replay::replay_trace`] uses)
/// so this can never accept a statement replay itself would reject.
///
/// A hop is established either by a `recall` line that names exactly the
/// expected `(relation, current_entity, next_entity)` triple and is present
/// in `example.prompt_facts`, or by a `derive` line whose `by`-refs compose
/// (through other valid `recall` lines, walked the same way
/// `replay_trace` walks them) from the current entity to the expected next
/// entity -- gated on that same triple being present in `prompt_facts`, so a
/// derive can't "shortcut" past a deliberately-redacted (abstention) fact.
fn longest_valid_prefix(example: &ReasonExample, organ_output: &str) -> usize {
    let fact_set: HashSet<(RelationKind, EntityId, EntityId)> = example
        .prompt_facts
        .iter()
        .map(|f| (f.relation, f.subject, f.object))
        .collect();

    // step number -> (subject, got) for every organ statement that parses as
    // a *valid* recall -- mirrors replay_trace's own bookkeeping so `derive`
    // by-refs resolve identically here.
    let mut recall_steps: HashMap<usize, (EntityId, EntityId)> = HashMap::new();
    let mut direct_hops: HashSet<(RelationKind, EntityId, EntityId)> = HashSet::new();
    let mut derives: Vec<(EntityId, EntityId, Vec<usize>)> = Vec::new();

    for raw_line in organ_output.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line) {
            ParsedLine::Recall {
                step,
                relation,
                subject,
                got,
            } => {
                if fact_set.contains(&(relation, subject, got)) {
                    recall_steps.insert(step, (subject, got));
                    direct_hops.insert((relation, subject, got));
                }
            }
            ParsedLine::Derive {
                subject,
                object,
                by,
                ..
            } => {
                derives.push((subject, object, by));
            }
            ParsedLine::Check { .. } | ParsedLine::CallAnswer { .. } | ParsedLine::Malformed => {}
        }
    }

    let mut cur = example.subject;
    let mut hops = 0usize;
    for hop in example.gold_chain.iter().take(example.depth) {
        let triple = (hop.relation, cur, hop.object);
        let established = direct_hops.contains(&triple)
            || (fact_set.contains(&triple)
                && derives.iter().any(|(subj, obj, by)| {
                    *subj == cur
                        && *obj == hop.object
                        && compose_by_refs(&recall_steps, *subj, by) == Some(*obj)
                }));
        if !established {
            break;
        }
        cur = hop.object;
        hops += 1;
    }
    hops
}

/// Walk a `derive` line's `by`-refs through `recall_steps`, exactly as
/// `replay_trace` does: each ref must name a valid recall step whose
/// recorded subject matches the running entity, and the chain must be
/// non-empty to advance at all. Returns `None` on any dangling/mismatched
/// ref.
fn compose_by_refs(
    recall_steps: &HashMap<usize, (EntityId, EntityId)>,
    subject: EntityId,
    by: &[usize],
) -> Option<EntityId> {
    let mut cur = subject;
    for r in by {
        match recall_steps.get(r) {
            Some(&(s, o)) if s == cur => cur = o,
            _ => return None,
        }
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::{ReasonConfig, ReasonDataset, TraceArm};

    fn dataset() -> ReasonDataset {
        ReasonDataset::new(ReasonConfig {
            seed: 11,
            entities_per_type: 40,
            distractor_factor: 2,
            train_depths: vec![2, 3, 4, 5],
            holdout_subject_frac: 0.2,
        })
        .unwrap()
    }

    fn one_answer_example(depth: usize) -> ReasonExample {
        // train_examples draws a random depth per call from train_depths;
        // draw a batch and pick the first that lands on the requested depth
        // (deterministic dataset/seed, so this is reproducible).
        dataset()
            .train_examples(200, TraceArm::Trace)
            .unwrap()
            .into_iter()
            .find(|e| e.depth == depth)
            .expect("requested depth is reachable from train_depths within 200 draws")
    }

    /// First hop-line count of a gold trace equals `example.depth` (used by
    /// several tests below to build a truncated organ output by taking a
    /// textual prefix of the gold trace).
    fn first_n_lines(text: &str, n: usize) -> String {
        text.lines().take(n).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn gold_trace_is_trivially_accepted() {
        let ex = one_answer_example(3);
        let outcome = handoff(&ex, &ex.target);
        assert!(outcome.organ_answer_used);
        assert_eq!(outcome.neural_hops, ex.depth);
        assert_eq!(outcome.symbolic_hops, 0);
        assert!(outcome.answer_correct);
        assert_eq!(outcome.answer, ex.gold_answer);
        assert_eq!(
            outcome.splice_entity,
            ex.gold_chain.last().unwrap().object.to_string()
        );
    }

    #[test]
    fn truncated_gold_trace_splices_the_missing_hops() {
        let ex = one_answer_example(4);
        for j in 1..ex.depth {
            let organ_output = first_n_lines(&ex.target, ex.depth - j);
            let outcome = handoff(&ex, &organ_output);
            assert!(
                !outcome.organ_answer_used,
                "truncated trace (j={j}) must not trivial-accept"
            );
            assert_eq!(
                outcome.neural_hops,
                ex.depth - j,
                "j={j}: neural_hops should cover the untouched prefix"
            );
            assert_eq!(
                outcome.symbolic_hops, j,
                "j={j}: symbolic must cover the rest"
            );
            assert!(outcome.answer_correct, "j={j}: {outcome:?}");
            assert_eq!(outcome.answer, ex.gold_answer);
            assert_eq!(
                outcome.composed_executability, 1.0,
                "j={j}: composed trace must fully replay: {outcome:?}"
            );
            assert_eq!(
                outcome.splice_entity,
                ex.gold_chain[outcome.neural_hops - 1].object.to_string(),
                "j={j}: splice_entity should be the last neurally-resolved entity"
            );
        }
    }

    #[test]
    fn empty_organ_output_is_fully_symbolic() {
        let ex = one_answer_example(3);
        let outcome = handoff(&ex, "");
        assert!(!outcome.organ_answer_used);
        assert_eq!(outcome.neural_hops, 0);
        assert_eq!(outcome.symbolic_hops, ex.depth);
        assert!(outcome.answer_correct);
        assert_eq!(outcome.answer, ex.gold_answer);
        assert_eq!(outcome.composed_executability, 1.0);
        assert_eq!(outcome.splice_entity, ex.subject.to_string());
    }

    #[test]
    fn empty_organ_output_on_abstention_example_yields_gap() {
        let examples = dataset().abstention_examples(20).unwrap();
        let ex = examples.first().expect("at least one abstention example");
        let outcome = handoff(ex, "");
        assert!(!outcome.organ_answer_used);
        assert_eq!(outcome.neural_hops, 0);
        assert_eq!(outcome.answer, "gap");
        assert!(outcome.answer_correct);
        assert_eq!(outcome.composed_executability, 1.0);
        assert_eq!(outcome.splice_entity, ex.subject.to_string());
    }

    #[test]
    fn off_path_valid_recall_does_not_count_as_progress() {
        let ex = one_answer_example(3);
        // A recall of a real prompt fact, but for a subject other than
        // `ex.subject` (a distractor whose subject necessarily differs --
        // genuinely valid against prompt_facts, but not on the query path
        // starting at `ex.subject`, so it must not be mistaken for hop 1).
        let off_path_fact = ex
            .prompt_facts
            .iter()
            .find(|f| !ex.gold_chain.contains(f) && f.subject != ex.subject)
            .expect("distractor_factor > 0 guarantees a non-gold prompt fact off the subject");
        let organ_output = format!(
            "step 1 | recall {} {} | got {}",
            off_path_fact.relation.word(),
            off_path_fact.subject,
            off_path_fact.object,
        );
        let outcome = handoff(&ex, &organ_output);
        assert_eq!(
            outcome.neural_hops, 0,
            "an off-path (even if individually valid) recall must not advance the walk"
        );
        assert_eq!(outcome.symbolic_hops, ex.depth);
        assert!(outcome.answer_correct);
    }

    #[test]
    fn malformed_soup_is_fully_symbolic() {
        let ex = one_answer_example(2);
        let outcome = handoff(&ex, "this is not a trace line\nneither is this\n12345");
        assert!(!outcome.organ_answer_used);
        assert_eq!(outcome.neural_hops, 0);
        assert_eq!(outcome.symbolic_hops, ex.depth);
        assert!(outcome.answer_correct);
        assert_eq!(outcome.composed_executability, 1.0);
    }

    #[test]
    fn handoff_never_panics_on_arbitrary_bytes() {
        let ex = one_answer_example(3);
        let weird_inputs = [
            "\u{0}\u{1}\u{2}",
            "step | | |",
            "step 1 | recall | got",
            "call answer | value ",
            "step 99999999999999999999999 | recall lives_in p000 | got c000",
        ];
        for input in weird_inputs {
            let outcome = handoff(&ex, input);
            // Total: just needs to return, not panic. Sanity-check the
            // invariant that symbolic_hops never exceeds the example depth.
            assert!(outcome.neural_hops <= ex.depth);
            assert!(outcome.symbolic_hops <= ex.depth);
        }
    }
}
