//! The trace replayer: ADR 092 decision D6, "engine replay is the scorer".
//!
//! [`replay_trace`] parses a model's raw output line-by-line and checks every
//! statement against the *example's own prompt facts* -- `recall` must match
//! a fact actually present in the prompt, `derive`'s `by` refs must cite
//! earlier `recall` steps that compose to the claimed subject/object, `check`
//! must name a relation/subject pair that is genuinely absent from the
//! prompt, and the final `call answer` must agree with a validated
//! `derive`/`check`. The parser is total: an unrecognised line is recorded in
//! [`ReplayReport::malformed_lines`], never a panic.

use std::collections::{HashMap, HashSet};

use super::GAP_VALUE;
use super::corpus::ReasonExample;
use super::kg::{EntityId, RelationKind};

/// The result of replaying one model output against its source
/// [`ReasonExample`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Every non-empty line in the output.
    pub statements_total: usize,
    /// Lines that validated against the prompt facts / prior steps.
    pub statements_valid: usize,
    /// `statements_valid / statements_total` (`0.0` if there were no
    /// statements at all).
    pub executability: f64,
    /// The value named by the (first) `call answer` line, if any -- an
    /// entity label or the literal `"gap"`.
    pub answer: Option<String>,
    /// Whether `answer` matches the example's `gold_answer` (a plain string
    /// comparison, independent of whether the trace that produced it was
    /// itself valid -- see the module docs on why these are separate axes).
    pub answer_correct: bool,
    /// Whether the model's final answer was the abstention value.
    pub abstained: bool,
    /// Whether at least one `check ... | gap` statement validated (the
    /// relation/subject pair it named was genuinely absent from the prompt).
    pub gap_confirmed: bool,
    /// Raw text of every line that didn't parse into a recognised shape.
    pub malformed_lines: Vec<String>,
    /// Step numbers of `derive` statements whose `by` references were
    /// dangling (pointed at a step that doesn't exist, wasn't a valid
    /// `recall`, or didn't compose to the claimed subject/object).
    pub broken_by_refs: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum ParsedLine {
    Recall {
        step: usize,
        relation: RelationKind,
        subject: EntityId,
        got: EntityId,
    },
    Derive {
        step: usize,
        subject: EntityId,
        object: EntityId,
        by: Vec<usize>,
    },
    Check {
        step: usize,
        relation: RelationKind,
        subject: EntityId,
    },
    CallAnswer {
        value: String,
    },
    Malformed,
}

/// Parse one trimmed, non-empty line. Total: never panics, falls back to
/// [`ParsedLine::Malformed`] for anything that doesn't match the ADR 092
/// grammar.
fn parse_line(line: &str) -> ParsedLine {
    let segs: Vec<&str> = line.split(" | ").map(str::trim).collect();
    if segs.len() < 2 {
        return ParsedLine::Malformed;
    }
    let head = segs[0];

    if head == "call answer" {
        let Some(value) = segs.get(1).and_then(|s| s.strip_prefix("value ")) else {
            return ParsedLine::Malformed;
        };
        return ParsedLine::CallAnswer {
            value: value.trim().to_string(),
        };
    }

    let Some(rest) = head.strip_prefix("step ") else {
        return ParsedLine::Malformed;
    };
    let Ok(step) = rest.trim().parse::<usize>() else {
        return ParsedLine::Malformed;
    };
    if segs.len() < 3 {
        return ParsedLine::Malformed;
    }
    let mut op_tokens = segs[1].split_whitespace();
    let result = segs[2];

    match op_tokens.next() {
        Some("recall") => {
            let (Some(rel_s), Some(subj_s)) = (op_tokens.next(), op_tokens.next()) else {
                return ParsedLine::Malformed;
            };
            let (Some(relation), Some(subject)) =
                (RelationKind::parse(rel_s), EntityId::parse(subj_s))
            else {
                return ParsedLine::Malformed;
            };
            let Some(got_s) = result.strip_prefix("got ") else {
                return ParsedLine::Malformed;
            };
            let Some(got) = EntityId::parse(got_s.trim()) else {
                return ParsedLine::Malformed;
            };
            ParsedLine::Recall {
                step,
                relation,
                subject,
                got,
            }
        }
        Some("derive") => {
            let rest_tokens: Vec<&str> = op_tokens.collect();
            if rest_tokens.len() < 3 {
                return ParsedLine::Malformed;
            }
            // rest_tokens[0] is the query-relation name; its label isn't
            // validated here -- the semantic check is the `by`-ref
            // composition below, which is what actually proves the subject
            // reaches the claimed object.
            let (Some(subject), Some(object)) = (
                EntityId::parse(rest_tokens[1]),
                EntityId::parse(rest_tokens[2]),
            ) else {
                return ParsedLine::Malformed;
            };
            let Some(by_s) = result.strip_prefix("by ") else {
                return ParsedLine::Malformed;
            };
            let by: Option<Vec<usize>> = by_s
                .split_whitespace()
                .map(|t| t.parse::<usize>().ok())
                .collect();
            let Some(by) = by else {
                return ParsedLine::Malformed;
            };
            if by.is_empty() {
                return ParsedLine::Malformed;
            }
            ParsedLine::Derive {
                step,
                subject,
                object,
                by,
            }
        }
        Some("check") => {
            let (Some(rel_s), Some(subj_s)) = (op_tokens.next(), op_tokens.next()) else {
                return ParsedLine::Malformed;
            };
            let (Some(relation), Some(subject)) =
                (RelationKind::parse(rel_s), EntityId::parse(subj_s))
            else {
                return ParsedLine::Malformed;
            };
            if result != GAP_VALUE {
                return ParsedLine::Malformed;
            }
            ParsedLine::Check {
                step,
                relation,
                subject,
            }
        }
        _ => ParsedLine::Malformed,
    }
}

/// Replay every statement in `output` against `example`'s own prompt facts
/// (ADR 092 D6). Total and never panics: unrecognised lines land in
/// `malformed_lines` instead of aborting the replay.
pub fn replay_trace(example: &ReasonExample, output: &str) -> ReplayReport {
    let fact_set: HashSet<(RelationKind, EntityId, EntityId)> = example
        .prompt_facts
        .iter()
        .map(|f| (f.relation, f.subject, f.object))
        .collect();
    let subj_rel_set: HashSet<(RelationKind, EntityId)> = example
        .prompt_facts
        .iter()
        .map(|f| (f.relation, f.subject))
        .collect();

    let mut statements_total = 0usize;
    let mut statements_valid = 0usize;
    let mut malformed_lines = Vec::new();
    let mut broken_by_refs = Vec::new();
    // step number (as printed by the model, not a recomputed counter) ->
    // (subject, got) for every *valid* recall step -- an off-by-one or
    // fabricated step number in a later `by` ref is exactly what this
    // catches (ADR 092 D2).
    let mut recall_steps: HashMap<usize, (EntityId, EntityId)> = HashMap::new();
    let mut valid_derive_objects: HashSet<EntityId> = HashSet::new();
    let mut gap_confirmed = false;
    let mut answer: Option<String> = None;

    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        statements_total += 1;

        match parse_line(line) {
            ParsedLine::Malformed => {
                malformed_lines.push(line.to_string());
            }
            ParsedLine::Recall {
                step,
                relation,
                subject,
                got,
            } => {
                if fact_set.contains(&(relation, subject, got)) {
                    statements_valid += 1;
                    recall_steps.insert(step, (subject, got));
                }
            }
            ParsedLine::Check {
                relation, subject, ..
            } => {
                if !subj_rel_set.contains(&(relation, subject)) {
                    statements_valid += 1;
                    gap_confirmed = true;
                }
            }
            ParsedLine::Derive {
                step,
                subject,
                object,
                by,
            } => {
                let mut cur = subject;
                let mut ok = true;
                for &r in &by {
                    match recall_steps.get(&r) {
                        Some(&(s, o)) if s == cur => cur = o,
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok && cur == object {
                    statements_valid += 1;
                    valid_derive_objects.insert(object);
                } else {
                    broken_by_refs.push(step);
                }
            }
            ParsedLine::CallAnswer { value } => {
                // Only the *first* call-answer line is treated as "the"
                // answer -- the format expects exactly one, and taking the
                // first keeps this deterministic if a model repeats itself.
                if answer.is_none() {
                    answer = Some(value.clone());
                }
                let valid = if value == GAP_VALUE {
                    gap_confirmed
                } else {
                    EntityId::parse(&value).is_some_and(|e| valid_derive_objects.contains(&e))
                };
                if valid {
                    statements_valid += 1;
                }
            }
        }
    }

    let answer_correct = answer.as_deref() == Some(example.gold_answer.as_str());
    let abstained = answer.as_deref() == Some(GAP_VALUE);
    let executability = if statements_total == 0 {
        0.0
    } else {
        statements_valid as f64 / statements_total as f64
    };

    ReplayReport {
        statements_total,
        statements_valid,
        executability,
        answer,
        answer_correct,
        abstained,
        gap_confirmed,
        malformed_lines,
        broken_by_refs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::{ReasonConfig, ReasonDataset, TraceArm};

    fn one_example(depth: usize) -> ReasonExample {
        let config = ReasonConfig {
            seed: 5,
            entities_per_type: 40,
            distractor_factor: 2,
            train_depths: vec![depth],
            holdout_subject_frac: 0.2,
        };
        let dataset = ReasonDataset::new(config).unwrap();
        dataset
            .train_examples(1, TraceArm::Trace)
            .unwrap()
            .remove(0)
    }

    #[test]
    fn gold_trace_replays_at_full_executability() {
        let ex = one_example(3);
        let report = replay_trace(&ex, &ex.target);
        assert_eq!(report.statements_valid, report.statements_total);
        assert_eq!(report.executability, 1.0);
        assert!(report.answer_correct);
        assert!(report.malformed_lines.is_empty());
        assert!(report.broken_by_refs.is_empty());
    }

    #[test]
    fn wrong_got_entity_is_caught() {
        let ex = one_example(3);
        // Corrupt the first `got <entity>` to a different-but-well-formed
        // entity label of the same type (so this exercises "fabricated/wrong
        // fact", not "malformed line") -- computed from the example's own
        // gold chain rather than a hardcoded label, so it can't drift out of
        // sync with the KG's actual assignment.
        let real = ex.gold_chain[0].object;
        // No modulus needed: `EntityId` doesn't validate bounds, and an
        // out-of-range fake entity is *still* correctly rejected below
        // because it simply isn't in `prompt_facts` either way.
        let fake = EntityId {
            ty: real.ty,
            index: real.index + 1,
        };
        assert_ne!(real, fake, "fixture must actually pick a different entity");
        let corrupted = ex
            .target
            .replacen(&format!("got {real}"), &format!("got {fake}"), 1);
        assert_ne!(
            corrupted, ex.target,
            "test fixture did not actually corrupt the trace"
        );
        let report = replay_trace(&ex, &corrupted);
        assert!(
            report.statements_valid < report.statements_total,
            "corrupted got-entity should reduce validity: {report:?}"
        );
    }

    #[test]
    fn dangling_by_ref_is_caught() {
        let ex = one_example(3);
        let by_line = format!("by 1 2 {}", 3 + 900); // step number that can't exist
        let corrupted = ex
            .target
            .lines()
            .map(|l| {
                if l.contains("| derive ") {
                    let head = l.split(" | by ").next().unwrap();
                    format!("{head} | {by_line}")
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_ne!(corrupted, ex.target);
        let report = replay_trace(&ex, &corrupted);
        assert!(
            !report.broken_by_refs.is_empty(),
            "dangling by-ref should be recorded: {report:?}"
        );
        assert!(report.statements_valid < report.statements_total);
    }

    #[test]
    fn fabricated_fact_is_caught() {
        let ex = one_example(2);
        // Replace the first recall's relation+subject with something that
        // (almost certainly) names no fact in the prompt at all.
        let corrupted = ex.target.replacen(
            &format!("recall {} {}", ex.gold_chain[0].relation.word(), ex.subject),
            "recall lives_in z999",
            1,
        );
        let report = replay_trace(&ex, &corrupted);
        assert!(
            report.statements_valid < report.statements_total,
            "fabricated fact should be rejected: {report:?}"
        );
    }

    #[test]
    fn malformed_lines_are_reported_not_panicked() {
        let ex = one_example(2);
        let report = replay_trace(&ex, "this is not a trace line at all\nnor is this");
        assert_eq!(report.malformed_lines.len(), 2);
        assert_eq!(report.statements_valid, 0);
        assert_eq!(report.answer, None);
        assert!(!report.answer_correct);
    }

    #[test]
    fn empty_output_has_zero_executability_not_nan() {
        let ex = one_example(2);
        let report = replay_trace(&ex, "");
        assert_eq!(report.statements_total, 0);
        assert_eq!(report.executability, 0.0);
    }
}
