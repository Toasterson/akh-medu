//! Aggregate scoring over a batch of reasoning outputs: [`score_reason_outputs`]
//! replays every `(example, output)` pair via [`super::replay_trace`] and
//! rolls the reports up into answer accuracy, executability, trace validity,
//! a per-depth breakdown, the abstention confusion matrix, and an
//! anti-collapse diversity check -- mirroring the metric style of
//! `crate::eval` for the tool-calling corpus.

use std::collections::{HashMap, HashSet};

use super::corpus::ReasonExample;
use super::replay::{ReplayReport, replay_trace};

/// Per-depth breakdown slice of a [`ReasonEvalSummary`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStats {
    pub n: usize,
    pub answer_accuracy: f64,
    pub executability_rate: f64,
}

/// The abstention (gap) confusion-matrix rates, mirroring
/// `crate::eval::AbstentionReport`'s shape for the reasoning corpus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbstentionStats {
    pub n_expected: usize,
    pub n_emitted: usize,
    /// Of the should-abstain examples, the fraction correctly abstained.
    pub recall: f64,
    /// Of the times the model abstained, the fraction that were correct.
    pub precision: f64,
}

/// The anti-collapse alarm over final answers, independent of correctness --
/// mirrors `crate::eval::DiversityReport`'s intent for the reasoning corpus's
/// answer values instead of tool calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReasonDiversityReport {
    /// Fraction of outputs whose final answer value is distinct from the
    /// rest of the batch.
    pub distinct_answer_ratio: f64,
    /// Shannon entropy (bits) over the answer-value distribution, normalized
    /// to `[0, 1]` by the number of distinct *gold* answers in the batch.
    pub answer_entropy: f64,
}

/// The full aggregate report over a scored batch.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasonEvalSummary {
    pub n: usize,
    pub answer_accuracy: f64,
    pub executability_rate: f64,
    /// Fraction of examples whose trace was **fully** valid: every emitted
    /// statement checked out and no line was malformed.
    pub trace_valid_rate: f64,
    pub per_depth: HashMap<usize, DepthStats>,
    pub abstention: AbstentionStats,
    pub diversity: ReasonDiversityReport,
    /// Whole-trace quality (Phase 51 P3-RL A1): redundancy, drift, and
    /// padding — the reward-hacking surfaces that per-statement replay is
    /// blind to. See [`super::quality`] and ADR 096.
    pub quality: super::quality::TraceQualitySummary,
}

/// Score a batch of raw model `outputs` against their source `examples`
/// (paired by index -- `outputs[i]` is scored against `examples[i]`; if the
/// two slices differ in length, the shorter one wins).
pub fn score_reason_outputs(examples: &[ReasonExample], outputs: &[String]) -> ReasonEvalSummary {
    let n = examples.len().min(outputs.len());
    let reports: Vec<(&ReasonExample, ReplayReport)> = (0..n)
        .map(|i| (&examples[i], replay_trace(&examples[i], &outputs[i])))
        .collect();

    if n == 0 {
        return ReasonEvalSummary {
            n: 0,
            answer_accuracy: 0.0,
            executability_rate: 0.0,
            trace_valid_rate: 0.0,
            per_depth: HashMap::new(),
            abstention: AbstentionStats {
                n_expected: 0,
                n_emitted: 0,
                recall: 0.0,
                precision: 0.0,
            },
            diversity: ReasonDiversityReport {
                distinct_answer_ratio: 0.0,
                answer_entropy: 0.0,
            },
            quality: super::quality::summarize_trace_quality(&[], &[]),
        };
    }

    let answer_accuracy =
        reports.iter().filter(|(_, r)| r.answer_correct).count() as f64 / n as f64;
    let executability_rate = reports.iter().map(|(_, r)| r.executability).sum::<f64>() / n as f64;
    let trace_valid_rate = reports
        .iter()
        .filter(|(_, r)| {
            r.statements_total > 0
                && r.statements_valid == r.statements_total
                && r.malformed_lines.is_empty()
        })
        .count() as f64
        / n as f64;

    let per_depth = per_depth_breakdown(&reports);
    let abstention = abstention_stats(&reports);
    let diversity = diversity_report(examples, &reports, n);
    // Scored over the same common prefix replay used, so the quality means
    // and the replay rates always describe the identical example set.
    let quality = super::quality::summarize_trace_quality(&examples[..n], &outputs[..n]);

    ReasonEvalSummary {
        n,
        answer_accuracy,
        executability_rate,
        trace_valid_rate,
        per_depth,
        abstention,
        diversity,
        quality,
    }
}

fn per_depth_breakdown(reports: &[(&ReasonExample, ReplayReport)]) -> HashMap<usize, DepthStats> {
    let mut raw: HashMap<usize, (usize, usize, f64)> = HashMap::new();
    for (ex, r) in reports {
        let entry = raw.entry(ex.depth).or_insert((0, 0, 0.0));
        entry.0 += 1;
        if r.answer_correct {
            entry.1 += 1;
        }
        entry.2 += r.executability;
    }
    raw.into_iter()
        .map(|(depth, (count, correct, exec_sum))| {
            (
                depth,
                DepthStats {
                    n: count,
                    answer_accuracy: correct as f64 / count.max(1) as f64,
                    executability_rate: exec_sum / count.max(1) as f64,
                },
            )
        })
        .collect()
}

fn abstention_stats(reports: &[(&ReasonExample, ReplayReport)]) -> AbstentionStats {
    let n_expected = reports.iter().filter(|(ex, _)| ex.is_abstention).count();
    let true_abstain = reports
        .iter()
        .filter(|(ex, r)| ex.is_abstention && r.abstained)
        .count();
    let false_abstain = reports
        .iter()
        .filter(|(ex, r)| !ex.is_abstention && r.abstained)
        .count();
    let n_emitted = true_abstain + false_abstain;
    AbstentionStats {
        n_expected,
        n_emitted,
        recall: if n_expected == 0 {
            0.0
        } else {
            true_abstain as f64 / n_expected as f64
        },
        precision: if n_emitted == 0 {
            0.0
        } else {
            true_abstain as f64 / n_emitted as f64
        },
    }
}

fn diversity_report(
    examples: &[ReasonExample],
    reports: &[(&ReasonExample, ReplayReport)],
    n: usize,
) -> ReasonDiversityReport {
    let gold_universe: HashSet<&str> = examples[..n]
        .iter()
        .map(|e| e.gold_answer.as_str())
        .collect();
    let universe_size = gold_universe.len().max(2);

    let answers: Vec<&str> = reports
        .iter()
        .map(|(_, r)| r.answer.as_deref().unwrap_or("<none>"))
        .collect();
    let distinct: HashSet<&str> = answers.iter().copied().collect();
    let distinct_answer_ratio = distinct.len() as f64 / n as f64;

    let mut counts: HashMap<&str, usize> = HashMap::new();
    for a in &answers {
        *counts.entry(*a).or_insert(0) += 1;
    }
    let total = n as f64;
    let raw_entropy: f64 = counts
        .values()
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum();
    let max_entropy = (universe_size as f64).log2();
    let answer_entropy = if max_entropy > 0.0 {
        (raw_entropy / max_entropy).min(1.0)
    } else {
        0.0
    };

    ReasonDiversityReport {
        distinct_answer_ratio,
        answer_entropy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason::{ReasonConfig, ReasonDataset, TraceArm};

    fn small_dataset() -> ReasonDataset {
        ReasonDataset::new(ReasonConfig {
            seed: 3,
            entities_per_type: 40,
            distractor_factor: 2,
            train_depths: vec![2, 3],
            holdout_subject_frac: 0.2,
        })
        .unwrap()
    }

    #[test]
    fn perfect_outputs_score_perfectly() {
        let dataset = small_dataset();
        let examples = dataset.train_examples(20, TraceArm::Trace).unwrap();
        let outputs: Vec<String> = examples.iter().map(|e| e.target.clone()).collect();
        let summary = score_reason_outputs(&examples, &outputs);
        assert_eq!(summary.n, 20);
        assert_eq!(summary.answer_accuracy, 1.0);
        assert_eq!(summary.executability_rate, 1.0);
        assert_eq!(summary.trace_valid_rate, 1.0);
        assert!(!summary.per_depth.is_empty());
        for stats in summary.per_depth.values() {
            assert_eq!(stats.answer_accuracy, 1.0);
        }
    }

    #[test]
    fn garbage_outputs_score_near_zero() {
        let dataset = small_dataset();
        let examples = dataset.train_examples(10, TraceArm::Trace).unwrap();
        let outputs: Vec<String> = examples
            .iter()
            .map(|_| "nonsense output".to_string())
            .collect();
        let summary = score_reason_outputs(&examples, &outputs);
        assert_eq!(summary.answer_accuracy, 0.0);
        assert_eq!(summary.executability_rate, 0.0);
        assert_eq!(summary.trace_valid_rate, 0.0);
    }

    #[test]
    fn abstention_batch_scores_recall_and_precision() {
        let dataset = small_dataset();
        let mut examples = dataset.abstention_examples(10).unwrap();
        let call_examples = dataset.train_examples(10, TraceArm::Trace).unwrap();
        examples.extend(call_examples);
        let outputs: Vec<String> = examples.iter().map(|e| e.target.clone()).collect();
        let summary = score_reason_outputs(&examples, &outputs);
        assert_eq!(summary.abstention.n_expected, 10);
        assert_eq!(summary.abstention.recall, 1.0);
        assert_eq!(summary.abstention.precision, 1.0);
    }

    #[test]
    fn empty_batch_does_not_panic() {
        let summary = score_reason_outputs(&[], &[]);
        assert_eq!(summary.n, 0);
        assert_eq!(summary.answer_accuracy, 0.0);
    }
}
