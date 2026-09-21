//! QA benchmark for evaluating LoRA adapter quality against agent-specific eval sets.
//!
//! Compares agent-specific QA accuracy between the base model and a fine-tuned
//! LoRA adapter.  The benchmark is intentionally lightweight — it runs offline
//! against a pre-generated evaluation JSONL and requires no embedding model.
//!
//! Accuracy is measured via keyword-overlap F1:
//! - Precision: fraction of output content-words that appear in the expected answer.
//! - Recall: fraction of expected content-words that appear in the output.
//! - F1: harmonic mean of precision and recall.
//!
//! For semantic evaluation, use `scripts/benchmark_adapter.py`.

use std::collections::HashSet;
use std::path::Path;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::lora::AgentId;
use super::qa_generator::QaPair;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors from benchmark loading and evaluation.
#[derive(Debug, Error, Diagnostic)]
pub enum BenchmarkError {
    #[error("evaluation set not found for '{agent_id}' at {path}")]
    #[diagnostic(
        code(akh::benchmark::not_found),
        help(
            "Generate a QA dataset first:\n  \
             python scripts/generate_qa_pairs.py --agent {agent_id}\n\
             Then run this benchmark."
        )
    )]
    EvalSetNotFound {
        agent_id: String,
        path: std::path::PathBuf,
    },

    #[error("I/O error reading eval set for '{agent_id}': {reason}")]
    #[diagnostic(code(akh::benchmark::io), help("Check that the path is readable."))]
    Io { agent_id: String, reason: String },

    #[error("malformed QA pair in eval set for '{agent_id}' at line {line}: {reason}")]
    #[diagnostic(
        code(akh::benchmark::parse),
        help("Regenerate the eval set: python scripts/generate_qa_pairs.py --agent {agent_id}")
    )]
    ParseError {
        agent_id: String,
        line: usize,
        reason: String,
    },

    #[error("inference error during benchmark for '{agent_id}': {reason}")]
    #[diagnostic(
        code(akh::benchmark::inference),
        help("Ensure the base model or adapter is loaded correctly.")
    )]
    Inference { agent_id: String, reason: String },
}

/// Result alias for benchmark operations.
pub type BenchmarkResult<T> = Result<T, BenchmarkError>;

// ── Benchmark configuration ───────────────────────────────────────────────────

/// Configuration for a QA benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Maximum number of eval pairs to use (None = all).
    pub max_pairs: Option<usize>,
    /// Minimum keyword-overlap F1 to consider a response correct (default: 0.5).
    pub accuracy_threshold: f32,
    /// Stop-words excluded from keyword matching.
    pub stop_words: HashSet<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            max_pairs: Some(100),
            accuracy_threshold: 0.5,
            stop_words: default_stop_words(),
        }
    }
}

fn default_stop_words() -> HashSet<String> {
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "and", "but", "or", "as",
        "at", "by", "for", "in", "of", "on", "to", "up", "with", "about", "above", "after",
        "again", "against", "all", "am", "any", "because", "before", "between", "during", "each",
        "few", "from", "further", "here", "how", "i", "if", "into", "it", "its", "itself", "just",
        "more", "most", "my", "myself", "now", "only", "other", "our", "out", "own", "same", "she",
        "he", "her", "him", "his", "than", "that", "their", "them", "then", "there", "these",
        "they", "this", "those", "through", "too", "under", "until", "us", "very", "was", "we",
        "what", "when", "where", "which", "while", "who", "whom", "why", "you", "your",
    ]
    .iter()
    .map(|&s| s.to_string())
    .collect()
}

// ── Per-pair result ───────────────────────────────────────────────────────────

/// Evaluation result for a single QA pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairResult {
    pub input: String,
    pub expected: String,
    pub actual: String,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub correct: bool,
}

// ── Aggregate benchmark result ────────────────────────────────────────────────

/// Aggregate benchmark result for one agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBenchmarkResult {
    pub agent_id: String,
    /// Number of pairs evaluated.
    pub pairs_evaluated: usize,
    /// Fraction of pairs where f1 >= accuracy_threshold.
    pub accuracy: f32,
    /// Mean keyword-overlap F1 across all pairs.
    pub mean_f1: f32,
    /// Whether this result meets the minimum accuracy bar.
    pub meets_bar: bool,
    /// Per-pair breakdown.
    pub pairs: Vec<PairResult>,
}

impl AgentBenchmarkResult {
    /// Minimum accuracy required to consider an adapter production-ready.
    pub const MIN_ACCURACY: f32 = 0.60;
}

// ── Benchmark engine ──────────────────────────────────────────────────────────

/// Evaluates adapter quality against an agent-specific QA eval set.
pub struct AdapterBenchmark {
    config: BenchmarkConfig,
}

impl AdapterBenchmark {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Load QA pairs from a JSONL file.
    pub fn load_eval_set(&self, agent: AgentId, eval_path: &Path) -> BenchmarkResult<Vec<QaPair>> {
        if !eval_path.exists() {
            return Err(BenchmarkError::EvalSetNotFound {
                agent_id: agent.to_string(),
                path: eval_path.to_path_buf(),
            });
        }

        let content = std::fs::read_to_string(eval_path).map_err(|e| BenchmarkError::Io {
            agent_id: agent.to_string(),
            reason: e.to_string(),
        })?;

        let mut pairs = Vec::new();
        for (i, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let pair: QaPair =
                serde_json::from_str(line).map_err(|e| BenchmarkError::ParseError {
                    agent_id: agent.to_string(),
                    line: i + 1,
                    reason: e.to_string(),
                })?;
            pairs.push(pair);
        }

        if let Some(max) = self.config.max_pairs {
            pairs.truncate(max);
        }

        Ok(pairs)
    }

    /// Evaluate a set of responses against expected answers.
    ///
    /// `inference_fn` takes an input prompt and returns the model's response.
    /// Pass a closure that calls Candle or a configured HTTP inference provider.
    pub fn evaluate<F>(
        &self,
        agent: AgentId,
        pairs: &[QaPair],
        mut inference_fn: F,
    ) -> BenchmarkResult<AgentBenchmarkResult>
    where
        F: FnMut(&str) -> Result<String, String>,
    {
        let mut pair_results = Vec::with_capacity(pairs.len());

        for pair in pairs {
            let actual = inference_fn(&pair.input).map_err(|e| BenchmarkError::Inference {
                agent_id: agent.to_string(),
                reason: e,
            })?;

            let (precision, recall, f1) = self.keyword_f1(&pair.output, &actual);
            let correct = f1 >= self.config.accuracy_threshold;

            pair_results.push(PairResult {
                input: pair.input.clone(),
                expected: pair.output.clone(),
                actual,
                precision,
                recall,
                f1,
                correct,
            });
        }

        let pairs_evaluated = pair_results.len();
        let correct_count = pair_results.iter().filter(|r| r.correct).count();
        let accuracy = if pairs_evaluated == 0 {
            0.0
        } else {
            correct_count as f32 / pairs_evaluated as f32
        };
        let mean_f1 = if pairs_evaluated == 0 {
            0.0
        } else {
            pair_results.iter().map(|r| r.f1).sum::<f32>() / pairs_evaluated as f32
        };

        Ok(AgentBenchmarkResult {
            agent_id: agent.to_string(),
            pairs_evaluated,
            accuracy,
            mean_f1,
            meets_bar: accuracy >= AgentBenchmarkResult::MIN_ACCURACY,
            pairs: pair_results,
        })
    }

    /// Compute keyword-overlap precision, recall, and F1 between expected and actual text.
    fn keyword_f1(&self, expected: &str, actual: &str) -> (f32, f32, f32) {
        let expected_tokens = self.tokenize(expected);
        let actual_tokens = self.tokenize(actual);

        if expected_tokens.is_empty() && actual_tokens.is_empty() {
            return (1.0, 1.0, 1.0);
        }
        if expected_tokens.is_empty() || actual_tokens.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let expected_set: HashSet<&str> = expected_tokens.iter().map(|s| s.as_str()).collect();
        let actual_set: HashSet<&str> = actual_tokens.iter().map(|s| s.as_str()).collect();

        let true_positives = expected_set.intersection(&actual_set).count() as f32;
        let precision = true_positives / actual_set.len() as f32;
        let recall = true_positives / expected_set.len() as f32;
        let f1 = if precision + recall < f32::EPSILON {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        (precision, recall, f1)
    }

    /// Tokenize text and filter out stop-words.
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphabetic())
            .filter(|t| !t.is_empty() && !self.config.stop_words.contains(*t))
            .map(|t| t.to_string())
            .collect()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::qa_generator::QaCategory;

    fn bench() -> AdapterBenchmark {
        AdapterBenchmark::new(BenchmarkConfig::default())
    }

    #[test]
    fn exact_match_yields_perfect_f1() {
        let b = bench();
        let text = "knowledge graph structure nodes edges relations";
        let (p, r, f1) = b.keyword_f1(text, text);
        assert!((f1 - 1.0).abs() < 1e-4, "f1={f1}");
        assert!((p - 1.0).abs() < 1e-4);
        assert!((r - 1.0).abs() < 1e-4);
    }

    #[test]
    fn disjoint_tokens_zero_f1() {
        let b = bench();
        let (_, _, f1) = b.keyword_f1("apple orange banana", "car truck bus");
        assert!((f1 - 0.0).abs() < 1e-4, "f1={f1}");
    }

    #[test]
    fn partial_overlap_nonzero_f1() {
        let b = bench();
        let (_, _, f1) = b.keyword_f1("knowledge graph representation", "knowledge schema");
        assert!(f1 > 0.0 && f1 < 1.0, "f1={f1}");
    }

    #[test]
    fn both_empty_yields_perfect_f1() {
        let b = bench();
        let (p, r, f1) = b.keyword_f1("", "");
        assert!((f1 - 1.0).abs() < 1e-4);
        assert!((p - 1.0).abs() < 1e-4);
        assert!((r - 1.0).abs() < 1e-4);
    }

    #[test]
    fn one_empty_yields_zero_f1() {
        let b = bench();
        let (_, _, f1) = b.keyword_f1("something here", "");
        assert!((f1 - 0.0).abs() < 1e-4);
    }

    #[test]
    fn stop_words_excluded_from_matching() {
        let b = bench();
        // Stop-words only ("the", "is", "a") → both empty after filtering.
        let (_, _, f1) = b.keyword_f1("the is a", "a is the");
        assert!((f1 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn evaluate_with_perfect_inference() {
        let b = bench();
        let pairs = vec![QaPair {
            input: "What is your role?".into(),
            output: "I am knowledge builder focused on structure precision".into(),
            agent: "tecton".into(),
            category: QaCategory::Identity,
        }];

        let result = b
            .evaluate(AgentId::Tecton, &pairs, |_| {
                Ok("I am knowledge builder focused on structure precision".into())
            })
            .unwrap();

        assert_eq!(result.pairs_evaluated, 1);
        assert!((result.accuracy - 1.0).abs() < 1e-4);
        assert!(result.pairs[0].correct);
        assert!(result.meets_bar);
    }

    #[test]
    fn evaluate_with_wrong_inference() {
        let b = bench();
        let pairs = vec![QaPair {
            input: "Describe your decision process.".into(),
            output: "balance truth weigh evidence carefully judgment".into(),
            agent: "maat".into(),
            category: QaCategory::Domain,
        }];

        let result = b
            .evaluate(AgentId::Maat, &pairs, |_| {
                Ok("sunshine cats rainbows playing outside".into())
            })
            .unwrap();

        assert_eq!(result.pairs_evaluated, 1);
        assert!(!result.pairs[0].correct);
    }

    #[test]
    fn evaluate_propagates_inference_error() {
        let b = bench();
        let pairs = vec![QaPair {
            input: "Question".into(),
            output: "Answer".into(),
            agent: "horus".into(),
            category: QaCategory::Domain,
        }];

        let result = b.evaluate(AgentId::Horus, &pairs, |_| Err("model error".into()));
        assert!(matches!(result, Err(BenchmarkError::Inference { .. })));
    }

    #[test]
    fn load_eval_set_missing_file_returns_err() {
        let b = bench();
        let result = b.load_eval_set(AgentId::Set, Path::new("/nonexistent/eval.jsonl"));
        assert!(matches!(
            result,
            Err(BenchmarkError::EvalSetNotFound { .. })
        ));
    }

    #[test]
    fn load_eval_set_valid_jsonl() {
        use crate::training::qa_generator::QaGenerator;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("seeds.jsonl");
        QaGenerator::write_seeds(AgentId::Thoth, &path).unwrap();

        let b = bench();
        let pairs = b.load_eval_set(AgentId::Thoth, &path).unwrap();
        assert!(!pairs.is_empty());
    }

    #[test]
    fn benchmark_config_defaults() {
        let cfg = BenchmarkConfig::default();
        assert!((cfg.accuracy_threshold - 0.5).abs() < 1e-4);
        assert_eq!(cfg.max_pairs, Some(100));
        assert!(!cfg.stop_words.is_empty());
    }

    #[test]
    fn agent_benchmark_result_min_accuracy() {
        assert!((AgentBenchmarkResult::MIN_ACCURACY - 0.60).abs() < 1e-4);
    }
}
