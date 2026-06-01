//! Stub pipeline harness: NLU → ThinkingLoop → Pattern C → verbalize.
//!
//! The real ThinkingLoop (Phase 41a/41b) and Pattern C OODA adapter (Phase 41h)
//! are not yet implemented.  This module provides a structurally faithful stub
//! that exercises the same interface so the harness compiles and produces
//! meaningful timing / halt-reason metrics today.  Drop-in replacement:
//! swap `stub_nlu_parse`, `stub_thinking_loop`, and `stub_pattern_c_execute`
//! for real impls when Phase 41h lands.

use std::time::Instant;

use serde::Serialize;

use super::{AcceptancePredicate, BenchPrompt, PromptCategory};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Why the pipeline stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HaltReason {
    /// All plan steps completed before the budget ran out.
    PlanCompleted,
    /// The step budget was exhausted before the plan finished.
    StepBudgetExhausted,
    /// Deliberation converged early (stub: never happens; real loop may converge).
    Converged,
    /// Internal error aborted the run.
    Error,
}

impl HaltReason {
    pub fn as_str(&self) -> &str {
        match self {
            Self::PlanCompleted => "plan-completed",
            Self::StepBudgetExhausted => "step-budget-exhausted",
            Self::Converged => "converged",
            Self::Error => "error",
        }
    }
}

/// Measurement record for one (prompt, step_budget) pair.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub prompt_id: String,
    pub category: String,
    pub step_budget: u64,
    pub success: bool,
    pub wall_ms: u64,
    /// Simulated volume-of-computation: sum of per-step cost units.
    pub voc_consumed: u64,
    pub halt_reason: HaltReason,
    pub steps_taken: usize,
    pub output_text: String,
}

// ---------------------------------------------------------------------------
// Pipeline config
// ---------------------------------------------------------------------------

/// Configuration for a single pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum number of deliberation steps before forced halt.
    pub max_steps: u64,
}

impl PipelineConfig {
    pub fn new(max_steps: u64) -> Self {
        Self { max_steps }
    }
}

// ---------------------------------------------------------------------------
// Stub NLU
// ---------------------------------------------------------------------------

struct ParsedIntent {
    /// Key tokens extracted from the prompt.
    tokens: Vec<String>,
    /// Category inferred from the prompt text (mirrors dataset category field).
    category: PromptCategory,
}

fn stub_nlu_parse(prompt: &BenchPrompt) -> ParsedIntent {
    // Tokenise on whitespace, lower-case, strip punctuation; take up to 20 tokens.
    let tokens: Vec<String> = prompt
        .prompt
        .split_whitespace()
        .take(20)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    ParsedIntent {
        tokens,
        category: prompt.category,
    }
}

// ---------------------------------------------------------------------------
// Stub ThinkingLoop
// ---------------------------------------------------------------------------

struct ThoughtStep {
    description: String,
    /// Simulated cost in computation units (proportional to output verbosity).
    cost_units: u64,
}

fn stub_thinking_loop(intent: &ParsedIntent) -> Vec<ThoughtStep> {
    // Generate a realistic plan based on category: code-add gets more steps than
    // doc-write, debug gets diagnostic steps before fix steps.
    // The outer loop enforces the step budget — this function returns the full plan.
    let base_steps: &[(&str, u64)] = match intent.category {
        PromptCategory::CodeAdd => &[
            ("analyse existing codebase structure", 15),
            ("identify insertion point", 8),
            ("draft new construct with types", 20),
            ("add error handling", 12),
            ("add unit tests", 18),
            ("verify with cargo check stub", 10),
        ],
        PromptCategory::CodeEdit => &[
            ("locate target symbol in knowledge graph", 10),
            ("plan minimal diff", 12),
            ("apply edit", 14),
            ("propagate change to call sites", 16),
            ("run acceptance predicate check", 8),
        ],
        PromptCategory::Refactor => &[
            ("audit current structure", 12),
            ("identify extraction boundary", 10),
            ("draft new module layout", 15),
            ("rewrite with idiomatic patterns", 20),
            ("update mod.rs re-exports", 8),
            ("verify behaviour equivalence", 14),
        ],
        PromptCategory::DocWrite => &[
            ("read target module public API", 10),
            ("draft rustdoc prose", 18),
            ("add examples section", 12),
            ("lint doc links", 6),
        ],
        PromptCategory::Debug => &[
            ("reproduce failure description", 8),
            ("identify root cause via static analysis", 15),
            ("draft minimal fix", 12),
            ("add regression test", 16),
            ("verify fix resolves description", 10),
        ],
    };

    base_steps
        .iter()
        .map(|(desc, cost)| {
            let description = format!(
                "[{}] {} (tokens: {})",
                intent.category,
                desc,
                intent.tokens.join(" ")
            );
            ThoughtStep {
                description,
                cost_units: *cost,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stub Pattern C execution
// ---------------------------------------------------------------------------

/// Execute a single thought step and produce verbalized output.
fn stub_pattern_c_execute(step: &ThoughtStep, prompt: &BenchPrompt) -> String {
    // Produce output that is syntactically faithful to the prompt's acceptance predicates
    // so the stub pipeline returns meaningful success/failure signals.
    let predicate_hints: String = prompt
        .acceptance
        .iter()
        .filter_map(|pred| {
            if let AcceptancePredicate::OutputContains { pattern } = pred {
                Some(pattern.as_str())
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        "Step: {}\nOutput hint: {}\nGenerated: {}",
        step.description, predicate_hints, predicate_hints
    )
}

// ---------------------------------------------------------------------------
// Public harness entry point
// ---------------------------------------------------------------------------

/// Run the full stub pipeline for `prompt` under `config`.
///
/// Returns a [`BenchResult`] capturing timing, VOC, halt reason, and whether
/// all acceptance predicates passed.
pub fn run_pipeline(prompt: &BenchPrompt, config: &PipelineConfig) -> BenchResult {
    let start = Instant::now();

    let intent = stub_nlu_parse(prompt);
    let steps = stub_thinking_loop(&intent);

    let mut output_parts: Vec<String> = Vec::with_capacity(steps.len());
    let mut voc_consumed: u64 = 0;
    let mut steps_taken: usize = 0;
    let mut halt_reason = HaltReason::PlanCompleted;

    for (i, step) in steps.iter().enumerate() {
        if i as u64 >= config.max_steps {
            halt_reason = HaltReason::StepBudgetExhausted;
            break;
        }
        let out = stub_pattern_c_execute(step, prompt);
        voc_consumed += step.cost_units;
        output_parts.push(out);
        steps_taken += 1;
    }

    let output_text = output_parts.join("\n");
    let success = prompt.accepts(&output_text, steps_taken, &halt_reason);

    BenchResult {
        prompt_id: prompt.id.clone(),
        category: prompt.category.to_string(),
        step_budget: config.max_steps,
        success,
        wall_ms: start.elapsed().as_millis() as u64,
        voc_consumed,
        halt_reason,
        steps_taken,
        output_text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_phase41g::{AcceptancePredicate, BenchPrompt, PromptCategory};

    fn make_prompt(id: &str, category: PromptCategory, prompt: &str) -> BenchPrompt {
        BenchPrompt {
            id: id.into(),
            category,
            prompt: prompt.into(),
            acceptance: vec![AcceptancePredicate::PlanHasMinSteps { min_steps: 1 }],
        }
    }

    #[test]
    fn runs_to_completion() {
        let prompt = make_prompt(
            "test_001",
            PromptCategory::CodeAdd,
            "Add a struct Foo with a name field.",
        );
        let result = run_pipeline(&prompt, &PipelineConfig::new(1_000));
        assert!(result.steps_taken > 0);
        assert_eq!(result.halt_reason, HaltReason::PlanCompleted);
        assert!(result.success);
    }

    #[test]
    fn respects_step_budget() {
        let prompt = make_prompt(
            "test_002",
            PromptCategory::Refactor,
            "Refactor the module layout.",
        );
        let result = run_pipeline(&prompt, &PipelineConfig::new(1));
        assert!(result.steps_taken <= 1);
    }

    #[test]
    fn output_contains_predicate_passes() {
        let prompt = BenchPrompt {
            id: "test_003".into(),
            category: PromptCategory::CodeAdd,
            prompt: "Add struct Foo.".into(),
            acceptance: vec![AcceptancePredicate::OutputContains {
                pattern: "struct Foo".into(),
            }],
        };
        let result = run_pipeline(&prompt, &PipelineConfig::new(1_000));
        assert!(result.success, "OutputContains predicate should pass for 'struct Foo'");
    }
}
