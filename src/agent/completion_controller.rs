//! Evidence-gated completion and VSA reasoning context for controlled turns.
//!
//! A caller opts a turn in by embedding one JSON directive in the prompt:
//! `<akh:control>{...}</akh:control>`.  The host, rather than the model, then
//! decides whether a declaration of completion is supported by fresh,
//! successful tool evidence.  Gates name tools instead of task domains, so the
//! same controller can govern code training, data migrations, deployments, or
//! any other workflow with observable tool boundaries.

use std::collections::HashSet;

use miette::Diagnostic;
use serde::Deserialize;

use crate::engine::Engine;
use crate::infer::InferenceQuery;

const OPEN_TAG: &str = "<akh:control>";
const CLOSE_TAG: &str = "</akh:control>";
const MAX_GATES: usize = 16;
const MAX_TOOLS_PER_GATE: usize = 16;
const MAX_VSA_SEEDS: usize = 8;

/// A host-enforced stage in an ordered completion contract.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceGate {
    /// Human-readable post-condition named in retry feedback.
    pub label: String,
    /// Any one successful matching tool predicate satisfies the gate.
    pub any_of: Vec<ToolEvidence>,
}

/// One observable tool result that may satisfy an evidence gate.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolEvidence {
    /// Exact registered tool name.
    pub tool: String,
    /// Optional exact argument subset. Unlisted arguments are ignored.
    #[serde(default)]
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// The JSON payload carried inside an `<akh:control>` prompt directive.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ControlDirective {
    stages: Vec<EvidenceGate>,
    #[serde(default)]
    vsa_seeds: Vec<String>,
}

/// A deterministic completion verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionDecision {
    /// Every ordered evidence gate has been satisfied this turn.
    Complete,
    /// The model must continue, with concrete host-generated feedback.
    Continue { feedback: String },
}

/// Invalid explicit control input.  An explicit but malformed directive fails
/// closed; silently treating it as an ordinary chat turn would discard the
/// caller's requested post-conditions.
#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum CompletionControlError {
    #[error("completion directive is missing closing </akh:control> tag")]
    #[diagnostic(
        code(akh::agent::completion_control::missing_close_tag),
        help("Close the JSON directive with </akh:control>.")
    )]
    MissingCloseTag,
    #[error("completion directive contains invalid JSON: {0}")]
    #[diagnostic(
        code(akh::agent::completion_control::invalid_json),
        help("Provide one valid JSON object between the control tags.")
    )]
    InvalidJson(#[from] serde_json::Error),
    #[error("completion directive must contain between 1 and {MAX_GATES} stages")]
    #[diagnostic(
        code(akh::agent::completion_control::invalid_stage_count),
        help("Add at least one evidence stage and keep the list bounded.")
    )]
    InvalidStageCount,
    #[error(
        "completion gate {index} must have a label and between 1 and {MAX_TOOLS_PER_GATE} tools"
    )]
    #[diagnostic(
        code(akh::agent::completion_control::invalid_gate),
        help("Give every gate a non-empty label and at least one non-empty tool name.")
    )]
    InvalidGate { index: usize },
    #[error("completion directive may name at most {MAX_VSA_SEEDS} VSA seeds")]
    #[diagnostic(
        code(akh::agent::completion_control::too_many_vsa_seeds),
        help("Keep only the most relevant exact symbol labels.")
    )]
    TooManyVsaSeeds,
    #[error("completion directive contains more than one <akh:control> block")]
    #[diagnostic(
        code(akh::agent::completion_control::multiple_directives),
        help("Merge the requirements into a single control block.")
    )]
    MultipleDirectives,
}

/// Per-turn ordered evidence controller.
///
/// `next_stage` only advances on a successful matching tool call. Evidence
/// from a later stage cannot be banked early, so a memory write performed
/// before verification does not certify the final implementation.
#[derive(Debug, Clone)]
pub struct CompletionController {
    directive: ControlDirective,
    next_stage: usize,
    last_failed_tool: Option<String>,
}

impl CompletionController {
    /// Parse an optional controller directive from a user prompt.
    pub fn from_prompt(prompt: &str) -> Result<Option<Self>, CompletionControlError> {
        let Some(open_at) = prompt.find(OPEN_TAG) else {
            return Ok(None);
        };
        let payload_start = open_at + OPEN_TAG.len();
        let rest = &prompt[payload_start..];
        let close_offset = rest
            .find(CLOSE_TAG)
            .ok_or(CompletionControlError::MissingCloseTag)?;
        let after = &rest[close_offset + CLOSE_TAG.len()..];
        if after.contains(OPEN_TAG) {
            return Err(CompletionControlError::MultipleDirectives);
        }

        let directive: ControlDirective = serde_json::from_str(rest[..close_offset].trim())?;
        validate_directive(&directive)?;
        Ok(Some(Self {
            directive,
            next_stage: 0,
            last_failed_tool: None,
        }))
    }

    /// Record one fresh tool result from this turn.
    pub fn observe(&mut self, tool: &str, arguments: &serde_json::Value, success: bool) {
        if let Some(gate) = self.directive.stages.get(self.next_stage)
            && gate.matches(tool, arguments)
        {
            if success {
                self.next_stage += 1;
                self.last_failed_tool = None;
            } else {
                self.last_failed_tool = Some(tool.to_string());
            }
            return;
        }

        // Re-running an earlier successful stage invalidates every later
        // stage. This makes evidence fresh with respect to subsequent state
        // changes: edit → verify → edit requires another verify before a
        // writeback can close the contract.
        if success
            && let Some(earlier) = self.directive.stages[..self.next_stage]
                .iter()
                .position(|gate| gate.matches(tool, arguments))
        {
            self.next_stage = earlier + 1;
            self.last_failed_tool = None;
        }
    }

    /// Decide whether the model's no-tool response may end the turn.
    pub fn completion_decision(&self) -> CompletionDecision {
        let Some(gate) = self.directive.stages.get(self.next_stage) else {
            return CompletionDecision::Complete;
        };
        let expected = gate.expectation();
        let failed = self
            .last_failed_tool
            .as_deref()
            .map(|tool| format!(" The latest `{tool}` call failed and does not count."))
            .unwrap_or_default();
        CompletionDecision::Continue {
            feedback: format!(
                "Completion contract is not satisfied. Next required evidence: `{label}` via a successful call to {expected}.{failed} Continue using tools; do not claim completion yet.",
                label = gate.label,
            ),
        }
    }

    /// Tool names that must remain offered even when retrieval scopes the
    /// catalog. Order is first appearance in the contract, with duplicates
    /// removed.
    pub fn required_tools(&self) -> Vec<&str> {
        let mut seen = HashSet::new();
        self.directive
            .stages
            .iter()
            .flat_map(|gate| gate.any_of.iter().map(|evidence| evidence.tool.as_str()))
            .filter(|tool| seen.insert(*tool))
            .collect()
    }

    /// Explicit VSA seed labels supplied by the caller.
    pub fn vsa_seeds(&self) -> &[String] {
        &self.directive.vsa_seeds
    }

    /// Human-readable system context stating the host-enforced contract.
    pub fn system_context(&self) -> String {
        let stages = self
            .directive
            .stages
            .iter()
            .enumerate()
            .map(|(index, gate)| format!("{}. {}: {}", index + 1, gate.label, gate.expectation()))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "## Completion Contract\nThe host enforces these ordered gates using only successful tool calls from this turn. Later evidence cannot satisfy an earlier gate. Do not declare completion until all gates pass.\n{stages}"
        )
    }

    /// Label of the next unsatisfied gate, if any.
    pub fn next_gate_label(&self) -> Option<&str> {
        self.directive
            .stages
            .get(self.next_stage)
            .map(|gate| gate.label.as_str())
    }
}

impl EvidenceGate {
    fn expectation(&self) -> String {
        self.any_of
            .iter()
            .map(ToolEvidence::expectation)
            .collect::<Vec<_>>()
            .join(" or ")
    }

    fn matches(&self, tool: &str, arguments: &serde_json::Value) -> bool {
        self.any_of
            .iter()
            .any(|evidence| evidence.matches(tool, arguments))
    }
}

impl ToolEvidence {
    fn expectation(&self) -> String {
        if self.arguments.is_empty() {
            self.tool.clone()
        } else {
            format!(
                "{} with arguments containing {}",
                self.tool,
                serde_json::Value::Object(self.arguments.clone())
            )
        }
    }

    fn matches(&self, tool: &str, arguments: &serde_json::Value) -> bool {
        self.tool == tool
            && self.arguments.iter().all(|(name, expected)| {
                arguments.as_object().and_then(|actual| actual.get(name)) == Some(expected)
            })
    }
}

fn validate_directive(directive: &ControlDirective) -> Result<(), CompletionControlError> {
    if directive.stages.is_empty() || directive.stages.len() > MAX_GATES {
        return Err(CompletionControlError::InvalidStageCount);
    }
    for (index, gate) in directive.stages.iter().enumerate() {
        if gate.label.trim().is_empty()
            || gate.any_of.is_empty()
            || gate.any_of.len() > MAX_TOOLS_PER_GATE
            || gate
                .any_of
                .iter()
                .any(|evidence| evidence.tool.trim().is_empty())
        {
            return Err(CompletionControlError::InvalidGate { index });
        }
    }
    if directive.vsa_seeds.len() > MAX_VSA_SEEDS {
        return Err(CompletionControlError::TooManyVsaSeeds);
    }
    Ok(())
}

/// Builds a bounded pre-action reasoning block from the engine's real
/// spreading-activation path. The block is explicitly advisory: Rust's
/// compiler/tests and other domain verifiers remain the proof boundary.
pub struct VsaReasoningContext;

impl VsaReasoningContext {
    /// Resolve explicit labels plus prompt keywords (including namespaced
    /// suffixes such as `typestate` → `rust:typestate`) and run VSA inference.
    pub fn build(engine: &Engine, prompt: &str, explicit_seeds: &[String]) -> Option<String> {
        let seeds = resolve_seeds(engine, prompt, explicit_seeds);
        if seeds.is_empty() {
            return None;
        }

        let result = engine
            .infer(&InferenceQuery {
                seeds: seeds.clone(),
                top_k: 8,
                max_depth: 2,
                min_confidence: 0.1,
                min_similarity: 0.35,
                verify_with_egraph: false,
                predicate_filter: None,
            })
            .ok()?;

        let seed_labels = seeds
            .iter()
            .map(|id| engine.resolve_label(*id))
            .collect::<Vec<_>>()
            .join(", ");
        let mut lines = Vec::new();
        for (symbol, confidence) in result.activations.iter().take(8) {
            let provenance = result
                .provenance
                .iter()
                .find(|record| record.derived_id == *symbol)
                .map(|record| {
                    let sources = record
                        .sources
                        .iter()
                        .map(|source| engine.resolve_label(*source))
                        .collect::<Vec<_>>();
                    let source_text = if sources.is_empty() {
                        "none".to_string()
                    } else {
                        sources.join(", ")
                    };
                    format!(
                        "{}; depth {}; sources: {source_text}",
                        crate::agent::explain::derivation_kind_prose(&record.kind),
                        record.depth
                    )
                })
                .unwrap_or_else(|| "activation provenance unavailable".to_string());
            lines.push(format!(
                "- {} ({confidence:.2}; {provenance})",
                engine.resolve_label(*symbol)
            ));
        }
        if lines.is_empty() {
            return None;
        }

        Some(format!(
            "## VSA Reasoning Context\nTreat these activations as associative evidence, not proof. Validate conclusions at the domain boundary.\nSeeds: {seed_labels}\nActivations:\n{}",
            lines.join("\n")
        ))
    }
}

fn resolve_seeds(
    engine: &Engine,
    prompt: &str,
    explicit_seeds: &[String],
) -> Vec<crate::symbol::SymbolId> {
    let mut seeds = Vec::new();
    let mut seen = HashSet::new();
    for label in explicit_seeds {
        if let Ok(id) = engine.resolve_symbol(label)
            && seen.insert(id)
        {
            seeds.push(id);
        }
    }

    let keywords = crate::nlu::kg_context::extract_keywords(prompt);
    for keyword in &keywords {
        if seeds.len() >= MAX_VSA_SEEDS {
            break;
        }
        if let Ok(id) = engine
            .resolve_symbol(keyword)
            .or_else(|_| engine.resolve_symbol(&keyword.to_lowercase()))
            && seen.insert(id)
        {
            seeds.push(id);
        }
    }

    if seeds.len() < MAX_VSA_SEEDS {
        let lowered: HashSet<String> = keywords.iter().map(|word| word.to_lowercase()).collect();
        let mut suffix_matches = engine
            .all_symbols()
            .into_iter()
            .filter(|meta| {
                meta.label
                    .rsplit([':', '/', '#'])
                    .next()
                    .is_some_and(|suffix| lowered.contains(&suffix.to_lowercase()))
            })
            .collect::<Vec<_>>();
        suffix_matches.sort_by(|a, b| a.label.cmp(&b.label));
        for meta in suffix_matches {
            if seeds.len() >= MAX_VSA_SEEDS {
                break;
            }
            if seen.insert(meta.id) {
                seeds.push(meta.id);
            }
        }
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::graph::Triple;

    const DIRECTIVE: &str = r#"<akh:control>{"stages":[{"label":"mutation","any_of":[{"tool":"edit"},{"tool":"file_io","arguments":{"action":"write"}}]},{"label":"verified","any_of":[{"tool":"compile_feedback"}]},{"label":"learned","any_of":[{"tool":"memory_write"}]}],"vsa_seeds":["rust:typestate"]}</akh:control>"#;

    #[test]
    fn parses_an_explicit_ordered_contract() {
        let controller = CompletionController::from_prompt(DIRECTIVE)
            .expect("valid directive")
            .expect("controller present");
        assert_eq!(
            controller.required_tools(),
            ["edit", "file_io", "compile_feedback", "memory_write"]
        );
        assert_eq!(controller.vsa_seeds(), ["rust:typestate"]);
        assert!(
            controller
                .system_context()
                .contains("successful tool calls from this turn")
        );
    }

    #[test]
    fn gates_are_successful_fresh_and_ordered() {
        let mut controller = CompletionController::from_prompt(DIRECTIVE)
            .unwrap()
            .unwrap();

        controller.observe("memory_write", &serde_json::json!({}), true);
        controller.observe("edit", &serde_json::json!({}), false);
        assert_eq!(controller.next_gate_label(), Some("mutation"));

        controller.observe("edit", &serde_json::json!({}), true);
        controller.observe("memory_write", &serde_json::json!({}), true);
        assert_eq!(controller.next_gate_label(), Some("verified"));

        controller.observe("compile_feedback", &serde_json::json!({}), true);
        assert_eq!(controller.next_gate_label(), Some("learned"));
        assert!(matches!(
            controller.completion_decision(),
            CompletionDecision::Continue { .. }
        ));

        controller.observe("memory_write", &serde_json::json!({}), true);
        assert_eq!(
            controller.completion_decision(),
            CompletionDecision::Complete
        );

        controller.observe("edit", &serde_json::json!({}), true);
        assert_eq!(
            controller.next_gate_label(),
            Some("verified"),
            "a new mutation makes prior verification and writeback stale"
        );
    }

    #[test]
    fn a_gate_can_require_exact_tool_arguments() {
        let prompt = r#"<akh:control>{"stages":[{"label":"tests","any_of":[{"tool":"compile_feedback","arguments":{"command":"test"}}]}]}</akh:control>"#;
        let mut controller = CompletionController::from_prompt(prompt).unwrap().unwrap();
        controller.observe(
            "compile_feedback",
            &serde_json::json!({"command":"check", "path":"."}),
            true,
        );
        assert_eq!(controller.next_gate_label(), Some("tests"));
        controller.observe(
            "compile_feedback",
            &serde_json::json!({"command":"test", "path":"."}),
            true,
        );
        assert_eq!(
            controller.completion_decision(),
            CompletionDecision::Complete
        );
    }

    #[test]
    fn file_io_read_does_not_satisfy_a_write_predicate() {
        let mut controller = CompletionController::from_prompt(DIRECTIVE)
            .unwrap()
            .unwrap();
        controller.observe(
            "file_io",
            &serde_json::json!({"action":"read", "path":"src/lib.rs"}),
            true,
        );
        assert_eq!(controller.next_gate_label(), Some("mutation"));
        controller.observe(
            "file_io",
            &serde_json::json!({"action":"write", "path":"src/lib.rs"}),
            true,
        );
        assert_eq!(controller.next_gate_label(), Some("verified"));
    }

    #[test]
    fn malformed_explicit_directive_fails_closed() {
        let error = CompletionController::from_prompt("<akh:control>{oops}</akh:control>")
            .expect_err("malformed control must not silently disable enforcement");
        assert!(error.to_string().contains("invalid JSON"));
    }

    #[test]
    fn vsa_context_resolves_namespaced_symbols_and_exposes_provenance() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        let typestate = engine.resolve_or_create_entity("rust:typestate").unwrap();
        let prevents = engine.resolve_or_create_relation("rust:prevents").unwrap();
        let invalid = engine
            .resolve_or_create_entity("rust:invalid-state")
            .unwrap();
        engine
            .add_triple(&Triple::new(typestate, prevents, invalid))
            .unwrap();

        let context = VsaReasoningContext::build(
            &engine,
            "Use typestate to prevent an invalid state",
            &["rust:typestate".into()],
        )
        .expect("a known seed should produce reasoning context");

        assert!(context.contains("rust:typestate"));
        assert!(context.contains("rust:invalid-state"));
        assert!(context.contains("graph edge"));
        assert!(context.contains("depth 1"));
        assert!(context.contains("sources: rust:typestate"));
        assert!(context.contains("associative evidence, not proof"));
    }
}
