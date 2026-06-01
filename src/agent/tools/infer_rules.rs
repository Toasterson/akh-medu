//! InferRulesTool: run forward-chaining inference rules on the KG.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::autonomous::rule_engine::RuleEngineConfig;
use crate::engine::Engine;
use std::collections::HashSet;

/// Agent tool that triggers forward-chaining rule inference.
pub struct InferRulesTool;

impl Tool for InferRulesTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "infer_rules".into(),
            description: "Run forward-chaining inference rules to derive new triples.".into(),
            parameters: vec![
                ToolParam {
                    name: "max_iterations".into(),
                    description: "Maximum forward-chaining iterations (default: 5).".into(),
                    required: false,
                },
                ToolParam {
                    name: "min_confidence".into(),
                    description: "Minimum confidence for derived triples (default: 0.1).".into(),
                    required: false,
                },
                ToolParam {
                    name: "cleanup".into(),
                    description: "Remove contaminated is-a/child-of triples before inference \
                                  (true/false, default: false). Use to retroactively purge \
                                  cross-domain type-hierarchy pollution from prior inference \
                                  bugs (e.g. agent identities linked to CI runners or storage \
                                  nodes)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "purge_self_referential".into(),
                    description: "Remove self-referential triples (A → p → A) before inference \
                                  (true/false, default: false). Use to retroactively purge \
                                  invalid self-links that were inserted before the guard was \
                                  active."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let max_iterations = input
            .get("max_iterations")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(5);
        let min_confidence = input
            .get("min_confidence")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.1);
        let run_cleanup = input
            .get("cleanup")
            .map(|s| matches!(s.as_ref(), "true" | "1" | "yes"))
            .unwrap_or(false);
        let purge_self_ref = input
            .get("purge_self_referential")
            .map(|s| matches!(s.as_ref(), "true" | "1" | "yes"))
            .unwrap_or(false);

        let mut lines = Vec::new();

        if run_cleanup {
            let purged = engine.purge_contaminated_inferences()?;
            if purged > 0 {
                lines.push(format!(
                    "Cleanup: removed {purged} contaminated is-a/child-of triple(s)."
                ));
            } else {
                lines.push("Cleanup: no contaminated triples found.".into());
            }
        }

        if purge_self_ref {
            let purged = engine.purge_self_referential_triples()?;
            if purged > 0 {
                lines.push(format!(
                    "Purge: removed {purged} self-referential triple(s)."
                ));
            } else {
                lines.push("Purge: no self-referential triples found.".into());
            }
        }

        let config = RuleEngineConfig {
            max_iterations,
            min_confidence,
            ..Default::default()
        };

        let result = engine.run_rules(config)?;

        lines.push(format!(
            "Derived {} new triple(s) in {} iteration(s){}",
            result.derived.len(),
            result.iterations,
            if result.reached_fixpoint {
                " (fixpoint reached)"
            } else {
                ""
            },
        ));

        for (rule, count) in &result.rule_stats {
            if *count > 0 {
                lines.push(format!("  {rule}: {count} derivation(s)"));
            }
        }

        let symbols: Vec<_> = result
            .derived
            .iter()
            .flat_map(|dt| [dt.triple.subject, dt.triple.predicate, dt.triple.object])
            .collect();

        Ok(ToolOutput::ok_with_symbols(lines.join("\n"), symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "infer_rules".into(),
            description: "Forward-chaining inference — derives new triples from rules.".into(),
            parameters: vec![
                ToolParamSchema::optional(
                    "max_iterations",
                    "Maximum forward-chaining iterations (default: 5).",
                ),
                ToolParamSchema::optional(
                    "min_confidence",
                    "Minimum confidence for derived triples (default: 0.1).",
                ),
                ToolParamSchema::optional(
                    "cleanup",
                    "Remove contaminated is-a/child-of triples before inference \
                     (true/false, default: false).",
                ),
                ToolParamSchema::optional(
                    "purge_self_referential",
                    "Remove self-referential triples (A → p → A) before inference \
                     (true/false, default: false).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([
                    Capability::Reason,
                    Capability::ReadKg,
                    Capability::WriteKg,
                ]),
                description: "Forward-chaining inference — derives new triples from rules.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
