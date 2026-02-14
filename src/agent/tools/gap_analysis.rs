//! GapAnalysisTool: analyze knowledge gaps in the KG.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::autonomous::gap::GapAnalysisConfig;
use crate::engine::Engine;
use std::collections::HashSet;

/// Agent tool that identifies knowledge gaps around goal symbols.
pub struct GapAnalysisTool;

impl Tool for GapAnalysisTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "gap_analysis".into(),
            description:
                "Identify knowledge gaps: dead ends, missing predicates, incomplete types.".into(),
            parameters: vec![
                ToolParam {
                    name: "goal".into(),
                    description: "Goal symbol name or ID to analyze around.".into(),
                    required: true,
                },
                ToolParam {
                    name: "max_gaps".into(),
                    description: "Maximum gaps to report (default: 10).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let goal_str = input.require("goal", "gap_analysis")?;
        let max_gaps = input
            .get("max_gaps")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);

        let goal_id = engine.resolve_symbol(goal_str)?;

        let config = GapAnalysisConfig {
            max_gaps,
            ..Default::default()
        };

        let result = engine.analyze_gaps(&[goal_id], config)?;

        let mut lines = Vec::new();
        lines.push(format!(
            "Analyzed {} entities: {} dead ends, coverage {:.0}%",
            result.entities_analyzed,
            result.dead_ends,
            result.coverage_score * 100.0,
        ));

        let mut symbols = vec![goal_id];

        for gap in &result.gaps {
            lines.push(format!("  [{:.2}] {}", gap.severity, gap.description));
            symbols.push(gap.entity);
            symbols.extend(&gap.suggested_predicates);
        }

        Ok(ToolOutput::ok_with_symbols(lines.join("\n"), symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "gap_analysis".into(),
            description: "Identifies missing knowledge via KG and VSA analysis — no side effects."
                .into(),
            parameters: vec![
                ToolParamSchema::required("goal", "Goal symbol name or ID to analyze around."),
                ToolParamSchema::optional("max_gaps", "Maximum gaps to report (default: 10)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg, Capability::VsaAccess]),
                description:
                    "Identifies missing knowledge via KG and VSA analysis — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
