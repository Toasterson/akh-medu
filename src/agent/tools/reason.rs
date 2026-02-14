//! Reason tool: simplify/rewrite an expression using e-graph reasoning.

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Simplify or rewrite a symbolic expression using e-graph reasoning.
pub struct ReasonTool;

impl Tool for ReasonTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "reason".into(),
            description: "Simplify or rewrite a symbolic expression using e-graph reasoning."
                .into(),
            parameters: vec![ToolParam {
                name: "expression".into(),
                description: "S-expression to simplify (e.g. \"(not (not x))\").".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let expr = input.require("expression", "reason")?;

        match engine.simplify_expression(expr) {
            Ok(simplified) => {
                let result = format!("Input: {expr}\nSimplified: {simplified}");
                Ok(ToolOutput::ok(result))
            }
            Err(e) => Err(AgentError::ToolExecution {
                tool_name: "reason".into(),
                message: format!("{e}"),
            }),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "reason".into(),
            description: "Simplify or rewrite a symbolic expression using e-graph reasoning."
                .into(),
            parameters: vec![ToolParamSchema::required(
                "expression",
                "S-expression to simplify (e.g. \"(not (not x))\").",
            )],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::Reason, Capability::ReadKg]),
                description: "Pure symbolic reasoning via e-graph rewriting — no side effects."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}
