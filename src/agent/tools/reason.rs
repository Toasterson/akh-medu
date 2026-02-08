//! Reason tool: simplify/rewrite an expression using e-graph reasoning.

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

/// Simplify or rewrite a symbolic expression using e-graph reasoning.
pub struct ReasonTool;

impl Tool for ReasonTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "reason".into(),
            description: "Simplify or rewrite a symbolic expression using e-graph reasoning.".into(),
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
}
