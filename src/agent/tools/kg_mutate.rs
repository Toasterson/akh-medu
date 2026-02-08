//! KG mutation tool: add a triple to the knowledge graph.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;
use crate::graph::Triple;

/// Add a triple to the knowledge graph.
pub struct KgMutateTool;

impl Tool for KgMutateTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "kg_mutate".into(),
            description: "Add a triple (subject, predicate, object) to the knowledge graph.".into(),
            parameters: vec![
                ToolParam {
                    name: "subject".into(),
                    description: "Subject symbol name or ID.".into(),
                    required: true,
                },
                ToolParam {
                    name: "predicate".into(),
                    description: "Predicate (relation) symbol name or ID.".into(),
                    required: true,
                },
                ToolParam {
                    name: "object".into(),
                    description: "Object symbol name or ID.".into(),
                    required: true,
                },
                ToolParam {
                    name: "confidence".into(),
                    description: "Confidence score 0.0-1.0 (default: 1.0).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let subject_str = input.require("subject", "kg_mutate")?;
        let predicate_str = input.require("predicate", "kg_mutate")?;
        let object_str = input.require("object", "kg_mutate")?;
        let confidence: f32 = input
            .get("confidence")
            .and_then(|c| c.parse().ok())
            .unwrap_or(1.0);

        let s = engine
            .resolve_or_create_entity(subject_str)?;
        let p = engine
            .resolve_or_create_relation(predicate_str)?;
        let o = engine
            .resolve_or_create_entity(object_str)?;

        engine
            .add_triple(&Triple::new(s, p, o).with_confidence(confidence))?;

        let result = format!(
            "Added triple: \"{}\" -> {} -> \"{}\" [{:.2}]",
            engine.resolve_label(s),
            engine.resolve_label(p),
            engine.resolve_label(o),
            confidence,
        );
        Ok(ToolOutput::ok_with_symbols(result, vec![s, p, o]))
    }
}
