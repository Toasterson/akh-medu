//! KG query tool: query triples from/to a symbol.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

/// Query triples from/to a symbol in the knowledge graph.
pub struct KgQueryTool;

impl Tool for KgQueryTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "kg_query".into(),
            description: "Query triples from/to a symbol in the knowledge graph.".into(),
            parameters: vec![
                ToolParam {
                    name: "symbol".into(),
                    description: "Symbol name or ID to query.".into(),
                    required: true,
                },
                ToolParam {
                    name: "direction".into(),
                    description: "Direction: 'from', 'to', or 'both' (default: both).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let symbol_str = input.require("symbol", "kg_query")?;
        let direction = input.get("direction").unwrap_or("both");

        let symbol_id = engine
            .resolve_symbol(symbol_str)?;

        let label = engine.resolve_label(symbol_id);
        let mut lines = Vec::new();
        let mut symbols = vec![symbol_id];

        if direction == "from" || direction == "both" {
            let triples = engine.triples_from(symbol_id);
            for t in &triples {
                lines.push(format!(
                    "\"{}\" -> {} -> \"{}\"  [{:.2}]",
                    label,
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                    t.confidence,
                ));
                symbols.push(t.object);
                symbols.push(t.predicate);
            }
        }

        if direction == "to" || direction == "both" {
            let triples = engine.triples_to(symbol_id);
            for t in &triples {
                lines.push(format!(
                    "\"{}\" -> {} -> \"{}\"  [{:.2}]",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    label,
                    t.confidence,
                ));
                symbols.push(t.subject);
                symbols.push(t.predicate);
            }
        }

        if lines.is_empty() {
            Ok(ToolOutput::ok_with_symbols(
                format!("No triples found for \"{label}\" (direction: {direction})."),
                symbols,
            ))
        } else {
            let result = format!(
                "Found {} triple(s) for \"{}\":\n{}",
                lines.len(),
                label,
                lines.join("\n")
            );
            Ok(ToolOutput::ok_with_symbols(result, symbols))
        }
    }
}
