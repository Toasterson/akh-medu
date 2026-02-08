//! Similarity search tool: find symbols similar to a given symbol via VSA.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

/// Find symbols similar to a given symbol using VSA hypervector similarity.
pub struct SimilaritySearchTool;

impl Tool for SimilaritySearchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "similarity_search".into(),
            description: "Find symbols similar to a given symbol via VSA hypervector similarity."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "symbol".into(),
                    description: "Symbol name or ID to search around.".into(),
                    required: true,
                },
                ToolParam {
                    name: "top_k".into(),
                    description: "Number of results to return (default: 5).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let symbol_str = input.require("symbol", "similarity_search")?;
        let top_k: usize = input
            .get("top_k")
            .and_then(|k| k.parse().ok())
            .unwrap_or(5);

        let symbol_id = engine
            .resolve_symbol(symbol_str)?;
        let label = engine.resolve_label(symbol_id);

        let results = engine
            .search_similar_to(symbol_id, top_k)?;

        if results.is_empty() {
            return Ok(ToolOutput::ok_with_symbols(
                format!("No similar symbols found for \"{label}\"."),
                vec![symbol_id],
            ));
        }

        let mut lines = Vec::new();
        let mut symbols = vec![symbol_id];
        for (i, sr) in results.iter().enumerate() {
            lines.push(format!(
                "  {}. \"{}\" / {} (similarity: {:.4})",
                i + 1,
                engine.resolve_label(sr.symbol_id),
                sr.symbol_id,
                sr.similarity,
            ));
            symbols.push(sr.symbol_id);
        }

        let result = format!(
            "Similar to \"{}\" ({} results):\n{}",
            label,
            results.len(),
            lines.join("\n")
        );
        Ok(ToolOutput::ok_with_symbols(result, symbols))
    }
}
