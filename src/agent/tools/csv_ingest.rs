//! CSV ingest tool: parse CSV files and add triples to the knowledge graph.
//!
//! Supports two formats:
//! - **SPO format**: each row is `(subject, predicate, object [, confidence])`
//! - **Entity format**: first column is subject, headers are predicates, cells are objects

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;

/// Ingest triples from a CSV file into the knowledge graph.
pub struct CsvIngestTool;

impl Tool for CsvIngestTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "csv_ingest".into(),
            description: "Ingest triples from a CSV file into the knowledge graph.".into(),
            parameters: vec![
                ToolParam {
                    name: "path".into(),
                    description: "Path to the CSV file.".into(),
                    required: true,
                },
                ToolParam {
                    name: "format".into(),
                    description: "CSV format: 'spo' (subject,predicate,object[,confidence]) \
                                  or 'entity' (first col = subject, headers = predicates, cells = objects). \
                                  Default: 'spo'."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "delimiter".into(),
                    description: "Column delimiter character. Default: ','.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path_str = input.require("path", "csv_ingest")?;
        let format = input.get("format").unwrap_or("spo");
        let delimiter = input
            .get("delimiter")
            .and_then(|d| d.chars().next())
            .unwrap_or(',');

        let content = match std::fs::read_to_string(path_str) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to read \"{path_str}\": {e}"
                )));
            }
        };

        match format {
            "spo" => parse_spo(&content, delimiter, engine),
            "entity" => parse_entity(&content, delimiter, engine),
            other => Ok(ToolOutput::err(format!(
                "Unknown CSV format: \"{other}\". Use 'spo' or 'entity'."
            ))),
        }
    }
}

/// Parse SPO format: each row is `subject, predicate, object [, confidence]`.
fn parse_spo(content: &str, delimiter: char, engine: &Engine) -> AgentResult<ToolOutput> {
    let mut ingested = 0usize;
    let mut errors = 0usize;
    let mut symbols = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).map(|f| f.trim()).collect();
        if fields.len() < 3 {
            errors += 1;
            continue;
        }

        // Skip header-like lines: if first non-comment line looks like a header.
        if line_num == 0 {
            let first_lower = fields[0].trim_matches('"').to_lowercase();
            if first_lower == "subject" || first_lower == "s" || first_lower == "entity" {
                continue;
            }
        }

        let subject = fields[0].trim_matches('"');
        let predicate = fields[1].trim_matches('"');
        let object = fields[2].trim_matches('"');
        let confidence: f32 = fields
            .get(3)
            .and_then(|c| c.trim_matches('"').parse().ok())
            .unwrap_or(1.0);

        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            errors += 1;
            continue;
        }

        match ingest_triple(engine, subject, predicate, object, confidence) {
            Ok(sym_ids) => {
                symbols.extend(sym_ids);
                ingested += 1;
            }
            Err(_) => errors += 1,
        }
    }

    let msg = format!(
        "CSV SPO ingest: {ingested} triples ingested, {errors} errors."
    );
    Ok(ToolOutput::ok_with_symbols(msg, symbols))
}

/// Parse entity format: first column is subject, headers are predicates, cells are objects.
fn parse_entity(content: &str, delimiter: char, engine: &Engine) -> AgentResult<ToolOutput> {
    let mut lines = content.lines();

    // First non-empty, non-comment line is the header.
    let header_line = loop {
        match lines.next() {
            Some(line) => {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    break trimmed;
                }
            }
            None => {
                return Ok(ToolOutput::err(
                    "CSV entity format: no header line found.",
                ));
            }
        }
    };

    let headers: Vec<&str> = header_line.split(delimiter).map(|h| h.trim().trim_matches('"')).collect();
    if headers.len() < 2 {
        return Ok(ToolOutput::err(
            "CSV entity format requires at least 2 columns (subject + 1 predicate).",
        ));
    }

    let predicates = &headers[1..];
    let mut ingested = 0usize;
    let mut errors = 0usize;
    let mut symbols = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split(delimiter).map(|f| f.trim().trim_matches('"')).collect();
        if fields.is_empty() {
            continue;
        }

        let subject = fields[0];
        if subject.is_empty() {
            continue;
        }

        for (i, predicate) in predicates.iter().enumerate() {
            let object = fields.get(i + 1).copied().unwrap_or("");
            if object.is_empty() || predicate.is_empty() {
                continue;
            }

            match ingest_triple(engine, subject, predicate, object, 1.0) {
                Ok(sym_ids) => {
                    symbols.extend(sym_ids);
                    ingested += 1;
                }
                Err(_) => errors += 1,
            }
        }
    }

    let msg = format!(
        "CSV entity ingest: {ingested} triples ingested, {errors} errors."
    );
    Ok(ToolOutput::ok_with_symbols(msg, symbols))
}

/// Resolve labels and add a single triple to the engine.
fn ingest_triple(
    engine: &Engine,
    subject: &str,
    predicate: &str,
    object: &str,
    confidence: f32,
) -> Result<Vec<crate::symbol::SymbolId>, crate::error::AkhError> {
    let s = engine.resolve_or_create_entity(subject)?;
    let p = engine.resolve_or_create_relation(predicate)?;
    let o = engine.resolve_or_create_entity(object)?;

    let triple = crate::graph::Triple::new(s, p, o).with_confidence(confidence);
    let _ = engine.add_triple(&triple); // ignore duplicate errors

    Ok(vec![s, p, o])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn spo_format_basic() {
        let engine = test_engine();
        let csv = "Dog,is-a,Animal\nCat,is-a,Animal,0.95\n";
        let result = parse_spo(csv, ',', &engine).unwrap();
        assert!(result.success);
        assert!(result.result.contains("2 triples ingested"));
    }

    #[test]
    fn spo_format_with_header() {
        let engine = test_engine();
        let csv = "subject,predicate,object,confidence\nDog,is-a,Animal,1.0\n";
        let result = parse_spo(csv, ',', &engine).unwrap();
        assert!(result.success);
        // Header line skipped, 1 triple ingested
        assert!(result.result.contains("1 triples ingested"));
    }

    #[test]
    fn spo_format_tab_delimiter() {
        let engine = test_engine();
        let csv = "Dog\tis-a\tAnimal\nCat\tis-a\tAnimal\n";
        let result = parse_spo(csv, '\t', &engine).unwrap();
        assert!(result.success);
        assert!(result.result.contains("2 triples ingested"));
    }

    #[test]
    fn entity_format_basic() {
        let engine = test_engine();
        let csv = "entity,is-a,lives-in\nDog,Animal,House\nCat,Animal,House\n";
        let result = parse_entity(csv, ',', &engine).unwrap();
        assert!(result.success);
        // Dog is-a Animal, Dog lives-in House, Cat is-a Animal, Cat lives-in House
        assert!(result.result.contains("4 triples ingested"));
    }

    #[test]
    fn entity_format_missing_cells() {
        let engine = test_engine();
        let csv = "entity,is-a,color\nDog,Animal,\nCat,Animal,Black\n";
        let result = parse_entity(csv, ',', &engine).unwrap();
        assert!(result.success);
        // Dog is-a Animal (color empty = skip), Cat is-a Animal, Cat color Black
        assert!(result.result.contains("3 triples ingested"));
    }

    #[test]
    fn csv_tool_execute() {
        let engine = test_engine();
        let dir = tempfile::TempDir::new().unwrap();
        let csv_path = dir.path().join("test.csv");
        std::fs::write(&csv_path, "Sun,is-a,Star\nEarth,orbits,Sun\n").unwrap();

        let input = ToolInput::new()
            .with_param("path", csv_path.to_str().unwrap())
            .with_param("format", "spo");

        let tool = CsvIngestTool;
        let result = tool.execute(&engine, input).unwrap();
        assert!(result.success);
        assert!(result.result.contains("2 triples ingested"));
    }
}
