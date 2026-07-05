//! Typed episodic-memory writer.
//!
//! The strongly-typed replacement for `journal_ingest` inside an LLM session:
//! the model reads source text itself, reasons, and emits a structured episode
//! (summary + tags + extracted facts) which this tool persists deterministically
//! — no internal NLU/LLM pass. Episodes are structurally identical to those made
//! by [`crate::agent::memory::index_cycle_episode`], so both `recall_episodes`
//! and `recall_recent_episodes` find them.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

use crate::agent::agent::AgentPredicates;
use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

/// Write one typed episodic memory entry to the knowledge graph.
pub struct MemoryWriteTool {
    predicates: AgentPredicates,
}

impl MemoryWriteTool {
    pub fn new(predicates: AgentPredicates) -> Self {
        Self { predicates }
    }
}

/// JSON-Schema fragment for the `facts` array: `[{subject, predicate, object, confidence?}]`.
fn facts_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "description": "Facts distilled from the source, each a (subject, predicate, object) triple.",
        "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject entity label."},
                "predicate": {"type": "string", "description": "Relation label."},
                "object": {"type": "string", "description": "Object entity label."},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0 (default 1.0)."}
            },
            "required": ["subject", "predicate", "object"]
        }
    })
}

/// JSON-Schema fragment for the `tags` array.
fn tags_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "description": "Short topical tags for this memory.",
        "items": {"type": "string"}
    })
}

impl Tool for MemoryWriteTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "memory_write".into(),
            description: "Record one episodic memory: a summary plus distilled facts. Use this \
                          (not a raw-text ingester) — read the source yourself, then emit the \
                          structured episode."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "summary".into(),
                    description: "One-sentence summary of what happened / was learned.".into(),
                    required: true,
                },
                ToolParam {
                    name: "facts".into(),
                    description: "Array of {subject, predicate, object, confidence?} triples.".into(),
                    required: false,
                },
                ToolParam {
                    name: "tags".into(),
                    description: "Array of short topical tags.".into(),
                    required: false,
                },
                ToolParam {
                    name: "source".into(),
                    description: "Where this memory came from (e.g. a file path). Used for \
                                  provenance and idempotency."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "origin".into(),
                    description: "\"openclaw\" to attribute as a self-import, else \"agent\" \
                                  (default)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "timestamp".into(),
                    description: "Optional unix timestamp for deterministic episode identity.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let summary = input.require("summary", "memory_write")?.to_string();
        let tags = input.get_str_list("tags");
        let source = input.get("source").unwrap_or("agent").to_string();
        let origin = input.get("origin").unwrap_or("agent").to_string();
        let timestamp = input.get_u64("timestamp");

        // Parse the typed facts array (LLM-supplied, no NLU).
        let facts = parse_facts(&input)?;

        // Deterministic, idempotent episode label: a re-run with the same source +
        // timestamp (or the same summary, when no timestamp) reuses the symbol.
        let discriminator = match timestamp {
            Some(ts) => ts.to_string(),
            None => {
                let mut h = DefaultHasher::new();
                summary.hash(&mut h);
                source.hash(&mut h);
                format!("{:x}", h.finish())
            }
        };
        let episode_label = format!("episode:{}:{}", slug(&source), discriminator);
        let episode_id = engine.resolve_or_create_entity(&episode_label)?;

        let mut symbols: Vec<SymbolId> = vec![episode_id];

        // Summary.
        let summary_label = if summary.len() > 80 {
            format!("summary:{}", &summary[..80])
        } else {
            format!("summary:{summary}")
        };
        let summary_sym = engine.resolve_or_create_entity(&summary_label)?;
        let _ = engine.add_triple(&Triple::new(episode_id, self.predicates.has_summary, summary_sym));

        // Tags (caller tags + a source tag for queryability).
        for tag in tags.iter().chain(std::iter::once(&format!("source:{source}"))) {
            let tag_sym = engine.resolve_or_create_entity(&format!("tag:{tag}"))?;
            let _ = engine.add_triple(&Triple::new(episode_id, self.predicates.has_tag, tag_sym));
        }

        // Facts — assert each triple and link the episode to the subject so
        // symbol-based recall surfaces this episode.
        for fact in &facts {
            let s = engine.resolve_or_create_entity(&fact.subject)?;
            let p = engine.resolve_or_create_relation(&fact.predicate)?;
            let o = engine.resolve_or_create_entity(&fact.object)?;
            let _ = engine.add_triple(&Triple::new(s, p, o).with_confidence(fact.confidence));
            let _ = engine.add_triple(&Triple::new(episode_id, self.predicates.learned, s));
            symbols.push(s);
        }

        // Mark as episodic memory so recall discovers it.
        let mem_type_sym = engine.resolve_or_create_entity("episodic_memory")?;
        let _ = engine.add_triple(&Triple::new(
            episode_id,
            self.predicates.memory_type,
            mem_type_sym,
        ));

        // Provenance: attribute self-imports distinctly so they're queryable/dedupable.
        let kind = if origin.eq_ignore_ascii_case("openclaw") {
            DerivationKind::OpenClawImport {
                source_path: source.clone(),
                kind: "journal".into(),
            }
        } else {
            DerivationKind::AgentConsolidation {
                reason: format!("memory_write from {source}"),
                relevance_score: 1.0,
            }
        };
        let mut prov = ProvenanceRecord::new(episode_id, kind).with_confidence(1.0);
        let _ = engine.store_provenance(&mut prov);

        Ok(ToolOutput::ok_with_symbols(
            format!(
                "Recorded episode \"{episode_label}\" ({} fact(s), {} tag(s)).",
                facts.len(),
                tags.len()
            ),
            symbols,
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "memory_write".into(),
            description: "Record one episodic memory (summary + distilled facts). The typed \
                          replacement for raw-text journal ingestion in an LLM session."
                .into(),
            parameters: vec![
                ToolParamSchema::required("summary", "One-sentence summary of what happened / was learned."),
                ToolParamSchema::optional_schema(
                    "facts",
                    "Facts distilled from the source.",
                    facts_schema(),
                ),
                ToolParamSchema::optional_schema("tags", "Short topical tags.", tags_schema()),
                ToolParamSchema::optional(
                    "source",
                    "Provenance source (e.g. a file path); also used for idempotency.",
                ),
                ToolParamSchema::optional(
                    "origin",
                    "\"openclaw\" to attribute as a self-import, else \"agent\" (default).",
                ),
                ToolParamSchema::optional(
                    "timestamp",
                    "Optional unix timestamp for deterministic episode identity.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg, Capability::MemoryAccess]),
                description: "Writes an episodic memory entry to the knowledge graph.".into(),
                shadow_triggers: vec!["remember".into(), "record memory".into()],
            },
            source: ToolSource::Native,
        }
    }
}

/// A single distilled fact.
struct Fact {
    subject: String,
    predicate: String,
    object: String,
    confidence: f32,
}

/// Parse the `facts` parameter from either a JSON array of objects or absent.
fn parse_facts(input: &ToolInput) -> AgentResult<Vec<Fact>> {
    let Some(value) = input.get_value("facts") else {
        return Ok(Vec::new());
    };
    // Tolerate a JSON string that itself contains the array (some backends
    // stringify nested args).
    let owned;
    let array = match value {
        serde_json::Value::Array(a) => a,
        serde_json::Value::String(s) => {
            owned = serde_json::from_str::<serde_json::Value>(s).map_err(|e| {
                AgentError::ToolExecution {
                    tool_name: "memory_write".into(),
                    message: format!("`facts` was a string but not valid JSON: {e}"),
                }
            })?;
            owned.as_array().ok_or_else(|| AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: "`facts` string did not decode to a JSON array".into(),
            })?
        }
        _ => {
            return Err(AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: "`facts` must be a JSON array of {subject, predicate, object}".into(),
            });
        }
    };

    let mut facts = Vec::with_capacity(array.len());
    for (i, item) in array.iter().enumerate() {
        let get = |k: &str| item.get(k).and_then(|v| v.as_str());
        let (Some(subject), Some(predicate), Some(object)) =
            (get("subject"), get("predicate"), get("object"))
        else {
            return Err(AgentError::ToolExecution {
                tool_name: "memory_write".into(),
                message: format!("fact[{i}] needs string subject, predicate, object"),
            });
        };
        let confidence = item
            .get("confidence")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);
        facts.push(Fact {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence,
        });
    }
    Ok(facts)
}

/// Lowercase a string into a compact label fragment (alphanumerics + dashes).
fn slug(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    let collapsed: String = trimmed.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    if collapsed.len() > 48 {
        collapsed[collapsed.len() - 48..].to_string()
    } else if collapsed.is_empty() {
        "agent".to_string()
    } else {
        collapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;
    use serde_json::json;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn sample_input() -> ToolInput {
        ToolInput::new()
            .with_value("summary", "Learned that Talos runs Kubernetes")
            .with_value("tags", json!(["k8s", "ops"]))
            .with_value(
                "facts",
                json!([{"subject": "talos", "predicate": "runs", "object": "kubernetes"}]),
            )
            .with_value("source", "~/.openclaw/workspace/memory/2026-05-08.md")
            .with_value("origin", "openclaw")
            .with_value("timestamp", 1_715_000_000u64)
    }

    #[test]
    fn writes_typed_episode_with_facts() {
        let engine = test_engine();
        let predicates = AgentPredicates::init(&engine).unwrap();
        let tool = MemoryWriteTool::new(predicates);

        let out = tool.execute(&engine, sample_input()).unwrap();
        assert!(out.success, "expected success, got: {}", out.result);

        // The fact's subject/object exist as entities.
        assert!(engine.resolve_symbol("talos").is_ok());
        assert!(engine.resolve_symbol("kubernetes").is_ok());
        // Episode marked episodic and discoverable by recall.
        let predicates = AgentPredicates::init(&engine).unwrap();
        let recent =
            crate::agent::memory::recall_recent_episodes(&engine, &predicates, 10).unwrap();
        assert!(!recent.is_empty(), "episode should be recallable");
    }

    #[test]
    fn re_running_same_source_is_idempotent() {
        let engine = test_engine();
        let predicates = AgentPredicates::init(&engine).unwrap();
        let tool = MemoryWriteTool::new(predicates);

        let first = tool.execute(&engine, sample_input()).unwrap();
        let second = tool.execute(&engine, sample_input()).unwrap();
        // Same deterministic episode entity reused (first symbol is the episode).
        assert_eq!(first.symbols_involved[0], second.symbols_involved[0]);
    }

    #[test]
    fn missing_summary_errors() {
        let engine = test_engine();
        let predicates = AgentPredicates::init(&engine).unwrap();
        let tool = MemoryWriteTool::new(predicates);
        let err = tool.execute(&engine, ToolInput::new());
        assert!(err.is_err());
    }
}
