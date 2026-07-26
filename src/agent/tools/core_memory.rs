//! Self-edit memory tools for the VirtualContextManager (Phase 44i; extended
//! ADR 081 slice 7).
//!
//! These let the model manage its own context across MemGPT's tiers through the
//! ADR-062 tool loop:
//! - [`CoreMemoryAppendTool`] / [`CoreMemoryReplaceTool`] edit the always-in-
//!   context **core** tier ([`crate::agent::context::core_memory`]).
//! - [`ArchivalSearchTool`] pages relevant facts in from the long-term **archival**
//!   tier — the KG, and (when a [`NoteCatalog`] is attached) note **bodies**,
//!   which otherwise never enter the resident context (ADR 081 decision 11).
//!   This is the sole path by which a note body may appear in a turn.
//! - [`QueryGraphTool`] (ADR 081 slice 7) runs a typed KG query — list the
//!   in-force types, entities of a type, or an entity's properties — using
//!   the same [`crate::schema::digest`] / `is-a` machinery the context
//!   manager's core-tier schema digest reads.
//!
//! Each editing tool is constructed with the akh's `agent_id` baked in (the tool
//! API only passes `&Engine`), so a registered tool always edits *its own* akh's
//! core memory. They are registered only when the akh runs the `virtual` context
//! strategy — today's tier-gating mechanism (small models get no self-edit
//! tools at all; capable models run `virtual` and get the full set, `archival_search`
//! + `query_graph` included). All are `Safe`: core memory is the akh's own
//! scratchpad, archival search / graph query are read-only.

use std::collections::HashSet;
use std::sync::Arc;

use crate::agent::context::core_memory;
use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::library::note_store::NoteCatalog;

/// Append a line to the akh's core memory.
pub struct CoreMemoryAppendTool {
    agent_id: String,
}

impl CoreMemoryAppendTool {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self { agent_id: agent_id.into() }
    }
}

impl Tool for CoreMemoryAppendTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "core_memory_append".into(),
            description: "Append a durable fact to your core memory (always kept in context). \
                          Use for the user's name, standing preferences, the current task."
                .into(),
            parameters: vec![ToolParam {
                name: "text".into(),
                description: "The line to remember.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let text = input.require("text", "core_memory_append")?;
        let count = core_memory::append(engine, &self.agent_id, text).map_err(|e| {
            crate::agent::error::AgentError::ToolExecution {
                tool_name: "core_memory_append".into(),
                message: e.to_string(),
            }
        })?;
        Ok(ToolOutput::ok(format!("Core memory updated ({count} line(s)).")))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "core_memory_append".into(),
            description: "Append a durable fact to your core memory (always kept in context)."
                .into(),
            parameters: vec![ToolParamSchema::required("text", "The line to remember.")],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Writes to the akh's own core-memory scratchpad.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Replace an existing core-memory line.
pub struct CoreMemoryReplaceTool {
    agent_id: String,
}

impl CoreMemoryReplaceTool {
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self { agent_id: agent_id.into() }
    }
}

impl Tool for CoreMemoryReplaceTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "core_memory_replace".into(),
            description: "Replace a line in your core memory (e.g. when a fact changes). \
                          `old` must match an existing line exactly."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "old".into(),
                    description: "The existing line to replace (exact text).".into(),
                    required: true,
                },
                ToolParam {
                    name: "new".into(),
                    description: "The replacement text (empty to delete the line).".into(),
                    required: true,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let old = input.require("old", "core_memory_replace")?;
        let new = input.get("new").unwrap_or("");
        let replaced = core_memory::replace(engine, &self.agent_id, old, new).map_err(|e| {
            crate::agent::error::AgentError::ToolExecution {
                tool_name: "core_memory_replace".into(),
                message: e.to_string(),
            }
        })?;
        if replaced {
            Ok(ToolOutput::ok("Core memory line replaced."))
        } else {
            Ok(ToolOutput::err(format!("No core-memory line matched: {old:?}")))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "core_memory_replace".into(),
            description: "Replace a line in your core memory (exact match on `old`).".into(),
            parameters: vec![
                ToolParamSchema::required("old", "The existing line to replace (exact text)."),
                ToolParamSchema::required("new", "The replacement text (empty to delete)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::MemoryAccess]),
                description: "Edits the akh's own core-memory scratchpad.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Search the long-term archival tier — the KG, plus (when attached) note
/// bodies — for facts relevant to a query, paging them into context on demand.
///
/// This is the **only** place a note body may enter the assembled context
/// (ADR 081 decision 11 — bodies live only in archival, never resident by
/// default). Without an attached [`NoteCatalog`] this behaves exactly as
/// before: KG-only search. Note search is a **pre-Heka fallback**
/// (`crate::library::note_store` module docs): case-insensitive substring
/// matching over title/frontmatter/body — swap in a Heka-backed
/// [`NoteCatalog`] once Phase 42 lands and this tool needs no changes.
#[derive(Default)]
pub struct ArchivalSearchTool {
    notes: Option<Arc<dyn NoteCatalog>>,
}

impl ArchivalSearchTool {
    /// KG-only archival search (today's default when no note catalog exists).
    pub fn new() -> Self {
        Self::default()
    }

    /// Archival search that also pages note bodies in on demand.
    pub fn with_notes(notes: Arc<dyn NoteCatalog>) -> Self {
        Self { notes: Some(notes) }
    }
}

/// Cap on note bodies returned per search — an archival page-in should stay
/// bounded, mirroring `agent::memory`'s `TURN_BODY_SUMMARY_CAP` precedent for
/// paged-in bodies.
const NOTE_BODY_RESULT_CAP: usize = 3;

impl Tool for ArchivalSearchTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "archival_search".into(),
            description: "Search your long-term archival memory (knowledge graph and note \
                          bodies) for facts relevant to a query, paging them into context on \
                          demand."
                .into(),
            parameters: vec![ToolParam {
                name: "query".into(),
                description: "What to look up.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let query = input.require("query", "archival_search")?;
        let mut cache = crate::nlu::kg_context::KgContextCache::default();
        let kg_block = crate::nlu::kg_context::build_kg_context(engine, query, &mut cache);

        let note_block = self.notes.as_ref().and_then(|catalog| {
            let hits = catalog.search(query, NOTE_BODY_RESULT_CAP);
            if hits.is_empty() {
                return None;
            }
            let mut out = String::from("Notes:\n");
            for note in &hits {
                out.push_str(&format!("### {} (note:{})\n{}\n\n", note.document.title, note.id, note.body));
            }
            Some(out)
        });

        let mut parts = Vec::new();
        if let Some(kg) = kg_block {
            parts.push(kg);
        }
        if let Some(notes) = note_block {
            parts.push(notes);
        }
        if parts.is_empty() {
            Ok(ToolOutput::ok("(no archival matches)"))
        } else {
            Ok(ToolOutput::ok(parts.join("\n")))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "archival_search".into(),
            description: "Search long-term archival memory (the knowledge graph and note bodies) \
                          for relevant facts."
                .into(),
            parameters: vec![ToolParamSchema::required("query", "What to look up.")],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg, Capability::MemoryAccess]),
                description: "Read-only knowledge-graph and note search — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Run a typed KG query against the akh's in-force schema (ADR 081 slice 7):
/// list the installed types/relations, list entities of a declared type, or
/// list an entity's properties. Reuses the same `is-a` / [`crate::schema::digest`]
/// machinery the context manager's core-tier schema digest reads, so what this
/// tool reports always matches what the model was told is in force.
///
/// Registered alongside the other self-edit tools only for capable models
/// (the `virtual` context strategy) — a small append-only model gets no query
/// tool and must rely on whatever KG context/recall is already assembled for it.
pub struct QueryGraphTool;

impl Tool for QueryGraphTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "query_graph".into(),
            description: "Run a typed query against your knowledge graph's in-force schema: \
                          list installed types ('types'), entities of a type \
                          ('entities_of_type' + 'type'), or an entity's properties \
                          ('properties' + 'entity')."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "mode".into(),
                    description: "'types' | 'entities_of_type' | 'properties'.".into(),
                    required: true,
                },
                ToolParam {
                    name: "type".into(),
                    description: "Type name (required for 'entities_of_type').".into(),
                    required: false,
                },
                ToolParam {
                    name: "entity".into(),
                    description: "Entity label (required for 'properties').".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let mode = input.require("mode", "query_graph")?;
        match mode {
            "types" => {
                let digest = crate::schema::digest(engine).map_err(|e| {
                    crate::agent::error::AgentError::ToolExecution {
                        tool_name: "query_graph".into(),
                        message: e.to_string(),
                    }
                })?;
                if digest.is_empty() {
                    Ok(ToolOutput::ok("(no schema pack installed)"))
                } else {
                    Ok(ToolOutput::ok(digest.render_compact()))
                }
            }
            "entities_of_type" => {
                let type_name = input.require("type", "query_graph")?;
                let Ok(is_a) = engine.resolve_symbol(crate::schema::compile::PRED_IS_A) else {
                    return Ok(ToolOutput::ok("(no schema pack installed)"));
                };
                let Ok(type_symbol) = engine.resolve_symbol(type_name) else {
                    return Ok(ToolOutput::ok(format!("(no such type: {type_name})")));
                };
                let entities: Vec<String> = engine
                    .triples_to(type_symbol)
                    .into_iter()
                    .filter(|t| t.predicate == is_a)
                    .map(|t| engine.resolve_label(t.subject))
                    .collect();
                if entities.is_empty() {
                    Ok(ToolOutput::ok(format!("(no entities of type {type_name})")))
                } else {
                    Ok(ToolOutput::ok(format!(
                        "Entities of type {type_name}:\n- {}",
                        entities.join("\n- ")
                    )))
                }
            }
            "properties" => {
                let entity_label = input.require("entity", "query_graph")?;
                let Ok(entity) = engine.resolve_symbol(entity_label) else {
                    return Ok(ToolOutput::ok(format!("(no such entity: {entity_label})")));
                };
                let lines: Vec<String> = engine
                    .triples_from(entity)
                    .into_iter()
                    .map(|t| {
                        format!(
                            "{}: {}",
                            engine.resolve_label(t.predicate),
                            engine.resolve_label(t.object)
                        )
                    })
                    .collect();
                if lines.is_empty() {
                    Ok(ToolOutput::ok(format!("(no properties for {entity_label})")))
                } else {
                    Ok(ToolOutput::ok(format!(
                        "Properties of {entity_label}:\n{}",
                        lines.join("\n")
                    )))
                }
            }
            other => Ok(ToolOutput::err(format!(
                "unknown mode {other:?} — expected 'types', 'entities_of_type', or 'properties'"
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "query_graph".into(),
            description: "Run a typed query against the in-force schema (types / entities of a \
                          type / an entity's properties)."
                .into(),
            parameters: vec![
                ToolParamSchema::required("mode", "'types' | 'entities_of_type' | 'properties'."),
                ToolParamSchema::optional("type", "Type name (for 'entities_of_type')."),
                ToolParamSchema::optional("entity", "Entity label (for 'properties')."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only typed knowledge-graph query — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let e = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (e, tmp)
    }

    #[test]
    fn append_then_read_back_via_store() {
        let (e, _t) = engine();
        let tool = CoreMemoryAppendTool::new("akh");
        let out = tool
            .execute(&e, ToolInput::new().with_param("text", "user likes Rust"))
            .unwrap();
        assert!(out.success);
        assert_eq!(core_memory::read(&e, "akh"), vec!["user likes Rust"]);
    }

    #[test]
    fn replace_reports_miss() {
        let (e, _t) = engine();
        let tool = CoreMemoryReplaceTool::new("akh");
        let out = tool
            .execute(
                &e,
                ToolInput::new().with_param("old", "nope").with_param("new", "x"),
            )
            .unwrap();
        assert!(!out.success, "replacing a non-existent line fails cleanly");
    }

    // -- ADR 081 slice 7: archival_search pages note bodies on demand ---------

    fn note(title: &str, body: &str) -> crate::library::note::Note {
        use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
        let doc = DocumentRecord {
            id: title.to_lowercase(),
            title: title.to_string(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 0,
        };
        crate::library::note::Note::new(doc, body)
    }

    #[test]
    fn archival_search_without_notes_stays_kg_only() {
        let (e, _t) = engine();
        let tool = ArchivalSearchTool::new();
        let out = tool.execute(&e, ToolInput::new().with_param("query", "Paris")).unwrap();
        assert!(out.success);
        assert_eq!(out.result, "(no archival matches)");
    }

    #[test]
    fn archival_search_with_notes_returns_the_body_on_demand() {
        use crate::library::note_store::InMemoryNoteCatalog;

        let (e, _t) = engine();
        let catalog = InMemoryNoteCatalog::new();
        catalog.insert(note("Paris", "Paris is the capital of France and has the Eiffel Tower."));
        let tool = ArchivalSearchTool::with_notes(std::sync::Arc::new(catalog));

        let out = tool.execute(&e, ToolInput::new().with_param("query", "Eiffel")).unwrap();
        assert!(out.success);
        assert!(
            out.result.contains("Eiffel Tower"),
            "the body pages in on explicit search: {}",
            out.result
        );
    }

    #[test]
    fn archival_search_with_notes_no_match_reports_no_matches() {
        use crate::library::note_store::InMemoryNoteCatalog;

        let (e, _t) = engine();
        let catalog = InMemoryNoteCatalog::new();
        catalog.insert(note("Paris", "Paris is the capital of France."));
        let tool = ArchivalSearchTool::with_notes(std::sync::Arc::new(catalog));

        let out = tool
            .execute(&e, ToolInput::new().with_param("query", "nonexistent-term"))
            .unwrap();
        assert_eq!(out.result, "(no archival matches)");
    }

    // -- ADR 081 slice 7: query_graph runs typed KG queries --------------------

    const SEED_PACK: &str = r#"{
        "pack": {"name": "query-graph-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "email", "value_type": "text"}]}
        ]
    }"#;

    fn install_seed(engine: &Engine) {
        let pack = crate::schema::compile::compile(SEED_PACK).unwrap();
        crate::schema::install::install_pack(engine, &pack).unwrap();
    }

    #[test]
    fn query_graph_types_lists_installed_schema() {
        let (e, _t) = engine();
        install_seed(&e);
        let out = QueryGraphTool
            .execute(&e, ToolInput::new().with_param("mode", "types"))
            .unwrap();
        assert!(out.success);
        assert!(out.result.contains("Person"), "{}", out.result);
    }

    #[test]
    fn query_graph_types_with_no_schema_reports_absence() {
        let (e, _t) = engine();
        let out = QueryGraphTool
            .execute(&e, ToolInput::new().with_param("mode", "types"))
            .unwrap();
        assert_eq!(out.result, "(no schema pack installed)");
    }

    #[test]
    fn query_graph_entities_of_type_lists_typed_entities() {
        let (e, _t) = engine();
        install_seed(&e);
        let jane = e.resolve_or_create_entity("Jane").unwrap();
        let person = e.resolve_symbol("Person").unwrap();
        let is_a = e.resolve_or_create_relation(crate::schema::compile::PRED_IS_A).unwrap();
        e.add_triple(&crate::graph::Triple::new(jane, is_a, person)).unwrap();

        let out = QueryGraphTool
            .execute(
                &e,
                ToolInput::new().with_param("mode", "entities_of_type").with_param("type", "Person"),
            )
            .unwrap();
        assert!(out.result.contains("Jane"), "{}", out.result);
    }

    #[test]
    fn query_graph_properties_lists_an_entitys_triples() {
        let (e, _t) = engine();
        install_seed(&e);
        let jane = e.resolve_or_create_entity("Jane").unwrap();
        let email_pred = e.resolve_or_create_relation("email").unwrap();
        let addr = e.resolve_or_create_entity("jane@example.org").unwrap();
        e.add_triple(&crate::graph::Triple::new(jane, email_pred, addr)).unwrap();

        let out = QueryGraphTool
            .execute(&e, ToolInput::new().with_param("mode", "properties").with_param("entity", "Jane"))
            .unwrap();
        assert!(out.result.contains("jane@example.org"), "{}", out.result);
    }

    #[test]
    fn query_graph_unknown_mode_fails_cleanly() {
        let (e, _t) = engine();
        let out = QueryGraphTool
            .execute(&e, ToolInput::new().with_param("mode", "nonsense"))
            .unwrap();
        assert!(!out.success);
    }
}
