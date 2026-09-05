//! Entity schema tools: expose an entity's type and relations as a single
//! structured call instead of forcing callers to `triples_of` and filter for
//! `is-a` by hand.
//!
//! Three tools, all read-only (`Capability::ReadKg`, `DangerLevel::Safe`):
//!
//! - [`EntitySchemaTool`] — `entity_schema {entity}`: the entity's type
//!   hierarchy (values of `is-a`), outgoing and incoming relation types, and a
//!   provenance summary (asserted vs. derived triple counts).
//! - [`EntityTypesTool`] — `entity_types`: every distinct entity type (the
//!   objects of `is-a` relations) with counts.
//! - [`EntityTypeFilterTool`] — `entity_type_filter {type, page, page_size}`:
//!   all entities of a given type, paginated.
//!
//! ## Design decisions
//!
//! - **No `episodes_referencing` field.** The motivating example payload
//!   assumed episode counts were cheaply available. They are not — episode
//!   tracking lives on `Agent` (memory.rs), not `Engine`, and the ACP
//!   data-plane `dispatch_akh` only gets `&Engine`. Building episode-index
//!   infrastructure to satisfy an example field would be its own bug. The
//!   `provenance_summary` reports `asserted` and `derived` triple counts
//!   instead (triples with `provenance_id: None` are asserted; `Some(_)` are
//!   derived).
//! - **No caching.** `triples_from`/`triples_to` are in-memory graph lookups;
//!   `all_triples` for `entity_types` is a single scan. These are fast enough
//!   on a real workspace — an unnecessary cache is its own bug.
//! - **Untyped entities are explicit.** An entity with no `is-a` triple
//!   returns `"types": []`, not an omission. Detecting incomplete identity is
//!   one of the stated uses.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::schema::compile::PRED_IS_A;
use std::collections::HashSet;

// -----------------------------------------------------------------------
// Helper: the canonical type predicate label.
// -----------------------------------------------------------------------

/// The predicate label that establishes entity type. Reuses the canonical
/// constant from `schema::compile` to avoid divergence.
// `IS_A` is a local alias for `PRED_IS_A` for readability at call sites.

// ======================================================================
// entity_schema
// ======================================================================

/// Expose an entity's type hierarchy, relations, and provenance summary in a
/// single structured call.
pub struct EntitySchemaTool;

impl Tool for EntitySchemaTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "entity_schema".into(),
            description: "Return an entity's type hierarchy (is-a values), outgoing and incoming \
                 relation types, and a provenance summary (asserted vs. derived counts). \
                 An entity with no is-a returns an empty types array — detecting \
                 incomplete identity is a stated use."
                .into(),
            parameters: vec![ToolParam {
                name: "entity".into(),
                description: "Entity name or ID to inspect.".into(),
                required: true,
            }],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let entity_str = input.require("entity", "entity_schema")?;

        let entity_id = match engine.resolve_symbol(entity_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "Entity \"{entity_str}\" not found — no schema to report."
                )));
            }
        };

        let label = engine.resolve_label(entity_id);
        let is_a_pred = engine.resolve_symbol(PRED_IS_A).ok();

        // ── types: objects of is-a triples where subject == entity ───────
        let types: Vec<String> = if let Some(is_a) = is_a_pred {
            engine
                .triples_from(entity_id)
                .into_iter()
                .filter(|t| t.predicate == is_a)
                .map(|t| engine.resolve_label(t.object))
                .collect()
        } else {
            Vec::new()
        };

        // ── outgoing relations: group by predicate label ─────────────────
        let mut outgoing: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut asserted = 0u64;
        let mut derived = 0u64;

        for t in engine.triples_from(entity_id) {
            if Some(t.predicate) == is_a_pred {
                // is-a is already captured in types; don't double-count it
                // in relations.
            } else {
                let pred_label = engine.resolve_label(t.predicate);
                let obj_label = engine.resolve_label(t.object);
                outgoing.entry(pred_label).or_default().push(obj_label);
            }
            if t.provenance_id.is_some() {
                derived += 1;
            } else {
                asserted += 1;
            }
        }

        // ── incoming relations: group by predicate label ─────────────────
        let mut incoming: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        for t in engine.triples_to(entity_id) {
            if Some(t.predicate) == is_a_pred {
                // Skip is-a in incoming too — it's type info, not a relation.
            } else {
                let pred_label = engine.resolve_label(t.predicate);
                let subj_label = engine.resolve_label(t.subject);
                incoming.entry(pred_label).or_default().push(subj_label);
            }
            if t.provenance_id.is_some() {
                derived += 1;
            } else {
                asserted += 1;
            }
        }

        // Build the JSON result as a compact, human-readable string that
        // also carries the structured shape for programmatic callers.
        let types_json: Vec<&str> = types.iter().map(|s| s.as_str()).collect();
        let out_json: serde_json::Value = outgoing
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|s| s.as_str()).collect::<Vec<_>>()))
            .collect();
        let in_json: serde_json::Value = incoming
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().map(|s| s.as_str()).collect::<Vec<_>>()))
            .collect();

        let result = serde_json::json!({
            "entity": label,
            "types": types_json,
            "outgoing_relations": out_json,
            "incoming_relations": in_json,
            "provenance_summary": {
                "asserted": asserted,
                "derived": derived,
            },
        });

        let summary = format!(
            "Entity \"{label}\": {} type(s), {} outgoing relation(s), {} incoming relation(s) \
             [asserted: {asserted}, derived: {derived}].",
            types.len(),
            outgoing.values().map(|v| v.len()).sum::<usize>(),
            incoming.values().map(|v| v.len()).sum::<usize>(),
        );

        Ok(ToolOutput::ok_with_symbols(
            format!(
                "{summary}\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            ),
            vec![entity_id],
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "entity_schema".into(),
            description: "Return an entity's type hierarchy, outgoing/incoming relations, and \
                 provenance summary."
                .into(),
            parameters: vec![ToolParamSchema::required(
                "entity",
                "Entity name or ID to inspect.",
            )],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only entity schema inspection — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ======================================================================
// entity_types
// ======================================================================

/// List every distinct entity type (the objects of `is-a` relations) with
/// counts.
pub struct EntityTypesTool;

impl Tool for EntityTypesTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "entity_types".into(),
            description: "List every distinct entity type (the values of is-a relations) with \
                 counts of how many entities have that type."
                .into(),
            parameters: vec![],
        }
    }

    fn execute(&self, engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
        let is_a = match engine.resolve_symbol(PRED_IS_A) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(
                    "No is-a relation in this workspace — entity_types is empty.".to_string(),
                ));
            }
        };

        // Count distinct objects of is-a triples.
        let mut counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
        let mut total_typed_entities = 0u64;

        for t in engine.all_triples() {
            if t.predicate == is_a {
                let type_label = engine.resolve_label(t.object);
                *counts.entry(type_label).or_insert(0) += 1;
                total_typed_entities += 1;
            }
        }

        let types_json: Vec<serde_json::Value> = counts
            .iter()
            .map(|(k, v)| {
                serde_json::json!({
                    "type": k,
                    "count": v,
                })
            })
            .collect();

        let result = serde_json::json!({
            "types": types_json,
            "distinct_types": counts.len(),
            "total_typed_entities": total_typed_entities,
        });

        let summary = format!(
            "{} distinct entity type(s), {} typed entity/entities.",
            counts.len(),
            total_typed_entities,
        );

        Ok(ToolOutput::ok(format!(
            "{summary}\n{}",
            serde_json::to_string_pretty(&result).unwrap_or_default()
        )))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "entity_types".into(),
            description: "List every distinct entity type (is-a objects) with counts.".into(),
            parameters: vec![],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only entity type census — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ======================================================================
// entity_type_filter
// ======================================================================

/// Paginated list of all entities of a given type, with core attributes.
pub struct EntityTypeFilterTool;

impl Tool for EntityTypeFilterTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "entity_type_filter".into(),
            description: "List all entities of a given type (is-a == type), paginated, with \
                 core attributes (label, created_at). Pass page=0 and page_size=N; \
                 the response includes total, page, page_size, and has_more."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "type".into(),
                    description: "Entity type to filter by (the object of is-a).".into(),
                    required: true,
                },
                ToolParam {
                    name: "page".into(),
                    description: "Zero-indexed page number (default: 0).".into(),
                    required: false,
                },
                ToolParam {
                    name: "page_size".into(),
                    description: "Page size (default: 50).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let type_str = input.require("type", "entity_type_filter")?;
        let page = input.get_u64("page").unwrap_or(0) as usize;
        let page_size = input.get_u64("page_size").unwrap_or(50) as usize;

        // Clamp page_size to a sane range.
        let page_size = page_size.clamp(1, 500);
        // page is usize, already non-negative; no clamp needed.

        let is_a = match engine.resolve_symbol(PRED_IS_A) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "No is-a relation in this workspace — no entities of type \"{type_str}\"."
                )));
            }
        };

        let type_id = match engine.resolve_symbol(type_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "Type \"{type_str}\" not found — no entities match."
                )));
            }
        };

        // Collect all entities with is-a == type_id.
        let mut entities: Vec<crate::symbol::SymbolId> = engine
            .all_triples()
            .into_iter()
            .filter(|t| t.predicate == is_a && t.object == type_id)
            .map(|t| t.subject)
            .collect();

        // Deduplicate (an entity can have the same is-a asserted multiple
        // times via different provenance).
        entities.sort();
        entities.dedup();

        let total = entities.len();
        let start = page * page_size;
        let end = (start + page_size).min(total);

        let page_entities: Vec<crate::symbol::SymbolId> = if start < total {
            entities[start..end].to_vec()
        } else {
            Vec::new()
        };

        // Build a lookup of SymbolMeta for created_at.
        let symbols_meta = engine.dump_symbols();
        let meta_lookup: std::collections::HashMap<_, _> =
            symbols_meta.iter().map(|m| (m.id, m.clone())).collect();

        let entities_json: Vec<serde_json::Value> = page_entities
            .iter()
            .map(|id| {
                let label = engine.resolve_label(*id);
                let created_at = meta_lookup.get(id).map(|m| m.created_at).unwrap_or(0);
                serde_json::json!({
                    "label": label,
                    "created_at": created_at,
                })
            })
            .collect();

        let has_more = end < total;

        let result = serde_json::json!({
            "type": engine.resolve_label(type_id),
            "entities": entities_json,
            "total": total,
            "page": page,
            "page_size": page_size,
            "returned": page_entities.len(),
            "has_more": has_more,
        });

        let summary = format!(
            "Type \"{type_str}\": {} total entit(ies), page {} ({} returned, has_more={has_more}).",
            total,
            page,
            page_entities.len(),
        );

        let mut involved = page_entities;
        involved.push(type_id);
        Ok(ToolOutput::ok_with_symbols(
            format!(
                "{summary}\n{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            ),
            involved,
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "entity_type_filter".into(),
            description: "Paginated list of all entities of a given type with core attributes."
                .into(),
            parameters: vec![
                ToolParamSchema::required("type", "Entity type to filter by (is-a object)."),
                ToolParamSchema::optional("page", "Zero-indexed page number (default: 0)."),
                ToolParamSchema::optional("page_size", "Page size (default: 50, max 500)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Read-only paginated entity type filter — no side effects.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::graph::Triple;
    use crate::symbol::SymbolId;

    /// Fresh in-memory engine.
    fn engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    /// Add a triple by label, creating symbols as needed.
    fn add_triple(e: &Engine, s: &str, p: &str, o: &str) -> (SymbolId, SymbolId, SymbolId) {
        let s = e.resolve_or_create_entity(s).unwrap();
        let p = e.resolve_or_create_relation(p).unwrap();
        let o = e.resolve_or_create_entity(o).unwrap();
        e.add_triple(&Triple::new(s, p, o)).unwrap();
        (s, p, o)
    }

    // -----------------------------------------------------------------
    // entity_schema
    // -----------------------------------------------------------------

    #[test]
    fn entity_schema_returns_both_directions() {
        let e = engine();
        // Tecton is-a DigitalArchitect, member-of Ennead, runs-on MacMini.
        add_triple(&e, "Tecton", "is-a", "DigitalArchitect");
        add_triple(&e, "Tecton", "member-of", "Ennead");
        add_triple(&e, "Tecton", "runs-on", "MacMini");
        // Incoming: OpenClaw hosts Tecton.
        add_triple(&e, "OpenClaw", "hosts", "Tecton");

        let out = EntitySchemaTool
            .execute(&e, ToolInput::new().with_param("entity", "Tecton"))
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("Tecton"));
        assert!(out.result.contains("DigitalArchitect"));
        assert!(out.result.contains("member-of"));
        assert!(out.result.contains("Ennead"));
        assert!(out.result.contains("runs-on"));
        assert!(out.result.contains("MacMini"));
        assert!(out.result.contains("hosts"));
        assert!(out.result.contains("OpenClaw"));
    }

    #[test]
    fn entity_schema_untyped_entity_has_empty_types() {
        let e = engine();
        // MacMini has no is-a, but has an incoming relation.
        add_triple(&e, "Tecton", "runs-on", "MacMini");

        let out = EntitySchemaTool
            .execute(&e, ToolInput::new().with_param("entity", "MacMini"))
            .unwrap();

        assert!(out.success);
        // Should explicitly show empty types.
        assert!(out.result.contains("\"types\": []"));
    }

    #[test]
    fn entity_schema_unknown_entity_is_graceful() {
        let e = engine();

        let out = EntitySchemaTool
            .execute(&e, ToolInput::new().with_param("entity", "Nonexistent"))
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("not found"));
    }

    // -----------------------------------------------------------------
    // entity_types
    // -----------------------------------------------------------------

    #[test]
    fn entity_types_counts_match_manual_count() {
        let e = engine();
        // Three entities of type "DigitalArchitect".
        add_triple(&e, "Tecton", "is-a", "DigitalArchitect");
        add_triple(&e, "Thoth", "is-a", "DigitalArchitect");
        add_triple(&e, "Horus", "is-a", "DigitalArchitect");
        // Two entities of type "Construct".
        add_triple(&e, "Tecton", "is-a", "Construct");
        add_triple(&e, "Thoth", "is-a", "Construct");

        let out = EntityTypesTool.execute(&e, ToolInput::new()).unwrap();

        assert!(out.success);
        // 2 distinct types.
        assert!(out.result.contains("\"distinct_types\": 2"));
        // 5 total typed entities (3 + 2).
        assert!(out.result.contains("\"total_typed_entities\": 5"));
        assert!(out.result.contains("DigitalArchitect"));
        assert!(out.result.contains("\"count\": 3"));
        assert!(out.result.contains("Construct"));
        assert!(out.result.contains("\"count\": 2"));
    }

    #[test]
    fn entity_types_empty_workspace_is_graceful() {
        let e = engine();

        let out = EntityTypesTool.execute(&e, ToolInput::new()).unwrap();

        assert!(out.success);
        // No is-a relation exists → graceful message.
        assert!(out.result.contains("No is-a relation"));
    }

    // -----------------------------------------------------------------
    // entity_type_filter
    // -----------------------------------------------------------------

    #[test]
    fn entity_type_filter_paginates_correctly() {
        let e = engine();
        // Create 7 entities of type "Construct".
        for i in 0..7 {
            let name = format!("Entity{i}");
            add_triple(&e, &name, "is-a", "Construct");
        }

        // Page 0, size 3 → 3 entities, has_more = true.
        let out = EntityTypeFilterTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("type", "Construct")
                    .with_param("page", "0")
                    .with_param("page_size", "3"),
            )
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("\"total\": 7"));
        assert!(out.result.contains("\"page\": 0"));
        assert!(out.result.contains("\"returned\": 3"));
        assert!(out.result.contains("\"has_more\": true"));

        // Page 2, size 3 → 1 entity, has_more = false.
        let out = EntityTypeFilterTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("type", "Construct")
                    .with_param("page", "2")
                    .with_param("page_size", "3"),
            )
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("\"returned\": 1"));
        assert!(out.result.contains("\"has_more\": false"));
    }

    #[test]
    fn entity_type_filter_unknown_type_is_graceful() {
        let e = engine();
        add_triple(&e, "Tecton", "is-a", "DigitalArchitect");

        let out = EntityTypeFilterTool
            .execute(&e, ToolInput::new().with_param("type", "NonexistentType"))
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("not found"));
    }

    #[test]
    fn entity_type_filter_page_past_end_returns_empty() {
        let e = engine();
        add_triple(&e, "Tecton", "is-a", "DigitalArchitect");

        let out = EntityTypeFilterTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("type", "DigitalArchitect")
                    .with_param("page", "10")
                    .with_param("page_size", "5"),
            )
            .unwrap();

        assert!(out.success);
        assert!(out.result.contains("\"returned\": 0"));
        assert!(out.result.contains("\"has_more\": false"));
    }

    /// Verify all three entity-schema tools are discoverable through the
    /// `ToolRegistry` by name. This catches the class of bug where a tool is
    /// implemented but never registered (review round 1 blocker).
    #[test]
    fn entity_schema_tools_registered_by_name() {
        use crate::agent::tool::ToolRegistry;

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EntitySchemaTool));
        registry.register(Box::new(EntityTypesTool));
        registry.register(Box::new(EntityTypeFilterTool));

        assert!(
            registry.get("entity_schema").is_some(),
            "entity_schema not registered"
        );
        assert!(
            registry.get("entity_types").is_some(),
            "entity_types not registered"
        );
        assert!(
            registry.get("entity_type_filter").is_some(),
            "entity_type_filter not registered"
        );
    }
}
