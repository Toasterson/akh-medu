//! KG delete tool: remove a triple from the knowledge graph.
//!
//! Companion to [`KgMutateTool`](super::kg_mutate::KgMutateTool): where
//! `kg_mutate` adds a triple, `kg_delete` retracts one. Without this, stale,
//! incorrect, or contaminated triples accumulate forever — agents can assert
//! but never retract.
//!
//! Semantics:
//! - Resolves `subject`/`predicate`/`object` by name or numeric ID, exactly
//!   like `kg_mutate`. Unlike `kg_mutate`, symbols are *looked up*, not created
//!   — deleting a triple whose symbols do not exist is a no-op, not an
//!   auto-creation.
//! - Delegates to [`Engine::remove_triple`], which retracts the triple from
//!   the KG, syncs the SPARQL store, and (when TMS is enabled) cascades the
//!   retraction through dependent support sets.
//! - Returns the count of triples removed (0 if the triple was not present
//!   or its symbols were unknown).
//! - The tool registry already audit-logs every tool invocation
//!   ([`ToolRegistry::execute`]); no extra ledger wiring is needed here.

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Remove a triple from the knowledge graph.
pub struct KgDeleteTool;

impl Tool for KgDeleteTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "kg_delete".into(),
            description:
                "Delete a triple (subject, predicate, object) from the knowledge graph. \
                 Returns the number of triples removed. Symbols are looked up, not \
                 created — deleting a non-existent triple is a no-op."
                    .into(),
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
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let subject_str = input.require("subject", "kg_delete")?;
        let predicate_str = input.require("predicate", "kg_delete")?;
        let object_str = input.require("object", "kg_delete")?;

        // Look up symbols WITHOUT auto-creating. A delete against a symbol that
        // does not exist cannot match any triple, so it is reported as 0 removed
        // rather than fabricating the very symbols the caller wanted to retract.
        let s = match engine.resolve_symbol(subject_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "Deleted 0 triple(s) — subject \"{subject_str}\" not found."
                )));
            }
        };
        let p = match engine.resolve_symbol(predicate_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "Deleted 0 triple(s) — predicate \"{predicate_str}\" not found."
                )));
            }
        };
        let o = match engine.resolve_symbol(object_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::ok(format!(
                    "Deleted 0 triple(s) — object \"{object_str}\" not found."
                )));
            }
        };

        let had_triple = engine.has_triple(s, p, o);
        let result = engine.remove_triple(s, p, o)?;

        let removed = if had_triple { 1 } else { 0 };

        // Count cascaded retractions (TMS support-set dependents) beyond the
        // directly removed triple's object.
        let cascaded = if result.retracted.len() > 1 {
            result.retracted.len() - 1
        } else {
            0
        };

        let label_s = engine.resolve_label(s);
        let label_p = engine.resolve_label(p);
        let label_o = engine.resolve_label(o);

        let summary = if removed == 0 {
            format!(
                "Deleted 0 triple(s) — \"{}\" -> {} -> \"{}\" was not present.",
                label_s, label_p, label_o,
            )
        } else if cascaded > 0 {
            format!(
                "Deleted 1 triple(s) — \"{}\" -> {} -> \"{}\" — plus {cascaded} \
                 cascaded retraction(s) from TMS support sets (depth {}).",
                label_s,
                label_p,
                label_o,
                result.cascade_depth,
            )
        } else {
            format!(
                "Deleted 1 triple(s) — \"{}\" -> {} -> \"{}\".",
                label_s, label_p, label_o,
            )
        };

        Ok(ToolOutput::ok_with_symbols(summary, vec![s, p, o]))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "kg_delete".into(),
            description:
                "Delete a triple (subject, predicate, object) from the knowledge graph. \
                 Returns the number of triples removed."
                    .into(),
            parameters: vec![
                ToolParamSchema::required("subject", "Subject symbol name or ID."),
                ToolParamSchema::required("predicate", "Predicate (relation) symbol name or ID."),
                ToolParamSchema::required("object", "Object symbol name or ID."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::WriteKg]),
                description:
                    "Destructively retracts a triple from the knowledge graph. When TMS \
                     is enabled, retraction cascades through dependent support sets."
                        .into(),
                shadow_triggers: vec!["delete triple".into(), "retract".into(), "remove".into()],
            },
            source: ToolSource::Native,
        }
    }
}
