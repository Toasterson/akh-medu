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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::permission::{PermissionDecision, PermissionMode, PermissionPolicy};
    use crate::engine::{Engine, EngineConfig};
    use crate::graph::Triple;
    use crate::provenance::DerivationKind;
    use crate::tms::SupportSet;

    /// Fresh engine backed by a temp directory. TMS is enabled by default.
    fn engine() -> (Engine, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let e = Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (e, tmp)
    }

    /// Helper: add a triple by label and return the resolved symbol IDs.
    fn add_triple(e: &Engine, s: &str, p: &str, o: &str) -> (SymbolId, SymbolId, SymbolId) {
        let s = e.resolve_or_create_entity(s).unwrap();
        let p = e.resolve_or_create_relation(p).unwrap();
        let o = e.resolve_or_create_entity(o).unwrap();
        e.add_triple(&Triple::new(s, p, o)).unwrap();
        (s, p, o)
    }

    // -----------------------------------------------------------------------
    // 1. Happy path: retract an existing triple
    // -----------------------------------------------------------------------

    #[test]
    fn delete_existing_triple_reports_one_removed() {
        let (e, _t) = engine();
        let (s, p, o) = add_triple(&e, "Tecton", "runs-on", "MacMini");
        assert!(e.has_triple(s, p, o));

        let out = KgDeleteTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("subject", "Tecton")
                    .with_param("predicate", "runs-on")
                    .with_param("object", "MacMini"),
            )
            .unwrap();

        assert!(out.success, "tool should succeed: {}", out.result);
        assert!(
            out.result.contains("Deleted 1 triple(s)"),
            "should report 1 removed: {}",
            out.result
        );
        assert!(!e.has_triple(s, p, o), "triple should be gone from the KG");
    }

    // -----------------------------------------------------------------------
    // 2. Absent triple: retracting something not present returns 0
    // -----------------------------------------------------------------------

    #[test]
    fn delete_absent_triple_returns_zero() {
        let (e, _t) = engine();
        // Symbols exist but the triple was never added.
        let _ = add_triple(&e, "Tecton", "runs-on", "MacMini");

        let out = KgDeleteTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("subject", "Tecton")
                    .with_param("predicate", "deity")
                    .with_param("object", "Ptah"),
            )
            .unwrap();

        assert!(out.success, "no error on absent triple: {}", out.result);
        assert!(
            out.result.contains("Deleted 0 triple(s)"),
            "should report 0 removed: {}",
            out.result
        );
    }

    // -----------------------------------------------------------------------
    // 3. Symbol resolution: name, numeric ID, and unresolvable name
    // -----------------------------------------------------------------------

    #[test]
    fn delete_by_numeric_id_works() {
        let (e, _t) = engine();
        let (s, p, o) = add_triple(&e, "Tecton", "runs-on", "MacMini");
        assert!(e.has_triple(s, p, o));

        let out = KgDeleteTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("subject", s.get().to_string())
                    .with_param("predicate", p.get().to_string())
                    .with_param("object", o.get().to_string()),
            )
            .unwrap();

        assert!(out.success, "delete by ID should succeed: {}", out.result);
        assert!(
            out.result.contains("Deleted 1 triple(s)"),
            "should report 1 removed: {}",
            out.result
        );
        assert!(!e.has_triple(s, p, o));
    }

    #[test]
    fn delete_unresolvable_subject_returns_zero() {
        let (e, _t) = engine();
        let out = KgDeleteTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("subject", "NonExistentEntity123")
                    .with_param("predicate", "is-a")
                    .with_param("object", "Thing"),
            )
            .unwrap();

        assert!(out.success);
        assert!(
            out.result.contains("Deleted 0 triple(s)"),
            "unresolvable subject → 0 removed: {}",
            out.result
        );
        assert!(
            out.result.contains("not found"),
            "should explain the symbol was not found: {}",
            out.result
        );
    }

    // -----------------------------------------------------------------------
    // 4. TMS cascade: retracting a premise cascades through dependents
    // -----------------------------------------------------------------------

    #[test]
    fn tms_cascade_removes_dependent_triples() {
        let (e, _t) = engine();

        // Base fact: (A, P, B) — B is a premise symbol.
        let (a, p, b) = add_triple(&e, "A", "P", "B");

        // Derived fact: (B, Q, C) — C is justified by premise B.
        let (b2, q, c) = add_triple(&e, "B", "Q", "C");
        assert_eq!(b, b2, "B should be the same symbol in both triples");

        e.add_triple_with_support(
            &Triple::new(b, q, c),
            SupportSet::new(vec![b], DerivationKind::Reasoned, 1.0),
        )
        .unwrap();

        // Unrelated fact: (X, R, Y) — no support dependency on B or C.
        let (x, r, y) = add_triple(&e, "X", "R", "Y");

        // Sanity: all three triples present before deletion.
        assert!(e.has_triple(a, p, b));
        assert!(e.has_triple(b, q, c));
        assert!(e.has_triple(x, r, y));

        // Retract the base fact (A, P, B). The engine removes the triple
        // from the KG, then calls tms.retract(B). B is a premise for C's
        // only support set → C loses all support → C is retracted → all
        // triples where C is subject or object are removed.
        let out = KgDeleteTool
            .execute(
                &e,
                ToolInput::new()
                    .with_param("subject", "A")
                    .with_param("predicate", "P")
                    .with_param("object", "B"),
            )
            .unwrap();

        assert!(out.success, "cascade delete should succeed: {}", out.result);
        assert!(
            out.result.contains("cascaded retraction"),
            "should report cascade: {}",
            out.result
        );

        // Collect the exact surviving triple set.
        let survivors: Vec<(String, String, String)> = e
            .all_triples()
            .iter()
            .map(|t| {
                (
                    e.resolve_label(t.subject),
                    e.resolve_label(t.predicate),
                    e.resolve_label(t.object),
                )
            })
            .collect();

        // The cascade should have removed both (A, P, B) and (B, Q, C).
        // Only the unrelated (X, R, Y) should survive.
        assert_eq!(
            survivors,
            vec![("X".to_string(), "R".to_string(), "Y".to_string())],
            "TMS cascade should remove the base fact and the dependent, \
             leaving only the unrelated triple"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Capability gating: Plan mode denies WriteKg tools
    // -----------------------------------------------------------------------

    #[test]
    fn plan_mode_denies_delete_tool() {
        let tool = KgDeleteTool;
        let manifest = tool.manifest();
        let policy = PermissionPolicy::new(PermissionMode::Plan);

        match policy.decide(&manifest) {
            PermissionDecision::Deny(msg) => {
                assert!(
                    msg.contains("plan mode") || msg.contains("read-only"),
                    "deny message should explain plan mode: {msg}"
                );
            }
            other => panic!("Plan mode should deny kg_delete, got {other:?}"),
        }
    }

    #[test]
    fn bypass_mode_allows_delete_tool() {
        let tool = KgDeleteTool;
        let manifest = tool.manifest();
        let policy = PermissionPolicy::new(PermissionMode::Bypass);

        assert_eq!(
            policy.decide(&manifest),
            PermissionDecision::Allow,
            "Bypass mode should allow kg_delete"
        );
    }
}
