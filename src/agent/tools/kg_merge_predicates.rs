//! `kg_merge_predicates`: fold alias predicate spellings onto one canonical
//! predicate (Phase 53j, ADR 113 memory hygiene).
//!
//! Every alias triple is re-asserted under the canonical predicate (keeping
//! confidence, timestamp, compartment and provenance link) and the alias
//! edge removed, with a `memory_hygiene` provenance record per move
//! ([`crate::schema::execute_merge`]). When the canonical predicate is
//! declared `single_valued`, the newest value across every spelling stays
//! live and the older ones are superseded.
//!
//! Modes:
//! - `explicit` (default): `aliases` onto `canonical`.
//! - `declared`: every alias declared by an installed schema pack.
//! - `collisions`: every exact structural spelling collision (what the
//!   daemon's hygiene pass merges on its own).
//!
//! It rewrites many triples at once, so it is `Dangerous`: plan mode denies
//! it and `/v1` scopes never expose it unless a scope or grant names it.

use std::collections::HashSet;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::schema::{DEFAULT_MERGE_LIMIT, MergeRequest, MergeSelection, execute_merge};

/// Tool name.
pub const KG_MERGE_PREDICATES: &str = "kg_merge_predicates";

/// Merge alias predicate spellings onto a canonical predicate.
pub struct KgMergePredicatesTool;

/// Build a [`MergeRequest`] from tool input. `Err` carries the message for
/// a failed [`ToolOutput`].
fn request_from(input: &ToolInput) -> Result<MergeRequest, String> {
    let dry_run = input.get_bool("dry_run").unwrap_or(false);
    let limit = input
        .get_u64("limit")
        .map(|l| usize::try_from(l).unwrap_or(usize::MAX))
        .unwrap_or(DEFAULT_MERGE_LIMIT);
    let selection = match input.get("mode").unwrap_or("explicit") {
        "explicit" => {
            let canonical = input
                .get("canonical")
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .ok_or_else(|| {
                    "explicit mode needs `canonical` (the predicate to keep) and `aliases` \
                     (the spellings to fold into it); or use mode=declared / mode=collisions"
                        .to_string()
                })?;
            MergeSelection::Explicit {
                canonical: canonical.to_string(),
                aliases: input.get_str_list("aliases"),
            }
        }
        "declared" => MergeSelection::Declared,
        "collisions" => MergeSelection::Collisions,
        other => {
            return Err(format!(
                "unknown mode \"{other}\": use explicit, declared or collisions"
            ));
        }
    };
    Ok(MergeRequest {
        selection,
        dry_run,
        limit,
    })
}

impl Tool for KgMergePredicatesTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: KG_MERGE_PREDICATES.into(),
            description: "Merge alias predicate spellings into one canonical predicate: every \
                 alias triple is re-asserted under the canonical predicate and the alias \
                 triple retracted, with provenance. Modes: explicit (canonical + aliases), \
                 declared (schema-pack aliases), collisions (spellings that differ only in \
                 case, spaces, '-' or '_')."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "mode".into(),
                    description: "explicit (default), declared, or collisions.".into(),
                    required: false,
                },
                ToolParam {
                    name: "canonical".into(),
                    description: "Explicit mode: the predicate to keep (label or id).".into(),
                    required: false,
                },
                ToolParam {
                    name: "aliases".into(),
                    description: "Explicit mode: predicates to fold into `canonical`.".into(),
                    required: false,
                },
                ToolParam {
                    name: "dry_run".into(),
                    description: "Count the triples that would move without changing anything."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "limit".into(),
                    description: "Most triples to move in this call (default 10000).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let request = match request_from(&input) {
            Ok(request) => request,
            Err(message) => return Ok(ToolOutput::err(message)),
        };
        let outcome = execute_merge(engine, &request)?;
        let verb = if outcome.dry_run {
            "Would move"
        } else {
            "Moved"
        };
        let mut summary = format!(
            "{verb} {} triple(s) onto {} canonical predicate(s)",
            outcome.moved,
            outcome.merged.len()
        );
        for merge in &outcome.merged {
            summary.push_str(&format!(
                "\n- {} <- {} ({} triple(s))",
                merge.canonical,
                merge.aliases.join(", "),
                merge.moved
            ));
        }
        if outcome.truncated {
            summary.push_str("\nStopped at `limit`; call again to continue.");
        }
        let symbols = outcome
            .merged
            .iter()
            .filter_map(|m| engine.lookup_symbol(&m.canonical).ok())
            .collect();
        Ok(ToolOutput::ok_with_symbols(summary, symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: KG_MERGE_PREDICATES.into(),
            description: "Merge alias predicate spellings into one canonical predicate \
                 (retract + re-assert with provenance)."
                .into(),
            parameters: vec![
                ToolParamSchema::optional("mode", "explicit (default), declared, or collisions."),
                ToolParamSchema::optional("canonical", "Explicit mode: the predicate to keep."),
                ToolParamSchema::optional(
                    "aliases",
                    "Explicit mode: predicates to fold into `canonical`.",
                ),
                ToolParamSchema::optional("dry_run", "Count without changing anything."),
                ToolParamSchema::optional("limit", "Most triples to move (default 10000)."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Rewrites every triple of the alias predicates onto the canonical \
                     predicate; superseded single-valued values are retracted."
                    .into(),
                shadow_triggers: Vec::new(),
                effect: crate::agent::tool_manifest::ToolEffect::Mutating,
            },
            source: ToolSource::Native,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::provenance::{DerivationKind, HygieneAction};

    fn engine() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(dir.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, dir)
    }

    fn input(value: serde_json::Value) -> ToolInput {
        let mut input = ToolInput::new();
        for (key, v) in value.as_object().unwrap() {
            input = match v {
                serde_json::Value::String(s) => input.with_param(key, s),
                other => input.with_value(key, other.clone()),
            };
        }
        input
    }

    fn put(engine: &Engine, s: &str, p: &str, o: &str) {
        engine
            .add_triple(&Triple::new(
                engine.resolve_or_create_entity(s).unwrap(),
                engine.resolve_or_create_relation(p).unwrap(),
                engine.resolve_or_create_entity(o).unwrap(),
            ))
            .unwrap();
    }

    #[test]
    fn explicit_merge_moves_alias_triples_with_provenance() {
        let (engine, _dir) = engine();
        put(&engine, "wi-1", "state", "open");
        put(&engine, "wi-2", "has-state", "done");
        put(&engine, "wi-3", "current state", "blocked");

        let dry = KgMergePredicatesTool
            .execute(
                &engine,
                input(serde_json::json!({
                    "canonical": "state",
                    "aliases": ["has-state", "current state"],
                    "dry_run": true,
                })),
            )
            .unwrap();
        assert!(dry.success);
        assert!(dry.result.starts_with("Would move 2"), "{}", dry.result);
        assert_eq!(
            engine
                .triples_with_predicate(engine.lookup_symbol("has-state").unwrap())
                .len(),
            1
        );

        let out = KgMergePredicatesTool
            .execute(
                &engine,
                input(serde_json::json!({
                    "canonical": "state",
                    "aliases": ["has-state", "current state"],
                })),
            )
            .unwrap();
        assert!(out.success, "{}", out.result);
        let state = engine.lookup_symbol("state").unwrap();
        assert_eq!(engine.triples_with_predicate(state).len(), 3);
        for alias in ["has-state", "current state"] {
            assert!(
                engine
                    .triples_with_predicate(engine.lookup_symbol(alias).unwrap())
                    .is_empty()
            );
        }
        let merges = engine
            .provenance_by_kind_tags(&[109])
            .unwrap()
            .into_iter()
            .filter(|r| {
                matches!(
                    r.kind,
                    DerivationKind::MemoryHygiene {
                        action: HygieneAction::PredicateMerge,
                        successor,
                        ..
                    } if successor == state
                )
            })
            .count();
        assert_eq!(merges, 2);
        // A merged fact is re-spelled, not retracted.
        let done = engine.lookup_symbol("done").unwrap();
        assert!(!engine.tms().was_retracted(done));
    }

    #[test]
    fn bad_requests_fail_with_guidance_and_change_nothing() {
        let (engine, _dir) = engine();
        put(&engine, "wi-1", "state", "open");
        let missing = KgMergePredicatesTool
            .execute(&engine, input(serde_json::json!({ "aliases": ["x"] })))
            .unwrap();
        assert!(!missing.success);
        assert!(missing.result.contains("canonical"));

        let unknown = KgMergePredicatesTool.execute(
            &engine,
            input(serde_json::json!({ "canonical": "state", "aliases": ["nope"] })),
        );
        assert!(unknown.is_err(), "an unknown alias is a typed merge error");

        let mode = KgMergePredicatesTool
            .execute(&engine, input(serde_json::json!({ "mode": "everything" })))
            .unwrap();
        assert!(!mode.success);
    }

    #[test]
    fn manifest_is_dangerous_kg_write() {
        let manifest = KgMergePredicatesTool.manifest();
        assert_eq!(manifest.danger.level, DangerLevel::Dangerous);
        assert!(manifest.danger.capabilities.contains(&Capability::WriteKg));
        assert!(manifest.danger.shadow_triggers.is_empty());
    }
}
