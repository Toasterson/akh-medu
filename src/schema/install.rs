//! Per-akh install of a compiled schema pack: bulk-insert-as-microtheory.
//!
//! ADR 081 decision 9: after the confirm-time compile, "per-akh install is a
//! dumb byte-fetch + bulk-insert-as-microtheory (no bespoke translation
//! code)". This module is that bulk-insert: every [`PortableTriple`] lands in
//! the KG under the pack's deterministic microtheory compartment
//! ([`CompiledSchemaPack::microtheory_label`]), and activation is a single
//! `ctx:specializes` import edge (decision 6: an akh's effective schema is
//! the import stack).
//!
//! The **global-immutable-type-name guard** (decision 5) is enforced here:
//! installing a pack that re-declares a type name already declared by a
//! *different* compartment is rejected with a rich diagnostic. Reinstalling
//! the same pack@version is idempotent.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::compile::{CompiledSchemaPack, PRED_IS_A, RELATION_MARKER, TYPE_MARKER};
use super::{SchemaError, SchemaResult};

/// Marker predicate: `(mt-symbol, akh:schema-installed, name@version)`.
const INSTALLED_PREDICATE: &str = "akh:schema-installed";

/// Report after installing a compiled schema pack.
#[derive(Debug, Clone)]
pub struct InstallReport {
    /// The Entity symbol representing the pack's microtheory.
    pub microtheory: SymbolId,
    /// The microtheory compartment label the triples were inserted under.
    pub microtheory_label: String,
    /// Number of triples inserted (0 when `already_installed`).
    pub triples_applied: usize,
    /// Whether this pack@version was already installed (no-op).
    pub already_installed: bool,
}

fn install_err(pack: &CompiledSchemaPack, message: impl std::fmt::Display) -> SchemaError {
    SchemaError::Install {
        pack: format!("{}@{}", pack.name, pack.version),
        message: message.to_string(),
    }
}

/// Install a compiled schema pack into an engine as a microtheory.
///
/// Enforces global type-name immutability (ADR 081 decision 5): if any type
/// declared by this pack is already declared as a type by a different
/// compartment, the install is rejected before anything is written.
/// Reinstalling the same pack@version is an idempotent no-op.
pub fn install_pack(engine: &Engine, pack: &CompiledSchemaPack) -> SchemaResult<InstallReport> {
    let mt_label = pack.microtheory_label();

    // --- Idempotency: already installed? ---
    let mt_existing = engine.resolve_symbol(&mt_label).ok();
    if let Some(mt) = mt_existing
        && engine.triples_from(mt).iter().any(|t| {
            engine.resolve_label(t.predicate) == INSTALLED_PREDICATE
        })
    {
        return Ok(InstallReport {
            microtheory: mt,
            microtheory_label: mt_label,
            triples_applied: 0,
            already_installed: true,
        });
    }

    // --- Global immutability guard (decision 5) ---
    // A type name declared by this pack must not already be declared as a
    // type (`name is-a onto:Type`) by any *other* compartment.
    if let Ok(type_marker) = engine.resolve_symbol(TYPE_MARKER) {
        for type_name in &pack.types {
            let Ok(existing) = engine.resolve_symbol(type_name) else {
                continue;
            };
            let redefined = engine
                .triples_from(existing)
                .into_iter()
                .find(|t| {
                    t.object == type_marker
                        && engine.resolve_label(t.predicate) == PRED_IS_A
                        && t.compartment_id.as_deref() != Some(mt_label.as_str())
                });
            if let Some(prior) = redefined {
                return Err(SchemaError::TypeRedefinition {
                    type_name: type_name.clone(),
                    existing_compartment: prior
                        .compartment_id
                        .unwrap_or_else(|| "(default graph)".to_string()),
                });
            }
        }
    }

    // --- Bulk insert under the pack's microtheory compartment ---
    let mut applied = 0usize;
    for pt in &pack.triples {
        // Seed convention: `(r, is-a, relation)` mints the subject as a Relation.
        let subject = if pt.predicate == PRED_IS_A && pt.object == RELATION_MARKER {
            engine.resolve_or_create_relation(&pt.subject)
        } else {
            engine.resolve_or_create_entity(&pt.subject)
        }
        .map_err(|e| install_err(pack, format!("subject '{}': {e}", pt.subject)))?;

        let predicate = engine
            .resolve_or_create_relation(&pt.predicate)
            .map_err(|e| install_err(pack, format!("predicate '{}': {e}", pt.predicate)))?;

        let object = engine
            .resolve_or_create_entity(&pt.object)
            .map_err(|e| install_err(pack, format!("object '{}': {e}", pt.object)))?;

        let triple = Triple::new(subject, predicate, object).with_compartment(mt_label.clone());
        engine.add_triple(&triple).map_err(|e| {
            install_err(
                pack,
                format!(
                    "triple ({}, {}, {}): {e}",
                    pt.subject, pt.predicate, pt.object
                ),
            )
        })?;
        applied += 1;
    }

    // --- Microtheory entity + installed marker ---
    let mt = engine
        .resolve_or_create_entity(&mt_label)
        .map_err(|e| install_err(pack, format!("microtheory entity: {e}")))?;
    let installed_pred = engine
        .resolve_or_create_relation(INSTALLED_PREDICATE)
        .map_err(|e| install_err(pack, format!("installed predicate: {e}")))?;
    let version_obj = engine
        .resolve_or_create_entity(&format!("{}@{}", pack.name, pack.version))
        .map_err(|e| install_err(pack, format!("version entity: {e}")))?;
    engine
        .add_triple(
            &Triple::new(mt, installed_pred, version_obj).with_compartment(mt_label.clone()),
        )
        .map_err(|e| install_err(pack, format!("installed marker: {e}")))?;

    Ok(InstallReport {
        microtheory: mt,
        microtheory_label: mt_label,
        triples_applied: applied,
        already_installed: false,
    })
}

/// Activate an installed pack for a context: one `ctx:specializes` import
/// edge from the importing context (e.g. an akh's effective-schema MT) to the
/// pack's microtheory (ADR 081 decision 6).
pub fn activate_pack(
    engine: &Engine,
    importer: SymbolId,
    pack: &CompiledSchemaPack,
) -> SchemaResult<()> {
    let mt = engine
        .resolve_symbol(&pack.microtheory_label())
        .map_err(|_| install_err(pack, "pack is not installed — run install_pack first"))?;
    let specializes = engine
        .resolve_or_create_relation("ctx:specializes")
        .map_err(|e| install_err(pack, format!("ctx:specializes: {e}")))?;
    engine
        .add_triple(&Triple::new(importer, specializes, mt))
        .map_err(|e| install_err(pack, format!("activation edge: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::schema::compile::compile;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: None,
            ..Default::default()
        })
        .unwrap()
    }

    const SRC: &str = r#"{
        "pack": {"name": "core-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "name", "value_type": "text", "required": true}]},
            {"name": "Organization"}
        ],
        "relations": [
            {"name": "worksFor", "arg1_type": "Person", "arg2_type": "Organization"}
        ]
    }"#;

    #[test]
    fn install_projects_types_into_kg() {
        let engine = test_engine();
        let pack = compile(SRC).unwrap();
        let report = install_pack(&engine, &pack).unwrap();
        assert!(!report.already_installed);
        assert!(report.triples_applied > 0);
        assert_eq!(report.microtheory_label, "mt:schema/core-test@0.1.0");

        // The typed core is queryable: Person is-a onto:Type.
        let person = engine.resolve_symbol("Person").unwrap();
        let is_a = engine.resolve_symbol("is-a").unwrap();
        let marker = engine.resolve_symbol(TYPE_MARKER).unwrap();
        assert!(
            engine
                .knowledge_graph()
                .objects_of(person, is_a)
                .contains(&marker)
        );

        // The relation was minted as a Relation and carries its arg shape.
        let works_for = engine.resolve_symbol("worksFor").unwrap();
        let arg1 = engine.resolve_symbol("onto:arg1type").unwrap();
        assert_eq!(
            engine.knowledge_graph().objects_of(works_for, arg1),
            vec![person]
        );

        // Every pack triple landed in the pack's compartment.
        let in_mt = engine
            .all_triples()
            .into_iter()
            .filter(|t| t.compartment_id.as_deref() == Some("mt:schema/core-test@0.1.0"))
            .count();
        assert!(in_mt >= report.triples_applied);
    }

    #[test]
    fn reinstall_same_pack_is_idempotent() {
        let engine = test_engine();
        let pack = compile(SRC).unwrap();
        install_pack(&engine, &pack).unwrap();
        let count_before = engine.all_triples().len();

        let again = install_pack(&engine, &pack).unwrap();
        assert!(again.already_installed);
        assert_eq!(again.triples_applied, 0);
        assert_eq!(engine.all_triples().len(), count_before);
    }

    #[test]
    fn redefining_a_global_type_is_rejected() {
        let engine = test_engine();
        install_pack(&engine, &compile(SRC).unwrap()).unwrap();

        // A different pack re-declaring "Person" must be rejected.
        let rival = compile(
            r#"{"pack": {"name": "rival", "version": "1.0.0"},
                "types": [{"name": "Person"}]}"#,
        )
        .unwrap();
        let err = install_pack(&engine, &rival).unwrap_err();
        assert!(matches!(
            err,
            SchemaError::TypeRedefinition { ref type_name, .. } if type_name == "Person"
        ));

        // Extension via is-a subtyping is the sanctioned path.
        let extension = compile(
            r#"{"pack": {"name": "hr", "version": "1.0.0"},
                "assumes": ["Person"],
                "types": [{"name": "Employee", "is_a": "Person",
                           "properties": [{"name": "employer", "value_type": "link", "required": true}]}]}"#,
        )
        .unwrap();
        let report = install_pack(&engine, &extension).unwrap();
        assert!(!report.already_installed);

        // Additive: Employee is-a Person edge exists, Person's view unchanged.
        let employee = engine.resolve_symbol("Employee").unwrap();
        let person = engine.resolve_symbol("Person").unwrap();
        let is_a = engine.resolve_symbol("is-a").unwrap();
        assert!(
            engine
                .knowledge_graph()
                .objects_of(employee, is_a)
                .contains(&person)
        );
    }

    #[test]
    fn activation_adds_specializes_edge() {
        let engine = test_engine();
        let pack = compile(SRC).unwrap();
        install_pack(&engine, &pack).unwrap();

        let akh_schema = engine.resolve_or_create_entity("mt:akh/test-effective").unwrap();
        activate_pack(&engine, akh_schema, &pack).unwrap();

        let specializes = engine.resolve_symbol("ctx:specializes").unwrap();
        let mt = engine.resolve_symbol("mt:schema/core-test@0.1.0").unwrap();
        assert!(
            engine
                .knowledge_graph()
                .objects_of(akh_schema, specializes)
                .contains(&mt)
        );
    }

    #[test]
    fn activation_requires_install() {
        let engine = test_engine();
        let pack = compile(SRC).unwrap();
        let importer = engine.resolve_or_create_entity("mt:akh/x").unwrap();
        assert!(matches!(
            activate_pack(&engine, importer, &pack),
            Err(SchemaError::Install { .. })
        ));
    }

    #[test]
    fn bundled_seed_pack_installs_end_to_end() {
        let engine = test_engine();
        let text = include_str!("../../data/schema-packs/schema-org-core.json");
        let pack = compile(text).unwrap();
        let report = install_pack(&engine, &pack).unwrap();
        assert!(report.triples_applied > 30);

        // Subtype chain: Person is-a Thing, Thing is-a onto:Type.
        let person = engine.resolve_symbol("Person").unwrap();
        let thing = engine.resolve_symbol("Thing").unwrap();
        let is_a = engine.resolve_symbol("is-a").unwrap();
        assert!(
            engine
                .knowledge_graph()
                .objects_of(person, is_a)
                .contains(&thing)
        );
    }

    #[test]
    fn installed_schema_triples_have_stable_identity() {
        // The install compartment is deterministic from (name, version), so
        // the same pack installed into two different engines yields the same
        // triple identities (the ADR-081 cross-akh property).
        let engine_a = test_engine();
        let engine_b = test_engine();
        let pack = compile(SRC).unwrap();

        // Skew engine_b's SymbolId allocation so ids differ between engines.
        for i in 0..7 {
            engine_b
                .resolve_or_create_entity(&format!("skew-{i}"))
                .unwrap();
        }

        install_pack(&engine_a, &pack).unwrap();
        install_pack(&engine_b, &pack).unwrap();

        let ids = |engine: &Engine| {
            let mut v: Vec<String> = engine
                .all_triples()
                .iter()
                .filter(|t| t.compartment_id.as_deref() == Some("mt:schema/core-test@0.1.0"))
                .map(|t| engine.triple_identity(t).to_hex())
                .collect();
            v.sort();
            v
        };
        assert_eq!(ids(&engine_a), ids(&engine_b));
        assert!(!ids(&engine_a).is_empty());
    }
}
