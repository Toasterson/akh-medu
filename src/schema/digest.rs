//! `get_graph_schema`-style digest (ADR 081 Slice 3, work-item 4's
//! "Acceptance" note in `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`
//! calls this out under Slice 2; it landed without it there because it needs
//! the effective-property machinery this slice built — see [`super::shape`]).
//!
//! Mirrors the GitLab-Orbit reference architecture's `get_graph_schema`: one
//! read-only introspection call surfacing the in-force schema for an
//! engine — every declared type with its effective (inherited-included)
//! property shape, and every declared relation with its argument-type shape —
//! as a single serializable artifact the operator, an ACP client, or an LLM
//! Librarian can inspect without walking the KG by hand.

use serde::{Deserialize, Serialize};

use crate::engine::Engine;

use super::compile::{PRED_ARG1_TYPE, PRED_ARG2_TYPE, PRED_IS_A, RELATION_MARKER, TYPE_MARKER};
use super::shape::{self, EffectiveProperty};
use super::source::ValueType;
use super::{SchemaError, SchemaResult};

/// One property in a [`TypeDigest`]'s effective shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyDigest {
    pub name: String,
    pub value_type: ValueType,
    pub required: bool,
    /// The type name this property is declared directly on (an ancestor when
    /// inherited).
    pub declared_on: String,
    /// Whether the property is declared directly on this type (`false` when
    /// inherited via `is-a`).
    pub direct: bool,
}

impl From<&EffectiveProperty> for PropertyDigest {
    fn from(p: &EffectiveProperty) -> Self {
        Self {
            name: p.name.clone(),
            value_type: p.value_type,
            required: p.required,
            declared_on: p.declared_on.clone(),
            direct: p.direct,
        }
    }
}

/// One installed type and its effective (own + inherited) property shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeDigest {
    pub name: String,
    /// Ancestry from this type up to its root (`chain[0] == name`).
    pub chain: Vec<String>,
    pub properties: Vec<PropertyDigest>,
}

/// One declared relation and its argument-type shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationDigest {
    pub name: String,
    pub arg1_type: Option<String>,
    pub arg2_type: Option<String>,
}

/// The in-force schema for an engine: every declared type (with its
/// effective shape) and relation currently resolvable in the KG.
///
/// Scoped to the whole engine, not a single workspace's `ctx:specializes`
/// activation stack — for a single-workspace-per-engine deployment (today's
/// default) these coincide; per-workspace filtering by the activation stack
/// is a natural follow-up once multiple workspaces routinely share one
/// engine process.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SchemaDigest {
    pub types: Vec<TypeDigest>,
    pub relations: Vec<RelationDigest>,
}

impl SchemaDigest {
    /// Whether no schema pack is installed at all — the digest carries
    /// nothing worth rendering into a prompt.
    pub fn is_empty(&self) -> bool {
        self.types.is_empty() && self.relations.is_empty()
    }

    /// Compact, token-cheap textual rendering for the working-context
    /// runtime's core tier (ADR 081 slice 7): one line per type (with its
    /// `is-a` ancestry and effective properties) and one line per relation.
    /// Not meant to be exhaustive — [`SchemaDigest`] itself (serialized) is
    /// the full introspection artifact; this is the "always in context"
    /// summary a model reads to know the in-force vocabulary at a glance.
    pub fn render_compact(&self) -> String {
        let mut out = String::new();
        for t in &self.types {
            out.push_str("- ");
            out.push_str(&t.name);
            if t.chain.len() > 1 {
                out.push_str(" (is-a ");
                out.push_str(&t.chain[1..].join(" -> "));
                out.push(')');
            }
            if !t.properties.is_empty() {
                let names: Vec<&str> = t.properties.iter().map(|p| p.name.as_str()).collect();
                out.push_str(" [");
                out.push_str(&names.join(", "));
                out.push(']');
            }
            out.push('\n');
        }
        if !self.relations.is_empty() {
            out.push_str("relations:\n");
            for r in &self.relations {
                out.push_str("- ");
                out.push_str(&r.name);
                out.push('(');
                out.push_str(r.arg1_type.as_deref().unwrap_or("?"));
                out.push_str(", ");
                out.push_str(r.arg2_type.as_deref().unwrap_or("?"));
                out.push_str(")\n");
            }
        }
        out
    }
}

/// Build the in-force schema digest for `engine`. Read-only; never mutates
/// the KG. An engine with no schema pack installed yields an empty digest,
/// not an error.
pub fn digest(engine: &Engine) -> SchemaResult<SchemaDigest> {
    let Ok(is_a) = engine.resolve_symbol(PRED_IS_A) else {
        return Ok(SchemaDigest {
            types: Vec::new(),
            relations: Vec::new(),
        });
    };
    let Ok(type_marker) = engine.resolve_symbol(TYPE_MARKER) else {
        return Ok(SchemaDigest {
            types: Vec::new(),
            relations: Vec::new(),
        });
    };

    let mut types = Vec::new();
    for type_symbol in engine.knowledge_graph().subjects_of(is_a, type_marker) {
        let effective = shape::effective_properties(engine, type_symbol).map_err(|e| {
            SchemaError::Introspection {
                message: format!(
                    "digest: failed to resolve effective properties for \"{}\": {e}",
                    engine.resolve_label(type_symbol)
                ),
            }
        })?;
        types.push(TypeDigest {
            name: effective.type_name.clone(),
            chain: effective.chain.clone(),
            properties: effective
                .properties
                .values()
                .map(PropertyDigest::from)
                .collect(),
        });
    }
    types.sort_by(|a, b| a.name.cmp(&b.name));

    let mut relations = Vec::new();
    if let Ok(relation_marker) = engine.resolve_symbol(RELATION_MARKER) {
        let arg1_pred = engine.resolve_symbol(PRED_ARG1_TYPE).ok();
        let arg2_pred = engine.resolve_symbol(PRED_ARG2_TYPE).ok();
        for relation_symbol in engine.knowledge_graph().subjects_of(is_a, relation_marker) {
            let name = engine.resolve_label(relation_symbol);
            let arg1_type = arg1_pred
                .and_then(|p| {
                    engine
                        .knowledge_graph()
                        .objects_of(relation_symbol, p)
                        .into_iter()
                        .next()
                })
                .map(|s| engine.resolve_label(s));
            let arg2_type = arg2_pred
                .and_then(|p| {
                    engine
                        .knowledge_graph()
                        .objects_of(relation_symbol, p)
                        .into_iter()
                        .next()
                })
                .map(|s| engine.resolve_label(s));
            relations.push(RelationDigest {
                name,
                arg1_type,
                arg2_type,
            });
        }
    }
    relations.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(SchemaDigest { types, relations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::schema::compile::compile;
    use crate::schema::install::install_pack;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: None,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn empty_engine_yields_empty_digest() {
        let engine = test_engine();
        let d = digest(&engine).unwrap();
        assert!(d.types.is_empty());
        assert!(d.relations.is_empty());
    }

    const SRC: &str = r#"{
        "pack": {"name": "core-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "name", "value_type": "text", "required": true},
                {"name": "url", "value_type": "link"}]},
            {"name": "Organization"},
            {"name": "Employee", "is_a": "Person", "properties": [
                {"name": "employer", "value_type": "link", "required": true}]}
        ],
        "relations": [
            {"name": "worksFor", "arg1_type": "Person", "arg2_type": "Organization"}
        ]
    }"#;

    #[test]
    fn digest_lists_types_with_effective_properties_and_relations() {
        let engine = test_engine();
        install_pack(&engine, &compile(SRC).unwrap()).unwrap();

        let d = digest(&engine).unwrap();
        let names: Vec<&str> = d.types.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["Employee", "Organization", "Person"]);

        let employee = d.types.iter().find(|t| t.name == "Employee").unwrap();
        assert_eq!(
            employee.chain,
            vec!["Employee".to_string(), "Person".to_string()]
        );
        let prop_names: Vec<&str> = employee
            .properties
            .iter()
            .map(|p| p.name.as_str())
            .collect();
        let mut sorted = prop_names.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["employer", "name", "url"]);
        // Inherited-vs-direct distinction survives into the digest.
        let name_prop = employee
            .properties
            .iter()
            .find(|p| p.name == "name")
            .unwrap();
        assert!(!name_prop.direct);
        assert_eq!(name_prop.declared_on, "Person");
        let employer_prop = employee
            .properties
            .iter()
            .find(|p| p.name == "employer")
            .unwrap();
        assert!(employer_prop.direct);

        // Person's own digest is unaffected by Employee (additive).
        let person = d.types.iter().find(|t| t.name == "Person").unwrap();
        assert!(!person.properties.iter().any(|p| p.name == "employer"));

        assert_eq!(d.relations.len(), 1);
        assert_eq!(d.relations[0].name, "worksFor");
        assert_eq!(d.relations[0].arg1_type.as_deref(), Some("Person"));
        assert_eq!(d.relations[0].arg2_type.as_deref(), Some("Organization"));
    }

    #[test]
    fn empty_digest_renders_nothing() {
        let engine = test_engine();
        let d = digest(&engine).unwrap();
        assert!(d.is_empty());
        assert_eq!(d.render_compact(), "");
    }

    #[test]
    fn render_compact_lists_types_ancestry_properties_and_relations() {
        let engine = test_engine();
        install_pack(&engine, &compile(SRC).unwrap()).unwrap();
        let d = digest(&engine).unwrap();
        assert!(!d.is_empty());
        let rendered = d.render_compact();
        assert!(rendered.contains("Employee (is-a Person)"), "{rendered}");
        assert!(rendered.contains("employer"), "{rendered}");
        assert!(
            rendered.contains("worksFor(Person, Organization)"),
            "{rendered}"
        );
    }

    #[test]
    fn digest_is_serializable() {
        let engine = test_engine();
        install_pack(&engine, &compile(SRC).unwrap()).unwrap();
        let d = digest(&engine).unwrap();
        let json = serde_json::to_string(&d).unwrap();
        let back: SchemaDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }
}
