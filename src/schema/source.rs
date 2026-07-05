//! Portable schema-pack source: the human-authored, GitHub-reviewable form.
//!
//! ADR 081 decision 9: packs are "authored/reviewed on GitHub in a portable
//! form" and compiled once at the human-confirm step. The v1 front-end is a
//! JSON document (JSON-LD-flavoured: schema.org type names, `is_a`
//! subtyping); Turtle/YAML front-ends can be added later behind
//! [`parse_source`] without touching the compiled artifact format.
//!
//! Vocabulary per ADR 081 decision 4 (layered borrow):
//! - entity types ← curated schema.org `Thing` subset + Akh-native types,
//! - property value types ← Obsidian's set ([`ValueType`]),
//! - relation names ← the existing predicate namespaces.

use serde::{Deserialize, Serialize};

use super::{SchemaError, SchemaResult};

/// Reserved label prefixes owned by the engine substrate; packs may not
/// declare names under them (they may still *reference* `assumes` types).
const RESERVED_PREFIXES: &[&str] = &["onto:", "ctx:", "agent:", "doc:", "akh:", "rel:", "mt:"];

/// Reserved bare names (seed/engine conventions).
const RESERVED_NAMES: &[&str] = &["relation", "is-a", "part-of"];

/// Whether a pack is destined for the shared registry or an akh's own
/// frontier (ADR 081 decision 7: shared = registry MTs, own = frontier MT).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PackScope {
    /// Shared registry pack — admission requires a human confirm.
    #[default]
    Shared,
    /// An akh's private frontier pack.
    Own,
}

impl std::fmt::Display for PackScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackScope::Shared => write!(f, "shared"),
            PackScope::Own => write!(f, "own"),
        }
    }
}

/// Obsidian frontmatter value types (ADR 081 decision 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    Text,
    List,
    Number,
    Date,
    Datetime,
    Checkbox,
    Link,
}

impl ValueType {
    /// The canonical KG entity label for this value type.
    pub fn entity_label(&self) -> &'static str {
        match self {
            ValueType::Text => "onto:value/text",
            ValueType::List => "onto:value/list",
            ValueType::Number => "onto:value/number",
            ValueType::Date => "onto:value/date",
            ValueType::Datetime => "onto:value/datetime",
            ValueType::Checkbox => "onto:value/checkbox",
            ValueType::Link => "onto:value/link",
        }
    }
}

/// Pack-level metadata (becomes the registry-row identity fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackMeta {
    /// Globally-unique pack name (part of the microtheory label).
    pub name: String,
    /// Pack version (semver-ish string; part of the microtheory label).
    pub version: String,
    /// Shared registry pack or private frontier pack.
    #[serde(default)]
    pub scope: PackScope,
    /// Where the portable source is curated (e.g. a GitHub URL).
    #[serde(default)]
    pub source_url: Option<String>,
    /// Human-readable description (kept in the artifact metadata, not lowered
    /// into triples — prose does not belong in the typed core).
    #[serde(default)]
    pub description: Option<String>,
}

/// A property carried by a type (additive; never mutates existing views).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDef {
    /// Property name (global, schema.org-flavoured — e.g. `name`, `url`).
    pub name: String,
    /// Obsidian value type of the property.
    pub value_type: ValueType,
    /// Whether the owning type's shape requires this property.
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// A globally-unique, immutable entity type declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDef {
    /// Global type name (e.g. `Person`). Never redefinable (ADR 081 dec. 5).
    pub name: String,
    /// Supertype: the only extension mechanism. Must name a type declared in
    /// this pack or listed in `assumes`. `None` = a root type.
    #[serde(default)]
    pub is_a: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    /// Properties this type adds (additive over inherited ones).
    #[serde(default)]
    pub properties: Vec<PropertyDef>,
}

/// A relation (predicate) declaration with optional argument-type shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationDef {
    /// Relation name (e.g. `worksFor`).
    pub name: String,
    /// Declared arity. Triples are binary; only 2 is accepted.
    #[serde(default = "default_arity")]
    pub arity: u64,
    /// Required type of the subject (arg 1), if constrained.
    #[serde(default)]
    pub arg1_type: Option<String>,
    /// Required type of the object (arg 2), if constrained.
    #[serde(default)]
    pub arg2_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

fn default_arity() -> u64 {
    2
}

/// A parsed + validated schema-pack source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaPackSource {
    pub pack: PackMeta,
    /// Type names this pack references but does not define (provided by
    /// other packs; verified against the registry at confirm time).
    #[serde(default)]
    pub assumes: Vec<String>,
    #[serde(default)]
    pub types: Vec<TypeDef>,
    #[serde(default)]
    pub relations: Vec<RelationDef>,
}

/// Parse and validate a portable schema-pack source (JSON front-end).
pub fn parse_source(text: &str) -> SchemaResult<SchemaPackSource> {
    let source: SchemaPackSource =
        serde_json::from_str(text).map_err(|e| SchemaError::Parse {
            message: e.to_string(),
        })?;
    validate(&source)?;
    Ok(source)
}

fn is_reserved(name: &str) -> bool {
    RESERVED_NAMES.contains(&name) || RESERVED_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Structural validation: metadata sanity, uniqueness, reserved names,
/// supertype/arg-type resolvability, arity.
pub fn validate(source: &SchemaPackSource) -> SchemaResult<()> {
    // --- metadata ---
    let name = source.pack.name.trim();
    if name.is_empty() || name.chars().any(char::is_whitespace) {
        return Err(SchemaError::InvalidMeta {
            message: format!(
                "pack.name {:?} must be non-empty and contain no whitespace",
                source.pack.name
            ),
        });
    }
    if source.pack.version.trim().is_empty() {
        return Err(SchemaError::InvalidMeta {
            message: "pack.version must be non-empty".into(),
        });
    }

    // --- type declarations (a name may be declared once, and never also
    //     listed in `assumes` — that would claim it is defined elsewhere) ---
    let assumed: std::collections::HashSet<&str> =
        source.assumes.iter().map(String::as_str).collect();
    let mut declared: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for t in &source.types {
        if is_reserved(&t.name) {
            return Err(SchemaError::ReservedName {
                name: t.name.clone(),
            });
        }
        if !declared.insert(&t.name) || assumed.contains(t.name.as_str()) {
            return Err(SchemaError::DuplicateType {
                name: t.name.clone(),
            });
        }
        let mut props = std::collections::HashSet::new();
        for p in &t.properties {
            if is_reserved(&p.name) {
                return Err(SchemaError::ReservedName {
                    name: p.name.clone(),
                });
            }
            if !props.insert(p.name.as_str()) {
                return Err(SchemaError::DuplicateProperty {
                    type_name: t.name.clone(),
                    property: p.name.clone(),
                });
            }
        }
    }

    // --- supertype resolvability (declared anywhere in pack, or assumed) ---
    let known_types: std::collections::HashSet<&str> =
        declared.union(&assumed).copied().collect();
    for t in &source.types {
        if let Some(ref sup) = t.is_a
            && !known_types.contains(sup.as_str())
        {
            return Err(SchemaError::UnknownSupertype {
                type_name: t.name.clone(),
                super_name: sup.clone(),
            });
        }
    }

    // --- relations ---
    let mut rel_names = std::collections::HashSet::new();
    for r in &source.relations {
        if is_reserved(&r.name) {
            return Err(SchemaError::ReservedName {
                name: r.name.clone(),
            });
        }
        if !rel_names.insert(r.name.as_str()) {
            return Err(SchemaError::DuplicateRelation {
                name: r.name.clone(),
            });
        }
        if r.arity != 2 {
            return Err(SchemaError::UnsupportedArity {
                relation: r.name.clone(),
                arity: r.arity,
            });
        }
        for (arg, ty) in [(1u8, &r.arg1_type), (2u8, &r.arg2_type)] {
            if let Some(ty) = ty
                && !known_types.contains(ty.as_str())
            {
                return Err(SchemaError::UnknownArgType {
                    relation: r.name.clone(),
                    arg,
                    type_name: ty.clone(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(name: &str, version: &str) -> String {
        format!(
            r#"{{"pack": {{"name": "{name}", "version": "{version}"}},
                 "types": [{{"name": "Person"}}]}}"#
        )
    }

    #[test]
    fn parses_minimal_pack() {
        let src = parse_source(&minimal("core", "0.1.0")).unwrap();
        assert_eq!(src.pack.name, "core");
        assert_eq!(src.pack.scope, PackScope::Shared);
        assert_eq!(src.types.len(), 1);
    }

    #[test]
    fn rejects_empty_name_and_version() {
        assert!(matches!(
            parse_source(&minimal("", "0.1.0")),
            Err(SchemaError::InvalidMeta { .. })
        ));
        assert!(matches!(
            parse_source(&minimal("core", " ")),
            Err(SchemaError::InvalidMeta { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_type() {
        let text = r#"{"pack": {"name": "p", "version": "1"},
            "types": [{"name": "Person"}, {"name": "Person"}]}"#;
        assert!(matches!(
            parse_source(text),
            Err(SchemaError::DuplicateType { name }) if name == "Person"
        ));
    }

    #[test]
    fn rejects_unknown_supertype_but_accepts_assumed() {
        let bad = r#"{"pack": {"name": "p", "version": "1"},
            "types": [{"name": "Employee", "is_a": "Person"}]}"#;
        assert!(matches!(
            parse_source(bad),
            Err(SchemaError::UnknownSupertype { super_name, .. }) if super_name == "Person"
        ));

        let good = r#"{"pack": {"name": "p", "version": "1"},
            "assumes": ["Person"],
            "types": [{"name": "Employee", "is_a": "Person"}]}"#;
        assert!(parse_source(good).is_ok());
    }

    #[test]
    fn rejects_reserved_names() {
        let text = r#"{"pack": {"name": "p", "version": "1"},
            "types": [{"name": "onto:Sneaky"}]}"#;
        assert!(matches!(
            parse_source(text),
            Err(SchemaError::ReservedName { .. })
        ));
        let rel = r#"{"pack": {"name": "p", "version": "1"},
            "relations": [{"name": "is-a"}]}"#;
        assert!(matches!(
            parse_source(rel),
            Err(SchemaError::ReservedName { .. })
        ));
    }

    #[test]
    fn rejects_non_binary_arity() {
        let text = r#"{"pack": {"name": "p", "version": "1"},
            "relations": [{"name": "between", "arity": 3}]}"#;
        assert!(matches!(
            parse_source(text),
            Err(SchemaError::UnsupportedArity { arity: 3, .. })
        ));
    }

    #[test]
    fn rejects_unknown_arg_type() {
        let text = r#"{"pack": {"name": "p", "version": "1"},
            "types": [{"name": "Person"}],
            "relations": [{"name": "worksFor", "arg1_type": "Person", "arg2_type": "Organization"}]}"#;
        assert!(matches!(
            parse_source(text),
            Err(SchemaError::UnknownArgType { arg: 2, .. })
        ));
    }

    #[test]
    fn rejects_duplicate_property() {
        let text = r#"{"pack": {"name": "p", "version": "1"},
            "types": [{"name": "Person", "properties": [
                {"name": "name", "value_type": "text"},
                {"name": "name", "value_type": "text"}]}]}"#;
        assert!(matches!(
            parse_source(text),
            Err(SchemaError::DuplicateProperty { .. })
        ));
    }

    #[test]
    fn value_type_labels_are_canonical() {
        assert_eq!(ValueType::Text.entity_label(), "onto:value/text");
        assert_eq!(ValueType::Datetime.entity_label(), "onto:value/datetime");
    }
}
