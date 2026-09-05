//! Migration: grandfather + schema-coverage report (ADR 081 Slice 9,
//! `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 9).
//!
//! ADR 081: *"Migration is no-flag-day: legacy triples are grandfathered as
//! frontier; the Librarian pass is the migration engine."* Concretely: every
//! triple already in the KG before (or outside) the typed projection keeps
//! working exactly as it always has — permissively, unconstrained, queryable.
//! Introducing the typed core changes **nothing** about how those triples
//! behave; it only adds a *lens* that can tell typed triples apart from
//! everything else.
//!
//! This module **is** that lens. [`classify_triple`] answers "what kind of
//! triple is this?" *structurally*, from the triple's own shape (predicate
//! label, compartment, and whether its subject/object carry a schema-pack
//! type) — no rewriting, no background job, no per-triple flag to set and
//! forget. [`coverage_report`] runs the lens over the live KG once and tallies
//! the result: a **read-only, non-LLM, opt-in dashboard** an operator can run
//! anytime to see whether the typed core is crystallizing (ADR 081, "typed
//! read-model that crystallizes an immutable typed core at runtime").
//!
//! ## Why classification, not a migration job
//!
//! A physical "grandfather" pass — walking every triple once and stamping it
//! `frontier` — would need to *keep* stamping every triple asserted from then
//! on (append-only writes, sleep consolidation, Librarian extraction, raw
//! `resolve_or_create_relation` + `add_triple` calls from a dozen call sites)
//! or it silently rots the moment a new legacy-shaped triple lands. Because
//! [`classify_triple`] is a pure function of a triple's *current* shape
//! (predicate namespace/label + whether the pack that would declare it is
//! actually installed), it never goes stale: a triple asserted five years ago
//! and a triple asserted five seconds ago classify identically, and a triple
//! that *becomes* typed (because its type's pack gets installed later, or the
//! Librarian retypes its subject) reclassifies for free on the next report —
//! no backfill required. That is "the immutable classification rule beats
//! data migration" from this slice's brief.
//!
//! ## The classification rule
//!
//! Three buckets ([`TripleClass`]):
//!
//! - **[`TripleClass::SchemaInfra`]** — the schema system's own bookkeeping,
//!   not data about the world: every triple a schema pack installs under its
//!   own `mt:schema/{name}@{version}` compartment ([`install::install_pack`]),
//!   plus any triple whose predicate lives in the `onto:` (type/property
//!   shape vocabulary) or `ctx:` (microtheory/import graph, e.g.
//!   `ctx:specializes` activation edges) namespaces wherever it's asserted.
//! - **[`TripleClass::TypedCore`]** — a *data* triple demonstrably backed by
//!   an installed pack:
//!   - `(subject, is-a, T)` where `T` is a pack-declared type (checked via
//!     the same `(T, is-a, onto:Type)` marker [`project::project_note`]
//!     uses) — this is the type-establishing edge itself, so it needs no
//!     further "subject already typed" condition (typing *is* the point of
//!     the triple);
//!   - `(subject, p, object)` where `p` is a property a pack declares on some
//!     type (`(T, onto:property, p)`) or a pack-declared relation
//!     (`(p, is-a, relation)`), **and** `subject` itself carries a
//!     pack-declared type via an `is-a` edge outside any pack's own
//!     compartment — "predicate declared in an installed pack, subject
//!     typed" from the brief, read literally;
//!   - the **front-loaded core relation namespaces** (see below) — asserted
//!     unconditionally as core, independent of subject typing, because they
//!     are substrate bookkeeping relations already governed by
//!     [`SchemaError::ReservedName`], not arbitrary entity properties a pack
//!     could plausibly disagree about.
//! - **[`TripleClass::Frontier`]** — everything else: an explicit
//!   `note:frontier/{key}` marker ([`project::FRONTIER_PREFIX`], from
//!   [`project::project_note`]'s "writes are never blocked" unknown-key
//!   path), an `is-a` edge to a type nobody has installed, or **any
//!   predicate this akh's installed packs don't declare** — which is exactly
//!   "grandfathered legacy": every triple asserted before schema-typing
//!   existed, or via a raw `resolve_or_create_relation` + `add_triple` call
//!   that never went through a pack, is frontier *by this same rule*, with no
//!   separate migration code path.
//!
//! ### Front-loaded core relation namespaces (issue 9's "front-load the safe
//! ### part")
//!
//! Issue 9 asks to "register the existing relation namespaces (`is-a`,
//! `part-of`, `doc:`, `agent:`, `ctx:`, the `ElicitPass` relations) into the
//! seed core vocabulary at install; entity typing stays lazy." Implemented
//! here as a fixed, code-verifiable set rather than a data migration:
//!
//! - [`FRONTLOADED_CORE_NAMES`] (`part-of`) and [`FRONTLOADED_CORE_PREFIXES`]
//!   (`agent:`, `doc:`, `akh:`, `rel:`) classify unconditionally as
//!   [`TripleClass::TypedCore`] — these mirror the reserved substrate
//!   namespaces [`SchemaError::ReservedName`] already refuses to let a pack
//!   redeclare, so they are core by construction, not by an installed pack.
//! - `is-a` is **not** blanket-front-loaded to `TypedCore`: it is handled by
//!   the type-establishing-edge rule above, so an `is-a` triple only counts
//!   as typed once its object names an actually-installed type. A five-year-
//!   old `(Socrates, is-a, Human)` triple with no `Human` pack installed
//!   stays `Frontier` — exactly the "entity typing stays lazy" half of the
//!   ask: is-a triples earn `TypedCore` only as packs get installed, never in
//!   bulk.
//! - `ctx:` is **not** front-loaded to `TypedCore` either — this module
//!   splits the issue's two-bucket ask (core vs. frontier) into three
//!   buckets, and `ctx:` bookkeeping (microtheory import edges) reads more
//!   usefully as [`TripleClass::SchemaInfra`] than as typed *data*: it is
//!   real and load-bearing, but it is schema machinery, not a fact about the
//!   world the way `(note, email, "jane@example.org")` is. Counting it as
//!   `TypedCore` would inflate "how much of my world-knowledge is typed" with
//!   plumbing.
//! - The `ElicitPass` relations are **not** front-loaded: `ElicitPass`
//!   ([`crate::bootstrap::knowledge_elicit::ElicitPass`]) names *pipeline
//!   passes* (`Taxonomy`, `Properties`, `Relationships`, `Procedures`,
//!   `Temporal`), not a fixed predicate vocabulary — there is no stable set
//!   of relation *labels* to hardcode here. Triples the Librarian/elicitation
//!   pipeline emits classify the same as any other triple, by predicate
//!   shape; a future slice (4/5 — Librarian projection / promotion) is the
//!   right place to register any predicates it standardizes on.
//!
//! ## No physical marking pass
//!
//! Per the brief: this slice deliberately does **not** add a "grandfather"
//! job that stamps existing triples. [`classify_triple`] already answers
//! "is this frontier?" for every triple, old or new, without a stored flag —
//! adding one would be a second, driftable source of truth for a question
//! the classification rule already answers for free.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::Serialize;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::compile::{PRED_IS_A, PRED_PROPERTY, RELATION_MARKER, TYPE_MARKER};
use super::project::FRONTIER_PREFIX;

/// Compartment-label prefix a schema pack installs its own triples under
/// ([`super::compile::CompiledSchemaPack::microtheory_label`]).
const SCHEMA_PACK_COMPARTMENT_PREFIX: &str = "mt:schema/";

/// Predicate-label prefixes that are schema-system bookkeeping, not data
/// about the world, wherever they are asserted (see module docs).
const SCHEMA_INFRA_PREFIXES: &[&str] = &["onto:", "ctx:"];

/// Predicate labels front-loaded into the core vocabulary unconditionally
/// (issue 9's "register the existing relation namespaces… into the seed core
/// vocabulary at install" — see module docs for why `is-a` and `ctx:` are
/// handled separately instead of listed here).
pub const FRONTLOADED_CORE_NAMES: &[&str] = &["part-of"];

/// Predicate-label prefixes front-loaded into the core vocabulary
/// unconditionally.
pub const FRONTLOADED_CORE_PREFIXES: &[&str] = &["agent:", "doc:", "akh:", "rel:"];

/// Default number of frontier predicates [`coverage_report`] lists, ranked by
/// occurrence count — the Librarian/promotion slice's recurrence signal
/// (issue 5: "a recurring frontier term generates a promotion proposal").
pub const DEFAULT_TOP_FRONTIER: usize = 20;

/// The compartment label used for triples in the default graph (no
/// `compartment_id`) — mirrors [`super::install::install_pack`]'s existing
/// `"(default graph)"` convention so reports and error messages agree.
pub const DEFAULT_GRAPH_LABEL: &str = "(default graph)";

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// What a triple *is*, structurally, with respect to the installed schema —
/// see the module docs for the exact rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TripleClass {
    /// A data triple backed by an installed schema pack.
    TypedCore,
    /// Everything not demonstrably backed by an installed pack — the
    /// grandfathering default. Includes explicit `note:frontier/` markers.
    Frontier,
    /// The schema system's own bookkeeping (pack compartments, `onto:`/`ctx:`
    /// predicates) — not data about the world.
    SchemaInfra,
}

/// Precomputed index over the live KG's installed-schema vocabulary, built
/// once per [`coverage_report`] call and consulted by [`classify_triple`] for
/// every triple. Three linear passes over the triple set (types →
/// relations/properties → typed subjects), not a per-triple graph query, so
/// classifying the whole KG stays `O(n)` rather than `O(n)` queries each
/// walking the graph again.
#[derive(Debug, Clone, Default)]
pub struct SchemaView {
    /// Symbols marked `(T, is-a, onto:Type)` by *some* installed pack —
    /// pack-declared types, regardless of which pack.
    declared_types: HashSet<SymbolId>,
    /// Property labels declared on some pack-declared type via
    /// `(T, onto:property, p)`.
    declared_property_names: HashSet<String>,
    /// Relation labels declared by some pack via `(p, is-a, relation)`.
    declared_relation_names: HashSet<String>,
    /// Subjects with an `is-a` edge to a `declared_types` member, asserted
    /// outside any pack's own compartment (i.e. an entity/note typing edge,
    /// not a pack's internal subtype-chain declaration).
    typed_subjects: HashSet<SymbolId>,
}

impl SchemaView {
    /// Build the index from the engine's current triple set.
    pub fn build(engine: &Engine) -> Self {
        Self::build_from(engine, &engine.all_triples())
    }

    /// Build the index from an already-materialized triple set (avoids a
    /// second `all_triples()` copy when the caller already has one, as
    /// [`coverage_report`] does).
    fn build_from(engine: &Engine, all: &[Triple]) -> Self {
        let is_a = engine.resolve_symbol(PRED_IS_A).ok();
        let type_marker = engine.resolve_symbol(TYPE_MARKER).ok();
        let relation_marker = engine.resolve_symbol(RELATION_MARKER).ok();
        let property_pred = engine.resolve_symbol(PRED_PROPERTY).ok();

        let mut declared_types = HashSet::new();
        let mut declared_relation_names = HashSet::new();
        if let Some(is_a) = is_a {
            for t in all {
                if t.predicate != is_a {
                    continue;
                }
                if Some(t.object) == type_marker {
                    declared_types.insert(t.subject);
                } else if Some(t.object) == relation_marker {
                    declared_relation_names.insert(engine.resolve_label(t.subject));
                }
            }
        }

        let mut declared_property_names = HashSet::new();
        if let Some(property_pred) = property_pred {
            for t in all {
                if t.predicate == property_pred && declared_types.contains(&t.subject) {
                    declared_property_names.insert(engine.resolve_label(t.object));
                }
            }
        }

        let mut typed_subjects = HashSet::new();
        if let Some(is_a) = is_a {
            for t in all {
                if t.predicate == is_a
                    && declared_types.contains(&t.object)
                    && !is_schema_pack_compartment(t.compartment_id.as_deref())
                {
                    typed_subjects.insert(t.subject);
                }
            }
        }

        Self {
            declared_types,
            declared_property_names,
            declared_relation_names,
            typed_subjects,
        }
    }
}

fn is_schema_pack_compartment(compartment_id: Option<&str>) -> bool {
    compartment_id
        .map(|c| c.starts_with(SCHEMA_PACK_COMPARTMENT_PREFIX))
        .unwrap_or(false)
}

/// Classify one triple against the installed-schema index — see the module
/// docs for the exact rule. Never fails: every triple lands in exactly one
/// bucket.
pub fn classify_triple(engine: &Engine, view: &SchemaView, triple: &Triple) -> TripleClass {
    if is_schema_pack_compartment(triple.compartment_id.as_deref()) {
        return TripleClass::SchemaInfra;
    }

    let predicate_label = engine.resolve_label(triple.predicate);

    if SCHEMA_INFRA_PREFIXES
        .iter()
        .any(|prefix| predicate_label.starts_with(prefix))
    {
        return TripleClass::SchemaInfra;
    }

    if predicate_label.starts_with(FRONTIER_PREFIX) {
        return TripleClass::Frontier;
    }

    if predicate_label == PRED_IS_A {
        // The type-establishing edge itself: core once (and only once) its
        // object names an installed type. No installed `T` ⇒ frontier — this
        // is the "entity typing stays lazy" half of the grandfathering rule.
        return if view.declared_types.contains(&triple.object) {
            TripleClass::TypedCore
        } else {
            TripleClass::Frontier
        };
    }

    if FRONTLOADED_CORE_NAMES.contains(&predicate_label.as_str())
        || FRONTLOADED_CORE_PREFIXES
            .iter()
            .any(|prefix| predicate_label.starts_with(prefix))
    {
        return TripleClass::TypedCore;
    }

    let predicate_declared = view.declared_property_names.contains(&predicate_label)
        || view.declared_relation_names.contains(&predicate_label);
    if predicate_declared && view.typed_subjects.contains(&triple.subject) {
        return TripleClass::TypedCore;
    }

    // Grandfathering default: an undeclared predicate (or a declared one on
    // an untyped subject) is frontier — no separate migration path needed.
    TripleClass::Frontier
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Per-compartment tally, keyed by compartment label
/// ([`DEFAULT_GRAPH_LABEL`] for triples with no compartment).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CompartmentBreakdown {
    pub typed_core: usize,
    pub frontier: usize,
    pub schema_infra: usize,
    pub total: usize,
}

/// The schema-coverage report: a non-LLM, read-only snapshot of how much of
/// the live KG is typed-core vs. frontier vs. schema bookkeeping. Run anytime
/// via [`coverage_report`] — this is visibility, not enforcement (ADR 081:
/// writes are never blocked).
#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    pub total_triples: usize,
    pub typed_core: usize,
    pub frontier: usize,
    pub schema_infra: usize,
    /// `typed_core / (typed_core + frontier) * 100` — percentage of *data*
    /// triples (excluding schema bookkeeping) that are typed. The more
    /// meaningful "is the core crystallizing?" number, since schema-infra
    /// triples aren't world-knowledge to type in the first place.
    pub percent_typed_of_data: f64,
    /// `typed_core / total_triples * 100` — the same numerator over every
    /// triple in the KG, schema bookkeeping included. Reported alongside
    /// `percent_typed_of_data` so a caller can see both views without
    /// recomputing.
    pub percent_typed_of_total: f64,
    pub per_compartment: BTreeMap<String, CompartmentBreakdown>,
    /// Frontier predicates ranked by occurrence count, descending (ties break
    /// lexicographically for determinism), truncated to the requested top-N.
    /// This is the Librarian/promotion slice's recurrence signal (issue 5).
    pub top_frontier_predicates: Vec<(String, usize)>,
}

/// Run the schema-coverage report over the live KG with the default top-N
/// frontier-predicate list length ([`DEFAULT_TOP_FRONTIER`]).
pub fn coverage_report(engine: &Engine) -> CoverageReport {
    coverage_report_top_n(engine, DEFAULT_TOP_FRONTIER)
}

/// Run the schema-coverage report with an explicit top-N frontier-predicate
/// list length.
pub fn coverage_report_top_n(engine: &Engine, top_n: usize) -> CoverageReport {
    let all = engine.all_triples();
    let view = SchemaView::build_from(engine, &all);

    let mut typed_core = 0usize;
    let mut frontier = 0usize;
    let mut schema_infra = 0usize;
    let mut frontier_predicate_counts: HashMap<String, usize> = HashMap::new();
    let mut per_compartment: BTreeMap<String, CompartmentBreakdown> = BTreeMap::new();

    for t in &all {
        let class = classify_triple(engine, &view, t);
        let compartment_label = t
            .compartment_id
            .clone()
            .unwrap_or_else(|| DEFAULT_GRAPH_LABEL.to_string());
        let entry = per_compartment.entry(compartment_label).or_default();
        entry.total += 1;

        match class {
            TripleClass::TypedCore => {
                typed_core += 1;
                entry.typed_core += 1;
            }
            TripleClass::Frontier => {
                frontier += 1;
                entry.frontier += 1;
                let predicate_label = engine.resolve_label(t.predicate);
                *frontier_predicate_counts
                    .entry(predicate_label)
                    .or_insert(0) += 1;
            }
            TripleClass::SchemaInfra => {
                schema_infra += 1;
                entry.schema_infra += 1;
            }
        }
    }

    let mut top_frontier_predicates: Vec<(String, usize)> =
        frontier_predicate_counts.into_iter().collect();
    top_frontier_predicates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top_frontier_predicates.truncate(top_n);

    let total_triples = all.len();
    let data_triples = typed_core + frontier;
    let percent_typed_of_data = if data_triples == 0 {
        0.0
    } else {
        (typed_core as f64 / data_triples as f64) * 100.0
    };
    let percent_typed_of_total = if total_triples == 0 {
        0.0
    } else {
        (typed_core as f64 / total_triples as f64) * 100.0
    };

    CoverageReport {
        total_triples,
        typed_core,
        frontier,
        schema_infra,
        percent_typed_of_data,
        percent_typed_of_total,
        per_compartment,
        top_frontier_predicates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
    use crate::library::note::{Frontmatter, Note};
    use crate::schema::compile::compile;
    use crate::schema::install::install_pack;
    use crate::schema::project::project_note;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: None,
            ..Default::default()
        })
        .unwrap()
    }

    const SEED_PACK: &str = r#"{
        "pack": {"name": "core-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "email", "value_type": "text"},
                {"name": "birthDate", "value_type": "date"}]},
            {"name": "Organization"}
        ],
        "relations": [
            {"name": "worksFor", "arg1_type": "Person", "arg2_type": "Organization"}
        ]
    }"#;

    fn install_seed(engine: &Engine) {
        let pack = compile(SEED_PACK).unwrap();
        install_pack(engine, &pack).unwrap();
    }

    fn sample_document(id: &str) -> DocumentRecord {
        DocumentRecord {
            id: id.into(),
            title: "Test Document".into(),
            source: DocumentSource::Inline,
            format: ContentFormat::PlainText,
            tags: vec![],
            chunk_count: 0,
            triple_count: 0,
            ingested_at: 1_700_000_000,
        }
    }

    fn person_note(doc_id: &str) -> Note {
        let mut fm = Frontmatter::new();
        fm.insert(
            "type",
            crate::library::note::FrontmatterValue::Text("Person".into()),
        );
        fm.insert(
            "email",
            crate::library::note::FrontmatterValue::Text("jane@example.org".into()),
        );
        fm.insert(
            "birthDate",
            crate::library::note::FrontmatterValue::Date("1990-01-01".into()),
        );
        fm.insert(
            "favoriteColor",
            crate::library::note::FrontmatterValue::Text("teal".into()),
        );
        Note::new(sample_document(doc_id), "Colleague of [[Alice Example]].").with_frontmatter(fm)
    }

    #[test]
    fn empty_engine_yields_zero_report() {
        let engine = test_engine();
        let report = coverage_report(&engine);
        assert_eq!(report.total_triples, 0);
        assert_eq!(report.typed_core, 0);
        assert_eq!(report.frontier, 0);
        assert_eq!(report.schema_infra, 0);
        assert_eq!(report.percent_typed_of_data, 0.0);
        assert_eq!(report.percent_typed_of_total, 0.0);
        assert!(report.top_frontier_predicates.is_empty());
        assert!(report.per_compartment.is_empty());
    }

    #[test]
    fn installed_pack_triples_are_schema_infra_only() {
        let engine = test_engine();
        install_seed(&engine);

        let report = coverage_report(&engine);
        assert_eq!(
            report.typed_core, 0,
            "no instance data yet, just the pack's own vocabulary"
        );
        assert_eq!(report.frontier, 0);
        assert!(report.schema_infra > 0);
        assert_eq!(report.total_triples, report.schema_infra);

        // The pack's compartment shows up in the per-compartment breakdown.
        let mt_label = "mt:schema/core-test@0.1.0";
        let entry = report.per_compartment.get(mt_label).unwrap();
        assert_eq!(entry.schema_infra, entry.total);
        assert_eq!(entry.typed_core, 0);
        assert_eq!(entry.frontier, 0);
    }

    #[test]
    fn typed_note_projects_is_a_and_properties_as_typed_core() {
        let engine = test_engine();
        install_seed(&engine);
        let note = person_note("doc-1");
        project_note(&engine, &note).unwrap();

        let report = coverage_report(&engine);
        // note is-a Person, email, birthDate ⇒ 3 typed-core data triples.
        assert_eq!(report.typed_core, 3);
        // favoriteColor (frontier marker) + links-to (undeclared predicate).
        assert_eq!(report.frontier, 2);
        assert!(report.schema_infra > 0);

        let default_graph = report.per_compartment.get(DEFAULT_GRAPH_LABEL).unwrap();
        assert_eq!(default_graph.typed_core, 3);
        assert_eq!(default_graph.frontier, 2);

        let frontier_labels: Vec<&str> = report
            .top_frontier_predicates
            .iter()
            .map(|(label, _)| label.as_str())
            .collect();
        assert!(frontier_labels.contains(&"note:frontier/favoriteColor"));
        assert!(frontier_labels.contains(&"links-to"));
    }

    #[test]
    fn is_a_to_uninstalled_type_is_frontier_not_typed_core() {
        let engine = test_engine();
        install_seed(&engine);

        // A legacy-shaped is-a triple whose object was never installed as a
        // schema-pack type — "entity typing stays lazy": this must NOT count
        // as typed just because `is-a` is a familiar predicate.
        let socrates = engine.resolve_or_create_entity("Socrates").unwrap();
        let human = engine.resolve_or_create_entity("Human").unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();
        engine
            .add_triple(&Triple::new(socrates, is_a, human))
            .unwrap();

        let report = coverage_report(&engine);
        assert_eq!(report.typed_core, 0);
        assert_eq!(report.frontier, 1);
    }

    #[test]
    fn raw_legacy_triple_via_resolve_or_create_is_frontier() {
        let engine = test_engine();
        install_seed(&engine);

        // Exactly the "legacy triples asserted outside the typed projection"
        // case: no note, no schema declaration, just a raw assert.
        let foo = engine.resolve_or_create_entity("Foo").unwrap();
        let baz = engine.resolve_or_create_entity("Baz").unwrap();
        let bar = engine.resolve_or_create_relation("bar").unwrap();
        engine.add_triple(&Triple::new(foo, bar, baz)).unwrap();

        let report = coverage_report(&engine);
        assert_eq!(report.frontier, 1);
        assert_eq!(report.typed_core, 0);
        assert_eq!(report.top_frontier_predicates, vec![("bar".to_string(), 1)]);
    }

    #[test]
    fn frontloaded_namespaces_are_typed_core_regardless_of_subject_typing() {
        let engine = test_engine();
        install_seed(&engine);

        // No pack declares "agent:goal" or "part-of" — they are front-loaded
        // core relation namespaces (issue 9), independent of subject typing.
        let subject = engine.resolve_or_create_entity("untyped-thing").unwrap();
        let object = engine.resolve_or_create_entity("some-goal").unwrap();
        let agent_pred = engine.resolve_or_create_relation("agent:goal").unwrap();
        engine
            .add_triple(&Triple::new(subject, agent_pred, object))
            .unwrap();

        let part_of = engine.resolve_or_create_relation("part-of").unwrap();
        let whole = engine.resolve_or_create_entity("whole-thing").unwrap();
        engine
            .add_triple(&Triple::new(subject, part_of, whole))
            .unwrap();

        let report = coverage_report(&engine);
        assert_eq!(report.typed_core, 2);
        assert_eq!(report.frontier, 0);
    }

    #[test]
    fn ctx_and_onto_predicates_are_schema_infra_outside_pack_compartments() {
        let engine = test_engine();
        install_seed(&engine);

        let a = engine
            .resolve_or_create_entity("mt:akh/test-effective")
            .unwrap();
        let mt = engine.resolve_symbol("mt:schema/core-test@0.1.0").unwrap();
        let specializes = engine
            .resolve_or_create_relation("ctx:specializes")
            .unwrap();
        engine.add_triple(&Triple::new(a, specializes, mt)).unwrap();

        let report = coverage_report(&engine);
        // The activation edge lives outside the pack's own compartment (it's
        // asserted by `activate_pack`, not `install_pack`) but still counts
        // as schema-infra by predicate namespace.
        assert!(report.schema_infra > 0);
        assert_eq!(report.frontier, 0);
        assert_eq!(report.typed_core, 0);
    }

    #[test]
    fn top_frontier_predicates_ordered_by_count_descending() {
        let engine = test_engine();
        install_seed(&engine);

        let common = engine.resolve_or_create_relation("common").unwrap();
        let rare = engine.resolve_or_create_relation("rare").unwrap();
        for i in 0..3 {
            let s = engine.resolve_or_create_entity(&format!("s{i}")).unwrap();
            let o = engine.resolve_or_create_entity(&format!("o{i}")).unwrap();
            engine.add_triple(&Triple::new(s, common, o)).unwrap();
        }
        let s = engine.resolve_or_create_entity("s-rare").unwrap();
        let o = engine.resolve_or_create_entity("o-rare").unwrap();
        engine.add_triple(&Triple::new(s, rare, o)).unwrap();

        let report = coverage_report(&engine);
        assert_eq!(
            report.top_frontier_predicates,
            vec![("common".to_string(), 3), ("rare".to_string(), 1)]
        );
    }

    #[test]
    fn top_n_truncates_frontier_predicate_list() {
        let engine = test_engine();
        install_seed(&engine);

        for i in 0..5 {
            let s = engine.resolve_or_create_entity(&format!("s{i}")).unwrap();
            let o = engine.resolve_or_create_entity(&format!("o{i}")).unwrap();
            let p = engine
                .resolve_or_create_relation(&format!("pred{i}"))
                .unwrap();
            engine.add_triple(&Triple::new(s, p, o)).unwrap();
        }

        let report = coverage_report_top_n(&engine, 2);
        assert_eq!(report.frontier, 5);
        assert_eq!(report.top_frontier_predicates.len(), 2);
    }

    #[test]
    fn percentages_are_sane_and_bounded() {
        let engine = test_engine();
        install_seed(&engine);
        project_note(&engine, &person_note("doc-2")).unwrap();

        let report = coverage_report(&engine);
        assert!(report.percent_typed_of_data >= 0.0 && report.percent_typed_of_data <= 100.0);
        assert!(report.percent_typed_of_total >= 0.0 && report.percent_typed_of_total <= 100.0);
        // typed_core=3, frontier=2 ⇒ 60%.
        assert!((report.percent_typed_of_data - 60.0).abs() < 1e-9);
        // percent_typed_of_data includes schema-infra in the denominator's
        // complement, so it must be <= percent_typed_of_data.
        assert!(report.percent_typed_of_total <= report.percent_typed_of_data);
    }

    #[test]
    fn reinstalling_existing_corpus_keeps_working_unchanged() {
        // "Existing triples keep working unchanged after the core is
        // introduced (no downtime)" — asserting raw legacy triples both
        // before and after a pack install must classify identically and must
        // not error or require any migration call.
        let engine = test_engine();
        let foo = engine.resolve_or_create_entity("Foo").unwrap();
        let baz = engine.resolve_or_create_entity("Baz").unwrap();
        let bar = engine.resolve_or_create_relation("bar").unwrap();
        engine.add_triple(&Triple::new(foo, bar, baz)).unwrap();

        let before = coverage_report(&engine);
        assert_eq!(before.frontier, 1);

        install_seed(&engine);

        let after = coverage_report(&engine);
        // The pre-existing triple is still there, still frontier, untouched.
        assert_eq!(after.frontier, 1);
        assert!(after.schema_infra > 0);
    }
}
