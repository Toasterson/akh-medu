//! Note → typed-KG projection (ADR 081 Slice 1: thin note → projection →
//! typed-KG round-trip, `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`
//! work-item 1).
//!
//! This is the **mechanical** projector — no LLM. A [`Note`] declares its
//! type via a `type` frontmatter key (e.g. `type: Person`); [`project_note`]
//! checks that name against the types already installed by
//! [`super::install::install_pack`] (the `(T, is-a, onto:Type)` triples a
//! schema pack projects into the KG) and, mechanically, emits:
//!
//! - `(note-entity, is-a, TypeSymbol)` when the type is known,
//! - one `(note-entity, property, value-entity)` triple per frontmatter key
//!   the type's schema declares (`onto:property` on the type),
//! - one `(note-entity, note:frontier/{key}, value-entity)` triple per
//!   frontmatter key the schema does **not** declare for that type — ADR 081
//!   decision 7, "writes are never blocked": unknown keys are still
//!   asserted, just distinguishably marked as frontier,
//! - one `(note-entity, links-to, target-entity)` triple per wikilink
//!   extracted from the note body (structural, independent of typing).
//!
//! Lives in `schema::` (not `library::`) because it is fundamentally a
//! *schema* operation: it reads the native-triple vocabulary
//! ([`super::compile::PRED_IS_A`], [`super::compile::PRED_PROPERTY`],
//! [`super::compile::TYPE_MARKER`]) that [`super::install::install_pack`]
//! projected, and extends [`SchemaError`] rather than
//! `crate::library::error::LibraryError`. `library::note` already depends on
//! `schema::source::ValueType`, so the reverse edge here (`schema` reading
//! `library::note::Note`) does not introduce a module cycle — both live in
//! one crate.
//!
//! **Slice 3 update** (`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`
//! work-item 3): a property no longer has to be declared *directly* on the
//! note's exact type to count as typed — [`super::shape::effective_properties`]
//! walks the installed `is-a` chain upward (e.g. `Person`'s `name`, inherited
//! from `Thing`), so inherited properties project as typed too. Each typed
//! property's direct-vs-inherited provenance is preserved on
//! [`ProjectionReport::typed_properties`]. Property-shape validation
//! ([`super::shape::check_value_shape`] + required-property completeness) is
//! also applied here: a value whose runtime [`crate::schema::source::ValueType`]
//! does not match the property's declared shape lands in frontier instead of
//! typed, and a missing required property is reported — both are advisory
//! ([`ProjectionReport::shape_violations`]), never blocking the write (ADR
//! 081 decision 7).
//!
//! **Deliberately out of scope for this slice** (see the issue list):
//! - Relation-valued frontmatter properties. Notes today only project
//!   `(note-entity, property, value-entity)` triples where the value is a
//!   literal-ish frontmatter value (text/number/date/list/link/checkbox) —
//!   there is no notion yet of a frontmatter key naming a *relation*
//!   (`RelationDef` with `arg1_type`/`arg2_type`) rather than a property.
//!   Declared relation arity/arg-type validation is therefore not exercised
//!   here; it applies once notes gain a way to author relation triples
//!   (work-item 5+ territory).
//!
//! **Slice 8 update** (work-item 8, "Evolution: diff-supersede + TMS +
//! multi-akh DS conflict"): [`reproject_note`] closes the gap the paragraph
//! above used to describe. Re-projecting an *unchanged* note stays
//! triple-level idempotent (unchanged); re-projecting a *changed* note now
//! diffs the new triple set against a durable per-note **projection
//! manifest** and retracts exactly the triples that dropped out — see
//! [`reproject_note`]'s doc comment for the manifest shape and why.

use std::collections::{BTreeMap, HashSet};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::graph::identity::TripleIdentity;
use crate::library::note::{Note, NoteId, FrontmatterValue};
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::compile::PRED_IS_A;
use super::shape::{self, EffectivePropertySet, ShapeViolation};
use super::{SchemaError, SchemaResult};

/// `note-entity --links-to--> target-entity`, one per wikilink.
pub const PRED_LINKS_TO: &str = "links-to";
/// Prefix for frontier-marker predicates: `note:frontier/{frontmatter-key}`.
/// Keeps frontier facts distinguishable from typed ones by predicate label
/// alone (ADR 081 decision 7), without needing a second compartment.
pub const FRONTIER_PREFIX: &str = "note:frontier/";
/// The frontmatter key that names a note's declared schema type.
pub const TYPE_FRONTMATTER_KEY: &str = "type";

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// What happened to a note's declared `type:` frontmatter key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeOutcome {
    /// The declared type is installed in this akh's effective schema; a
    /// `(note-entity, is-a, TypeSymbol)` triple was asserted.
    Typed {
        type_symbol: SymbolId,
        type_name: String,
    },
    /// A `type:` key was present but does not name an installed type. No
    /// `is-a` triple was asserted; every frontmatter field (including the
    /// `type` key itself) landed in frontier.
    UnknownType { type_name: String },
    /// No `type:` frontmatter key at all. No `is-a` triple was asserted;
    /// every frontmatter field (if any) landed in frontier.
    Untyped,
}

impl TypeOutcome {
    /// Whether the note projected onto a known type.
    pub fn is_typed(&self) -> bool {
        matches!(self, TypeOutcome::Typed { .. })
    }

    /// A diagnostic-friendly explanation for the non-happy paths — `None`
    /// when the note typed cleanly. ADR 081: typing errors surface at read
    /// time, never block the write, so this is informational, not an
    /// [`SchemaError`].
    pub fn diagnostic(&self) -> Option<String> {
        match self {
            TypeOutcome::Typed { .. } => None,
            TypeOutcome::UnknownType { type_name } => Some(format!(
                "type \"{type_name}\" is not installed in this akh's effective schema — the \
                 note's frontmatter projected entirely to frontier instead of the typed core. \
                 Install (and activate) the schema pack that declares \"{type_name}\", then \
                 re-project the note."
            )),
            TypeOutcome::Untyped => None,
        }
    }
}

/// One frontmatter key that projected as a typed property (Slice 3): which
/// type it is declared on, and whether that declaration is direct (on the
/// note's exact type) or inherited via the `is-a` chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedPropertyProjection {
    /// The frontmatter key / property name.
    pub key: String,
    /// The type this property is declared directly on (the note's exact
    /// type when [`Self::direct`] is `true`, an ancestor otherwise).
    pub declared_on: String,
    /// Whether the property is declared directly on the note's exact type
    /// (not inherited via `is-a`).
    pub direct: bool,
}

/// Result of mechanically projecting one [`Note`] into the typed KG.
#[derive(Debug, Clone)]
pub struct ProjectionReport {
    pub note_id: NoteId,
    /// The `note:{id}` KG entity the note projected onto.
    pub note_entity: SymbolId,
    pub type_outcome: TypeOutcome,
    /// Frontmatter keys that matched a property the declared type carries
    /// (own or inherited) *and* passed shape validation. Equal to
    /// `typed_properties.len()`.
    pub typed_property_count: usize,
    /// Frontmatter keys that did not (unknown type, unknown key, or a shape
    /// violation demoted an otherwise-matching key to frontier).
    pub frontier_property_count: usize,
    /// Per-key detail for every typed property, including the
    /// direct-vs-inherited distinction (Slice 3; feeds the coverage report,
    /// work-item 9).
    pub typed_properties: Vec<TypedPropertyProjection>,
    /// Property-shape violations detected during projection (Slice 3): a
    /// type mismatch demotes the value to frontier instead of typed; a
    /// missing required property is reported without demoting anything.
    /// Never blocks the write (ADR 081 decision 7) — advisory only.
    pub shape_violations: Vec<ShapeViolation>,
    /// Wikilinks projected as `links-to` triples.
    pub wikilink_count: usize,
    /// `None` when the engine has no provenance ledger configured
    /// (in-memory engine, `EngineConfig.data_dir: None`) — projection still
    /// succeeds without persisted provenance (mirrors
    /// `agent::harvest::mint_turn_episode`'s `let _ = engine.store_provenance(..)`
    /// tolerance, made visible here as `None` instead of silently dropped).
    pub provenance_id: Option<ProvenanceId>,
    /// The exact `(subject, predicate, object)` triples this projection pass
    /// asserted: the `is-a` triple (if typed), one triple per frontmatter key
    /// (typed or frontier), and one `links-to` triple per wikilink. Asserting
    /// an unchanged triple again is a no-op at the KG level (idempotent
    /// dedup), so this list is the projector's *full current intent*, not
    /// just what changed this call — [`reproject_note`] diffs it against the
    /// prior manifest to find out what changed.
    pub asserted_triples: Vec<Triple>,
    /// Diff-based supersede counts (ADR 081 decision 12, work-item 8).
    /// `None` for a plain [`project_note`] call — no manifest diffing is
    /// performed, so there is nothing to report. `Some` when this report was
    /// produced by [`reproject_note`].
    pub diff: Option<ProjectionDiff>,
}

/// How a [`reproject_note`] call's new triple set compared to the note's
/// prior projection manifest (work-item 8 acceptance: "editing a fact
/// retracts only the changed triple(s); unchanged triples keep their
/// original provenance/confidence").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProjectionDiff {
    /// Triples newly asserted this pass (present now, absent from the prior
    /// manifest).
    pub added: usize,
    /// Triples unchanged since the prior projection (present in both the new
    /// set and the prior manifest) — untouched, no provenance/confidence
    /// churn.
    pub retained: usize,
    /// Triples retracted this pass (present in the prior manifest, absent
    /// from the new set) — removed via [`Engine::remove_triple`]'s TMS
    /// cascade, so `TruthMaintenanceSystem::was_retracted` sees them and
    /// Phase 44g recall hygiene filters them out, while the retraction
    /// itself is preserved as a [`DerivationKind::NoteReprojection`]
    /// provenance record (history is never deleted, only marked superseded).
    pub retracted: usize,
}

impl ProjectionDiff {
    /// True when this reprojection changed nothing — an ideal Retain outcome.
    pub fn is_unchanged(&self) -> bool {
        self.added == 0 && self.retracted == 0
    }
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// Mechanically project a [`Note`] into the typed KG. No LLM; never fails on
/// a typing problem (ADR 081 decision 7, "writes are never blocked") — the
/// only errors this returns are genuine engine-side failures (symbol
/// allocation, constraint violations), reported via [`SchemaError::Projection`].
///
/// Idempotent at the triple level for an *unchanged* note: re-projecting
/// asserts the same `(subject, predicate, object)` triples, which
/// [`crate::graph::index::KnowledgeIndex::insert_triple`] already dedups
/// (existing edge confidence/provenance is updated in place rather than a
/// duplicate edge being created). A *changed* note simply adds the new
/// triples; diff-based supersede is work-item 8, not this slice.
pub fn project_note(engine: &Engine, note: &Note) -> SchemaResult<ProjectionReport> {
    let note_label = format!("note:{}", note.id.as_str());
    let note_entity = engine
        .resolve_or_create_entity(&note_label)
        .map_err(|e| projection_err(&note.id, format!("note entity: {e}")))?;
    let is_a = engine
        .resolve_or_create_relation(PRED_IS_A)
        .map_err(|e| projection_err(&note.id, format!("is-a relation: {e}")))?;

    let mut sources = vec![note_entity];
    let mut asserted_triples = Vec::new();

    // --- Decide the type outcome first: it gates which frontmatter keys
    // count as "typed" vs. "frontier". ---
    let type_outcome = match note.frontmatter.get(TYPE_FRONTMATTER_KEY) {
        None => TypeOutcome::Untyped,
        Some(value) => {
            let type_name = value_label(value);
            match resolve_installed_type(engine, &type_name) {
                Some(type_symbol) => TypeOutcome::Typed {
                    type_symbol,
                    type_name,
                },
                None => TypeOutcome::UnknownType { type_name },
            }
        }
    };

    let effective = if let TypeOutcome::Typed { type_symbol, .. } = &type_outcome {
        let type_triple = Triple::new(note_entity, is_a, *type_symbol);
        engine
            .add_triple(&type_triple)
            .map_err(|e| projection_err(&note.id, format!("type triple: {e}")))?;
        asserted_triples.push(type_triple);
        sources.push(*type_symbol);
        // Slice 3: walk the is-a chain for the effective (own + inherited)
        // property set. A subtype cycle is a schema-authoring bug, not a
        // note-write problem — degrade to "no declared properties" (every
        // key lands in frontier) instead of failing the write (ADR 081
        // decision 7).
        shape::effective_properties(engine, *type_symbol)
            .unwrap_or_else(|_| EffectivePropertySet::empty(engine.resolve_label(*type_symbol)))
    } else {
        EffectivePropertySet::default()
    };

    // --- Frontmatter properties: typed vs. frontier, shape-validated. ---
    let mut typed_property_count = 0usize;
    let mut frontier_property_count = 0usize;
    let mut typed_properties = Vec::new();
    let mut shape_violations = Vec::new();
    for (key, value) in note.frontmatter.iter() {
        // The `type` key itself is framing metadata, already captured by the
        // `is-a` triple above when the type is known. When the type is
        // unknown/absent it still lands in frontier below, alongside every
        // other field, so it is never silently dropped.
        if key == TYPE_FRONTMATTER_KEY && matches!(type_outcome, TypeOutcome::Typed { .. }) {
            continue;
        }

        let value_entity = engine
            .resolve_or_create_entity(&value_label(value))
            .map_err(|e| projection_err(&note.id, format!("value entity '{key}': {e}")))?;
        sources.push(value_entity);

        let property = effective.get(key);
        let violation = property.and_then(|p| shape::check_value_shape(p, value));
        let has_violation = violation.is_some();
        if let Some(v) = violation {
            shape_violations.push(v);
        }

        match property {
            Some(property) if !has_violation => {
                let predicate = engine
                    .resolve_or_create_relation(key)
                    .map_err(|e| projection_err(&note.id, format!("property predicate '{key}': {e}")))?;
                let property_triple = Triple::new(note_entity, predicate, value_entity);
                engine
                    .add_triple(&property_triple)
                    .map_err(|e| projection_err(&note.id, format!("property triple '{key}': {e}")))?;
                asserted_triples.push(property_triple);
                typed_property_count += 1;
                typed_properties.push(TypedPropertyProjection {
                    key: key.to_string(),
                    declared_on: property.declared_on.clone(),
                    direct: property.direct,
                });
            }
            _ => {
                // Unknown key, or a shape violation demoted a known
                // property — either way the value is still written, just
                // marked frontier instead of typed (ADR 081 decision 7).
                let frontier_label = format!("{FRONTIER_PREFIX}{key}");
                let predicate = engine
                    .resolve_or_create_relation(&frontier_label)
                    .map_err(|e| projection_err(&note.id, format!("frontier predicate '{key}': {e}")))?;
                let frontier_triple = Triple::new(note_entity, predicate, value_entity);
                engine
                    .add_triple(&frontier_triple)
                    .map_err(|e| projection_err(&note.id, format!("frontier triple '{key}': {e}")))?;
                asserted_triples.push(frontier_triple);
                frontier_property_count += 1;
            }
        }
    }

    // --- Required-property completeness (Slice 3): advisory only, never
    // blocks the write. ---
    if matches!(type_outcome, TypeOutcome::Typed { .. }) {
        for prop in effective.required() {
            if note.frontmatter.get(&prop.name).is_none() {
                shape_violations.push(ShapeViolation::MissingRequired {
                    property: prop.name.clone(),
                    declared_on: prop.declared_on.clone(),
                });
            }
        }
    }

    // --- Wikilinks: structural, independent of typing. ---
    let links_to = engine
        .resolve_or_create_relation(PRED_LINKS_TO)
        .map_err(|e| projection_err(&note.id, format!("links-to relation: {e}")))?;
    let mut wikilink_count = 0usize;
    for link in &note.wikilinks {
        let target = engine
            .resolve_or_create_entity(&link.target)
            .map_err(|e| projection_err(&note.id, format!("wikilink target '{}': {e}", link.target)))?;
        let link_triple = Triple::new(note_entity, links_to, target);
        engine
            .add_triple(&link_triple)
            .map_err(|e| projection_err(&note.id, format!("wikilink triple '{}': {e}", link.target)))?;
        asserted_triples.push(link_triple);
        sources.push(target);
        wikilink_count += 1;
    }

    // --- Provenance: one record per projection call (see ProjectionReport
    // docs — re-projection is triple-level idempotent, not provenance-level;
    // each call still records that a projection pass happened). ---
    let mut prov = ProvenanceRecord::new(
        note_entity,
        DerivationKind::AgentConsolidation {
            reason: format!("note_projection:{}", note.id),
            relevance_score: 1.0,
        },
    )
    .with_sources(sources)
    .with_confidence(1.0);
    let provenance_id = engine.store_provenance(&mut prov).ok();

    Ok(ProjectionReport {
        note_id: note.id.clone(),
        note_entity,
        type_outcome,
        typed_property_count,
        frontier_property_count,
        typed_properties,
        shape_violations,
        wikilink_count,
        provenance_id,
        asserted_triples,
        diff: None,
    })
}

// ---------------------------------------------------------------------------
// Slice 8: diff-based supersede (projection manifest + reproject_note)
// ---------------------------------------------------------------------------

/// Compartment the per-note projection manifest lives in.
///
/// Deliberately reuses [`super::coverage`]'s `"mt:schema/"`
/// schema-pack-compartment prefix (`is_schema_pack_compartment`), even
/// though a manifest is not a schema *pack* — the manifest is schema-system
/// bookkeeping about a note's *last projection*, exactly the same category
/// of "not data about the world" the coverage report already carves out as
/// [`super::coverage::TripleClass::SchemaInfra`]. Piggy-backing on the
/// existing prefix check means the coverage report classifies manifest
/// triples correctly for free, with no changes to `schema::coverage`.
pub const MANIFEST_COMPARTMENT: &str = "mt:schema/projection-manifest";

/// `(note-manifest-entity, projection:asserted, {triple-identity entity})` —
/// the manifest predicate. One triple per [`TripleIdentity`] the note's last
/// successful [`reproject_note`] (or first [`project_note`]) call asserted.
pub const PRED_PROJECTION_ASSERTED: &str = "projection:asserted";

/// Label prefix for the per-identity marker entity a manifest triple points
/// at. `{prefix}{identity.to_hex()}` — one KG entity per unique
/// [`TripleIdentity`] ever seen by any manifest, shared across notes (the
/// hex is content-addressed, so this is naturally deduplicated).
const IDENTITY_LABEL_PREFIX: &str = "triple-identity:";

fn manifest_entity_label(note_id: &NoteId) -> String {
    format!("note-manifest:{}", note_id.as_str())
}

fn identity_entity_label(identity: &TripleIdentity) -> String {
    format!("{IDENTITY_LABEL_PREFIX}{}", identity.to_hex())
}

/// Resolve (creating if absent) the manifest entity for `note_id`, and read
/// back the set of [`TripleIdentity`] hashes its current manifest triples
/// record. Empty for a note that has never been projected.
fn read_manifest(engine: &Engine, note_id: &NoteId) -> SchemaResult<(SymbolId, HashSet<TripleIdentity>)> {
    let manifest_entity = engine
        .resolve_or_create_entity(&manifest_entity_label(note_id))
        .map_err(|e| projection_err(note_id, format!("manifest entity: {e}")))?;
    let asserted_pred = engine
        .resolve_or_create_relation(PRED_PROJECTION_ASSERTED)
        .map_err(|e| projection_err(note_id, format!("manifest predicate: {e}")))?;

    let identities = engine
        .triples_from(manifest_entity)
        .into_iter()
        .filter(|t| t.predicate == asserted_pred)
        .filter_map(|t| {
            engine
                .resolve_label(t.object)
                .strip_prefix(IDENTITY_LABEL_PREFIX)
                .and_then(|hex| TripleIdentity::from_hex(hex).ok())
        })
        .collect();

    Ok((manifest_entity, identities))
}

/// Overwrite the manifest to match `new_identities`: assert a marker triple
/// for every newly-added identity, retract the marker for every identity
/// that dropped out. `prior` is the set [`read_manifest`] returned earlier in
/// the same call — passed in rather than re-read so the diff is computed
/// against a single consistent snapshot.
fn write_manifest(
    engine: &Engine,
    note_id: &NoteId,
    manifest_entity: SymbolId,
    prior: &HashSet<TripleIdentity>,
    new_identities: &HashSet<TripleIdentity>,
) -> SchemaResult<()> {
    let asserted_pred = engine
        .resolve_or_create_relation(PRED_PROJECTION_ASSERTED)
        .map_err(|e| projection_err(note_id, format!("manifest predicate: {e}")))?;

    for identity in new_identities.difference(prior) {
        let marker = engine
            .resolve_or_create_entity(&identity_entity_label(identity))
            .map_err(|e| projection_err(note_id, format!("manifest marker: {e}")))?;
        engine
            .add_triple(
                &Triple::new(manifest_entity, asserted_pred, marker)
                    .with_compartment(MANIFEST_COMPARTMENT.to_string()),
            )
            .map_err(|e| projection_err(note_id, format!("manifest assert: {e}")))?;
    }

    for identity in prior.difference(new_identities) {
        let marker = engine
            .resolve_or_create_entity(&identity_entity_label(identity))
            .map_err(|e| projection_err(note_id, format!("manifest marker: {e}")))?;
        let _ = engine.remove_triple(manifest_entity, asserted_pred, marker);
    }

    Ok(())
}

/// Re-project a [`Note`] into the typed KG, diffing the new triple set
/// against the note's prior **projection manifest** and retracting exactly
/// what changed (ADR 081 decision 12, work-item 8).
///
/// ## Why a manifest, and why KG-native triples
///
/// Diffing needs an answer to "what did the *last* projection of this note
/// assert?" — [`project_note`] itself is write-only (it only ever calls
/// [`Engine::add_triple`]), so without a separate record there is no way to
/// tell "this triple is still correct, leave it" from "this triple's
/// frontmatter key was deleted, retract it": both look identical from the
/// KG's point of view (the triple is simply present).
///
/// The manifest is `(note-manifest:{id}, projection:asserted,
/// {triple-identity:{hex}})` triples — plain KG data, not an
/// engine-side cache:
/// - **Survives snapshot/wake for free.** Phase 38's snapshot/restore
///   machinery (`Engine::all_triples` / `Engine::restore_triples`) already
///   captures and reloads every KG triple; the manifest needs no bespoke
///   (de)serialization or a new Postgres table to persist across a
///   relocate/wake cycle.
/// - **Auditable.** The manifest is queryable with the same tools as any
///   other fact (`engine.triples_from(manifest_entity)`), satisfying the
///   acceptance criterion "the prior note version + the change are
///   auditable" without a separate history table.
/// - **A separate subject, not `note_entity` itself.** Manifest triples hang
///   off a dedicated `note-manifest:{id}` entity rather than `note_entity`
///   so [`typed_view`]'s classification (which walks
///   `engine.triples_from(note_entity)` and buckets anything it does not
///   recognize as a typed property) is never touched by bookkeeping data.
///
/// ## The diff
///
/// 1. Read the prior manifest (before this call changes anything).
/// 2. Run [`project_note`] unchanged — it asserts the note's *current*
///    intended triple set (idempotent dedup handles unchanged triples; see
///    [`ProjectionReport::asserted_triples`] for the exact list this pass
///    intended, independent of whatever else happens to already be in the
///    KG).
/// 3. Diff `asserted_triples`' [`TripleIdentity`] set against the prior
///    manifest: identities in both are **retained** (untouched — no
///    provenance/confidence churn, since `add_triple`'s dedup path only
///    raises confidence to a max and never overwrites an existing
///    `provenance_id` with `None`); identities only in the new set are
///    **added** (already asserted by step 2); identities only in the prior
///    manifest are **retracted**.
/// 4. Retract each dropped identity via [`Engine::remove_triple`] — the
///    existing Phase 9c TMS cascade marks the triple's object
///    `was_retracted` (Phase 44g recall hygiene already filters on that),
///    while a [`DerivationKind::NoteReprojection`] provenance record is
///    stored *before* the retraction so the "who/what superseded this" audit
///    trail survives even though the TMS's own support-set bookkeeping does
///    not retain retracted entries.
/// 5. Overwrite the manifest to the new identity set.
pub fn reproject_note(engine: &Engine, note: &Note) -> SchemaResult<ProjectionReport> {
    let (manifest_entity, prior_identities) = read_manifest(engine, &note.id)?;

    let mut report = project_note(engine, note)?;

    let new_identities: HashSet<TripleIdentity> = report
        .asserted_triples
        .iter()
        .map(|t| engine.triple_identity(t))
        .collect();

    let added = new_identities.difference(&prior_identities).count();
    let retained = new_identities.intersection(&prior_identities).count();

    let mut retracted = 0usize;
    for identity in prior_identities.difference(&new_identities) {
        let Some(stale) = engine.triple_by_identity(identity) else {
            // Already gone (removed by something else, or never resolvable
            // any more) — nothing to retract, nothing to record.
            continue;
        };

        // Supersede provenance recorded *before* removal (work-item 8, item
        // 4): a durable "who/what superseded this fact" audit trail that
        // outlives the TMS's own support-set bookkeeping.
        let mut prov = ProvenanceRecord::new(
            stale.object,
            DerivationKind::NoteReprojection {
                note_id: note.id.to_string(),
                reason: "note re-projection: frontmatter/body no longer asserts this fact".into(),
            },
        )
        .with_sources(vec![manifest_entity, stale.subject, stale.predicate])
        .with_confidence(0.0);
        let _ = engine.store_provenance(&mut prov);

        // TMS-cascaded removal: marks `was_retracted` for Phase 44g recall
        // hygiene, logs to the WAL, and syncs the SPARQL store — the same
        // path any other fact retraction takes.
        let _ = engine.remove_triple(stale.subject, stale.predicate, stale.object);
        retracted += 1;
    }

    write_manifest(engine, &note.id, manifest_entity, &prior_identities, &new_identities)?;

    report.diff = Some(ProjectionDiff {
        added,
        retained,
        retracted,
    });
    Ok(report)
}

fn projection_err(note_id: &NoteId, message: impl std::fmt::Display) -> SchemaError {
    SchemaError::Projection {
        note_id: note_id.as_str().to_string(),
        message: message.to_string(),
    }
}

/// Render a frontmatter value to the literal-entity label it projects onto.
/// Mirrors (but does not reuse — the note carrier's is private)
/// [`crate::library::note::Frontmatter::to_block`]'s per-value rendering, so
/// the same value always projects onto the same label call over call.
fn value_label(value: &FrontmatterValue) -> String {
    match value {
        FrontmatterValue::Text(s) => s.clone(),
        FrontmatterValue::Number(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                n.to_string()
            }
        }
        FrontmatterValue::Checkbox(b) => b.to_string(),
        FrontmatterValue::Date(d) => d.clone(),
        FrontmatterValue::Datetime(dt) => dt.clone(),
        FrontmatterValue::List(items) => items.join("; "),
        FrontmatterValue::Link(link) => link.target.clone(),
    }
}

/// Resolve `type_name` to its `SymbolId` iff it is declared as a type in the
/// installed schema (`(type_name, is-a, onto:Type)`). `None` covers both
/// "no such symbol at all" and "symbol exists but isn't marked as a type" —
/// both are equally "not an installed type" from a projection's point of view.
fn resolve_installed_type(engine: &Engine, type_name: &str) -> Option<SymbolId> {
    let type_symbol = engine.resolve_symbol(type_name).ok()?;
    shape::is_installed_type(engine, type_symbol).then_some(type_symbol)
}

// ---------------------------------------------------------------------------
// Round-trip read-back
// ---------------------------------------------------------------------------

/// A note's projected triples, read back into a typed view: the proof the
/// write → project → read spine round-trips.
#[derive(Debug, Clone)]
pub struct TypedNoteView {
    /// The `note:{id}` KG entity.
    pub note_entity: SymbolId,
    /// `None` when the note is untyped or its declared type was unknown at
    /// projection time.
    pub type_name: Option<String>,
    /// Property label → asserted value labels (schema-declared properties).
    pub typed_properties: BTreeMap<String, Vec<String>>,
    /// Frontmatter key → asserted value labels (unknown type and/or unknown
    /// key — ADR 081 decision 7, still written, just marked frontier).
    pub frontier_properties: BTreeMap<String, Vec<String>>,
    /// Wikilink targets.
    pub links: Vec<String>,
}

/// Read a previously-[`project_note`]d note back as a [`TypedNoteView`].
/// Errs only if the note was never projected (no `note:{id}` entity exists).
pub fn typed_view(engine: &Engine, note_id: &NoteId) -> SchemaResult<TypedNoteView> {
    let note_entity = note_of_entity(engine, note_id)?;

    let mut view = TypedNoteView {
        note_entity,
        type_name: None,
        typed_properties: BTreeMap::new(),
        frontier_properties: BTreeMap::new(),
        links: Vec::new(),
    };

    for triple in engine.triples_from(note_entity) {
        let predicate_label = engine.resolve_label(triple.predicate);
        let object_label = engine.resolve_label(triple.object);

        if predicate_label == PRED_IS_A {
            view.type_name = Some(object_label);
        } else if predicate_label == PRED_LINKS_TO {
            view.links.push(object_label);
        } else if let Some(key) = predicate_label.strip_prefix(FRONTIER_PREFIX) {
            view.frontier_properties
                .entry(key.to_string())
                .or_default()
                .push(object_label);
        } else {
            view.typed_properties
                .entry(predicate_label)
                .or_default()
                .push(object_label);
        }
    }

    Ok(view)
}

/// Resolve a [`NoteId`] to its projected `note:{id}` KG entity.
pub fn note_of_entity(engine: &Engine, note_id: &NoteId) -> SchemaResult<SymbolId> {
    let note_label = format!("note:{}", note_id.as_str());
    engine.resolve_symbol(&note_label).map_err(|e| SchemaError::Projection {
        note_id: note_id.as_str().to_string(),
        message: format!("note was never projected (no `{note_label}` entity found): {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::library::note::{Frontmatter, TaskRef};
    use crate::library::model::{ContentFormat, DocumentRecord, DocumentSource};
    use crate::schema::compile::compile;
    use crate::schema::install::install_pack;
    use crate::vsa::Dimension;

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

    /// A persistent (temp-dir-backed) engine — `store_provenance` errors
    /// with `NoPersistence` on a pure in-memory engine (mirrors
    /// `agent::harvest`'s test setup); this slice's acceptance criterion
    /// "the triple carries provenance back to the source note" needs a real
    /// ledger to check against.
    fn test_engine() -> (tempfile::TempDir, crate::engine::Engine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = crate::engine::Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        (dir, engine)
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

    fn install_seed(engine: &crate::engine::Engine) {
        let pack = compile(SEED_PACK).unwrap();
        install_pack(engine, &pack).unwrap();
    }

    fn person_note() -> Note {
        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Person".into()));
        fm.insert("email", FrontmatterValue::Text("jane@example.org".into()));
        fm.insert("birthDate", FrontmatterValue::Date("1990-01-01".into()));
        fm.insert("favoriteColor", FrontmatterValue::Text("teal".into()));
        Note::new(sample_document("doc-1"), "Colleague of [[Alice Example]].")
            .with_frontmatter(fm)
            .with_tasks(Vec::<TaskRef>::new())
    }

    #[test]
    fn typed_note_projects_is_a_and_declared_properties() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let note = person_note();

        let report = project_note(&engine, &note).unwrap();

        assert!(report.type_outcome.is_typed());
        assert!(report.type_outcome.diagnostic().is_none());
        assert_eq!(report.typed_property_count, 2, "email + birthDate");
        assert_eq!(report.frontier_property_count, 1, "favoriteColor only");
        assert_eq!(report.wikilink_count, 1);
        assert!(report.provenance_id.is_some());

        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(view.type_name.as_deref(), Some("Person"));
        assert_eq!(
            view.typed_properties.get("email"),
            Some(&vec!["jane@example.org".to_string()])
        );
        assert_eq!(
            view.typed_properties.get("birthDate"),
            Some(&vec!["1990-01-01".to_string()])
        );
        assert_eq!(
            view.frontier_properties.get("favoriteColor"),
            Some(&vec!["teal".to_string()])
        );
        assert_eq!(view.links, vec!["Alice Example".to_string()]);

        // Provenance carries the note back to its source (acceptance: "the
        // triple carries provenance back to the source note + version").
        let prov = engine.provenance_of(report.note_entity).unwrap();
        assert!(prov.iter().any(|p| matches!(
            &p.kind,
            DerivationKind::AgentConsolidation { reason, .. }
                if reason == &format!("note_projection:{}", note.id)
        )));
    }

    #[test]
    fn unknown_type_lands_entirely_in_frontier() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Wizard".into()));
        fm.insert("email", FrontmatterValue::Text("merlin@example.org".into()));
        let note = Note::new(sample_document("doc-2"), "no links here").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();

        assert!(!report.type_outcome.is_typed());
        assert!(matches!(&report.type_outcome, TypeOutcome::UnknownType { type_name } if type_name == "Wizard"));
        assert!(report.type_outcome.diagnostic().unwrap().contains("Wizard"));
        assert_eq!(report.typed_property_count, 0);
        // Both `type` and `email` land in frontier — nothing is dropped.
        assert_eq!(report.frontier_property_count, 2);

        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(view.type_name, None);
        assert!(view.typed_properties.is_empty());
        assert_eq!(
            view.frontier_properties.get("type"),
            Some(&vec!["Wizard".to_string()])
        );
        assert_eq!(
            view.frontier_properties.get("email"),
            Some(&vec!["merlin@example.org".to_string()])
        );
    }

    #[test]
    fn note_without_type_key_produces_no_type_triple_and_does_not_crash() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("email", FrontmatterValue::Text("no-type@example.org".into()));
        let note = Note::new(sample_document("doc-3"), "plain body").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();
        assert_eq!(report.type_outcome, TypeOutcome::Untyped);
        assert!(report.type_outcome.diagnostic().is_none());
        assert_eq!(report.frontier_property_count, 1);

        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(view.type_name, None);
    }

    #[test]
    fn frontmatter_less_note_projects_only_wikilinks() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let note = Note::new(sample_document("doc-4"), "See [[Some Target]].");
        let report = project_note(&engine, &note).unwrap();

        assert_eq!(report.type_outcome, TypeOutcome::Untyped);
        assert_eq!(report.typed_property_count, 0);
        assert_eq!(report.frontier_property_count, 0);
        assert_eq!(report.wikilink_count, 1);
    }

    #[test]
    fn reprojecting_unchanged_note_does_not_duplicate_triples() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let note = person_note();

        let first = project_note(&engine, &note).unwrap();
        let triples_after_first = engine.triples_from(first.note_entity).len();

        let second = project_note(&engine, &note).unwrap();
        let triples_after_second = engine.triples_from(second.note_entity).len();

        assert_eq!(
            triples_after_first, triples_after_second,
            "re-projecting an unchanged note must not duplicate (subject, predicate, object) triples"
        );
        // Both calls still succeed and still record provenance (provenance
        // itself is not triple-deduped — each projection pass is logged;
        // see the ProjectionReport docs).
        assert!(first.provenance_id.is_some());
        assert!(second.provenance_id.is_some());
        assert_ne!(first.provenance_id, second.provenance_id);
    }

    #[test]
    fn typed_view_errs_for_a_note_never_projected() {
        let (_dir, engine) = test_engine();
        let never_projected = NoteId::new();
        assert!(matches!(
            typed_view(&engine, &never_projected),
            Err(SchemaError::Projection { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Slice 3: is-a subtyping + property-shape validation, exercised at
    // the note-projection level (unit coverage of the chain walk itself
    // lives in `schema::shape`; this proves the acceptance criteria in
    // `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 3
    // end-to-end through `project_note`).
    // -----------------------------------------------------------------

    /// A second, independently-installed pack that extends `SEED_PACK`'s
    /// `Person` via `is_a` — proves the chain walk crosses pack boundaries
    /// (ADR 081 decision 5: type names are global).
    const EMPLOYEE_PACK: &str = r#"{
        "pack": {"name": "hr-test", "version": "0.1.0"},
        "assumes": ["Person"],
        "types": [
            {"name": "Employee", "is_a": "Person", "properties": [
                {"name": "employer", "value_type": "text", "required": true}]}
        ]
    }"#;

    fn install_employee(engine: &crate::engine::Engine) {
        let pack = compile(EMPLOYEE_PACK).unwrap();
        install_pack(engine, &pack).unwrap();
    }

    #[test]
    fn inherited_property_projects_as_typed_for_a_subtype_note() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        install_employee(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Employee".into()));
        fm.insert("email", FrontmatterValue::Text("jane@example.org".into())); // Person's, inherited
        fm.insert("employer", FrontmatterValue::Text("Acme".into())); // Employee's own
        let note = Note::new(sample_document("doc-employee-1"), "no links").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();

        assert!(report.type_outcome.is_typed());
        assert_eq!(report.typed_property_count, 2, "email (inherited) + employer (own)");
        assert_eq!(report.frontier_property_count, 0);
        assert!(report.shape_violations.is_empty());

        let email = report
            .typed_properties
            .iter()
            .find(|p| p.key == "email")
            .expect("email must project as typed, not frontier — it is inherited from Person");
        assert!(!email.direct, "email is inherited, not declared directly on Employee");
        assert_eq!(email.declared_on, "Person");

        let employer = report
            .typed_properties
            .iter()
            .find(|p| p.key == "employer")
            .unwrap();
        assert!(employer.direct);
        assert_eq!(employer.declared_on, "Employee");

        // Read-back confirms the inherited property landed as typed, not
        // frontier, in the KG itself.
        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(view.type_name.as_deref(), Some("Employee"));
        assert_eq!(
            view.typed_properties.get("email"),
            Some(&vec!["jane@example.org".to_string()])
        );
        assert!(view.frontier_properties.is_empty());
    }

    #[test]
    fn missing_required_property_surfaces_diagnostic_but_does_not_block_the_write() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        install_employee(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Employee".into()));
        // `employer` is required by Employee's shape but omitted here.
        let note = Note::new(sample_document("doc-employee-2"), "").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();

        // The write is never blocked (ADR 081 decision 7): the note still
        // projects onto its declared type.
        assert!(report.type_outcome.is_typed());

        assert_eq!(report.shape_violations.len(), 1);
        let violation = &report.shape_violations[0];
        assert!(matches!(
            violation,
            ShapeViolation::MissingRequired { property, declared_on }
                if property == "employer" && declared_on == "Employee"
        ));

        // The violation carries a rich miette diagnostic (help text), per
        // CLAUDE.md's "use miette's diagnostic pattern" convention.
        use miette::Diagnostic as _;
        let help = violation.help().expect("missing-required carries help text").to_string();
        assert!(help.contains("employer"));
    }

    #[test]
    fn completing_the_required_property_validates_cleanly() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        install_employee(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Employee".into()));
        fm.insert("employer", FrontmatterValue::Text("Acme".into()));
        let note = Note::new(sample_document("doc-employee-3"), "").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();

        assert!(report.type_outcome.is_typed());
        assert!(report.shape_violations.is_empty());
        assert_eq!(report.typed_property_count, 1);
        assert!(report.typed_properties.iter().any(|p| p.key == "employer"));
    }

    #[test]
    fn value_type_mismatch_demotes_to_frontier_and_is_reported() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);

        let mut fm = Frontmatter::new();
        fm.insert("type", FrontmatterValue::Text("Person".into()));
        // `email` is declared `text`; a Number is a shape violation.
        fm.insert("email", FrontmatterValue::Number(42.0));
        let note = Note::new(sample_document("doc-mismatch-1"), "").with_frontmatter(fm);

        let report = project_note(&engine, &note).unwrap();

        assert!(report.type_outcome.is_typed());
        assert_eq!(report.typed_property_count, 0);
        assert_eq!(report.frontier_property_count, 1, "the mismatched value lands in frontier");
        assert!(!report.typed_properties.iter().any(|p| p.key == "email"));

        assert_eq!(report.shape_violations.len(), 1);
        assert!(matches!(
            &report.shape_violations[0],
            ShapeViolation::TypeMismatch { property, expected, got, .. }
                if property == "email"
                    && *expected == crate::schema::source::ValueType::Text
                    && *got == crate::schema::source::ValueType::Number
        ));

        // The value was still written — just under the frontier predicate.
        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(view.frontier_properties.get("email"), Some(&vec!["42".to_string()]));
        assert!(view.typed_properties.get("email").is_none());
    }

    #[test]
    fn additive_extension_leaves_the_ancestor_types_projection_unaffected() {
        // ADR 081 decision 5: adding Employee must never change how a
        // Person note (Employee's ancestor) itself projects.
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        install_employee(&engine);

        let note = person_note(); // typed "Person", not "Employee"
        let report = project_note(&engine, &note).unwrap();

        assert!(report.type_outcome.is_typed());
        assert_eq!(report.typed_property_count, 2, "email + birthDate, exactly as before Employee existed");
        assert!(!report.typed_properties.iter().any(|p| p.key == "employer"));
    }

    // -----------------------------------------------------------------
    // Slice 8: diff-based supersede (`reproject_note`) —
    // `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md` work-item 8.
    // -----------------------------------------------------------------

    #[test]
    fn first_reprojection_reports_everything_as_added() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let note = person_note();

        let report = reproject_note(&engine, &note).unwrap();

        let diff = report.diff.expect("reproject_note always sets diff");
        assert_eq!(diff.retracted, 0);
        assert_eq!(diff.retained, 0);
        assert_eq!(diff.added, report.asserted_triples.len());
        assert!(diff.added > 0);
    }

    #[test]
    fn reprojecting_an_unchanged_note_retains_everything_and_retracts_nothing() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let note = person_note();

        let first = reproject_note(&engine, &note).unwrap();
        let first_diff = first.diff.unwrap();

        let second = reproject_note(&engine, &note).unwrap();
        let second_diff = second.diff.unwrap();

        assert_eq!(second_diff.added, 0);
        assert_eq!(second_diff.retracted, 0);
        assert_eq!(second_diff.retained, first_diff.added, "everything from pass 1 is retained in pass 2");
    }

    #[test]
    fn editing_a_typed_property_retracts_only_the_changed_triple() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let mut note = person_note();

        let first = reproject_note(&engine, &note).unwrap();
        assert!(first.type_outcome.is_typed());

        // The birthDate triple, present before the edit — used below to
        // prove it is untouched (acceptance: "unchanged triples keep their
        // original provenance/confidence").
        let birth_date_triple = first
            .asserted_triples
            .iter()
            .find(|t| engine.resolve_label(t.predicate) == "birthDate")
            .cloned()
            .expect("birthDate triple was asserted");
        let old_email_triple = first
            .asserted_triples
            .iter()
            .find(|t| engine.resolve_label(t.predicate) == "email")
            .cloned()
            .expect("email triple was asserted");

        // Edit just the email value — same NoteId (set_frontmatter preserves
        // identity, mirroring an in-place note edit), one changed field.
        let mut fm = note.frontmatter.clone();
        fm.insert("email", FrontmatterValue::Text("jane.doe@example.org".into()));
        note.set_frontmatter(fm);

        let second = reproject_note(&engine, &note).unwrap();
        let diff = second.diff.unwrap();

        assert_eq!(diff.added, 1, "only the new email value is newly added");
        assert_eq!(diff.retracted, 1, "only the old email value is retracted");
        // birthDate + is-a + favoriteColor + wikilink all retained unchanged.
        assert_eq!(diff.retained, first.asserted_triples.len() - 1);

        // The old email triple is gone from the live KG and marked
        // superseded — discoverable via `was_retracted`, the exact primitive
        // Phase 44g recall hygiene already filters recall on.
        assert!(!engine.has_triple(old_email_triple.subject, old_email_triple.predicate, old_email_triple.object));
        assert!(
            engine.tms().was_retracted(old_email_triple.object),
            "retracted triple's object must be discoverable via TruthMaintenanceSystem::was_retracted"
        );

        // The new email value is live.
        let view = typed_view(&engine, &note.id).unwrap();
        assert_eq!(
            view.typed_properties.get("email"),
            Some(&vec!["jane.doe@example.org".to_string()])
        );

        // birthDate — untouched — keeps its exact confidence (no churn).
        assert!(engine.has_triple(birth_date_triple.subject, birth_date_triple.predicate, birth_date_triple.object));
        let live_birth_date = engine
            .triples_from(birth_date_triple.subject)
            .into_iter()
            .find(|t| t.predicate == birth_date_triple.predicate && t.object == birth_date_triple.object)
            .expect("birthDate triple still live");
        assert!((live_birth_date.confidence - birth_date_triple.confidence).abs() < 1e-6);

        // A NoteReprojection provenance record documents who/what
        // superseded the old email fact — the audit trail work-item 8 asks
        // for, surviving independently of the TMS's own bookkeeping.
        let prov = engine.provenance_of(old_email_triple.object).unwrap();
        assert!(prov.iter().any(|p| matches!(
            &p.kind,
            DerivationKind::NoteReprojection { note_id, .. } if note_id == note.id.as_str()
        )));
    }

    #[test]
    fn removing_a_frontier_key_retracts_its_triple() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let mut note = person_note(); // carries `favoriteColor` (frontier)

        let first = reproject_note(&engine, &note).unwrap();
        let frontier_triple = first
            .asserted_triples
            .iter()
            .find(|t| engine.resolve_label(t.predicate) == format!("{FRONTIER_PREFIX}favoriteColor"))
            .cloned()
            .expect("favoriteColor frontier triple was asserted");

        // Remove the key entirely (`Frontmatter` has no `remove` — rebuild
        // without it).
        let mut fm = Frontmatter::new();
        for (key, value) in note.frontmatter.iter() {
            if key != "favoriteColor" {
                fm.insert(key, value.clone());
            }
        }
        note.set_frontmatter(fm);

        let second = reproject_note(&engine, &note).unwrap();
        let diff = second.diff.unwrap();

        assert_eq!(diff.added, 0);
        assert_eq!(diff.retracted, 1);
        assert!(!engine.has_triple(frontier_triple.subject, frontier_triple.predicate, frontier_triple.object));
        assert!(engine.tms().was_retracted(frontier_triple.object));

        let view = typed_view(&engine, &note.id).unwrap();
        assert!(view.frontier_properties.get("favoriteColor").is_none());
    }

    #[test]
    fn manifest_is_ordinary_kg_data_and_survives_a_bulk_restore_round_trip() {
        // Proof that the manifest needs no bespoke (de)serialization: it is
        // plain KG triples, so the same bulk-restore path Phase 38c's wake
        // uses (`Engine::restore_triples`, exercised the same way
        // `engine::tests::identity_index_rebuilds_after_bulk_restore` proves
        // the identity index survives it) round-trips it correctly.
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        let note = person_note();

        let first = reproject_note(&engine, &note).unwrap();
        let asserted_before = first.asserted_triples.len();

        // Simulate the wake bulk-load path: re-load every live triple
        // (including the manifest's own bookkeeping triples) via the same
        // bypass-add_triple path Phase 38c's wake uses.
        let all = engine.all_triples();
        engine.restore_triples(&all).unwrap();

        // Re-projecting again must still see the manifest correctly: an
        // unchanged note retains everything and adds/retracts nothing, which
        // is only possible if the manifest triples survived the round trip.
        let second = reproject_note(&engine, &note).unwrap();
        let diff = second.diff.unwrap();
        assert_eq!(diff.added, 0, "manifest content survived the bulk-restore round trip");
        assert_eq!(diff.retracted, 0);
        assert_eq!(diff.retained, asserted_before);
    }
}
