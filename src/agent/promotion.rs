//! Frontier → core promotion + human-confirm + aliasing — ADR 081 Slice 5
//! (`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`, work-item 5).
//!
//! Closes the governance loop the coverage report (Slice 9,
//! `schema::coverage`) and the Librarian projection (Slice 4,
//! `agent::librarian`) set up:
//!
//! - [`promotion_candidates`] finds frontier predicates (mechanically, no
//!   LLM) that have recurred across enough distinct subjects with a
//!   consistent value shape — the "does this deserve a proposal at all?"
//!   gate.
//! - [`run_promotion_pass`] asks the [`LibrarianModel`] seam
//!   ([`PromotionRequest`]/[`PromotionAdvice`], defined alongside
//!   `agent::librarian`'s extraction seam since it is the same LLM
//!   dependency) to name/type/alias each new candidate, and stores the
//!   result as a **pending** [`PromotionProposal`] — a `schema:proposal/{predicate}`
//!   KG entity. Nothing here ever writes to the shared registry: storing a
//!   proposal is structurally incapable of admitting anything (the function
//!   doesn't even take a [`SchemaRegistry`] parameter).
//! - [`promote_proposal`] is the human-confirm step (ADR 081 decision 8):
//!   for a non-alias proposal it mints a tiny schema pack extending the
//!   proposed type via `is-a` subtyping (or a standalone type when none is
//!   proposed), admits it through the caller-supplied [`SchemaRegistry`],
//!   and installs + activates it; for an `alias_of` proposal it records an
//!   `onto:alias-of` edge instead — no new pack, no new property (R5: the
//!   core stays small).
//! - [`reject_proposal`] marks a proposal rejected so it never re-surfaces
//!   ([`run_promotion_pass`] skips any predicate that already has a
//!   proposal, confirmed or rejected).
//!
//! ## Deferred: the cross-akh (`≥M akhs`) dimension
//!
//! The plan's acceptance signal is "recurring across ≥N notes / ≥M akhs".
//! [`promotion_candidates`] implements the single-akh `≥N distinct subjects`
//! half only. An aggregate view across akhs would need a cross-akh query
//! over frontier usage — the registry stores *admitted* packs, not frontier
//! usage stats, and no such telemetry channel exists yet. Faking a "≥M
//! akhs" count from single-akh data would misrepresent consensus, so this
//! dimension is explicitly deferred rather than approximated. A follow-up
//! slice can add it once a cross-akh frontier-stats channel exists (e.g. a
//! `akhomed_frontier_stats` table each akh's coverage report populates).
//!
//! ## Deferred: frontier *type* promotion (`is-a`)
//!
//! [`promotion_candidates`] skips the `is-a` frontier key entirely (frontier
//! `is-a` triples to an uninstalled type all tally under the single key
//! `"is-a"` in `schema::coverage`/`agent::librarian::frontier_terms` — there
//! is no per-target-type breakdown to threshold against without a further
//! query). Promoting a recurring frontier *type* (as opposed to a property
//! on an existing type) is a distinct, not-yet-built shape; only property
//! promotion is implemented here.
//!
//! ## Why a new subtype instead of adding to the existing type directly
//!
//! `schema::install::install_pack`'s global-immutability guard (ADR 081
//! decision 5) rejects re-declaring `(T, is-a, onto:Type)` for a `T` some
//! *other* compartment already declared — so a promotion pack can never
//! redeclare an existing type just to attach a new property to it (even a
//! new *version* of the same pack would count as a different compartment
//! and get rejected the same way). Instead a promotion mints a **new,
//! uniquely-named subtype** ([`promoted_subtype_name`]) that `is-a`s the
//! proposed type and carries just the promoted property — exactly the
//! sanctioned additive-extension path `schema::source`'s `assumes` + `is_a`
//! mechanism already supports (the same shape `schema::install`'s own test
//! `redefining_a_global_type_is_rejected` uses for its "assumes"-based `hr`
//! extension). `schema::shape::effective_properties` already walks the
//! `is-a` chain, so anything typed as the new subtype sees both the
//! inherited base-type properties and the promoted one. When the advisor
//! proposes no existing type (or proposes one that isn't actually
//! installed — an LLM can hallucinate a name), the mini-pack declares a
//! standalone root type instead ([`promoted_standalone_type_name`]); see
//! [`PromotionOutcome::Promoted::supertype`] for how a caller observes which
//! path was taken.
//!
//! ## Why one pack per proposal, not one growing pack
//!
//! Each promotion mints a pack named `akh-frontier-promotion-{predicate}` at
//! a fixed version (`1.0.0`) — a pack scoped to exactly the one type/
//! property it introduces, rather than a single `akh-frontier-promotions`
//! pack bumped on every confirm. One pack per proposal keeps each artifact
//! self-contained and independently auditable ("which pack backs proposal
//! X?" — its name says so), sidesteps tracking a shared "next version"
//! counter across concurrent promotions from different akhs (a real risk:
//! any akh with registry write access can confirm a proposal), and costs
//! nothing extra — the registry already indexes packs by `(name, version)`,
//! not by a single growing lineage.
//!
//! ## Aliasing: recorded, not re-asserted
//!
//! An `alias_of` promotion records exactly one triple —
//! `(frontier_predicate, onto:alias-of, core_term)` — and nothing else.
//! Historical frontier triples using the aliased predicate stay classified
//! `Frontier` by `schema::coverage::classify_triple`: making them
//! reclassify as `TypedCore` would need `SchemaView` to resolve a predicate
//! through the alias map before checking `declared_property_names` (and
//! still requires the *subject* to be independently typed) — a real but
//! non-trivial change to code that Slice 9 already ships fully tested. Per
//! this slice's brief ("extend classify... IF that's a small change;
//! otherwise defer"), that extension is deferred. New writes are
//! unaffected: from the moment `promote_proposal` returns, `schema::digest`
//! and `agent::librarian::schema_digest` both see the new pack, so the
//! Librarian's *next* extraction pass files the aliased predicate as typed
//! going forward without any backfill — the same lazy-crystallization story
//! `schema::coverage`'s module docs describe for ordinary promotion.
//!
//! ## Never-blocked invariant
//!
//! Nothing in this module touches the frontier write path
//! (`agent::librarian::assert_candidate`, `schema::project::project_note`).
//! A predicate below threshold, with no proposal, or with a rejected
//! proposal keeps landing in frontier exactly as before — promotion is
//! strictly additive read+propose+confirm machinery layered on top.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::library::note::parse_value;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::schema::project::FRONTIER_PREFIX;
use crate::schema::source::ValueType;
use crate::schema::{RegistryRow, SchemaRegistry, activate_pack, compile, install_pack};
use crate::symbol::SymbolId;

use super::error::{AgentError, AgentResult};
use super::librarian::{LibrarianModel, PromotionAdvice, PromotionRequest, SchemaTypeDigest};

// ---------------------------------------------------------------------------
// Thresholds + candidate detection (mechanical, no LLM)
// ---------------------------------------------------------------------------

/// Tuning for [`promotion_candidates`].
#[derive(Debug, Clone)]
pub struct PromotionThresholds {
    /// Minimum distinct subjects a frontier predicate must have accrued to
    /// be worth proposing (the "≥N notes" half of the plan's "≥N notes /
    /// ≥M akhs" signal — see the module docs' deferred cross-akh section).
    pub min_distinct_subjects: usize,
}

impl Default for PromotionThresholds {
    fn default() -> Self {
        Self {
            min_distinct_subjects: 3,
        }
    }
}

/// Cap on how many example subjects/values [`promotion_candidates`] carries
/// per candidate (prompt-sizing + stored-proposal-sizing guard).
const MAX_EXAMPLES: usize = 5;

/// A frontier predicate that has crossed [`PromotionThresholds`] and is
/// eligible for a promotion proposal.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PromotionCandidate {
    /// The frontier predicate, unprefixed (no `note:frontier/`).
    pub predicate: String,
    pub distinct_subjects: usize,
    /// The single [`ValueType`] every observed object shares — a candidate
    /// with inconsistent shape across its observations is excluded before
    /// this struct is ever built (see [`promotion_candidates`]).
    pub value_type: ValueType,
    pub example_subjects: Vec<String>,
    pub example_values: Vec<String>,
}

/// Find frontier predicates recurring across at least
/// `thresholds.min_distinct_subjects` distinct subjects with a **consistent**
/// value shape (every observed object infers to the same [`ValueType`] via
/// [`crate::library::note::parse_value`] — the same textual heuristic that
/// already types frontmatter). A predicate with mixed shapes (e.g. some
/// numeric, some free text) is excluded outright rather than guessing —
/// inconsistent usage is exactly the signal that a term isn't ready for the
/// core yet. Pure and read-only; never mutates the KG. Deterministically
/// ordered (most-recurring first, ties broken lexicographically).
pub fn promotion_candidates(
    engine: &Engine,
    thresholds: &PromotionThresholds,
) -> Vec<PromotionCandidate> {
    let mut candidates = Vec::new();

    for symbol in engine.item_memory().all_symbols() {
        let label = engine.resolve_label(symbol);
        let Some(predicate) = label.strip_prefix(FRONTIER_PREFIX) else {
            continue;
        };
        if predicate == crate::schema::compile::PRED_IS_A {
            continue; // deferred — see module docs "frontier type promotion"
        }

        let pairs = engine.knowledge_graph().triples_for_predicate(symbol);
        if pairs.is_empty() {
            continue;
        }

        let mut distinct_subjects: BTreeSet<SymbolId> = BTreeSet::new();
        let mut shape: Option<ValueType> = None;
        let mut consistent = true;
        let mut example_subjects = Vec::new();
        let mut example_values = Vec::new();

        for (subject, object) in &pairs {
            distinct_subjects.insert(*subject);
            let object_label = engine.resolve_label(*object);
            let observed = parse_value(&object_label).value_type();
            match shape {
                None => shape = Some(observed),
                Some(existing) if existing == observed => {}
                Some(_) => consistent = false,
            }
            if example_subjects.len() < MAX_EXAMPLES {
                example_subjects.push(engine.resolve_label(*subject));
            }
            if example_values.len() < MAX_EXAMPLES {
                example_values.push(object_label);
            }
        }

        if !consistent {
            continue;
        }
        let Some(value_type) = shape else { continue };
        if distinct_subjects.len() < thresholds.min_distinct_subjects {
            continue;
        }

        candidates.push(PromotionCandidate {
            predicate: predicate.to_string(),
            distinct_subjects: distinct_subjects.len(),
            value_type,
            example_subjects,
            example_values,
        });
    }

    candidates.sort_by(|a, b| {
        b.distinct_subjects
            .cmp(&a.distinct_subjects)
            .then_with(|| a.predicate.cmp(&b.predicate))
    });
    candidates
}

// ---------------------------------------------------------------------------
// Proposal storage (KG entities — durable, queryable, never auto-applied)
// ---------------------------------------------------------------------------

/// Subject-label prefix for a proposal entity: `schema:proposal/{predicate}`.
/// One proposal per predicate — [`run_promotion_pass`] skips a predicate
/// that already has one, regardless of status.
const PROPOSAL_ID_PREFIX: &str = "schema:proposal/";

/// Bookkeeping predicates a proposal's triples use, all under the `onto:`
/// namespace so they classify as [`crate::schema::coverage::TripleClass::SchemaInfra`]
/// (schema apparatus, not world data) for free — `schema::coverage`'s
/// `SCHEMA_INFRA_PREFIXES` already includes `onto:`.
const PRED_PROPOSAL_MARKER: &str = "onto:proposal-marker";
const PROPOSAL_MARKER_OBJ: &str = "onto:Proposal";
const PRED_PROPOSAL_PREDICATE: &str = "onto:proposal-predicate";
const PRED_PROPOSAL_STATUS: &str = "onto:proposal-status";
const STATUS_PENDING_OBJ: &str = "onto:proposal-status/pending";
const STATUS_CONFIRMED_OBJ: &str = "onto:proposal-status/confirmed";
const STATUS_REJECTED_OBJ: &str = "onto:proposal-status/rejected";
const PRED_PROPOSAL_DISTINCT_SUBJECTS: &str = "onto:proposal-distinct-subjects";
const PRED_PROPOSAL_VALUE_TYPE: &str = "onto:proposal-value-type";
const PRED_PROPOSAL_TYPE: &str = "onto:proposal-type";
const PRED_PROPOSAL_PROPERTY_NAME: &str = "onto:proposal-property-name";
const PRED_PROPOSAL_ALIAS_OF: &str = "onto:proposal-alias-of";
const PRED_PROPOSAL_RATIONALE: &str = "onto:proposal-rationale";
const PRED_PROPOSAL_MODEL: &str = "onto:proposal-model";
const PRED_PROPOSAL_REJECT_REASON: &str = "onto:proposal-reject-reason";

/// The recorded alias edge: `(frontier_predicate, onto:alias-of, core_term)`.
/// Also `onto:`-prefixed, so it too reads as schema apparatus in coverage.
const PRED_ALIAS_OF: &str = "onto:alias-of";

fn proposal_label(predicate: &str) -> String {
    format!("{PROPOSAL_ID_PREFIX}{predicate}")
}

/// A promotion proposal's lifecycle state. Append-only in storage (see
/// [`proposal_status`]): a status transition *adds* a new status triple
/// rather than retracting the old one, so no diff/supersede machinery is
/// needed here — [`proposal_status`] just checks for a terminal marker
/// (confirmed/rejected) before falling back to pending.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Confirmed,
    Rejected,
}

impl std::fmt::Display for ProposalStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProposalStatus::Pending => write!(f, "pending"),
            ProposalStatus::Confirmed => write!(f, "confirmed"),
            ProposalStatus::Rejected => write!(f, "rejected"),
        }
    }
}

/// A stored promotion proposal — the durable, queryable record
/// [`run_promotion_pass`] writes and [`promote_proposal`]/[`reject_proposal`]
/// act on. `id` and `predicate` are the same string today (the frontier
/// predicate name doubles as the proposal id — see the module docs); kept as
/// two fields so callers reading this as JSON have a stable `id` key even if
/// that identity choice changes later.
#[derive(Debug, Clone, Serialize)]
pub struct PromotionProposal {
    pub id: String,
    pub predicate: String,
    pub status: ProposalStatus,
    pub distinct_subjects: usize,
    pub value_type: ValueType,
    pub proposed_type: Option<String>,
    pub proposed_property_name: String,
    pub alias_of: Option<String>,
    pub rationale: String,
    /// The [`LibrarianModel::label`] that produced this proposal.
    pub model: String,
}

/// Store a new pending proposal for `candidate`, per `advice`. Idempotent
/// only in the sense that re-calling this for an already-proposed predicate
/// adds a *second* set of proposal triples under the same subject (harmless
/// but redundant) — callers (`run_promotion_pass`) are responsible for the
/// "don't re-propose" skip via [`has_proposal`].
fn store_proposal(
    engine: &Engine,
    candidate: &PromotionCandidate,
    advice: &PromotionAdvice,
    model_label: &str,
) -> AgentResult<SymbolId> {
    let proposal = engine.resolve_or_create_entity(&proposal_label(&candidate.predicate))?;

    let assert_str = |pred_label: &str, object_label: &str| -> AgentResult<()> {
        let pred = engine.resolve_or_create_relation(pred_label)?;
        let object = engine.resolve_or_create_entity(object_label)?;
        engine.add_triple(&Triple::new(proposal, pred, object))?;
        Ok(())
    };

    assert_str(PRED_PROPOSAL_MARKER, PROPOSAL_MARKER_OBJ)?;
    assert_str(PRED_PROPOSAL_PREDICATE, &candidate.predicate)?;
    assert_str(PRED_PROPOSAL_STATUS, STATUS_PENDING_OBJ)?;
    assert_str(
        PRED_PROPOSAL_DISTINCT_SUBJECTS,
        &candidate.distinct_subjects.to_string(),
    )?;
    assert_str(PRED_PROPOSAL_VALUE_TYPE, candidate.value_type.entity_label())?;
    if let Some(t) = &advice.proposed_type {
        assert_str(PRED_PROPOSAL_TYPE, t)?;
    }
    assert_str(PRED_PROPOSAL_PROPERTY_NAME, &advice.proposed_property_name)?;
    if let Some(a) = &advice.alias_of {
        assert_str(PRED_PROPOSAL_ALIAS_OF, a)?;
    }
    if !advice.rationale.trim().is_empty() {
        assert_str(PRED_PROPOSAL_RATIONALE, &advice.rationale)?;
    }
    assert_str(PRED_PROPOSAL_MODEL, model_label)?;

    Ok(proposal)
}

/// Whether a proposal already exists for `predicate` (any status).
fn has_proposal(engine: &Engine, predicate: &str) -> bool {
    engine.resolve_symbol(&proposal_label(predicate)).is_ok()
}

fn single_object_label(
    engine: &Engine,
    subject: SymbolId,
    pred_label: &str,
) -> AgentResult<Option<String>> {
    let Ok(pred) = engine.resolve_symbol(pred_label) else {
        return Ok(None);
    };
    Ok(engine
        .knowledge_graph()
        .objects_of(subject, pred)
        .into_iter()
        .next()
        .map(|o| engine.resolve_label(o)))
}

fn proposal_status(engine: &Engine, proposal: SymbolId) -> AgentResult<ProposalStatus> {
    let Ok(status_pred) = engine.resolve_symbol(PRED_PROPOSAL_STATUS) else {
        return Ok(ProposalStatus::Pending);
    };
    let labels: Vec<String> = engine
        .knowledge_graph()
        .objects_of(proposal, status_pred)
        .into_iter()
        .map(|o| engine.resolve_label(o))
        .collect();
    if labels.iter().any(|l| l == STATUS_CONFIRMED_OBJ) {
        return Ok(ProposalStatus::Confirmed);
    }
    if labels.iter().any(|l| l == STATUS_REJECTED_OBJ) {
        return Ok(ProposalStatus::Rejected);
    }
    Ok(ProposalStatus::Pending)
}

fn read_proposal(
    engine: &Engine,
    proposal: SymbolId,
    predicate: &str,
) -> AgentResult<PromotionProposal> {
    let status = proposal_status(engine, proposal)?;
    let distinct_subjects = single_object_label(engine, proposal, PRED_PROPOSAL_DISTINCT_SUBJECTS)?
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let value_type = single_object_label(engine, proposal, PRED_PROPOSAL_VALUE_TYPE)?
        .and_then(|l| ValueType::from_entity_label(&l))
        .unwrap_or(ValueType::Text);
    let proposed_type = single_object_label(engine, proposal, PRED_PROPOSAL_TYPE)?;
    let proposed_property_name = single_object_label(engine, proposal, PRED_PROPOSAL_PROPERTY_NAME)?
        .unwrap_or_else(|| predicate.to_string());
    let alias_of = single_object_label(engine, proposal, PRED_PROPOSAL_ALIAS_OF)?;
    let rationale = single_object_label(engine, proposal, PRED_PROPOSAL_RATIONALE)?.unwrap_or_default();
    let model = single_object_label(engine, proposal, PRED_PROPOSAL_MODEL)?.unwrap_or_default();

    Ok(PromotionProposal {
        id: predicate.to_string(),
        predicate: predicate.to_string(),
        status,
        distinct_subjects,
        value_type,
        proposed_type,
        proposed_property_name,
        alias_of,
        rationale,
        model,
    })
}

/// Fetch the stored proposal for `predicate`, if one exists (any status).
pub fn get_proposal(engine: &Engine, predicate: &str) -> AgentResult<Option<PromotionProposal>> {
    let Ok(proposal) = engine.resolve_symbol(&proposal_label(predicate)) else {
        return Ok(None);
    };
    Ok(Some(read_proposal(engine, proposal, predicate)?))
}

/// List every proposal still in [`ProposalStatus::Pending`], sorted by
/// predicate for deterministic output (`akh schema proposals`).
pub fn list_pending_proposals(engine: &Engine) -> AgentResult<Vec<PromotionProposal>> {
    let mut proposals = Vec::new();
    for symbol in engine.item_memory().all_symbols() {
        let label = engine.resolve_label(symbol);
        let Some(predicate) = label.strip_prefix(PROPOSAL_ID_PREFIX) else {
            continue;
        };
        let proposal = read_proposal(engine, symbol, predicate)?;
        if proposal.status == ProposalStatus::Pending {
            proposals.push(proposal);
        }
    }
    proposals.sort_by(|a, b| a.predicate.cmp(&b.predicate));
    Ok(proposals)
}

fn require_pending(
    engine: &Engine,
    predicate: &str,
) -> AgentResult<(SymbolId, PromotionProposal)> {
    let Some(proposal) = get_proposal(engine, predicate)? else {
        return Err(AgentError::PromotionNotFound {
            predicate: predicate.to_string(),
        });
    };
    if proposal.status != ProposalStatus::Pending {
        return Err(AgentError::PromotionNotPending {
            predicate: predicate.to_string(),
            status: proposal.status.to_string(),
        });
    }
    let symbol = engine.resolve_symbol(&proposal_label(predicate))?;
    Ok((symbol, proposal))
}

fn mark_status(engine: &Engine, proposal: SymbolId, status_obj: &str) -> AgentResult<()> {
    let status_pred = engine.resolve_or_create_relation(PRED_PROPOSAL_STATUS)?;
    let object = engine.resolve_or_create_entity(status_obj)?;
    engine.add_triple(&Triple::new(proposal, status_pred, object))?;
    Ok(())
}

/// Reject a pending proposal so it never re-surfaces
/// ([`run_promotion_pass`] skips any predicate with an existing proposal,
/// confirmed or rejected). Errors if no proposal exists, or it already left
/// `pending`.
pub fn reject_proposal(
    engine: &Engine,
    predicate: &str,
    reason: Option<&str>,
) -> AgentResult<()> {
    let (proposal, _) = require_pending(engine, predicate)?;
    if let Some(reason) = reason
        && !reason.trim().is_empty()
    {
        let pred = engine.resolve_or_create_relation(PRED_PROPOSAL_REJECT_REASON)?;
        let object = engine.resolve_or_create_entity(reason)?;
        engine.add_triple(&Triple::new(proposal, pred, object))?;
    }
    mark_status(engine, proposal, STATUS_REJECTED_OBJ)
}

// ---------------------------------------------------------------------------
// Promotion pass — mechanical candidates → LLM advice → stored proposals
// ---------------------------------------------------------------------------

/// Outcome of one [`run_promotion_pass`] call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PromotionReport {
    pub candidates_considered: usize,
    pub proposals_created: usize,
    /// Candidates skipped because a proposal (any status) already exists —
    /// this is the "rejected proposals do not re-surface" behavior in
    /// aggregate form.
    pub already_proposed_skipped: usize,
    /// `(predicate, reason)` for candidates whose model call failed. Not
    /// fatal to the pass (unlike `agent::librarian::run_librarian_pass`'s
    /// batch-atomic watermark — there is no watermark here, so a failed
    /// candidate simply has no proposal stored and is retried automatically
    /// on the next pass).
    pub failed: Vec<(String, String)>,
}

fn schema_type_digests(engine: &Engine) -> AgentResult<Vec<SchemaTypeDigest>> {
    let digest = crate::schema::digest(engine)?;
    Ok(digest
        .types
        .into_iter()
        .map(|t| SchemaTypeDigest {
            name: t.name,
            properties: t.properties.into_iter().map(|p| p.name).collect(),
        })
        .collect())
}

/// Find candidates ([`promotion_candidates`]) and, for every one not already
/// proposed, ask `model` for [`PromotionAdvice`] and store the result as a
/// pending [`PromotionProposal`]. Never touches a [`SchemaRegistry`] — see
/// the module docs' "never-blocked invariant" note.
pub fn run_promotion_pass(
    engine: &Engine,
    model: &dyn LibrarianModel,
    thresholds: &PromotionThresholds,
) -> AgentResult<PromotionReport> {
    let candidates = promotion_candidates(engine, thresholds);
    let schema_types = schema_type_digests(engine)?;

    let mut report = PromotionReport {
        candidates_considered: candidates.len(),
        ..Default::default()
    };

    for candidate in &candidates {
        if has_proposal(engine, &candidate.predicate) {
            report.already_proposed_skipped += 1;
            continue;
        }

        let request = PromotionRequest {
            predicate: candidate.predicate.clone(),
            distinct_subjects: candidate.distinct_subjects,
            example_subjects: candidate.example_subjects.clone(),
            example_values: candidate.example_values.clone(),
            schema_types: schema_types.clone(),
        };

        match model.propose_promotion(&request) {
            Ok(advice) => {
                store_proposal(engine, candidate, &advice, model.label())?;
                report.proposals_created += 1;
            }
            Err(e) => {
                report.failed.push((candidate.predicate.clone(), e.to_string()));
            }
        }
    }

    Ok(report)
}

// ---------------------------------------------------------------------------
// Promotion (human-confirm admission) + aliasing
// ---------------------------------------------------------------------------

/// Fixed version every promotion pack is minted at — see the module docs'
/// "why one pack per proposal" section for why this never needs to bump.
const PROMOTION_PACK_VERSION: &str = "1.0.0";

/// Result of a successful [`promote_proposal`] call.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind")]
pub enum PromotionOutcome {
    /// A new schema pack was compiled, admitted, installed, and activated.
    Promoted {
        pack_name: String,
        pack_version: String,
        /// The newly-minted type name carrying the promoted property.
        type_name: String,
        /// The installed type `type_name` extends via `is-a`, or `None` when
        /// the proposal named no type (or named one that turned out not to
        /// be actually installed — a safe fallback, not an error; see the
        /// module docs).
        supertype: Option<String>,
        property_name: String,
        microtheory_label: String,
    },
    /// The frontier predicate was aliased onto an existing core term — no
    /// new pack, no new property (R5: the core stays small).
    Aliased { alias_of: String },
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Deterministic subtype name for a promoted `(base_type, property)` pair —
/// see the module docs "Why a new subtype" section.
fn promoted_subtype_name(base_type: &str, property: &str) -> String {
    format!("{base_type}With{}", capitalize(property))
}

/// Deterministic standalone type name when no (valid, installed) supertype
/// is available.
fn promoted_standalone_type_name(property: &str) -> String {
    format!("{}Fact", capitalize(property))
}

/// Sanitize a frontier predicate into a pack-name fragment: `pack.name` must
/// be non-empty and whitespace-free ([`crate::schema::source::validate`]);
/// frontier predicates are ordinary camelCase tokens today, so this is a
/// defensive fallback rather than a load-bearing transform.
fn sanitize_pack_name_fragment(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Build the portable JSON source for a promotion pack: one type
/// (`type_name`, optionally `is_a base_type`) carrying exactly one property
/// (`property_name: value_type`).
fn build_promotion_pack_source(
    pack_name: &str,
    type_name: &str,
    base_type: Option<&str>,
    property_name: &str,
    value_type: ValueType,
) -> String {
    let assumes: Vec<&str> = base_type.into_iter().collect();
    serde_json::json!({
        "pack": { "name": pack_name, "version": PROMOTION_PACK_VERSION, "scope": "shared" },
        "assumes": assumes,
        "types": [{
            "name": type_name,
            "is_a": base_type,
            "properties": [{ "name": property_name, "value_type": value_type }],
        }],
    })
    .to_string()
}

/// Promote a pending proposal (ADR 081 decision 8 — this call itself *is*
/// the human confirm; callers are responsible for gating it behind an
/// explicit `--yes`/confirmation UI, mirroring `schema::compile`'s existing
/// `akh schema admit --yes` precedent, which is equally trusting of its
/// caller). `importer` is the workspace's effective-schema context entity
/// ([`crate::schema::install::effective_schema_label`]) the new pack
/// activates into — unused for an [`PromotionOutcome::Aliased`] outcome
/// (aliasing installs nothing).
///
/// Errors if no proposal exists for `predicate`, or it already left
/// `pending` (double-promotion/double-rejection guard).
pub fn promote_proposal(
    engine: &Engine,
    registry: &dyn SchemaRegistry,
    predicate: &str,
    confirmed_by: &str,
    importer: SymbolId,
) -> AgentResult<PromotionOutcome> {
    let (proposal, fields) = require_pending(engine, predicate)?;

    if let Some(alias_of) = &fields.alias_of {
        record_alias(engine, predicate, alias_of)?;
        mark_status(engine, proposal, STATUS_CONFIRMED_OBJ)?;
        return Ok(PromotionOutcome::Aliased {
            alias_of: alias_of.clone(),
        });
    }

    let supertype = fields.proposed_type.as_deref().filter(|t| {
        engine
            .resolve_symbol(t)
            .map(|s| crate::schema::is_installed_type(engine, s))
            .unwrap_or(false)
    });

    let type_name = match supertype {
        Some(base) => promoted_subtype_name(base, &fields.proposed_property_name),
        None => promoted_standalone_type_name(&fields.proposed_property_name),
    };
    let pack_name = format!(
        "akh-frontier-promotion-{}",
        sanitize_pack_name_fragment(predicate)
    );

    let source_text = build_promotion_pack_source(
        &pack_name,
        &type_name,
        supertype,
        &fields.proposed_property_name,
        fields.value_type,
    );
    let pack = compile(&source_text)?;
    let row = RegistryRow::from_pack(&pack, confirmed_by)?;
    registry.admit(row)?;
    install_pack(engine, &pack)?;
    activate_pack(engine, importer, &pack)?;

    mark_status(engine, proposal, STATUS_CONFIRMED_OBJ)?;

    Ok(PromotionOutcome::Promoted {
        pack_name: pack.name.clone(),
        pack_version: pack.version.clone(),
        type_name,
        supertype: supertype.map(str::to_string),
        property_name: fields.proposed_property_name,
        microtheory_label: pack.microtheory_label(),
    })
}

fn record_alias(engine: &Engine, predicate: &str, alias_of: &str) -> AgentResult<()> {
    let subject = engine.resolve_or_create_entity(predicate)?;
    let pred = engine.resolve_or_create_relation(PRED_ALIAS_OF)?;
    let object = engine.resolve_or_create_entity(alias_of)?;
    engine.add_triple(&Triple::new(subject, pred, object))?;

    let mut prov = ProvenanceRecord::new(
        subject,
        DerivationKind::SchemaDiscovered {
            pattern_type: "frontier-promotion-alias".to_string(),
        },
    )
    .with_sources(vec![subject, pred, object])
    .with_confidence(1.0);
    let _ = engine.store_provenance(&mut prov);
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::schema::compile::PRED_IS_A;
    use crate::schema::{RegistryRow as Row, check_admission};
    use crate::vsa::Dimension;
    use std::sync::Mutex;

    fn test_engine() -> (tempfile::TempDir, Engine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        (dir, engine)
    }

    const SEED_PACK: &str = r#"{
        "pack": {"name": "promotion-test", "version": "0.1.0"},
        "types": [
            {"name": "Person", "properties": [
                {"name": "email", "value_type": "text"}]}
        ]
    }"#;

    fn install_seed(engine: &Engine) {
        let pack = crate::schema::compile(SEED_PACK).unwrap();
        crate::schema::install_pack(engine, &pack).unwrap();
    }

    /// Assert `count` frontier triples `(subject_i, note:frontier/{predicate},
    /// value)` directly (bypassing the Librarian), mirroring how
    /// `schema::project::project_note`/`agent::librarian::assert_candidate`
    /// both land unknown keys under `FRONTIER_PREFIX`.
    fn assert_frontier_triples(engine: &Engine, predicate: &str, values: &[(&str, &str)]) {
        let pred = engine
            .resolve_or_create_relation(&format!("{FRONTIER_PREFIX}{predicate}"))
            .unwrap();
        for (subject, value) in values {
            let s = engine.resolve_or_create_entity(subject).unwrap();
            let o = engine.resolve_or_create_entity(value).unwrap();
            engine.add_triple(&Triple::new(s, pred, o)).unwrap();
        }
    }

    /// A trivial in-memory [`SchemaRegistry`] — same shape as
    /// `schema::compile::tests::InMemoryRegistry` (that one is private to
    /// its module), used here to exercise [`promote_proposal`]'s registry
    /// admission without Postgres.
    #[derive(Default)]
    struct InMemoryRegistry {
        rows: Mutex<Vec<Row>>,
    }

    impl SchemaRegistry for InMemoryRegistry {
        fn admit(&self, row: Row) -> crate::schema::SchemaResult<()> {
            check_admission(self, &row)?;
            self.rows.lock().unwrap().push(row);
            Ok(())
        }

        fn fetch(&self, name: &str, version: &str) -> crate::schema::SchemaResult<Option<Row>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .find(|r| r.name == name && r.version == version)
                .cloned())
        }

        fn list(&self) -> crate::schema::SchemaResult<Vec<(String, String)>> {
            Ok(self
                .rows
                .lock()
                .unwrap()
                .iter()
                .map(|r| (r.name.clone(), r.version.clone()))
                .collect())
        }
    }

    // --- promotion_candidates ------------------------------------------

    #[test]
    fn candidate_requires_min_distinct_subjects() {
        let (_dir, engine) = test_engine();
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red")],
        );

        let thresholds = PromotionThresholds {
            min_distinct_subjects: 3,
        };
        assert!(promotion_candidates(&engine, &thresholds).is_empty());

        let looser = PromotionThresholds {
            min_distinct_subjects: 2,
        };
        let candidates = promotion_candidates(&engine, &looser);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].predicate, "favoriteColor");
        assert_eq!(candidates[0].distinct_subjects, 2);
        assert_eq!(candidates[0].value_type, ValueType::Text);
    }

    #[test]
    fn inconsistent_value_shape_excludes_candidate() {
        let (_dir, engine) = test_engine();
        // Three distinct subjects clears the threshold, but the shapes
        // disagree (text vs. number vs. date) — must be excluded.
        assert_frontier_triples(
            &engine,
            "mixedShape",
            &[("A", "teal"), ("B", "42"), ("C", "2024-01-01")],
        );

        let candidates = promotion_candidates(&engine, &PromotionThresholds::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn consistent_numeric_shape_is_detected() {
        let (_dir, engine) = test_engine();
        assert_frontier_triples(
            &engine,
            "score",
            &[("A", "1"), ("B", "2"), ("C", "3")],
        );
        let candidates = promotion_candidates(&engine, &PromotionThresholds::default());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].value_type, ValueType::Number);
    }

    #[test]
    fn frontier_is_a_predicate_is_never_a_candidate() {
        let (_dir, engine) = test_engine();
        let frontier_is_a = engine
            .resolve_or_create_relation(&format!("{FRONTIER_PREFIX}{PRED_IS_A}"))
            .unwrap();
        for i in 0..5 {
            let s = engine.resolve_or_create_entity(&format!("s{i}")).unwrap();
            let o = engine.resolve_or_create_entity("SomeUninstalledType").unwrap();
            engine.add_triple(&Triple::new(s, frontier_is_a, o)).unwrap();
        }
        assert!(promotion_candidates(&engine, &PromotionThresholds::default()).is_empty());
    }

    #[test]
    fn below_threshold_terms_never_proposed() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(&engine, "rare", &[("A", "x")]); // 1 subject, default min 3

        let model = crate::agent::librarian::FakeLibrarianModel::new();
        let report =
            run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        assert_eq!(report.candidates_considered, 0);
        assert_eq!(report.proposals_created, 0);
        assert!(list_pending_proposals(&engine).unwrap().is_empty());
    }

    // --- proposal storage -----------------------------------------------

    #[test]
    fn proposal_round_trips_through_storage() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );

        let model = crate::agent::librarian::FakeLibrarianModel::new().with_proposal(
            "favoriteColor",
            crate::agent::librarian::PromotionAdvice {
                proposed_type: Some("Person".into()),
                proposed_property_name: "favoriteColor".into(),
                alias_of: None,
                rationale: "recurs on Person subjects".into(),
            },
        );

        let report =
            run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        assert_eq!(report.proposals_created, 1);
        assert_eq!(report.already_proposed_skipped, 0);

        let proposal = get_proposal(&engine, "favoriteColor").unwrap().unwrap();
        assert_eq!(proposal.id, "favoriteColor");
        assert_eq!(proposal.status, ProposalStatus::Pending);
        assert_eq!(proposal.distinct_subjects, 3);
        assert_eq!(proposal.value_type, ValueType::Text);
        assert_eq!(proposal.proposed_type.as_deref(), Some("Person"));
        assert_eq!(proposal.proposed_property_name, "favoriteColor");
        assert_eq!(proposal.rationale, "recurs on Person subjects");
        assert_eq!(proposal.model, "fake");

        let pending = list_pending_proposals(&engine).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].predicate, "favoriteColor");
    }

    #[test]
    fn run_promotion_pass_skips_already_proposed_predicates() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new();

        let first = run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        assert_eq!(first.proposals_created, 1);

        let second =
            run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        assert_eq!(second.proposals_created, 0);
        assert_eq!(second.already_proposed_skipped, 1);
        assert_eq!(list_pending_proposals(&engine).unwrap().len(), 1);
    }

    #[test]
    fn model_failure_is_recorded_and_retried_next_pass() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let failing =
            crate::agent::librarian::FakeLibrarianModel::new().failing_on("favoriteColor");

        let report =
            run_promotion_pass(&engine, &failing, &PromotionThresholds::default()).unwrap();
        assert_eq!(report.proposals_created, 0);
        assert_eq!(report.failed.len(), 1);
        assert_eq!(report.failed[0].0, "favoriteColor");
        assert!(list_pending_proposals(&engine).unwrap().is_empty());

        // No proposal was stored, so a healthy model retries it successfully.
        let healthy = crate::agent::librarian::FakeLibrarianModel::new();
        let retry =
            run_promotion_pass(&engine, &healthy, &PromotionThresholds::default()).unwrap();
        assert_eq!(retry.proposals_created, 1);
    }

    // --- reject -----------------------------------------------------------

    #[test]
    fn rejected_proposal_does_not_resurface_on_next_pass() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new();
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();

        reject_proposal(&engine, "favoriteColor", Some("noise, not a real property")).unwrap();
        let proposal = get_proposal(&engine, "favoriteColor").unwrap().unwrap();
        assert_eq!(proposal.status, ProposalStatus::Rejected);
        assert!(list_pending_proposals(&engine).unwrap().is_empty());

        // Even with fresh recurring evidence, a re-run must not re-propose.
        assert_frontier_triples(&engine, "favoriteColor", &[("Cid", "green")]);
        let report =
            run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        assert_eq!(report.proposals_created, 0);
        assert_eq!(report.already_proposed_skipped, 1);
    }

    #[test]
    fn rejecting_a_non_pending_proposal_errors() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new();
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        reject_proposal(&engine, "favoriteColor", None).unwrap();

        assert!(matches!(
            reject_proposal(&engine, "favoriteColor", None),
            Err(AgentError::PromotionNotPending { .. })
        ));
    }

    #[test]
    fn rejecting_unknown_predicate_errors() {
        let (_dir, engine) = test_engine();
        assert!(matches!(
            reject_proposal(&engine, "neverProposed", None),
            Err(AgentError::PromotionNotFound { .. })
        ));
    }

    // --- promote (registry admission) --------------------------------------

    fn importer(engine: &Engine) -> SymbolId {
        engine
            .resolve_or_create_entity("mt:akh/test-effective")
            .unwrap()
    }

    #[test]
    fn promote_admits_pack_and_property_becomes_queryable() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new().with_proposal(
            "favoriteColor",
            crate::agent::librarian::PromotionAdvice {
                proposed_type: Some("Person".into()),
                proposed_property_name: "favoriteColor".into(),
                alias_of: None,
                rationale: "recurs on Person subjects".into(),
            },
        );
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();

        let registry = InMemoryRegistry::default();
        let outcome = promote_proposal(
            &engine,
            &registry,
            "favoriteColor",
            "toasty",
            importer(&engine),
        )
        .unwrap();

        let PromotionOutcome::Promoted {
            pack_name,
            pack_version,
            type_name,
            supertype,
            property_name,
            ..
        } = outcome
        else {
            panic!("expected a Promoted outcome");
        };
        assert_eq!(pack_name, "akh-frontier-promotion-favoriteColor");
        assert_eq!(pack_version, "1.0.0");
        assert_eq!(type_name, "PersonWithFavoriteColor");
        assert_eq!(supertype.as_deref(), Some("Person"));
        assert_eq!(property_name, "favoriteColor");

        // Admitted to the registry.
        assert_eq!(
            registry.list().unwrap(),
            vec![(pack_name.clone(), "1.0.0".to_string())]
        );

        // Installed + activated: the new type carries the property and
        // is-a's Person; `schema::digest` sees it in the engine's live
        // schema (what future Librarian passes/projections consult).
        let digest = crate::schema::digest(&engine).unwrap();
        let t = digest
            .types
            .iter()
            .find(|t| t.name == "PersonWithFavoriteColor")
            .expect("promoted type present in digest");
        assert!(t.chain.contains(&"Person".to_string()));
        assert!(t.properties.iter().any(|p| p.name == "favoriteColor"));

        // The proposal is now confirmed, not pending.
        let proposal = get_proposal(&engine, "favoriteColor").unwrap().unwrap();
        assert_eq!(proposal.status, ProposalStatus::Confirmed);
        assert!(list_pending_proposals(&engine).unwrap().is_empty());
    }

    #[test]
    fn promote_falls_back_to_standalone_type_when_no_type_proposed() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "shoeSize",
            &[("Jane", "9"), ("Bob", "10"), ("Amy", "8")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new(); // default advice: no type

        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();
        let registry = InMemoryRegistry::default();
        let outcome =
            promote_proposal(&engine, &registry, "shoeSize", "toasty", importer(&engine)).unwrap();

        let PromotionOutcome::Promoted {
            type_name,
            supertype,
            ..
        } = outcome
        else {
            panic!("expected a Promoted outcome");
        };
        assert_eq!(type_name, "ShoeSizeFact");
        assert!(supertype.is_none());
    }

    #[test]
    fn promote_falls_back_to_standalone_when_proposed_type_is_not_installed() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "nickname",
            &[("Jane", "J"), ("Bob", "B"), ("Amy", "A")],
        );
        // The LLM hallucinates a type that was never installed.
        let model = crate::agent::librarian::FakeLibrarianModel::new().with_proposal(
            "nickname",
            crate::agent::librarian::PromotionAdvice {
                proposed_type: Some("Nonexistent".into()),
                proposed_property_name: "nickname".into(),
                alias_of: None,
                rationale: "hallucinated".into(),
            },
        );
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();

        let registry = InMemoryRegistry::default();
        let outcome =
            promote_proposal(&engine, &registry, "nickname", "toasty", importer(&engine)).unwrap();
        let PromotionOutcome::Promoted { supertype, .. } = outcome else {
            panic!("expected a Promoted outcome");
        };
        assert!(supertype.is_none(), "hallucinated supertype must not be trusted");
    }

    #[test]
    fn promoting_a_non_pending_proposal_errors() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "favoriteColor",
            &[("Jane", "teal"), ("Bob", "red"), ("Amy", "blue")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new();
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();

        let registry = InMemoryRegistry::default();
        promote_proposal(
            &engine,
            &registry,
            "favoriteColor",
            "toasty",
            importer(&engine),
        )
        .unwrap();

        assert!(matches!(
            promote_proposal(&engine, &registry, "favoriteColor", "toasty", importer(&engine)),
            Err(AgentError::PromotionNotPending { .. })
        ));
    }

    #[test]
    fn promoting_unknown_predicate_errors() {
        let (_dir, engine) = test_engine();
        let registry = InMemoryRegistry::default();
        assert!(matches!(
            promote_proposal(&engine, &registry, "neverProposed", "toasty", importer(&engine)),
            Err(AgentError::PromotionNotFound { .. })
        ));
    }

    // --- alias --------------------------------------------------------------

    #[test]
    fn promote_alias_records_alias_triple_and_creates_no_pack() {
        let (_dir, engine) = test_engine();
        install_seed(&engine);
        assert_frontier_triples(
            &engine,
            "mailAddress",
            &[("Jane", "jane@example.org"), ("Bob", "bob@example.org"), ("Amy", "amy@x.org")],
        );
        let model = crate::agent::librarian::FakeLibrarianModel::new().with_proposal(
            "mailAddress",
            crate::agent::librarian::PromotionAdvice {
                proposed_type: None,
                proposed_property_name: "mailAddress".into(),
                alias_of: Some("email".into()),
                rationale: "synonym of the existing `email` property".into(),
            },
        );
        run_promotion_pass(&engine, &model, &PromotionThresholds::default()).unwrap();

        let registry = InMemoryRegistry::default();
        let outcome = promote_proposal(
            &engine,
            &registry,
            "mailAddress",
            "toasty",
            importer(&engine),
        )
        .unwrap();

        assert!(matches!(
            outcome,
            PromotionOutcome::Aliased { ref alias_of } if alias_of == "email"
        ));
        // No new pack — registry untouched.
        assert!(registry.list().unwrap().is_empty());

        // The alias edge is queryable.
        let mail_address = engine.resolve_symbol("mailAddress").unwrap();
        let alias_of_pred = engine.resolve_symbol(PRED_ALIAS_OF).unwrap();
        let email = engine.resolve_symbol("email").unwrap();
        assert!(
            engine
                .knowledge_graph()
                .objects_of(mail_address, alias_of_pred)
                .contains(&email)
        );

        let proposal = get_proposal(&engine, "mailAddress").unwrap().unwrap();
        assert_eq!(proposal.status, ProposalStatus::Confirmed);
    }
}
