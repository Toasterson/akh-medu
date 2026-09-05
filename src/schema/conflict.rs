//! Multi-akh Dempster-Shafer conflict resolution (ADR 081 Slice 8, work-item
//! 8, `docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`).
//!
//! ADR 081 decision 12: when two sources (two akhs, or two Librarian runs)
//! assert different objects for the same `(subject, predicate)` in the
//! **shared core**, the conflict must never resolve by last-writer-wins.
//! Both assertions survive in the KG (the engine's default
//! [`crate::engine::ContradictionPolicy::Warn`] already does this — it logs
//! and inserts, never silently drops one side), and this module makes the
//! disagreement *legible*: each candidate object gets a Dempster-Shafer
//! belief interval, weighted by the reliability of the source(s) that
//! asserted it, reusing the Phase 17 evidence-theory machinery
//! ([`crate::agent::evidence`]) and Phase 18 source-reliability machinery
//! ([`crate::agent::source_reliability`]) wholesale — no new probability
//! math lives here.
//!
//! ## Attributing a source to a triple
//!
//! A [`crate::graph::Triple`] carries no "who asserted this" field, and
//! generic [`crate::provenance::ProvenanceRecord::sources`] is already
//! overloaded (project_note, for instance, fills it with the note entity,
//! its type, and every value entity — the KG symbols a derivation *depends*
//! on, not "who is vouching for it"). Rather than overload that further,
//! source attribution here is explicit: [`assert_with_source`] is the one
//! way to register a source-attributed assertion, and it stores a dedicated
//! [`crate::provenance::DerivationKind::SourceAttributed`] provenance record
//! alongside the triple. [`conflicting_assertions`] reads exactly those
//! records back — no attempt is made to infer a source from provenance
//! written by other paths (`project_note`, the Librarian, etc.); those
//! triples simply never appear as a `SourcedAssertion` and cannot enter a
//! conflict comparison. Wiring multi-akh sync to call `assert_with_source`
//! is out of this slice's scope (no networking) — see ADR 081 decision 12's
//! "the unit is two conflicting assertions ... in one engine".
//!
//! ## Scope
//!
//! This module is intentionally *not* a full evidence-management service —
//! [`crate::agent::evidence::EvidenceManager`] already is one, with alerting
//! and persistence. This module answers one narrower question over the live
//! KG: "for this `(subject, predicate)`, what does every source-attributed
//! candidate object's belief interval look like, side by side?" —
//! [`conflicting_assertions`] (representation + query) and
//! [`resolve_conflict`] (the belief-weighted resolution helper) are that
//! answer.

use std::collections::HashMap;

use crate::agent::evidence::{BeliefInterval, MassFunction};
use crate::agent::source_reliability::SourceReliabilityManager;
use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Representation
// ---------------------------------------------------------------------------

/// One source's assertion of a candidate object for a `(subject, predicate)`
/// pair, as recorded by [`assert_with_source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcedAssertion {
    /// The candidate object value this source asserts.
    pub object: SymbolId,
    /// Who/what asserted it — an akh identity, a note entity, a Librarian
    /// run marker, etc.
    pub source: SymbolId,
}

/// Every source-attributed assertion found for a `(subject, predicate)`
/// pair, with `distinct_objects() > 1` meaning sources disagree — the
/// "conflict" [`conflicting_assertions`] is named for.
#[derive(Debug, Clone)]
pub struct AssertionConflict {
    pub subject: SymbolId,
    pub predicate: SymbolId,
    pub assertions: Vec<SourcedAssertion>,
}

impl AssertionConflict {
    /// Number of distinct candidate objects asserted.
    pub fn distinct_objects(&self) -> usize {
        let mut seen: Vec<SymbolId> = self.assertions.iter().map(|a| a.object).collect();
        seen.sort_by_key(|s| s.get());
        seen.dedup();
        seen.len()
    }

    /// Whether sources actually disagree (more than one candidate object).
    pub fn is_conflicting(&self) -> bool {
        self.distinct_objects() > 1
    }
}

/// A candidate object's resolved belief, after combining every supporting
/// source's evidence via Dempster's rule, each discounted by that source's
/// Admiralty-rated reliability (Phase 18). Neither candidate "wins" —
/// callers see every [`ResolvedAssertion`] for the conflict and decide what
/// to do with the disagreement (surface it, wait for more evidence, etc.).
#[derive(Debug, Clone)]
pub struct ResolvedAssertion {
    pub object: SymbolId,
    pub interval: BeliefInterval,
    /// Every source that asserted this candidate object.
    pub supporting_sources: Vec<SymbolId>,
}

// ---------------------------------------------------------------------------
// Registering a source-attributed assertion
// ---------------------------------------------------------------------------

/// Assert `(subject, predicate, object)` into the KG with an explicit source
/// attribution, so a later disagreement about the same `(subject,
/// predicate)` can be found by [`conflicting_assertions`] and resolved by
/// [`resolve_conflict`] instead of one assertion silently overwriting the
/// other (ADR 081 decision 12).
///
/// Tolerates an engine with no provenance ledger configured (in-memory,
/// `EngineConfig.data_dir: None`) the same way `schema::project::project_note`
/// does: the triple still asserts, but with no persisted provenance there is
/// nothing for `conflicting_assertions` to read back, so this degrades to
/// "the triple exists, but is invisible to conflict detection" rather than
/// failing the write.
pub fn assert_with_source(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
    object: SymbolId,
    source: SymbolId,
    confidence: f32,
) -> AkhResult<()> {
    let triple = Triple::new(subject, predicate, object).with_confidence(confidence);
    engine.add_triple(&triple)?;

    let mut prov = ProvenanceRecord::new(
        object,
        DerivationKind::SourceAttributed {
            subject,
            predicate,
            source,
        },
    )
    .with_sources(vec![subject, predicate, source])
    .with_confidence(confidence);
    let _ = engine.store_provenance(&mut prov);

    Ok(())
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// Find every source-attributed assertion (registered via
/// [`assert_with_source`]) for a live `(subject, predicate)` triple.
///
/// Returns `None` when there is nothing to compare (no source-attributed
/// candidates at all, or only one) — a single, unanimous assertion is not a
/// conflict. Otherwise returns `Some` with every `SourcedAssertion` found,
/// whether or not they actually disagree — see
/// [`AssertionConflict::is_conflicting`].
pub fn conflicting_assertions(
    engine: &Engine,
    subject: SymbolId,
    predicate: SymbolId,
) -> Option<AssertionConflict> {
    let candidate_objects: Vec<SymbolId> = engine
        .triples_from(subject)
        .into_iter()
        .filter(|t| t.predicate == predicate)
        .map(|t| t.object)
        .collect();

    let mut assertions = Vec::new();
    for object in candidate_objects {
        let Ok(records) = engine.provenance_of(object) else {
            continue; // No persistence configured — nothing to attribute.
        };
        for record in records {
            if let DerivationKind::SourceAttributed {
                subject: s,
                predicate: p,
                source,
            } = record.kind
                && s == subject
                && p == predicate
            {
                assertions.push(SourcedAssertion { object, source });
            }
        }
    }

    if assertions.len() < 2 {
        return None;
    }

    Some(AssertionConflict {
        subject,
        predicate,
        assertions,
    })
}

// ---------------------------------------------------------------------------
// Belief-weighted resolution
// ---------------------------------------------------------------------------

/// Resolve a conflict into a per-candidate-object [`BeliefInterval`],
/// reusing Phase 17's Dempster-Shafer combination
/// ([`MassFunction::combine_multiple`]) and Phase 18's source-reliability
/// discounting ([`SourceReliabilityManager::rate_source`]).
///
/// Each source's support is `MassFunction::from_confidence(1.0)` (the source
/// asserted the fact outright — "confidence" here is about the *source's*
/// reliability, not a per-assertion scalar the caller has to supply)
/// discounted by that source's rated reliability factor before combining —
/// an unreliable source's assertion is attenuated toward ignorance rather
/// than either counted at face value or thrown away, so a lone low-trust
/// source never silently beats a well-attested rival, and never gets
/// silently dropped either.
///
/// Results are sorted by descending belief so the strongest candidate sorts
/// first — informational only, this function never picks a winner for the
/// caller.
pub fn resolve_conflict(
    conflict: &AssertionConflict,
    reliability: &SourceReliabilityManager,
) -> Vec<ResolvedAssertion> {
    let mut by_object: HashMap<SymbolId, Vec<SymbolId>> = HashMap::new();
    for a in &conflict.assertions {
        by_object.entry(a.object).or_default().push(a.source);
    }

    let mut out: Vec<ResolvedAssertion> = by_object
        .into_iter()
        .map(|(object, sources)| {
            let masses: Vec<MassFunction> = sources
                .iter()
                .map(|source| {
                    let rating = reliability.rate_source(*source);
                    MassFunction::from_confidence(1.0).discount(rating.source.to_factor())
                })
                .collect();
            let combined =
                MassFunction::combine_multiple(&masses).unwrap_or_else(|_| MassFunction::vacuous());
            ResolvedAssertion {
                object,
                interval: combined.belief_interval(),
                supporting_sources: sources,
            }
        })
        .collect();

    out.sort_by(|a, b| {
        b.interval
            .belief
            .partial_cmp(&a.interval.belief)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::source_reliability::TrustDimension;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

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

    #[test]
    fn single_source_is_not_a_conflict() {
        let (_dir, engine) = test_engine();
        let subject = engine.resolve_or_create_entity("Ticket-42").unwrap();
        let predicate = engine.resolve_or_create_relation("status").unwrap();
        let object = engine.resolve_or_create_entity("Open").unwrap();
        let source = engine.resolve_or_create_entity("akh:Tecton").unwrap();

        assert_with_source(&engine, subject, predicate, object, source, 0.9).unwrap();

        assert!(conflicting_assertions(&engine, subject, predicate).is_none());
    }

    #[test]
    fn two_sources_agreeing_is_not_a_conflicting_object_set() {
        let (_dir, engine) = test_engine();
        let subject = engine.resolve_or_create_entity("Ticket-43").unwrap();
        let predicate = engine.resolve_or_create_relation("status").unwrap();
        let object = engine.resolve_or_create_entity("Open").unwrap();
        let akh_a = engine.resolve_or_create_entity("akh:Tecton").unwrap();
        let akh_b = engine.resolve_or_create_entity("akh:Thoth").unwrap();

        assert_with_source(&engine, subject, predicate, object, akh_a, 0.9).unwrap();
        assert_with_source(&engine, subject, predicate, object, akh_b, 0.9).unwrap();

        let conflict = conflicting_assertions(&engine, subject, predicate).unwrap();
        assert_eq!(conflict.assertions.len(), 2);
        assert_eq!(conflict.distinct_objects(), 1, "same object, two witnesses");
        assert!(!conflict.is_conflicting());
    }

    #[test]
    fn disagreeing_sources_yield_a_belief_interval_per_candidate_neither_silently_wins() {
        let (_dir, engine) = test_engine();
        let subject = engine.resolve_or_create_entity("Ticket-44").unwrap();
        let predicate = engine.resolve_or_create_relation("status").unwrap();
        let open = engine.resolve_or_create_entity("Open").unwrap();
        let closed = engine.resolve_or_create_entity("Closed").unwrap();
        let akh_a = engine.resolve_or_create_entity("akh:Tecton").unwrap();
        let akh_b = engine.resolve_or_create_entity("akh:Thoth").unwrap();

        // Two akhs disagree: Tecton says Open, Thoth says Closed.
        assert_with_source(&engine, subject, predicate, open, akh_a, 0.9).unwrap();
        assert_with_source(&engine, subject, predicate, closed, akh_b, 0.9).unwrap();

        // Both triples are actually present — the engine's default
        // Contradiction::Warn policy never silently drops either side.
        assert!(engine.has_triple(subject, predicate, open));
        assert!(engine.has_triple(subject, predicate, closed));

        let conflict = conflicting_assertions(&engine, subject, predicate).unwrap();
        assert!(conflict.is_conflicting());
        assert_eq!(conflict.distinct_objects(), 2);

        // Equal, untracked reliability (both sources default to
        // "CannotJudge" — no observations yet): the resolution must not
        // silently prefer one candidate over the other.
        let reliability = SourceReliabilityManager::new();
        let resolved = resolve_conflict(&conflict, &reliability);
        assert_eq!(resolved.len(), 2, "both candidates get their own interval");
        let bel_open = resolved.iter().find(|r| r.object == open).unwrap().interval;
        let bel_closed = resolved
            .iter()
            .find(|r| r.object == closed)
            .unwrap()
            .interval;
        assert!(
            (bel_open.belief - bel_closed.belief).abs() < 1e-6,
            "equal-reliability sources must not silently favor one candidate: {bel_open:?} vs {bel_closed:?}"
        );
        assert!(
            bel_open.belief > 0.0,
            "an asserted candidate still carries some belief"
        );
    }

    #[test]
    fn reliability_weighting_shifts_belief_toward_the_more_trusted_source() {
        let (_dir, engine) = test_engine();
        let subject = engine.resolve_or_create_entity("Ticket-45").unwrap();
        let predicate = engine.resolve_or_create_relation("status").unwrap();
        let open = engine.resolve_or_create_entity("Open").unwrap();
        let closed = engine.resolve_or_create_entity("Closed").unwrap();
        let trusted = engine.resolve_or_create_entity("akh:Tecton").unwrap();
        let flaky = engine.resolve_or_create_entity("akh:Untrusted").unwrap();

        assert_with_source(&engine, subject, predicate, open, trusted, 0.9).unwrap();
        assert_with_source(&engine, subject, predicate, closed, flaky, 0.9).unwrap();

        let mut reliability = SourceReliabilityManager::new();
        // `trusted` has a long history of accurate claims.
        for _ in 0..20 {
            reliability.verify_claim(trusted, true);
        }
        // `flaky` has a long history of being wrong.
        for _ in 0..20 {
            reliability.update_trust(flaky, TrustDimension::Competence, false, 1.0);
            reliability.update_trust(flaky, TrustDimension::Integrity, false, 1.0);
        }

        let conflict = conflicting_assertions(&engine, subject, predicate).unwrap();
        let resolved = resolve_conflict(&conflict, &reliability);

        let bel_open = resolved.iter().find(|r| r.object == open).unwrap().interval;
        let bel_closed = resolved
            .iter()
            .find(|r| r.object == closed)
            .unwrap()
            .interval;
        assert!(
            bel_open.belief > bel_closed.belief,
            "the trusted source's candidate must out-weigh the flaky source's: \
             open={bel_open:?} closed={bel_closed:?}"
        );
        // Neither is driven all the way to zero — the flaky source's
        // assertion still exists in the KG and still carries *some* mass;
        // this is attenuation, not erasure.
        assert!(
            bel_closed.plausibility > 0.0,
            "the losing candidate is not silently erased"
        );
    }
}
