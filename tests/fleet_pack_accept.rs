//! Fleet memory pack acceptance experiment (Phase 53 follow-up, Anima WI
//! 01a0d569-1cf1 item 7): the built-in `fleet-memory` pack installed next to
//! `roman-memory` on a real akhomed SQLite store, through the real engine
//! (`snapshot::wake`).
//!
//! Ignored by default. Point it at a **private copy** of a store — never the
//! live file:
//!
//! ```sh
//! mkdir -p /tmp/claude-fleetpack
//! sqlite3 "file:$ROOT/state/core/akh.sqlite3?mode=ro" ".backup /tmp/claude-fleetpack/akh.sqlite3"
//! AKH_FLEETPACK_ACCEPT_DB=/tmp/claude-fleetpack/akh.sqlite3 \
//!   cargo test --features server --test fleet_pack_accept -- --ignored --nocapture
//! ```
//!
//! Optional: `AKH_FLEETPACK_ACCEPT_AKHS` (comma list, default
//! `janus,minerva,vesta`).
//!
//! Per akh: wake (which converges the store's active registry packs, i.e.
//! `roman-memory`), bring roman-memory's hygiene to a fixed point (the state
//! the live daemon keeps), record every roman-memory relation, then ensure
//! the fleet pack and run hygiene to a fixed point again. It asserts that
//! roman-memory's declarations and live triples are unchanged, that no
//! declaration conflicts, and that the final pass is a no-op.
#![cfg(feature = "postgres")]

use std::collections::BTreeMap;
use std::sync::Arc;

use akh_medu::engine::Engine;
use akh_medu::schema::{
    BuiltinPack, BuiltinPackOutcome, DefaultPackPolicy, HygieneOptions, HygieneReport,
    MergeRequest, MergeSelection, PredicateCardinality, declared_alias_target,
    declared_cardinality, ensure_builtin_pack, execute_merge, install_pack, run_hygiene,
};
use akh_medu::symbol::SymbolId;

fn assert_private_copy(path: &str) {
    assert!(
        !path.contains("roman-akhs/state"),
        "AKH_FLEETPACK_ACCEPT_DB must be a private copy, not the live store: {path}"
    );
}

/// `(live triples, subjects, subjects with several values)` of a predicate.
fn stats(engine: &Engine, label: &str) -> (usize, usize, usize) {
    let Ok(p) = engine.lookup_symbol(label) else {
        return (0, 0, 0);
    };
    let mut per_subject: BTreeMap<(SymbolId, Option<String>), usize> = BTreeMap::new();
    let triples = engine.triples_with_predicate(p);
    for t in &triples {
        *per_subject
            .entry((t.subject, t.compartment_id.clone()))
            .or_default() += 1;
    }
    (
        triples.len(),
        per_subject.len(),
        per_subject.values().filter(|&&n| n > 1).count(),
    )
}

fn cardinality(engine: &Engine, label: &str) -> PredicateCardinality {
    engine
        .lookup_symbol(label)
        .map(|p| declared_cardinality(engine, p))
        .unwrap_or_default()
}

fn alias_target(engine: &Engine, label: &str) -> Option<String> {
    let p = engine.lookup_symbol(label).ok()?;
    declared_alias_target(engine, p).map(|t| engine.resolve_label(t))
}

/// Hygiene to a fixed point: passes until one is not truncated, the
/// declared-alias merge, then one more pass. Returns every pass report and
/// the triples the merge moved.
fn converge(engine: &Engine) -> (Vec<HygieneReport>, usize) {
    let options = HygieneOptions::for_engine(engine, akh_medu::schema::DEFAULT_HYGIENE_BATCH);
    let mut passes = Vec::new();
    loop {
        let report = run_hygiene(engine, options).expect("hygiene pass");
        let truncated = report.truncated;
        passes.push(report);
        if !truncated || passes.len() >= 50 {
            break;
        }
    }
    let merged = execute_merge(
        engine,
        &MergeRequest {
            selection: MergeSelection::Declared,
            dry_run: false,
            limit: akh_medu::schema::DEFAULT_MERGE_LIMIT,
        },
    )
    .expect("declared merge")
    .moved;
    passes.push(run_hygiene(engine, options).expect("post-merge pass"));
    (passes, merged)
}

fn summarize(passes: &[HygieneReport]) -> String {
    format!(
        "passes {} mutations {} superseded {} repaired {} merged families {} / triples {} provenance {}",
        passes.len(),
        passes.iter().map(|p| p.mutations).sum::<usize>(),
        passes.iter().map(|p| p.superseded).sum::<usize>(),
        passes.iter().map(|p| p.repaired).sum::<usize>(),
        passes.iter().map(|p| p.merged.len()).sum::<usize>(),
        passes
            .iter()
            .map(HygieneReport::merged_triples)
            .sum::<usize>(),
        passes.iter().map(|p| p.provenance_records).sum::<usize>(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs AKH_FLEETPACK_ACCEPT_DB pointing at a private copy of an akhomed SQLite store"]
async fn fleet_pack_next_to_roman_memory_on_a_copied_store() {
    let Ok(db_path) = std::env::var("AKH_FLEETPACK_ACCEPT_DB") else {
        eprintln!("AKH_FLEETPACK_ACCEPT_DB not set; skipping");
        return;
    };
    assert_private_copy(&db_path);
    let akhs =
        std::env::var("AKH_FLEETPACK_ACCEPT_AKHS").unwrap_or_else(|_| "janus,minerva,vesta".into());
    let roman_source = include_str!("../deploy/roman-mini/schemas/roman-memory.json");
    let roman = akh_medu::schema::compile(roman_source).expect("compile roman-memory");
    let roman_relations = akh_medu::schema::parse_source(roman_source)
        .unwrap()
        .relations;
    let fleet = BuiltinPack::FleetMemory.compile().expect("fleet pack");
    let fleet_relations = akh_medu::schema::parse_source(BuiltinPack::FleetMemory.source())
        .unwrap()
        .relations;

    let options = akhomed_store::PoolOptions::new(format!("sqlite://{db_path}?mode=rw"));
    let db = akhomed_store::connect(&options).await.expect("open copy");

    for name in akhs.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        let dir = tempfile::tempdir().unwrap();
        let wake_started = std::time::Instant::now();
        let engine: Arc<Engine> = akh_medu::snapshot::wake(
            dir.path(),
            akh_medu::wal::akh_id_for(name),
            akh_medu::wal::workspace_id_for(name),
            &db,
        )
        .await
        .unwrap_or_else(|e| panic!("wake {name}: {e}"));
        println!("\n=== {name} === (wake {:?})", wake_started.elapsed());

        let run = {
            let engine = Arc::clone(&engine);
            let roman = roman.clone();
            let roman_relations = roman_relations.clone();
            let fleet_relations = fleet_relations.clone();
            let fleet_id = fleet.id();
            let name = name.to_string();
            tokio::task::spawn_blocking(move || {
                let roman_converged = engine.lookup_symbol(&roman.microtheory_label()).is_ok();
                if !roman_converged {
                    install_pack(&engine, &roman).expect("install roman-memory");
                }
                println!(
                    "roman-memory: {}",
                    if roman_converged {
                        "converged from the copy's registry at wake"
                    } else {
                        "not active in the copy; installed from deploy/"
                    }
                );
                let facts_before = engine.all_triples().len();

                // 1. roman-memory alone, to its fixed point.
                let totals = engine.fact_totals();
                let (roman_passes, roman_merged) = converge(&engine);
                let roman_delta = engine.fact_totals().since(totals);
                println!(
                    "roman-only baseline: {}; declared-alias merge moved {roman_merged}; asserted {} retracted {}",
                    summarize(&roman_passes),
                    roman_delta.asserted,
                    roman_delta.retracted
                );

                // 2. Record roman-memory's relations.
                let roman_before: Vec<(String, (usize, usize, usize), PredicateCardinality)> =
                    roman_relations
                        .iter()
                        .map(|r| {
                            (
                                r.name.clone(),
                                stats(&engine, &r.name),
                                cardinality(&engine, &r.name),
                            )
                        })
                        .collect();
                let fleet_only: Vec<&str> = fleet_relations
                    .iter()
                    .map(|r| r.name.as_str())
                    .filter(|n| !roman_relations.iter().any(|r| r.name == *n))
                    .collect();
                let fleet_only_before: Vec<(&str, (usize, usize, usize))> = fleet_only
                    .iter()
                    .map(|n| (*n, stats(&engine, n)))
                    .collect();

                // 3. The fleet pack.
                let outcome = ensure_builtin_pack(
                    &engine,
                    BuiltinPack::FleetMemory,
                    &name,
                    DefaultPackPolicy::Enabled,
                )
                .expect("ensure fleet pack");
                println!("fleet pack: {outcome:?}");
                assert!(matches!(outcome, BuiltinPackOutcome::Installed { .. }));

                // 4. Conflicts with roman-memory: relations both declare, and
                //    any effective declaration that is not roman-memory's.
                let overlap: Vec<&str> = fleet_relations
                    .iter()
                    .map(|r| r.name.as_str())
                    .filter(|n| roman_relations.iter().any(|r| r.name == *n))
                    .collect();
                let mut conflicts = Vec::new();
                for r in &roman_relations {
                    let expected = PredicateCardinality::SingleValued {
                        temporal: r.temporal,
                    };
                    if cardinality(&engine, &r.name) != expected {
                        conflicts.push(format!("cardinality of {}", r.name));
                    }
                    for alias in &r.aliases {
                        if alias_target(&engine, alias).as_deref() != Some(r.name.as_str()) {
                            conflicts.push(format!("alias {alias}"));
                        }
                    }
                }
                println!(
                    "overlapping relations ({}): {overlap:?}; declaration conflicts with roman-memory: {} {conflicts:?}",
                    overlap.len(),
                    conflicts.len()
                );

                // 5. Hygiene with both packs, to a fixed point.
                let totals = engine.fact_totals();
                let (fleet_passes, fleet_merged) = converge(&engine);
                let fleet_delta = engine.fact_totals().since(totals);
                let idempotent = run_hygiene(
                    &engine,
                    HygieneOptions::for_engine(&engine, akh_medu::schema::DEFAULT_HYGIENE_BATCH),
                )
                .expect("idempotency pass");
                println!(
                    "with {fleet_id}: {}; declared-alias merge moved {fleet_merged}; asserted {} retracted {}; idempotency pass mutations {} noop {}",
                    summarize(&fleet_passes),
                    fleet_delta.asserted,
                    fleet_delta.retracted,
                    idempotent.mutations,
                    idempotent.is_noop()
                );
                for (label, before) in &fleet_only_before {
                    println!(
                        "[fleet-only] {label}: before (live, subjects, multi) {before:?} after {:?}",
                        stats(&engine, label)
                    );
                }
                let mut roman_changed = Vec::new();
                for (label, before, card) in &roman_before {
                    let after = stats(&engine, label);
                    if after != *before || cardinality(&engine, label) != *card {
                        roman_changed.push(format!("{label}: {before:?} -> {after:?}"));
                    }
                }
                println!(
                    "roman-memory relations changed by the fleet pack: {} {roman_changed:?}",
                    roman_changed.len()
                );
                println!(
                    "facts: before {facts_before} after {}",
                    engine.all_triples().len()
                );
                (conflicts, roman_changed, idempotent.is_noop())
            })
            .await
            .unwrap()
        };
        let (conflicts, roman_changed, noop) = run;
        assert!(conflicts.is_empty(), "{name}: {conflicts:?}");
        assert!(roman_changed.is_empty(), "{name}: {roman_changed:?}");
        assert!(noop, "{name}: the last pass must be a no-op");
    }
}
