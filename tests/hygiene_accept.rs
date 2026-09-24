//! Phase 53j acceptance experiment: the memory-hygiene pass over a real
//! akhomed SQLite store, through the real engine (`snapshot::wake`).
//!
//! Ignored by default. Point it at a **private copy** of a store — never the
//! live file:
//!
//! ```sh
//! mkdir -p /tmp/claude-53j
//! sqlite3 "file:$ROOT/state/core/akh.sqlite3?mode=ro" ".backup /tmp/claude-53j/akh.sqlite3"
//! AKH_HYGIENE_ACCEPT_DB=/tmp/claude-53j/akh.sqlite3 \
//!   cargo test --features server --test hygiene_accept -- --ignored --nocapture
//! ```
//!
//! Optional: `AKH_HYGIENE_ACCEPT_AKHS` (comma list, default
//! `janus,minerva,vesta`) and `AKH_HYGIENE_ACCEPT_PACK` (default
//! `deploy/roman-mini/schemas/roman-memory.json`).
//!
//! Per akh it wakes the copy into a temp data dir, installs the schema pack,
//! runs hygiene passes until one is not truncated, merges the declared
//! aliases (the cutover's explicit step), runs one more pass (must be a
//! no-op), and prints before/after numbers. It asserts that every subject of
//! every declared single-valued predicate ends with exactly one live value
//! and that no alias spelling keeps a live triple.
#![cfg(feature = "postgres")]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use akh_medu::engine::Engine;
use akh_medu::schema::{
    HygieneOptions, MergeRequest, MergeSelection, PgSchemaRegistry, RegistryRow, SchemaRegistry,
    declared_alias_target, declared_cardinality, execute_merge, install_pack, run_hygiene,
};
use akh_medu::symbol::SymbolId;

/// Refuse anything that looks like the live Roman state directory.
fn assert_private_copy(path: &str) {
    assert!(
        !path.contains("roman-akhs/state"),
        "AKH_HYGIENE_ACCEPT_DB must be a private copy, not the live store: {path}"
    );
}

#[derive(Debug, Default)]
struct FamilyStats {
    per_spelling: BTreeMap<String, usize>,
    subjects: usize,
    multi_valued_subjects: usize,
    max_per_subject: usize,
}

/// Live triples of `canonical` and every spelling that is a declared alias
/// of it, grouped per `(subject, compartment)`.
fn family_stats(engine: &Engine, canonical: SymbolId, spellings: &[SymbolId]) -> FamilyStats {
    let mut stats = FamilyStats::default();
    let mut per_subject: HashMap<(SymbolId, Option<String>), usize> = HashMap::new();
    for &p in std::iter::once(&canonical).chain(spellings) {
        let triples = engine.triples_with_predicate(p);
        if !triples.is_empty() {
            stats
                .per_spelling
                .insert(engine.resolve_label(p), triples.len());
        }
        for t in triples {
            *per_subject
                .entry((t.subject, t.compartment_id))
                .or_default() += 1;
        }
    }
    stats.subjects = per_subject.len();
    stats.multi_valued_subjects = per_subject.values().filter(|&&n| n > 1).count();
    stats.max_per_subject = per_subject.values().copied().max().unwrap_or(0);
    stats
}

/// Every predicate symbol whose label is `canonical` or one of the pack's
/// declared aliases for it (resolved by label before the pack exists).
fn spellings_of(engine: &Engine, labels: &[String]) -> Vec<SymbolId> {
    labels
        .iter()
        .filter_map(|l| engine.lookup_symbol(l).ok())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs AKH_HYGIENE_ACCEPT_DB pointing at a private copy of an akhomed SQLite store"]
async fn hygiene_pass_on_a_copied_store() {
    let Ok(db_path) = std::env::var("AKH_HYGIENE_ACCEPT_DB") else {
        eprintln!("AKH_HYGIENE_ACCEPT_DB not set; skipping");
        return;
    };
    assert_private_copy(&db_path);
    let akhs =
        std::env::var("AKH_HYGIENE_ACCEPT_AKHS").unwrap_or_else(|_| "janus,minerva,vesta".into());
    let pack_path = std::env::var("AKH_HYGIENE_ACCEPT_PACK").unwrap_or_else(|_| {
        format!(
            "{}/deploy/roman-mini/schemas/roman-memory.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let source = std::fs::read_to_string(&pack_path).expect("read schema pack");
    let pack = akh_medu::schema::compile(&source).expect("compile schema pack");
    let parsed = akh_medu::schema::parse_source(&source).unwrap();
    let families: Vec<(String, Vec<String>)> = parsed
        .relations
        .iter()
        .map(|r| (r.name.clone(), r.aliases.clone()))
        .collect();

    let options = akhomed_store::PoolOptions::new(format!("sqlite://{db_path}?mode=rw"));
    let db = akhomed_store::connect(&options).await.expect("open copy");

    let mut grand_retracted = 0u64;
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
        let wake_elapsed = wake_started.elapsed();

        let facts_before = engine.all_triples().len();
        let before: Vec<(String, FamilyStats)> = families
            .iter()
            .map(|(canonical, aliases)| {
                let mut labels = vec![canonical.clone()];
                labels.extend(aliases.iter().cloned());
                let ids = spellings_of(&engine, &labels);
                let stats = match ids.split_first() {
                    Some((&first, rest)) => family_stats(&engine, first, rest),
                    None => FamilyStats::default(),
                };
                (canonical.clone(), stats)
            })
            .collect();

        let run = {
            let engine = Arc::clone(&engine);
            let pack = pack.clone();
            let db = db.clone();
            let name = name.to_string();
            tokio::task::spawn_blocking(move || {
                let totals_before = engine.fact_totals();
                let started = std::time::Instant::now();
                // The cutover path: admit once to the store's registry
                // (`akh schema admit --yes`), fetch + decode + install +
                // record the activation (`akh -w <akh> schema install`).
                let registry = PgSchemaRegistry::new(&db);
                if registry.fetch(&pack.name, &pack.version).unwrap().is_none() {
                    let blob = pack.to_blob().unwrap();
                    registry
                        .admit(RegistryRow {
                            name: pack.name.clone(),
                            version: pack.version.clone(),
                            scope: pack.scope,
                            source_url: pack.source_url.clone(),
                            checksum: akh_medu::schema::compile::checksum_hex(&blob),
                            confirmed_by: "hygiene-accept".into(),
                            blob,
                        })
                        .expect("admit pack to the copy's registry");
                }
                let fetched = registry
                    .fetch(&pack.name, &pack.version)
                    .unwrap()
                    .expect("admitted pack")
                    .decode()
                    .expect("decode registry blob");
                let install = install_pack(&engine, &fetched).expect("install pack");
                akh_medu::schema::registry_pg::record_activation(
                    &db,
                    akh_medu::wal::workspace_id_for(&name),
                    &fetched.microtheory_label(),
                    akh_medu::wal::akh_id_for(&name),
                )
                .expect("record activation");
                let mut passes = Vec::new();
                loop {
                    let report = run_hygiene(
                        &engine,
                        HygieneOptions::for_engine(
                            &engine,
                            akh_medu::schema::DEFAULT_HYGIENE_BATCH,
                        ),
                    )
                    .expect("hygiene pass");
                    let truncated = report.truncated;
                    passes.push(report);
                    if !truncated || passes.len() >= 50 {
                        break;
                    }
                }
                let declared = execute_merge(
                    &engine,
                    &MergeRequest {
                        selection: MergeSelection::Declared,
                        dry_run: false,
                        limit: akh_medu::schema::DEFAULT_MERGE_LIMIT,
                    },
                )
                .expect("declared merge");
                let after_merge = run_hygiene(
                    &engine,
                    HygieneOptions::for_engine(&engine, akh_medu::schema::DEFAULT_HYGIENE_BATCH),
                )
                .expect("post-merge pass");
                let idempotent = run_hygiene(
                    &engine,
                    HygieneOptions::for_engine(&engine, akh_medu::schema::DEFAULT_HYGIENE_BATCH),
                )
                .expect("idempotency pass");
                let totals = engine.fact_totals().since(totals_before);
                println!(
                    "hygiene wall time (install + passes + merge + 2 checks): {:?}",
                    started.elapsed()
                );
                (install, passes, declared, after_merge, idempotent, totals)
            })
            .await
            .unwrap()
        };
        let (install, passes, declared, after_merge, idempotent, totals) = run;
        grand_retracted += totals.retracted;

        let provenance = engine
            .provenance_by_kind_tags(&[akh_medu::provenance::DerivationKind::tag_for_name(
                "memory_hygiene",
            )
            .unwrap()])
            .unwrap();
        let mut by_action: BTreeMap<&'static str, usize> = BTreeMap::new();
        for record in &provenance {
            if let akh_medu::provenance::DerivationKind::MemoryHygiene { action, .. } = &record.kind
            {
                *by_action.entry(action.as_str()).or_default() += 1;
            }
        }

        println!("\n=== {name} === (wake {wake_elapsed:?})");
        println!(
            "facts before: {facts_before}  after: {}  (pack triples applied: {})",
            engine.all_triples().len(),
            install.triples_applied
        );
        for (i, pass) in passes.iter().enumerate() {
            println!(
                "pass {}: merged {} families / {} triples (e.g. {:?}) superseded {} repaired {} mutations {} truncated {} provenance {}",
                i + 1,
                pass.merged.len(),
                pass.merged_triples(),
                pass.merged
                    .iter()
                    .take(4)
                    .map(|m| format!("{}<-{:?}:{}", m.canonical, m.aliases, m.moved))
                    .collect::<Vec<_>>(),
                pass.superseded,
                pass.repaired,
                pass.mutations,
                pass.truncated,
                pass.provenance_records,
            );
        }
        if let Some(first) = passes.first() {
            println!(
                "first pass reports: ambiguous families {}, pending declared aliases {:?}",
                first.ambiguous_families.len(),
                first
                    .pending_declared_aliases
                    .iter()
                    .map(|a| format!("{}->{}:{}", a.alias, a.canonical, a.live_triples))
                    .collect::<Vec<_>>()
            );
        }
        println!(
            "declared-alias merge: moved {} {:?}",
            declared.moved,
            declared
                .merged
                .iter()
                .map(|m| format!("{}<-{:?}:{}", m.canonical, m.aliases, m.moved))
                .collect::<Vec<_>>()
        );
        println!(
            "post-merge pass: superseded {} mutations {}; idempotency pass: mutations {} noop {}",
            after_merge.superseded,
            after_merge.mutations,
            idempotent.mutations,
            idempotent.is_noop()
        );
        println!(
            "fact delta: asserted {} retracted {}; memory_hygiene provenance {} {:?}",
            totals.asserted,
            totals.retracted,
            provenance.len(),
            by_action
        );
        println!(
            "unresolved (undeclared multi-value) top: {:?}",
            idempotent
                .unresolved
                .iter()
                .take(8)
                .map(|u| format!(
                    "{}:{}x(max {})",
                    u.predicate, u.subjects_with_multiple_values, u.max_values
                ))
                .collect::<Vec<_>>()
        );

        assert!(idempotent.is_noop(), "{name}: second pass must be a no-op");
        for (canonical, stats_before) in &before {
            let Ok(canonical_id) = engine.lookup_symbol(canonical) else {
                continue;
            };
            assert!(declared_cardinality(&engine, canonical_id).is_single_valued());
            let alias_ids: Vec<SymbolId> = families
                .iter()
                .find(|(c, _)| c == canonical)
                .map(|(_, aliases)| spellings_of(&engine, aliases))
                .unwrap_or_default();
            for alias in &alias_ids {
                assert_eq!(declared_alias_target(&engine, *alias), Some(canonical_id));
            }
            let after = family_stats(&engine, canonical_id, &alias_ids);
            println!(
                "[family] {canonical}: before {:?} subjects {} multi {} max {} | after {:?} subjects {} multi {} max {}",
                stats_before.per_spelling,
                stats_before.subjects,
                stats_before.multi_valued_subjects,
                stats_before.max_per_subject,
                after.per_spelling,
                after.subjects,
                after.multi_valued_subjects,
                after.max_per_subject,
            );
            assert_eq!(
                after.subjects, stats_before.subjects,
                "{name}/{canonical}: no subject may lose its current value"
            );
            assert_eq!(
                after.multi_valued_subjects, 0,
                "{name}/{canonical}: exactly one live value per subject"
            );
            assert!(
                after.per_spelling.keys().all(|k| k == canonical),
                "{name}/{canonical}: alias spellings must be empty: {:?}",
                after.per_spelling
            );
        }
    }
    println!("\ntotal retracted across akhs: {grand_retracted}");
}
