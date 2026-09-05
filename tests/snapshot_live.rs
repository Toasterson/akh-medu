//! Live-Postgres test for the Phase 38c snapshot writer.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5432/akhmedu?sslmode=disable \
//!     cargo test --features server --test snapshot_live -- --ignored
#![cfg(feature = "postgres")]

use std::collections::BTreeSet;
use std::sync::Arc;

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::provenance::{
    DerivationKind, ProvenanceId, ProvenanceRecord, RULE_APPLICATION_V2_SENTINEL,
    RULE_SUPPORT_TRIPLES_V1_SENTINEL,
};
use akh_medu::snapshot::{
    capture_snapshot_at_fence, capture_snapshot_at_watermark, persist_snapshot, wake,
    write_snapshot,
};
use akh_medu::symbol::SymbolKind;
use akhomed_store::sea_orm::{self, ConnectionTrait, DbBackend, Statement, TransactionTrait};
use akhomed_store::uuid::Uuid;
use akhomed_store::{MutationRow, append_mutations, connect, ensure_akh, run_migrations};

async fn count(db: &sea_orm::DatabaseConnection, sql: &str) -> i64 {
    let row = db
        .query_one(Statement::from_string(DbBackend::Postgres, sql.to_owned()))
        .await
        .unwrap()
        .unwrap();
    row.try_get::<i64>("", "c").unwrap()
}

async fn exec(db: &sea_orm::DatabaseConnection, sql: &str) {
    db.execute(Statement::from_string(DbBackend::Postgres, sql.to_owned()))
        .await
        .unwrap();
}

async fn waiting_advisory_locks(db: &sea_orm::DatabaseConnection) -> i64 {
    let row = db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT count(*)::bigint AS c FROM pg_locks \
             WHERE locktype = 'advisory' AND NOT granted",
        ))
        .await
        .unwrap()
        .unwrap();
    row.try_get::<i64>("", "c").unwrap()
}

async fn wait_for_advisory_waiters(db: &sea_orm::DatabaseConnection, minimum: i64) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if waiting_advisory_locks(db).await >= minimum {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("snapshot publisher did not reach the controlled advisory-lock barrier");
}

/// Sorted `(subject, predicate, object)` set of an engine's live triples.
fn triple_set(engine: &Engine) -> BTreeSet<(u64, u64, u64)> {
    engine
        .all_triples()
        .iter()
        .map(|t| (t.subject.get(), t.predicate.get(), t.object.get()))
        .collect()
}

/// Sorted label set of an engine's registry.
fn label_set(engine: &Engine) -> BTreeSet<String> {
    engine.dump_symbols().into_iter().map(|m| m.label).collect()
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn snapshot_populates_relational_tables() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();

    // Build an engine with a few symbols + triples.
    let tmp = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(tmp.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");

    // A WAL row that must survive the snapshot (38c does not truncate).
    ensure_akh(&db, akh_id, "snap-test", None).await.unwrap();
    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[MutationRow {
            kind: "assert".into(),
            payload: serde_json::json!({ "subject": sun.id.get() }),
        }],
    )
    .await
    .unwrap();

    let expected_triples = engine.all_triples().len() as i64;
    let expected_symbols = engine.dump_symbols().len() as i64;

    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    assert_eq!(
        count(&db, &format!("SELECT count(*)::bigint AS c FROM akhomed_facts WHERE workspace_id = '{workspace_id}'")).await,
        expected_triples,
        "facts rows must equal live KG triple count",
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT count(*)::bigint AS c FROM akhomed_symbols WHERE akh_id = '{akh_id}'")
        )
        .await,
        expected_symbols,
        "symbols rows must equal registry size",
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT count(*)::bigint AS c FROM akhomed_drives WHERE akh_id = '{akh_id}'")
        )
        .await,
        4,
        "four drive rows",
    );
    assert_eq!(
        count(
            &db,
            &format!(
                "SELECT count(*)::bigint AS c FROM akhomed_akh_state WHERE akh_id = '{akh_id}'"
            )
        )
        .await,
        1,
        "one akh_state row",
    );
    // WAL not truncated by the snapshot.
    assert_eq!(
        count(
            &db,
            &format!(
                "SELECT count(*)::bigint AS c FROM akhomed_mutation_log WHERE akh_id = '{akh_id}'"
            )
        )
        .await,
        1,
        "snapshot must not truncate the WAL",
    );
}

/// End-to-end relocation: snapshot on a source host, append a WAL tail newer
/// than the watermark, then `wake` into a fresh data dir against the same
/// Postgres. The woken engine must have identical symbol labels, the snapshot's
/// triples with the tail applied (one assert + one retract), and the restored
/// `cycle_count`.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn wake_reconstructs_identical_akh() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();

    // --- Source host ---
    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    let planet = engine.create_symbol(SymbolKind::Entity, "Planet").unwrap();
    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
    ensure_akh(&db, akh_id, "wake-test", None).await.unwrap();
    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    let src_labels = label_set(&engine);

    // Make cycle_count restoration observable (no DB default; set directly).
    exec(
        &db,
        &format!("UPDATE akhomed_akh_state SET cycle_count = 42 WHERE akh_id = '{akh_id}'"),
    )
    .await;

    // --- WAL tail (seq > watermark): assert (Sun is-a Planet), retract (Sun is-a Star) ---
    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[
            MutationRow {
                kind: "assert".into(),
                payload: serde_json::json!({
                    "subject": sun.id.get(),
                    "predicate": is_a.id.get(),
                    "object": planet.id.get(),
                    "confidence": 1.0,
                    "compartment": null,
                }),
            },
            MutationRow {
                kind: "retract".into(),
                payload: serde_json::json!({
                    "subject": sun.id.get(),
                    "predicate": is_a.id.get(),
                    "object": star.id.get(),
                }),
            },
        ],
    )
    .await
    .unwrap();

    // --- Wake on a fresh host/dir ---
    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .expect("wake");

    assert_eq!(
        label_set(&woken),
        src_labels,
        "symbol labels must survive relocation",
    );

    let triples = triple_set(&woken);
    assert!(
        triples.contains(&(sun.id.get(), is_a.id.get(), planet.id.get())),
        "asserted tail triple must be replayed: {triples:?}",
    );
    assert!(
        !triples.contains(&(sun.id.get(), is_a.id.get(), star.id.get())),
        "retracted tail triple must be absent: {triples:?}",
    );

    // cycle_count is restored via the agent:* meta the wake rebuilt.
    let woken_agent = Agent::resume(Arc::clone(&woken), AgentConfig::default()).expect("resume");
    assert_eq!(
        woken_agent.cycle_count(),
        42,
        "cycle_count must be restored"
    );
}

/// Regression for the capture/watermark race: the WAL lower bound is sampled
/// before capture, then a replay-complete assertion lands after capture but
/// before the captured rows are persisted. A cold wake must recover the tail's
/// fresh symbols, full provenance record, edge metadata, and both allocators.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn post_capture_assertion_is_replayed_with_symbols_and_provenance() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    ensure_akh(&db, akh_id, "capture-race-test", None)
        .await
        .unwrap();

    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let base_subject = engine.resolve_or_create_entity("SnapshotBase").unwrap();
    let base_predicate = engine.resolve_or_create_relation("contains").unwrap();
    let base_object = engine.resolve_or_create_entity("BaseFact").unwrap();
    engine
        .add_triple(&Triple::new(base_subject, base_predicate, base_object))
        .unwrap();
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();

    let watermark = akhomed_store::snapshots::current_watermark(&db, akh_id)
        .await
        .unwrap();
    let captured = capture_snapshot_at_watermark(&agent, akh_id, workspace_id, watermark).unwrap();

    // This entire fact family is newer than the captured in-memory image.
    let tail_subject = engine.resolve_or_create_entity("TailOnlySubject").unwrap();
    let tail_predicate = engine
        .resolve_or_create_relation("tail-only-predicate")
        .unwrap();
    let tail_object = engine.resolve_or_create_entity("TailOnlyObject").unwrap();
    let tail_symbols = [tail_subject, tail_predicate, tail_object]
        .into_iter()
        .map(|id| engine.get_symbol_meta(id).unwrap())
        .collect::<Vec<_>>();
    let mut provenance = ProvenanceRecord::new(tail_object, DerivationKind::Extracted)
        .with_sources(vec![tail_subject])
        .with_confidence(0.77)
        .with_depth(3);
    provenance.timestamp = 1_700_123_456;
    let provenance_id = engine.store_provenance(&mut provenance).unwrap();
    let mut tail_triple = Triple::new(tail_subject, tail_predicate, tail_object)
        .with_confidence(0.77)
        .with_provenance(provenance_id.get());
    tail_triple.timestamp = 1_700_123_456;
    tail_triple.compartment_id = Some("tail-only".into());
    engine.add_triple(&tail_triple).unwrap();

    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[MutationRow {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": akh_medu::mutation::ASSERT_MUTATION_VERSION,
                "subject": tail_subject.get(),
                "predicate": tail_predicate.get(),
                "object": tail_object.get(),
                "confidence": tail_triple.confidence,
                "timestamp": tail_triple.timestamp,
                "compartment": tail_triple.compartment_id,
                "provenance_id": provenance_id.get(),
                "provenance": provenance,
                "symbols": tail_symbols,
            }),
        }],
    )
    .await
    .unwrap();

    // Persisting must retain the pre-capture lower bound, not sample after the
    // assertion above and accidentally exclude it from both image and tail.
    persist_snapshot(captured, &db).await.unwrap();

    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .unwrap();
    let restored_subject = woken.lookup_symbol("TailOnlySubject").unwrap();
    let restored_predicate = woken.lookup_symbol("tail-only-predicate").unwrap();
    let restored_object = woken.lookup_symbol("TailOnlyObject").unwrap();
    assert_eq!(restored_subject, tail_subject);
    assert_eq!(restored_predicate, tail_predicate);
    assert_eq!(restored_object, tail_object);
    let restored = woken
        .triples_from(restored_subject)
        .into_iter()
        .find(|triple| triple.predicate == restored_predicate && triple.object == restored_object)
        .expect("post-capture assertion replayed");
    assert_eq!(restored.confidence, 0.77);
    assert_eq!(restored.timestamp, 1_700_123_456);
    assert_eq!(restored.compartment_id.as_deref(), Some("tail-only"));
    assert_eq!(restored.provenance_id, Some(provenance_id.get()));
    let restored_record = woken.get_provenance(provenance_id).unwrap();
    assert_eq!(
        bincode::serialize(&restored_record).unwrap(),
        bincode::serialize(&provenance).unwrap()
    );
    for expected in &tail_symbols {
        let actual = woken.get_symbol_meta(expected.id).unwrap();
        assert_eq!(actual.kind, expected.kind);
        assert_eq!(actual.label, expected.label);
        assert_eq!(actual.created_at, expected.created_at);
        assert_eq!(actual.source, expected.source);
    }

    let max_tail_symbol = tail_symbols.iter().map(|meta| meta.id.get()).max().unwrap();
    let after_tail = woken
        .create_symbol(SymbolKind::Entity, "AllocatedAfterTail")
        .unwrap();
    assert!(after_tail.id.get() > max_tail_symbol);
    let mut after_record = ProvenanceRecord::new(after_tail.id, DerivationKind::Reasoned);
    assert!(woken.store_provenance(&mut after_record).unwrap().get() > provenance_id.get());
}

#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn wake_preserves_provenance_links_and_allocator_identity() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    engine
        .ingest_label_triples(&[
            ("Dog".into(), "is-a".into(), "Mammal".into(), 1.0),
            ("Mammal".into(), "is-a".into(), "Animal".into(), 1.0),
        ])
        .unwrap();
    engine
        .run_rules(akh_medu::autonomous::RuleEngineConfig::default())
        .unwrap();
    let dog = engine.lookup_symbol("Dog").unwrap();
    let is_a = engine.lookup_symbol("is-a").unwrap();
    let animal = engine.lookup_symbol("Animal").unwrap();
    let source_triple = engine
        .triples_from(dog)
        .into_iter()
        .find(|triple| triple.predicate == is_a && triple.object == animal)
        .expect("source inferred edge");
    let source_provenance_id = source_triple.provenance_id.expect("linked source edge");
    let source_max_id = engine
        .all_provenance()
        .unwrap()
        .into_iter()
        .filter_map(|record| record.id.map(ProvenanceId::get))
        .max()
        .unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
    ensure_akh(&db, akh_id, "provenance-wake-test", None)
        .await
        .unwrap();
    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .expect("wake");
    let restored = woken
        .triples_from(dog)
        .into_iter()
        .find(|triple| triple.predicate == is_a && triple.object == animal)
        .expect("restored inferred edge");
    assert_eq!(restored.provenance_id, Some(source_provenance_id));
    let restored_record = woken
        .get_provenance(ProvenanceId::new(source_provenance_id).unwrap())
        .expect("restored provenance record");
    match restored_record.kind {
        DerivationKind::RuleInference { antecedents, .. } => {
            assert_eq!(antecedents[0].get(), RULE_SUPPORT_TRIPLES_V1_SENTINEL);
            assert_eq!(antecedents[1].get(), RULE_APPLICATION_V2_SENTINEL);
            assert_eq!(
                antecedents.len(),
                11,
                "one exact conclusion and two support triples"
            );
        }
        other => panic!("unexpected restored derivation: {other:?}"),
    }

    woken
        .ingest_label_triples(&[(
            "FreshAssertion".into(),
            "states".into(),
            "FreshClaim".into(),
            0.9,
        )])
        .unwrap();
    let fresh = woken
        .triples_from(woken.lookup_symbol("FreshAssertion").unwrap())
        .into_iter()
        .find(|triple| triple.predicate == woken.lookup_symbol("states").unwrap())
        .unwrap();
    assert!(
        fresh.provenance_id.unwrap() > source_max_id,
        "new provenance must allocate beyond the restored ledger watermark"
    );
    assert!(matches!(
        woken
            .get_provenance(ProvenanceId::new(source_provenance_id).unwrap())
            .unwrap()
            .kind,
        DerivationKind::RuleInference { .. }
    ));
}

/// Lazy-open hydration (Anima WI 019fe636-94e1): a k8s tmpfs pod boots with an
/// empty `kg_dir` on every restart. `akhomed`'s `get_engine` must hydrate from
/// Postgres before constructing the engine a plain open would otherwise hand
/// back seed-pack-empty. `ServerState::try_hydrate_from_store` lives in the
/// `akhomed` binary crate (not exported from the library), so this test can't
/// call it directly; instead it exercises the same two-step composition that
/// function performs — `wake` into the cold dir, drop that engine handle
/// (redb is single-writer), then reopen via `Engine::new` exactly as
/// `get_engine` does afterwards — and asserts the reopened engine sees the
/// full source KG rather than an empty/seed-pack-only one.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn open_hydrates_cold_workspace_from_store() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();

    // --- Populate + snapshot an akh, as if it had lived on some host before. ---
    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
    let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
    engine
        .add_triple(&Triple::new(sun.id, is_a.id, star.id))
        .unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
    ensure_akh(&db, akh_id, "open-hydrate-test", None)
        .await
        .unwrap();
    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    let src_labels = label_set(&engine);
    let src_triples = triple_set(&engine);

    // --- Cold pod: a fresh dir with no redb (the tmpfs-restart scenario). ---
    let cold = tempfile::tempdir().unwrap();
    let kg_dir = cold.path().join("kg");
    assert!(
        !kg_dir.join("akh-medu.redb").exists(),
        "precondition: cold dir must have no redb yet"
    );

    // Step 1 of try_hydrate_from_store: the store has state for this akh
    // (mirrors its `load_akh_state` gate), so hydrate — `wake` populates the
    // redb in `kg_dir`, then its engine handle must be dropped before anyone
    // reopens the same redb file.
    let hydrating = wake(&kg_dir, akh_id, workspace_id, &db)
        .await
        .expect("hydration wake");
    drop(hydrating);

    // Step 2: what `get_engine` does immediately afterwards — open the now-
    // populated redb as a normal Engine, exactly like a plain lazy open would.
    let reopened = Engine::new(EngineConfig {
        data_dir: Some(kg_dir),
        ..Default::default()
    })
    .expect("reopen hydrated redb");

    assert_eq!(
        label_set(&reopened),
        src_labels,
        "reopened engine must see the source akh's symbols, not a seed-pack-empty one"
    );
    assert_eq!(
        triple_set(&reopened),
        src_triples,
        "reopened engine must see the source akh's triples"
    );
    assert!(
        !reopened.all_triples().is_empty(),
        "hydrated-then-reopened engine must not be empty"
    );
}

/// A missing optional cognitive table must not abort wake — the Akh still comes
/// back with its KG and symbols, just with that subsystem at defaults.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn degraded_wake_succeeds_with_missing_table() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();

    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let a = engine.create_symbol(SymbolKind::Entity, "Maat").unwrap();
    let rel = engine
        .create_symbol(SymbolKind::Relation, "embodies")
        .unwrap();
    let b = engine.create_symbol(SymbolKind::Entity, "Order").unwrap();
    engine.add_triple(&Triple::new(a.id, rel.id, b.id)).unwrap();

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
    ensure_akh(&db, akh_id, "degraded-test", None)
        .await
        .unwrap();
    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    let src_labels = label_set(&engine);
    let src_triples = triple_set(&engine);

    // Simulate a partial/old snapshot: the drives table never got written.
    exec(
        &db,
        &format!("DELETE FROM akhomed_drives WHERE akh_id = '{akh_id}'"),
    )
    .await;

    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .expect("degraded wake must still succeed");

    assert_eq!(
        label_set(&woken),
        src_labels,
        "labels intact despite missing table"
    );
    assert_eq!(
        triple_set(&woken),
        src_triples,
        "KG intact despite missing table"
    );

    // Resume must still work; drives fall back to defaults (4 of them).
    let woken_agent = Agent::resume(Arc::clone(&woken), AgentConfig::default()).expect("resume");
    assert_eq!(
        woken_agent.drives().drives.len(),
        4,
        "drives default-rebuilt"
    );
}

/// Force newer and older captures to queue behind the same publication lock.
/// Once released, the newer image must commit atomically and the stale writer
/// must fail before it can reintroduce the retracted edge.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn overlapping_snapshot_publishers_reject_stale_retraction_image() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    ensure_akh(&db, akh_id, "snapshot-overlap", None)
        .await
        .unwrap();

    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let subject = engine
        .create_symbol(SymbolKind::Entity, "OverlapSubject")
        .unwrap();
    let predicate = engine
        .create_symbol(SymbolKind::Relation, "overlap-predicate")
        .unwrap();
    let object = engine
        .create_symbol(SymbolKind::Entity, "OverlapObject")
        .unwrap();
    engine
        .add_triple(&Triple::new(subject.id, predicate.id, object.id))
        .unwrap();
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();

    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[MutationRow {
            kind: "turn".into(),
            payload: serde_json::json!({"phase": "before-retraction"}),
        }],
    )
    .await
    .unwrap();
    let old_watermark = akhomed_store::snapshots::current_watermark(&db, akh_id)
        .await
        .unwrap();
    let old = capture_snapshot_at_watermark(&agent, akh_id, workspace_id, old_watermark).unwrap();

    engine
        .remove_triple(subject.id, predicate.id, object.id)
        .unwrap();
    append_mutations(
        &db,
        akh_id,
        workspace_id,
        &[MutationRow {
            kind: "retract".into(),
            payload: serde_json::json!({
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
            }),
        }],
    )
    .await
    .unwrap();
    let new_watermark = akhomed_store::snapshots::current_watermark(&db, akh_id)
        .await
        .unwrap();
    assert!(new_watermark > old_watermark);
    let new = capture_snapshot_at_watermark(&agent, akh_id, workspace_id, new_watermark).unwrap();

    let blocker = db.begin().await.unwrap();
    blocker
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            [format!("snapshot:{workspace_id}").into()],
        ))
        .await
        .unwrap();
    let baseline_waiters = waiting_advisory_locks(&db).await;

    let newer_db = db.clone();
    let newer = tokio::spawn(async move { persist_snapshot(new, &newer_db).await });
    wait_for_advisory_waiters(&db, baseline_waiters + 1).await;

    let older_db = db.clone();
    let older = tokio::spawn(async move { persist_snapshot(old, &older_db).await });
    wait_for_advisory_waiters(&db, baseline_waiters + 2).await;
    blocker.commit().await.unwrap();

    newer
        .await
        .unwrap()
        .expect("newer snapshot publishes first");
    let stale = older
        .await
        .unwrap()
        .expect_err("older capture must fail closed after newer publication");
    assert!(stale.to_string().contains("stale snapshot"), "{stale}");

    let state = akhomed_store::snapshots::load_akh_state(&db, akh_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(state.watermark, new_watermark);
    assert!(
        akhomed_store::snapshots::load_workspace_facts(&db, workspace_id)
            .await
            .unwrap()
            .is_empty(),
        "the old edge must not coexist with the newer recovery watermark"
    );

    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .unwrap();
    assert!(!woken.has_triple(subject.id, predicate.id, object.id));
}

/// A WAL watermark cannot distinguish two divergent captures when no mutation
/// row was appended between them. Residency epoch is the cross-process fence:
/// after ownership advances, the old equal-watermark image must fail closed.
#[tokio::test]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn equal_watermark_snapshot_from_previous_residency_is_rejected() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let opts = akhomed_store::PoolOptions::new(url);
    let db = connect(&opts).await.unwrap();
    run_migrations(&db).await.unwrap();
    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    ensure_akh(&db, akh_id, "equal-watermark-fence", None)
        .await
        .unwrap();

    let engine = Arc::new(Engine::new(EngineConfig::default()).unwrap());
    let subject = engine
        .create_symbol(SymbolKind::Entity, "FenceSubject")
        .unwrap();
    let predicate = engine
        .create_symbol(SymbolKind::Relation, "fence-predicate")
        .unwrap();
    let object = engine
        .create_symbol(SymbolKind::Entity, "FenceObject")
        .unwrap();
    engine
        .add_triple(&Triple::new(subject.id, predicate.id, object.id))
        .unwrap();
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();
    let watermark = akhomed_store::snapshots::current_watermark(&db, akh_id)
        .await
        .unwrap();

    let old_session = akhomed_store::StablePgSession::connect(&opts)
        .await
        .unwrap();
    assert!(old_session.try_acquire_akh_lock(akh_id).await.unwrap());
    let old_epoch = old_session
        .claim_akh_residency(akh_id, workspace_id)
        .await
        .unwrap();
    let stale =
        capture_snapshot_at_fence(&agent, akh_id, workspace_id, watermark, old_epoch).unwrap();
    engine
        .remove_triple(subject.id, predicate.id, object.id)
        .unwrap();
    assert!(old_session.release_akh_lock(akh_id).await.unwrap());
    let new_session = akhomed_store::StablePgSession::connect(&opts)
        .await
        .unwrap();
    assert!(new_session.try_acquire_akh_lock(akh_id).await.unwrap());
    let new_epoch = new_session
        .claim_akh_residency(akh_id, workspace_id)
        .await
        .unwrap();
    let current =
        capture_snapshot_at_fence(&agent, akh_id, workspace_id, watermark, new_epoch).unwrap();
    persist_snapshot(current, &db).await.unwrap();
    let error = persist_snapshot(stale, &db)
        .await
        .expect_err("previous owner must be fenced at equal watermark");
    assert!(
        error.to_string().contains("stale snapshot owner"),
        "{error}"
    );
    assert!(
        akhomed_store::snapshots::load_workspace_facts(&db, workspace_id)
            .await
            .unwrap()
            .is_empty()
    );
}
