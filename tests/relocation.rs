//! Phase 38g — randomized crash-recovery / relocation property test.
//!
//! Complements `snapshot_live::wake_reconstructs_identical_akh` (a fixed
//! scenario) and `akhomed-store::tests::live_pg` (the advisory-lock split-brain
//! guard). Here we drive a *random* sequence of asserts/retracts into the WAL
//! after a snapshot, then wake on a fresh data dir against the same Postgres and
//! assert the recovered triple set exactly equals snapshot ⊕ WAL-tail — i.e.
//! recovery loses nothing that was committed to the WAL, across many seeds.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5432/akhmedu?sslmode=disable \
//!     cargo test --features server --test relocation -- --ignored
#![cfg(feature = "postgres")]

use std::collections::BTreeSet;
use std::sync::Arc;

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::snapshot::{wake, write_snapshot};
use akh_medu::symbol::SymbolKind;
use akhomed_store::uuid::Uuid;
use akhomed_store::{MutationRow, append_mutations, connect, ensure_akh, run_migrations};
use rand::{Rng, SeedableRng, rngs::StdRng};

fn triple_set(engine: &Engine) -> BTreeSet<(u64, u64, u64)> {
    engine
        .all_triples()
        .iter()
        .map(|t| (t.subject.get(), t.predicate.get(), t.object.get()))
        .collect()
}

/// For one seed: snapshot a small KG, push a random assert/retract WAL tail,
/// wake on a fresh dir, and assert the recovered set == the model's expectation.
async fn recover_for_seed(url: &str, seed: u64) {
    let db = connect(&akhomed_store::PoolOptions::new(url.to_owned()))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    let mut rng = StdRng::seed_from_u64(seed);

    // --- Source host: symbols + an initial (pre-snapshot) triple set ---
    let src = tempfile::tempdir().unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(src.path().join("kg")),
            ..Default::default()
        })
        .unwrap(),
    );
    let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap().id;
    let ents: Vec<_> = (0..6)
        .map(|i| engine.create_symbol(SymbolKind::Entity, &format!("E{i}")).unwrap().id)
        .collect();

    // Seed a small pre-snapshot triple set among the test entities.
    for _ in 0..8 {
        let s = ents[rng.gen_range(0..ents.len())];
        let o = ents[rng.gen_range(0..ents.len())];
        if s == o {
            continue;
        }
        engine.add_triple(&Triple::new(s, is_a, o)).unwrap();
    }

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");

    // `Agent::new` bootstraps the well-known `agent:` predicates *and* the causal
    // world-model schemas derived from the tool registry into the engine, so the
    // snapshot captures the engine's *entire* triple set — not just the asserts
    // above. The recovery expectation is therefore (full snapshot baseline) ⊕
    // (WAL tail); seed `model` from the live engine here rather than tracking
    // individual asserts, then let the random tail below mutate it.
    let mut model: BTreeSet<(u64, u64, u64)> = triple_set(&engine);

    ensure_akh(&db, akh_id, "reloc-test", None).await.unwrap();
    write_snapshot(&agent, akh_id, workspace_id, &db)
        .await
        .expect("snapshot");

    // --- Random WAL tail (seq > watermark): asserts and retracts ---
    let mut tail: Vec<MutationRow> = Vec::new();
    for _ in 0..40 {
        let s = ents[rng.gen_range(0..ents.len())];
        let o = ents[rng.gen_range(0..ents.len())];
        if s == o {
            continue;
        }
        let key = (s.get(), is_a.get(), o.get());
        if rng.gen_bool(0.6) {
            tail.push(MutationRow {
                kind: "assert".into(),
                payload: serde_json::json!({
                    "subject": s.get(), "predicate": is_a.get(), "object": o.get(),
                    "confidence": 1.0, "compartment": null,
                }),
            });
            model.insert(key);
        } else {
            tail.push(MutationRow {
                kind: "retract".into(),
                payload: serde_json::json!({
                    "subject": s.get(), "predicate": is_a.get(), "object": o.get(),
                }),
            });
            model.remove(&key);
        }
    }
    append_mutations(&db, akh_id, workspace_id, &tail).await.unwrap();

    // --- "Kill" the source (drop the engine) and wake on a fresh dir ---
    drop(agent);
    drop(engine);
    let dst = tempfile::tempdir().unwrap();
    let woken = wake(&dst.path().join("kg"), akh_id, workspace_id, &db)
        .await
        .expect("wake");

    assert_eq!(
        triple_set(&woken),
        model,
        "seed {seed}: recovered set must equal snapshot ⊕ WAL tail (bounded dataloss = 0 here)",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn random_assert_retract_tail_recovers_exactly() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    // A handful of seeds = a lightweight property test over random workloads.
    for seed in [1u64, 7, 42, 1234, 99999] {
        recover_for_seed(&url, seed).await;
    }
}
