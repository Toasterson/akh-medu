//! Live-Postgres test for the ADR 091 sleep-cycle harvest phase (increment 2):
//! `kind='turn'` WAL rows → `episode:turn:*` KG symbols, watermarked.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test --features postgres --test harvest_live -- --ignored
#![cfg(feature = "postgres")]

use akh_medu::agent::harvest::harvest_daily_log;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akhomed_store::uuid::Uuid;
use akhomed_store::{
    MutationRow, append_mutations, connect, ensure_akh, get_harvest_watermark, run_migrations,
};
use std::sync::Arc;

fn turn_payload(seq_hint: i64, user_prompt: &str, agent_answer: &str) -> serde_json::Value {
    serde_json::json!({
        "v": 1,
        "ts": format!("2026-07-{:02}T00:00:00.000000000Z", 10 + seq_hint),
        "workspace": "harvest-live",
        "model": "test-model",
        "user_prompt": user_prompt,
        "agent_answer": agent_answer,
        "tools": [],
    })
}

/// Appends 3 valid turn rows, harvests them into episodes, asserts the
/// watermark and episode count, then harvests again and asserts idempotence
/// (zero new episodes, watermark unchanged).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn harvest_mints_episodes_and_is_idempotent() {
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
    ensure_akh(&db, akh_id, "harvest-live", None).await.unwrap();

    // Watermark starts at 0 for a fresh akh.
    assert_eq!(get_harvest_watermark(&db, akh_id).await.unwrap(), 0);

    let rows = vec![
        MutationRow {
            kind: "turn".to_string(),
            payload: turn_payload(1, "What is the capital of France?", "Paris."),
        },
        MutationRow {
            kind: "turn".to_string(),
            payload: turn_payload(2, "Who wrote Hamlet?", "William Shakespeare wrote Hamlet."),
        },
        MutationRow {
            kind: "turn".to_string(),
            payload: turn_payload(3, "What is 2+2?", "2+2 equals 4."),
        },
    ];
    let inserted = append_mutations(&db, akh_id, workspace_id, &rows)
        .await
        .unwrap();
    assert_eq!(inserted, 3);

    let engine = Arc::new(Engine::new(EngineConfig::default()).unwrap());
    // `Agent::new` bootstraps the well-known `agent:` predicates (init is
    // crate-private; the agent constructor is the public path to them).
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();
    let predicates = agent.predicates();

    // First harvest: mints 3 episodes, advances the watermark.
    let metrics =
        tokio::task::block_in_place(|| harvest_daily_log(&db, &engine, predicates, akh_id))
            .unwrap();
    assert_eq!(metrics.turns_harvested, 3);
    assert_eq!(metrics.episodes_minted, 3);
    assert!(metrics.last_seq > 0);

    let watermark = get_harvest_watermark(&db, akh_id).await.unwrap();
    assert_eq!(watermark, metrics.last_seq);

    // Every minted episode symbol exists and resolves.
    let episode_syms: Vec<_> = engine
        .all_symbols()
        .into_iter()
        .filter(|meta| engine.resolve_label(meta.id).starts_with("episode:turn:"))
        .collect();
    assert_eq!(
        episode_syms.len(),
        3,
        "exactly 3 episode:turn:* symbols minted"
    );

    // Second harvest: nothing new since the watermark — idempotent.
    let metrics2 =
        tokio::task::block_in_place(|| harvest_daily_log(&db, &engine, predicates, akh_id))
            .unwrap();
    assert_eq!(
        metrics2.turns_harvested, 0,
        "re-harvest finds nothing past the watermark"
    );
    assert_eq!(metrics2.episodes_minted, 0);
    assert_eq!(
        metrics2.last_seq, watermark,
        "watermark unchanged on a no-op harvest"
    );

    let episode_syms_after: Vec<_> = engine
        .all_symbols()
        .into_iter()
        .filter(|meta| engine.resolve_label(meta.id).starts_with("episode:turn:"))
        .collect();
    assert_eq!(
        episode_syms_after.len(),
        3,
        "no duplicate episodes minted on re-harvest"
    );
}
