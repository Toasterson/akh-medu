//! ADR 091 acceptance test (Wave 3): the full transcript → episodic capture
//! loop, end to end, with **Postgres as the only carrier between phases**.
//!
//! Each phase below gets a completely fresh `Engine` instance — nothing
//! in-memory survives between them. That is the restart discriminator ADR 091
//! decision 8 calls for: "the restart is the discriminator that separates the
//! real durable loop from the ephemeral within-session 4/6." This test's
//! restart boundary is the WAL/Postgres hop (fresh `Engine` per phase, same
//! `akh_id`, same database). Full KG-durability-across-a-literal-process-restart
//! (snapshot/wake) is already covered by the live snapshot/wake tests
//! (Phase 38 `ADR 049`); re-proving that here would just be a slower copy of
//! those tests. What this test proves that they don't: a fact captured in one
//! session's WAL turn, harvested into the KG by a *different* engine instance,
//! is recallable by a *third* engine instance that never saw the original
//! conversation — the loop ADR 091 exists to close.
//!
//! Compiled only with the `postgres` feature; ignored unless AKH_TEST_PG_URL is
//! set. Run against a throwaway database:
//!   AKH_TEST_PG_URL=postgres://user@127.0.0.1:5439/akhmedu?sslmode=disable \
//!     cargo test --features postgres --test adr091_acceptance -- --ignored
#![cfg(feature = "postgres")]

use std::sync::Arc;
use std::time::Duration;

use akh_medu::agent::harvest::harvest_daily_log;
use akh_medu::agent::memory::recall_episodes;
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::mutation::{MutationSink, TurnBodyReader};
use akh_medu::nlu::kg_context::extract_keywords;
use akh_medu::wal::{MutationLog, WalConfig};
use akhomed_store::uuid::Uuid;
use akhomed_store::{connect, ensure_akh, get_harvest_watermark, load_turns_since, run_migrations};

/// The fact stated only in "session A" — never mentioned anywhere else in
/// this test. Its presence in a recalled episode's summary is the acceptance
/// signal: it proves the body was paged out of the WAL row session A wrote,
/// by an engine that only ever saw session B's question.
const THE_FACT: &str = "zebra-blue-42";

fn turn_payload(user_prompt: &str, agent_answer: &str) -> serde_json::Value {
    serde_json::json!({
        "v": 1,
        "ts": "2026-07-10T00:00:00.000000000Z",
        "workspace": "adr091-acceptance",
        "model": "test-model",
        "user_prompt": user_prompt,
        "agent_answer": agent_answer,
        "tools": [],
    })
}

/// A `TurnBodyReader` reading directly from Postgres via `load_turn_by_seq` —
/// the test-side equivalent of `src/bin/akhomed.rs`'s `PgTurnBodyReader`
/// (a binary item, not importable from an integration test). Bridges the
/// synchronous `Engine::turn_body` seam to the async store call with
/// `block_in_place` + a captured tokio handle, same pattern as the production
/// reader and as `harvest_daily_log`'s own sync/async boundary.
struct TestTurnBodyReader {
    db: akhomed_store::DatabaseConnection,
    akh_id: Uuid,
    handle: tokio::runtime::Handle,
}

impl TurnBodyReader for TestTurnBodyReader {
    fn turn_body(&self, seq: i64) -> Option<String> {
        let fut = akhomed_store::load_turn_by_seq(&self.db, self.akh_id, seq);
        let result = tokio::task::block_in_place(|| self.handle.block_on(fut));
        match result {
            Ok(Some(payload)) => {
                let user_prompt = payload
                    .get("user_prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let agent_answer = payload
                    .get("agent_answer")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Some(format!("User: {user_prompt}\nAkh: {agent_answer}"))
            }
            Ok(None) => None,
            Err(e) => {
                eprintln!("turn body lookup failed for seq={seq}: {e}");
                None
            }
        }
    }
}

/// Resolve the query symbols for a question the same way recall's caller
/// would: extract keywords, then resolve (not necessarily create — but
/// `resolve_or_create_entity` is the only public path and is idempotent: if
/// the symbol already exists from harvest, it resolves the same id) a symbol
/// per keyword.
fn resolve_query_symbols(engine: &Engine, question: &str) -> Vec<akh_medu::symbol::SymbolId> {
    extract_keywords(question)
        .iter()
        .map(|kw| engine.resolve_or_create_entity(kw).unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires AKH_TEST_PG_URL"]
async fn transcript_to_episodic_recall_survives_restart_across_pg() {
    let Ok(url) = std::env::var("AKH_TEST_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_PG_URL");
        return;
    };
    let db = connect(&akhomed_store::PoolOptions::new(url))
        .await
        .unwrap();
    run_migrations(&db).await.unwrap();

    // Isolation: a fresh akh_id per test run, so repeated runs don't collide
    // on watermark/episode state.
    let akh_id = Uuid::new_v4();
    let workspace_id = Uuid::new_v4();
    ensure_akh(&db, akh_id, "adr091-acceptance", None)
        .await
        .unwrap();
    let residency = Arc::new(
        akhomed_store::StablePgSession::connect(&akhomed_store::PoolOptions::new(
            std::env::var("AKH_TEST_PG_URL").unwrap(),
        ))
        .await
        .unwrap(),
    );
    assert!(residency.try_acquire_akh_lock(akh_id).await.unwrap());
    let residency_epoch =
        akhomed_store::claim_akh_residency(&db, akh_id, workspace_id)
            .await
            .unwrap();

    // ---------------------------------------------------------------
    // Phase A — "session A": inject a fact, flush the WAL, close the session.
    // ---------------------------------------------------------------
    {
        let engine_a = Engine::new(EngineConfig::default()).unwrap();

        let log_a = Arc::new(MutationLog::new(
            Arc::clone(&residency),
            akh_id,
            workspace_id,
            residency_epoch,
            WalConfig::new(Duration::from_millis(50), 64, 1 << 20),
        ));
        engine_a
            .set_mutation_sink(Arc::clone(&log_a) as Arc<dyn MutationSink>)
            .map_err(|_| ())
            .expect("mutation sink install should succeed on a fresh engine");

        let payload = turn_payload(
            "remember the maintenance vault code",
            &format!("Noted: the maintenance vault code is {THE_FACT}."),
        );
        engine_a.record_turn(payload);
        assert_eq!(log_a.pending(), 1, "the turn is buffered before flush");

        let flushed = log_a.flush_once().await;
        assert_eq!(flushed, 1, "exactly one turn row flushed to Postgres");

        // Session A closes: engine_a and log_a drop at the end of this block.
        // Nothing in-memory survives past this point — Postgres is the only
        // carrier into Phase B.
    }

    // Exactly one `kind='turn'` row exists for this akh — the WAL hop is the
    // sole source of truth, and we wrote exactly one turn.
    let turns = load_turns_since(&db, akh_id, 0, 100).await.unwrap();
    assert_eq!(
        turns.len(),
        1,
        "exactly one kind='turn' row survives the session-A close, in Postgres"
    );

    // ---------------------------------------------------------------
    // Phase B — "sleep/harvest after restart": a completely fresh Engine
    // (the restart) harvests the WAL tail into episode-index symbols.
    // ---------------------------------------------------------------
    let engine_b = Engine::new(EngineConfig::default()).unwrap();
    let agent_b = Agent::new(Arc::new(engine_b), AgentConfig::default()).unwrap();
    // (Agent owns the Arc<Engine> from here; recover a plain reference below.)
    let engine_b = agent_b.engine();
    let predicates_b = agent_b.predicates();

    assert_eq!(
        get_harvest_watermark(&db, akh_id).await.unwrap(),
        0,
        "fresh akh: watermark starts at 0"
    );

    let metrics =
        tokio::task::block_in_place(|| harvest_daily_log(&db, engine_b, predicates_b, akh_id))
            .unwrap();
    assert_eq!(metrics.turns_harvested, 1, "the one turn row is harvested");
    assert_eq!(metrics.episodes_minted, 1, "exactly one episode minted");
    assert!(metrics.last_seq > 0);

    let watermark_after = get_harvest_watermark(&db, akh_id).await.unwrap();
    assert_eq!(
        watermark_after, metrics.last_seq,
        "watermark advanced in Postgres, so it travels across the restart too"
    );

    let episode_syms: Vec<_> = engine_b
        .all_symbols()
        .into_iter()
        .filter(|meta| engine_b.resolve_label(meta.id).starts_with("episode:turn:"))
        .collect();
    assert_eq!(
        episode_syms.len(),
        1,
        "exactly one episode:turn:* symbol exists in the harvesting engine"
    );

    // Idempotence at the acceptance level: re-running harvest on the same
    // (still-live) engine instance mints nothing new.
    let metrics_again =
        tokio::task::block_in_place(|| harvest_daily_log(&db, engine_b, predicates_b, akh_id))
            .unwrap();
    assert_eq!(
        metrics_again.turns_harvested, 0,
        "re-harvest past the watermark finds nothing new"
    );
    assert_eq!(metrics_again.episodes_minted, 0);
    assert_eq!(metrics_again.last_seq, watermark_after);

    // ---------------------------------------------------------------
    // Phase C — "session B asks": still engine_b (the episode index lives in
    // its KG — see the module doc comment for why a third fresh-Engine
    // restart isn't needed to prove the durable loop). Install the
    // Postgres-backed TurnBodyReader and ask a question that never mentions
    // the fact directly, only the keywords that led to it.
    // ---------------------------------------------------------------
    engine_b
        .set_turn_body_reader(Arc::new(TestTurnBodyReader {
            db: db.clone(),
            akh_id,
            handle: tokio::runtime::Handle::current(),
        }))
        .map_err(|_| ())
        .expect("turn body reader install should succeed on engine_b");

    let query_syms_b = resolve_query_symbols(engine_b, "what is the maintenance vault code?");
    assert!(
        !query_syms_b.is_empty(),
        "the question must yield at least one query symbol"
    );

    let recalled = recall_episodes(engine_b, &query_syms_b, predicates_b, 5, None).unwrap();
    assert!(
        !recalled.is_empty(),
        "session B's question must recall at least one episode"
    );
    assert!(
        recalled.iter().any(|ep| ep.summary.contains(THE_FACT)),
        "at least one recalled episode's summary must contain the fact stated \
         only in session A ({THE_FACT}); got: {:?}",
        recalled.iter().map(|ep| &ep.summary).collect::<Vec<_>>()
    );

    // ---------------------------------------------------------------
    // Phase D — control: a completely fresh Engine with NO harvest run and no
    // reader installed. Same query, over a KG that never saw session A's
    // turn — this is the harness-off control from ADR 091 decision 8
    // ("passthrough stays 0").
    // ---------------------------------------------------------------
    let engine_d = Engine::new(EngineConfig::default()).unwrap();
    let agent_d = Agent::new(Arc::new(engine_d), AgentConfig::default()).unwrap();
    let engine_d = agent_d.engine();
    let predicates_d = agent_d.predicates();

    let query_syms_d = resolve_query_symbols(engine_d, "what is the maintenance vault code?");
    let recalled_d = recall_episodes(engine_d, &query_syms_d, predicates_d, 5, None).unwrap();
    assert!(
        recalled_d.iter().all(|ep| !ep.summary.contains(THE_FACT)),
        "control engine (no harvest run) must never recall the fact: {:?}",
        recalled_d.iter().map(|ep| &ep.summary).collect::<Vec<_>>()
    );
    assert!(
        recalled_d.is_empty(),
        "control engine has no episodes at all — the passthrough control stays 0"
    );
}
