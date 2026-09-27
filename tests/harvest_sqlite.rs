//! SQLite-backed test of the ADR 091 transcript harvest (Phase 53b): the
//! Mac-resident Roman akhs run on the standalone SQLite backing (ADR 113
//! decision 5), so harvest must work there, not only on Postgres
//! (`harvest_live.rs`).
//!
//! Needs no external service: each test opens a tempdir
//! `sqlite://…?mode=rwc` store, like `sqlite_identity_roundtrip.rs`. Turn rows
//! are seeded through the residency-fenced SQLite session (`StablePgSession`
//! over `LocalSession`), the same path the WAL flusher uses.
//!
//!   cargo test --features postgres --test harvest_sqlite
//!   cargo test --features server --test harvest_sqlite   # adds the daemon test
#![cfg(feature = "postgres")]

use std::sync::Arc;

use akh_medu::agent::harvest::{HARVEST_BATCH_LIMIT, harvest_daily_log, harvest_tick};
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akhomed_store::uuid::Uuid;
use akhomed_store::{
    DatabaseConnection, MutationRow, PoolOptions, StablePgSession, append_mutations_fenced,
    ensure_akh, get_harvest_watermark,
};

struct SqliteFixture {
    _dir: tempfile::TempDir,
    db: DatabaseConnection,
    session: StablePgSession,
    akh_id: Uuid,
    workspace_id: Uuid,
    epoch: i64,
}

async fn sqlite_fixture(name: &str) -> SqliteFixture {
    let dir = tempfile::tempdir().unwrap();
    let options = PoolOptions::new(format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("harvest.sqlite3").display()
    ));
    akhomed_store::run_migrations_locked(&options)
        .await
        .unwrap();
    let db = akhomed_store::connect(&options).await.unwrap();
    let akh_id = akh_medu::wal::akh_id_for(name);
    let workspace_id = akh_medu::wal::workspace_id_for(name);
    ensure_akh(&db, akh_id, name, None).await.unwrap();

    let session = StablePgSession::connect(&options).await.unwrap();
    assert!(session.try_acquire_akh_lock(akh_id).await.unwrap());
    let epoch = session
        .claim_akh_residency(akh_id, workspace_id)
        .await
        .unwrap();
    SqliteFixture {
        _dir: dir,
        db,
        session,
        akh_id,
        workspace_id,
        epoch,
    }
}

impl SqliteFixture {
    async fn seed_turns(&self, count: usize) {
        let rows: Vec<MutationRow> = (0..count)
            .map(|i| MutationRow {
                kind: "turn".to_string(),
                payload: serde_json::json!({
                    "v": 1,
                    "ts": format!("2026-09-24T00:00:{:02}.000000000Z", i % 60),
                    "workspace": "harvest-sqlite",
                    "model": "test-model",
                    "user_prompt": format!("Question number {i} about sqlite harvest"),
                    "agent_answer": format!("Answer number {i} from the akh"),
                    "tools": [],
                }),
            })
            .collect();
        let inserted = append_mutations_fenced(
            &self.session,
            self.akh_id,
            self.workspace_id,
            self.epoch,
            &rows,
        )
        .await
        .unwrap();
        assert_eq!(inserted, count as u64);
    }
}

fn agent() -> (Arc<Engine>, Agent) {
    let engine = Arc::new(Engine::new(EngineConfig::default()).unwrap());
    // `Agent::new` bootstraps the well-known `agent:` predicates.
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).unwrap();
    (engine, agent)
}

fn episode_count(engine: &Engine) -> usize {
    engine
        .all_symbols()
        .into_iter()
        .filter(|meta| engine.resolve_label(meta.id).starts_with("episode:turn:"))
        .count()
}

/// The watermark advances past harvested turns and a second pass is a no-op.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sqlite_harvest_advances_watermark_and_is_idempotent() {
    let fx = sqlite_fixture("harvest-sqlite-idempotent").await;
    assert_eq!(get_harvest_watermark(&fx.db, fx.akh_id).await.unwrap(), 0);
    fx.seed_turns(3).await;

    let (engine, agent) = agent();
    let predicates = agent.predicates();

    let first =
        tokio::task::block_in_place(|| harvest_daily_log(&fx.db, &engine, predicates, fx.akh_id))
            .unwrap();
    assert_eq!(first.turns_harvested, 3);
    assert_eq!(first.episodes_minted, 3);
    assert!(first.last_seq > 0);
    let watermark = get_harvest_watermark(&fx.db, fx.akh_id).await.unwrap();
    assert_eq!(
        watermark, first.last_seq,
        "watermark advanced to the last turn"
    );
    assert_eq!(episode_count(&engine), 3);

    let second =
        tokio::task::block_in_place(|| harvest_daily_log(&fx.db, &engine, predicates, fx.akh_id))
            .unwrap();
    assert_eq!(second.turns_harvested, 0, "nothing past the watermark");
    assert_eq!(second.episodes_minted, 0);
    assert_eq!(second.last_seq, watermark);
    assert_eq!(
        get_harvest_watermark(&fx.db, fx.akh_id).await.unwrap(),
        watermark
    );
    assert_eq!(episode_count(&engine), 3, "no duplicate episodes");
}

/// A cutover-sized backlog (more than one batch) drains in one bounded tick,
/// and a capped tick reports that more may remain.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sqlite_harvest_tick_drains_a_multi_batch_backlog() {
    let fx = sqlite_fixture("harvest-sqlite-backlog").await;
    let backlog = HARVEST_BATCH_LIMIT as usize + 44;
    fx.seed_turns(backlog).await;
    let (engine, agent) = agent();
    let predicates = agent.predicates();

    // Capped at one batch: a full batch means the backlog may remain.
    let capped =
        tokio::task::block_in_place(|| harvest_tick(&fx.db, &engine, predicates, fx.akh_id, 1))
            .unwrap();
    assert_eq!(capped.batches.len(), 1);
    assert_eq!(capped.turns_harvested(), HARVEST_BATCH_LIMIT as usize);
    assert!(capped.backlog_may_remain);

    // The next tick drains the rest and stops early on the short batch.
    let rest = tokio::task::block_in_place(|| {
        harvest_tick(
            &fx.db,
            &engine,
            predicates,
            fx.akh_id,
            akh_medu::agent::autonomy::HARVEST_MAX_BATCHES_PER_TICK,
        )
    })
    .unwrap();
    assert_eq!(rest.batches.len(), 1);
    assert_eq!(rest.turns_harvested(), 44);
    assert!(!rest.backlog_may_remain);
    assert_eq!(episode_count(&engine), backlog);
    assert_eq!(
        get_harvest_watermark(&fx.db, fx.akh_id).await.unwrap(),
        rest.last_seq().unwrap()
    );
}

/// The daemon's sleep tick runs harvest as its own task before the sleep
/// cycle, against the SQLite store, and reports it with a fact delta.
/// (That a *failing* sleep cycle cannot skip harvest is proven at the
/// scheduling seam in `agent::daemon::tests`, where the task bodies can be
/// injected.)
#[cfg(feature = "daemon")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_sleep_tick_harvests_sqlite_turns() {
    use akh_medu::agent::autonomy::TaskOutcome;
    use akh_medu::agent::{AgentDaemon, DaemonConfig};

    let fx = sqlite_fixture("harvest-sqlite-daemon").await;
    fx.seed_turns(5).await;
    let (_engine, agent) = agent();
    let shared = Arc::new(std::sync::RwLock::new(agent));
    let mut daemon = AgentDaemon::new(Arc::clone(&shared), DaemonConfig::default())
        .with_db(fx.db.clone(), fx.akh_id)
        .with_workspace("harvest-sqlite-daemon");

    let tick = tokio::task::block_in_place(|| daemon.run_sleep_tick());
    assert_eq!(tick.harvest.outcome, TaskOutcome::Ran);
    assert!(
        tick.harvest.facts.asserted > 0,
        "harvest minted episode links"
    );
    assert!(get_harvest_watermark(&fx.db, fx.akh_id).await.unwrap() > 0);

    let again = tokio::task::block_in_place(|| daemon.run_sleep_tick());
    assert_eq!(again.harvest.outcome, TaskOutcome::Ran);
    assert_eq!(again.harvest.facts.asserted, 0, "second tick is idempotent");
}
