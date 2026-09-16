#![cfg(feature = "postgres")]

use akh_medu::{
    agent::{Agent, AgentConfig},
    compartment::psyche::Psyche,
    engine::{Engine, EngineConfig},
};
use std::sync::Arc;

#[tokio::test]
async fn optional_psyche_fields_survive_sqlite_wake_and_local_session_resume() {
    let dir = tempfile::tempdir().unwrap();
    let options = akhomed_store::PoolOptions::new(format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("identity.sqlite3").display()
    ));
    akhomed_store::run_migrations_locked(&options)
        .await
        .unwrap();
    let db = akhomed_store::connect(&options).await.unwrap();
    let id = akh_medu::wal::akh_id_for("sqlite-regression");
    let workspace = akh_medu::wal::workspace_id_for("sqlite-regression");
    akhomed_store::ensure_akh(&db, id, "sqlite-regression", None)
        .await
        .unwrap();
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(dir.path().join("initial")),
            ..Default::default()
        })
        .unwrap(),
    );
    let mut psyche = Psyche::default();
    psyche.persona.name = "SQLite regression identity".into();
    psyche.mark_awakened();
    assert!(psyche.ennead_role.is_none());
    assert!(psyche.avatar_hash.is_none());
    engine.compartments().unwrap().force_set_psyche(psyche);
    let agent = Agent::new(engine, AgentConfig::default()).unwrap();
    let snapshot = akh_medu::snapshot::capture_snapshot(&agent, id, workspace).unwrap();
    akh_medu::snapshot::persist_snapshot(snapshot, &db)
        .await
        .unwrap();
    drop(agent);

    let recovered_path = dir.path().join("fresh-cache");
    let recovered = akh_medu::snapshot::wake(&recovered_path, id, workspace, &db)
        .await
        .unwrap();
    let restored = Agent::resume(recovered, AgentConfig::default()).unwrap();
    let psyche = restored
        .psyche()
        .expect("relational wake must retain identity");
    assert!(psyche.is_awakened());
    assert_eq!(psyche.persona.name, "SQLite regression identity");
    restored.persist_session().unwrap();
    drop(restored);

    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(recovered_path),
            ..Default::default()
        })
        .unwrap(),
    );
    let resumed = Agent::resume(engine, AgentConfig::default()).unwrap();
    assert_eq!(
        resumed.psyche().unwrap().persona.name,
        "SQLite regression identity"
    );
    assert!(resumed.psyche().unwrap().is_awakened());
}
