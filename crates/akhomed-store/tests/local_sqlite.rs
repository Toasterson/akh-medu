use akhomed_store::{MutationRow, PoolOptions, StablePgSession};

#[tokio::test]
async fn sqlite_migrations_residency_wal_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let opts = PoolOptions::new(format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("path with spaces/akh.sqlite3").display()
    ));
    akhomed_store::run_migrations_locked(&opts).await.unwrap();
    akhomed_store::run_migrations_locked(&opts).await.unwrap();
    let db = akhomed_store::connect(&opts).await.unwrap();
    let id = akhomed_store::uuid::Uuid::new_v4();
    let workspace = akhomed_store::uuid::Uuid::new_v4();
    akhomed_store::ensure_akh(&db, id, "Janus", None)
        .await
        .unwrap();
    let first = StablePgSession::connect(&opts).await.unwrap();
    let second = StablePgSession::connect(&opts).await.unwrap();
    assert!(first.try_acquire_akh_lock(id).await.unwrap());
    assert!(!second.try_acquire_akh_lock(id).await.unwrap());
    let epoch = first.claim_akh_residency(id, workspace).await.unwrap();
    let rows = vec![MutationRow {
        kind: "assert".into(),
        payload: serde_json::json!({"test": "survives reopen"}),
    }];
    assert_eq!(
        akhomed_store::append_mutations_fenced(&first, id, workspace, epoch, &rows)
            .await
            .unwrap(),
        1
    );
    assert!(first.release_akh_lock(id).await.unwrap());
    assert!(
        akhomed_store::append_mutations_fenced(&first, id, workspace, epoch, &rows)
            .await
            .is_err()
    );
    assert!(second.try_acquire_akh_lock(id).await.unwrap());
    let next = second.claim_akh_residency(id, workspace).await.unwrap();
    assert!(next > epoch);
    assert!(
        akhomed_store::append_mutations_fenced(&second, id, workspace, epoch, &rows)
            .await
            .is_err()
    );
    drop(first);
    drop(second);
    drop(db);
    let db = akhomed_store::connect(&opts).await.unwrap();
    assert!(
        akhomed_store::snapshots::current_watermark(&db, id)
            .await
            .unwrap()
            > 0
    );
    let symbols = (1..=8000)
        .map(|n| akhomed_store::entity::symbol::Model {
            akh_id: id,
            id: n,
            kind: "Entity".into(),
            glyph_codepoint: None,
            label: format!("symbol-{n}"),
        })
        .collect();
    akhomed_store::snapshots::replace_symbols(&db, id, symbols)
        .await
        .unwrap();
    assert_eq!(
        akhomed_store::snapshots::load_symbols(&db, id)
            .await
            .unwrap()
            .len(),
        8000
    );
}

#[tokio::test]
async fn sqlite_parallel_migrations_are_serialized() {
    let dir = tempfile::tempdir().unwrap();
    let opts = PoolOptions::new(format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("migration.sqlite3").display()
    ));
    let (a, b) = tokio::join!(
        akhomed_store::run_migrations_locked(&opts),
        akhomed_store::run_migrations_locked(&opts)
    );
    a.unwrap();
    b.unwrap();
}

#[tokio::test]
async fn sqlite_lock_child() {
    let Ok(url) = std::env::var("AKH_TEST_SQLITE_CHILD_URL") else {
        return;
    };
    let id =
        akhomed_store::uuid::Uuid::parse_str(&std::env::var("AKH_TEST_SQLITE_CHILD_ID").unwrap())
            .unwrap();
    let owner = StablePgSession::connect(&PoolOptions::new(url))
        .await
        .unwrap();
    assert!(owner.try_acquire_akh_lock(id).await.unwrap());
    std::fs::write(
        std::env::var("AKH_TEST_SQLITE_CHILD_READY").unwrap(),
        "ready",
    )
    .unwrap();
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;
}

#[tokio::test]
async fn sqlite_process_death_releases_residency() {
    let dir = tempfile::tempdir().unwrap();
    let opts = PoolOptions::new(format!(
        "sqlite://{}?mode=rwc",
        dir.path().join("crash.sqlite3").display()
    ));
    akhomed_store::run_migrations_locked(&opts).await.unwrap();
    let id = akhomed_store::uuid::Uuid::new_v4();
    let ready = dir.path().join("ready");
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "sqlite_lock_child"])
        .env("AKH_TEST_SQLITE_CHILD_URL", &opts.url)
        .env("AKH_TEST_SQLITE_CHILD_ID", id.to_string())
        .env("AKH_TEST_SQLITE_CHILD_READY", &ready)
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    while !ready.exists() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    if !ready.exists() {
        let _ = child.kill();
        let _ = child.wait();
        panic!("child did not acquire lock");
    }
    let other = StablePgSession::connect(&opts).await.unwrap();
    let refused = !other.try_acquire_akh_lock(id).await.unwrap();
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(refused);
    assert!(other.try_acquire_akh_lock(id).await.unwrap());
}
