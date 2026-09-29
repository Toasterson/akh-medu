//! ADR 120: per-akh SQLite layout and the offline split tool.

use std::path::{Path, PathBuf};
use std::time::Duration;

use akhomed_store::entity::{akh_state, drive, fact, goal, provenance_snapshot, symbol};
use akhomed_store::sea_orm::sqlx;
use akhomed_store::snapshots::{self, SnapshotPublication};
use akhomed_store::split::{self, SplitError, SplitRequest};
use akhomed_store::uuid::Uuid;
use akhomed_store::{
    AkhStoreName, DatabaseConnection, LayoutError, MutationRow, PoolOptions, StablePgSession,
    StoreError, StoreLayout, StoreTopology,
};

fn sqlite_url(path: &Path) -> String {
    format!("sqlite://{}?mode=rwc", path.display())
}

fn options(path: &Path, busy: Duration) -> PoolOptions {
    PoolOptions {
        url: sqlite_url(path),
        max_connections: 1,
        connect_timeout: busy,
    }
}

fn state(akh_id: Uuid, watermark: i64) -> akh_state::Model {
    akh_state::Model {
        akh_id,
        watermark,
        harvest_watermark: 0,
        librarian_watermark: 0,
        allocator_next: 1_000,
        cycle_count: 7,
        session_start_cycle: 0,
        sleep_phase: "Wake".into(),
        last_sleep_cycle: 0,
        psyche: None,
        extra: Some(serde_json::json!({"complete": true})),
        updated_at: akhomed_store::chrono::Utc::now(),
    }
}

/// A publication with `symbols` symbols and half as many facts, so a
/// snapshot is a real multi-table write.
fn publication(name: &str, epoch: i64, symbols: i64, watermark: i64) -> SnapshotPublication {
    let akh_id = akhomed_store::akh_id_for(name);
    let workspace_id = akhomed_store::workspace_id_for(name);
    let now = akhomed_store::chrono::Utc::now();
    SnapshotPublication {
        residency_epoch: epoch,
        symbols: (1..=symbols)
            .map(|id| symbol::Model {
                akh_id,
                id,
                kind: "Entity".into(),
                glyph_codepoint: None,
                label: format!("{name}-symbol-{id}"),
            })
            .collect(),
        facts: (1..=symbols / 2)
            .map(|id| fact::Model {
                workspace_id,
                subject: id,
                predicate: 1,
                object: id + 1,
                akh_id,
                confidence: 0.75,
                asserted_at: now,
                retracted_at: None,
                provenance_id: Some(id),
                compartment: None,
            })
            .collect(),
        provenance: (1..=3)
            .map(|id| provenance_snapshot::Model {
                akh_id,
                provenance_id: id,
                workspace_id,
                derived_symbol: id,
                sources: serde_json::json!([id]),
                derivation: serde_json::json!({"kind": "test"}),
                confidence: 0.5,
                depth: 1,
                created_at: now,
            })
            .collect(),
        drives: vec![drive::Model {
            akh_id,
            kind: "curiosity".into(),
            strength: 0.25,
            last_computed: 3,
            last_triple_count: 4,
            last_symbol_count: 5,
            threshold: 0.6,
        }],
        agenda: Vec::new(),
        curiosity: Vec::new(),
        goals: vec![goal::Model {
            akh_id,
            id: 42,
            description: format!("{name} goal"),
            status: "Active".into(),
            priority: 3,
            success_criteria: "done".into(),
            parent: None,
            created_at: 1,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            detail: None,
        }],
        goal_hierarchy: Vec::new(),
        goal_dependencies: Vec::new(),
        failures: Vec::new(),
        efforts: Vec::new(),
        methods: Vec::new(),
        method_steps: Vec::new(),
        competence_tools: Vec::new(),
        competence_categories: Vec::new(),
        competence_calibration: Vec::new(),
        sleep_metrics: Vec::new(),
        state: state(akh_id, watermark),
    }
}

/// One resident akh: its store, its residency session and fenced epoch.
struct Resident {
    name: String,
    db: DatabaseConnection,
    session: StablePgSession,
    epoch: i64,
}

async fn resident(topology: &StoreTopology, host: &DatabaseConnection, name: &str) -> Resident {
    let store_name = AkhStoreName::new(name).unwrap();
    let db = topology.open_akh_store(&store_name, host).await.unwrap();
    let akh_id = store_name.akh_id();
    akhomed_store::ensure_akh(&db, akh_id, name, None)
        .await
        .unwrap();
    let session = topology.connect_residency(&store_name, &db).await.unwrap();
    assert!(session.try_acquire_akh_lock(akh_id).await.unwrap());
    let epoch = session
        .claim_akh_residency(akh_id, store_name.workspace_id())
        .await
        .unwrap();
    Resident {
        name: name.to_owned(),
        db,
        session,
        epoch,
    }
}

async fn host_store(topology: &StoreTopology) -> DatabaseConnection {
    akhomed_store::run_migrations_locked(topology.host())
        .await
        .unwrap();
    akhomed_store::connect(topology.host()).await.unwrap()
}

/// Hold SQLite's write lock on `path` from a separate connection until the
/// returned connection is rolled back or dropped.
async fn hold_write_lock(path: &Path) -> sqlx::SqliteConnection {
    use sqlx::ConnectOptions as _;
    let mut conn = sqlx::sqlite::SqliteConnectOptions::new()
        .filename(path)
        .connect()
        .await
        .unwrap();
    sqlx::query("BEGIN IMMEDIATE")
        .execute(&mut conn)
        .await
        .unwrap();
    conn
}

#[tokio::test]
async fn shared_layout_contends_but_per_akh_snapshots_do_not() {
    let busy = Duration::from_millis(100);

    // Shared layout: akh B's snapshot waits on akh A's write lock and fails
    // as lock contention, not as a generic database error.
    let dir = tempfile::tempdir().unwrap();
    let shared_path = dir.path().join("shared.sqlite3");
    let shared = StoreTopology::new(options(&shared_path, busy), StoreLayout::Shared).unwrap();
    let host = host_store(&shared).await;
    let a = resident(&shared, &host, "alpha").await;
    let b = resident(&shared, &host, "beta").await;
    let blocker = hold_write_lock(&shared_path).await;
    let error = snapshots::publish_snapshot(
        &b.db,
        akhomed_store::workspace_id_for(&b.name),
        akhomed_store::akh_id_for(&b.name),
        publication(&b.name, b.epoch, 50, 0),
    )
    .await
    .expect_err("a held write lock on the shared file blocks every akh");
    assert!(error.is_lock_contention(), "{error:?}");
    let diagnostic: &dyn miette::Diagnostic = &error;
    assert_eq!(
        diagnostic.code().unwrap().to_string(),
        "akhomed_store::sqlite_busy"
    );
    let help = diagnostic.help().unwrap().to_string();
    assert!(
        help.contains("per_akh") && help.contains("busy_timeout"),
        "{help}"
    );
    drop(blocker);
    drop((a, b));

    // Per-akh layout: the same held lock on alpha's file leaves beta alone.
    let per_dir = tempfile::tempdir().unwrap();
    let host_path = per_dir.path().join("host.sqlite3");
    let per_akh = StoreTopology::new(options(&host_path, busy), StoreLayout::PerAkh).unwrap();
    let host = host_store(&per_akh).await;
    let a = resident(&per_akh, &host, "alpha").await;
    let b = resident(&per_akh, &host, "beta").await;
    let alpha_path = per_akh
        .akh_path(&AkhStoreName::new("alpha").unwrap())
        .unwrap();
    assert_eq!(alpha_path, per_dir.path().join("akh.alpha.sqlite3"));
    let blocker = hold_write_lock(&alpha_path).await;
    snapshots::publish_snapshot(
        &b.db,
        akhomed_store::workspace_id_for(&b.name),
        akhomed_store::akh_id_for(&b.name),
        publication(&b.name, b.epoch, 50, 0),
    )
    .await
    .expect("beta's file is not locked by alpha's writer");
    drop(blocker);
    drop(a);

    // No per-akh table row lands in the host file.
    let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM akhomed_akh")
        .fetch_one(host.get_sqlite_connection_pool())
        .await
        .unwrap();
    assert_eq!(rows, 0);
}

#[tokio::test]
async fn two_akhs_snapshot_and_flush_concurrently_without_sqlite_busy() {
    // Per-akh: no akh waits on another akh's writer, and one akh's snapshot
    // and WAL share one connection, so no write ever meets SQLITE_BUSY (which
    // a read-then-write transaction would hit at once, whatever busy_timeout).
    let dir = tempfile::tempdir().unwrap();
    let topology = StoreTopology::new(
        options(&dir.path().join("host.sqlite3"), Duration::from_secs(10)),
        StoreLayout::PerAkh,
    )
    .unwrap();
    let host = host_store(&topology).await;
    let short = topology.clone();

    let mut residents = Vec::new();
    for name in ["janus", "minerva"] {
        residents.push(resident(&short, &host, name).await);
    }
    let (first, second) = tokio::join!(writer(&residents[0]), writer(&residents[1]));
    for result in [first, second] {
        if let Err(error) = result {
            panic!(
                "per-akh writers must not contend (lock contention: {}): {error:?}",
                error.is_lock_contention()
            );
        }
    }
    for r in &residents {
        let akh_id = akhomed_store::akh_id_for(&r.name);
        assert_eq!(
            snapshots::load_symbols(&r.db, akh_id).await.unwrap().len(),
            2_000
        );
        assert_eq!(
            snapshots::current_watermark(&r.db, akh_id).await.unwrap(),
            15
        );
    }
}

/// Fifteen snapshot rounds for one akh, with its fenced WAL flushing through
/// the residency session at the same time (the daemon's real shape).
async fn writer(r: &Resident) -> Result<(), StoreError> {
    let akh_id = akhomed_store::akh_id_for(&r.name);
    let workspace_id = akhomed_store::workspace_id_for(&r.name);
    let snapshots = async {
        for round in 0..15 {
            snapshots::publish_snapshot(
                &r.db,
                workspace_id,
                akh_id,
                publication(&r.name, r.epoch, 2_000, round),
            )
            .await?;
        }
        Ok::<_, StoreError>(())
    };
    let wal = async {
        for round in 0..15 {
            akhomed_store::append_mutations_fenced(
                &r.session,
                akh_id,
                workspace_id,
                r.epoch,
                &[MutationRow {
                    kind: "assert".into(),
                    payload: serde_json::json!({"round": round}),
                }],
            )
            .await?;
            tokio::task::yield_now().await;
        }
        Ok::<_, StoreError>(())
    };
    let (snapshots, wal) = tokio::join!(snapshots, wal);
    snapshots.and(wal)
}

#[tokio::test]
async fn residency_locks_anchor_at_the_host_database() {
    let dir = tempfile::tempdir().unwrap();
    let host_path = dir.path().join("host.sqlite3");
    let topology = StoreTopology::new(
        options(&host_path, Duration::from_secs(5)),
        StoreLayout::PerAkh,
    )
    .unwrap();
    let host = host_store(&topology).await;
    let owner = resident(&topology, &host, "vesta").await;
    let name = AkhStoreName::new("vesta").unwrap();
    let rival = topology.connect_residency(&name, &owner.db).await.unwrap();
    assert!(!rival.try_acquire_akh_lock(name.akh_id()).await.unwrap());
    let lock = PathBuf::from(format!(
        "{}.akh-{}.lock",
        host_path.display(),
        name.akh_id()
    ));
    assert!(lock.exists(), "residency lock sits beside the host file");
    // The fenced WAL writes land in the akh's own file.
    let rows = [MutationRow {
        kind: "assert".into(),
        payload: serde_json::json!({"x": 1}),
    }];
    akhomed_store::append_mutations_fenced(
        &owner.session,
        name.akh_id(),
        name.workspace_id(),
        owner.epoch,
        &rows,
    )
    .await
    .unwrap();
    assert_eq!(
        snapshots::current_watermark(&owner.db, name.akh_id())
            .await
            .unwrap(),
        1
    );
}

/// Build a realistic shared-layout source with two akhs plus host rows.
async fn shared_source(path: &Path) {
    let topology =
        StoreTopology::new(options(path, Duration::from_secs(5)), StoreLayout::Shared).unwrap();
    let host = host_store(&topology).await;
    for (name, symbols) in [("janus", 120), ("vesta", 40)] {
        let r = resident(&topology, &host, name).await;
        let akh_id = akhomed_store::akh_id_for(name);
        let workspace_id = akhomed_store::workspace_id_for(name);
        let rows: Vec<MutationRow> = (0..symbols)
            .map(|n| MutationRow {
                kind: if n % 5 == 0 { "turn" } else { "assert" }.into(),
                payload: serde_json::json!({"n": n, "who": name, "ratio": 0.123_456_789_012_345_67}),
            })
            .collect();
        akhomed_store::append_mutations_fenced(&r.session, akh_id, workspace_id, r.epoch, &rows)
            .await
            .unwrap();
        let watermark = snapshots::current_watermark(&r.db, akh_id).await.unwrap();
        snapshots::publish_snapshot(
            &r.db,
            workspace_id,
            akh_id,
            publication(name, r.epoch, symbols, watermark),
        )
        .await
        .unwrap();
        akhomed_store::set_harvest_watermark(&r.db, akh_id, watermark - 1)
            .await
            .unwrap();
        akhomed_store::join_workspace(
            &host,
            workspace_id,
            akh_id,
            "member",
            akhomed_store::chrono::Utc::now(),
        )
        .await
        .unwrap();
        akhomed_store::record_pack_activation(
            &host,
            workspace_id,
            "mt:schema/test@1",
            akh_id,
            akhomed_store::chrono::Utc::now(),
        )
        .await
        .unwrap();
        assert!(r.session.release_akh_lock(akh_id).await.unwrap());
    }
    akhomed_store::admit_pack(
        &host,
        akhomed_store::SchemaPackRow {
            name: "test".into(),
            version: "1".into(),
            scope: "shared".into(),
            source_url: None,
            checksum: "abc".into(),
            confirmed_by: "tester".into(),
            blob: vec![0, 1, 2, 255],
        },
    )
    .await
    .unwrap();
    host.close().await.unwrap();
}

fn sha256(path: &Path) -> String {
    use sha2::Digest as _;
    let bytes = std::fs::read(path).unwrap_or_default();
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn request(source: &Path, host: &Path) -> SplitRequest {
    SplitRequest {
        source: source.to_path_buf(),
        host: host.to_path_buf(),
        allow_orphans: false,
        busy_timeout: Duration::from_secs(5),
    }
}

fn count(tables: &[split::TableCount], table: &str) -> u64 {
    tables
        .iter()
        .find(|t| t.table == table)
        .map_or(0, |t| t.rows)
}

#[tokio::test]
async fn split_copies_verifies_and_leaves_the_source_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("akh.sqlite3");
    shared_source(&source).await;
    let before = sha256(&source);
    let host = dir.path().join("host.sqlite3");
    let evidence_path = dir.path().join("evidence/split.json");

    // Dry run: per-akh counts, nothing created.
    let plan = split::plan(&request(&source, &host)).await.unwrap();
    assert!(plan.in_use.is_empty());
    assert!(plan.orphans.is_empty(), "{:?}", plan.orphans);
    let janus = plan.akhs.iter().find(|a| a.name == "janus").unwrap();
    assert_eq!(count(&janus.tables, "akhomed_symbols"), 120);
    assert_eq!(count(&janus.tables, "akhomed_facts"), 60);
    assert_eq!(count(&janus.tables, "akhomed_mutation_log"), 120);
    assert_eq!(count(&janus.tables, "akhomed_akh"), 1);
    assert_eq!(janus.path, dir.path().join("akh.janus.sqlite3"));
    let vesta = plan.akhs.iter().find(|a| a.name == "vesta").unwrap();
    assert_eq!(count(&vesta.tables, "akhomed_symbols"), 40);
    assert_eq!(count(&plan.host_tables, "workspaces_membership"), 2);
    assert_eq!(count(&plan.host_tables, "akhomed_schema_registry"), 1);
    assert!(!host.exists() && !janus.path.exists());

    // A running daemon (a held residency lock beside the source) blocks it.
    {
        let shared = StablePgSession::connect(&options(&source, Duration::from_secs(5)))
            .await
            .unwrap();
        assert!(
            shared
                .try_acquire_akh_lock(akhomed_store::akh_id_for("vesta"))
                .await
                .unwrap()
        );
        let dry = split::plan(&request(&source, &host)).await.unwrap();
        assert_eq!(dry.in_use.len(), 1, "dry run reports, does not refuse");
        let refused = split::execute(&request(&source, &host), &evidence_path)
            .await
            .unwrap_err();
        assert!(
            matches!(refused, SplitError::SourceInUse { .. }),
            "{refused:?}"
        );
        assert!(!host.exists());
    }

    // The real run.
    let evidence = split::execute(&request(&source, &host), &evidence_path)
        .await
        .unwrap();
    assert!(evidence.verified);
    assert_eq!(evidence.source.sha256_before, before);
    assert_eq!(evidence.source.sha256_after, before);
    assert_eq!(sha256(&source), before, "source file is never modified");
    assert_eq!(evidence.akhs.len(), 2);
    for akh in &evidence.akhs {
        for table in &akh.target.tables {
            assert_eq!(table.source_rows, table.target_rows);
            assert_eq!(table.source_sha256, table.target_sha256);
        }
    }
    let written: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&evidence_path).unwrap()).unwrap();
    assert_eq!(written["schema"], split::EVIDENCE_SCHEMA);
    assert_eq!(written["verified"], true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for path in [&host, &janus.path, &evidence_path] {
            let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{}", path.display());
        }
    }

    // The per-akh daemon reads the same state back.
    let topology =
        StoreTopology::new(options(&host, Duration::from_secs(5)), StoreLayout::PerAkh).unwrap();
    let host_db = akhomed_store::connect(topology.host()).await.unwrap();
    for (name, symbols) in [("janus", 120usize), ("vesta", 40)] {
        let store_name = AkhStoreName::new(name).unwrap();
        let db = topology
            .open_akh_store(&store_name, &host_db)
            .await
            .unwrap();
        let akh_id = store_name.akh_id();
        assert_eq!(
            snapshots::load_symbols(&db, akh_id).await.unwrap().len(),
            symbols
        );
        assert_eq!(
            snapshots::load_workspace_facts(&db, store_name.workspace_id())
                .await
                .unwrap()
                .len(),
            symbols / 2
        );
        let state = snapshots::load_akh_state(&db, akh_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            akhomed_store::get_harvest_watermark(&db, akh_id)
                .await
                .unwrap(),
            state.watermark - 1
        );
        assert!(
            snapshots::load_mutation_tail(&db, akh_id, 0)
                .await
                .unwrap()
                .len()
                == symbols
        );
        // A new WAL append continues the akh's sequence past the copied rows.
        akhomed_store::append_mutations(
            &db,
            akh_id,
            store_name.workspace_id(),
            &[MutationRow {
                kind: "assert".into(),
                payload: serde_json::json!({}),
            }],
        )
        .await
        .unwrap();
        assert!(snapshots::current_watermark(&db, akh_id).await.unwrap() > state.watermark);
        assert_eq!(
            akhomed_store::list_members(&host_db, store_name.workspace_id())
                .await
                .unwrap()
                .len(),
            1
        );
    }
    assert!(
        akhomed_store::get_pack(&host_db, "test", "1")
            .await
            .unwrap()
            .is_some()
    );

    // A second run refuses to overwrite its targets.
    let again = split::execute(
        &request(&source, &host),
        &dir.path().join("evidence/again.json"),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(again, SplitError::TargetExists { .. }),
        "{again:?}"
    );

    // Pointing a per-akh daemon at the unsplit source is refused, not ignored.
    let legacy = StoreTopology::new(
        options(&source, Duration::from_secs(5)),
        StoreLayout::PerAkh,
    )
    .unwrap();
    let legacy_host = akhomed_store::connect(legacy.host()).await.unwrap();
    let refused = legacy
        .open_akh_store(&AkhStoreName::new("janus").unwrap(), &legacy_host)
        .await
        .unwrap_err();
    assert!(
        matches!(&refused, StoreError::Layout(e) if matches!(**e, LayoutError::UnmigratedAkh { .. })),
        "{refused:?}"
    );
}

#[tokio::test]
async fn split_refuses_orphans_unless_allowed_and_cleans_up() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("akh.sqlite3");
    shared_source(&source).await;
    {
        let db = akhomed_store::connect(&options(&source, Duration::from_secs(5)))
            .await
            .unwrap();
        akhomed_store::append_mutations(
            &db,
            Uuid::new_v4(),
            Uuid::new_v4(),
            &[MutationRow {
                kind: "assert".into(),
                payload: serde_json::json!({"orphan": true}),
            }],
        )
        .await
        .unwrap();
        db.close().await.unwrap();
    }
    let host = dir.path().join("host.sqlite3");
    let plan = split::plan(&request(&source, &host)).await.unwrap();
    assert_eq!(count(&plan.orphans, "akhomed_mutation_log"), 1);
    let refused = split::execute(&request(&source, &host), &dir.path().join("e.json"))
        .await
        .unwrap_err();
    assert!(
        matches!(refused, SplitError::Orphans { rows: 1, .. }),
        "{refused:?}"
    );
    assert!(
        !host.exists(),
        "no target is created before the orphan check"
    );

    let mut allow = request(&source, &host);
    allow.allow_orphans = true;
    let evidence = split::execute(&allow, &dir.path().join("e.json"))
        .await
        .unwrap();
    assert_eq!(evidence.orphans_left_in_source.len(), 1);
}

#[tokio::test]
async fn split_refuses_the_source_as_its_own_host() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("akh.sqlite3");
    shared_source(&source).await;
    let refused = split::execute(&request(&source, &source), &dir.path().join("e.json"))
        .await
        .unwrap_err();
    assert!(
        matches!(refused, SplitError::HostIsSource { .. }),
        "{refused:?}"
    );
}
