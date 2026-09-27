//! Live-Postgres integration test for increment 3 (`akh-harvest`).
//!
//! Ignored by default; run against a throwaway database from
//! `scripts/pg-standalone.sh` (see `tests/live_pg.rs`'s header for the boot
//! dance) or the real `harvest-pg` reference deployment:
//!
//! ```text
//! AKH_TEST_TH_PG_URL='postgres://akh:akh@127.0.0.1:55432/transcripts?sslmode=disable' \
//!   cargo test -p akh-transcripts --test live_harvest -- --ignored
//! ```
//!
//! This test never touches real Claude Code / Codex / OpenCode session
//! stores — it points [`HarvestConfig`] at a synthetic fixture tree under a
//! tempdir, so it is safe to run against a shared database without
//! polluting anyone else's transcript rows (its `session_id` is unique to
//! this test).

use akh_transcripts::harvest::{HarvestConfig, RunOptions, run};
use akh_transcripts::pool::run_migrations_locked;

fn write_fixture_session(root: &std::path::Path) -> std::path::PathBuf {
    let slug = "live-harvest-test-project";
    let session_id = "11111111-1111-1111-1111-111111111111";
    let path = root.join(slug).join(format!("{session_id}.jsonl"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(
        &path,
        "{\"type\":\"user\",\"sessionId\":\"11111111-1111-1111-1111-111111111111\",\"timestamp\":\"2026-09-19T00:00:00Z\",\"message\":{\"content\":\"synthetic fixture, safe to insert repeatedly\"}}\n\
         {\"type\":\"assistant\",\"sessionId\":\"11111111-1111-1111-1111-111111111111\",\"timestamp\":\"2026-09-19T00:00:01Z\",\"message\":{\"content\":\"ack\"}}\n",
    )
    .unwrap();
    path
}

/// The core acceptance property from the increment-3 plan: two consecutive
/// runs over unchanged input insert once and then insert nothing. A third
/// run after deleting the watermark state re-scans from zero and still nets
/// zero new rows, because `ON CONFLICT DO NOTHING` dedupes the re-read.
#[tokio::test]
#[ignore = "requires AKH_TEST_TH_PG_URL (cnpg-timescale or plain image: vector required)"]
async fn second_run_over_unchanged_input_inserts_nothing() {
    let Ok(url) = std::env::var("AKH_TEST_TH_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_TH_PG_URL");
        return;
    };

    run_migrations_locked(&url).await.expect("migrate");

    let claude_root = tempfile::tempdir().expect("claude root tempdir");
    write_fixture_session(claude_root.path());
    let state_dir = tempfile::tempdir().expect("state dir tempdir");

    let cfg = HarvestConfig {
        database_url: url,
        machine: "live-harvest-test-runner".to_string(),
        state_dir: state_dir.path().to_path_buf(),
        claude_root: claude_root.path().to_path_buf(),
        // Nonexistent on purpose: a missing root is a warning, not a
        // failure (source-file-lifecycle.md), so these sources are
        // harmlessly skipped for this test.
        codex_root: claude_root.path().join("no-codex-here"),
        opencode_db: claude_root.path().join("no-opencode-here.db"),
    };
    let opts = RunOptions::default();

    let first = run(&cfg, &opts).await.expect("first run");
    assert_eq!(
        first.failed_streams, 0,
        "fixture must read cleanly: {first:?}"
    );
    assert_eq!(
        first.inserted, 2,
        "both fixture records should insert on a fresh run: {first:?}"
    );
    assert_eq!(first.quarantined, 0);

    let second = run(&cfg, &opts).await.expect("second run");
    assert_eq!(
        second.inserted, 0,
        "an unchanged source file, resumed from its watermark, must insert nothing: {second:?}"
    );
    assert_eq!(second.failed_streams, 0);

    // Delete watermark state and re-run: a full re-scan collides into
    // ON CONFLICT DO NOTHING rather than duplicating rows.
    std::fs::remove_dir_all(state_dir.path()).expect("remove state dir");
    let third = run(&cfg, &opts)
        .await
        .expect("third run after watermark loss");
    assert_eq!(
        third.inserted, 0,
        "losing the watermark must re-scan harmlessly, not duplicate rows: {third:?}"
    );
    assert_eq!(
        third.deduped, 2,
        "the full re-read should dedupe both rows: {third:?}"
    );
}
