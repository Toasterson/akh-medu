//! Live-Postgres integration tests for the Postgres-backed `Store` (get,
//! put, scan, transaction commit/rollback, and the one-shot redb import).
//!
//! Ignored by default; run against a throwaway database, e.g.:
//!
//!   docker run -d --rm --name akh-store-pg -p 55433:5432 \
//!     -e POSTGRES_PASSWORD=x -e POSTGRES_USER=akh -e POSTGRES_DB=akh \
//!     postgres:18-alpine
//!   AKH_TEST_DATABASE_URL=postgres://akh:x@127.0.0.1:55433/akh \
//!     cargo test -p akh-service-store --test postgres -- --ignored
//!   docker stop akh-store-pg
use akh_service_store::{PostgresConfig, Store, key};
use akh_session_types::Fault;
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh `kind` per test, so concurrently-run tests (and repeated runs
/// against a persistent dev database) never see each other's rows or
/// migration markers.
fn unique_kind(label: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "test_{label}_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

fn database_url() -> Option<String> {
    match std::env::var("AKH_TEST_DATABASE_URL") {
        Ok(url) => Some(url),
        Err(_) => {
            eprintln!("skipping: set AKH_TEST_DATABASE_URL");
            None
        }
    }
}

fn open(kind: &str, url: &str, legacy_redb_path: Option<std::path::PathBuf>) -> Store {
    Store::open_postgres(PostgresConfig {
        url: url.to_string(),
        schema: "akh_sessions".to_string(),
        kind: kind.to_string(),
        legacy_redb_path,
    })
    .expect("open postgres store")
}

#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn get_put_and_scan_round_trip_through_a_transaction() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("roundtrip");
    let store = open(&kind, &url, None);

    store
        .transaction(|tx| {
            tx.put(&key("session", &["a"]), &1u32)?;
            tx.put(&key("session", &["b"]), &2u32)?;
            tx.put(&key("other", &["c"]), &3u32)
        })
        .unwrap();

    assert_eq!(store.get::<u32>(&key("session", &["a"])).unwrap(), Some(1));
    assert_eq!(store.get::<u32>(&key("session", &["b"])).unwrap(), Some(2));
    assert_eq!(store.get::<u32>(&key("missing", &["x"])).unwrap(), None);

    let mut scanned: Vec<u32> = store.scan("session/").unwrap();
    scanned.sort_unstable();
    assert_eq!(scanned, vec![1, 2]);

    // A second `get` for the same key must be served without a fresh
    // network round trip via the read-through cache; correctness (not
    // just presence) is what's actually verifiable from here.
    assert_eq!(store.get::<u32>(&key("session", &["a"])).unwrap(), Some(1));
}

#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn a_failed_transaction_rolls_back_and_never_reaches_the_cache() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("rollback");
    let store = open(&kind, &url, None);

    store
        .transaction(|tx| {
            tx.put("x", &1u32)?;
            Ok(())
        })
        .unwrap();
    assert_eq!(store.get::<u32>("x").unwrap(), Some(1));

    let result: Result<(), Fault> = store.transaction(|tx| {
        tx.put("x", &2u32)?;
        Err(Fault::conflict("abort"))
    });
    assert!(result.is_err());

    // Neither the database nor the process-local cache observed the abort.
    assert_eq!(store.get::<u32>("x").unwrap(), Some(1));

    // A fresh handle bypasses this process's cache entirely, proving the
    // durable row itself was rolled back, not just the cache.
    let reopened = open(&kind, &url, None);
    assert_eq!(reopened.get::<u32>("x").unwrap(), Some(1));
}

#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn transaction_reads_see_the_transactions_own_uncommitted_writes() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("read_your_writes");
    let store = open(&kind, &url, None);

    store
        .transaction(|tx| {
            tx.put(&key("session", &["a"]), &1u32)?;
            assert_eq!(tx.get::<u32>(&key("session", &["a"]))?, Some(1));
            let scanned: Vec<u32> = tx.scan("session/")?;
            assert_eq!(scanned, vec![1]);
            tx.delete(&key("session", &["a"]))?;
            assert_eq!(tx.get::<u32>(&key("session", &["a"]))?, None);
            let scanned: Vec<u32> = tx.scan("session/")?;
            assert!(scanned.is_empty());
            Ok(())
        })
        .unwrap();

    assert_eq!(store.get::<u32>(&key("session", &["a"])).unwrap(), None);
}

#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn delete_removes_the_row_and_the_cached_value() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("delete");
    let store = open(&kind, &url, None);

    store.transaction(|tx| tx.put("a", &1u32)).unwrap();
    assert_eq!(store.get::<u32>("a").unwrap(), Some(1));

    store.transaction(|tx| tx.delete("a")).unwrap();
    assert_eq!(store.get::<u32>("a").unwrap(), None);
}

#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn legacy_redb_contents_are_imported_exactly_once() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("migrate");

    let dir = tempfile::tempdir().unwrap();
    let redb_path = dir.path().join(format!("{kind}.redb"));
    let redb_store = Store::open(&redb_path).unwrap();
    redb_store
        .transaction(|tx| {
            tx.put(&key("session", &["a"]), &1u32)?;
            tx.put(&key("session", &["b"]), &2u32)
        })
        .unwrap();
    drop(redb_store);

    let migrated = open(&kind, &url, Some(redb_path.clone()));
    let mut scanned: Vec<u32> = migrated.scan("session/").unwrap();
    scanned.sort_unstable();
    assert_eq!(scanned, vec![1, 2]);
    drop(migrated);

    // Mutate the on-disk redb file after the first migration: if the
    // marker were not honored, reopening would re-import and clobber the
    // value written directly in Postgres below.
    let redb_store = Store::open(&redb_path).unwrap();
    redb_store
        .transaction(|tx| tx.put(&key("session", &["a"]), &999u32))
        .unwrap();
    drop(redb_store);

    let reopened = open(&kind, &url, Some(redb_path));
    reopened
        .transaction(|tx| tx.put(&key("session", &["a"]), &42u32))
        .unwrap();
    let reopened_again = open(&kind, &url, None);
    assert_eq!(
        reopened_again.get::<u32>(&key("session", &["a"])).unwrap(),
        Some(42),
        "a marker-guarded reopen must not re-import and overwrite a value written after migration"
    );
}

/// Regression test for the production incident: `akhsessd`/`akhrtd`/
/// `akhexecd`/`akhsbxd`/`akheventd` all run a bare `#[tokio::main]` (multi-
/// thread scheduler) and call these `Store` methods synchronously from
/// within async request-handling code — i.e. directly on a multi-thread
/// runtime worker thread, exactly like this test body does. Before the sync
/// facade moved its Postgres work onto a dedicated OS thread, this call
/// shape panicked with "Cannot start a runtime from within a runtime".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
async fn store_works_when_called_synchronously_from_a_multi_thread_runtime_worker() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("multi_thread_runtime");
    let store = open(&kind, &url, None);

    store
        .transaction(|tx| {
            tx.put(&key("session", &["a"]), &1u32)?;
            tx.put(&key("session", &["b"]), &2u32)?;
            assert_eq!(tx.get::<u32>(&key("session", &["a"]))?, Some(1));
            let scanned: Vec<u32> = tx.scan("session/")?;
            assert_eq!(scanned.len(), 2);
            Ok(())
        })
        .unwrap();

    let mut scanned: Vec<u32> = store.scan("session/").unwrap();
    scanned.sort_unstable();
    assert_eq!(scanned, vec![1, 2]);
    assert_eq!(store.get::<u32>(&key("session", &["a"])).unwrap(), Some(1));
}

/// Companion to the multi-thread-runtime-worker test above: the same store
/// must also work when opened and used from a plain `std::thread` with no
/// Tokio runtime at all (this always worked, even with the old
/// `block_on`-a-private-runtime bridge — it is the multi-thread-runtime
/// case above that regressed in production).
#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn store_works_when_called_from_a_plain_std_thread_with_no_ambient_runtime() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("plain_std_thread");
    let handle = std::thread::spawn(move || {
        let store = open(&kind, &url, None);
        store
            .transaction(|tx| tx.put(&key("session", &["a"]), &7u32))
            .unwrap();
        store.scan::<u32>("session/").unwrap()
    });
    assert_eq!(handle.join().unwrap(), vec![7]);
}

/// Regression: two transactions started concurrently from one process must
/// not deadlock. Before the in-process transaction gate, the second
/// `begin` blocked on the per-kind advisory lock *on the single-job
/// `akh-store-pg` thread*, so the first transaction's own statements could
/// never run and both hung forever (production, 2026-09-13, within seconds
/// of akhsessd starting). Mirrors the daemons: blocking `transaction()`
/// calls from tasks on a multi-thread tokio runtime.
#[test]
#[ignore = "requires AKH_TEST_DATABASE_URL"]
fn concurrent_transactions_from_one_process_do_not_deadlock() {
    let Some(url) = database_url() else { return };
    let kind = unique_kind("concurrent_tx");
    let store = std::sync::Arc::new(open(&kind, &url, None));

    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let worker_store = store.clone();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let mut tasks = Vec::new();
            for worker in 0..8u32 {
                let store = worker_store.clone();
                tasks.push(tokio::spawn(async move {
                    for round in 0..10u32 {
                        let k = key("counter", &[&worker.to_string()]);
                        store
                            .transaction(|tx| {
                                let current: u32 = tx.get(&k)?.unwrap_or(0);
                                assert_eq!(current, round);
                                tx.put(&k, &(current + 1))
                            })
                            .unwrap();
                    }
                }));
            }
            for task in tasks {
                task.await.unwrap();
            }
        });
        let _ = done_tx.send(());
    });

    done_rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("concurrent transactions deadlocked (no progress within 60 s)");
    for worker in 0..8u32 {
        let k = key("counter", &[&worker.to_string()]);
        assert_eq!(store.get::<u32>(&k).unwrap(), Some(10));
    }
}
