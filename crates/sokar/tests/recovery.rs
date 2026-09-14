mod common;
use bytes::Bytes;
use common::*;
use futures::stream;
use sokar::{
    SokarError,
    index::{Index, PgIndex},
};
use sokar_client::*;
use std::sync::Arc;
#[tokio::test]
async fn reindex_rejects_future_schema_atomically_and_gc_protects_orphans() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let c = fixture.collection().await;
    let a = server
        .client
        .upload(c.id, &options("file"), "content")
        .await
        .unwrap()
        .artifact;
    let store = fixture.state.stores.get(&c.store).unwrap();
    let mut orphan = a.clone();
    orphan.id = ArtifactId::new();
    orphan.path = "orphan".parse().unwrap();
    store.write_json(&orphan.key(), &orphan).await.unwrap();
    server.client.delete_artifact(a.id).await.unwrap();
    assert!(store.op.exists(&a.blob.key).await.unwrap());
    let counts = fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(counts.artifacts, 1);
    assert_eq!(fixture.memory.artifact(orphan.id).await.unwrap(), orphan);
    let bad_id = ArtifactId::new();
    let key = format!("artifacts/{bad_id}/manifest.json");
    store
        .write_json(&key, &serde_json::json!({"schema":"sokar.artifact.v2"}))
        .await
        .unwrap();
    assert!(matches!(
        fixture.state.reindex(&c.store).await,
        Err(SokarError::Client(
            sokar_client::Error::ManifestSchema { .. }
        ))
    ));
    assert_eq!(fixture.memory.artifact(orphan.id).await.unwrap(), orphan);
    store.delete(&key).await.unwrap();
    store
        .write_json(
            sokar::store::FORMAT_KEY,
            &serde_json::json!({"schema":"sokar.layout.v2"}),
        )
        .await
        .unwrap();
    assert!(matches!(
        store.initialize().await,
        Err(SokarError::Client(
            sokar_client::Error::ManifestSchema { .. }
        ))
    ));
}
#[tokio::test]
async fn concurrent_idempotency_and_failed_staging_leave_no_uploads() {
    let fixture = Fixture::new(false).await;
    let server = fixture.serve().await;
    let c = fixture.collection().await;
    let mut opts = options("same");
    opts.idempotency_key = Some("concurrent".into());
    let (a, b) = tokio::join!(
        server.client.upload(c.id, &opts, "data"),
        server.client.upload(c.id, &opts, "data")
    );
    let a = a.unwrap();
    assert_eq!(a, b.unwrap());
    assert_eq!(
        fixture
            .memory
            .refcount(&c.store, a.artifact.blob.digest)
            .await,
        1
    );
    let store = fixture.state.stores.get(&c.store).unwrap();
    let input = stream::iter([
        Ok(Bytes::from_static(b"1234")),
        Err(SokarError::Invalid("broken stream".into())),
    ]);
    assert!(store.stage(input, 100).await.is_err());
    assert!(
        store
            .stage(stream::iter([Ok(Bytes::from_static(b"1234"))]), 3)
            .await
            .is_err()
    );
    let uploads = store
        .op
        .list_with("uploads/")
        .recursive(true)
        .await
        .unwrap();
    assert!(uploads.iter().all(|e| !e.metadata().is_file()));
}
#[tokio::test]
async fn postgres_drop_schema_reindex_fs_contract() {
    // The URL must point at a disposable Sokar test database. Never uses Anima.
    let Ok(url) = std::env::var("SOKAR_TEST_DATABASE_URL") else {
        eprintln!("SKIP: SOKAR_TEST_DATABASE_URL unset");
        return;
    };
    let index = Arc::new(PgIndex::connect(&url).await.unwrap());
    index.migrate().await.unwrap();
    let fixture = Fixture::with_index(true, Some(index.clone())).await;
    let server = fixture.serve().await;
    let c = server
        .client
        .create_collection(&create(&format!("pg-{}", uuid::Uuid::now_v7())))
        .await
        .unwrap();
    let a = server
        .client
        .upload(c.id, &options("a"), "postgres recovery")
        .await
        .unwrap()
        .artifact;
    let b = server
        .client
        .upload(c.id, &options("b"), "postgres recovery")
        .await
        .unwrap()
        .artifact;
    let mut opts = options("receipt");
    opts.idempotency_key = Some("pg-retry".into());
    let receipt = server.client.upload(c.id, &opts, "receipt").await.unwrap();
    fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(
        server.client.upload(c.id, &opts, "receipt").await.unwrap(),
        receipt
    );
    server
        .client
        .delete_artifact(receipt.artifact.id)
        .await
        .unwrap();
    sqlx::query("DROP SCHEMA sokar CASCADE")
        .execute(index.pool())
        .await
        .unwrap();
    index.migrate().await.unwrap();
    assert!(index.collection(c.id).await.is_err());
    let counts = fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!(
        (counts.collections, counts.artifacts, counts.blobs),
        (1, 2, 1)
    );
    assert_eq!(index.collection(c.id).await.unwrap(), c);
    assert_eq!(index.artifact(a.id).await.unwrap(), a);
    assert_eq!(index.artifact(b.id).await.unwrap(), b);
    let refs: i32 =
        sqlx::query_scalar("SELECT refcount FROM sokar.blob WHERE store=$1 AND digest=$2")
            .bind(c.store.as_str())
            .bind(a.blob.digest.0.as_slice())
            .fetch_one(index.pool())
            .await
            .unwrap();
    assert_eq!(refs, 2);
    assert_eq!(fixture.state.reindex(&c.store).await.unwrap().artifacts, 2);
    server.client.delete_artifact(a.id).await.unwrap();
    server.client.delete_artifact(b.id).await.unwrap();
    let store = fixture.state.stores.get(&c.store).unwrap();
    assert!(store.op.exists(&a.blob.key).await.unwrap());
    let refs: i32 =
        sqlx::query_scalar("SELECT refcount FROM sokar.blob WHERE store=$1 AND digest=$2")
            .bind(c.store.as_str())
            .bind(a.blob.digest.0.as_slice())
            .fetch_one(index.pool())
            .await
            .unwrap();
    assert_eq!(refs, 0);
    // Run additional backend contracts within this one disposable-schema test to
    // avoid parallel DROP SCHEMA interfering with another PostgreSQL test.
    let extra = Fixture::with_index(true, Some(index.clone())).await;
    receipts_and_tombstones(&extra).await;
    concurrent_store_transactions(index.as_ref()).await;
}
#[tokio::test]
#[ignore = "requires local MinIO; SOKAR_MINIO_{ENDPOINT,BUCKET,ACCESS_KEY,SECRET_KEY} environment references"]
async fn minio_s3_contract() {
    use sokar::{config::*, store::StoreRegistry};
    let config = StoreConfig {
        name: "minio".parse().unwrap(),
        service: ServiceKind::S3,
        prefix: Some(format!("sokar-contract-{}", uuid::Uuid::now_v7())),
        options: [
            ("endpoint", "SOKAR_MINIO_ENDPOINT"),
            ("bucket", "SOKAR_MINIO_BUCKET"),
            ("access_key_id", "SOKAR_MINIO_ACCESS_KEY"),
            ("secret_access_key", "SOKAR_MINIO_SECRET_KEY"),
            ("region", "SOKAR_MINIO_REGION"),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), SecretRef::Env(value.into())))
        .collect(),
    };
    let stores = StoreRegistry::build(&[config]).await.unwrap();
    stores.check(true).await.unwrap();
    let store = stores.get(&"minio".parse().unwrap()).unwrap();
    assert!(store.cap.presign_read);
    let mut a = store
        .stage(stream::iter([Ok(Bytes::from_static(b"s3 contract"))]), 100)
        .await
        .unwrap();
    assert_eq!(a.promote().await.unwrap(), UploadOutcome::Stored);
    a.cleanup().await.unwrap();
    let mut b = store
        .stage(stream::iter([Ok(Bytes::from_static(b"s3 contract"))]), 100)
        .await
        .unwrap();
    assert_eq!(b.promote().await.unwrap(), UploadOutcome::Deduplicated);
    b.cleanup().await.unwrap();
    assert_eq!(
        store.op.read(&a.digest.key()).await.unwrap().to_bytes(),
        "s3 contract"
    );
    store.delete(&a.digest.key()).await.unwrap();
    store.delete(sokar::store::FORMAT_KEY).await.unwrap();
}

#[tokio::test]
async fn identical_content_in_two_stores_has_independent_refcounts_and_reindex() {
    use sokar::{config::*, store::StoreRegistry};
    let fixture = Fixture::new(false).await;
    let mut config = (*fixture.config).clone();
    config.stores.push(StoreConfig {
        name: "second".parse().unwrap(),
        service: ServiceKind::Memory,
        prefix: None,
        options: Default::default(),
    });
    let stores = Arc::new(StoreRegistry::build(&config.stores).await.unwrap());
    let state = sokar::Sokar::new(Arc::new(config), stores.clone(), fixture.memory.clone());
    let a = state
        .create_collection("test", create("first"))
        .await
        .unwrap();
    let mut input = create("second");
    input.store = Some("second".parse().unwrap());
    let b = state.create_collection("test", input).await.unwrap();
    let mut artifacts = Vec::new();
    for c in [&a, &b] {
        let store = stores.get(&c.store).unwrap();
        let stage = store
            .stage(
                stream::iter([Ok(Bytes::from_static(b"same across stores"))]),
                100,
            )
            .await
            .unwrap();
        let result = state
            .finish_upload("test", c, options("same"), stage)
            .await
            .unwrap();
        assert_eq!(result.outcome, UploadOutcome::Stored);
        artifacts.push(result.artifact);
    }
    assert_eq!(artifacts[0].blob.digest, artifacts[1].blob.digest);
    state.reindex(&a.store).await.unwrap();
    assert_eq!(
        fixture.memory.artifact(artifacts[1].id).await.unwrap(),
        artifacts[1]
    );
    // An imported UUID must not move a collection belonging to another store.
    let mut collision = b.clone();
    collision.store = a.store.clone();
    stores
        .get(&a.store)
        .unwrap()
        .write_json(&collision.key(), &collision)
        .await
        .unwrap();
    assert!(matches!(
        state.reindex(&a.store).await,
        Err(SokarError::StoreMismatch { .. })
    ));
    assert_eq!(fixture.memory.collection(b.id).await.unwrap(), b);
    stores
        .get(&a.store)
        .unwrap()
        .delete(&collision.key())
        .await
        .unwrap();
    state.delete_artifact(artifacts[0].id).await.unwrap();
    assert!(
        stores
            .get(&b.store)
            .unwrap()
            .op
            .exists(&artifacts[1].blob.key)
            .await
            .unwrap()
    );
    assert_eq!(
        fixture
            .memory
            .refcount(&b.store, artifacts[1].blob.digest)
            .await,
        1
    );
}

#[tokio::test]
async fn duplicate_scope_manifests_choose_earliest_and_count_orphans() {
    let fixture = Fixture::new(false).await;
    let c = fixture.collection().await;
    let store = fixture.state.stores.get(&c.store).unwrap();
    let mut duplicate = c.clone();
    duplicate.id = CollectionId::new();
    assert!(c.id < duplicate.id);
    // Reverse write order exercises independence from listing/insertion order.
    store
        .write_json(&duplicate.key(), &duplicate)
        .await
        .unwrap();
    store.write_json(&c.key(), &c).await.unwrap();
    fixture.memory.clear().await;
    for _ in 0..2 {
        let counts = fixture.state.reindex(&c.store).await.unwrap();
        assert_eq!(
            (counts.collections, counts.artifacts, counts.orphans),
            (1, 0, 1)
        );
        assert_eq!(fixture.memory.by_scope(&c.scope.key()).await.unwrap(), c);
        assert!(matches!(
            fixture.memory.collection(duplicate.id).await,
            Err(SokarError::NotFound)
        ));
        // Recovery never deletes the losing portable manifest.
        assert!(store.op.exists(&duplicate.key()).await.unwrap());
    }
}

async fn receipts_and_tombstones(fixture: &Fixture) {
    use sokar::index::MutationScope;
    let server = fixture.serve().await;
    let c = fixture
        .state
        .create_collection("test-service", create("delete-replay"))
        .await
        .unwrap();
    let store = fixture.state.stores.get(&c.store).unwrap();
    let mut opts = options("replay");
    opts.idempotency_key = Some("delete-replay-key".into());
    let first = server.client.upload(c.id, &opts, "original").await.unwrap();
    // Capture a receipt to simulate a stale pre-upgrade cache later.
    let mut tx = fixture
        .state
        .index
        .begin(MutationScope::Store(c.store.clone()))
        .await
        .unwrap();
    let old = tx
        .idempotency("test-service", c.id, "delete-replay-key")
        .await
        .unwrap()
        .unwrap();
    drop(tx);
    server
        .client
        .delete_artifact(first.artifact.id)
        .await
        .unwrap();
    let mut tx = fixture
        .state
        .index
        .begin(MutationScope::Store(c.store.clone()))
        .await
        .unwrap();
    assert!(
        tx.idempotency("test-service", c.id, "delete-replay-key")
            .await
            .unwrap()
            .is_none()
    );
    drop(tx);
    assert!(store.op.exists(&first.artifact.blob.key).await.unwrap());
    let tombstone: Artifact = store.read_json(&first.artifact.key()).await.unwrap();
    assert!(tombstone.deleted_at.is_some());
    fixture.state.reindex(&c.store).await.unwrap();
    assert!(matches!(
        fixture.state.index.artifact(first.artifact.id).await,
        Err(SokarError::NotFound)
    ));
    let second = server.client.upload(c.id, &opts, "original").await.unwrap();
    assert_ne!(first.artifact.id, second.artifact.id);
    assert_eq!(second.outcome, UploadOutcome::Deduplicated);
    server
        .client
        .delete_artifact(second.artifact.id)
        .await
        .unwrap();
    // Even a receipt with a different fingerprint is ignored when its artifact is gone.
    let mut tx = fixture
        .state
        .index
        .begin(MutationScope::Store(c.store.clone()))
        .await
        .unwrap();
    tx.put_idempotency(&old).await.unwrap();
    tx.commit().await.unwrap();
    let third = server
        .client
        .upload(c.id, &opts, "different")
        .await
        .unwrap();
    assert_ne!(third.artifact.id, first.artifact.id);
    assert_eq!(
        server
            .client
            .upload(c.id, &opts, "different")
            .await
            .unwrap(),
        third
    );
    let conflict = server
        .client
        .upload(c.id, &opts, "mismatch")
        .await
        .unwrap_err();
    assert_eq!(
        conflict.problem().unwrap().code,
        "sokar::idempotency_mismatch"
    );
    let manifests = store
        .op
        .list_with("artifacts/")
        .recursive(true)
        .await
        .unwrap();
    assert_eq!(
        manifests
            .iter()
            .filter(|e| e.path().ends_with("/manifest.json"))
            .count(),
        3
    );
    let uploads = store
        .op
        .list_with("uploads/")
        .recursive(true)
        .await
        .unwrap();
    assert!(uploads.iter().all(|e| !e.metadata().is_file()));
    // Simulate future janitor collection of bytes referenced only by tombstones.
    store.delete(&first.artifact.blob.key).await.unwrap();
    let counts = fixture.state.reindex(&c.store).await.unwrap();
    assert_eq!((counts.artifacts, counts.blobs), (1, 1));
    assert_eq!(
        fixture
            .state
            .index
            .artifact(third.artifact.id)
            .await
            .unwrap(),
        third.artifact
    );
}

#[tokio::test]
async fn replay_after_delete_purges_receipts_and_restores_tombstones() {
    receipts_and_tombstones(&Fixture::new(false).await).await;
}

async fn concurrent_store_transactions(index: &dyn Index) {
    use sokar::index::MutationScope;
    use std::time::Duration;
    let store_a: StoreName = "concurrent-a".parse().unwrap();
    let store_b: StoreName = "concurrent-b".parse().unwrap();
    let collection = |store: StoreName| Collection {
        schema: Collection::SCHEMA.into(),
        id: CollectionId::new(),
        scope: CollectionScope::Named {
            name: format!("scope-{}", uuid::Uuid::now_v7()),
        },
        owner: "test".into(),
        store,
        retention: RetentionPolicy::Forever,
        quota: Quota::default(),
        created_at: time::OffsetDateTime::now_utc(),
    };
    let a = collection(store_a.clone());
    let b = collection(store_b.clone());
    let mut global = index.begin(MutationScope::Global).await.unwrap();
    global.put_collection(&a).await.unwrap();
    global.put_collection(&b).await.unwrap();
    global.commit().await.unwrap();
    // Both implementations must expose the same typed scope collision.
    let mut collision = a.clone();
    collision.id = CollectionId::new();
    let mut global = index.begin(MutationScope::Global).await.unwrap();
    assert!(matches!(
        global.put_collection(&collision).await,
        Err(SokarError::ScopeExists(_))
    ));
    drop(global);
    let artifact = |c: &Collection| Artifact {
        schema: Artifact::SCHEMA.into(),
        id: ArtifactId::new(),
        collection: c.id,
        path: "concurrent".parse().unwrap(),
        kind: ArtifactKind::Document,
        blob: BlobRef {
            digest: BlobDigest([8; 32]),
            size: 1,
            media_type: "text/plain".parse().unwrap(),
            key: BlobDigest([8; 32]).key(),
        },
        labels: Labels::new(),
        created_by: "test".into(),
        created_at: c.created_at,
        expires_at: None,
        previous: None,
        deleted_at: None,
    };
    let aa = artifact(&a);
    let bb = artifact(&b);
    let mut first = index
        .begin(MutationScope::Store(store_a.clone()))
        .await
        .unwrap();
    first.put_artifact(&store_a, &aa).await.unwrap();
    // This must acquire, mutate and commit while A is still held, not merely start.
    tokio::time::timeout(Duration::from_secs(2), async {
        let mut second = index
            .begin(MutationScope::Store(store_b.clone()))
            .await
            .unwrap();
        second.put_artifact(&store_b, &bb).await.unwrap();
        second.commit().await.unwrap();
    })
    .await
    .expect("independent store blocked on first store");
    assert!(
        tokio::time::timeout(
            Duration::from_millis(50),
            index.begin(MutationScope::Store(store_a.clone()))
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(
            Duration::from_millis(50),
            index.begin(MutationScope::Global)
        )
        .await
        .is_err()
    );
    first.commit().await.unwrap();
    assert_eq!(index.artifact(aa.id).await.unwrap().id, aa.id);
    assert_eq!(index.artifact(bb.id).await.unwrap().id, bb.id);
    let global = index.begin(MutationScope::Global).await.unwrap();
    assert!(
        tokio::time::timeout(
            Duration::from_millis(50),
            index.begin(MutationScope::Store(store_b))
        )
        .await
        .is_err()
    );
    drop(global);
}

#[tokio::test]
async fn two_memory_stores_mutate_concurrently_without_lost_commits() {
    concurrent_store_transactions(&sokar::index::MemoryIndex::default()).await;
}
