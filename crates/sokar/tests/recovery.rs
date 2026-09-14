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
        Err(SokarError::ManifestSchema { .. })
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
        Err(SokarError::ManifestSchema { .. })
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
    assert!(!store.op.exists(&a.blob.key).await.unwrap());
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
        Err(SokarError::Conflict(_))
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
