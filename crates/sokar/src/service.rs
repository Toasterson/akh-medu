use crate::{
    SokarError, SokarResult,
    authz::{Authorizer, ServiceOnly},
    config::Config,
    index::{Idempotency, Index, ReindexCounts},
    store::{StagedUpload, StoreRegistry},
};
use futures::TryStreamExt;
use sha2::{Digest, Sha256};
use sokar_client::*;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use time::OffsetDateTime;

pub struct Sokar {
    pub config: Arc<Config>,
    pub stores: Arc<StoreRegistry>,
    pub index: Arc<dyn Index>,
    pub authorizer: Arc<dyn Authorizer>,
}
impl Sokar {
    pub fn new(config: Arc<Config>, stores: Arc<StoreRegistry>, index: Arc<dyn Index>) -> Self {
        Self {
            config,
            stores,
            index,
            authorizer: Arc::new(ServiceOnly),
        }
    }
    pub async fn create_collection(
        &self,
        owner: &str,
        input: CreateCollection,
    ) -> SokarResult<Collection> {
        let store = input
            .store
            .unwrap_or_else(|| self.config.default_store.clone());
        let storage = self.stores.get(&store)?;
        let mut tx = self.index.begin().await?;
        if tx.by_scope(&input.scope.key()).await?.is_some() {
            return Err(SokarError::Conflict(
                "collection scope already exists".into(),
            ));
        }
        let c = Collection {
            schema: Collection::SCHEMA.into(),
            id: CollectionId::new(),
            scope: input.scope,
            owner: owner.into(),
            store,
            retention: input.retention,
            quota: input.quota,
            created_at: now(),
        };
        // Bucket manifests are truth: write manifest BEFORE the index transaction.
        storage.write_json(&c.key(), &c).await?;
        tx.put_collection(&c).await?;
        tx.commit().await?;
        Ok(c)
    }
    pub async fn finish_upload(
        &self,
        principal: &str,
        collection: &Collection,
        options: UploadOptions,
        mut staged: StagedUpload,
    ) -> SokarResult<UploadResponse> {
        if staged.store.name != collection.store {
            return Err(SokarError::Conflict(
                "staging store differs from collection".into(),
            ));
        }
        let fingerprint = hex::encode(Sha256::digest(serde_json::to_vec(&(
            collection.id,
            &options.path,
            &options.media_type,
            options.kind,
            &options.labels,
            staged.digest,
            staged.size,
        ))?));
        let mut tx = self.index.begin().await?;
        if let Some(key) = &options.idempotency_key {
            if key.is_empty() || key.len() > 256 {
                return Err(SokarError::Invalid(
                    "idempotency key must be 1..256 bytes".into(),
                ));
            }
            if let Some(old) = tx.idempotency(principal, collection.id, key).await? {
                if old.fingerprint != fingerprint {
                    return Err(SokarError::Conflict(
                        "idempotency key reused with different content or metadata".into(),
                    ));
                }
                tx.artifact(old.response.artifact.id).await?;
                staged.cleanup().await?;
                return Ok(old.response);
            }
        }
        let previous = tx.latest(collection.id, &options.path).await?;
        let created_at = previous.as_ref().map_or_else(now, |a| {
            now().max(a.created_at + time::Duration::microseconds(1))
        });
        let artifact = Artifact {
            schema: Artifact::SCHEMA.into(),
            id: ArtifactId::new(),
            collection: collection.id,
            path: options.path,
            kind: options.kind,
            blob: BlobRef {
                digest: staged.digest,
                size: staged.size,
                media_type: options.media_type,
                key: staged.digest.key(),
            },
            labels: options.labels,
            created_by: principal.into(),
            created_at,
            expires_at: None,
            previous: previous.map(|a| a.id),
        };
        let outcome = staged.promote().await?;
        staged.cleanup().await?;
        staged.store.write_json(&artifact.key(), &artifact).await?;
        tx.put_artifact(&collection.store, &artifact).await?;
        let response = UploadResponse { artifact, outcome };
        if let Some(key) = options.idempotency_key {
            tx.put_idempotency(&Idempotency {
                principal: principal.into(),
                collection: collection.id,
                key,
                fingerprint,
                response: response.clone(),
                expires_at: now() + time::Duration::hours(24),
            })
            .await?;
        }
        tx.commit().await?;
        Ok(response)
    }
    pub async fn delete_artifact(&self, id: ArtifactId) -> SokarResult<()> {
        let artifact = self.index.artifact(id).await?;
        let collection = self.index.collection(artifact.collection).await?;
        let store = self.stores.get(&collection.store)?;
        let mut tx = self.index.begin().await?;
        let remaining = tx.delete_artifact(&collection.store, id).await?;
        store.delete(&artifact.key()).await?;
        // Synchronous 52a GC, under the same index transaction/mutation lock.
        // An orphan manifest from an earlier failed index commit must protect its bytes.
        if remaining == 0 {
            let manifests = artifact_manifests(&store).await?;
            if !manifests
                .iter()
                .any(|a| a.blob.digest == artifact.blob.digest)
            {
                store.delete(&artifact.blob.key).await?;
            }
        }
        tx.commit().await?;
        Ok(())
    }
    pub async fn reindex(&self, name: &StoreName) -> SokarResult<ReindexCounts> {
        let store = self.stores.get(name)?;
        let mut tx = self.index.begin().await?;
        store.check(false).await?;
        let mut collections = Vec::new();
        let mut lister = store
            .op
            .lister_with("collections/")
            .recursive(true)
            .await
            .map_err(|e| store.error("list collections", e))?;
        while let Some(entry) = lister
            .try_next()
            .await
            .map_err(|e| store.error("list collections", e))?
        {
            if !entry.path().ends_with("/manifest.json") {
                continue;
            }
            let c: Collection = store
                .read_manifest(entry.path(), Collection::SCHEMA)
                .await?;
            c.validate()?;
            if c.key() != entry.path() || &c.store != name {
                return Err(SokarError::Invalid(
                    "collection manifest key/store mismatch".into(),
                ));
            }
            collections.push(c);
        }
        let artifacts = artifact_manifests(&store).await?;
        let ids: HashSet<_> = collections.iter().map(|c| c.id).collect();
        let mut blobs = HashMap::new();
        for a in &artifacts {
            if !ids.contains(&a.collection) {
                return Err(SokarError::Invalid(
                    "artifact references a missing collection manifest".into(),
                ));
            }
            if let Some(size) = blobs.insert(a.blob.digest, a.blob.size) {
                if size != a.blob.size {
                    return Err(SokarError::Conflict(
                        "manifests disagree on blob size".into(),
                    ));
                }
            } else {
                let metadata = store
                    .op
                    .stat(&a.blob.key)
                    .await
                    .map_err(|e| store.error("stat reindex blob", e))?;
                if metadata.content_length() != a.blob.size {
                    return Err(SokarError::Conflict(
                        "blob size differs from manifest".into(),
                    ));
                }
            }
        }
        let counts = tx.replace_store(name, &collections, &artifacts).await?;
        tx.commit().await?;
        Ok(counts)
    }
}
async fn artifact_manifests(store: &crate::store::Store) -> SokarResult<Vec<Artifact>> {
    let mut result = Vec::new();
    let mut lister = store
        .op
        .lister_with("artifacts/")
        .recursive(true)
        .await
        .map_err(|e| store.error("list artifacts", e))?;
    while let Some(entry) = lister
        .try_next()
        .await
        .map_err(|e| store.error("list artifacts", e))?
    {
        if !entry.path().ends_with("/manifest.json") {
            continue;
        }
        let a: Artifact = store.read_manifest(entry.path(), Artifact::SCHEMA).await?;
        a.validate()?;
        if a.key() != entry.path() {
            return Err(SokarError::Invalid("artifact manifest key mismatch".into()));
        }
        result.push(a);
    }
    Ok(result)
}
/// PostgreSQL stores microseconds; keep JSON manifests and keyset cursors identical.
fn now() -> OffsetDateTime {
    let value = OffsetDateTime::now_utc();
    value
        .replace_nanosecond(value.nanosecond() / 1000 * 1000)
        .expect("valid nanoseconds")
}
