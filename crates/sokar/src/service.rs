use crate::{
    SokarError, SokarResult,
    authz::{Authorizer, ServiceOnly},
    config::Config,
    index::{Idempotency, Index, MutationScope, ReindexCounts},
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
        Ok(self.create_collection_outcome(owner, input).await?.0)
    }
    /// The boolean distinguishes a newly created collection (201) from replay (200).
    pub async fn create_collection_outcome(
        &self,
        owner: &str,
        input: CreateCollection,
    ) -> SokarResult<(Collection, bool)> {
        let mut tx = self.index.begin(MutationScope::Global).await?;
        if let Some(existing) = tx.by_scope(&input.scope.key()).await? {
            if input
                .store
                .as_ref()
                .is_none_or(|store| store == &existing.store)
            {
                return Ok((existing, false));
            }
            return Err(SokarError::ScopeExists(input.scope.key()));
        }
        let store = input
            .store
            .unwrap_or_else(|| self.config.default_store.clone());
        let storage = self.stores.get(&store)?;
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
        // Bucket manifests are truth: write manifest BEFORE inserting the index row.
        storage.write_json(&c.key(), &c).await?;
        tx.put_collection(&c).await?;
        tx.commit().await?;
        Ok((c, true))
    }
    pub async fn finish_upload(
        &self,
        principal: &str,
        collection: &Collection,
        options: UploadOptions,
        mut staged: StagedUpload,
    ) -> SokarResult<UploadResponse> {
        // Every early return runs explicit cleanup, and cleanup failure must never
        // replace the original upload error. Drop remains a cancellation fallback.
        let result = self
            .finish_staged(principal, collection, options, &mut staged)
            .await;
        if let Err(cleanup) = staged.cleanup().await {
            tracing::warn!(error = %cleanup, key = %staged.key, "staging cleanup failed");
        }
        result
    }
    async fn finish_staged(
        &self,
        principal: &str,
        collection: &Collection,
        options: UploadOptions,
        staged: &mut StagedUpload,
    ) -> SokarResult<UploadResponse> {
        // Reject malformed keys before any index access or mutation lock.
        if options
            .idempotency_key
            .as_ref()
            .is_some_and(|key| key.is_empty() || key.len() > 256)
        {
            return Err(SokarError::Invalid(
                "idempotency key must be 1..256 bytes".into(),
            ));
        }
        if staged.store.name != collection.store {
            return Err(SokarError::StoreMismatch {
                expected: collection.store.clone(),
                actual: staged.store.name.clone(),
            });
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
        let outcome = staged.promote().await?;
        let mut previous = match self.index.latest(collection.id, &options.path).await {
            Ok(a) => Some(a),
            Err(SokarError::NotFound) => None,
            Err(e) => return Err(e),
        };
        let mut artifact = Artifact {
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
            created_at: now(),
            expires_at: None,
            previous: None,
            deleted_at: None,
        };
        loop {
            artifact.previous = previous.as_ref().map(|a| a.id);
            artifact.created_at = previous.as_ref().map_or_else(now, |a| {
                now().max(a.created_at + time::Duration::microseconds(1))
            });
            // Bytes -> manifest -> row. No bucket I/O while the index lock is held.
            staged.store.write_json(&artifact.key(), &artifact).await?;
            let mut tx = self
                .index
                .begin(MutationScope::Store(collection.store.clone()))
                .await?;
            if let Some(key) = &options.idempotency_key
                && let Some(old) = tx.idempotency(principal, collection.id, key).await?
            {
                match tx.artifact(old.response.artifact.id).await {
                    Ok(_) => {
                        drop(tx);
                        // A receipt wins: remove only our unused manifest, never bytes.
                        let cleanup_manifest = staged.store.delete(&artifact.key()).await;
                        if old.fingerprint != fingerprint {
                            if let Err(e) = cleanup_manifest {
                                tracing::warn!(error = %e, key = %artifact.key(), "unused manifest cleanup failed");
                            }
                            if let Err(e) = staged.cleanup().await {
                                tracing::warn!(error = %e, key = %staged.key, "staging cleanup failed");
                            }
                            return Err(SokarError::IdempotencyMismatch);
                        }
                        cleanup_manifest?;
                        return Ok(old.response);
                    }
                    Err(SokarError::NotFound) => {
                        tx.purge_idempotency_for(old.response.artifact.id).await?;
                    }
                    Err(e) => return Err(e),
                }
            }
            let latest = tx.latest(collection.id, &artifact.path).await?;
            if latest.as_ref().map(|a| a.id) != artifact.previous {
                // Another finalizer won. Release the lock before rewriting the
                // manifest with its predecessor and microsecond-monotonic timestamp.
                drop(tx);
                previous = latest;
                continue;
            }
            tx.put_artifact(&collection.store, &artifact).await?;
            let response = UploadResponse { artifact, outcome };
            if let Some(key) = &options.idempotency_key {
                tx.put_idempotency(&Idempotency {
                    principal: principal.into(),
                    collection: collection.id,
                    key: key.clone(),
                    fingerprint,
                    response: response.clone(),
                    expires_at: now() + time::Duration::hours(24),
                })
                .await?;
            }
            tx.commit().await?;
            return Ok(response);
        }
    }
    /// Bounded full-content read. Never silently truncate, including if backend
    /// bytes have drifted from their manifest size. Check metadata before I/O.
    pub async fn read_inline(
        &self,
        id: ArtifactId,
        limit: usize,
    ) -> SokarResult<(Artifact, Vec<u8>)> {
        let artifact = self.index.artifact(id).await?;
        if artifact.blob.size > limit as u64 {
            return Err(SokarError::InlineLimit { limit });
        }
        let collection = self.index.collection(artifact.collection).await?;
        let store = self.stores.get(&collection.store)?;
        let reader = store
            .op
            .reader(&artifact.blob.key)
            .await
            .map_err(|e| store.error("open inline content", e))?;
        let mut stream = reader
            .into_bytes_stream(..)
            .await
            .map_err(|e| store.error("stream inline content", e))?;
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.try_next().await.map_err(SokarError::Io)? {
            if chunk.len() > limit.saturating_sub(bytes.len()) {
                return Err(SokarError::InlineLimit { limit });
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.len() as u64 != artifact.blob.size {
            return Err(SokarError::Conflict(
                "blob size differs from manifest".into(),
            ));
        }
        Ok((artifact, bytes))
    }
    pub async fn delete_artifact(&self, id: ArtifactId) -> SokarResult<()> {
        let mut artifact = self.index.artifact(id).await?;
        let collection = self.index.collection(artifact.collection).await?;
        let store = self.stores.get(&collection.store)?;
        artifact.deleted_at = Some(now());
        // Durable tombstone first. Manifest and bytes survive commit failure and
        // are reconciled by reindex; physical GC belongs exclusively to 52b.
        store.write_json(&artifact.key(), &artifact).await?;
        let mut tx = self
            .index
            .begin(MutationScope::Store(collection.store.clone()))
            .await?;
        tx.delete_artifact(&collection.store, &artifact).await?;
        tx.purge_idempotency_for(id).await?;
        tx.commit().await?;
        Ok(())
    }
    pub async fn reindex(&self, name: &StoreName) -> SokarResult<ReindexCounts> {
        let store = self.stores.get(name)?;
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
            if &c.store != name {
                return Err(SokarError::StoreMismatch {
                    expected: name.clone(),
                    actual: c.store,
                });
            }
            if c.key() != entry.path() {
                return Err(SokarError::Invalid(
                    "collection manifest key/store mismatch".into(),
                ));
            }
            collections.push(c);
        }
        // Retry after a manifest-only collection creation can leave duplicate scopes.
        // UUIDv7 ordering chooses the earliest deterministically, regardless of list order.
        collections.sort_by_key(|c| c.id);
        let mut scopes = HashSet::new();
        let mut orphan_collections = HashSet::new();
        collections.retain(|c| {
            if scopes.insert(c.scope.key()) { true } else {
                orphan_collections.insert(c.id);
                tracing::warn!(collection = %c.id, scope = %c.scope.key(), "duplicate scope manifest orphaned");
                false
            }
        });
        let mut orphans = orphan_collections.len();
        let mut artifacts = artifact_manifests(&store).await?;
        artifacts.retain(|a| {
            if orphan_collections.contains(&a.collection) {
                orphans += 1;
                tracing::warn!(artifact = %a.id, collection = %a.collection, "artifact of duplicate scope orphaned");
                false
            } else { true }
        });
        let ids: HashSet<_> = collections.iter().map(|c| c.id).collect();
        let mut blobs = HashMap::new();
        for a in &artifacts {
            if !ids.contains(&a.collection) {
                return Err(SokarError::Invalid(
                    "artifact references a missing collection manifest".into(),
                ));
            }
            // Tombstones hold no live blob reference; bytes may already have been
            // collected by a future janitor and are not required for restoration.
            if a.deleted_at.is_some() {
                continue;
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
        let mut tx = self.index.begin(MutationScope::Global).await?;
        // Cross-store uniqueness is decided under the global lock. A store scanned
        // later may contain the earliest UUID, so it must evict an indexed loser.
        let mut winners = Vec::new();
        for collection in collections {
            if let Some(existing) = tx.by_scope(&collection.scope.key()).await?
                && existing.store != *name
                && existing.id != collection.id
            {
                if existing.id < collection.id {
                    orphan_collections.insert(collection.id);
                    orphans += 1;
                    tracing::warn!(store = %name, collection = %collection.id, winner = %existing.id,
                        "cross-store duplicate scope manifest orphaned");
                    continue;
                }
                let removed = tx.orphan_collection(existing.id).await?;
                orphans += 1 + removed;
                tracing::warn!(store = %existing.store, collection = %existing.id, winner = %collection.id,
                    artifacts = removed, "cross-store duplicate scope and its artifacts orphaned");
            }
            winners.push(collection);
        }
        artifacts.retain(|a| {
            if orphan_collections.contains(&a.collection) {
                orphans += 1;
                tracing::warn!(store = %name, artifact = %a.id, collection = %a.collection,
                    "artifact of cross-store duplicate scope orphaned");
                false
            } else {
                true
            }
        });
        let mut counts = tx.replace_store(name, &winners, &artifacts).await?;
        counts.orphans = orphans;
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
