use super::*;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::{Mutex, OwnedMutexGuard};
#[derive(Default, Clone)]
struct Data {
    collections: HashMap<CollectionId, Collection>,
    artifacts: HashMap<ArtifactId, Artifact>,
    tombstones: HashMap<ArtifactId, Artifact>,
    refs: HashMap<(StoreName, BlobDigest), u64>,
    idempotency: HashMap<(String, CollectionId, String), Idempotency>,
}
impl Data {
    fn scope(&self, key: &str) -> Option<Collection> {
        self.collections
            .values()
            .find(|c| c.scope.key() == key)
            .cloned()
    }
    fn latest(&self, collection: CollectionId, path: &ArtifactPath) -> Option<Artifact> {
        self.artifacts
            .values()
            .filter(|a| a.collection == collection && &a.path == path)
            .max_by_key(|a| (a.created_at, a.id))
            .cloned()
    }
}
#[derive(Default)]
pub struct MemoryIndex {
    data: Arc<Mutex<Data>>,
}
impl MemoryIndex {
    pub async fn clear(&self) {
        *self.data.lock().await = Data::default();
    }
    pub async fn refcount(&self, store: &StoreName, digest: BlobDigest) -> u64 {
        *self
            .data
            .lock()
            .await
            .refs
            .get(&(store.clone(), digest))
            .unwrap_or(&0)
    }
}
#[async_trait]
impl Index for MemoryIndex {
    async fn ping(&self) -> SokarResult<()> {
        Ok(())
    }
    async fn collection(&self, id: CollectionId) -> SokarResult<Collection> {
        self.data
            .lock()
            .await
            .collections
            .get(&id)
            .cloned()
            .ok_or(SokarError::NotFound)
    }
    async fn by_scope(&self, key: &str) -> SokarResult<Collection> {
        self.data
            .lock()
            .await
            .scope(key)
            .ok_or(SokarError::NotFound)
    }
    async fn artifact(&self, id: ArtifactId) -> SokarResult<Artifact> {
        self.data
            .lock()
            .await
            .artifacts
            .get(&id)
            .cloned()
            .ok_or(SokarError::NotFound)
    }
    async fn latest(&self, collection: CollectionId, path: &ArtifactPath) -> SokarResult<Artifact> {
        self.data
            .lock()
            .await
            .latest(collection, path)
            .ok_or(SokarError::NotFound)
    }
    async fn list(
        &self,
        collection: CollectionId,
        query: &ListArtifacts,
    ) -> SokarResult<ArtifactPage> {
        let limit = page_limit(query)?;
        let after = query.cursor.as_deref().map(parse_cursor).transpose()?;
        let data = self.data.lock().await;
        let mut values: Vec<_> = data
            .artifacts
            .values()
            .filter(|a| {
                a.collection == collection
                    && query
                        .prefix
                        .as_ref()
                        .is_none_or(|p| a.path.as_str().starts_with(p))
                    && query.kind.is_none_or(|k| a.kind == k)
                    && after.is_none_or(|p| (a.created_at, a.id) > p)
            })
            .cloned()
            .collect();
        values.sort_by_key(|a| (a.created_at, a.id));
        Ok(page(values, limit))
    }
    async fn begin(&self) -> SokarResult<Box<dyn IndexTransaction>> {
        let guard = self.data.clone().lock_owned().await;
        let working = guard.clone();
        Ok(Box::new(MemoryTransaction { guard, working }))
    }
}
struct MemoryTransaction {
    guard: OwnedMutexGuard<Data>,
    working: Data,
}
#[async_trait]
impl IndexTransaction for MemoryTransaction {
    async fn by_scope(&mut self, key: &str) -> SokarResult<Option<Collection>> {
        Ok(self.working.scope(key))
    }
    async fn latest(
        &mut self,
        collection: CollectionId,
        path: &ArtifactPath,
    ) -> SokarResult<Option<Artifact>> {
        Ok(self.working.latest(collection, path))
    }
    async fn artifact(&mut self, id: ArtifactId) -> SokarResult<Artifact> {
        self.working
            .artifacts
            .get(&id)
            .cloned()
            .ok_or(SokarError::NotFound)
    }
    async fn idempotency(
        &mut self,
        principal: &str,
        collection: CollectionId,
        key: &str,
    ) -> SokarResult<Option<Idempotency>> {
        Ok(self
            .working
            .idempotency
            .get(&(principal.into(), collection, key.into()))
            .filter(|i| i.expires_at > OffsetDateTime::now_utc())
            .cloned())
    }
    async fn put_collection(&mut self, c: &Collection) -> SokarResult<()> {
        if self
            .working
            .collections
            .get(&c.id)
            .is_some_and(|old| old.store != c.store)
        {
            return Err(SokarError::Conflict(
                "collection UUID belongs to another store".into(),
            ));
        }
        if self
            .working
            .scope(&c.scope.key())
            .is_some_and(|old| old.id != c.id)
        {
            return Err(SokarError::Conflict("scope already exists".into()));
        }
        self.working.collections.insert(c.id, c.clone());
        Ok(())
    }
    async fn put_artifact(&mut self, store: &StoreName, a: &Artifact) -> SokarResult<()> {
        let c = self
            .working
            .collections
            .get(&a.collection)
            .ok_or(SokarError::NotFound)?;
        if &c.store != store {
            return Err(SokarError::Conflict(
                "artifact store differs from collection".into(),
            ));
        }
        if self.working.artifacts.contains_key(&a.id) {
            return Err(SokarError::Conflict("artifact already exists".into()));
        }
        *self
            .working
            .refs
            .entry((store.clone(), a.blob.digest))
            .or_default() += 1;
        self.working.artifacts.insert(a.id, a.clone());
        Ok(())
    }
    async fn put_idempotency(&mut self, i: &Idempotency) -> SokarResult<()> {
        self.working
            .idempotency
            .retain(|_, v| v.expires_at > OffsetDateTime::now_utc());
        self.working.idempotency.insert(
            (i.principal.clone(), i.collection, i.key.clone()),
            i.clone(),
        );
        Ok(())
    }
    async fn delete_artifact(&mut self, store: &StoreName, id: ArtifactId) -> SokarResult<u64> {
        let a = self
            .working
            .artifacts
            .remove(&id)
            .ok_or(SokarError::NotFound)?;
        let key = (store.clone(), a.blob.digest);
        let count = self
            .working
            .refs
            .get_mut(&key)
            .ok_or_else(|| SokarError::Conflict("missing blob index".into()))?;
        *count = count
            .checked_sub(1)
            .ok_or_else(|| SokarError::Conflict("invalid refcount".into()))?;
        let remaining = *count;
        if remaining == 0 {
            self.working.refs.remove(&key);
        }
        self.working.tombstones.insert(id, a);
        Ok(remaining)
    }
    async fn replace_store(
        &mut self,
        store: &StoreName,
        collections: &[Collection],
        artifacts: &[Artifact],
    ) -> SokarResult<ReindexCounts> {
        let ids: std::collections::HashSet<_> = self
            .working
            .collections
            .values()
            .filter(|c| &c.store == store)
            .map(|c| c.id)
            .collect();
        self.working
            .artifacts
            .retain(|_, a| !ids.contains(&a.collection));
        self.working
            .tombstones
            .retain(|_, a| !ids.contains(&a.collection));
        let artifact_ids: std::collections::HashSet<_> = artifacts.iter().map(|a| a.id).collect();
        self.working.idempotency.retain(|_, i| {
            !ids.contains(&i.collection)
                || (artifact_ids.contains(&i.response.artifact.id)
                    && i.expires_at > OffsetDateTime::now_utc())
        });
        self.working.collections.retain(|_, c| &c.store != store);
        self.working.refs.retain(|(s, _), _| s != store);
        for c in collections {
            self.put_collection(c).await?;
        }
        for a in artifacts {
            self.put_artifact(store, a).await?;
        }
        Ok(ReindexCounts {
            collections: collections.len(),
            artifacts: artifacts.len(),
            blobs: self.working.refs.keys().filter(|(s, _)| s == store).count(),
        })
    }
    async fn commit(mut self: Box<Self>) -> SokarResult<()> {
        *self.guard = std::mem::take(&mut self.working);
        Ok(())
    }
}
