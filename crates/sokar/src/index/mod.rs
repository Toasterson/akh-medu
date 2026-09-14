//! Transactional index seam. Lock order is global before store, never reversed.
//! Store mutations take shared global + exclusive store; global mutations take
//! exclusive global, excluding every store mutation during collection/reindex changes.
mod memory;
mod postgres;
use crate::{SokarError, SokarResult};
use async_trait::async_trait;
pub use memory::MemoryIndex;
pub use postgres::PgIndex;
use sokar_client::*;
use time::OffsetDateTime;

#[derive(Debug, Clone)]
pub struct Idempotency {
    pub principal: String,
    pub collection: CollectionId,
    pub key: String,
    pub fingerprint: String,
    pub response: UploadResponse,
    pub expires_at: OffsetDateTime,
}
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ReindexCounts {
    pub collections: usize,
    pub artifacts: usize,
    pub blobs: usize,
    pub orphans: usize,
}
#[derive(Debug, Clone)]
pub enum MutationScope {
    Global,
    Store(StoreName),
}
#[async_trait]
pub trait Index: Send + Sync {
    async fn ping(&self) -> SokarResult<()>;
    async fn collection(&self, id: CollectionId) -> SokarResult<Collection>;
    async fn by_scope(&self, key: &str) -> SokarResult<Collection>;
    async fn artifact(&self, id: ArtifactId) -> SokarResult<Artifact>;
    async fn latest(&self, collection: CollectionId, path: &ArtifactPath) -> SokarResult<Artifact>;
    async fn list(
        &self,
        collection: CollectionId,
        query: &ListArtifacts,
    ) -> SokarResult<ArtifactPage>;
    async fn begin(&self, scope: MutationScope) -> SokarResult<Box<dyn IndexTransaction>>;
}
#[async_trait]
pub trait IndexTransaction: Send {
    async fn by_scope(&mut self, key: &str) -> SokarResult<Option<Collection>>;
    async fn latest(
        &mut self,
        collection: CollectionId,
        path: &ArtifactPath,
    ) -> SokarResult<Option<Artifact>>;
    async fn artifact(&mut self, id: ArtifactId) -> SokarResult<Artifact>;
    async fn idempotency(
        &mut self,
        principal: &str,
        collection: CollectionId,
        key: &str,
    ) -> SokarResult<Option<Idempotency>>;
    async fn put_collection(&mut self, collection: &Collection) -> SokarResult<()>;
    async fn put_artifact(&mut self, store: &StoreName, artifact: &Artifact) -> SokarResult<()>;
    async fn put_idempotency(&mut self, entry: &Idempotency) -> SokarResult<()>;
    async fn purge_idempotency_for(&mut self, artifact_id: ArtifactId) -> SokarResult<()>;
    /// Tombstones the artifact; returns remaining references in this store.
    async fn delete_artifact(&mut self, store: &StoreName, artifact: &Artifact)
    -> SokarResult<u64>;
    /// Remove a losing scope's index rows and receipts, retaining physical manifests
    /// and bytes. Requires the global lock; returns the number of orphaned artifacts.
    async fn orphan_collection(&mut self, id: CollectionId) -> SokarResult<usize>;
    /// Replace only this store's index with the fully validated manifest inventory.
    async fn replace_store(
        &mut self,
        store: &StoreName,
        collections: &[Collection],
        artifacts: &[Artifact],
    ) -> SokarResult<ReindexCounts>;
    async fn commit(self: Box<Self>) -> SokarResult<()>;
}
pub fn cursor(artifact: &Artifact) -> String {
    format!(
        "{}:{}",
        artifact.created_at.unix_timestamp_nanos(),
        artifact.id
    )
}
pub fn parse_cursor(value: &str) -> SokarResult<(OffsetDateTime, ArtifactId)> {
    let (time, id) = value
        .split_once(':')
        .ok_or_else(|| SokarError::Invalid("invalid cursor".into()))?;
    let time = time
        .parse::<i128>()
        .ok()
        .and_then(|t| OffsetDateTime::from_unix_timestamp_nanos(t).ok())
        .ok_or_else(|| SokarError::Invalid("invalid cursor timestamp".into()))?;
    Ok((time, id.parse()?))
}
pub fn page_limit(query: &ListArtifacts) -> SokarResult<usize> {
    let limit = query.limit.unwrap_or(100);
    if !(1..=1000).contains(&limit) {
        return Err(SokarError::Invalid("limit must be 1..1000".into()));
    }
    Ok(limit as usize)
}
pub fn page(mut artifacts: Vec<Artifact>, limit: usize) -> ArtifactPage {
    let more = artifacts.len() > limit;
    artifacts.truncate(limit);
    let next_cursor = if more {
        artifacts.last().map(cursor)
    } else {
        None
    };
    ArtifactPage {
        artifacts,
        next_cursor,
    }
}
