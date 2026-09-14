use super::*;
use serde::de::DeserializeOwned;
use sqlx::{PgPool, Postgres, Row, Transaction, postgres::PgPoolOptions};
use std::time::Duration;
const COLLECTION: &str = "SELECT manifest FROM sokar.collection WHERE id = $1";
const SCOPE: &str = "SELECT manifest FROM sokar.collection WHERE scope_key = $1";
const ARTIFACT: &str = "SELECT manifest FROM sokar.artifact WHERE id = $1 AND deleted_at IS NULL";
const LATEST: &str = "SELECT manifest FROM sokar.artifact WHERE collection = $1 AND path = $2 AND deleted_at IS NULL ORDER BY created_at DESC, id DESC LIMIT 1";
fn decode<T: DeserializeOwned>(value: Option<serde_json::Value>) -> SokarResult<T> {
    Ok(serde_json::from_value(value.ok_or(SokarError::NotFound)?)?)
}
fn optional<T: DeserializeOwned>(value: Option<serde_json::Value>) -> SokarResult<Option<T>> {
    value
        .map(serde_json::from_value)
        .transpose()
        .map_err(Into::into)
}
fn db_error(error: sqlx::Error) -> SokarError {
    if error
        .as_database_error()
        .is_some_and(|e| e.is_unique_violation())
    {
        SokarError::Conflict("index uniqueness constraint".into())
    } else {
        error.into()
    }
}

pub struct PgIndex {
    pool: PgPool,
}
impl PgIndex {
    pub async fn connect(url: &str) -> SokarResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .acquire_timeout(Duration::from_secs(10))
            .connect(url)
            .await?;
        Ok(Self { pool })
    }
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
    pub async fn migrate(&self) -> SokarResult<()> {
        // Session lock is held by this connection, never a random pooled connection.
        // close_on_drop also releases it if migration is cancelled or fails.
        let mut connection = self.pool.acquire().await?;
        connection.close_on_drop();
        sqlx::query("SELECT pg_advisory_lock(hashtext('sokar.migrations'))")
            .execute(&mut *connection)
            .await?;
        // Keep sqlx's migration ledger IN sokar, so DROP SCHEMA removes it too.
        sqlx::query("CREATE SCHEMA IF NOT EXISTS sokar")
            .execute(&mut *connection)
            .await?;
        sqlx::query("SET search_path TO sokar, public")
            .execute(&mut *connection)
            .await?;
        let result = sqlx::migrate!("./migrations").run(&mut *connection).await;
        sqlx::query("SELECT pg_advisory_unlock(hashtext('sokar.migrations'))")
            .execute(&mut *connection)
            .await?;
        result?;
        Ok(())
    }
}
#[async_trait]
impl Index for PgIndex {
    async fn ping(&self) -> SokarResult<()> {
        sqlx::query("SELECT 1 FROM sokar.collection LIMIT 1")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
    async fn collection(&self, id: CollectionId) -> SokarResult<Collection> {
        decode(
            sqlx::query_scalar(COLLECTION)
                .bind(id.as_uuid())
                .fetch_optional(&self.pool)
                .await?,
        )
    }
    async fn by_scope(&self, key: &str) -> SokarResult<Collection> {
        decode(
            sqlx::query_scalar(SCOPE)
                .bind(key)
                .fetch_optional(&self.pool)
                .await?,
        )
    }
    async fn artifact(&self, id: ArtifactId) -> SokarResult<Artifact> {
        decode(
            sqlx::query_scalar(ARTIFACT)
                .bind(id.as_uuid())
                .fetch_optional(&self.pool)
                .await?,
        )
    }
    async fn latest(&self, collection: CollectionId, path: &ArtifactPath) -> SokarResult<Artifact> {
        decode(
            sqlx::query_scalar(LATEST)
                .bind(collection.as_uuid())
                .bind(path.as_str())
                .fetch_optional(&self.pool)
                .await?,
        )
    }
    async fn list(
        &self,
        collection: CollectionId,
        query: &ListArtifacts,
    ) -> SokarResult<ArtifactPage> {
        let limit = page_limit(query)?;
        let cursor = query.cursor.as_deref().map(parse_cursor).transpose()?;
        let rows: Vec<serde_json::Value> = sqlx::query_scalar("SELECT manifest FROM sokar.artifact WHERE collection=$1 AND deleted_at IS NULL AND ($2::text IS NULL OR starts_with(path,$2)) AND ($3::text IS NULL OR kind=$3) AND ($4::timestamptz IS NULL OR (created_at,id) > ($4,$5)) ORDER BY created_at,id LIMIT $6")
            .bind(collection.as_uuid()).bind(&query.prefix).bind(query.kind.map(|k| k.to_string()))
            .bind(cursor.map(|c| c.0)).bind(cursor.map(|c| c.1.as_uuid())).bind((limit+1) as i64).fetch_all(&self.pool).await?;
        let values = rows
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<_, _>>()?;
        Ok(page(values, limit))
    }
    async fn begin(&self, scope: MutationScope) -> SokarResult<Box<dyn IndexTransaction>> {
        let mut tx = self.pool.begin().await?;
        // Always global before store. Shared global admits independent stores while
        // exclusive global excludes all artifact mutations for collection/reindex work.
        match scope {
            MutationScope::Global => {
                sqlx::query("SELECT pg_advisory_xact_lock(hashtext('sokar.mutations'))")
                    .execute(&mut *tx)
                    .await?;
            }
            MutationScope::Store(store) => {
                sqlx::query("SELECT pg_advisory_xact_lock_shared(hashtext('sokar.mutations'))")
                    .execute(&mut *tx)
                    .await?;
                sqlx::query("SELECT pg_advisory_xact_lock(hashtext('sokar.store:' || $1))")
                    .bind(store.as_str())
                    .execute(&mut *tx)
                    .await?;
            }
        }
        Ok(Box::new(PgTransaction { tx }))
    }
}
struct PgTransaction {
    tx: Transaction<'static, Postgres>,
}
#[async_trait]
impl IndexTransaction for PgTransaction {
    async fn by_scope(&mut self, key: &str) -> SokarResult<Option<Collection>> {
        optional(
            sqlx::query_scalar(SCOPE)
                .bind(key)
                .fetch_optional(&mut *self.tx)
                .await?,
        )
    }
    async fn latest(
        &mut self,
        collection: CollectionId,
        path: &ArtifactPath,
    ) -> SokarResult<Option<Artifact>> {
        optional(
            sqlx::query_scalar(LATEST)
                .bind(collection.as_uuid())
                .bind(path.as_str())
                .fetch_optional(&mut *self.tx)
                .await?,
        )
    }
    async fn artifact(&mut self, id: ArtifactId) -> SokarResult<Artifact> {
        decode(
            sqlx::query_scalar(ARTIFACT)
                .bind(id.as_uuid())
                .fetch_optional(&mut *self.tx)
                .await?,
        )
    }
    async fn idempotency(
        &mut self,
        principal: &str,
        collection: CollectionId,
        key: &str,
    ) -> SokarResult<Option<Idempotency>> {
        let row = sqlx::query("SELECT fingerprint,response,expires_at FROM sokar.idempotency WHERE principal=$1 AND collection=$2 AND key=$3 AND expires_at>now()")
            .bind(principal).bind(collection.as_uuid()).bind(key).fetch_optional(&mut *self.tx).await?;
        row.map(|r| {
            Ok(Idempotency {
                principal: principal.into(),
                collection,
                key: key.into(),
                fingerprint: r.try_get("fingerprint")?,
                response: serde_json::from_value(r.try_get("response")?)?,
                expires_at: r.try_get("expires_at")?,
            })
        })
        .transpose()
    }
    async fn put_collection(&mut self, c: &Collection) -> SokarResult<()> {
        let existing: Option<String> =
            sqlx::query_scalar("SELECT store FROM sokar.collection WHERE id=$1")
                .bind(c.id.as_uuid())
                .fetch_optional(&mut *self.tx)
                .await?;
        if let Some(store) = existing
            && store != c.store.as_str()
        {
            return Err(SokarError::StoreMismatch {
                expected: store.parse()?,
                actual: c.store.clone(),
            });
        }
        sqlx::query("INSERT INTO sokar.collection(id,scope,scope_key,owner,store,retention,quota,created_at,manifest) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT(id) DO UPDATE SET scope=EXCLUDED.scope,scope_key=EXCLUDED.scope_key,owner=EXCLUDED.owner,store=EXCLUDED.store,retention=EXCLUDED.retention,quota=EXCLUDED.quota,created_at=EXCLUDED.created_at,manifest=EXCLUDED.manifest")
            .bind(c.id.as_uuid()).bind(serde_json::to_value(&c.scope)?).bind(c.scope.key()).bind(&c.owner).bind(c.store.as_str()).bind(serde_json::to_value(&c.retention)?).bind(serde_json::to_value(&c.quota)?).bind(c.created_at).bind(serde_json::to_value(c)?)
            .execute(&mut *self.tx).await.map_err(|e| {
                if e.as_database_error().is_some_and(|e| e.is_unique_violation() && e.constraint() == Some("collection_scope_key_key")) {
                    SokarError::ScopeExists(c.scope.key())
                } else { db_error(e) }
            })?;
        Ok(())
    }
    async fn put_artifact(&mut self, store: &StoreName, a: &Artifact) -> SokarResult<()> {
        let collection: Collection = decode(
            sqlx::query_scalar(COLLECTION)
                .bind(a.collection.as_uuid())
                .fetch_optional(&mut *self.tx)
                .await?,
        )?;
        if &collection.store != store {
            return Err(SokarError::StoreMismatch {
                expected: collection.store.clone(),
                actual: store.clone(),
            });
        }
        let size = i64::try_from(a.blob.size)
            .map_err(|_| SokarError::Invalid("blob too large for index".into()))?;
        sqlx::query("INSERT INTO sokar.blob(store,digest,size,media_type,refcount,created_at) VALUES($1,$2,$3,$4,$6,$5) ON CONFLICT(store,digest) DO UPDATE SET refcount=sokar.blob.refcount+EXCLUDED.refcount")
            .bind(store.as_str()).bind(a.blob.digest.0.as_slice()).bind(size).bind(a.blob.media_type.as_str()).bind(a.created_at).bind(i32::from(a.deleted_at.is_none())).execute(&mut *self.tx).await?;
        sqlx::query("INSERT INTO sokar.artifact(id,collection,path,kind,store,digest,media_type,labels,created_by,created_at,expires_at,previous,manifest,deleted_at) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)")
            .bind(a.id.as_uuid()).bind(a.collection.as_uuid()).bind(a.path.as_str()).bind(a.kind.to_string()).bind(store.as_str()).bind(a.blob.digest.0.as_slice()).bind(a.blob.media_type.as_str()).bind(serde_json::to_value(&a.labels)?).bind(&a.created_by).bind(a.created_at).bind(a.expires_at).bind(a.previous.map(|i| i.as_uuid())).bind(serde_json::to_value(a)?).bind(a.deleted_at)
            .execute(&mut *self.tx).await.map_err(db_error)?;
        Ok(())
    }
    async fn put_idempotency(&mut self, i: &Idempotency) -> SokarResult<()> {
        sqlx::query("DELETE FROM sokar.idempotency WHERE collection=$1 AND expires_at<=now()")
            .bind(i.collection.as_uuid())
            .execute(&mut *self.tx)
            .await?;
        sqlx::query("INSERT INTO sokar.idempotency(principal,collection,key,fingerprint,response,expires_at) VALUES($1,$2,$3,$4,$5,$6) ON CONFLICT(principal,collection,key) DO UPDATE SET fingerprint=EXCLUDED.fingerprint,response=EXCLUDED.response,expires_at=EXCLUDED.expires_at")
            .bind(&i.principal).bind(i.collection.as_uuid()).bind(&i.key).bind(&i.fingerprint).bind(serde_json::to_value(&i.response)?).bind(i.expires_at).execute(&mut *self.tx).await?;
        Ok(())
    }
    async fn purge_idempotency_for(&mut self, artifact_id: ArtifactId) -> SokarResult<()> {
        sqlx::query("DELETE FROM sokar.idempotency WHERE response->'artifact'->>'id'=$1")
            .bind(artifact_id.to_string())
            .execute(&mut *self.tx)
            .await?;
        Ok(())
    }
    async fn delete_artifact(
        &mut self,
        store: &StoreName,
        artifact: &Artifact,
    ) -> SokarResult<u64> {
        let a = self.artifact(artifact.id).await?;
        sqlx::query("UPDATE sokar.artifact SET deleted_at=$2,manifest=$3 WHERE id=$1")
            .bind(artifact.id.as_uuid())
            .bind(artifact.deleted_at)
            .bind(serde_json::to_value(artifact)?)
            .execute(&mut *self.tx)
            .await?;
        let remaining: i32 = sqlx::query_scalar("UPDATE sokar.blob SET refcount=refcount-1 WHERE store=$1 AND digest=$2 AND refcount>0 RETURNING refcount")
            .bind(store.as_str()).bind(a.blob.digest.0.as_slice()).fetch_one(&mut *self.tx).await?;
        Ok(remaining as u64)
    }
    async fn replace_store(
        &mut self,
        store: &StoreName,
        collections: &[Collection],
        artifacts: &[Artifact],
    ) -> SokarResult<ReindexCounts> {
        let artifact_ids: std::collections::HashSet<_> = artifacts
            .iter()
            .filter(|a| a.deleted_at.is_none())
            .map(|a| a.id)
            .collect();
        let rows = sqlx::query("SELECT principal,collection,key,fingerprint,response,expires_at FROM sokar.idempotency WHERE collection IN (SELECT id FROM sokar.collection WHERE store=$1) AND expires_at>now()")
            .bind(store.as_str()).fetch_all(&mut *self.tx).await?;
        let mut receipts = Vec::new();
        for row in rows {
            let response: UploadResponse = serde_json::from_value(row.try_get("response")?)?;
            if artifact_ids.contains(&response.artifact.id) {
                receipts.push(Idempotency {
                    principal: row.try_get("principal")?,
                    collection: CollectionId::try_from(
                        row.try_get::<uuid::Uuid, _>("collection")?,
                    )?,
                    key: row.try_get("key")?,
                    fingerprint: row.try_get("fingerprint")?,
                    response,
                    expires_at: row.try_get("expires_at")?,
                });
            }
        }
        sqlx::query("DELETE FROM sokar.idempotency WHERE collection IN (SELECT id FROM sokar.collection WHERE store=$1)").bind(store.as_str()).execute(&mut *self.tx).await?;
        sqlx::query("DELETE FROM sokar.artifact WHERE store=$1")
            .bind(store.as_str())
            .execute(&mut *self.tx)
            .await?;
        sqlx::query("DELETE FROM sokar.blob WHERE store=$1")
            .bind(store.as_str())
            .execute(&mut *self.tx)
            .await?;
        sqlx::query("DELETE FROM sokar.collection WHERE store=$1")
            .bind(store.as_str())
            .execute(&mut *self.tx)
            .await?;
        for c in collections {
            self.put_collection(c).await?;
        }
        for a in artifacts {
            self.put_artifact(store, a).await?;
        }
        for receipt in receipts {
            self.put_idempotency(&receipt).await?;
        }
        let count: i64 =
            sqlx::query_scalar("SELECT count(*) FROM sokar.blob WHERE store=$1 AND refcount>0")
                .bind(store.as_str())
                .fetch_one(&mut *self.tx)
                .await?;
        Ok(ReindexCounts {
            collections: collections.len(),
            artifacts: artifacts.iter().filter(|a| a.deleted_at.is_none()).count(),
            orphans: 0,
            blobs: count as usize,
        })
    }
    async fn commit(self: Box<Self>) -> SokarResult<()> {
        self.tx.commit().await?;
        Ok(())
    }
}
