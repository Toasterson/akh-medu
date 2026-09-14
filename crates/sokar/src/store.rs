use crate::{
    SokarError, SokarResult,
    config::{ServiceKind, StoreConfig},
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use opendal::{ErrorKind, Operator, layers::*, services};
use prometheus_client::registry::Registry;
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use sokar_client::{BlobDigest, StoreName, UploadOutcome};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub const FORMAT_KEY: &str = "_sokar/format.json";
pub const FORMAT_SCHEMA: &str = "sokar.layout.v1";
#[derive(Debug)]
pub struct StoreCapability {
    pub presign_read: bool,
    pub presign_write: bool,
    pub ranged_read: bool,
    pub conditional_write: AtomicBool,
}
impl From<opendal::Capability> for StoreCapability {
    fn from(cap: opendal::Capability) -> Self {
        Self {
            presign_read: cap.presign_read,
            presign_write: cap.presign_write,
            // 0.59 has no read_with_range field. Its range-class availability flag
            // is read_with_suffix (including the default SimulateLayer on fs).
            ranged_read: cap.read_with_suffix,
            conditional_write: AtomicBool::new(if cap.copy {
                cap.copy_with_if_not_exists
            } else {
                cap.write_with_if_not_exists
            }),
        }
    }
}
pub struct Store {
    pub name: StoreName,
    pub op: Operator,
    pub cap: StoreCapability,
}
pub struct StoreRegistry {
    pub stores: HashMap<StoreName, Arc<Store>>,
    pub metrics: Registry,
}
impl StoreRegistry {
    pub async fn build(configs: &[StoreConfig]) -> SokarResult<Self> {
        let mut metrics = Registry::default();
        let mut stores = HashMap::new();
        let mut configs: Vec<_> = configs.iter().collect();
        configs.sort_by(|a, b| a.name.cmp(&b.name));
        for config in configs {
            let keys = match config.service {
                ServiceKind::S3 => serde_json::to_value(services::S3Config::default())?,
                ServiceKind::Fs => serde_json::to_value(services::FsConfig::default())?,
                ServiceKind::Gcs => serde_json::to_value(services::GcsConfig::default())?,
                ServiceKind::Azblob => serde_json::to_value(services::AzblobConfig::default())?,
                ServiceKind::Memory => serde_json::to_value(services::MemoryConfig::default())?,
            };
            let invalid = |message: &str| {
                SokarError::storage(
                    &config.name,
                    "configure",
                    opendal::Error::new(ErrorKind::ConfigInvalid, message),
                )
            };
            let mut options = std::collections::BTreeMap::new();
            for (key, value) in &config.options {
                if keys.get(key).is_none() {
                    return Err(invalid(&format!("unknown canonical store option: {key}")));
                }
                options.insert(key.clone(), value.resolve()?);
            }
            if let Some(prefix) = &config.prefix {
                let prefix = prefix
                    .parse::<sokar_client::ArtifactPath>()
                    .map_err(|_| invalid("prefix must be a relative artifact path"))?;
                let root = options.entry("root".into()).or_default();
                *root = format!("{}/{}/", root.trim_end_matches('/'), prefix);
            }
            if matches!(config.service, ServiceKind::S3) {
                options
                    .entry("disable_config_load".into())
                    .or_insert("true".into());
                options
                    .entry("disable_ec2_metadata".into())
                    .or_insert("true".into());
            }
            let registry =
                metrics.sub_registry_with_label(("store".into(), config.name.to_string().into()));
            let op = Operator::via_iter(config.service.as_str(), options)
                .map_err(|e| SokarError::storage(&config.name, "configure", e))?
                .layer(RetryLayer::new().with_jitter())
                .layer(TimeoutLayer::new())
                .layer(ConcurrentLimitLayer::new(32))
                .layer(TracingLayer::new())
                .layer(PrometheusClientLayer::builder().register(registry));
            let cap = op.info().capability();
            if !(cap.read && cap.write && cap.stat && cap.list && cap.delete) {
                return Err(invalid("store requires read/write/stat/list/delete"));
            }
            let store = Arc::new(Store {
                name: config.name.clone(),
                op,
                cap: StoreCapability::from(cap),
            });
            store.initialize().await?;
            if stores.insert(config.name.clone(), store).is_some() {
                return Err(invalid("duplicate store name"));
            }
        }
        Ok(Self { stores, metrics })
    }
    pub fn get(&self, name: &StoreName) -> SokarResult<Arc<Store>> {
        self.stores
            .get(name)
            .cloned()
            .ok_or_else(|| SokarError::UnknownStore(name.clone()))
    }
    pub fn sorted_names(&self) -> Vec<StoreName> {
        let mut names: Vec<_> = self.stores.keys().cloned().collect();
        names.sort();
        names
    }
    pub async fn check(&self, probe: bool) -> SokarResult<()> {
        for name in self.sorted_names() {
            self.get(&name)?.check(probe).await?;
        }
        Ok(())
    }
}
impl Store {
    pub fn error(&self, op: &'static str, e: opendal::Error) -> SokarError {
        SokarError::storage(&self.name, op, e)
    }
    pub async fn initialize(&self) -> SokarResult<()> {
        match self.op.stat(FORMAT_KEY).await {
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {
                let marker = serde_json::json!({"schema": FORMAT_SCHEMA, "created_at": time::OffsetDateTime::now_utc().format(&time::format_description::well_known::Rfc3339).expect("UTC")});
                let result = self
                    .op
                    .write_with(FORMAT_KEY, serde_json::to_vec(&marker)?)
                    .if_not_exists(self.op.info().capability().write_with_if_not_exists)
                    .await;
                match result {
                    Ok(_) => {}
                    Err(e) if dedup_error(&e) => {}
                    Err(e) => return Err(self.error("initialize", e)),
                }
            }
            Err(e) => return Err(self.error("stat format", e)),
        }
        self.check(false).await?;
        self.probe_conditional_copy().await
    }
    pub async fn check(&self, probe: bool) -> SokarResult<()> {
        let marker: serde_json::Value = self.read_json(FORMAT_KEY).await?;
        sokar_client::check_schema(
            marker.get("schema").and_then(|v| v.as_str()).unwrap_or(""),
            FORMAT_SCHEMA,
        )?;
        if probe {
            let key = format!("_sokar/probe/{}", uuid::Uuid::now_v7());
            self.op
                .write(&key, "sokar probe")
                .await
                .map_err(|e| self.error("write probe", e))?;
            self.delete(&key).await?;
            self.probe_conditional_copy().await?;
        }
        Ok(())
    }
    fn downgrade_conditional_copy(&self, reason: &str) {
        self.cap.conditional_write.store(false, Ordering::Relaxed);
        tracing::warn!(store = %self.name, %reason, "conditional copy disabled; using stat-then-copy");
    }
    /// Advertising a precondition does not prove that the configured endpoint honors it.
    /// A downgrade is sticky for this Store, including across subsequent health checks.
    async fn probe_conditional_copy(&self) -> SokarResult<()> {
        let cap = self.op.info().capability();
        if !(cap.copy && cap.copy_with_if_not_exists) {
            return Ok(());
        }
        let source = format!("_sokar/probe/{}", uuid::Uuid::now_v7());
        let target = format!("{source}-copy");
        let probe: SokarResult<()> = async {
            self.op
                .write(&source, "sokar conditional copy probe")
                .await
                .map_err(|e| self.error("write copy probe", e))?;
            self.op
                .copy_with(&source, &target)
                .if_not_exists(true)
                .await
                .map_err(|e| self.error("first conditional copy probe", e))?;
            match self
                .op
                .copy_with(&source, &target)
                .if_not_exists(true)
                .await
            {
                Err(e) if dedup_error(&e) => Ok(()),
                Err(e) => Err(self.error("second conditional copy probe", e)),
                Ok(_) => Err(SokarError::Invalid(
                    "conditional copy overwrote an existing probe".into(),
                )),
            }
        }
        .await;
        if let Err(error) = probe {
            self.downgrade_conditional_copy(&error.to_string());
        }
        // Attempt both deletions even if one fails; check/build surface cleanup failures.
        let source_cleanup = self.delete(&source).await;
        let target_cleanup = self.delete(&target).await;
        source_cleanup.and(target_cleanup)
    }
    pub async fn read_json<T: DeserializeOwned>(&self, key: &str) -> SokarResult<T> {
        let data = self
            .op
            .read(key)
            .await
            .map_err(|e| self.error("read manifest", e))?;
        if data.len() > 16 * 1024 * 1024 {
            return Err(SokarError::Invalid("manifest exceeds 16 MiB".into()));
        }
        Ok(serde_json::from_slice(&data.to_bytes())?)
    }
    pub async fn read_manifest<T: DeserializeOwned>(
        &self,
        key: &str,
        schema: &'static str,
    ) -> SokarResult<T> {
        let value: serde_json::Value = self.read_json(key).await?;
        sokar_client::check_schema(
            value.get("schema").and_then(|v| v.as_str()).unwrap_or(""),
            schema,
        )?;
        Ok(serde_json::from_value(value)?)
    }
    pub async fn write_json(&self, key: &str, value: &impl Serialize) -> SokarResult<()> {
        self.op
            .write(key, serde_json::to_vec(value)?)
            .await
            .map_err(|e| self.error("write manifest", e))?;
        Ok(())
    }
    pub async fn delete(&self, key: &str) -> SokarResult<()> {
        self.op
            .delete(key)
            .await
            .map_err(|e| self.error("delete", e))?;
        Ok(())
    }
    /// Keep memory bounded by incoming chunks and the backend writer buffer.
    pub async fn stage(
        self: &Arc<Self>,
        stream: impl Stream<Item = SokarResult<Bytes>>,
        max_bytes: usize,
    ) -> SokarResult<StagedUpload> {
        let mut stage = StagedUpload {
            store: self.clone(),
            key: format!("uploads/{}", uuid::Uuid::now_v7()),
            digest: BlobDigest([0; 32]),
            size: 0,
            cleaned: false,
        };
        let mut writer = self
            .op
            .writer_with(&stage.key)
            .chunk(8 * 1024 * 1024)
            .await
            .map_err(|e| self.error("open staging", e))?;
        let mut hash = Sha256::new();
        futures::pin_mut!(stream);
        let result: SokarResult<()> = async {
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                stage.size = stage
                    .size
                    .checked_add(chunk.len() as u64)
                    .ok_or(SokarError::BodyLimit)?;
                if stage.size > max_bytes as u64 {
                    return Err(SokarError::BodyLimit);
                }
                hash.update(&chunk);
                writer
                    .write(chunk)
                    .await
                    .map_err(|e| self.error("write staging", e))?;
            }
            writer
                .close()
                .await
                .map_err(|e| self.error("close staging", e))?;
            Ok(())
        }
        .await;
        if let Err(e) = result {
            let _ = writer.abort().await;
            if let Err(cleanup) = stage.cleanup().await {
                tracing::warn!(error = %cleanup, key = %stage.key, "staging cleanup failed");
            }
            return Err(e);
        }
        stage.digest = BlobDigest(hash.finalize().into());
        Ok(stage)
    }
}
fn dedup_error(e: &opendal::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::ConditionNotMatch | ErrorKind::AlreadyExists
    )
}
pub struct StagedUpload {
    pub store: Arc<Store>,
    pub key: String,
    pub digest: BlobDigest,
    pub size: u64,
    cleaned: bool,
}
impl StagedUpload {
    /// Runs outside index locks. Non-conditional stat-then-copy has a TOCTOU window:
    /// concurrent writes are safe only because a digest names byte-identical content.
    /// Index locks cannot serialize bucket writes across independent indexes; a bucket
    /// owner marker is deferred to 52b.
    pub async fn promote(&self) -> SokarResult<UploadOutcome> {
        let conditional = self.store.cap.conditional_write.load(Ordering::Relaxed);
        match self.promote_once(conditional).await {
            Err(SokarError::Storage { ref source, .. })
                if conditional
                    && self.store.op.info().capability().copy
                    && !dedup_error(source) =>
            {
                self.store.downgrade_conditional_copy(&source.to_string());
                // One fallback only: ordinary copy errors must still reach the caller.
                self.promote_once(false).await
            }
            result => result,
        }
    }
    async fn promote_once(&self, conditional: bool) -> SokarResult<UploadOutcome> {
        let final_key = self.digest.key();
        if !conditional {
            match self.store.op.stat(&final_key).await {
                Ok(meta) if meta.content_length() == self.size => {
                    return Ok(UploadOutcome::Deduplicated);
                }
                Ok(_) => {
                    return Err(SokarError::Conflict(
                        "stored digest has a different size".into(),
                    ));
                }
                Err(e) if e.kind() == ErrorKind::NotFound => {}
                Err(e) => return Err(self.store.error("stat blob", e)),
            }
        }
        let result = if self.store.op.info().capability().copy {
            self.store
                .op
                .copy_with(&self.key, &final_key)
                .if_not_exists(conditional)
                .await
                .map(|_| ())
        } else {
            // OpenDAL's memory backend has no copy operation. Relay through bounded
            // chunks. Concurrent promotion is safe only for byte-identical digest content.
            async {
                let reader = self.store.op.reader(&self.key).await?;
                let mut stream = reader.into_bytes_stream(..).await?;
                let mut writer = self
                    .store
                    .op
                    .writer_with(&final_key)
                    .if_not_exists(conditional)
                    .await?;
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.map_err(|e| {
                        opendal::Error::new(ErrorKind::Unexpected, "staging relay failed")
                            .set_source(e)
                    })?;
                    writer.write(chunk).await?;
                }
                writer.close().await?;
                Ok(())
            }
            .await
        };
        match result {
            Ok(_) => Ok(UploadOutcome::Stored),
            Err(e) if dedup_error(&e) => Ok(UploadOutcome::Deduplicated),
            Err(e) => Err(self.store.error("promote blob", e)),
        }
    }
    pub async fn cleanup(&mut self) -> SokarResult<()> {
        if self.cleaned {
            return Ok(());
        }
        self.store.delete(&self.key).await?;
        self.cleaned = true;
        Ok(())
    }
}
impl Drop for StagedUpload {
    fn drop(&mut self) {
        if self.cleaned {
            return;
        }
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                let store = self.store.clone();
                let key = self.key.clone();
                handle.spawn(async move {
                    if let Err(e) = store.delete(&key).await {
                        tracing::warn!(error = %e, %key, "staging cleanup failed");
                    }
                });
            }
            Err(_) => {
                tracing::warn!(key = %self.key, "staging cleanup unavailable without a runtime")
            }
        }
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
