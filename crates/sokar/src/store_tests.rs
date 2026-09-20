use super::*;
use opendal::{Capability, OperationContext, Result, raw::*};
use std::sync::atomic::AtomicUsize;

// 0 honors conditions; 1 rejects them; 2 silently ignores them; 3 rejects all copy.
#[derive(Debug, Default)]
struct Behavior {
    mode: AtomicUsize,
    conditional: AtomicUsize,
    ordinary: AtomicUsize,
}
#[derive(Debug)]
struct GatewayLayer(Arc<Behavior>);
impl Layer for GatewayLayer {
    fn apply_service(&self, inner: Servicer) -> Servicer {
        Arc::new(Gateway {
            inner,
            behavior: self.0.clone(),
        })
    }
}
#[derive(Debug)]
struct Gateway {
    inner: Servicer,
    behavior: Arc<Behavior>,
}
impl Service for Gateway {
    type Reader = oio::Reader;
    type Writer = oio::Writer;
    type Lister = oio::Lister;
    type Deleter = oio::Deleter;
    type Copier = oio::Copier;
    type Composer = oio::Composer;

    fn info(&self) -> ServiceInfo {
        self.inner.info()
    }

    fn capability(&self) -> Capability {
        opendal::Capability {
            copy: true,
            copy_with_if_not_exists: true,
            ..self.inner.capability()
        }
    }

    async fn create_dir(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpCreateDir,
    ) -> Result<RpCreateDir> {
        self.inner.create_dir(ctx, path, args).await
    }

    async fn stat(&self, ctx: &OperationContext, path: &str, args: OpStat) -> Result<RpStat> {
        self.inner.stat(ctx, path, args).await
    }

    fn read(&self, ctx: &OperationContext, path: &str, args: OpRead) -> Result<Self::Reader> {
        self.inner.read(ctx, path, args)
    }

    fn write(&self, ctx: &OperationContext, path: &str, args: OpWrite) -> Result<Self::Writer> {
        self.inner.write(ctx, path, args)
    }

    fn delete(&self, ctx: &OperationContext) -> Result<Self::Deleter> {
        self.inner.delete(ctx)
    }

    fn list(&self, ctx: &OperationContext, path: &str, args: OpList) -> Result<Self::Lister> {
        self.inner.list(ctx, path, args)
    }

    fn copy(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpCopy,
    ) -> Result<Self::Copier> {
        let conditional = args.if_not_exists();
        let calls = if conditional {
            &self.behavior.conditional
        } else {
            &self.behavior.ordinary
        };
        calls.fetch_add(1, Ordering::Relaxed);
        let mode = self.behavior.mode.load(Ordering::Relaxed);
        let inner = self.inner.clone();
        let ctx = ctx.clone();
        let from = from.to_owned();
        let to = to.to_owned();
        Ok(Box::new(oio::OneShotCopier::new(async move {
            if mode == 3 || (mode == 1 && conditional) {
                return Err(opendal::Error::new(
                    ErrorKind::Unsupported,
                    "fake gateway rejects copy condition",
                ));
            }
            if conditional && mode == 0 {
                match inner.stat(&ctx, &to, OpStat::default()).await {
                    Ok(_) => {
                        return Err(opendal::Error::new(
                            ErrorKind::ConditionNotMatch,
                            "destination exists",
                        ));
                    }
                    Err(e) if e.kind() == ErrorKind::NotFound => {}
                    Err(e) => return Err(e),
                }
            }
            let mut copier = inner.copy(&ctx, &from, &to, OpCopy::default())?;
            while oio::Copy::next(&mut copier).await?.is_some() {}
            oio::Copy::close(&mut copier).await
        })))
    }

    fn compose(&self, ctx: &OperationContext, to: &str, args: OpCompose) -> Result<Self::Composer> {
        self.inner.compose(ctx, to, args)
    }

    async fn rename(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpRename,
    ) -> Result<RpRename> {
        self.inner.rename(ctx, from, to, args).await
    }

    async fn restore(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpRestore,
    ) -> Result<RpRestore> {
        self.inner.restore(ctx, path, args).await
    }

    async fn presign(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpPresign,
    ) -> Result<RpPresign> {
        self.inner.presign(ctx, path, args).await
    }
}

async fn fake_store(mode: usize) -> (tempfile::TempDir, Arc<Store>, Arc<Behavior>) {
    let dir = tempfile::tempdir().unwrap();
    let behavior = Arc::new(Behavior::default());
    behavior.mode.store(mode, Ordering::Relaxed);
    let op = Operator::new(services::Fs::default().root(dir.path().to_str().unwrap()))
        .unwrap()
        .layer(GatewayLayer(behavior.clone()));
    let store = Arc::new(Store {
        name: "gateway".parse().unwrap(),
        cap: StoreCapability::from(op.info().capability()),
        op,
    });
    store.initialize().await.unwrap();
    (dir, store, behavior)
}
async fn staged(store: &Arc<Store>, bytes: &'static [u8]) -> StagedUpload {
    store
        .stage(futures::stream::iter([Ok(Bytes::from_static(bytes))]), 100)
        .await
        .unwrap()
}
async fn assert_probes_clean(store: &Store) {
    assert!(
        store
            .op
            .list_with("_sokar/probe/")
            .recursive(true)
            .await
            .unwrap()
            .iter()
            .all(|entry| !entry.metadata().is_file())
    );
}
#[tokio::test]
async fn probe_rejecting_and_ignoring_gateways_downgrades_and_cleans() {
    for mode in [1, 2] {
        let (_dir, store, behavior) = fake_store(mode).await;
        assert!(!store.cap.conditional_write.load(Ordering::Relaxed));
        assert_eq!(
            behavior.conditional.load(Ordering::Relaxed),
            if mode == 1 { 1 } else { 2 }
        );
        assert_probes_clean(&store).await;
        store.check(true).await.unwrap(); // --check-stores path re-probes the endpoint.
        assert_eq!(
            behavior.conditional.load(Ordering::Relaxed),
            if mode == 1 { 2 } else { 4 }
        );
        assert_probes_clean(&store).await;
        let mut upload = staged(&store, b"data").await;
        assert_eq!(upload.promote().await.unwrap(), UploadOutcome::Stored);
        assert_eq!(upload.promote().await.unwrap(), UploadOutcome::Deduplicated);
        assert_eq!(behavior.ordinary.load(Ordering::Relaxed), 1);
        assert_eq!(
            store
                .op
                .read(&upload.digest.key())
                .await
                .unwrap()
                .to_bytes(),
            Bytes::from_static(b"data")
        );
        upload.cleanup().await.unwrap();
    }
}
#[tokio::test]
async fn verified_copy_dedups_then_runtime_failure_falls_back_once_and_stays_disabled() {
    let (_dir, store, behavior) = fake_store(0).await;
    assert!(store.cap.conditional_write.load(Ordering::Relaxed));
    assert_eq!(behavior.conditional.load(Ordering::Relaxed), 2);
    assert_probes_clean(&store).await;
    let mut first = staged(&store, b"first").await;
    assert_eq!(first.promote().await.unwrap(), UploadOutcome::Stored);
    assert_eq!(first.promote().await.unwrap(), UploadOutcome::Deduplicated);
    assert!(store.cap.conditional_write.load(Ordering::Relaxed));
    assert_eq!(behavior.ordinary.load(Ordering::Relaxed), 0);
    behavior.mode.store(1, Ordering::Relaxed);
    let mut second = staged(&store, b"second").await;
    assert_eq!(second.promote().await.unwrap(), UploadOutcome::Stored);
    assert!(!store.cap.conditional_write.load(Ordering::Relaxed));
    assert_eq!(behavior.ordinary.load(Ordering::Relaxed), 1);
    assert_eq!(behavior.conditional.load(Ordering::Relaxed), 5);
    assert_eq!(second.promote().await.unwrap(), UploadOutcome::Deduplicated);
    assert_eq!(behavior.conditional.load(Ordering::Relaxed), 5);
    behavior.mode.store(0, Ordering::Relaxed);
    store.check(true).await.unwrap();
    assert!(!store.cap.conditional_write.load(Ordering::Relaxed));
    assert_probes_clean(&store).await;
    first.cleanup().await.unwrap();
    second.cleanup().await.unwrap();
}
#[tokio::test]
async fn runtime_fallback_failure_is_returned_after_one_attempt() {
    let (_dir, store, behavior) = fake_store(0).await;
    behavior.mode.store(3, Ordering::Relaxed);
    let mut upload = staged(&store, b"fails").await;
    assert!(matches!(
        upload.promote().await,
        Err(SokarError::Storage {
            kind: ErrorKind::Unsupported,
            ..
        })
    ));
    assert_eq!(behavior.conditional.load(Ordering::Relaxed), 3);
    assert_eq!(behavior.ordinary.load(Ordering::Relaxed), 1);
    assert!(!store.cap.conditional_write.load(Ordering::Relaxed));
    upload.cleanup().await.unwrap();
}
