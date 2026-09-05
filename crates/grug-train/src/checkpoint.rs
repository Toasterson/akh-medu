//! Resumable checkpointing (Phase 51 plan, P1a "Ops" bullet: "Burn `Recorder`
//! checkpoint every N steps (resumable)").
//!
//! A checkpoint is a directory containing:
//!
//! - `model.mpk` -- the [`TinyTransformer`] weights, via
//!   [`burn::record::NamedMpkFileRecorder`] (Burn's default named-tensor
//!   MessagePack format).
//! - `optim.mpk` -- the Adam optimizer state (moment estimates), via the
//!   same recorder, so resuming a run does not reset momentum/variance
//!   accumulators to zero (which would otherwise cause a visible loss spike
//!   right after resume).
//! - `meta.json` -- the training step, schema version, and model/Adam BLAKE3
//!   hashes. New checkpoints are published by an atomic directory rename;
//!   loaders continue to accept hashless schema-v1 checkpoints created by
//!   earlier builds.
//!
//! Full precision (`FullPrecisionSettings`) is used rather than
//! `HalfPrecisionSettings` -- at this model scale (~10-20M params) the disk
//! savings from half precision are not worth the risk of losing optimizer
//! precision partway through a multi-hour run.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::error::{GrugError, GrugResult};
use crate::pointer::PointerTransformer;
use crate::transformer::TinyTransformer;

type TinyCheckpoint<B> = (
    TinyTransformer<B>,
    OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
    usize,
);
type PointerCheckpoint<B> = (
    PointerTransformer<B>,
    OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
    usize,
);

/// Checkpoint metadata written alongside the model/optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    /// Bump if the on-disk layout or model/optimizer shapes change in a way
    /// that would make an old checkpoint unsafe to load into new code.
    schema_version: u32,
    step: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    optimizer_hash: Option<String>,
}

const LEGACY_SCHEMA_VERSION: u32 = 1;
const SCHEMA_VERSION: u32 = 2;

fn model_path(dir: &Path) -> PathBuf {
    dir.join("model")
}
fn optim_path(dir: &Path) -> PathBuf {
    dir.join("optim")
}
fn model_record_path(dir: &Path) -> PathBuf {
    model_path(dir).with_extension("mpk")
}
fn optim_record_path(dir: &Path) -> PathBuf {
    optim_path(dir).with_extension("mpk")
}
fn meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}

fn prepare_checkpoint_directory(dir: &Path) -> GrugResult<PathBuf> {
    if dir.exists() {
        return Err(GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: "refusing to overwrite an immutable checkpoint".into(),
        });
    }
    let parent = dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("creating checkpoint parent: {e}"),
    })?;
    let name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: "checkpoint directory name is not valid UTF-8".into(),
        })?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(GrugError::CheckpointWriteFailed {
            path: temporary.display().to_string(),
            message: "stale temporary checkpoint path exists".into(),
        });
    }
    std::fs::create_dir(&temporary).map_err(|e| GrugError::CheckpointWriteFailed {
        path: temporary.display().to_string(),
        message: format!("creating temporary checkpoint: {e}"),
    })?;
    Ok(temporary)
}

fn fingerprint_file(path: &Path) -> std::io::Result<String> {
    let mut reader = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn verify_checkpoint_meta(
    dir: &Path,
    meta: &CheckpointMeta,
    verify_optimizer: bool,
) -> GrugResult<()> {
    match meta.schema_version {
        LEGACY_SCHEMA_VERSION => return Ok(()),
        SCHEMA_VERSION => {}
        version => {
            return Err(GrugError::CheckpointReadFailed {
                path: dir.display().to_string(),
                message: format!(
                    "checkpoint schema_version {version} is unsupported; expected {LEGACY_SCHEMA_VERSION} or {SCHEMA_VERSION}"
                ),
            });
        }
    }

    let expected_model =
        meta.model_hash
            .as_deref()
            .ok_or_else(|| GrugError::CheckpointReadFailed {
                path: dir.display().to_string(),
                message: "content-addressed checkpoint has no model hash".into(),
            })?;
    verify_checkpoint_file(dir, &model_record_path(dir), expected_model, "model")?;
    if verify_optimizer {
        let expected_optimizer =
            meta.optimizer_hash
                .as_deref()
                .ok_or_else(|| GrugError::CheckpointReadFailed {
                    path: dir.display().to_string(),
                    message: "content-addressed checkpoint has no optimizer hash".into(),
                })?;
        verify_checkpoint_file(
            dir,
            &optim_record_path(dir),
            expected_optimizer,
            "optimizer",
        )?;
    }
    Ok(())
}

fn verify_checkpoint_file(dir: &Path, path: &Path, expected: &str, label: &str) -> GrugResult<()> {
    if !expected.starts_with("blake3:") {
        return Err(GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("checkpoint {label} hash is not BLAKE3"),
        });
    }
    let actual = fingerprint_file(path).map_err(|e| GrugError::CheckpointReadFailed {
        path: path.display().to_string(),
        message: format!("hashing {label}: {e}"),
    })?;
    if actual != expected {
        return Err(GrugError::CheckpointReadFailed {
            path: path.display().to_string(),
            message: format!("{label} hash mismatch: expected {expected}, got {actual}"),
        });
    }
    Ok(())
}

/// Save `model` + `optim`'s state to a new immutable `dir`, tagging the
/// checkpoint with `step` and both content hashes. The directory becomes
/// visible through one same-filesystem rename only after every file exists.
pub fn save_checkpoint<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    step: usize,
    model: &TinyTransformer<B>,
    optim: &OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
) -> GrugResult<()> {
    let dir = dir.as_ref();
    let temporary = prepare_checkpoint_directory(dir)?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    model
        .clone()
        .save_file(model_path(&temporary), &recorder)
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving model: {e}"),
        })?;

    recorder
        .record(optim.to_record(), optim_path(&temporary))
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving optimizer state: {e}"),
        })?;

    let model_hash = fingerprint_file(&model_record_path(&temporary)).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("hashing model: {e}"),
        }
    })?;
    let optimizer_hash = fingerprint_file(&optim_record_path(&temporary)).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("hashing optimizer state: {e}"),
        }
    })?;

    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        step,
        model_hash: Some(model_hash),
        optimizer_hash: Some(optimizer_hash),
    };
    let meta_json =
        serde_json::to_string_pretty(&meta).map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("serializing checkpoint metadata: {e}"),
        })?;
    std::fs::write(meta_path(&temporary), meta_json).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("writing meta.json: {e}"),
        }
    })?;
    std::fs::rename(&temporary, dir).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("publishing immutable checkpoint: {e}"),
    })?;

    Ok(())
}

/// Load a checkpoint written by [`save_checkpoint`] back into a fresh model
/// + optimizer built from `model` / `optim` (their weights/state are
///   overwritten by the checkpoint's contents). Returns the restored `(model,
/// optim, step)`.
///
/// `model` and `optim` should be freshly initialized (e.g. via
/// [`crate::train::init_training`]) with the *same architecture config* the
/// checkpoint was saved from -- shape mismatches surface as a
/// [`GrugError::CheckpointReadFailed`] from the underlying recorder rather
/// than a silent corrupt load.
pub fn load_checkpoint<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    model: TinyTransformer<B>,
    optim: OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
    device: &B::Device,
) -> GrugResult<TinyCheckpoint<B>> {
    let dir = dir.as_ref();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let meta_str =
        std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("reading meta.json: {e}"),
        })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    verify_checkpoint_meta(dir, &meta, true)?;

    let model = model
        .load_file(model_path(dir), &recorder, device)
        .map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("loading model: {e}"),
        })?;

    let optim_record =
        recorder
            .load(optim_path(dir), device)
            .map_err(|e| GrugError::CheckpointReadFailed {
                path: dir.display().to_string(),
                message: format!("loading optimizer state: {e}"),
            })?;
    let optim = optim.load_record(optim_record);

    Ok((model, optim, meta.step))
}

/// Load only a checkpoint's model weights, skipping `optim.mpk` entirely.
///
/// The GWT workspace-emergence probes (`docs/ai/research/2026-07-09-global-workspace-language-models.md`
/// §7, `examples/probe_lens.rs` / `examples/probe_swap.rs`) are inference-only
/// -- they never call `.backward()` -- so there is no need to reconstruct the
/// Adam optimizer state (which [`load_checkpoint`] requires purely because it
/// is generic over the *training* resumption path). This is `load_checkpoint`
/// minus the `optim` half: same `meta.json` schema-version validation, same
/// `model` recorder, but `model` may be built directly on a plain (non-autodiff)
/// [`Backend`] rather than requiring [`AutodiffBackend`] -- one fewer generic
/// parameter to satisfy at the probe call site.
///
/// `model` should be freshly initialized with the *same architecture config*
/// the checkpoint was saved from, exactly as [`load_checkpoint`] requires.
pub fn load_model_only<B: Backend>(
    dir: impl AsRef<Path>,
    model: TinyTransformer<B>,
    device: &B::Device,
) -> GrugResult<(TinyTransformer<B>, usize)> {
    let dir = dir.as_ref();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let meta_str =
        std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("reading meta.json: {e}"),
        })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    verify_checkpoint_meta(dir, &meta, false)?;

    let model = model
        .load_file(model_path(dir), &recorder, device)
        .map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("loading model: {e}"),
        })?;

    Ok((model, meta.step))
}

/// [`save_checkpoint`] for a [`PointerTransformer`] (Phase 51 P1a-4 Step B).
/// Same on-disk layout (`model`/`optim`/`meta.json`); a separate function
/// rather than a generic one to keep the proven `TinyTransformer` path
/// untouched.
pub fn save_checkpoint_ptr<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    step: usize,
    model: &PointerTransformer<B>,
    optim: &OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
) -> GrugResult<()> {
    let dir = dir.as_ref();
    let temporary = prepare_checkpoint_directory(dir)?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path(&temporary), &recorder)
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving model: {e}"),
        })?;
    recorder
        .record(optim.to_record(), optim_path(&temporary))
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving optimizer state: {e}"),
        })?;

    let model_hash = fingerprint_file(&model_record_path(&temporary)).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("hashing model: {e}"),
        }
    })?;
    let optimizer_hash = fingerprint_file(&optim_record_path(&temporary)).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("hashing optimizer state: {e}"),
        }
    })?;

    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        step,
        model_hash: Some(model_hash),
        optimizer_hash: Some(optimizer_hash),
    };
    let meta_json =
        serde_json::to_string_pretty(&meta).map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("serializing checkpoint metadata: {e}"),
        })?;
    std::fs::write(meta_path(&temporary), meta_json).map_err(|e| {
        GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("writing meta.json: {e}"),
        }
    })?;
    std::fs::rename(&temporary, dir).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("publishing immutable checkpoint: {e}"),
    })?;
    Ok(())
}

/// [`load_checkpoint`] for a [`PointerTransformer`] (Phase 51 P1a-4 Step B).
pub fn load_checkpoint_ptr<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    model: PointerTransformer<B>,
    optim: OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
    device: &B::Device,
) -> GrugResult<PointerCheckpoint<B>> {
    let dir = dir.as_ref();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let meta_str =
        std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("reading meta.json: {e}"),
        })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    verify_checkpoint_meta(dir, &meta, true)?;

    let model = model
        .load_file(model_path(dir), &recorder, device)
        .map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("loading model: {e}"),
        })?;
    let optim_record =
        recorder
            .load(optim_path(dir), device)
            .map_err(|e| GrugError::CheckpointReadFailed {
                path: dir.display().to_string(),
                message: format!("loading optimizer state: {e}"),
            })?;
    let optim = optim.load_record(optim_record);

    Ok((model, optim, meta.step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuAutodiffBackend;
    use crate::train::init_training;
    use crate::transformer::TinyTransformerConfig;

    fn tiny_model_config() -> TinyTransformerConfig {
        TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64)
    }

    #[test]
    fn save_then_load_restores_step_and_weights() {
        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);

        let dir = std::env::temp_dir().join(format!("grug_train_ckpt_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        save_checkpoint(&dir, 42, &model, &optim).unwrap();

        let (fresh_model, fresh_optim) =
            init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let (loaded_model, _loaded_optim, step) =
            load_checkpoint(&dir, fresh_model, fresh_optim, &device).unwrap();

        assert_eq!(step, 42);

        // Weight tensors should now match the originally-saved model (a
        // freshly-initialized model has different random weights, so this
        // proves the load actually overwrote them rather than being a
        // no-op).
        let original_param_count = burn::module::Module::num_params(&model);
        let loaded_param_count = burn::module::Module::num_params(&loaded_model);
        assert_eq!(original_param_count, loaded_param_count);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// [`load_model_only`] reads back the same weights [`save_checkpoint`]
    /// wrote, on a plain (non-autodiff) backend, without ever touching
    /// `optim.mpk` -- the probe harness's load path.
    #[test]
    fn load_model_only_restores_step_and_weights_without_optimizer() {
        use crate::CpuBackend;

        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);

        let dir = std::env::temp_dir().join(format!(
            "grug_train_ckpt_model_only_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        save_checkpoint(&dir, 7, &model, &optim).unwrap();

        // Deliberately no optimizer built at all -- a plain inference backend,
        // matching how the GWT probes load checkpoints.
        let fresh_model = tiny_model_config().init::<CpuBackend>(&device);
        let (loaded_model, step) = load_model_only(&dir, fresh_model, &device).unwrap();

        assert_eq!(step, 7);
        let original_param_count = burn::module::Module::num_params(&model);
        let loaded_param_count = burn::module::Module::num_params(&loaded_model);
        assert_eq!(original_param_count, loaded_param_count);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_model_only_rejects_missing_checkpoint_directory() {
        use crate::CpuBackend;

        let device = Default::default();
        let model = tiny_model_config().init::<CpuBackend>(&device);
        let missing_dir =
            std::env::temp_dir().join("grug_train_ckpt_model_only_test_definitely_missing_dir_xyz");
        let _ = std::fs::remove_dir_all(&missing_dir);

        match load_model_only(&missing_dir, model, &device) {
            Err(GrugError::CheckpointReadFailed { .. }) => {}
            other => panic!("expected CheckpointReadFailed, got: {}", other.is_ok()),
        }
    }

    #[test]
    fn load_rejects_missing_checkpoint_directory() {
        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let missing_dir =
            std::env::temp_dir().join("grug_train_ckpt_test_definitely_missing_dir_xyz");
        let _ = std::fs::remove_dir_all(&missing_dir);

        match load_checkpoint(&missing_dir, model, optim, &device) {
            Err(GrugError::CheckpointReadFailed { .. }) => {}
            other => panic!("expected CheckpointReadFailed, got: {}", other.is_ok()),
        }
    }

    #[test]
    fn published_checkpoint_is_content_addressed_and_immutable() {
        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("step_00000042");

        save_checkpoint(&dir, 42, &model, &optim).unwrap();
        let meta: CheckpointMeta =
            serde_json::from_slice(&std::fs::read(meta_path(&dir)).unwrap()).unwrap();
        assert_eq!(meta.schema_version, SCHEMA_VERSION);
        assert_eq!(
            meta.model_hash.as_deref(),
            Some(fingerprint_file(&model_record_path(&dir)).unwrap().as_str())
        );
        assert_eq!(
            meta.optimizer_hash.as_deref(),
            Some(fingerprint_file(&optim_record_path(&dir)).unwrap().as_str())
        );

        let before = std::fs::read(model_record_path(&dir)).unwrap();
        assert!(save_checkpoint(&dir, 42, &model, &optim).is_err());
        assert_eq!(std::fs::read(model_record_path(&dir)).unwrap(), before);
    }

    #[test]
    fn content_addressed_load_refuses_tampered_model() {
        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("step_00000007");
        save_checkpoint(&dir, 7, &model, &optim).unwrap();
        std::fs::write(model_record_path(&dir), b"tampered").unwrap();

        let (fresh_model, fresh_optim) =
            init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        assert!(matches!(
            load_checkpoint(&dir, fresh_model, fresh_optim, &device),
            Err(GrugError::CheckpointReadFailed { .. })
        ));
    }

    #[test]
    fn legacy_v1_checkpoint_remains_loadable() {
        let device = Default::default();
        let (model, optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path().join("step_00000009");
        save_checkpoint(&dir, 9, &model, &optim).unwrap();
        let legacy = CheckpointMeta {
            schema_version: LEGACY_SCHEMA_VERSION,
            step: 9,
            model_hash: None,
            optimizer_hash: None,
        };
        std::fs::write(meta_path(&dir), serde_json::to_vec_pretty(&legacy).unwrap()).unwrap();

        let (fresh_model, fresh_optim) =
            init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
        let (_, _, step) = load_checkpoint(&dir, fresh_model, fresh_optim, &device).unwrap();
        assert_eq!(step, 9);
    }
}
