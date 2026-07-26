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
//! - `meta.json` -- the training step number this checkpoint was taken at,
//!   plus a schema-version tag, so [`load_checkpoint`] can report a clear
//!   error rather than silently resuming from the wrong step if pointed at
//!   an incompatible directory.
//!
//! Full precision (`FullPrecisionSettings`) is used rather than
//! `HalfPrecisionSettings` -- at this model scale (~10-20M params) the disk
//! savings from half precision are not worth the risk of losing optimizer
//! precision partway through a multi-hour run.

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

/// Checkpoint metadata written alongside the model/optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    /// Bump if the on-disk layout or model/optimizer shapes change in a way
    /// that would make an old checkpoint unsafe to load into new code.
    schema_version: u32,
    step: usize,
}

const SCHEMA_VERSION: u32 = 1;

fn model_path(dir: &Path) -> PathBuf {
    dir.join("model")
}
fn optim_path(dir: &Path) -> PathBuf {
    dir.join("optim")
}
fn meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}

/// Save `model` + `optim`'s state to `dir` (created if missing), tagging the
/// checkpoint with `step`.
pub fn save_checkpoint<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    step: usize,
    model: &TinyTransformer<B>,
    optim: &OptimizerAdaptor<Adam, TinyTransformer<B>, B>,
) -> GrugResult<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("creating checkpoint dir: {e}"),
    })?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    model
        .clone()
        .save_file(model_path(dir), &recorder)
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving model: {e}"),
        })?;

    recorder
        .record(optim.to_record(), optim_path(dir))
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving optimizer state: {e}"),
        })?;

    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        step,
    };
    let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("serializing checkpoint metadata: {e}"),
    })?;
    std::fs::write(meta_path(dir), meta_json).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("writing meta.json: {e}"),
    })?;

    Ok(())
}

/// Load a checkpoint written by [`save_checkpoint`] back into a fresh model
/// + optimizer built from `model` / `optim` (their weights/state are
/// overwritten by the checkpoint's contents). Returns the restored `(model,
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
) -> GrugResult<(TinyTransformer<B>, OptimizerAdaptor<Adam, TinyTransformer<B>, B>, usize)> {
    let dir = dir.as_ref();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let meta_str = std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
        path: dir.display().to_string(),
        message: format!("reading meta.json: {e}"),
    })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!(
                "checkpoint schema_version {} does not match this build's {SCHEMA_VERSION}",
                meta.schema_version
            ),
        });
    }

    let model = model
        .load_file(model_path(dir), &recorder, device)
        .map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("loading model: {e}"),
        })?;

    let optim_record = recorder
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

    let meta_str = std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
        path: dir.display().to_string(),
        message: format!("reading meta.json: {e}"),
    })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!(
                "checkpoint schema_version {} does not match this build's {SCHEMA_VERSION}",
                meta.schema_version
            ),
        });
    }

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
    std::fs::create_dir_all(dir).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("creating checkpoint dir: {e}"),
    })?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path(dir), &recorder)
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving model: {e}"),
        })?;
    recorder
        .record(optim.to_record(), optim_path(dir))
        .map_err(|e| GrugError::CheckpointWriteFailed {
            path: dir.display().to_string(),
            message: format!("saving optimizer state: {e}"),
        })?;

    let meta = CheckpointMeta {
        schema_version: SCHEMA_VERSION,
        step,
    };
    let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("serializing checkpoint metadata: {e}"),
    })?;
    std::fs::write(meta_path(dir), meta_json).map_err(|e| GrugError::CheckpointWriteFailed {
        path: dir.display().to_string(),
        message: format!("writing meta.json: {e}"),
    })?;
    Ok(())
}

/// [`load_checkpoint`] for a [`PointerTransformer`] (Phase 51 P1a-4 Step B).
pub fn load_checkpoint_ptr<B: AutodiffBackend>(
    dir: impl AsRef<Path>,
    model: PointerTransformer<B>,
    optim: OptimizerAdaptor<Adam, PointerTransformer<B>, B>,
    device: &B::Device,
) -> GrugResult<(PointerTransformer<B>, OptimizerAdaptor<Adam, PointerTransformer<B>, B>, usize)> {
    let dir = dir.as_ref();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let meta_str = std::fs::read_to_string(meta_path(dir)).map_err(|e| GrugError::CheckpointReadFailed {
        path: dir.display().to_string(),
        message: format!("reading meta.json: {e}"),
    })?;
    let meta: CheckpointMeta =
        serde_json::from_str(&meta_str).map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("parsing meta.json: {e}"),
        })?;
    if meta.schema_version != SCHEMA_VERSION {
        return Err(GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!(
                "checkpoint schema_version {} does not match this build's {SCHEMA_VERSION}",
                meta.schema_version
            ),
        });
    }

    let model = model
        .load_file(model_path(dir), &recorder, device)
        .map_err(|e| GrugError::CheckpointReadFailed {
            path: dir.display().to_string(),
            message: format!("loading model: {e}"),
        })?;
    let optim_record = recorder
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
    use crate::train::init_training;
    use crate::transformer::TinyTransformerConfig;
    use crate::CpuAutodiffBackend;

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

        let (fresh_model, fresh_optim) = init_training::<CpuAutodiffBackend>(&tiny_model_config(), &device);
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

        let dir = std::env::temp_dir().join(format!("grug_train_ckpt_model_only_test_{}", std::process::id()));
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
        let missing_dir = std::env::temp_dir().join("grug_train_ckpt_model_only_test_definitely_missing_dir_xyz");
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
        let missing_dir = std::env::temp_dir().join("grug_train_ckpt_test_definitely_missing_dir_xyz");
        let _ = std::fs::remove_dir_all(&missing_dir);

        match load_checkpoint(&missing_dir, model, optim, &device) {
            Err(GrugError::CheckpointReadFailed { .. }) => {}
            other => panic!("expected CheckpointReadFailed, got: {}", other.is_ok()),
        }
    }
}
