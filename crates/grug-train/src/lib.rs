//! `grug-train` — Phase 51 (Grug-Brain BabyLLM) Burn beachhead.
//!
//! This crate exists to answer exactly one question (the P0 / R1 gate from
//! [ADR 087](../../../docs/ai/decisions/087-grug-brain-baby-llm-training.md)
//! and the [Phase 51 plan](../../../docs/ai/plans/2026-07-05-phase51-grug-brain-baby-llm.md)):
//! **does Burn compile and train (autograd) in this workspace, on CPU?**
//!
//! It is deliberately not the transformer, not the tokenizer, and not the
//! data pipeline -- those are separate follow-on tasks once this gate is
//! green. What lives here is the smallest real module (a 2-layer MLP) plus
//! a test that proves the full loop: forward -> loss -> `.backward()` ->
//! non-zero gradients -> optimizer step -> loss decreases.
//!
//! # Backend
//!
//! We use `burn-ndarray` (pure-Rust CPU tensor backend) wrapped in
//! `burn::backend::Autodiff` to get reverse-mode autodiff. See
//! [`CpuAutodiffBackend`] for the type alias every downstream Phase 51 crate
//! should reuse for CPU work.
//!
//! Only three Burn features are enabled (`default-features = false`):
//! `ndarray`, `autodiff`, `std`. Notably *not* `train`, which would pull in
//! `dataset` -> `burn-dataset` -> `rusqlite` -> `libsqlite3-sys 0.35`,
//! conflicting with seshat's `sea-orm` -> `sqlx` -> `libsqlite3-sys 0.26`
//! chain elsewhere in this workspace. We roll our own tiny training loop
//! instead of using `burn::train`'s `Learner`.

pub mod bpe;
pub mod checkpoint;
pub mod error;
pub mod eval;
pub mod foundation;
pub mod generate;
pub mod knowledge_runtime;
pub mod knowledge_train;
pub mod mixed_precision;
pub mod model;
pub mod multimodal;
pub mod multimodal_runtime;
pub mod pointer;
pub mod reason_train;
pub mod remediation;
pub mod rule_train;
pub mod tokenizer;
pub mod train;
pub mod transformer;
pub mod vision;

pub use bpe::BpeTokenizer;
pub use checkpoint::{load_checkpoint, load_model_only, save_checkpoint};
pub use error::{GrugError, GrugResult};
pub use eval::{
    AxisEvalLogRow, EvalLogRow, FrozenAbstentionHoldout, FrozenHoldout, FrozenRetrievalHoldout,
    FrozenTwoAxisHoldout, RetrievalEval, run_abstention_eval_fmt, run_eval, run_eval_and_log,
    run_retrieval_eval_fmt, run_two_axis_eval, run_two_axis_eval_and_log,
    run_two_axis_eval_constrained, run_two_axis_eval_fmt, run_two_axis_eval_ptr,
    run_two_axis_eval_ptr_constrained, run_two_axis_eval_raw_and_repaired_fmt,
};
pub use foundation::{
    FOUNDATION_MODEL_ARTIFACT_SCHEMA_VERSION, FOUNDATION_MODEL_MANIFEST, FoundationError,
    FoundationLineage, FoundationModelArtifact, VerifiedFoundationBundle, build_causal_batch,
    causal_batch_loss, foundation_training_languages, foundation_training_sources,
    source_identity as foundation_source_identity, verify_foundation_bundle,
};
pub use generate::{
    ConstrainedDecode, GenerateConfig, generate, generate_constrained, generate_from_prompt_ids,
    generate_from_prompt_ids_cached, generate_from_prompt_ids_cached_constrained, generate_ptr,
    generate_ptr_constrained,
};
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
pub use knowledge_runtime::GpuKnowledgeModel;
pub use knowledge_runtime::{
    CpuKnowledgeModel, DecodingMode, KnowledgeModel, KnowledgeRuntimeError,
    VerifiedKnowledgeBundle, verify_active_knowledge_bundle, verify_candidate_knowledge_bundle,
};
pub use knowledge_train::{
    CertifiedKnowledgeExam, KNOWLEDGE_MODEL_ARTIFACT_SCHEMA_VERSION,
    KNOWLEDGE_MODEL_CERTIFICATION_SCHEMA_VERSION, KnowledgeAdaptationLineage, KnowledgeEvalSlice,
    KnowledgeModelArtifact, KnowledgeModelCertification, KnowledgeModelShape, KnowledgeModelSize,
    KnowledgeSteeringLineage, KnowledgeTeacherForcedEval, build_knowledge_batch,
    compact_knowledge_prompt_ids, encode_knowledge_record, fingerprint_artifact_file,
    fingerprint_knowledge_records, format_knowledge_prompt, generate_knowledge_outputs,
    score_knowledge_outputs, validate_knowledge_steering_share,
};
pub use mixed_precision::{DEFAULT_LOSS_SCALE, MixedPrecisionTrainer};
pub use model::{Mlp, MlpConfig};
pub use multimodal::{
    CompactVisionShape, MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
    MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION, MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION,
    MultimodalAuditEvidence, MultimodalModelArtifact, MultimodalModelCertification,
    MultimodalNetworkDeniedReplayEvidence, MultimodalTextRetentionEvidence, VisionCorpusLineage,
};
#[cfg(any(feature = "gpu", feature = "cuda", feature = "metal"))]
pub use multimodal_runtime::GpuMultimodalModel;
pub use multimodal_runtime::{
    CpuMultimodalModel, MultimodalModel, MultimodalRuntimeError, NativePromptImage,
};
pub use pointer::{PointerParts, PointerTransformer, PointerTransformerConfig};
pub use reason_train::{
    build_reason_batch, encode_reason_example, format_reason_prompt, generate_reason_outputs,
};
pub use remediation::{
    DEFAULT_REMEDIATION_MAX_SHARE_PPM, RemediationLineage, RemediationSupplement,
    read_sibling_generator_metadata, validate_remediation_share,
};
pub use rule_train::{
    build_rule_batch, encode_rule_example, format_rule_prompt, generate_rule_outputs,
};
pub use tokenizer::{ByteTokenizer, Vocab};
pub use train::text_batch_loss;
pub use transformer::{
    Attention, AttentionBackend, CausalSelfAttention, PositionalEncoding, TinyTransformer,
    TinyTransformerConfig,
};
pub use vision::{
    CompactConvVisionStem, CompactConvVisionStemConfig, CompactVisionTransformer,
    CompactVisionTransformerConfig, NativeVisionTransformer, NativeVisionTransformerConfig,
    OCR_PROMPT, PatchGrid, VisionError, VisionPatchStem, VisionPatchStemConfig, VisualTokenBudget,
    build_vision_ocr_batch, compact_vision_batch_loss, generate_compact_vision_ocr,
    generate_vision_ocr, init_compact_vision_training, init_vision_training, patchify_rgb,
    rgb_batch_to_nchw, stack_patch_grids, to_eval_compact_vision_model, to_eval_vision_model,
    train_compact_text_step_guarded, train_compact_vision_step, train_compact_vision_step_guarded,
    train_vision_step, visual_token_budget,
};

#[cfg(feature = "grad_checkpointing")]
use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
use burn::backend::{Autodiff, NdArray};

/// The plain (non-differentiable) CPU inference backend: `burn-ndarray`.
///
/// Use this for forward-only / eval-only code paths.
pub type CpuBackend = NdArray<f32>;

/// The CPU *training* backend: `burn-ndarray` wrapped in `Autodiff`.
///
/// `Autodiff<B>` is Burn's backend decorator that records the computation
/// graph and gives every tensor `.backward()` / `.grad()`. Any module
/// generic over `B: burn::tensor::backend::AutodiffBackend` can be trained
/// with this backend; swap it for an opt-in GPU backend (CUDA/ROCm) without
/// touching model code.
///
/// `Autodiff<B, C = NoCheckpointing>` also takes a `C: CheckpointStrategy`
/// type parameter (ADR 104 D3): `NoCheckpointing` (the implicit default,
/// used here unless the `grad_checkpointing` cargo feature is enabled)
/// keeps every forward activation live for the backward pass, exactly the
/// pre-ADR-104 behavior. Enabling `grad_checkpointing` switches this alias
/// to `BalancedCheckpointing`, which discards and recomputes selected
/// activations during `.backward()` to trade some recompute time for lower
/// peak training memory -- needed to fit an 8192-context run's `[batch,
/// heads, seq, seq]` attention activations (see `PositionalEncoding::Rope`
/// / `attention_chunk` in `transformer.rs`) on a single GPU. Since every
/// trainer in this crate (`train.rs`, `examples/train_foundation.rs`,
/// `examples/train_knowledge.rs`) is generic over `B: AutodiffBackend`
/// rather than naming `Autodiff<CpuBackend>`/`Autodiff<GpuBackend>`
/// directly, flipping this one alias (and its GPU counterpart below) is
/// the entire change needed to opt every trainer into checkpointing.
#[cfg(not(feature = "grad_checkpointing"))]
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
/// See the non-`grad_checkpointing` variant above.
#[cfg(feature = "grad_checkpointing")]
pub type CpuAutodiffBackend = Autodiff<CpuBackend, BalancedCheckpointing>;

/// The GPU inference backend, behind the opt-in `gpu` cargo feature: Burn's
/// ROCm/HIP (CubeCL) backend, targeting this box's 7900 XTX (gfx1100).
///
/// Requires `ROCM_PATH` or `CUBECL_ROCM_PATH` to point at the ROCm install
/// (e.g. `/opt/rocm` or `/opt/rocm-6.3.4`) at build time.
#[cfg(feature = "gpu")]
pub type GpuBackend = burn::backend::Rocm<f32>;

/// The GPU *training* backend: [`GpuBackend`] wrapped in `Autodiff`.
///
/// A pure type-alias swap for [`CpuAutodiffBackend`] -- every model in this
/// crate is generic over `B: AutodiffBackend`, so training on ROCm requires
/// no model-code changes, only using this alias (and a
/// `burn::backend::rocm::RocmDevice`) instead of `CpuAutodiffBackend`. See
/// [`CpuAutodiffBackend`] for the `grad_checkpointing` feature this alias
/// also honors.
#[cfg(all(feature = "gpu", not(feature = "grad_checkpointing")))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
/// See the non-`grad_checkpointing` variant above.
#[cfg(all(feature = "gpu", feature = "grad_checkpointing"))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend, BalancedCheckpointing>;

/// The GPU inference backend under the `cuda` feature: Burn's CUDA (CubeCL)
/// backend, for NVIDIA cloud training. Same forward+backward CubeCL
/// story as the ROCm path, different vendor runtime.
///
/// Requires the CUDA runtime and NVRTC libraries (`CUDA_PATH` when they are
/// not on the default loader path).
/// **Mutually exclusive with `gpu`** — enable exactly one.
#[cfg(all(feature = "cuda", not(feature = "gpu")))]
pub type GpuBackend = burn::backend::Cuda<f32>;

/// The CUDA *training* backend: [`GpuBackend`] wrapped in `Autodiff`. See
/// [`CpuAutodiffBackend`] for the `grad_checkpointing` feature this alias
/// also honors.
#[cfg(all(
    feature = "cuda",
    not(feature = "gpu"),
    not(feature = "grad_checkpointing")
))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
/// See the non-`grad_checkpointing` variant above.
#[cfg(all(feature = "cuda", not(feature = "gpu"), feature = "grad_checkpointing"))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend, BalancedCheckpointing>;

/// The GPU inference backend under the `metal` feature (ADR 106, Phase 51
/// P6.1): Burn's `burn-wgpu` (CubeCL) backend, targeting Apple Silicon via
/// the Metal Shading Language compiler target. Same forward+backward CubeCL
/// story as the ROCm/CUDA paths, but routed through `wgpu` (Burn's
/// `metal` feature = `["wgpu", "burn-wgpu/metal", ...]`) instead of a
/// vendor-specific runtime -- no separate toolchain install beyond the
/// Metal support macOS/Xcode already ships.
///
/// `burn::backend::Metal<F>` is itself a type alias for
/// `burn::backend::Wgpu<F, i32, u8>` (`burn-wgpu` picks the Metal graphics
/// API automatically on macOS via `AutoGraphicsApi`/`WgpuDevice::DefaultDevice`
/// -- see [`gpu_device`] below).
/// **Mutually exclusive with `gpu` and `cuda`** — enable exactly one.
/// Under `metal_f16` (ADR 107 decision 3c, Phase 51 P6.1b(c)) this becomes
/// [`burn::backend::Metal<burn::tensor::f16>`] instead -- see the variant
/// below and [`mixed_precision`].
#[cfg(all(
    feature = "metal",
    not(feature = "metal_f16"),
    not(feature = "gpu"),
    not(feature = "cuda")
))]
pub type GpuBackend = burn::backend::Metal<f32>;

/// The `metal_f16` variant of [`GpuBackend`] above (ADR 107 decision 3c,
/// Phase 51 P6.1b(c)): the trainable forward/backward graph runs in f16
/// instead of f32, which is where the actual activation-memory win from
/// [`mixed_precision`] comes from. Burn 0.21 has no mixed precision of its
/// own, so on its own this would also put every trainer's Adam moments in
/// f16 (ADR 107's whole-backend f16 measurement, which converged visibly
/// slower) -- `mixed_precision::MixedPrecisionTrainer` pairs this backend
/// with [`GpuMasterAutodiffBackend`] (f32) to keep the optimizer state at
/// full precision. Implies `metal` (`metal_f16 = ["metal"]` in
/// `Cargo.toml`), so it is mutually exclusive with `gpu`/`cuda` via the
/// same `compile_error!`s those already trigger against `metal`.
#[cfg(all(feature = "metal_f16", not(feature = "gpu"), not(feature = "cuda")))]
pub type GpuBackend = burn::backend::Metal<burn::tensor::f16>;

/// The Metal *training* backend: [`GpuBackend`] wrapped in `Autodiff`. See
/// [`CpuAutodiffBackend`] for the `grad_checkpointing` feature this alias
/// also honors. Unchanged by `metal_f16` -- it wraps whichever `GpuBackend`
/// (f32 or f16) is active.
#[cfg(all(
    feature = "metal",
    not(feature = "gpu"),
    not(feature = "cuda"),
    not(feature = "grad_checkpointing")
))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
/// See the non-`grad_checkpointing` variant above.
#[cfg(all(
    feature = "metal",
    not(feature = "gpu"),
    not(feature = "cuda"),
    feature = "grad_checkpointing"
))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend, BalancedCheckpointing>;

/// The f32 *master-weights* backend for [`mixed_precision`] under
/// `metal_f16` (ADR 107 decision 3c, Phase 51 P6.1b(c)) -- the same
/// physical Metal device as [`GpuBackend`]/[`GpuAutodiffBackend`] (both are
/// `Wgpu<F, i32, u8>` for different `F`, so both use `WgpuDevice`; see
/// [`gpu_device`]), just a different float element. Only the master's
/// weights and Adam moments live here; the trainable forward/backward graph
/// stays on [`GpuAutodiffBackend`] (f16).
#[cfg(feature = "metal_f16")]
pub type GpuMasterBackend = burn::backend::Metal<f32>;

/// [`GpuMasterBackend`] wrapped in `Autodiff` -- required because
/// `burn::optim::Optimizer::step` is only defined for `M:
/// AutodiffModule<B>, B: AutodiffBackend` (see `mixed_precision.rs`), even
/// though the master itself never runs a forward/backward pass. Honors
/// `grad_checkpointing` for consistency with [`GpuAutodiffBackend`], though
/// the master's parameter-only "graph" never triggers checkpointing in
/// practice.
#[cfg(all(feature = "metal_f16", not(feature = "grad_checkpointing")))]
pub type GpuMasterAutodiffBackend = Autodiff<GpuMasterBackend>;
/// See the non-`grad_checkpointing` variant above.
#[cfg(all(feature = "metal_f16", feature = "grad_checkpointing"))]
pub type GpuMasterAutodiffBackend = Autodiff<GpuMasterBackend, BalancedCheckpointing>;

/// The default device for whichever GPU backend is enabled — the seam that
/// keeps example/binary code **vendor-agnostic**: call `gpu_device()` instead
/// of naming `RocmDevice`/`CudaDevice`/`WgpuDevice`, and the same source
/// builds for the local 7900 XTX (`--features gpu`), an NVIDIA cloud box
/// (`--features cuda`), and a local Apple-silicon Mac (`--features metal`).
#[cfg(feature = "gpu")]
pub fn gpu_device() -> burn::backend::rocm::RocmDevice {
    Default::default()
}

/// See the `gpu` variant above.
#[cfg(all(feature = "cuda", not(feature = "gpu")))]
pub fn gpu_device() -> burn::backend::cuda::CudaDevice {
    Default::default()
}

/// See the `gpu` variant above. `WgpuDevice::default()` is
/// `WgpuDevice::DefaultDevice`, which asks `wgpu` for the best available
/// device under the active graphics API -- on macOS that resolves to the
/// system's Metal device, no index/selection needed for the single-GPU
/// Apple-silicon boxes this feature targets.
#[cfg(all(feature = "metal", not(feature = "gpu"), not(feature = "cuda")))]
pub fn gpu_device() -> burn::backend::wgpu::WgpuDevice {
    Default::default()
}

// Enabling more than one vendor backend at once is a configuration error:
// the `GpuBackend`/`GpuAutodiffBackend`/`gpu_device` items would collide, and
// only one CubeCL runtime can own the process's device context. Fail loudly
// at compile time rather than producing a confusing type error.
#[cfg(all(feature = "gpu", feature = "cuda"))]
compile_error!(
    "grug-train: features `gpu` (ROCm/AMD) and `cuda` (NVIDIA) are mutually exclusive — \
     enable exactly one (local 7900 XTX: --features gpu; NVIDIA cloud: --features cuda)."
);
#[cfg(all(feature = "gpu", feature = "metal"))]
compile_error!(
    "grug-train: features `gpu` (ROCm/AMD) and `metal` (Apple Silicon) are mutually exclusive \
     — enable exactly one (local 7900 XTX: --features gpu; local Apple Silicon: --features metal)."
);
#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!(
    "grug-train: features `cuda` (NVIDIA) and `metal` (Apple Silicon) are mutually exclusive — \
     enable exactly one (NVIDIA cloud: --features cuda; local Apple Silicon: --features metal)."
);
