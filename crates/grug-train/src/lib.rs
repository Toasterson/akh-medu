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
pub mod model;
pub mod multimodal;
pub mod multimodal_runtime;
pub mod pointer;
pub mod reason_train;
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
#[cfg(any(feature = "gpu", feature = "cuda"))]
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
pub use model::{Mlp, MlpConfig};
pub use multimodal::{
    CompactVisionShape, MULTIMODAL_EVIDENCE_SCHEMA_VERSION,
    MULTIMODAL_MODEL_ARTIFACT_SCHEMA_VERSION, MULTIMODAL_MODEL_CERTIFICATION_SCHEMA_VERSION,
    MultimodalAuditEvidence, MultimodalModelArtifact, MultimodalModelCertification,
    MultimodalNetworkDeniedReplayEvidence, MultimodalTextRetentionEvidence, VisionCorpusLineage,
};
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub use multimodal_runtime::GpuMultimodalModel;
pub use multimodal_runtime::{
    CpuMultimodalModel, MultimodalModel, MultimodalRuntimeError, NativePromptImage,
};
pub use pointer::{PointerParts, PointerTransformer, PointerTransformerConfig};
pub use reason_train::{
    build_reason_batch, encode_reason_example, format_reason_prompt, generate_reason_outputs,
};
pub use rule_train::{
    build_rule_batch, encode_rule_example, format_rule_prompt, generate_rule_outputs,
};
pub use tokenizer::{ByteTokenizer, Vocab};
pub use train::text_batch_loss;
pub use transformer::{Attention, CausalSelfAttention, TinyTransformer, TinyTransformerConfig};
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
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;

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
/// `burn::backend::rocm::RocmDevice`) instead of `CpuAutodiffBackend`.
#[cfg(feature = "gpu")]
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

/// The GPU inference backend under the `cuda` feature: Burn's CUDA (CubeCL)
/// backend, for NVIDIA cloud training. Same forward+backward CubeCL
/// story as the ROCm path, different vendor runtime.
///
/// Requires the CUDA runtime and NVRTC libraries (`CUDA_PATH` when they are
/// not on the default loader path).
/// **Mutually exclusive with `gpu`** — enable exactly one.
#[cfg(all(feature = "cuda", not(feature = "gpu")))]
pub type GpuBackend = burn::backend::Cuda<f32>;

/// The CUDA *training* backend: [`GpuBackend`] wrapped in `Autodiff`.
#[cfg(all(feature = "cuda", not(feature = "gpu")))]
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

/// The default device for whichever GPU backend is enabled — the seam that
/// keeps example/binary code **vendor-agnostic**: call `gpu_device()` instead
/// of naming `RocmDevice`/`CudaDevice`, and the same source builds for the
/// local 7900 XTX (`--features gpu`) and an NVIDIA cloud box
/// (`--features cuda`).
#[cfg(feature = "gpu")]
pub fn gpu_device() -> burn::backend::rocm::RocmDevice {
    Default::default()
}

/// See the `gpu` variant above.
#[cfg(all(feature = "cuda", not(feature = "gpu")))]
pub fn gpu_device() -> burn::backend::cuda::CudaDevice {
    Default::default()
}

// Enabling both vendor backends at once is a configuration error: the
// `GpuBackend`/`GpuAutodiffBackend`/`gpu_device` items would collide, and only
// one CubeCL runtime can own the process's device context. Fail loudly at
// compile time rather than producing a confusing type error.
#[cfg(all(feature = "gpu", feature = "cuda"))]
compile_error!(
    "grug-train: features `gpu` (ROCm/AMD) and `cuda` (NVIDIA) are mutually exclusive — \
     enable exactly one (local 7900 XTX: --features gpu; NVIDIA cloud: --features cuda)."
);
