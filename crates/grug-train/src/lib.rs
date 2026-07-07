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
pub mod generate;
pub mod model;
pub mod tokenizer;
pub mod train;
pub mod transformer;

pub use bpe::BpeTokenizer;
pub use checkpoint::{load_checkpoint, save_checkpoint};
pub use error::{GrugError, GrugResult};
pub use eval::{
    run_eval, run_eval_and_log, run_two_axis_eval, run_two_axis_eval_and_log,
    run_two_axis_eval_constrained, AxisEvalLogRow, EvalLogRow, FrozenHoldout, FrozenTwoAxisHoldout,
};
pub use generate::{generate, generate_constrained, GenerateConfig};
pub use model::{Mlp, MlpConfig};
pub use tokenizer::{ByteTokenizer, Vocab};
pub use transformer::{Attention, CausalSelfAttention, TinyTransformer, TinyTransformerConfig};

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
/// with this backend; swap it for a GPU backend (wgpu/cuda/rocm) later
/// without touching model code (Phase 51 P0 gate note: GPU backends are
/// deliberately *not* wired yet).
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
