#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Autodiff
//!
//! This autodiff library is a part of the Burn project. It is a standalone crate
//! that can be used to perform automatic differentiation on tensors. It is
//! designed to be used with the Burn Tensor crate, but it can be used with any
//! tensor library that implements the `Backend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Checkpoint module.
pub mod checkpoint;
#[cfg(feature = "distributed")]
/// Distributed utils.
pub mod distributed;
/// Gradients module.
pub mod grads;
/// Operation module.
pub mod ops;

// akh-medu vendor patch (P6.1b, ADR 107 / docs/ai/decisions/107-training-memory-turboquant-vs-recompute-attention.md):
// `graph` and `tensor` were `pub(crate)` in upstream 0.21.0, which makes
// `AutodiffTensor<B>` (and the `NodeRef`/`Node`/`Requirement`/`Step` types the
// public `ops::Backward`/`ops::OpsPrep` API already takes and returns in its
// signatures) unreachable from any downstream crate, even though every field
// and method on them is individually `pub`. That blocks exactly the
// "backend extension: custom autodiff op" pattern the Burn book describes
// (and which `ops/tensor.rs`'s own `Matmul`/`Add`/etc. use internally) from
// ever being implemented outside this crate. This patch does not add,
// remove, or change any item -- it only widens two module declarations from
// `pub(crate)` to `pub` so their already-`pub` contents become reachable.
// Zero behavior change for every existing caller.
/// Graph module. Made `pub` (see the akh-medu vendor patch note above) so a
/// downstream crate can hold `NodeRef`s and `Ops::parents` values.
// Its contents were never documented to the `missing_docs` bar this crate
// otherwise holds itself to (they were `pub(crate)`, so the lint never
// checked them); silence rather than opportunistically half-document code
// this patch didn't otherwise touch.
#[allow(missing_docs)]
pub mod graph;
// Exported for backend extension
pub use graph::NodeId;
/// Tensor module. Made `pub` (see the akh-medu vendor patch note above) so a
/// downstream crate can construct/inspect `AutodiffTensor<B>` -- the type
/// `ops::OpsPrep::finish` already returns -- to implement a custom
/// `ops::Backward<B, N>` op.
#[allow(missing_docs)]
pub mod tensor;
pub(crate) mod utils;

mod backend;

pub(crate) mod runtime;

pub use backend::*;

/// A facade around for HashMap and HashSet.
/// This avoids elaborate import wrangling having to happen in every module.
mod collections {
    #[cfg(not(feature = "std"))]
    pub use hashbrown::{HashMap, HashSet};
    #[cfg(feature = "std")]
    pub use std::collections::{HashMap, HashSet};
}
