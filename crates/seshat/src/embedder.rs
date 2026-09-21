//! ONNX-based sentence embedding using all-MiniLM-L6-v2.
//!
//! This module is a thin re-export over the shared `akh-embed` crate, which
//! both Seshat and `akh-transcripts` now depend on for their sentence
//! embedder — see plan increment 4 ("akh-embed extraction") and
//! `docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md` §3. All
//! `crate::embedder::X` paths that existed before extraction keep resolving
//! from here.

pub use akh_embed::{EMBEDDING_DIM, SentenceEmbedder, SharedEmbedder, resolve_model_dir};
