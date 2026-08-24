//! Vendored + patched candle-transformers quantized model loaders.
//!
//! The Qwen2 and Gemma3 loaders began as Candle 0.10.2 copies, patched to expose
//! the **post-norm last-position hidden state** before the output (lm_head)
//! projection via `hidden_states()` / `forward_with_hidden()`. Upstream's
//! `ModelWeights` fields (`norm`, `output`, `layers`) are private, so the hidden
//! state cannot be reached without owning the model code — hence this in-tree
//! fork. Gemma4 and Qwen35 are in-tree implementations rather than upstream
//! Candle loaders; Qwen35 is derived from the custom Qwen3.8 GGUF contract.
//!
//! The hidden state is what the Neural→VSA bridge (Phase 26) consumes; it is
//! the decisive reason the engine uses Candle rather than a C++/HTTP backend.
//!
//! In the legacy copied loaders, only import paths (`crate::` →
//! `candle_transformers::`, with `candle` aliased to `candle_core`) and the
//! hidden-state methods differ from their recorded upstream versions.

pub mod quantized_gemma3;
pub mod quantized_gemma4;
pub mod quantized_qwen2;
pub mod quantized_qwen35;
