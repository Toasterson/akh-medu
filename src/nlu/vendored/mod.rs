//! Vendored + patched candle-transformers quantized model loaders.
//!
//! These are copies of `candle_transformers::models::quantized_qwen2` and
//! `…::quantized_gemma3` (v0.10.2), patched to expose the **post-norm
//! last-position hidden state** before the output (lm_head) projection via
//! `hidden_states()` / `forward_with_hidden()`. Upstream's `ModelWeights`
//! fields (`norm`, `output`, `layers`) are private, so the hidden state cannot
//! be reached without owning the model code — hence this in-tree fork.
//!
//! The hidden state is what the Neural→VSA bridge (Phase 26) consumes; it is
//! the decisive reason the engine uses Candle rather than a C++/HTTP backend.
//!
//! Only the import paths changed (`crate::` → `candle_transformers::`, and
//! `candle` aliased to `candle_core`) plus the added hidden-state methods; the
//! transformer math is byte-for-byte upstream.

pub mod quantized_gemma3;
pub mod quantized_gemma4;
pub mod quantized_qwen2;
