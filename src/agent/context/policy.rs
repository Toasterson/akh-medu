//! Deferred policy seams (Phase 44k / 44l) — **recorded, not built**.
//!
//! Two axes of the assembly runtime are gated on infrastructure that does not
//! exist on this branch. They are defined here as traits + no-op defaults so the
//! seam is stable and the wiring point is obvious — but their real bodies are
//! intentionally absent, because implementing them now would be speculation
//! against an API that isn't there. Each carries its gate in the doc comment.
//!
//! - **44k [`KvCachePolicy`]** — Candle KV-cache retention (which cached prefix to
//!   keep resident across turns, to skip re-prefill). Gated on Candle exposing
//!   KV-cache control (ADR 037); today the Candle backend owns its cache opaquely,
//!   so there is nothing for a policy to drive. The default
//!   [`BackendManagedKv`] defers entirely to the backend — today's behaviour.
//! - **44l [`RetrievalPolicy`]** — the retrieval swap-axis (KG recall ↔
//!   Heka/pgvector RAG). Gated on Phase 42 (Heka) landing, which is **absent** on
//!   this branch — episodic recall is pure-KG — so there is no second retriever to
//!   choose between. The default [`KgOnlyRetrieval`] selects the only source that
//!   exists.

use super::types::AssembledContext;

/// 44k — which portion of a Candle KV cache to retain across turns. **Deferred.**
///
/// See the module docs: gated on ADR 037 Candle KV-cache control.
pub trait KvCachePolicy: Send + Sync {
    /// Tokens of the cached prefix to keep resident for the next turn, given the
    /// just-assembled context. `None` ⇒ let the backend manage its own cache
    /// (the only behaviour available today).
    fn retain_prefix_tokens(&self, _assembled: &AssembledContext) -> Option<usize> {
        None
    }
}

/// The no-op KV policy: defer entirely to the backend (current behaviour).
#[derive(Debug, Default, Clone, Copy)]
pub struct BackendManagedKv;
impl KvCachePolicy for BackendManagedKv {}

/// Where recalled context comes from (44l). Only [`KnowledgeGraph`] exists on this
/// branch; `Heka` (Phase 42 pgvector RAG) is the deferred second source.
///
/// [`KnowledgeGraph`]: RetrievalSource::KnowledgeGraph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalSource {
    /// Pure-KG episodic recall (`agent::memory::recall_episodes`) — today's only
    /// retriever.
    KnowledgeGraph,
}

/// 44l — which retriever supplies recalled context. **Deferred** (only KG exists).
///
/// See the module docs: gated on Phase 42 (Heka/pgvector) landing.
pub trait RetrievalPolicy: Send + Sync {
    /// The retrieval source to use. Defaults to the only one present.
    fn source(&self) -> RetrievalSource {
        RetrievalSource::KnowledgeGraph
    }
}

/// The only retrieval policy available today: pure-KG recall.
#[derive(Debug, Default, Clone, Copy)]
pub struct KgOnlyRetrieval;
impl RetrievalPolicy for KgOnlyRetrieval {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn defaults_are_no_ops() {
        let assembled = AssembledContext {
            messages: Vec::new(),
            tools: Vec::new(),
            section_tokens: BTreeMap::new(),
            dropped: Vec::new(),
        };
        assert_eq!(BackendManagedKv.retain_prefix_tokens(&assembled), None);
        assert_eq!(KgOnlyRetrieval.source(), RetrievalSource::KnowledgeGraph);
    }
}
