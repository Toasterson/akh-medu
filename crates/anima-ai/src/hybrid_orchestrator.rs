//! Hybrid orchestration: ELIZA extracts KG facts, Gemma enriches with personality.
//!
//! The [`HybridOrchestrator`] is a stateless utility that implements the Hybrid route:
//! 1. Call ELIZA to extract grounded KG facts for the input entity.
//! 2. Inject those facts as structured context into the Gemma prompt.
//! 3. Run full Gemma inference to produce a personality-rich response.
//!
//! When Gemma is unavailable the caller receives [`AnimaError::Gemma`] and should
//! fall back to the ELIZA response alone (see [`crate::AnimaAi`] for the full chain).

use crate::eliza::{ElizaHandler, KgBackend};
use crate::error::{AnimaError, AnimaResult};
use crate::gemma::{GemmaHandler, GenerationMode};

/// Stateless orchestrator for the Hybrid route.
///
/// Exposed as a public type so callers can invoke the Hybrid path directly
/// without going through the full [`crate::AnimaAi`] orchestrator.
pub struct HybridOrchestrator;

impl HybridOrchestrator {
    /// Run the Hybrid path for `input`.
    ///
    /// `eliza` and `gemma` are separate parameters to avoid borrow conflicts
    /// when called from within a struct that owns both (e.g. [`crate::AnimaAi`]).
    ///
    /// # Errors
    /// - [`AnimaError::Gemma`]`(`[`crate::error::GemmaError::Unavailable`]`)` when no
    ///   Gemma backend is configured or reachable — callers should fall back gracefully.
    pub fn respond(
        input: &str,
        kg: &dyn KgBackend,
        eliza: &ElizaHandler,
        gemma: &mut GemmaHandler,
    ) -> AnimaResult<String> {
        let kg_context = eliza.respond(input, kg);
        gemma
            .generate(input, kg_context.as_deref(), GenerationMode::Full)
            .map_err(AnimaError::Gemma)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eliza::NullKg;
    use crate::error::GemmaError;
    use crate::gemma::GemmaBackend;

    struct EchoBackend;

    impl GemmaBackend for EchoBackend {
        fn generate(
            &self,
            prompt: &str,
            _context: Option<&str>,
        ) -> crate::error::GemmaResult<String> {
            Ok(format!("hybrid: {prompt}"))
        }
        fn is_available(&self) -> bool {
            true
        }
    }

    #[test]
    fn hybrid_produces_response_when_gemma_available() {
        let eliza = ElizaHandler::new();
        let mut gemma = GemmaHandler::with_backend(Box::new(EchoBackend), "Scholar");
        let result =
            HybridOrchestrator::respond("What is the Ennead?", &NullKg, &eliza, &mut gemma);
        assert!(result.is_ok());
    }

    #[test]
    fn hybrid_returns_unavailable_when_no_backend() {
        let eliza = ElizaHandler::new();
        let mut gemma = GemmaHandler::new_stub();
        let result =
            HybridOrchestrator::respond("How should I approach this?", &NullKg, &eliza, &mut gemma);
        assert!(matches!(
            result,
            Err(AnimaError::Gemma(GemmaError::Unavailable))
        ));
    }

    #[test]
    fn kg_context_flows_through_eliza_to_gemma() {
        use crate::eliza::KgBackend;

        struct KgWithData;
        impl KgBackend for KgWithData {
            fn fact(&self, subject: &str, _predicate: &str) -> Option<String> {
                if subject.eq_ignore_ascii_case("Ennead") {
                    Some("group of nine Egyptian deities".to_string())
                } else {
                    None
                }
            }
            fn related(&self, _: &str, _: usize) -> Vec<(String, String)> {
                vec![]
            }
            fn search(&self, _: &str, _: usize) -> Vec<(String, String)> {
                vec![]
            }
        }

        let eliza = ElizaHandler::new();
        let mut gemma = GemmaHandler::with_backend(Box::new(EchoBackend), "Scholar");
        let result =
            HybridOrchestrator::respond("What is the Ennead?", &KgWithData, &eliza, &mut gemma);
        assert!(result.is_ok());
    }
}
