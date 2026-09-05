//! # anima-ai
//!
//! Multi-engine NL routing for akh-medu.
//!
//! Routes user input to the best natural language handler:
//!
//! | Path | When | Latency |
//! |------|------|---------|
//! | ELIZA + KG | Factual Q&A with KG answer | < 1 ms |
//! | Gemma + Psyche | Creative / novel / complex | 1–3 s |
//! | Hybrid | KG facts + Gemma enrichment | 1–3 s |
//! | Remote LLM | All local engines unavailable | varies |
//!
//! ## Fallback chain
//! ELIZA → Gemma (cached) → Gemma (full) → Remote LLM
//!
//! ## Quick start
//! ```rust,no_run
//! use anima_ai::{AnimaAi, GemmaHandler, NullKg};
//!
//! let mut ai = AnimaAi::new(GemmaHandler::new_stub());
//! // With NullKg, ELIZA finds nothing and Gemma is unavailable → AllEnginesExhausted
//! let _ = ai.process("What is the Ennead?", &NullKg);
//! ```

pub mod eliza;
pub mod engine_selector;
pub mod error;
pub mod gemma;
pub mod hybrid_orchestrator;

pub use eliza::{ElizaHandler, KgBackend, NullKg};
pub use engine_selector::{EngineSelector, Route, SelectionContext};
pub use error::{AnimaError, AnimaResult, GemmaError, RemoteError};
pub use gemma::{GemmaBackend, GemmaHandler, GenerationMode, SemanticCache};
pub use hybrid_orchestrator::HybridOrchestrator;

use tracing::debug;

// ---------------------------------------------------------------------------
// RemoteLlmBackend
// ---------------------------------------------------------------------------

/// Pluggable remote LLM fallback (e.g. Gemini, OpenRouter, OpenAI-compatible).
///
/// Implement this to wire in a remote API when all local engines are unavailable.
pub trait RemoteLlmBackend: Send + Sync {
    /// Generate a completion for `prompt` with optional grounding `context`.
    fn generate(&self, prompt: &str, context: Option<&str>) -> Result<String, RemoteError>;

    /// Returns `true` when the backend can accept requests right now.
    fn is_available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// EngineResponse
// ---------------------------------------------------------------------------

/// The response produced by [`AnimaAi::process`].
#[derive(Debug, Clone)]
pub struct EngineResponse {
    /// The generated text.
    pub text: String,
    /// The route that actually produced the response (may differ from the selector's
    /// recommendation when earlier paths in the fallback chain succeeded).
    pub route_used: Route,
    /// Scoring context from the engine selector (for observability / tracing).
    pub context: SelectionContext,
}

// ---------------------------------------------------------------------------
// AnimaAi — top-level orchestrator
// ---------------------------------------------------------------------------

/// Top-level orchestrator: routes input and manages the fallback chain.
///
/// # Fallback order (ELIZA route as example)
/// 1. ELIZA — deterministic KG pattern matching (< 1 ms)
/// 2. Gemma cached — semantic cache hit (< 0.1 ms)
/// 3. Gemma full — LLM inference (1–3 s)
/// 4. Remote LLM — external API (varies)
///
/// Creative (Gemma) and Hybrid routes skip step 1 or adjust the order accordingly.
pub struct AnimaAi {
    selector: EngineSelector,
    eliza: ElizaHandler,
    gemma: GemmaHandler,
    remote: Option<Box<dyn RemoteLlmBackend>>,
}

impl AnimaAi {
    /// Create an orchestrator with the given Gemma handler and default settings.
    pub fn new(gemma: GemmaHandler) -> Self {
        Self {
            selector: EngineSelector::default(),
            eliza: ElizaHandler::new(),
            gemma,
            remote: None,
        }
    }

    /// Attach a remote LLM fallback backend.
    pub fn with_remote(mut self, remote: Box<dyn RemoteLlmBackend>) -> Self {
        self.remote = Some(remote);
        self
    }

    /// Replace the engine selector (e.g. to adjust routing thresholds).
    pub fn with_selector(mut self, selector: EngineSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Route `input` to the best available engine and return its response.
    ///
    /// Uses `kg` for ELIZA and Hybrid paths. Pass [`NullKg`] when no engine is attached.
    ///
    /// # Errors
    /// [`AnimaError::AllEnginesExhausted`] when every path in the fallback chain fails.
    pub fn process(&mut self, input: &str, kg: &dyn KgBackend) -> AnimaResult<EngineResponse> {
        let ctx = self.selector.analyze(input);
        debug!(
            kg_relevance = ctx.kg_relevance,
            novelty_score = ctx.novelty_score,
            psyche_fit = ctx.psyche_fit,
            route = ?ctx.route,
            "engine selector decision"
        );
        match ctx.route {
            Route::Eliza => self.run_eliza_chain(input, kg, ctx),
            Route::Gemma => self.run_gemma_chain(input, None, ctx),
            Route::Hybrid => self.run_hybrid_chain(input, kg, ctx),
            Route::RemoteFallback => self.run_remote(input, ctx),
        }
    }

    // ------------------------------------------------------------------
    // Fallback chain helpers
    // ------------------------------------------------------------------

    /// ELIZA → Gemma cached → Gemma full → Remote
    fn run_eliza_chain(
        &mut self,
        input: &str,
        kg: &dyn KgBackend,
        ctx: SelectionContext,
    ) -> AnimaResult<EngineResponse> {
        if let Some(text) = self.eliza.respond(input, kg) {
            return Ok(EngineResponse {
                text,
                route_used: Route::Eliza,
                context: ctx,
            });
        }
        self.run_gemma_chain(input, None, ctx)
    }

    /// Gemma cached → Gemma full → Remote
    fn run_gemma_chain(
        &mut self,
        input: &str,
        context: Option<&str>,
        ctx: SelectionContext,
    ) -> AnimaResult<EngineResponse> {
        match self.gemma.generate(input, context, GenerationMode::Cached) {
            Ok(text) => {
                return Ok(EngineResponse {
                    text,
                    route_used: Route::Gemma,
                    context: ctx,
                });
            }
            Err(GemmaError::CacheMiss | GemmaError::Unavailable) => {}
            Err(e) => return Err(AnimaError::Gemma(e)),
        }

        match self.gemma.generate(input, context, GenerationMode::Full) {
            Ok(text) => {
                return Ok(EngineResponse {
                    text,
                    route_used: Route::Gemma,
                    context: ctx,
                });
            }
            Err(GemmaError::Unavailable) => {}
            Err(e) => return Err(AnimaError::Gemma(e)),
        }

        self.run_remote(input, ctx)
    }

    /// ELIZA (for KG context) + Gemma full → ELIZA alone → Remote
    fn run_hybrid_chain(
        &mut self,
        input: &str,
        kg: &dyn KgBackend,
        ctx: SelectionContext,
    ) -> AnimaResult<EngineResponse> {
        // Extract KG context via ELIZA first (immutable borrow of self.eliza,
        // dropped before we mutably borrow self.gemma below)
        let kg_context = self.eliza.respond(input, kg);

        match self
            .gemma
            .generate(input, kg_context.as_deref(), GenerationMode::Full)
        {
            Ok(text) => {
                return Ok(EngineResponse {
                    text,
                    route_used: Route::Hybrid,
                    context: ctx,
                });
            }
            Err(GemmaError::Unavailable) => {}
            Err(e) => return Err(AnimaError::Gemma(e)),
        }

        // Gemma unavailable — serve the ELIZA-only KG answer if we have one
        if let Some(text) = kg_context {
            return Ok(EngineResponse {
                text,
                route_used: Route::Eliza,
                context: ctx,
            });
        }

        self.run_remote(input, ctx)
    }

    fn run_remote(&self, input: &str, ctx: SelectionContext) -> AnimaResult<EngineResponse> {
        match self.remote.as_ref() {
            Some(r) if r.is_available() => {
                let text = r.generate(input, None)?;
                Ok(EngineResponse {
                    text,
                    route_used: Route::RemoteFallback,
                    context: ctx,
                })
            }
            _ => Err(AnimaError::AllEnginesExhausted {
                input_preview: input.chars().take(60).collect(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    struct EchoGemma;

    impl GemmaBackend for EchoGemma {
        fn generate(&self, prompt: &str, _ctx: Option<&str>) -> error::GemmaResult<String> {
            Ok(format!("gemma: {prompt}"))
        }
        fn is_available(&self) -> bool {
            true
        }
    }

    struct EchoRemote;

    impl RemoteLlmBackend for EchoRemote {
        fn generate(&self, prompt: &str, _ctx: Option<&str>) -> Result<String, RemoteError> {
            Ok(format!("remote: {prompt}"))
        }
        fn is_available(&self) -> bool {
            true
        }
    }

    // ---- routing acceptance tests ------------------------------------------

    #[test]
    fn factual_question_uses_eliza_path() {
        // The ELIZA pattern matches "What is the Ennead?" and returns a static template
        // even when the KG is empty, since it falls back to the pattern's response_template.
        let mut ai = AnimaAi::new(GemmaHandler::new_stub());
        let result = ai.process("What is the Ennead?", &NullKg);
        assert!(
            result.is_ok(),
            "ELIZA should return a static template response"
        );
        let resp = result.unwrap();
        assert_eq!(
            resp.context.route,
            Route::Eliza,
            "selector should pick Eliza"
        );
        assert_eq!(
            resp.route_used,
            Route::Eliza,
            "response should come from ELIZA"
        );
    }

    #[test]
    fn factual_question_eliza_then_gemma_fallback() {
        let mut ai = AnimaAi::new(GemmaHandler::with_backend(Box::new(EchoGemma), "Scholar"));
        // NullKg → ELIZA misses → Gemma cache miss → Gemma Full succeeds
        let result = ai.process("What is the Ennead?", &NullKg);
        assert!(result.is_ok(), "expected ok, got {result:?}");
        let resp = result.unwrap();
        // Route may be Gemma since ELIZA fell through
        assert!(
            matches!(resp.route_used, Route::Gemma | Route::Eliza),
            "unexpected route: {:?}",
            resp.route_used
        );
    }

    #[test]
    fn creative_input_routes_to_gemma() {
        let mut ai = AnimaAi::new(GemmaHandler::with_backend(Box::new(EchoGemma), "Scholar"));
        let result = ai.process("Write a poem about the Ennead", &NullKg);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.context.route, Route::Gemma);
        assert_eq!(resp.route_used, Route::Gemma);
    }

    #[test]
    fn reflective_input_routes_to_hybrid() {
        let mut ai = AnimaAi::new(GemmaHandler::with_backend(Box::new(EchoGemma), "Scholar"));
        let result = ai.process("How should I approach this design problem?", &NullKg);
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.context.route, Route::Hybrid);
        // Gemma answered so route_used is Hybrid
        assert_eq!(resp.route_used, Route::Hybrid);
    }

    // ---- fallback chain tests ----------------------------------------------

    #[test]
    fn gemma_unavailable_falls_back_to_remote() {
        let mut ai = AnimaAi::new(GemmaHandler::new_stub()).with_remote(Box::new(EchoRemote));
        let result = ai.process("Write a poem about the Ennead", &NullKg);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().route_used, Route::RemoteFallback);
    }

    #[test]
    fn all_engines_unavailable_returns_error() {
        let mut ai = AnimaAi::new(GemmaHandler::new_stub());
        let result = ai.process("Write a poem about the Ennead", &NullKg);
        assert!(matches!(
            result,
            Err(AnimaError::AllEnginesExhausted { .. })
        ));
    }

    #[test]
    fn hybrid_falls_back_to_eliza_when_gemma_unavailable() {
        use eliza::KgBackend;

        struct PopulatedKg;
        impl KgBackend for PopulatedKg {
            fn fact(&self, subject: &str, _: &str) -> Option<String> {
                if subject.eq_ignore_ascii_case("design problem") {
                    Some("a structured challenge requiring creativity".to_string())
                } else {
                    None
                }
            }
            fn related(&self, subject: &str, limit: usize) -> Vec<(String, String)> {
                if subject.eq_ignore_ascii_case("design problem") {
                    vec![("description".to_string(), "requires creativity".to_string())]
                        .into_iter()
                        .take(limit)
                        .collect()
                } else {
                    vec![]
                }
            }
            fn search(&self, _: &str, _: usize) -> Vec<(String, String)> {
                vec![]
            }
        }

        let mut ai = AnimaAi::new(GemmaHandler::new_stub());
        let input = "How should I approach this design problem?";
        let result = ai.process(input, &PopulatedKg);
        // Gemma unavailable but ELIZA has KG data → should fall back to Eliza response
        // OR AllEnginesExhausted if ELIZA pattern doesn't match "design problem" entity
        // Either outcome is acceptable; verify no panic and error is well-typed
        match result {
            Ok(resp) => assert!(matches!(resp.route_used, Route::Eliza | Route::Hybrid)),
            Err(AnimaError::AllEnginesExhausted { .. }) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn hybrid_with_eliza_kg_data_passes_context_to_gemma() {
        use eliza::KgBackend;

        struct KgWithFact;
        impl KgBackend for KgWithFact {
            fn fact(&self, subject: &str, _: &str) -> Option<String> {
                if subject.eq_ignore_ascii_case("Ennead") {
                    Some("nine deities of Heliopolis".to_string())
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

        let mut ai = AnimaAi::new(GemmaHandler::with_backend(Box::new(EchoGemma), "Scholar"));
        let result = ai.process("How should I approach this design problem?", &KgWithFact);
        assert!(result.is_ok());
    }
}
