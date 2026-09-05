//! Real end-to-end agentic flow with the Candle **gemma4-12b** backing (Phase 47 demo).
//!
//! Builds an `Engine` + `Agent`, attaches gemma4-12b as the chat/agentic backing with Phase-47
//! speculative decode enabled (via the shared `InferenceOptions`), and drives `run_agentic_turn`
//! on a multi-step task that *requires tool calls* — so we can watch how well gemma4-12b adapts to
//! agentic work: tool selection, argument synthesis (grammar-constrained JSON), and multi-step
//! reasoning over its own knowledge graph.
//!
//! Run (gemma4 GGUF must be under `<models>/llm/`):
//! ```sh
//! AKH_LLM_MODEL=gemma4-12b AKH_AGENT_MAX_TOOLS=14 AKH_SPEC_DECODE=1 \
//!   cargo run --release --example agentic_flow --features hip
//! ```

#[cfg(feature = "candle-backend")]
fn main() -> miette::Result<()> {
    use akh_medu::agent::chat_backend::CandleChatBackend;
    use akh_medu::agent::{Agent, AgentConfig};
    use akh_medu::engine::{Engine, EngineConfig};
    use akh_medu::message::{AkhMessage, MessageSink};
    use akh_medu::nlu::candle_backend::CandleBackend;
    use akh_medu::paths::AkhPaths;
    use std::sync::{Arc, Mutex};

    // Print every structured agent message so the whole OODA trace is visible.
    struct TraceSink;
    impl MessageSink for TraceSink {
        fn emit(&self, msg: &AkhMessage) {
            match msg {
                AkhMessage::Reasoning { step, .. } => println!("  \x1b[36m[think]\x1b[0m {step}"),
                AkhMessage::ToolResult {
                    tool,
                    success,
                    output,
                } => {
                    let mark = if *success {
                        "\x1b[32m✓\x1b[0m"
                    } else {
                        "\x1b[31m✗\x1b[0m"
                    };
                    println!(
                        "  \x1b[33m[tool {mark} {tool}]\x1b[0m {}",
                        output.replace('\n', " ")
                    );
                }
                AkhMessage::Narrative { text, .. } => println!("  \x1b[32m[answer]\x1b[0m {text}"),
                AkhMessage::Fact { text, .. } => println!("  [fact] {text}"),
                AkhMessage::System { text } => println!("  [system] {text}"),
                AkhMessage::Error { code, message, .. } => {
                    println!("  \x1b[31m[error {code}]\x1b[0m {message}")
                }
                other => println!("  [msg] {other:?}"),
            }
        }
    }

    let paths = AkhPaths::resolve()?;
    let models = paths.models_dir();

    // Engine + Agent (Agent::new registers the full tool catalog).
    let engine = Arc::new(Engine::new(EngineConfig::default())?);
    let mut agent = Agent::new(Arc::clone(&engine), AgentConfig::default())?;

    // gemma4-12b as the in-process agentic backing, with Phase-47 spec decode enabled through
    // the shared inference options (the same path `akhomed` + the `inference_config` tool use).
    let mut cb = CandleBackend::try_load(&models).ok_or_else(|| {
        miette::miette!(
            "no Candle GGUF under {}/llm/ (set AKH_LLM_MODEL=gemma4-12b)",
            models.display()
        )
    })?;
    let opts = engine.inference_options().clone();
    opts.set_spec_decode(true);
    cb.set_inference_options(opts);
    let system_prompt = "You are an akh: a precise neuro-symbolic assistant. You can call tools \
        to read and write your knowledge graph. When a task asks you to store or look up a fact, \
        CALL THE APPROPRIATE TOOL rather than answering from memory. Otherwise answer directly."
        .to_string();
    agent.set_chat_backend(Arc::new(CandleChatBackend::new(
        Arc::new(Mutex::new(cb)),
        system_prompt,
    )));

    let task = std::env::var("AKH_AGENT_TASK").unwrap_or_else(|_| {
        "Use your tools to do this in two steps. Step 1: store the fact that Ada Lovelace \
        wrote the first computer algorithm. Step 2: query your knowledge graph to answer the \
        question — who wrote the first computer algorithm? Give the final answer in one sentence."
            .to_string()
    });

    println!("model        : gemma4-12b (Candle, Device::Hip)  spec_decode: on");
    println!("task         : {task}\n");
    println!("=== AGENTIC TRACE (run_agentic_turn) ===");
    let t0 = std::time::Instant::now();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let outcome =
        agent.run_agentic_turn(&engine, "agentic-demo", &task, &TraceSink, &cancel, false);
    println!(
        "=== {} in {:.1}s ===",
        if outcome.is_some() {
            "completed"
        } else {
            "no backend / fell back"
        },
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("This example requires the `candle-backend` feature (use --features hip).");
    std::process::exit(1);
}
