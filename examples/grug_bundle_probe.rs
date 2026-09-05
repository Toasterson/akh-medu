//! Verify and optionally exercise a local BabyLLM artifact through the same
//! [`ChatBackend`](akh_medu::agent::chat_backend::ChatBackend) boundary used by
//! `akhomed`.

#[cfg(feature = "grug-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::PathBuf;

    use akh_medu::agent::chat_backend::{ChatBackend, GrugChatBackend};
    use clap::Parser;
    use serde_json::Value;

    #[derive(Parser)]
    struct Args {
        /// Directory containing active-model.json.
        #[arg(long)]
        bundle: PathBuf,
        /// When present, run one user turn after verifying the bundle.
        #[arg(long)]
        prompt: Option<String>,
        /// Optional OpenAI-style JSON tool array offered to the probe turn.
        #[arg(long)]
        tools: Option<PathBuf>,
        /// Small values are useful for an untrained wiring smoke test.
        #[arg(long, default_value_t = 384)]
        max_new_tokens: usize,
        /// Load candidate-model.json for offline evaluation. Without this flag,
        /// only the exam-certified active-model.json is accepted.
        #[arg(long)]
        candidate: bool,
    }

    let args = Args::parse();
    let backend = if args.candidate {
        GrugChatBackend::load_candidate_for_evaluation(&args.bundle)?
    } else {
        GrugChatBackend::load(&args.bundle)?
    }
    .with_max_new_tokens(args.max_new_tokens)?;
    eprintln!(
        "verified params={} step={} context={} contract={} multimodal={}",
        backend.parameter_count(),
        backend.checkpoint_step(),
        backend.capabilities().context_window.unwrap_or_default(),
        backend.contract_hash(),
        backend.is_multimodal(),
    );

    let Some(prompt) = args.prompt else {
        return Ok(());
    };
    let tools: Vec<Value> = match args.tools {
        Some(path) => serde_json::from_reader(std::fs::File::open(path)?)?,
        None => Vec::new(),
    };
    let conversation = vec![serde_json::json!({"role": "user", "content": prompt})];
    let turn = backend.chat_with_tools("probe", &conversation, &tools, 0)?;
    serde_json::to_writer_pretty(
        std::io::stdout().lock(),
        &serde_json::json!({
            "content": turn.content,
            "tool_calls": turn.tool_calls.iter().map(|call| serde_json::json!({
                "name": call.name,
                "arguments": call.arguments,
            })).collect::<Vec<_>>(),
            "prompt_tokens": turn.prompt_tokens,
            "completion_tokens": turn.completion_tokens,
        }),
    )?;
    println!();
    Ok(())
}

#[cfg(not(feature = "grug-backend"))]
fn main() {
    eprintln!("this example requires --features grug-backend");
    std::process::exit(2);
}
