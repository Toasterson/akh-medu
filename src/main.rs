//! akh-medu CLI: neuro-symbolic AI engine.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::symbol::SymbolId;
use akh_medu::vsa::Dimension;

#[derive(Parser)]
#[command(name = "akh-medu", version, about = "Neuro-symbolic AI engine")]
struct Cli {
    /// Data directory for persistent storage.
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    /// Hypervector dimension.
    #[arg(long, global = true, default_value = "10000")]
    dimension: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new akh-medu data directory.
    Init,

    /// Ingest triples from a JSON file.
    Ingest {
        /// Path to JSON file with triples.
        #[arg(long)]
        file: PathBuf,
    },

    /// Query the knowledge base.
    Query {
        /// Seed symbol IDs (comma-separated).
        #[arg(long)]
        seeds: String,

        /// Number of results to return.
        #[arg(long, default_value = "10")]
        top_k: usize,
    },

    /// Show engine info and statistics.
    Info,
}

fn main() -> Result<()> {
    miette::set_hook(Box::new(|_| {
        Box::new(
            miette::MietteHandlerOpts::new()
                .terminal_links(true)
                .unicode(true)
                .context_lines(3)
                .build(),
        )
    }))
    .ok(); // Ignore error if hook already set (e.g., in tests)

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let config = EngineConfig {
        dimension: Dimension(cli.dimension),
        data_dir: cli.data_dir.clone(),
        ..Default::default()
    };

    match cli.command {
        Commands::Init => {
            let data_dir = cli.data_dir.unwrap_or_else(|| PathBuf::from(".akh-medu"));
            let config = EngineConfig {
                data_dir: Some(data_dir.clone()),
                dimension: Dimension(cli.dimension),
                ..Default::default()
            };
            let engine = Engine::new(config).into_diagnostic()?;
            println!("Initialized akh-medu at {}", data_dir.display());
            println!("{}", engine.info());
        }

        Commands::Ingest { file } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let content = std::fs::read_to_string(&file).into_diagnostic()?;

            // Simple JSON format: [{"s": 1, "p": 2, "o": 3}, ...]
            let triples: Vec<serde_json::Value> =
                serde_json::from_str(&content).into_diagnostic()?;

            let mut count = 0;
            for val in &triples {
                let s = val["s"].as_u64().unwrap_or(0);
                let p = val["p"].as_u64().unwrap_or(0);
                let o = val["o"].as_u64().unwrap_or(0);

                if let (Some(s), Some(p), Some(o)) =
                    (SymbolId::new(s), SymbolId::new(p), SymbolId::new(o))
                {
                    engine.add_triple(&Triple::new(s, p, o)).into_diagnostic()?;
                    count += 1;
                }
            }
            println!("Ingested {count} triples from {}", file.display());
            println!("{}", engine.info());
        }

        Commands::Query { seeds, top_k } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let seed_ids: Vec<SymbolId> = seeds
                .split(',')
                .filter_map(|s| {
                    let raw: u64 = s.trim().parse().ok()?;
                    SymbolId::new(raw)
                })
                .collect();

            if seed_ids.is_empty() {
                miette::bail!("no valid seed symbol IDs provided");
            }

            // Ensure seeds exist in item memory
            for &seed in &seed_ids {
                engine.item_memory().get_or_create(engine.ops(), seed);
            }

            // Bundle seed vectors and search
            let ops = engine.ops();
            let vecs: Vec<_> = seed_ids
                .iter()
                .map(|&id| engine.item_memory().get_or_create(ops, id))
                .collect();
            let refs: Vec<&_> = vecs.iter().collect();
            let bundled = ops.bundle(&refs).into_diagnostic()?;

            let results = engine.search_similar(&bundled, top_k).into_diagnostic()?;

            println!("Query results (top {top_k}):");
            for (i, r) in results.iter().enumerate() {
                println!("  {}. {} (similarity: {:.4})", i + 1, r.symbol_id, r.similarity);
            }
        }

        Commands::Info => {
            let engine = Engine::new(config).into_diagnostic()?;
            println!("{}", engine.info());
        }
    }

    Ok(())
}
