//! akh-medu CLI: neuro-symbolic AI engine.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
use akh_medu::provenance::DerivationKind;
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

    /// Query the knowledge base using spreading-activation inference.
    Query {
        /// Seed symbol IDs (comma-separated).
        #[arg(long)]
        seeds: String,

        /// Number of results to return.
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Maximum inference depth.
        #[arg(long, default_value = "1")]
        max_depth: usize,
    },

    /// Show engine info and statistics.
    Info,

    /// Manage skillpacks.
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },
}

#[derive(Subcommand)]
enum SkillAction {
    /// List all discovered skillpacks.
    List,
    /// Load (discover + warm + activate) a skillpack.
    Load {
        /// Skill name (directory name under skills/).
        name: String,
    },
    /// Unload (deactivate) a skillpack.
    Unload {
        /// Skill name.
        name: String,
    },
    /// Show details about a skillpack.
    Info {
        /// Skill name.
        name: String,
    },
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

        Commands::Query {
            seeds,
            top_k,
            max_depth,
        } => {
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

            let query = InferenceQuery {
                seeds: seed_ids,
                top_k,
                max_depth,
                ..Default::default()
            };

            let result = engine.infer(&query).into_diagnostic()?;

            println!("Inference results (top {top_k}, depth {max_depth}):");
            for (i, (sym, confidence)) in result.activations.iter().enumerate() {
                println!("  {}. {} (confidence: {:.4})", i + 1, sym, confidence);
            }

            if !result.provenance.is_empty() {
                println!("\nProvenance:");
                for record in &result.provenance {
                    let kind_desc = match &record.kind {
                        DerivationKind::Extracted => "extracted".to_string(),
                        DerivationKind::Seed => "seed".to_string(),
                        DerivationKind::GraphEdge { from, predicate } => {
                            format!("graph edge from {} via {}", from, predicate)
                        }
                        DerivationKind::VsaRecovery {
                            from,
                            predicate,
                            similarity,
                        } => {
                            format!(
                                "VSA recovery from {} via {} (sim: {:.4})",
                                from, predicate, similarity
                            )
                        }
                        DerivationKind::Analogy { a, b, c } => {
                            format!("analogy {}:{} :: {}:?", a, b, c)
                        }
                        DerivationKind::FillerRecovery {
                            subject,
                            predicate,
                        } => {
                            format!("filler recovery ({}, {})", subject, predicate)
                        }
                        DerivationKind::Reasoned => "reasoned".to_string(),
                        DerivationKind::Aggregated => "aggregated".to_string(),
                    };
                    println!(
                        "  {} depth={} confidence={:.4} [{}]",
                        record.derived_id, record.depth, record.confidence, kind_desc
                    );
                }
            }
        }

        Commands::Info => {
            let engine = Engine::new(config).into_diagnostic()?;
            println!("{}", engine.info());
        }

        Commands::Skill { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                SkillAction::List => {
                    let skills = engine.list_skills();
                    if skills.is_empty() {
                        println!("No skillpacks discovered.");
                    } else {
                        println!("Skillpacks ({}):", skills.len());
                        for skill in &skills {
                            println!(
                                "  {} ({}) [{}] - {}",
                                skill.id, skill.version, skill.state, skill.description
                            );
                        }
                    }
                }
                SkillAction::Load { name } => {
                    let activation = engine.load_skill(&name).into_diagnostic()?;
                    println!("Loaded skill: {}", activation.skill_id);
                    println!("  triples: {}", activation.triples_loaded);
                    println!("  rules:   {}", activation.rules_loaded);
                    println!("  memory:  {} bytes", activation.memory_bytes);
                }
                SkillAction::Unload { name } => {
                    engine.unload_skill(&name).into_diagnostic()?;
                    println!("Unloaded skill: {name}");
                }
                SkillAction::Info { name } => {
                    let info = engine.skill_info(&name).into_diagnostic()?;
                    println!("Skill: {}", info.id);
                    println!("  name:        {}", info.name);
                    println!("  version:     {}", info.version);
                    println!("  description: {}", info.description);
                    println!("  state:       {}", info.state);
                    println!("  domains:     {}", info.domains.join(", "));
                    println!("  triples:     {}", info.triple_count);
                    println!("  rules:       {}", info.rule_count);
                }
            }
        }
    }

    Ok(())
}
