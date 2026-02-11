//! akh-medu CLI: neuro-symbolic AI engine.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::autonomous::{GapAnalysisConfig, RuleEngineConfig, SchemaDiscoveryConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::error::EngineError;
use akh_medu::glyph;
use akh_medu::graph::traverse::TraversalConfig;
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;
use akh_medu::pipeline::{Pipeline, PipelineData, PipelineStage, StageConfig, StageKind};
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

    /// Ingest triples from a file (JSON, CSV, or text format).
    Ingest {
        /// Path to file with triples.
        #[arg(long)]
        file: PathBuf,

        /// File format: "json" (default), "csv", or "text".
        #[arg(long, default_value = "json")]
        format: String,

        /// CSV format: "spo" (subject,predicate,object) or "entity" (headers=predicates).
        #[arg(long, default_value = "spo")]
        csv_format: String,

        /// Maximum sentences to process for text format.
        #[arg(long, default_value = "100")]
        max_sentences: usize,
    },

    /// Load all bundled skills, run grounding, run inference.
    Bootstrap,

    /// Query the knowledge base using spreading-activation inference.
    Query {
        /// Seed symbols (comma-separated names or IDs, e.g. "Sun,Moon" or "1,2").
        #[arg(long)]
        seeds: String,

        /// Number of results to return.
        #[arg(long, default_value = "10")]
        top_k: usize,

        /// Maximum inference depth.
        #[arg(long, default_value = "1")]
        max_depth: usize,
    },

    /// Traverse the knowledge graph from seed nodes using BFS.
    Traverse {
        /// Seed symbols (comma-separated names or IDs).
        #[arg(long)]
        seeds: String,

        /// Maximum traversal depth.
        #[arg(long, default_value = "3")]
        max_depth: usize,

        /// Only follow these predicates (comma-separated, optional).
        #[arg(long)]
        predicates: Option<String>,

        /// Minimum confidence threshold.
        #[arg(long, default_value = "0.0")]
        min_confidence: f32,

        /// Maximum number of triples to collect.
        #[arg(long, default_value = "1000")]
        max_results: usize,

        /// Output format: "summary" or "json".
        #[arg(long, default_value = "summary")]
        format: String,
    },

    /// Run a SPARQL query against the knowledge graph.
    Sparql {
        /// Inline SPARQL query string.
        #[arg(long)]
        query: Option<String>,

        /// Path to a SPARQL query file (.rq).
        #[arg(long)]
        file: Option<PathBuf>,
    },

    /// Simplify a symbolic expression using e-graph rewriting.
    Reason {
        /// Expression to simplify (e.g. "(not (not x))").
        #[arg(long)]
        expr: String,

        /// Show rule count and extra details.
        #[arg(long)]
        verbose: bool,
    },

    /// Search for symbols similar to a given symbol via VSA.
    Search {
        /// Symbol name or numeric ID to search around.
        #[arg(long)]
        symbol: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Compute an analogy: A:B :: C:? via VSA bind/unbind.
    Analogy {
        /// First symbol (A).
        #[arg(long)]
        a: String,

        /// Second symbol (B).
        #[arg(long)]
        b: String,

        /// Third symbol (C).
        #[arg(long)]
        c: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Recover the filler for a (subject, predicate) pair via VSA unbind.
    Filler {
        /// Subject symbol name or ID.
        #[arg(long)]
        subject: String,

        /// Predicate symbol name or ID.
        #[arg(long)]
        predicate: String,

        /// Number of results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },

    /// Show engine info and statistics.
    Info,

    /// List and inspect symbols.
    Symbols {
        #[command(subcommand)]
        action: SymbolAction,
    },

    /// Export engine data as JSON.
    Export {
        #[command(subcommand)]
        action: ExportAction,
    },

    /// Manage skillpacks.
    Skill {
        #[command(subcommand)]
        action: SkillAction,
    },

    /// Run processing pipelines.
    Pipeline {
        #[command(subcommand)]
        action: PipelineAction,
    },

    /// Graph analytics: centrality, components, paths.
    Analytics {
        #[command(subcommand)]
        action: AnalyticsAction,
    },

    /// Render knowledge in hieroglyphic notation.
    Render {
        /// Entity or symbol to render (label or ID).
        #[arg(long)]
        entity: Option<String>,
        /// Depth of subgraph to render (default: 1).
        #[arg(long, default_value = "1")]
        depth: usize,
        /// Show all triples (when no entity specified).
        #[arg(long)]
        all: bool,
        /// Show glyph legend.
        #[arg(long)]
        legend: bool,
        /// Disable color output.
        #[arg(long)]
        no_color: bool,
    },

    /// Run the autonomous agent.
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },

    /// Interactive chat with the knowledge base.
    Chat {
        /// Skill pack to load on start (optional).
        #[arg(long)]
        skill: Option<String>,
        /// Ollama model override (default: llama3.2).
        #[arg(long)]
        model: Option<String>,
        /// Disable Ollama (regex-only mode).
        #[arg(long)]
        no_ollama: bool,
    },

    /// Ingest Rust source code into the knowledge graph.
    CodeIngest {
        /// File or directory path to ingest.
        #[arg(long)]
        path: PathBuf,
        /// Scan subdirectories recursively (default: true).
        #[arg(long, default_value = "true")]
        recursive: bool,
        /// Run forward-chaining code rules after ingestion.
        #[arg(long)]
        run_rules: bool,
        /// Maximum number of files to process.
        #[arg(long, default_value = "200")]
        max_files: usize,
        /// Run semantic enrichment (role classification, importance, data flow).
        #[arg(long)]
        enrich: bool,
    },

    /// Run semantic enrichment on existing code knowledge.
    ///
    /// Classifies module roles, computes importance, and detects data flow.
    /// Persists results as `semantic:*` triples in the KG.
    Enrich,

    /// Generate documentation from code knowledge in the KG.
    DocGen {
        /// Target: "architecture", "module:<name>", "type:<name>", "dependencies".
        #[arg(long)]
        target: String,
        /// Output format: "markdown" (default), "json", or "both".
        #[arg(long, default_value = "markdown")]
        format: String,
        /// Output file path (stdout if omitted).
        #[arg(long)]
        output: Option<PathBuf>,
        /// Use LLM to polish Markdown output.
        #[arg(long)]
        polish: bool,
    },

    /// Bidirectional grammar system: translate between prose and symbols.
    Grammar {
        #[command(subcommand)]
        action: GrammarAction,
    },
}

#[derive(Subcommand)]
enum SymbolAction {
    /// List all registered symbols.
    List,
    /// Show details of a specific symbol (by name or ID).
    Show {
        /// Symbol name or numeric ID.
        name_or_id: String,
    },
}

#[derive(Subcommand)]
enum ExportAction {
    /// Export the symbol table as JSON.
    Symbols,
    /// Export all triples as JSON.
    Triples,
    /// Export provenance chain for a symbol as JSON.
    Provenance {
        /// Symbol name or numeric ID.
        name_or_id: String,
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
    /// Scaffold a new skillpack with template files.
    Scaffold {
        /// Name for the new skillpack.
        name: String,
    },
}

#[derive(Subcommand)]
enum PipelineAction {
    /// List available built-in pipelines.
    List,
    /// Run the query pipeline (Retrieve -> Infer -> Reason).
    Query {
        /// Seed symbols (comma-separated names or IDs).
        #[arg(long)]
        seeds: String,
        /// Maximum traversal depth for retrieve stage.
        #[arg(long, default_value = "3")]
        max_depth: usize,
        /// Maximum inference depth.
        #[arg(long, default_value = "1")]
        infer_depth: usize,
        /// Output format: "summary" or "json".
        #[arg(long, default_value = "summary")]
        format: String,
    },
    /// Run a custom pipeline from named stages.
    Run {
        /// Comma-separated stage names: retrieve,infer,reason,extract.
        #[arg(long)]
        stages: String,
        /// Seed symbols (comma-separated names or IDs).
        #[arg(long)]
        seeds: String,
        /// Output format: "summary" or "json".
        #[arg(long, default_value = "summary")]
        format: String,
    },
}

#[derive(Subcommand)]
enum AnalyticsAction {
    /// Compute degree centrality for all nodes.
    Degree {
        /// Number of top results.
        #[arg(long, default_value = "10")]
        top_k: usize,
    },
    /// Compute PageRank scores.
    Pagerank {
        /// Damping factor (default 0.85).
        #[arg(long, default_value = "0.85")]
        damping: f64,
        /// Number of iterations.
        #[arg(long, default_value = "20")]
        iterations: usize,
        /// Number of top results.
        #[arg(long, default_value = "10")]
        top_k: usize,
    },
    /// Find strongly connected components.
    Components,
    /// Find shortest path between two symbols.
    Path {
        /// Start symbol name or ID.
        #[arg(long)]
        from: String,
        /// End symbol name or ID.
        #[arg(long)]
        to: String,
    },
}

#[derive(Subcommand)]
enum AgentAction {
    /// Run one OODA cycle.
    Cycle {
        /// Goal description.
        #[arg(long)]
        goal: String,
        /// Goal priority (0-255).
        #[arg(long, default_value = "128")]
        priority: u8,
    },
    /// Run agent until goals complete or max cycles reached.
    Run {
        /// Goal descriptions (comma-separated).
        #[arg(long)]
        goals: String,
        /// Maximum OODA cycles.
        #[arg(long, default_value = "10")]
        max_cycles: usize,
        /// Ollama model override.
        #[arg(long)]
        model: Option<String>,
        /// Disable Ollama LLM polishing.
        #[arg(long)]
        no_ollama: bool,
        /// Fresh start: ignore persisted goals from previous sessions.
        #[arg(long)]
        fresh: bool,
    },
    /// Trigger memory consolidation.
    Consolidate,
    /// Recall episodic memories.
    Recall {
        /// Query symbols (comma-separated names or IDs).
        #[arg(long)]
        query: String,
        /// Maximum results.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },
    /// List registered tools.
    Tools,
    /// Resume a previously persisted session.
    Resume {
        /// Maximum OODA cycles.
        #[arg(long, default_value = "10")]
        max_cycles: usize,
    },
    /// Interactive REPL: run cycles one at a time with user input between each.
    Repl {
        /// Goal descriptions (comma-separated). Omit to resume existing goals.
        #[arg(long)]
        goals: Option<String>,
    },
    /// Generate and display a plan for a goal.
    Plan {
        /// Goal description.
        #[arg(long)]
        goal: String,
        /// Goal priority (0-255).
        #[arg(long, default_value = "128")]
        priority: u8,
    },
    /// Run reflection on the current agent state.
    Reflect,
    /// Run forward-chaining inference rules.
    Infer {
        /// Maximum forward-chaining iterations.
        #[arg(long, default_value = "5")]
        max_iterations: usize,
        /// Minimum confidence for derived triples.
        #[arg(long, default_value = "0.1")]
        min_confidence: f32,
    },
    /// Analyze knowledge gaps around a goal.
    Gaps {
        /// Goal symbol name or ID.
        #[arg(long)]
        goal: String,
        /// Maximum gaps to report.
        #[arg(long, default_value = "10")]
        max_gaps: usize,
    },
    /// Discover schema patterns from the knowledge graph.
    Schema,
    /// Interactive chat: ask questions, agent explores and answers in prose.
    Chat {
        /// Ollama model override (default: llama3.2).
        #[arg(long)]
        model: Option<String>,
        /// Disable Ollama LLM polishing.
        #[arg(long)]
        no_ollama: bool,
        /// Maximum OODA cycles per question.
        #[arg(long, default_value = "5")]
        max_cycles: usize,
        /// Fresh start: ignore persisted session and goals.
        #[arg(long)]
        fresh: bool,
    },
}

#[derive(Subcommand)]
enum GrammarAction {
    /// List available grammar archetypes.
    List,
    /// Parse prose into abstract syntax and display the result.
    Parse {
        /// The prose text to parse.
        input: String,
    },
    /// Linearize a triple through a grammar archetype.
    Linearize {
        /// Subject entity.
        #[arg(long)]
        subject: String,
        /// Predicate relation.
        #[arg(long)]
        predicate: String,
        /// Object entity.
        #[arg(long)]
        object: String,
        /// Grammar archetype to use (default: all three).
        #[arg(long)]
        archetype: Option<String>,
        /// Confidence value (0.0-1.0, optional).
        #[arg(long)]
        confidence: Option<f32>,
    },
    /// Compare all archetypes side-by-side on the same input.
    Compare {
        /// Subject entity.
        #[arg(long)]
        subject: String,
        /// Predicate relation.
        #[arg(long)]
        predicate: String,
        /// Object entity.
        #[arg(long)]
        object: String,
        /// Confidence value (0.0-1.0, optional).
        #[arg(long)]
        confidence: Option<f32>,
    },
    /// Load a custom grammar archetype from a TOML file.
    Load {
        /// Path to the TOML grammar definition.
        #[arg(long)]
        file: PathBuf,
        /// Linearize a test triple after loading to verify.
        #[arg(long)]
        test: bool,
    },
    /// Render existing KG triples through a grammar archetype.
    Render {
        /// Entity to render triples for (label or ID).
        #[arg(long)]
        entity: String,
        /// Grammar archetype (default: formal).
        #[arg(long, default_value = "formal")]
        archetype: String,
        /// Maximum triples to render.
        #[arg(long, default_value = "20")]
        max_triples: usize,
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
                .unwrap_or_else(|_| {
                    // Default: show akh-medu info, silence noisy deps (egg, hnsw).
                    tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
                }),
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

        Commands::Ingest { file, format, csv_format, max_sentences } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match format.as_str() {
                "json" => {
                    let content = std::fs::read_to_string(&file).into_diagnostic()?;
                    let triples: Vec<serde_json::Value> =
                        serde_json::from_str(&content).into_diagnostic()?;

                    if triples.is_empty() {
                        println!("No triples found in {}", file.display());
                        return Ok(());
                    }

                    let first = &triples[0];
                    let is_label_format = first.get("subject").is_some();
                    let is_numeric_format = first.get("s").is_some();

                    if is_label_format {
                        let mut label_triples = Vec::new();
                        for (i, val) in triples.iter().enumerate() {
                            let subject = val["subject"].as_str().ok_or_else(|| {
                                EngineError::IngestFormat {
                                    message: format!("triple {i}: missing or non-string 'subject' field"),
                                }
                            }).into_diagnostic()?;
                            let predicate = val["predicate"].as_str().ok_or_else(|| {
                                EngineError::IngestFormat {
                                    message: format!("triple {i}: missing or non-string 'predicate' field"),
                                }
                            }).into_diagnostic()?;
                            let object = val["object"].as_str().ok_or_else(|| {
                                EngineError::IngestFormat {
                                    message: format!("triple {i}: missing or non-string 'object' field"),
                                }
                            }).into_diagnostic()?;
                            let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                            label_triples.push((
                                subject.to_string(),
                                predicate.to_string(),
                                object.to_string(),
                                confidence,
                            ));
                        }

                        let (created, ingested) = engine
                            .ingest_label_triples(&label_triples)
                            .into_diagnostic()?;
                        let _ = engine.persist();
                        println!(
                            "Ingested {ingested} triples ({created} new symbols) from {}",
                            file.display()
                        );
                    } else if is_numeric_format {
                        let mut count = 0;
                        for val in &triples {
                            let s = val["s"].as_u64().unwrap_or(0);
                            let p = val["p"].as_u64().unwrap_or(0);
                            let o = val["o"].as_u64().unwrap_or(0);
                            let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                            if let (Some(s), Some(p), Some(o)) =
                                (SymbolId::new(s), SymbolId::new(p), SymbolId::new(o))
                            {
                                engine
                                    .add_triple(&Triple::new(s, p, o).with_confidence(confidence))
                                    .into_diagnostic()?;
                                count += 1;
                            }
                        }
                        println!("Ingested {count} triples from {}", file.display());
                    } else {
                        return Err(EngineError::IngestFormat {
                            message: "unrecognized triple format in first element".into(),
                        })
                        .into_diagnostic();
                    }
                }
                "csv" => {
                    use akh_medu::agent::tools::CsvIngestTool;
                    use akh_medu::agent::tool::{Tool, ToolInput};

                    let input = ToolInput::new()
                        .with_param("path", file.to_str().unwrap_or(""))
                        .with_param("format", &csv_format);

                    let tool = CsvIngestTool;
                    let output = tool.execute(&engine, input).into_diagnostic()?;
                    println!("{}", output.result);
                }
                "text" => {
                    use akh_medu::agent::tools::TextIngestTool;
                    use akh_medu::agent::tool::{Tool, ToolInput};

                    let input = ToolInput::new()
                        .with_param("text", &format!("file:{}", file.display()))
                        .with_param("max_sentences", &max_sentences.to_string());

                    let tool = TextIngestTool;
                    let output = tool.execute(&engine, input).into_diagnostic()?;
                    println!("{}", output.result);
                }
                other => {
                    miette::bail!("Unknown format: \"{other}\". Use json, csv, or text.");
                }
            }

            // Auto-ground symbols after ingest.
            let ops = engine.ops();
            let im = engine.item_memory();
            let grounding_config = akh_medu::vsa::grounding::GroundingConfig::default();
            match akh_medu::vsa::grounding::ground_all(&engine, ops, im, &grounding_config) {
                Ok(result) => {
                    if result.symbols_updated > 0 {
                        println!(
                            "Grounding: {} symbols updated in {} round(s).",
                            result.symbols_updated, result.rounds_completed,
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Grounding warning: {e}");
                }
            }

            let _ = engine.persist();
            println!("{}", engine.info());
        }

        Commands::Bootstrap => {
            let engine = Engine::new(config).into_diagnostic()?;

            let skill_names = ["astronomy", "common_sense", "geography", "science", "language"];
            let mut total_triples = 0usize;
            let mut total_rules = 0usize;
            let mut skills_loaded = 0usize;

            for name in &skill_names {
                match engine.load_skill(name) {
                    Ok(activation) => {
                        println!(
                            "Loading skill: {}... {} triples, {} rules",
                            name, activation.triples_loaded, activation.rules_loaded,
                        );
                        total_triples += activation.triples_loaded;
                        total_rules += activation.rules_loaded;
                        skills_loaded += 1;
                    }
                    Err(e) => {
                        eprintln!("  Skipping {name}: {e}");
                    }
                }
            }

            // Run grounding.
            let ops = engine.ops();
            let im = engine.item_memory();
            let grounding_config = akh_medu::vsa::grounding::GroundingConfig::default();
            match akh_medu::vsa::grounding::ground_all(&engine, ops, im, &grounding_config) {
                Ok(result) => {
                    println!(
                        "Grounding symbols... {} round(s), {} symbols updated",
                        result.rounds_completed, result.symbols_updated,
                    );
                }
                Err(e) => {
                    eprintln!("Grounding warning: {e}");
                }
            }

            // Run forward-chaining inference.
            let rule_config = akh_medu::autonomous::RuleEngineConfig::default();
            match engine.run_rules(rule_config) {
                Ok(result) => {
                    let derived_count = result.derived.len();
                    println!(
                        "Running inference... derived {} new triples",
                        derived_count,
                    );
                    total_triples += derived_count;
                }
                Err(e) => {
                    eprintln!("Inference warning: {e}");
                }
            }

            let _ = engine.persist();
            println!(
                "Bootstrap complete: {} base + derived = {} total triples, {} skills, {} rules.",
                total_triples - total_rules, total_triples, skills_loaded, total_rules,
            );
        }

        Commands::Query {
            seeds,
            top_k,
            max_depth,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                .split(',')
                .map(|s| engine.resolve_symbol(s.trim()))
                .collect();
            let seed_ids = seed_ids.into_diagnostic()?;

            if seed_ids.is_empty() {
                miette::bail!("no valid seed symbols provided");
            }

            let query = InferenceQuery {
                seeds: seed_ids,
                top_k,
                max_depth,
                ..Default::default()
            };

            let result = engine.infer(&query).into_diagnostic()?;

            println!("Inference results (top {top_k}, depth {max_depth}):");
            for (i, (sym_id, confidence)) in result.activations.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (confidence: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    confidence
                );
            }

            if !result.provenance.is_empty() {
                println!("\nProvenance:");
                for record in &result.provenance {
                    let derived_label = engine.resolve_label(record.derived_id);
                    let kind_desc = format_derivation_kind(&record.kind, &engine);
                    println!(
                        "  \"{}\" / {} depth={} confidence={:.4} [{}]",
                        derived_label,
                        record.derived_id,
                        record.depth,
                        record.confidence,
                        kind_desc
                    );
                }
            }
        }

        Commands::Traverse {
            seeds,
            max_depth,
            predicates,
            min_confidence,
            max_results,
            format,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                .split(',')
                .map(|s| engine.resolve_symbol(s.trim()))
                .collect();
            let seed_ids = seed_ids.into_diagnostic()?;

            let predicate_filter: HashSet<SymbolId> = if let Some(ref preds) = predicates {
                let ids: std::result::Result<Vec<SymbolId>, _> = preds
                    .split(',')
                    .map(|s| engine.resolve_symbol(s.trim()))
                    .collect();
                ids.into_diagnostic()?.into_iter().collect()
            } else {
                HashSet::new()
            };

            let traverse_config = TraversalConfig {
                max_depth,
                predicate_filter,
                min_confidence,
                max_results,
            };

            let result = engine.traverse(&seed_ids, traverse_config).into_diagnostic()?;

            if format == "json" {
                let json_triples: Vec<serde_json::Value> = result
                    .triples
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "subject": engine.resolve_label(t.subject),
                            "predicate": engine.resolve_label(t.predicate),
                            "object": engine.resolve_label(t.object),
                            "confidence": t.confidence,
                        })
                    })
                    .collect();
                let json = serde_json::to_string_pretty(&json_triples).into_diagnostic()?;
                println!("{json}");
            } else {
                println!(
                    "Traversal: {} triples, {} nodes, depth {}",
                    result.triples.len(),
                    result.visited.len(),
                    result.depth_reached
                );
                for t in &result.triples {
                    println!(
                        "  \"{}\" -> {} -> \"{}\"  [{:.2}]",
                        engine.resolve_label(t.subject),
                        engine.resolve_label(t.predicate),
                        engine.resolve_label(t.object),
                        t.confidence,
                    );
                }
            }
        }

        Commands::Sparql { query, file } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let sparql_str = if let Some(q) = query {
                q
            } else if let Some(path) = file {
                std::fs::read_to_string(&path).into_diagnostic()?
            } else {
                miette::bail!("provide either --query or --file for SPARQL");
            };

            let results = engine.sparql_query(&sparql_str).into_diagnostic()?;

            if results.is_empty() {
                println!("No results.");
            } else {
                // Print header from first row's variable names.
                if let Some(first_row) = results.first() {
                    let header: Vec<&str> = first_row.iter().map(|(k, _)| k.as_str()).collect();
                    println!("{}", header.join("\t"));
                }
                for row in &results {
                    let vals: Vec<&str> = row.iter().map(|(_, v)| v.as_str()).collect();
                    println!("{}", vals.join("\t"));
                }
            }
        }

        Commands::Reason { expr, verbose } => {
            let engine = Engine::new(config).into_diagnostic()?;

            if verbose {
                let rules = engine.all_rules();
                println!("Active rules: {}", rules.len());
            }

            println!("Input:      {expr}");
            let simplified = engine.simplify_expression(&expr).into_diagnostic()?;
            println!("Simplified: {simplified}");
        }

        Commands::Search { symbol, top_k } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let sym_id = engine.resolve_symbol(&symbol).into_diagnostic()?;
            let label = engine.resolve_label(sym_id);

            let results = engine.search_similar_to(sym_id, top_k).into_diagnostic()?;

            println!("Similar to \"{label}\" (top {top_k}):");
            for (i, sr) in results.iter().enumerate() {
                let sr_label = engine.resolve_label(sr.symbol_id);
                println!(
                    "  {}. \"{}\" / {} (similarity: {:.4})",
                    i + 1,
                    sr_label,
                    sr.symbol_id,
                    sr.similarity
                );
            }
        }

        Commands::Analogy { a, b, c, top_k } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let a_id = engine.resolve_symbol(&a).into_diagnostic()?;
            let b_id = engine.resolve_symbol(&b).into_diagnostic()?;
            let c_id = engine.resolve_symbol(&c).into_diagnostic()?;

            let a_label = engine.resolve_label(a_id);
            let b_label = engine.resolve_label(b_id);
            let c_label = engine.resolve_label(c_id);

            let results = engine.infer_analogy(a_id, b_id, c_id, top_k).into_diagnostic()?;

            println!("Analogy: \"{a_label}\" : \"{b_label}\" :: \"{c_label}\" : ?");
            for (i, (sym_id, confidence)) in results.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (confidence: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    confidence
                );
            }
        }

        Commands::Filler {
            subject,
            predicate,
            top_k,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;
            let subj_id = engine.resolve_symbol(&subject).into_diagnostic()?;
            let pred_id = engine.resolve_symbol(&predicate).into_diagnostic()?;

            let subj_label = engine.resolve_label(subj_id);
            let pred_label = engine.resolve_label(pred_id);

            let results = engine
                .recover_filler(subj_id, pred_id, top_k)
                .into_diagnostic()?;

            println!("Filler for (\"{subj_label}\", \"{pred_label}\"):");
            for (i, (sym_id, similarity)) in results.iter().enumerate() {
                let label = engine.resolve_label(*sym_id);
                println!(
                    "  {}. \"{}\" / {} (similarity: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    similarity
                );
            }
        }

        Commands::Info => {
            let engine = Engine::new(config).into_diagnostic()?;
            println!("{}", engine.info());
        }

        Commands::Symbols { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                SymbolAction::List => {
                    let symbols = engine.all_symbols();
                    if symbols.is_empty() {
                        println!("No symbols registered.");
                    } else {
                        println!("Symbols ({}):", symbols.len());
                        for meta in &symbols {
                            println!(
                                "  {} / {} [{}]",
                                meta.label, meta.id, meta.kind
                            );
                        }
                    }
                }
                SymbolAction::Show { name_or_id } => {
                    let id = engine.resolve_symbol(&name_or_id).into_diagnostic()?;
                    let meta = engine.get_symbol_meta(id).into_diagnostic()?;
                    println!("Symbol: \"{}\"", meta.label);
                    println!("  id:         {}", meta.id);
                    println!("  kind:       {}", meta.kind);
                    println!("  created_at: {}", meta.created_at);

                    let from = engine.triples_from(id);
                    if !from.is_empty() {
                        println!("  outgoing triples ({}):", from.len());
                        for t in &from {
                            println!(
                                "    -> {} -> \"{}\"",
                                engine.resolve_label(t.predicate),
                                engine.resolve_label(t.object)
                            );
                        }
                    }

                    let to = engine.triples_to(id);
                    if !to.is_empty() {
                        println!("  incoming triples ({}):", to.len());
                        for t in &to {
                            println!(
                                "    \"{}\" -> {} ->",
                                engine.resolve_label(t.subject),
                                engine.resolve_label(t.predicate)
                            );
                        }
                    }
                }
            }
        }

        Commands::Export { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                ExportAction::Symbols => {
                    let exports = engine.export_symbol_table();
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Triples => {
                    let exports = engine.export_triples();
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Provenance { name_or_id } => {
                    let id = engine.resolve_symbol(&name_or_id).into_diagnostic()?;
                    let exports = engine.export_provenance_chain(id).into_diagnostic()?;
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
            }
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
                SkillAction::Scaffold { name } => {
                    let data_dir = cli
                        .data_dir
                        .as_deref()
                        .unwrap_or_else(|| std::path::Path::new(".akh-medu"));
                    let skill_dir = data_dir.join("skills").join(&name);
                    std::fs::create_dir_all(&skill_dir).into_diagnostic()?;

                    // Write skill.json template.
                    let manifest = serde_json::json!({
                        "id": name,
                        "name": name,
                        "version": "0.1.0",
                        "description": format!("{name} knowledge domain"),
                        "domains": [&name],
                        "weight_size_bytes": 0,
                        "triples_file": "triples.json",
                        "rules_file": "rules.txt"
                    });
                    std::fs::write(
                        skill_dir.join("skill.json"),
                        serde_json::to_string_pretty(&manifest).into_diagnostic()?,
                    )
                    .into_diagnostic()?;

                    // Write example triples.json in label-based format.
                    let triples = serde_json::json!([
                        {
                            "subject": "ExampleEntity",
                            "predicate": "is-a",
                            "object": "Category",
                            "confidence": 1.0
                        }
                    ]);
                    std::fs::write(
                        skill_dir.join("triples.json"),
                        serde_json::to_string_pretty(&triples).into_diagnostic()?,
                    )
                    .into_diagnostic()?;

                    // Write rules.txt template.
                    std::fs::write(
                        skill_dir.join("rules.txt"),
                        "# Rewrite rules for this skillpack.\n\
                         # Format: <lhs-pattern> => <rhs-pattern>\n\
                         # Example:\n\
                         # (similar ?x ?y) => (similar ?y ?x)\n",
                    )
                    .into_diagnostic()?;

                    println!("Scaffolded skill '{}' at {}", name, skill_dir.display());
                    println!("  skill.json   - manifest (edit name, domains, description)");
                    println!("  triples.json - knowledge triples (label-based format)");
                    println!("  rules.txt    - rewrite rules");
                }
            }
        }

        Commands::Pipeline { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                PipelineAction::List => {
                    println!("Built-in pipelines:");
                    println!();
                    let query = Pipeline::query_pipeline();
                    println!("  \"{}\" - {} stages:", query.name, query.stages.len());
                    for (i, stage) in query.stages.iter().enumerate() {
                        println!("    [{}] {} ({:?})", i + 1, stage.name, stage.kind);
                    }
                    println!();
                    let ingest = Pipeline::ingest_pipeline();
                    println!("  \"{}\" - {} stage(s):", ingest.name, ingest.stages.len());
                    for (i, stage) in ingest.stages.iter().enumerate() {
                        println!("    [{}] {} ({:?})", i + 1, stage.name, stage.kind);
                    }
                }
                PipelineAction::Query {
                    seeds,
                    max_depth,
                    infer_depth,
                    format,
                } => {
                    let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                        .split(',')
                        .map(|s| engine.resolve_symbol(s.trim()))
                        .collect();
                    let seed_ids = seed_ids.into_diagnostic()?;

                    let mut pipeline = Pipeline::query_pipeline();
                    // Apply custom config to retrieve stage.
                    if let Some(stage) = pipeline.stages.first_mut() {
                        stage.config = StageConfig::Retrieve {
                            traversal: TraversalConfig {
                                max_depth,
                                ..Default::default()
                            },
                        };
                    }
                    // Apply custom config to infer stage.
                    if let Some(stage) = pipeline.stages.get_mut(1) {
                        stage.config = StageConfig::Infer {
                            query_template: InferenceQuery {
                                max_depth: infer_depth,
                                ..Default::default()
                            },
                        };
                    }

                    let output = engine
                        .run_pipeline(&pipeline, PipelineData::Seeds(seed_ids))
                        .into_diagnostic()?;

                    if format == "json" {
                        print_pipeline_output_json(&output, &engine);
                    } else {
                        print_pipeline_output_summary(&output, &engine);
                    }
                }
                PipelineAction::Run {
                    stages,
                    seeds,
                    format,
                } => {
                    let seed_ids: std::result::Result<Vec<SymbolId>, _> = seeds
                        .split(',')
                        .map(|s| engine.resolve_symbol(s.trim()))
                        .collect();
                    let seed_ids = seed_ids.into_diagnostic()?;

                    let stage_list: Vec<PipelineStage> = stages
                        .split(',')
                        .map(|s| {
                            let name = s.trim().to_lowercase();
                            let kind = match name.as_str() {
                                "retrieve" => StageKind::Retrieve,
                                "infer" => StageKind::Infer,
                                "reason" => StageKind::Reason,
                                "extract" => StageKind::ExtractTriples,
                                other => {
                                    eprintln!("Unknown stage: {other}, defaulting to Retrieve");
                                    StageKind::Retrieve
                                }
                            };
                            PipelineStage {
                                name: name.clone(),
                                kind,
                                config: StageConfig::Default,
                            }
                        })
                        .collect();

                    let pipeline = Pipeline {
                        name: "custom".into(),
                        stages: stage_list,
                    };

                    let output = engine
                        .run_pipeline(&pipeline, PipelineData::Seeds(seed_ids))
                        .into_diagnostic()?;

                    if format == "json" {
                        print_pipeline_output_json(&output, &engine);
                    } else {
                        print_pipeline_output_summary(&output, &engine);
                    }
                }
            }
        }

        Commands::Analytics { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                AnalyticsAction::Degree { top_k } => {
                    let results = engine.degree_centrality();
                    if results.is_empty() {
                        println!("No nodes in graph.");
                    } else {
                        println!("Degree centrality (top {top_k}):");
                        for (i, dc) in results.iter().take(top_k).enumerate() {
                            let label = engine.resolve_label(dc.symbol);
                            println!(
                                "  {}. \"{}\" / {} — in: {}, out: {}, total: {}",
                                i + 1,
                                label,
                                dc.symbol,
                                dc.in_degree,
                                dc.out_degree,
                                dc.total
                            );
                        }
                    }
                }
                AnalyticsAction::Pagerank {
                    damping,
                    iterations,
                    top_k,
                } => {
                    let results = engine.pagerank(damping, iterations).into_diagnostic()?;
                    if results.is_empty() {
                        println!("No nodes in graph.");
                    } else {
                        println!("PageRank (damping={damping}, iterations={iterations}, top {top_k}):");
                        for (i, pr) in results.iter().take(top_k).enumerate() {
                            let label = engine.resolve_label(pr.symbol);
                            println!(
                                "  {}. \"{}\" / {} — score: {:.6}",
                                i + 1,
                                label,
                                pr.symbol,
                                pr.score
                            );
                        }
                    }
                }
                AnalyticsAction::Components => {
                    let components = engine.strongly_connected_components().into_diagnostic()?;
                    if components.is_empty() {
                        println!("No components found.");
                    } else {
                        println!("Strongly connected components ({}):", components.len());
                        for comp in &components {
                            let labels: Vec<String> = comp
                                .members
                                .iter()
                                .take(10)
                                .map(|s| engine.resolve_label(*s))
                                .collect();
                            let suffix = if comp.size > 10 {
                                format!(" ... and {} more", comp.size - 10)
                            } else {
                                String::new()
                            };
                            println!(
                                "  Component {} (size {}): [{}]{}",
                                comp.id,
                                comp.size,
                                labels.join(", "),
                                suffix
                            );
                        }
                    }
                }
                AnalyticsAction::Path { from, to } => {
                    let from_id = engine.resolve_symbol(&from).into_diagnostic()?;
                    let to_id = engine.resolve_symbol(&to).into_diagnostic()?;

                    let from_label = engine.resolve_label(from_id);
                    let to_label = engine.resolve_label(to_id);

                    match engine.shortest_path(from_id, to_id).into_diagnostic()? {
                        Some(path) => {
                            let labels: Vec<String> =
                                path.iter().map(|s| engine.resolve_label(*s)).collect();
                            println!(
                                "Shortest path from \"{}\" to \"{}\" ({} hops):",
                                from_label,
                                to_label,
                                path.len() - 1
                            );
                            println!("  {}", labels.join(" -> "));
                        }
                        None => {
                            println!(
                                "No path found from \"{}\" to \"{}\".",
                                from_label, to_label
                            );
                        }
                    }
                }
            }
        }

        Commands::Render {
            entity,
            depth,
            all,
            legend,
            no_color,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            let render_config = glyph::RenderConfig {
                color: !no_color,
                notation: glyph::NotationConfig {
                    use_pua: glyph::catalog::font_available(),
                    show_confidence: true,
                    show_provenance: false,
                    show_sigils: true,
                    compact: false,
                },
                ..Default::default()
            };

            if legend {
                println!("{}", glyph::render::render_legend(&render_config));
            } else if let Some(ref name) = entity {
                let sym_id = engine.resolve_symbol(name).into_diagnostic()?;
                let result = engine.extract_subgraph(&[sym_id], depth).into_diagnostic()?;
                if result.triples.is_empty() {
                    println!("No triples found around \"{}\".", name);
                } else {
                    println!(
                        "{}",
                        glyph::render::render_to_terminal(
                            &engine,
                            &result.triples,
                            &render_config,
                        )
                    );
                }
            } else if all {
                let triples = engine.all_triples();
                if triples.is_empty() {
                    println!("No triples in knowledge graph.");
                } else {
                    println!(
                        "{}",
                        glyph::render::render_to_terminal(&engine, &triples, &render_config)
                    );
                }
            } else {
                println!("Usage: render --entity <name> [--depth N] | render --all | render --legend");
            }
        }

        Commands::Agent { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);

            match action {
                AgentAction::Cycle { goal, priority } => {
                    let agent_config = AgentConfig::default();
                    let mut agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    agent
                        .add_goal(&goal, priority, "Agent-determined completion")
                        .into_diagnostic()?;

                    let result = agent.run_cycle().into_diagnostic()?;

                    println!("OODA Cycle {}", result.cycle_number);
                    println!(
                        "  Observe: {} active goals, {} WM entries",
                        result.observation.active_goals.len(),
                        result.observation.working_memory_size,
                    );
                    println!(
                        "  Orient:  {} relevant triples, {} inferences, pressure {:.2}",
                        result.orientation.relevant_knowledge.len(),
                        result.orientation.inferences.len(),
                        result.orientation.memory_pressure,
                    );
                    println!(
                        "  Decide:  tool={}, goal=\"{}\"",
                        result.decision.chosen_tool,
                        engine.resolve_label(result.decision.goal_id),
                    );
                    println!("  Reason:  {}", result.decision.reasoning);
                    println!(
                        "  Act:     success={}, symbols={}",
                        result.action_result.tool_output.success,
                        result.action_result.tool_output.symbols_involved.len(),
                    );
                    println!(
                        "  Result:  {}",
                        if result.action_result.tool_output.result.len() > 120 {
                            format!("{}...", &result.action_result.tool_output.result[..120])
                        } else {
                            result.action_result.tool_output.result.clone()
                        }
                    );

                    agent.persist_session().into_diagnostic()?;
                }

                AgentAction::Run {
                    goals,
                    max_cycles,
                    model,
                    no_ollama,
                    fresh,
                } => {
                    let agent_config = AgentConfig {
                        max_cycles,
                        ..Default::default()
                    };
                    let mut agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    if fresh {
                        agent.clear_goals();
                    }

                    // Optionally set up Ollama for narrative polishing.
                    if !no_ollama {
                        let mut ollama_config = akh_medu::agent::OllamaConfig::default();
                        if let Some(ref m) = model {
                            ollama_config.model = m.clone();
                        }
                        let mut client = akh_medu::agent::OllamaClient::new(ollama_config);
                        if client.probe() {
                            if let Err(e) = client.ensure_model() {
                                eprintln!("Warning: {e}");
                            }
                            agent.set_llm_client(client);
                        }
                    }

                    for goal_str in goals.split(',') {
                        let goal = goal_str.trim();
                        if !goal.is_empty() {
                            agent
                                .add_goal(goal, 128, "Agent-determined completion")
                                .into_diagnostic()?;
                        }
                    }

                    let run_result = agent.run_until_complete();

                    // Show summary regardless of how the run ended.
                    match &run_result {
                        Ok(results) => {
                            println!(
                                "Agent completed: {} cycles, {} goals",
                                results.len(),
                                agent.goals().len(),
                            );
                        }
                        Err(e) => {
                            println!("Agent stopped: {e}");
                        }
                    }

                    // Always display goal status.
                    println!("\nGoals:");
                    for g in agent.goals() {
                        println!(
                            "  [{}] {}: {}",
                            g.status,
                            engine.resolve_label(g.symbol_id),
                            g.description,
                        );
                    }

                    // Synthesize narrative from findings.
                    let summary = agent.synthesize_findings(&goals);
                    println!("\n{}", summary.overview);
                    for section in &summary.sections {
                        println!("\n## {}", section.heading);
                        println!("{}", section.prose);
                    }
                    if !summary.gaps.is_empty() {
                        println!("\nOpen questions:");
                        for gap in &summary.gaps {
                            println!("  - {gap}");
                        }
                    }

                    agent.persist_session().into_diagnostic()?;
                }

                AgentAction::Consolidate => {
                    let agent_config = AgentConfig::default();
                    let mut agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    let result = agent.consolidate().into_diagnostic()?;
                    println!("Consolidation complete:");
                    println!("  entries scored:    {}", result.entries_scored);
                    println!("  entries persisted: {}", result.entries_persisted);
                    println!("  entries evicted:   {}", result.entries_evicted);
                    println!("  episodes created:  {}", result.episodes_created.len());
                    for ep in &result.episodes_created {
                        println!("    {}", engine.resolve_label(*ep));
                    }

                    let _ = engine.persist();
                }

                AgentAction::Recall { query, top_k } => {
                    let agent_config = AgentConfig::default();
                    let agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    let query_ids: std::result::Result<Vec<SymbolId>, _> = query
                        .split(',')
                        .map(|s| engine.resolve_symbol(s.trim()))
                        .collect();
                    let query_ids = query_ids.into_diagnostic()?;

                    let episodes = agent.recall(&query_ids, top_k).into_diagnostic()?;
                    if episodes.is_empty() {
                        println!("No episodic memories found.");
                    } else {
                        println!("Recalled {} episode(s):", episodes.len());
                        for ep in &episodes {
                            println!(
                                "  {} — \"{}\" (learnings: {}, tags: {})",
                                engine.resolve_label(ep.symbol_id),
                                ep.summary,
                                ep.learnings.len(),
                                ep.tags.len(),
                            );
                        }
                    }
                }

                AgentAction::Tools => {
                    let agent_config = AgentConfig::default();
                    let agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    let tools = agent.list_tools();
                    println!("Registered tools ({}):", tools.len());
                    for sig in &tools {
                        println!("  {} — {}", sig.name, sig.description);
                        for param in &sig.parameters {
                            let req = if param.required { " (required)" } else { "" };
                            println!("    --{}{}: {}", param.name, req, param.description);
                        }
                    }
                }

                AgentAction::Resume { max_cycles } => {
                    if !Agent::has_persisted_session(&engine) {
                        miette::bail!(
                            "No persisted session found. Run `agent run` or `agent repl` first."
                        );
                    }

                    let agent_config = AgentConfig {
                        max_cycles,
                        ..Default::default()
                    };
                    let mut agent =
                        Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    println!(
                        "Resumed session: cycle {}, {} WM entries, {} goals",
                        agent.cycle_count(),
                        agent.working_memory().len(),
                        agent.goals().len(),
                    );

                    for g in agent.goals() {
                        println!(
                            "  [{}] {} — {}",
                            g.status,
                            engine.resolve_label(g.symbol_id),
                            g.description,
                        );
                    }

                    match agent.run_until_complete() {
                        Ok(results) => {
                            println!(
                                "Agent completed: {} cycles, {} goals",
                                results.len(),
                                agent.goals().len(),
                            );
                            for g in agent.goals() {
                                println!(
                                    "  {} [{}]: {}",
                                    engine.resolve_label(g.symbol_id),
                                    g.status,
                                    g.description,
                                );
                            }
                        }
                        Err(e) => {
                            println!("Agent stopped: {e}");
                        }
                    }

                    agent.persist_session().into_diagnostic()?;
                }

                AgentAction::Repl { goals } => {
                    // Resume or create a fresh agent.
                    let agent_config = AgentConfig::default();
                    let mut agent = if goals.is_none()
                        && Agent::has_persisted_session(&engine)
                    {
                        let a =
                            Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?;
                        println!(
                            "Resumed session: cycle {}, {} WM entries, {} goals",
                            a.cycle_count(),
                            a.working_memory().len(),
                            a.goals().len(),
                        );
                        a
                    } else {
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                    };

                    // Add goals if provided.
                    if let Some(ref goals_str) = goals {
                        for goal_str in goals_str.split(',') {
                            let goal = goal_str.trim();
                            if !goal.is_empty() {
                                agent
                                    .add_goal(goal, 128, "Agent-determined completion")
                                    .into_diagnostic()?;
                            }
                        }
                    }

                    println!("Agent REPL — q:quit, c:consolidate, p:plan, r:reflect, s:status, t:tools, i:infer, g:gaps, d:schema, h:hiero, hl:legend, Enter:cycle");
                    print_repl_status(&agent, &engine);

                    let stdin = std::io::stdin();
                    let mut input = String::new();

                    loop {
                        input.clear();
                        print!("> ");
                        // Flush stdout so the prompt appears before blocking on read.
                        use std::io::Write;
                        std::io::stdout().flush().ok();

                        if stdin.read_line(&mut input).into_diagnostic()? == 0 {
                            // EOF
                            break;
                        }

                        let cmd = input.trim();

                        match cmd {
                            "q" | "quit" | "exit" => break,
                            "c" | "consolidate" => {
                                match agent.consolidate() {
                                    Ok(r) => println!(
                                        "Consolidated: {} persisted, {} evicted",
                                        r.entries_persisted, r.entries_evicted,
                                    ),
                                    Err(e) => println!("Consolidation error: {e}"),
                                }
                            }
                            "s" | "status" => {
                                print_repl_status(&agent, &engine);
                            }
                            "t" | "tools" => {
                                for sig in agent.list_tools() {
                                    println!("  {} — {}", sig.name, sig.description);
                                }
                            }
                            "p" | "plan" => {
                                let active = agent.goals().iter()
                                    .find(|g| matches!(g.status, akh_medu::agent::GoalStatus::Active));
                                if let Some(goal) = active {
                                    let gid = goal.symbol_id;
                                    match agent.plan_goal(gid) {
                                        Ok(plan) => {
                                            println!("Plan: {}", plan.strategy);
                                            for step in &plan.steps {
                                                let status = match &step.status {
                                                    akh_medu::agent::StepStatus::Pending => ".",
                                                    akh_medu::agent::StepStatus::Active => ">",
                                                    akh_medu::agent::StepStatus::Completed => "✓",
                                                    akh_medu::agent::StepStatus::Failed { .. } => "✗",
                                                    akh_medu::agent::StepStatus::Skipped => "-",
                                                };
                                                println!(
                                                    "  [{}] {}: {}",
                                                    status, step.tool_name, step.rationale,
                                                );
                                            }
                                        }
                                        Err(e) => println!("Plan error: {e}"),
                                    }
                                } else {
                                    println!("No active goals to plan for.");
                                }
                            }
                            "r" | "reflect" => {
                                match agent.reflect() {
                                    Ok(result) => {
                                        println!("{}", result.summary);
                                        for adj in &result.adjustments {
                                            match adj {
                                                akh_medu::agent::Adjustment::IncreasePriority { from, to, reason, .. } =>
                                                    println!("  [+] {} → {}: {}", from, to, reason),
                                                akh_medu::agent::Adjustment::DecreasePriority { from, to, reason, .. } =>
                                                    println!("  [-] {} → {}: {}", from, to, reason),
                                                akh_medu::agent::Adjustment::SuggestNewGoal { description, reason, .. } =>
                                                    println!("  [new] {}: {}", description, reason),
                                                akh_medu::agent::Adjustment::SuggestAbandon { reason, .. } =>
                                                    println!("  [abandon] {}", reason),
                                            }
                                        }
                                    }
                                    Err(e) => println!("Reflect error: {e}"),
                                }
                            }
                            "i" | "infer" => {
                                let rule_config = RuleEngineConfig {
                                    max_iterations: 5,
                                    min_confidence: 0.1,
                                    ..Default::default()
                                };
                                match engine.run_rules(rule_config) {
                                    Ok(result) => {
                                        println!(
                                            "Derived {} triple(s) in {} iteration(s){}",
                                            result.derived.len(),
                                            result.iterations,
                                            if result.reached_fixpoint { " (fixpoint)" } else { "" },
                                        );
                                        for dt in &result.derived {
                                            println!(
                                                "  [{}] \"{}\" -> {} -> \"{}\"",
                                                dt.rule_name,
                                                engine.resolve_label(dt.triple.subject),
                                                engine.resolve_label(dt.triple.predicate),
                                                engine.resolve_label(dt.triple.object),
                                            );
                                        }
                                    }
                                    Err(e) => println!("Infer error: {e}"),
                                }
                            }
                            cmd if cmd.starts_with("g ") || cmd.starts_with("gaps ") => {
                                let goal_name = cmd
                                    .strip_prefix("gaps ")
                                    .or_else(|| cmd.strip_prefix("g "))
                                    .unwrap_or("")
                                    .trim();
                                if goal_name.is_empty() {
                                    println!("Usage: g <goal-symbol>");
                                } else {
                                    match engine.resolve_symbol(goal_name) {
                                        Ok(goal_id) => {
                                            let gap_config = GapAnalysisConfig {
                                                max_gaps: 10,
                                                ..Default::default()
                                            };
                                            match engine.analyze_gaps(&[goal_id], gap_config) {
                                                Ok(result) => {
                                                    println!(
                                                        "Gaps: {} analyzed, {} dead ends, {:.0}% coverage",
                                                        result.entities_analyzed,
                                                        result.dead_ends,
                                                        result.coverage_score * 100.0,
                                                    );
                                                    for gap in &result.gaps {
                                                        println!(
                                                            "  [{:.2}] \"{}\" — {}",
                                                            gap.severity,
                                                            engine.resolve_label(gap.entity),
                                                            gap.description,
                                                        );
                                                    }
                                                }
                                                Err(e) => println!("Gap analysis error: {e}"),
                                            }
                                        }
                                        Err(e) => println!("Symbol resolve error: {e}"),
                                    }
                                }
                            }
                            "hl" | "legend" => {
                                let rc = glyph::RenderConfig {
                                    color: true,
                                    notation: glyph::NotationConfig {
                                        use_pua: glyph::catalog::font_available(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                };
                                println!("{}", glyph::render::render_legend(&rc));
                            }
                            cmd if cmd == "h" || cmd == "hiero" || cmd.starts_with("h ") || cmd.starts_with("hiero ") => {
                                let entity_name = cmd
                                    .strip_prefix("hiero ")
                                    .or_else(|| cmd.strip_prefix("h "))
                                    .unwrap_or("")
                                    .trim();

                                let rc = glyph::RenderConfig {
                                    color: true,
                                    notation: glyph::NotationConfig {
                                        use_pua: glyph::catalog::font_available(),
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                };

                                if entity_name.is_empty() {
                                    // Render all triples.
                                    let triples = engine.all_triples();
                                    if triples.is_empty() {
                                        println!("No triples in knowledge graph.");
                                    } else {
                                        println!(
                                            "{}",
                                            glyph::render::render_to_terminal(&engine, &triples, &rc)
                                        );
                                    }
                                } else {
                                    match engine.resolve_symbol(entity_name) {
                                        Ok(sym_id) => {
                                            match engine.extract_subgraph(&[sym_id], 1) {
                                                Ok(result) => {
                                                    if result.triples.is_empty() {
                                                        println!("No triples around \"{}\".", entity_name);
                                                    } else {
                                                        println!(
                                                            "{}",
                                                            glyph::render::render_to_terminal(
                                                                &engine,
                                                                &result.triples,
                                                                &rc,
                                                            )
                                                        );
                                                    }
                                                }
                                                Err(e) => println!("Subgraph error: {e}"),
                                            }
                                        }
                                        Err(e) => println!("Symbol error: {e}"),
                                    }
                                }
                            }
                            "d" | "schema" => {
                                let schema_config = SchemaDiscoveryConfig::default();
                                match engine.discover_schema(schema_config) {
                                    Ok(result) => {
                                        if result.types.is_empty() {
                                            println!("No schema patterns discovered.");
                                        } else {
                                            println!("Discovered {} type(s):", result.types.len());
                                            for dt in &result.types {
                                                let name = dt
                                                    .type_symbol
                                                    .map(|s| engine.resolve_label(s))
                                                    .unwrap_or_else(|| {
                                                        format!("cluster({})", engine.resolve_label(dt.exemplar))
                                                    });
                                                println!(
                                                    "  {} — {} members",
                                                    name, dt.members.len(),
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => println!("Schema error: {e}"),
                                }
                            }
                            cmd if cmd.starts_with("goal ") => {
                                let desc = cmd.strip_prefix("goal ").unwrap_or("").trim();
                                if desc.is_empty() {
                                    println!("Usage: goal <description>");
                                } else {
                                    match agent.add_goal(desc, 128, "Agent-determined completion") {
                                        Ok(id) => println!("Added goal: {}", engine.resolve_label(id)),
                                        Err(e) => println!("Error adding goal: {e}"),
                                    }
                                }
                            }
                            _ => {
                                // Default: run one OODA cycle.
                                match agent.run_cycle() {
                                    Ok(result) => {
                                        println!(
                                            "Cycle {} — tool={}, goal=\"{}\", progress={:?}",
                                            result.cycle_number,
                                            result.decision.chosen_tool,
                                            engine.resolve_label(result.decision.goal_id),
                                            result.action_result.goal_progress,
                                        );
                                        println!(
                                            "  Result: {}",
                                            if result.action_result.tool_output.result.len() > 100 {
                                                format!(
                                                    "{}...",
                                                    &result.action_result.tool_output.result[..100]
                                                )
                                            } else {
                                                result.action_result.tool_output.result.clone()
                                            }
                                        );
                                    }
                                    Err(e) => println!("Cycle error: {e}"),
                                }
                            }
                        }
                    }

                    // Persist session on exit.
                    agent.persist_session().into_diagnostic()?;
                    println!("Session persisted. Use `agent resume` to continue.");
                }

                AgentAction::Plan { goal, priority } => {
                    let agent_config = AgentConfig::default();
                    let mut agent =
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

                    let goal_id = agent
                        .add_goal(&goal, priority, "Agent-determined completion")
                        .into_diagnostic()?;

                    let plan = agent.plan_goal(goal_id).into_diagnostic()?;

                    println!("Plan for \"{}\" (attempt {}):", goal, plan.attempt + 1);
                    println!("  Strategy: {}", plan.strategy);
                    for step in &plan.steps {
                        let status = match &step.status {
                            akh_medu::agent::StepStatus::Pending => "pending",
                            akh_medu::agent::StepStatus::Active => "active",
                            akh_medu::agent::StepStatus::Completed => "done",
                            akh_medu::agent::StepStatus::Failed { .. } => "FAILED",
                            akh_medu::agent::StepStatus::Skipped => "skipped",
                        };
                        println!(
                            "  [{}] Step {}: {} — {}",
                            status, step.index, step.tool_name, step.rationale,
                        );
                    }
                }

                AgentAction::Reflect => {
                    let agent_config = AgentConfig::default();
                    let mut agent = if Agent::has_persisted_session(&engine) {
                        Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
                    } else {
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                    };

                    let result = agent.reflect().into_diagnostic()?;

                    println!("{}", result.summary);
                    if !result.tool_insights.is_empty() {
                        println!("\nTool effectiveness:");
                        for ti in &result.tool_insights {
                            let flag = if ti.flagged_ineffective { " [!]" } else { "" };
                            println!(
                                "  {} — {}/{} success ({:.0}%){}",
                                ti.tool_name,
                                ti.successes,
                                ti.invocations,
                                ti.success_rate * 100.0,
                                flag,
                            );
                        }
                    }
                    if !result.goal_insights.is_empty() {
                        println!("\nGoal progress:");
                        for gi in &result.goal_insights {
                            let stag = if gi.is_stagnant { " [stagnant]" } else { "" };
                            println!(
                                "  {} — {} cycles worked{}",
                                gi.description, gi.cycles_worked, stag,
                            );
                        }
                    }
                    if !result.adjustments.is_empty() {
                        println!("\nRecommended adjustments:");
                        for adj in &result.adjustments {
                            match adj {
                                akh_medu::agent::Adjustment::IncreasePriority {
                                    from, to, reason, ..
                                } => println!("  [+] Priority {} → {}: {}", from, to, reason),
                                akh_medu::agent::Adjustment::DecreasePriority {
                                    from, to, reason, ..
                                } => println!("  [-] Priority {} → {}: {}", from, to, reason),
                                akh_medu::agent::Adjustment::SuggestNewGoal {
                                    description,
                                    reason,
                                    ..
                                } => println!("  [new] \"{}\": {}", description, reason),
                                akh_medu::agent::Adjustment::SuggestAbandon {
                                    reason, ..
                                } => println!("  [abandon] {}", reason),
                            }
                        }
                    }
                }

                AgentAction::Infer {
                    max_iterations,
                    min_confidence,
                } => {
                    let rule_config = RuleEngineConfig {
                        max_iterations,
                        min_confidence,
                        ..Default::default()
                    };

                    let result = engine.run_rules(rule_config).into_diagnostic()?;

                    println!(
                        "Derived {} new triple(s) in {} iteration(s){}",
                        result.derived.len(),
                        result.iterations,
                        if result.reached_fixpoint {
                            " (fixpoint reached)"
                        } else {
                            ""
                        },
                    );

                    for (rule, count) in &result.rule_stats {
                        if *count > 0 {
                            println!("  {rule}: {count} derivation(s)");
                        }
                    }

                    for dt in &result.derived {
                        println!(
                            "  [{}] \"{}\" -> {} -> \"{}\" (conf: {:.2})",
                            dt.rule_name,
                            engine.resolve_label(dt.triple.subject),
                            engine.resolve_label(dt.triple.predicate),
                            engine.resolve_label(dt.triple.object),
                            dt.confidence,
                        );
                    }
                }

                AgentAction::Gaps { goal, max_gaps } => {
                    let goal_id = engine.resolve_symbol(&goal).into_diagnostic()?;

                    let gap_config = GapAnalysisConfig {
                        max_gaps,
                        ..Default::default()
                    };

                    let result = engine
                        .analyze_gaps(&[goal_id], gap_config)
                        .into_diagnostic()?;

                    println!(
                        "Gap analysis for \"{}\": {} entities analyzed, {} dead ends, coverage {:.0}%",
                        engine.resolve_label(goal_id),
                        result.entities_analyzed,
                        result.dead_ends,
                        result.coverage_score * 100.0,
                    );

                    for gap in &result.gaps {
                        println!(
                            "  [{:.2}] \"{}\" — {}",
                            gap.severity,
                            engine.resolve_label(gap.entity),
                            gap.description,
                        );
                        if !gap.suggested_predicates.is_empty() {
                            let preds: Vec<String> = gap
                                .suggested_predicates
                                .iter()
                                .map(|p| engine.resolve_label(*p))
                                .collect();
                            println!("    suggested predicates: {}", preds.join(", "));
                        }
                    }
                }

                AgentAction::Schema => {
                    let schema_config = SchemaDiscoveryConfig::default();

                    let result = engine
                        .discover_schema(schema_config)
                        .into_diagnostic()?;

                    if result.types.is_empty()
                        && result.co_occurring_predicates.is_empty()
                        && result.relation_hierarchies.is_empty()
                    {
                        println!("No schema patterns discovered (insufficient data).");
                    } else {
                        if !result.types.is_empty() {
                            println!("Discovered types ({}):", result.types.len());
                            for dt in &result.types {
                                let name = dt
                                    .type_symbol
                                    .map(|s| engine.resolve_label(s))
                                    .unwrap_or_else(|| {
                                        format!("cluster({})", engine.resolve_label(dt.exemplar))
                                    });
                                println!(
                                    "  {} — {} members, {} typical predicates",
                                    name,
                                    dt.members.len(),
                                    dt.typical_predicates.len(),
                                );
                                for pp in &dt.typical_predicates {
                                    println!(
                                        "    {} ({:.0}% coverage)",
                                        engine.resolve_label(pp.predicate),
                                        pp.coverage * 100.0,
                                    );
                                }
                            }
                        }

                        if !result.co_occurring_predicates.is_empty() {
                            println!(
                                "\nCo-occurring predicates ({}):",
                                result.co_occurring_predicates.len()
                            );
                            for (p1, p2, strength) in &result.co_occurring_predicates {
                                println!(
                                    "  {} <-> {} ({:.0}%)",
                                    engine.resolve_label(*p1),
                                    engine.resolve_label(*p2),
                                    strength * 100.0,
                                );
                            }
                        }

                        if !result.relation_hierarchies.is_empty() {
                            println!(
                                "\nRelation hierarchies ({}):",
                                result.relation_hierarchies.len()
                            );
                            for rh in &result.relation_hierarchies {
                                println!(
                                    "  {} => {} ({:.0}%)",
                                    engine.resolve_label(rh.specific),
                                    engine.resolve_label(rh.general),
                                    rh.implication_strength * 100.0,
                                );
                            }
                        }
                    }
                }

                AgentAction::Chat {
                    model,
                    no_ollama,
                    max_cycles,
                    fresh,
                } => {
                    let agent_config = AgentConfig {
                        max_cycles,
                        ..Default::default()
                    };

                    // Resume or create fresh agent.
                    let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
                        let a = Agent::resume(Arc::clone(&engine), agent_config)
                            .into_diagnostic()?;
                        println!(
                            "Resumed session: cycle {}, {} WM entries, {} goals",
                            a.cycle_count(),
                            a.working_memory().len(),
                            a.goals().len(),
                        );
                        a
                    } else {
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                    };

                    if fresh {
                        agent.clear_goals();
                    }

                    // Optionally set up Ollama.
                    if !no_ollama {
                        let mut ollama_config = akh_medu::agent::OllamaConfig::default();
                        if let Some(ref m) = model {
                            ollama_config.model = m.clone();
                        }
                        let mut client = akh_medu::agent::OllamaClient::new(ollama_config);
                        if client.probe() {
                            if let Err(e) = client.ensure_model() {
                                eprintln!("Warning: {e}");
                            } else {
                                println!("LLM available (model: {})", client.model());
                            }
                            agent.set_llm_client(client);
                        } else {
                            println!("No LLM — using template-based responses.");
                        }
                    }

                    // Restore or create conversation state.
                    let mut conversation = {
                        let store = engine.store();
                        store
                            .get_meta(b"agent:chat:conversation")
                            .ok()
                            .flatten()
                            .and_then(|bytes| {
                                akh_medu::agent::Conversation::from_bytes(&bytes).ok()
                            })
                            .unwrap_or_else(|| akh_medu::agent::Conversation::new(100))
                    };

                    println!("akh-medu agent chat (type 'help' for commands, 'quit' to exit)\n");

                    use std::io::Write as _;
                    let stdin = std::io::stdin();
                    let mut input = String::new();

                    loop {
                        input.clear();
                        print!("> ");
                        std::io::stdout().flush().ok();

                        if stdin.read_line(&mut input).into_diagnostic()? == 0 {
                            break; // EOF
                        }

                        let cmd = input.trim();
                        if cmd.is_empty() {
                            continue;
                        }

                        match cmd {
                            "quit" | "exit" | "q" => break,
                            "help" | "h" => {
                                println!("Commands:");
                                println!("  <question>  — ask about the knowledge base");
                                println!("  status      — show goal status");
                                println!("  goals       — list active goals");
                                println!("  help        — this message");
                                println!("  quit        — exit chat");
                                continue;
                            }
                            "status" | "s" => {
                                println!("Cycle: {}, WM entries: {}, Goals: {}",
                                    agent.cycle_count(),
                                    agent.working_memory().len(),
                                    agent.goals().len(),
                                );
                                for g in agent.goals() {
                                    println!("  [{}] {}", g.status, g.description);
                                }
                                continue;
                            }
                            "goals" | "g" => {
                                if agent.goals().is_empty() {
                                    println!("No active goals.");
                                } else {
                                    for g in agent.goals() {
                                        println!(
                                            "  [{}] {}: {}",
                                            g.status,
                                            engine.resolve_label(g.symbol_id),
                                            g.description,
                                        );
                                    }
                                }
                                continue;
                            }
                            _ => {}
                        }

                        // Treat input as a question — create temporary goal and run cycles.
                        let question = cmd.to_string();
                        let goal_desc = format!("chat: {question}");
                        let goal_id = match agent
                            .add_goal(&goal_desc, 200, "Agent-determined completion")
                        {
                            Ok(id) => id,
                            Err(e) => {
                                eprintln!("Error: {e}");
                                continue;
                            }
                        };

                        // Run cycles for this question.
                        match agent.run_until_complete() {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("(agent stopped: {e})");
                            }
                        }

                        // Synthesize and display findings.
                        let summary = agent.synthesize_findings(&question);
                        println!("\n{}", summary.overview);
                        for section in &summary.sections {
                            println!("\n## {}", section.heading);
                            println!("{}", section.prose);
                        }
                        if !summary.gaps.is_empty() {
                            println!("\nOpen questions:");
                            for gap in &summary.gaps {
                                println!("  - {gap}");
                            }
                        }
                        println!();

                        // Mark the chat goal as completed.
                        let _ = agent.complete_goal(goal_id);

                        // Store exchange in conversation.
                        let response_text = format!("{summary}");
                        conversation.add_turn(question, response_text);
                    }

                    // Persist agent session.
                    agent.persist_session().into_diagnostic()?;

                    // Persist conversation.
                    if let Ok(bytes) = conversation.to_bytes() {
                        let _ = engine.store().put_meta(b"agent:chat:conversation", &bytes);
                    }

                    println!("Session saved.");
                }
            }
        }

        Commands::Chat { skill, model, no_ollama } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);

            // Load skill if specified.
            if let Some(ref skill_name) = skill {
                match engine.load_skill(skill_name) {
                    Ok(activation) => {
                        println!(
                            "Loaded skill: {} ({} triples, {} rules)",
                            skill_name, activation.triples_loaded, activation.rules_loaded,
                        );
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to load skill \"{skill_name}\": {e}");
                    }
                }

                // Run grounding after skill load.
                let ops = engine.ops();
                let im = engine.item_memory();
                let grounding_config = akh_medu::vsa::grounding::GroundingConfig::default();
                if let Ok(result) = akh_medu::vsa::grounding::ground_all(&engine, ops, im, &grounding_config) {
                    if result.symbols_updated > 0 {
                        println!(
                            "Grounded {} symbols in {} round(s).",
                            result.symbols_updated, result.rounds_completed,
                        );
                    }
                }
            }

            // Set up Ollama client (optional).
            let mut ollama = if no_ollama {
                None
            } else {
                let mut ollama_config = akh_medu::agent::OllamaConfig::default();
                if let Some(ref m) = model {
                    ollama_config.model = m.clone();
                }
                let mut client = akh_medu::agent::OllamaClient::new(ollama_config);
                if client.probe() {
                    println!("Ollama available (model: {})", client.model());
                    Some(client)
                } else {
                    println!("Ollama not available, using regex-only mode.");
                    None
                }
            };

            // Set up agent for goal-based interactions.
            let agent_config = AgentConfig {
                max_cycles: 20,
                ..Default::default()
            };
            let mut agent = Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;
            let mut conversation = akh_medu::agent::Conversation::new(100);

            println!("akh-medu chat (type 'help' for commands, 'quit' to exit)");
            println!();

            loop {
                eprint!("> ");
                let mut input = String::new();
                match std::io::stdin().read_line(&mut input) {
                    Ok(0) => break, // EOF
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("Read error: {e}");
                        break;
                    }
                }

                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "quit" || trimmed == "exit" || trimmed == "q" {
                    break;
                }

                let intent = akh_medu::agent::classify_intent(trimmed);
                let response = match intent {
                    akh_medu::agent::UserIntent::Help => {
                        "Commands:\n  \
                         <question>?    Query the knowledge base\n  \
                         <fact>         Assert a fact (e.g., \"Dogs are mammals\")\n  \
                         find <topic>   Set an agent goal\n  \
                         run [N]        Run N OODA cycles\n  \
                         status         Show agent status\n  \
                         show <entity>  Render hieroglyphic notation\n  \
                         help           Show this help\n  \
                         quit           Exit"
                            .to_string()
                    }

                    akh_medu::agent::UserIntent::ShowStatus => {
                        let goals = agent.goals();
                        let active = akh_medu::agent::goal::active_goals(&goals);
                        format!(
                            "Cycle: {}, Goals: {} active / {} total, WM: {} entries, Triples: {}",
                            agent.cycle_count(),
                            active.len(),
                            goals.len(),
                            agent.working_memory().len(),
                            engine.all_triples().len(),
                        )
                    }

                    akh_medu::agent::UserIntent::Query { subject } => {
                        // KG search via engine.
                        let mut lines = Vec::new();
                        match engine.resolve_symbol(&subject) {
                            Ok(sym_id) => {
                                let from_triples = engine.triples_from(sym_id);
                                let to_triples = engine.triples_to(sym_id);
                                let label = engine.resolve_label(sym_id);

                                for t in &from_triples {
                                    lines.push(format!(
                                        "  {} -> {} -> {} [{:.2}]",
                                        label,
                                        engine.resolve_label(t.predicate),
                                        engine.resolve_label(t.object),
                                        t.confidence,
                                    ));
                                }
                                for t in &to_triples {
                                    lines.push(format!(
                                        "  {} -> {} -> {} [{:.2}]",
                                        engine.resolve_label(t.subject),
                                        engine.resolve_label(t.predicate),
                                        label,
                                        t.confidence,
                                    ));
                                }

                                if lines.is_empty() {
                                    // Try similarity search.
                                    if let Ok(similar) = engine.search_similar_to(sym_id, 5) {
                                        if !similar.is_empty() {
                                            lines.push(format!("No triples for \"{subject}\", but similar symbols:"));
                                            for sr in &similar {
                                                lines.push(format!(
                                                    "  {} (similarity: {:.3})",
                                                    engine.resolve_label(sr.symbol_id),
                                                    sr.similarity,
                                                ));
                                            }
                                        }
                                    }
                                }

                                if lines.is_empty() {
                                    format!("No information found for \"{subject}\".")
                                } else {
                                    // Optionally synthesize NL answer via Ollama.
                                    let kg_text = lines.join("\n");
                                    if let Some(ref mut client) = ollama {
                                        match client.generate(
                                            &format!(
                                                "Based on these facts:\n{}\n\nAnswer: {}",
                                                kg_text, trimmed,
                                            ),
                                            Some("You are a helpful assistant. Answer the question using only the provided facts. Be concise."),
                                        ) {
                                            Ok(answer) => format!("{answer}\n\nKG facts:\n{kg_text}"),
                                            Err(_) => kg_text,
                                        }
                                    } else {
                                        kg_text
                                    }
                                }
                            }
                            Err(_) => format!("Symbol \"{subject}\" not found in knowledge base."),
                        }
                    }

                    akh_medu::agent::UserIntent::Assert { text } => {
                        // Extract triples using regex (and optionally LLM).
                        use akh_medu::agent::tools::TextIngestTool;
                        use akh_medu::agent::tool::Tool;

                        let tool_input = akh_medu::agent::ToolInput::new()
                            .with_param("text", &text);
                        match TextIngestTool.execute(&engine, tool_input) {
                            Ok(output) => {
                                // Auto-ground after ingest.
                                let ops = engine.ops();
                                let im = engine.item_memory();
                                let gc = akh_medu::vsa::grounding::GroundingConfig::default();
                                let _ = akh_medu::vsa::grounding::ground_all(&engine, ops, im, &gc);
                                output.result
                            }
                            Err(e) => format!("Extraction error: {e}"),
                        }
                    }

                    akh_medu::agent::UserIntent::SetGoal { description } => {
                        match agent.add_goal(&description, 128, "Agent-determined completion") {
                            Ok(_) => {
                                // Run a few OODA cycles.
                                let mut results = Vec::new();
                                for _ in 0..5 {
                                    match agent.run_cycle() {
                                        Ok(r) => {
                                            results.push(format!(
                                                "  [{}] {} -> {}",
                                                r.cycle_number,
                                                r.decision.chosen_tool,
                                                if r.action_result.tool_output.result.len() > 80 {
                                                    format!("{}...", &r.action_result.tool_output.result[..80])
                                                } else {
                                                    r.action_result.tool_output.result.clone()
                                                },
                                            ));
                                        }
                                        Err(e) => {
                                            results.push(format!("  Error: {e}"));
                                            break;
                                        }
                                    }
                                    // Check if goal completed.
                                    let active = akh_medu::agent::goal::active_goals(agent.goals());
                                    if active.is_empty() {
                                        break;
                                    }
                                }
                                format!("Goal set: \"{description}\"\n{}", results.join("\n"))
                            }
                            Err(e) => format!("Failed to set goal: {e}"),
                        }
                    }

                    akh_medu::agent::UserIntent::RunAgent { cycles } => {
                        let n = cycles.unwrap_or(1);
                        let mut results = Vec::new();
                        for _ in 0..n {
                            match agent.run_cycle() {
                                Ok(r) => {
                                    results.push(format!(
                                        "  [{}] {} -> {:?}",
                                        r.cycle_number,
                                        r.decision.chosen_tool,
                                        r.action_result.goal_progress,
                                    ));
                                }
                                Err(e) => {
                                    results.push(format!("  Error: {e}"));
                                    break;
                                }
                            }
                        }
                        if results.is_empty() {
                            "No active goals to run.".to_string()
                        } else {
                            results.join("\n")
                        }
                    }

                    akh_medu::agent::UserIntent::RenderHiero { entity } => {
                        let render_config = akh_medu::glyph::RenderConfig {
                            color: true,
                            notation: akh_medu::glyph::NotationConfig {
                                use_pua: akh_medu::glyph::catalog::font_available(),
                                show_confidence: true,
                                show_provenance: false,
                                show_sigils: true,
                                compact: false,
                            },
                            ..Default::default()
                        };

                        if let Some(ref name) = entity {
                            match engine.resolve_symbol(name) {
                                Ok(sym_id) => {
                                    match engine.extract_subgraph(&[sym_id], 1) {
                                        Ok(result) if !result.triples.is_empty() => {
                                            akh_medu::glyph::render::render_to_terminal(
                                                &engine, &result.triples, &render_config,
                                            )
                                        }
                                        _ => format!("No triples found around \"{name}\"."),
                                    }
                                }
                                Err(_) => format!("Symbol \"{name}\" not found."),
                            }
                        } else {
                            let triples = engine.all_triples();
                            if triples.is_empty() {
                                "No triples in knowledge graph.".to_string()
                            } else {
                                akh_medu::glyph::render::render_to_terminal(
                                    &engine, &triples, &render_config,
                                )
                            }
                        }
                    }

                    akh_medu::agent::UserIntent::Freeform { text } => {
                        if let Some(ref mut client) = ollama {
                            // Send to Ollama with KG context.
                            let symbols = engine.all_symbols();
                            let context = if symbols.len() <= 20 {
                                symbols.iter().map(|s| s.label.clone()).collect::<Vec<_>>().join(", ")
                            } else {
                                format!("{} symbols in knowledge base", symbols.len())
                            };

                            match client.generate(
                                &text,
                                Some(&format!(
                                    "You are a helpful assistant. The knowledge base contains: {}. \
                                     Answer based on this context when relevant.",
                                    context,
                                )),
                            ) {
                                Ok(answer) => answer,
                                Err(_) => {
                                    "I don't understand that input. Type 'help' for commands.".to_string()
                                }
                            }
                        } else {
                            "I don't understand that input. Type 'help' for commands.".to_string()
                        }
                    }
                };

                println!("{response}");
                println!();

                conversation.add_turn(trimmed.to_string(), response);
            }

            // Persist session.
            agent.persist_session().into_diagnostic()?;

            // Persist conversation.
            if let Ok(bytes) = conversation.to_bytes() {
                let _ = engine.store().put_meta(b"chat:conversation", &bytes);
            }

            println!("Session saved.");
        }

        Commands::CodeIngest {
            path,
            recursive,
            run_rules,
            max_files,
            enrich,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            use akh_medu::agent::tool::{Tool, ToolInput};
            use akh_medu::agent::tools::CodeIngestTool;

            let input = ToolInput::new()
                .with_param("path", path.to_str().unwrap_or(""))
                .with_param("recursive", &recursive.to_string())
                .with_param("max_files", &max_files.to_string());

            let tool = CodeIngestTool;
            let output = tool.execute(&engine, input).into_diagnostic()?;
            println!("{}", output.result);

            if run_rules {
                let rule_config = RuleEngineConfig::default();
                let result = engine.run_code_rules(rule_config).into_diagnostic()?;
                println!(
                    "Rules: {} derived triple(s) in {} iteration(s){}.",
                    result.derived.len(),
                    result.iterations,
                    if result.reached_fixpoint {
                        " (fixpoint)"
                    } else {
                        ""
                    },
                );
            }

            // Auto-ground after ingestion.
            let ops = engine.ops();
            let im = engine.item_memory();
            let grounding_config = akh_medu::vsa::grounding::GroundingConfig::default();
            match akh_medu::vsa::grounding::ground_all(&engine, ops, im, &grounding_config) {
                Ok(result) => {
                    if result.symbols_updated > 0 {
                        println!(
                            "Grounding: {} symbols updated in {} round(s).",
                            result.symbols_updated, result.rounds_completed,
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "grounding skipped");
                }
            }

            // Semantic enrichment (optional).
            if enrich {
                match akh_medu::agent::semantic_enrichment::enrich(&engine) {
                    Ok(result) => {
                        println!(
                            "Enrichment: {} role(s), {} importance score(s), {} flow edge(s).",
                            result.roles_enriched,
                            result.importance_enriched,
                            result.flows_detected,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "semantic enrichment skipped");
                    }
                }
            }

            let _ = engine.persist();
            println!("{}", engine.info());
        }

        Commands::Enrich => {
            let engine = Engine::new(config).into_diagnostic()?;

            match akh_medu::agent::semantic_enrichment::enrich(&engine) {
                Ok(result) => {
                    println!(
                        "Semantic enrichment complete:\n  Roles classified: {}\n  Importance scores: {}\n  Flow edges: {}",
                        result.roles_enriched,
                        result.importance_enriched,
                        result.flows_detected,
                    );
                }
                Err(e) => {
                    eprintln!("Enrichment failed: {e}");
                    std::process::exit(1);
                }
            }

            let _ = engine.persist();
        }

        Commands::Grammar { action } => {
            use akh_medu::grammar::{AbsTree, ConcreteGrammar, GrammarRegistry};
            use akh_medu::grammar::concrete::LinContext;
            use akh_medu::grammar::parser::{parse_prose, ParseResult};
            use akh_medu::grammar::concrete::ParseContext;
            use akh_medu::grammar::bridge::triple_to_abs;
            use akh_medu::grammar::custom::CustomGrammar;

            match action {
                GrammarAction::List => {
                    let reg = GrammarRegistry::new();
                    println!("Available grammar archetypes:\n");
                    let mut names = reg.list();
                    names.sort();
                    for name in names {
                        let grammar = reg.get(name).unwrap();
                        let default_marker = if name == reg.default_name() {
                            " (default)"
                        } else {
                            ""
                        };
                        println!("  {}{}", name, default_marker);
                        println!("    {}", grammar.description());
                        println!();
                    }
                }

                GrammarAction::Parse { input } => {
                    let engine = Engine::new(config).into_diagnostic()?;
                    let ctx = ParseContext::with_engine(
                        engine.registry(),
                        &engine.ops(),
                        engine.item_memory(),
                    );
                    let result = parse_prose(&input, &ctx);
                    match &result {
                        ParseResult::Facts(facts) => {
                            println!("Parsed {} fact(s):\n", facts.len());
                            let reg = GrammarRegistry::new();
                            let lin_ctx = LinContext::with_registry(engine.registry());
                            for (i, fact) in facts.iter().enumerate() {
                                println!("  {}. [{}]", i + 1, fact.cat());
                                // Show in all three archetypes
                                for archetype in &["formal", "terse", "narrative"] {
                                    if let Ok(prose) = reg.get(archetype).unwrap().linearize(fact, &lin_ctx) {
                                        println!("     {:<10} {}", format!("{archetype}:"), prose);
                                    }
                                }
                                println!();
                            }
                        }
                        ParseResult::Query { subject, .. } => {
                            println!("Parsed as query: subject = \"{subject}\"");
                        }
                        ParseResult::Command(cmd) => {
                            println!("Parsed as command: {cmd:?}");
                        }
                        ParseResult::Goal { description } => {
                            println!("Parsed as goal: \"{description}\"");
                        }
                        ParseResult::Freeform { text, .. } => {
                            println!("Could not parse into structured form.");
                            println!("Freeform text: \"{text}\"");
                        }
                    }
                }

                GrammarAction::Linearize {
                    subject,
                    predicate,
                    object,
                    archetype,
                    confidence,
                } => {
                    let reg = GrammarRegistry::new();

                    let tree = if let Some(conf) = confidence {
                        AbsTree::triple_with_confidence(
                            AbsTree::entity(&subject),
                            AbsTree::relation(&predicate),
                            AbsTree::entity(&object),
                            conf,
                        )
                    } else {
                        AbsTree::triple(
                            AbsTree::entity(&subject),
                            AbsTree::relation(&predicate),
                            AbsTree::entity(&object),
                        )
                    };

                    if let Some(name) = archetype {
                        match reg.linearize(&name, &tree) {
                            Ok(prose) => println!("{prose}"),
                            Err(e) => {
                                eprintln!("Error: {e}");
                                std::process::exit(1);
                            }
                        }
                    } else {
                        // Show all archetypes
                        let mut names = reg.list();
                        names.sort();
                        for name in names {
                            if let Ok(prose) = reg.linearize(name, &tree) {
                                println!("{:<10} {}", format!("{name}:"), prose);
                            }
                        }
                    }
                }

                GrammarAction::Compare {
                    subject,
                    predicate,
                    object,
                    confidence,
                } => {
                    let reg = GrammarRegistry::new();

                    let tree = if let Some(conf) = confidence {
                        AbsTree::triple_with_confidence(
                            AbsTree::entity(&subject),
                            AbsTree::relation(&predicate),
                            AbsTree::entity(&object),
                            conf,
                        )
                    } else {
                        AbsTree::triple(
                            AbsTree::entity(&subject),
                            AbsTree::relation(&predicate),
                            AbsTree::entity(&object),
                        )
                    };

                    println!("Triple: ({subject}, {predicate}, {object})");
                    if let Some(conf) = confidence {
                        println!("Confidence: {conf:.2}");
                    }
                    println!();

                    let mut names = reg.list();
                    names.sort();
                    let ctx = LinContext::default();
                    for name in names {
                        let grammar = reg.get(name).unwrap();
                        println!("── {} ──", name);
                        println!("  {}", grammar.description());
                        match grammar.linearize(&tree, &ctx) {
                            Ok(prose) => println!("  → {prose}"),
                            Err(e) => println!("  ✗ {e}"),
                        }
                        println!();
                    }
                }

                GrammarAction::Load { file, test } => {
                    let content = std::fs::read_to_string(&file).into_diagnostic()?;
                    let grammar = CustomGrammar::from_toml(&content).into_diagnostic()?;
                    let name = grammar.name().to_string();
                    let desc = grammar.description().to_string();
                    println!("Loaded custom grammar: \"{name}\"");
                    println!("  {desc}");

                    if test {
                        let ctx = LinContext::default();
                        let tree = AbsTree::triple(
                            AbsTree::entity("Dog"),
                            AbsTree::relation("is-a"),
                            AbsTree::entity("Mammal"),
                        );
                        match grammar.linearize(&tree, &ctx) {
                            Ok(prose) => println!("\n  Test triple: {prose}"),
                            Err(e) => println!("\n  Test failed: {e}"),
                        }

                        let gap = AbsTree::gap(AbsTree::entity("Dog"), "no habitat data");
                        match grammar.linearize(&gap, &ctx) {
                            Ok(prose) => println!("  Test gap:    {prose}"),
                            Err(e) => println!("  Test failed: {e}"),
                        }

                        let sim = AbsTree::similarity(
                            AbsTree::entity("Dog"),
                            AbsTree::entity("Wolf"),
                            0.87,
                        );
                        match grammar.linearize(&sim, &ctx) {
                            Ok(prose) => println!("  Test sim:    {prose}"),
                            Err(e) => println!("  Test failed: {e}"),
                        }
                    }
                }

                GrammarAction::Render {
                    entity,
                    archetype,
                    max_triples,
                } => {
                    let engine = Engine::new(config).into_diagnostic()?;
                    let reg = GrammarRegistry::new();
                    let grammar = reg.get(&archetype).into_diagnostic()?;
                    let lin_ctx = LinContext::with_registry(engine.registry());

                    // Resolve the entity
                    let symbol_id = engine.resolve_symbol(&entity).into_diagnostic()?;

                    // Get triples from and to this entity
                    let from_triples = engine.triples_from(symbol_id);
                    let to_triples = engine.triples_to(symbol_id);

                    let mut all_triples: Vec<_> = from_triples.into_iter().chain(to_triples).collect();
                    all_triples.truncate(max_triples);

                    if all_triples.is_empty() {
                        println!("No triples found for '{entity}'.");
                        return Ok(());
                    }

                    let entity_label = engine.resolve_label(symbol_id);
                    println!(
                        "Rendering {} triple(s) for '{}' via [{}] archetype:\n",
                        all_triples.len(),
                        entity_label,
                        archetype,
                    );

                    for triple in &all_triples {
                        let tree = triple_to_abs(triple, engine.registry());
                        match grammar.linearize(&tree, &lin_ctx) {
                            Ok(prose) => println!("  {prose}"),
                            Err(e) => println!("  (error: {e})"),
                        }
                    }
                }
            }
        }

        Commands::DocGen {
            target,
            format,
            output,
            polish,
        } => {
            let engine = Engine::new(config).into_diagnostic()?;

            use akh_medu::agent::tool::{Tool, ToolInput};
            use akh_medu::agent::tools::DocGenTool;

            let input = ToolInput::new()
                .with_param("target", &target)
                .with_param("format", &format)
                .with_param("polish", &polish.to_string());

            let tool = DocGenTool;
            let tool_output = tool.execute(&engine, input).into_diagnostic()?;

            if let Some(ref out_path) = output {
                std::fs::write(out_path, &tool_output.result).into_diagnostic()?;
                println!("Documentation written to {}", out_path.display());

                // Write JSON sidecar if format is "both".
                if format == "both" {
                    let json_path = out_path.with_extension("json");
                    // The result contains both separated by ---
                    if let Some(json_part) = tool_output.result.split("\n\n---\n\n").nth(1) {
                        std::fs::write(&json_path, json_part).into_diagnostic()?;
                        println!("JSON sidecar written to {}", json_path.display());
                    }
                }
            } else {
                println!("{}", tool_output.result);
            }
        }
    }

    Ok(())
}

/// Print agent REPL status line.
fn print_repl_status(agent: &Agent, engine: &Engine) {
    println!(
        "  cycle: {}, WM: {}/{}, goals: {} active / {} total",
        agent.cycle_count(),
        agent.working_memory().len(),
        agent.working_memory().capacity(),
        agent.goals().iter().filter(|g| matches!(g.status, akh_medu::agent::GoalStatus::Active)).count(),
        agent.goals().len(),
    );
    for g in agent.goals() {
        println!(
            "    [{}] {}",
            g.status,
            engine.resolve_label(g.symbol_id),
        );
    }
}

/// Print pipeline output in summary format.
fn print_pipeline_output_summary(
    output: &akh_medu::pipeline::PipelineOutput,
    engine: &Engine,
) {
    println!(
        "Pipeline — {} stages executed",
        output.stages_executed
    );
    for (i, (name, data)) in output.stage_results.iter().enumerate() {
        let summary = format_pipeline_data_summary(data, engine);
        println!("  [{}/{}] {}: {}", i + 1, output.stages_executed, name, summary);
    }
}

/// Print pipeline output in JSON format.
fn print_pipeline_output_json(
    output: &akh_medu::pipeline::PipelineOutput,
    engine: &Engine,
) {
    let stages: Vec<serde_json::Value> = output
        .stage_results
        .iter()
        .map(|(name, data)| {
            serde_json::json!({
                "stage": name,
                "summary": format_pipeline_data_summary(data, engine),
            })
        })
        .collect();
    let json = serde_json::json!({
        "stages_executed": output.stages_executed,
        "stages": stages,
        "result": format_pipeline_data_summary(&output.result, engine),
    });
    println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
}

/// Format a PipelineData variant as a one-line summary.
fn format_pipeline_data_summary(data: &akh_medu::pipeline::PipelineData, engine: &Engine) -> String {
    use akh_medu::pipeline::PipelineData;
    match data {
        PipelineData::Seeds(seeds) => {
            let labels: Vec<String> = seeds.iter().take(5).map(|s| engine.resolve_label(*s)).collect();
            format!("{} seeds [{}]", seeds.len(), labels.join(", "))
        }
        PipelineData::Triples(triples) => {
            format!("{} triples", triples.len())
        }
        PipelineData::Traversal(result) => {
            format!(
                "{} triples, {} nodes visited, depth {}",
                result.triples.len(),
                result.visited.len(),
                result.depth_reached
            )
        }
        PipelineData::Inference(result) => {
            if result.activations.is_empty() {
                "0 activations".to_string()
            } else {
                let top = &result.activations[0];
                format!(
                    "{} activations (top: \"{}\" {:.4})",
                    result.activations.len(),
                    engine.resolve_label(top.0),
                    top.1
                )
            }
        }
        PipelineData::Reasoning(result) => {
            format!(
                "\"{}\" (cost: {}, saturated: {})",
                result.simplified_expr, result.cost, result.saturated
            )
        }
    }
}

/// Format a DerivationKind with human-readable label resolution.
fn format_derivation_kind(kind: &DerivationKind, engine: &Engine) -> String {
    match kind {
        DerivationKind::Extracted => "extracted".to_string(),
        DerivationKind::Seed => "seed".to_string(),
        DerivationKind::GraphEdge { from, predicate } => {
            format!(
                "graph edge from \"{}\" via \"{}\"",
                engine.resolve_label(*from),
                engine.resolve_label(*predicate)
            )
        }
        DerivationKind::VsaRecovery {
            from,
            predicate,
            similarity,
        } => {
            format!(
                "VSA recovery from \"{}\" via \"{}\" (sim: {:.4})",
                engine.resolve_label(*from),
                engine.resolve_label(*predicate),
                similarity
            )
        }
        DerivationKind::Analogy { a, b, c } => {
            format!(
                "analogy \"{}\":\"{}\" :: \"{}\":?",
                engine.resolve_label(*a),
                engine.resolve_label(*b),
                engine.resolve_label(*c)
            )
        }
        DerivationKind::FillerRecovery {
            subject,
            predicate,
        } => {
            format!(
                "filler recovery (\"{}\", \"{}\")",
                engine.resolve_label(*subject),
                engine.resolve_label(*predicate)
            )
        }
        DerivationKind::Reasoned => "reasoned".to_string(),
        DerivationKind::Aggregated => "aggregated".to_string(),
        DerivationKind::AgentDecision { goal, cycle } => {
            format!(
                "agent decision for \"{}\" at cycle {}",
                engine.resolve_label(*goal),
                cycle
            )
        }
        DerivationKind::AgentConsolidation {
            reason,
            relevance_score,
        } => {
            format!(
                "agent consolidation (relevance: {:.2}): {}",
                relevance_score, reason
            )
        }
        DerivationKind::RuleInference {
            rule_name,
            antecedents,
        } => {
            let ant_labels: Vec<String> = antecedents
                .iter()
                .map(|s| engine.resolve_label(*s))
                .collect();
            format!(
                "rule inference [{}] from [{}]",
                rule_name,
                ant_labels.join(", ")
            )
        }
        DerivationKind::FusedInference {
            path_count,
            interference_signal,
        } => {
            format!(
                "fused inference ({} paths, interference: {:.2})",
                path_count, interference_signal
            )
        }
        DerivationKind::GapIdentified {
            gap_kind,
            severity,
        } => {
            format!(
                "gap identified [{}] (severity: {:.2})",
                gap_kind, severity
            )
        }
        DerivationKind::SchemaDiscovered { pattern_type } => {
            format!("schema discovered [{}]", pattern_type)
        }
        DerivationKind::SemanticEnrichment { source } => {
            format!("semantic enrichment [{}]", source)
        }
    }
}
