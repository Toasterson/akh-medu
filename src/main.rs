//! akh CLI: neuro-symbolic AI engine.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::autonomous::{GapAnalysisConfig, RuleEngineConfig, SchemaDiscoveryConfig};
use akh_medu::client::{AkhClient, discover_server};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::error::EngineError;
use akh_medu::glyph;
use akh_medu::grammar::Language;
use akh_medu::graph::Triple;
use akh_medu::graph::traverse::TraversalConfig;
use akh_medu::infer::InferenceQuery;
use akh_medu::pipeline::{Pipeline, PipelineData, PipelineStage, StageConfig, StageKind};
use akh_medu::provenance::DerivationKind;
use akh_medu::symbol::SymbolId;
use akh_medu::vsa::Dimension;

#[derive(Parser)]
#[command(name = "akh", version, about = "Neuro-symbolic AI engine")]
struct Cli {
    /// Data directory for persistent storage (overrides XDG workspace path).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    /// Workspace name (default: "default"). Workspaces live under XDG data dir.
    #[arg(short = 'w', long, global = true, default_value = "default")]
    workspace: String,

    /// Hypervector dimension.
    #[arg(long, global = true, default_value = "10000")]
    dimension: usize,

    /// Default language for parsing (en, ru, ar, fr, es, auto).
    #[arg(long, global = true, default_value = "auto")]
    language: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new workspace (creates XDG directory structure).
    Init,

    /// Manage workspaces.
    Workspace {
        #[command(subcommand)]
        action: WorkspaceAction,
    },

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

    /// Manage seed packs (knowledge bootstrapping).
    Seed {
        #[command(subcommand)]
        action: SeedAction,
    },

    /// Interactive chat with the knowledge base (launches TUI).
    Chat {
        /// Skill pack to load on start (optional).
        #[arg(long)]
        skill: Option<String>,
        /// Headless mode: use plain stdin/stdout instead of TUI.
        #[arg(long)]
        headless: bool,
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

    /// Pre-process text chunks for the Eleutherios integration pipeline.
    ///
    /// Reads JSONL from stdin, extracts entities/claims, writes JSONL to stdout.
    Preprocess {
        /// Output format: "jsonl" (default) or "json".
        #[arg(long, default_value = "jsonl")]
        format: String,

        /// Override language (en, ru, ar, fr, es). Default: auto-detect.
        #[arg(long)]
        language: Option<String>,

        /// Enrich extracted entities with context from the shared content library.
        #[arg(long)]
        library_context: bool,
    },

    /// Manage cross-lingual equivalence mappings.
    Equivalences {
        #[command(subcommand)]
        action: EquivalenceAction,
    },

    /// Manage the shared content library (ingest books, websites, documents).
    Library {
        #[command(subcommand)]
        action: LibraryAction,
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
    /// Install a skill from a local directory into the workspace.
    Install {
        /// Path to skill directory containing skill.json, triples.json, rules.txt.
        path: String,
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
    /// Interactive REPL (launches TUI).
    Repl {
        /// Goal descriptions (comma-separated). Omit to resume existing goals.
        #[arg(long)]
        goals: Option<String>,
        /// Headless mode: use plain stdin/stdout instead of TUI.
        #[arg(long)]
        headless: bool,
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
    /// Interactive chat (launches TUI).
    Chat {
        /// Maximum OODA cycles per question.
        #[arg(long, default_value = "5")]
        max_cycles: usize,
        /// Fresh start: ignore persisted session and goals.
        #[arg(long)]
        fresh: bool,
        /// Headless mode: use plain stdin/stdout instead of TUI.
        #[arg(long)]
        headless: bool,
    },
    /// Run as a background daemon with scheduled learning tasks.
    #[cfg(feature = "daemon")]
    Daemon {
        /// Maximum OODA cycles (0 = unlimited).
        #[arg(long, default_value = "0")]
        max_cycles: usize,
        /// Fresh start: ignore persisted session.
        #[arg(long)]
        fresh: bool,
        /// Equivalence learning interval in seconds.
        #[arg(long, default_value = "300")]
        equiv_interval: u64,
        /// Reflection interval in seconds.
        #[arg(long, default_value = "180")]
        reflect_interval: u64,
        /// Rule inference interval in seconds.
        #[arg(long, default_value = "600")]
        rules_interval: u64,
        /// Session persist interval in seconds.
        #[arg(long, default_value = "60")]
        persist_interval: u64,
    },
    /// Stop a running background daemon (requires akhomed).
    DaemonStop,
    /// Show background daemon status (requires akhomed).
    DaemonStatus,
}

#[derive(Subcommand)]
enum GrammarAction {
    /// List available grammar archetypes.
    List,
    /// Parse prose into abstract syntax and display the result.
    Parse {
        /// The prose text to parse.
        input: String,
        /// Commit parsed facts to the knowledge graph.
        #[arg(long)]
        ingest: bool,
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

#[derive(Subcommand)]
enum EquivalenceAction {
    /// List all learned equivalences.
    List,
    /// Show equivalence statistics (counts by source).
    Stats,
    /// Run all learning strategies to discover new equivalences.
    Learn,
    /// Export learned equivalences as JSON to stdout.
    Export,
    /// Import equivalences from JSON on stdin.
    Import,
}

#[derive(Subcommand)]
enum LibraryAction {
    /// Add a document to the library (file path or URL).
    Add {
        /// File path or URL to ingest.
        source: String,
        /// Override document title.
        #[arg(long)]
        title: Option<String>,
        /// Tags for categorization (comma-separated).
        #[arg(long)]
        tags: Option<String>,
        /// Override format detection (html, pdf, epub, text).
        #[arg(long)]
        format: Option<String>,
    },
    /// List all documents in the library.
    List,
    /// Search library content by text similarity.
    Search {
        /// Query text to search for.
        #[arg(long)]
        query: String,
        /// Maximum results to return.
        #[arg(long, default_value = "5")]
        top_k: usize,
    },
    /// Remove a document from the library.
    Remove {
        /// Document ID (slug) to remove.
        id: String,
    },
    /// Show detailed information about a document.
    Info {
        /// Document ID (slug).
        id: String,
    },
    /// Watch a directory for new files and auto-ingest them.
    Watch {
        /// Directory to watch (defaults to the library inbox).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum SeedAction {
    /// List available seed packs.
    List,
    /// Apply a seed pack to the current workspace.
    Apply {
        /// Seed pack ID.
        pack: String,
    },
    /// Show which seeds have been applied.
    Status,
}

#[derive(Subcommand)]
enum WorkspaceAction {
    /// List all workspaces.
    List,
    /// Create a new workspace.
    Create {
        /// Workspace name.
        name: String,
        /// Ennead role to assign (e.g. Architect, Investigator). Write-once.
        #[arg(long)]
        role: Option<String>,
    },
    /// Delete a workspace and all its data.
    Delete {
        /// Workspace name.
        name: String,
    },
    /// Show workspace info.
    Info {
        /// Workspace name (defaults to current workspace).
        name: Option<String>,
    },
    /// Assign an Ennead role to an existing workspace (write-once).
    AssignRole {
        /// Workspace name.
        name: String,
        /// Role to assign (e.g. Architect, Investigator, Executor).
        role: String,
    },
}

/// Resolve an [`AkhClient`]: prefer a running akhomed server, fall back to local engine.
fn resolve_client(
    workspace: &str,
    config: EngineConfig,
    xdg_paths: Option<&akh_medu::paths::AkhPaths>,
) -> Result<AkhClient> {
    if let Some(paths) = xdg_paths {
        if let Some(server) = discover_server(paths) {
            return Ok(AkhClient::remote(&server, workspace));
        }
    }
    eprintln!("warning: akhomed not running, using local engine");
    let engine = Engine::new(config).into_diagnostic()?;
    Ok(AkhClient::local(Arc::new(engine)))
}

/// Get a local [`Arc<Engine>`] from an [`AkhClient`], creating one if needed.
///
/// Commands that need deep engine access (Agent, TUI, etc.) call this.
fn require_local_engine(
    client: &AkhClient,
    config: EngineConfig,
) -> Result<Arc<Engine>> {
    if let Some(engine) = client.engine() {
        Ok(Arc::clone(engine))
    } else {
        eprintln!("warning: this command requires a local engine, ignoring remote server");
        Ok(Arc::new(Engine::new(config).into_diagnostic()?))
    }
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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // Default: show akh-medu info, silence noisy deps (egg, hnsw).
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();

    let cli = Cli::parse();

    let language = Language::from_code(&cli.language).unwrap_or(Language::Auto);

    // Resolve data directory: explicit --data-dir wins, otherwise XDG workspace.
    let (data_dir, xdg_paths) = if let Some(ref explicit) = cli.data_dir {
        (Some(explicit.clone()), None)
    } else {
        match akh_medu::paths::AkhPaths::resolve() {
            Ok(paths) => {
                let ws = paths.workspace(&cli.workspace);
                let dir = ws.kg_dir.clone();
                (Some(dir), Some(paths))
            }
            Err(_) => (None, None),
        }
    };

    let config = EngineConfig {
        dimension: Dimension(cli.dimension),
        data_dir: data_dir.clone(),
        language,
        ..Default::default()
    };

    match cli.command {
        Commands::Init => {
            // Resolve the effective data directory.
            let effective_dir = if let Some(ref explicit) = cli.data_dir {
                explicit.clone()
            } else if let Some(ref paths) = xdg_paths {
                let ws_paths = paths.workspace(&cli.workspace);
                paths.ensure_dirs().into_diagnostic()?;
                ws_paths.ensure_dirs().into_diagnostic()?;

                // Save default workspace config if it doesn't exist.
                let config_path = paths.workspace_config_file(&cli.workspace);
                if !config_path.exists() {
                    let ws_config = akh_medu::workspace::WorkspaceConfig::with_name(&cli.workspace);
                    ws_config.save(&config_path).into_diagnostic()?;
                }

                ws_paths.kg_dir.clone()
            } else {
                PathBuf::from(".akh-medu")
            };

            let config = EngineConfig {
                data_dir: Some(effective_dir.clone()),
                dimension: Dimension(cli.dimension),
                language,
                ..Default::default()
            };
            let engine = Engine::new(config).into_diagnostic()?;
            println!(
                "Initialized workspace \"{}\" at {}",
                cli.workspace,
                effective_dir.display()
            );
            println!("{}", engine.info());
        }

        Commands::Workspace { action } => {
            let paths = xdg_paths.ok_or_else(|| {
                miette::miette!("Cannot resolve XDG paths. Set HOME environment variable.")
            })?;
            paths.ensure_dirs().into_diagnostic()?;

            let mgr = akh_medu::workspace::WorkspaceManager::new(paths);

            match action {
                WorkspaceAction::List => {
                    let names = mgr.list();
                    if names.is_empty() {
                        println!(
                            "No workspaces found. Create one with: akh workspace create <name>"
                        );
                    } else {
                        println!("Workspaces:");
                        for name in &names {
                            println!("  {name}");
                        }
                    }
                }
                WorkspaceAction::Create { name, role } => {
                    let ws_config = akh_medu::workspace::WorkspaceConfig::with_name(&name);
                    let ws_paths = mgr.create(ws_config).into_diagnostic()?;
                    println!(
                        "Created workspace \"{name}\" at {}",
                        ws_paths.root.display()
                    );

                    if let Some(ref role_name) = role {
                        let engine_config = EngineConfig {
                            dimension: Dimension(cli.dimension),
                            data_dir: Some(ws_paths.kg_dir.clone()),
                            language,
                            ..Default::default()
                        };
                        let engine = Engine::new(engine_config).into_diagnostic()?;
                        engine.assign_role(role_name).into_diagnostic()?;
                        engine.persist().into_diagnostic()?;
                        println!("Assigned role \"{role_name}\" to workspace \"{name}\".");
                    }
                }
                WorkspaceAction::Delete { name } => {
                    mgr.delete(&name).into_diagnostic()?;
                    println!("Deleted workspace \"{name}\".");
                }
                WorkspaceAction::Info { name } => {
                    let ws_name = name.as_deref().unwrap_or(&cli.workspace);
                    let info = mgr.info(ws_name).into_diagnostic()?;
                    println!("Workspace: {}", info.name);
                    println!("  Dimension: {}", info.dimension);
                    println!("  Encoding: {}", info.encoding);
                    println!("  Language: {}", info.language);
                    println!("  Max memory: {} MB", info.max_memory_mb);
                    println!(
                        "  Seed packs: {}",
                        if info.seed_packs.is_empty() {
                            "(none)".to_string()
                        } else {
                            info.seed_packs.join(", ")
                        }
                    );
                    let ws_paths = mgr.paths().workspace(ws_name);
                    println!("  Data dir: {}", ws_paths.root.display());

                    // Show assigned role if any.
                    let engine_config = EngineConfig {
                        dimension: Dimension(cli.dimension),
                        data_dir: Some(ws_paths.kg_dir.clone()),
                        language,
                        ..Default::default()
                    };
                    if let Ok(engine) = Engine::new(engine_config) {
                        if let Some(role) = engine.assigned_role() {
                            println!("  Role: {role}");
                        }
                    }
                }
                WorkspaceAction::AssignRole { name, role } => {
                    let client = resolve_client(&name, config, Some(mgr.paths()))?;
                    client.assign_role(&role).into_diagnostic()?;
                    println!("Assigned role \"{role}\" to workspace \"{name}\".");
                }
            }
        }

        Commands::Seed { action } => {
            let seeds_dir = xdg_paths
                .as_ref()
                .map(|p| p.seeds_dir())
                .unwrap_or_else(|| PathBuf::from(".akh-medu/seeds"));

            let registry = akh_medu::seeds::SeedRegistry::discover(&seeds_dir);

            match action {
                SeedAction::List => {
                    let packs = registry.list();
                    if packs.is_empty() {
                        println!("No seed packs available.");
                    } else {
                        println!("Available seed packs:");
                        for pack in packs {
                            let source = match &pack.source {
                                akh_medu::seeds::SeedSource::Bundled => "bundled",
                                akh_medu::seeds::SeedSource::External(_) => "external",
                            };
                            println!(
                                "  {} v{} ({}) — {} [{} triples]",
                                pack.id,
                                pack.version,
                                source,
                                pack.description,
                                pack.triples.len(),
                            );
                        }
                    }
                }
                SeedAction::Apply { pack } => {
                    let engine = Engine::new(config).into_diagnostic()?;
                    let report = registry.apply(&pack, &engine).into_diagnostic()?;
                    if report.already_applied {
                        println!("Seed \"{}\" was already applied.", report.id);
                    } else {
                        println!(
                            "Applied seed \"{}\": {} triples added, {} skipped.",
                            report.id, report.triples_applied, report.triples_skipped,
                        );
                    }
                }
                SeedAction::Status => {
                    let engine = Engine::new(config).into_diagnostic()?;
                    let packs = registry.list();
                    println!("Seed status for workspace \"{}\":", cli.workspace);
                    for pack in packs {
                        let applied = akh_medu::seeds::is_seed_applied_public(&engine, &pack.id);
                        let status = if applied { "applied" } else { "not applied" };
                        println!("  {} — {status}", pack.id);
                    }
                }
            }
        }

        Commands::Ingest {
            file,
            format,
            csv_format,
            max_sentences,
        } => {
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
                            let subject = val["subject"]
                                .as_str()
                                .ok_or_else(|| EngineError::IngestFormat {
                                    message: format!(
                                        "triple {i}: missing or non-string 'subject' field"
                                    ),
                                })
                                .into_diagnostic()?;
                            let predicate = val["predicate"]
                                .as_str()
                                .ok_or_else(|| EngineError::IngestFormat {
                                    message: format!(
                                        "triple {i}: missing or non-string 'predicate' field"
                                    ),
                                })
                                .into_diagnostic()?;
                            let object = val["object"]
                                .as_str()
                                .ok_or_else(|| EngineError::IngestFormat {
                                    message: format!(
                                        "triple {i}: missing or non-string 'object' field"
                                    ),
                                })
                                .into_diagnostic()?;
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
                    use akh_medu::agent::tool::{Tool, ToolInput};
                    use akh_medu::agent::tools::CsvIngestTool;

                    let input = ToolInput::new()
                        .with_param("path", file.to_str().unwrap_or(""))
                        .with_param("format", &csv_format);

                    let tool = CsvIngestTool;
                    let output = tool.execute(&engine, input).into_diagnostic()?;
                    println!("{}", output.result);
                }
                "text" => {
                    use akh_medu::agent::tool::{Tool, ToolInput};
                    use akh_medu::agent::tools::TextIngestTool;

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

            let skill_names = [
                "astronomy",
                "common_sense",
                "geography",
                "science",
                "language",
            ];
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
                    println!("Running inference... derived {} new triples", derived_count,);
                    total_triples += derived_count;
                }
                Err(e) => {
                    eprintln!("Inference warning: {e}");
                }
            }

            let _ = engine.persist();
            println!(
                "Bootstrap complete: {} base + derived = {} total triples, {} skills, {} rules.",
                total_triples - total_rules,
                total_triples,
                skills_loaded,
                total_rules,
            );
        }

        Commands::Query {
            seeds,
            top_k,
            max_depth,
        } => {
            let client = resolve_client(&cli.workspace, config.clone(), xdg_paths.as_ref())?;

            let seed_ids: Vec<SymbolId> = seeds
                .split(',')
                .map(|s| client.resolve_symbol(s.trim()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .into_diagnostic()?;

            if seed_ids.is_empty() {
                miette::bail!("no valid seed symbols provided");
            }

            let query = InferenceQuery {
                seeds: seed_ids,
                top_k,
                max_depth,
                ..Default::default()
            };

            let result = client.infer(&query).into_diagnostic()?;

            println!("Inference results (top {top_k}, depth {max_depth}):");
            for (i, (sym_id, confidence)) in result.activations.iter().enumerate() {
                let label = client.resolve_label(*sym_id).unwrap_or_else(|_| format!("{sym_id}"));
                println!(
                    "  {}. \"{}\" / {} (confidence: {:.4})",
                    i + 1,
                    label,
                    sym_id,
                    confidence
                );
            }

            if !result.provenance.is_empty() {
                let engine = require_local_engine(&client, config)?;
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
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;

            let seed_ids: Vec<SymbolId> = seeds
                .split(',')
                .map(|s| client.resolve_symbol(s.trim()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .into_diagnostic()?;

            let predicate_filter: HashSet<SymbolId> = if let Some(ref preds) = predicates {
                preds
                    .split(',')
                    .map(|s| client.resolve_symbol(s.trim()))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .into_diagnostic()?
                    .into_iter()
                    .collect()
            } else {
                HashSet::new()
            };

            let traverse_config = TraversalConfig {
                max_depth,
                predicate_filter,
                min_confidence,
                max_results,
            };

            let result = client
                .traverse(&seed_ids, traverse_config)
                .into_diagnostic()?;

            if format == "json" {
                let json_triples: Vec<serde_json::Value> = result
                    .triples
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "subject": client.resolve_label(t.subject).unwrap_or_else(|_| format!("{}", t.subject)),
                            "predicate": client.resolve_label(t.predicate).unwrap_or_else(|_| format!("{}", t.predicate)),
                            "object": client.resolve_label(t.object).unwrap_or_else(|_| format!("{}", t.object)),
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
                        client.resolve_label(t.subject).unwrap_or_else(|_| format!("{}", t.subject)),
                        client.resolve_label(t.predicate).unwrap_or_else(|_| format!("{}", t.predicate)),
                        client.resolve_label(t.object).unwrap_or_else(|_| format!("{}", t.object)),
                        t.confidence,
                    );
                }
            }
        }

        Commands::Sparql { query, file } => {
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;

            let sparql_str = if let Some(q) = query {
                q
            } else if let Some(path) = file {
                std::fs::read_to_string(&path).into_diagnostic()?
            } else {
                miette::bail!("provide either --query or --file for SPARQL");
            };

            let results = client.sparql_query(&sparql_str).into_diagnostic()?;

            if results.is_empty() {
                println!("No results.");
            } else {
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
            let client = resolve_client(&cli.workspace, config.clone(), xdg_paths.as_ref())?;

            if verbose {
                let engine = require_local_engine(&client, config)?;
                let rules = engine.all_rules();
                println!("Active rules: {}", rules.len());
            }

            println!("Input:      {expr}");
            let simplified = client.simplify_expression(&expr).into_diagnostic()?;
            println!("Simplified: {simplified}");
        }

        Commands::Search { symbol, top_k } => {
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;
            let sym_id = client.resolve_symbol(&symbol).into_diagnostic()?;
            let label = client.resolve_label(sym_id).unwrap_or_else(|_| symbol.clone());

            let results = client.search_similar_to(sym_id, top_k).into_diagnostic()?;

            println!("Similar to \"{label}\" (top {top_k}):");
            for (i, sr) in results.iter().enumerate() {
                let sr_label = client.resolve_label(sr.symbol_id).unwrap_or_else(|_| format!("{}", sr.symbol_id));
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
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;
            let a_id = client.resolve_symbol(&a).into_diagnostic()?;
            let b_id = client.resolve_symbol(&b).into_diagnostic()?;
            let c_id = client.resolve_symbol(&c).into_diagnostic()?;

            let a_label = client.resolve_label(a_id).unwrap_or_else(|_| a.clone());
            let b_label = client.resolve_label(b_id).unwrap_or_else(|_| b.clone());
            let c_label = client.resolve_label(c_id).unwrap_or_else(|_| c.clone());

            let results = client
                .infer_analogy(a_id, b_id, c_id, top_k)
                .into_diagnostic()?;

            println!("Analogy: \"{a_label}\" : \"{b_label}\" :: \"{c_label}\" : ?");
            for (i, (sym_id, confidence)) in results.iter().enumerate() {
                let label = client.resolve_label(*sym_id).unwrap_or_else(|_| format!("{sym_id}"));
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
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;
            let subj_id = client.resolve_symbol(&subject).into_diagnostic()?;
            let pred_id = client.resolve_symbol(&predicate).into_diagnostic()?;

            let subj_label = client.resolve_label(subj_id).unwrap_or_else(|_| subject.clone());
            let pred_label = client.resolve_label(pred_id).unwrap_or_else(|_| predicate.clone());

            let results = client
                .recover_filler(subj_id, pred_id, top_k)
                .into_diagnostic()?;

            println!("Filler for (\"{subj_label}\", \"{pred_label}\"):");
            for (i, (sym_id, similarity)) in results.iter().enumerate() {
                let label = client.resolve_label(*sym_id).unwrap_or_else(|_| format!("{sym_id}"));
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
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;
            let info = client.info().into_diagnostic()?;
            println!("{info}");
        }

        Commands::Symbols { action } => {
            let client = resolve_client(&cli.workspace, config.clone(), xdg_paths.as_ref())?;

            match action {
                SymbolAction::List => {
                    let symbols = client.all_symbols().into_diagnostic()?;
                    if symbols.is_empty() {
                        println!("No symbols registered.");
                    } else {
                        println!("Symbols ({}):", symbols.len());
                        for meta in &symbols {
                            println!("  {} / {} [{}]", meta.label, meta.id, meta.kind);
                        }
                    }
                }
                SymbolAction::Show { name_or_id } => {
                    let id = client.resolve_symbol(&name_or_id).into_diagnostic()?;
                    // For Show we need engine-level detail; use local fallback.
                    let engine = require_local_engine(&client, config.clone())?;
                    let meta = engine.get_symbol_meta(id).into_diagnostic()?;
                    println!("Symbol: \"{}\"", meta.label);
                    println!("  id:         {}", meta.id);
                    println!("  kind:       {}", meta.kind);
                    println!("  created_at: {}", meta.created_at);

                    let from = client.triples_from(id).into_diagnostic()?;
                    if !from.is_empty() {
                        println!("  outgoing triples ({}):", from.len());
                        for t in &from {
                            let pred = client.resolve_label(t.predicate).unwrap_or_else(|_| format!("{}", t.predicate));
                            let obj = client.resolve_label(t.object).unwrap_or_else(|_| format!("{}", t.object));
                            println!("    -> {pred} -> \"{obj}\"");
                        }
                    }

                    let to = client.triples_to(id).into_diagnostic()?;
                    if !to.is_empty() {
                        println!("  incoming triples ({}):", to.len());
                        for t in &to {
                            let subj = client.resolve_label(t.subject).unwrap_or_else(|_| format!("{}", t.subject));
                            let pred = client.resolve_label(t.predicate).unwrap_or_else(|_| format!("{}", t.predicate));
                            println!("    \"{subj}\" -> {pred} ->");
                        }
                    }
                }
            }
        }

        Commands::Export { action } => {
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;

            match action {
                ExportAction::Symbols => {
                    let exports = client.export_symbols().into_diagnostic()?;
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Triples => {
                    let exports = client.export_triples().into_diagnostic()?;
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
                ExportAction::Provenance { name_or_id } => {
                    let id = client.resolve_symbol(&name_or_id).into_diagnostic()?;
                    let exports = client.export_provenance(id).into_diagnostic()?;
                    let json = serde_json::to_string_pretty(&exports).into_diagnostic()?;
                    println!("{json}");
                }
            }
        }

        Commands::Skill { action } => {
            match action {
                // Scaffold stays local-only (writes template files to user's filesystem).
                SkillAction::Scaffold { name } => {
                    let skill_base = data_dir
                        .as_deref()
                        .unwrap_or_else(|| std::path::Path::new(".akh-medu"));
                    let skill_dir = skill_base.join("skills").join(&name);
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

                // All other skill actions route through AkhClient.
                other => {
                    let client =
                        resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;

                    match other {
                        SkillAction::List => {
                            let skills = client.list_skills().into_diagnostic()?;
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
                            let activation =
                                client.load_skill(&name).into_diagnostic()?;
                            println!("Loaded skill: {}", activation.skill_id);
                            println!("  triples: {}", activation.triples_loaded);
                            println!("  rules:   {}", activation.rules_loaded);
                            println!("  memory:  {} bytes", activation.memory_bytes);
                        }
                        SkillAction::Unload { name } => {
                            client.unload_skill(&name).into_diagnostic()?;
                            println!("Unloaded skill: {name}");
                        }
                        SkillAction::Info { name } => {
                            let info =
                                client.skill_info(&name).into_diagnostic()?;
                            println!("Skill: {}", info.id);
                            println!("  name:        {}", info.name);
                            println!("  version:     {}", info.version);
                            println!("  description: {}", info.description);
                            println!("  state:       {}", info.state);
                            println!("  domains:     {}", info.domains.join(", "));
                            println!("  triples:     {}", info.triple_count);
                            println!("  rules:       {}", info.rule_count);
                        }
                        SkillAction::Install { path } => {
                            let skill_path = std::path::Path::new(&path);

                            // Read skill.json (required).
                            let manifest_path = skill_path.join("skill.json");
                            let manifest_content = std::fs::read_to_string(&manifest_path)
                                .into_diagnostic()?;
                            let manifest: akh_medu::skills::SkillManifest =
                                serde_json::from_str(&manifest_content).into_diagnostic()?;

                            // Read triples.json (optional, default empty).
                            let triples_path = skill_path.join("triples.json");
                            let triples: Vec<akh_medu::skills::LabelTriple> =
                                if triples_path.exists() {
                                    let content =
                                        std::fs::read_to_string(&triples_path).into_diagnostic()?;
                                    serde_json::from_str(&content).into_diagnostic()?
                                } else {
                                    vec![]
                                };

                            // Read rules.txt (optional, default empty).
                            let rules_path = skill_path.join("rules.txt");
                            let rules = if rules_path.exists() {
                                std::fs::read_to_string(&rules_path).into_diagnostic()?
                            } else {
                                String::new()
                            };

                            let payload = akh_medu::skills::SkillInstallPayload {
                                manifest,
                                triples,
                                rules,
                            };

                            let activation =
                                client.install_skill(&payload).into_diagnostic()?;
                            println!("Installed skill: {}", activation.skill_id);
                            println!("  triples: {}", activation.triples_loaded);
                            println!("  rules:   {}", activation.rules_loaded);
                            println!("  memory:  {} bytes", activation.memory_bytes);
                        }
                        SkillAction::Scaffold { .. } => unreachable!(),
                    }
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
            let client = resolve_client(&cli.workspace, config, xdg_paths.as_ref())?;

            match action {
                AnalyticsAction::Degree { top_k } => {
                    let results = client.degree_centrality().into_diagnostic()?;
                    if results.is_empty() {
                        println!("No nodes in graph.");
                    } else {
                        println!("Degree centrality (top {top_k}):");
                        for (i, dc) in results.iter().take(top_k).enumerate() {
                            let label = client.resolve_label(dc.symbol).unwrap_or_else(|_| format!("{}", dc.symbol));
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
                    let results = client.pagerank(damping, iterations).into_diagnostic()?;
                    if results.is_empty() {
                        println!("No nodes in graph.");
                    } else {
                        println!(
                            "PageRank (damping={damping}, iterations={iterations}, top {top_k}):"
                        );
                        for (i, pr) in results.iter().take(top_k).enumerate() {
                            let label = client.resolve_label(pr.symbol).unwrap_or_else(|_| format!("{}", pr.symbol));
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
                    let components = client.strongly_connected_components().into_diagnostic()?;
                    if components.is_empty() {
                        println!("No components found.");
                    } else {
                        println!("Strongly connected components ({}):", components.len());
                        for comp in &components {
                            let labels: Vec<String> = comp
                                .members
                                .iter()
                                .take(10)
                                .map(|s| client.resolve_label(*s).unwrap_or_else(|_| format!("{s}")))
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
                    let from_id = client.resolve_symbol(&from).into_diagnostic()?;
                    let to_id = client.resolve_symbol(&to).into_diagnostic()?;

                    let from_label = client.resolve_label(from_id).unwrap_or_else(|_| from.clone());
                    let to_label = client.resolve_label(to_id).unwrap_or_else(|_| to.clone());

                    match client.shortest_path(from_id, to_id).into_diagnostic()? {
                        Some(path) => {
                            let labels: Vec<String> =
                                path.iter().map(|s| client.resolve_label(*s).unwrap_or_else(|_| format!("{s}"))).collect();
                            println!(
                                "Shortest path from \"{}\" to \"{}\" ({} hops):",
                                from_label,
                                to_label,
                                path.len() - 1
                            );
                            println!("  {}", labels.join(" -> "));
                        }
                        None => {
                            println!("No path found from \"{}\" to \"{}\".", from_label, to_label);
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
                let result = engine
                    .extract_subgraph(&[sym_id], depth)
                    .into_diagnostic()?;
                if result.triples.is_empty() {
                    println!("No triples found around \"{}\".", name);
                } else {
                    println!(
                        "{}",
                        glyph::render::render_to_terminal(&engine, &result.triples, &render_config,)
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
                println!(
                    "Usage: render --entity <name> [--depth N] | render --all | render --legend"
                );
            }
        }

        Commands::Agent { action } => {
            // Handle daemon subcommands that only need a client (no local engine).
            match &action {
                AgentAction::DaemonStop => {
                    let client = resolve_client(
                        &cli.workspace,
                        config,
                        xdg_paths.as_ref(),
                    )?;
                    client.stop_daemon().into_diagnostic()?;
                    println!("Daemon stopped.");
                    return Ok(());
                }
                AgentAction::DaemonStatus => {
                    let client = resolve_client(
                        &cli.workspace,
                        config,
                        xdg_paths.as_ref(),
                    )?;
                    let status = client.daemon_status().into_diagnostic()?;
                    println!("Daemon status:");
                    println!("  Running:    {}", status.running);
                    println!("  Cycles:     {}", status.total_cycles);
                    println!("  Started at: {}", status.started_at);
                    println!("  Triggers:   {}", status.trigger_count);
                    return Ok(());
                }
                _ => {}
            }

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
                    let agent = Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

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
                    let agent = Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;

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

                AgentAction::Repl { goals, headless } => {
                    if !headless {
                        // TUI mode.
                        let agent_config = AgentConfig::default();
                        let fresh = goals.is_none();
                        let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
                            Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
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

                        let ws_name = cli.workspace.clone();
                        let mut tui =
                            akh_medu::tui::AkhTui::new_local(ws_name, Arc::clone(&engine), agent);
                        tui.run()?;
                    } else {
                        // Headless REPL (legacy stdin/stdout mode).
                        let agent_config = AgentConfig::default();
                        let mut agent = if goals.is_none() && Agent::has_persisted_session(&engine)
                        {
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

                        println!("Agent REPL (headless) — q:quit, s:status, Enter:cycle");
                        print_repl_status(&agent, &engine);

                        let stdin = std::io::stdin();
                        let mut input = String::new();

                        loop {
                            input.clear();
                            print!("> ");
                            use std::io::Write;
                            std::io::stdout().flush().ok();

                            if stdin.read_line(&mut input).into_diagnostic()? == 0 {
                                break;
                            }

                            let cmd = input.trim();
                            match cmd {
                                "q" | "quit" | "exit" => break,
                                "s" | "status" => print_repl_status(&agent, &engine),
                                _ => match agent.run_cycle() {
                                    Ok(result) => {
                                        println!(
                                            "Cycle {} — tool={}, progress={:?}",
                                            result.cycle_number,
                                            result.decision.chosen_tool,
                                            result.action_result.goal_progress,
                                        );
                                        let output = &result.action_result.tool_output.result;
                                        if output.len() > 100 {
                                            println!("  Result: {}...", &output[..100]);
                                        } else {
                                            println!("  Result: {output}");
                                        }
                                    }
                                    Err(e) => println!("Cycle error: {e}"),
                                },
                            }
                        }

                        agent.persist_session().into_diagnostic()?;
                        println!("Session persisted.");
                    }
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
                                    from,
                                    to,
                                    reason,
                                    ..
                                } => println!("  [+] Priority {} → {}: {}", from, to, reason),
                                akh_medu::agent::Adjustment::DecreasePriority {
                                    from,
                                    to,
                                    reason,
                                    ..
                                } => println!("  [-] Priority {} → {}: {}", from, to, reason),
                                akh_medu::agent::Adjustment::SuggestNewGoal {
                                    description,
                                    reason,
                                    ..
                                } => println!("  [new] \"{}\": {}", description, reason),
                                akh_medu::agent::Adjustment::SuggestAbandon { reason, .. } => {
                                    println!("  [abandon] {}", reason)
                                }
                                akh_medu::agent::Adjustment::ReformulateGoal { relaxed_criteria, reason, .. } => {
                                    println!("  [reformulate] \"{}\": {}", relaxed_criteria, reason)
                                }
                                akh_medu::agent::Adjustment::SuspendGoal { reason, .. } => {
                                    println!("  [suspend] {}", reason)
                                }
                                akh_medu::agent::Adjustment::ReviseBeliefs { retract, assert_syms, .. } => {
                                    println!("  [revise] retract {} / assert {}", retract.len(), assert_syms.len())
                                }
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

                    let result = engine.discover_schema(schema_config).into_diagnostic()?;

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
                    max_cycles,
                    fresh,
                    headless,
                } => {
                    if !headless {
                        // TUI mode.
                        let agent_config = AgentConfig {
                            max_cycles,
                            ..Default::default()
                        };
                        let ws_name = cli.workspace.clone();
                        akh_medu::tui::launch(&ws_name, Arc::clone(&engine), agent_config, fresh)?;
                    } else {
                        // Headless mode (legacy stdin/stdout).
                        let agent_config = AgentConfig {
                            max_cycles,
                            ..Default::default()
                        };
                        let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
                            Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
                        } else {
                            Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                        };
                        if fresh {
                            agent.clear_goals();
                        }

                        println!("akh agent chat (headless). Type 'quit' to exit.\n");
                        use std::io::Write as _;
                        let stdin = std::io::stdin();
                        let mut input = String::new();

                        loop {
                            input.clear();
                            print!("> ");
                            std::io::stdout().flush().ok();
                            if stdin.read_line(&mut input).into_diagnostic()? == 0 {
                                break;
                            }
                            let cmd = input.trim();
                            if cmd.is_empty() {
                                continue;
                            }
                            if cmd == "quit" || cmd == "exit" || cmd == "q" {
                                break;
                            }

                            let question = cmd.to_string();
                            let goal_desc = format!("chat: {question}");
                            let goal_id = match agent.add_goal(
                                &goal_desc,
                                200,
                                "Agent-determined completion",
                            ) {
                                Ok(id) => id,
                                Err(e) => {
                                    eprintln!("Error: {e}");
                                    continue;
                                }
                            };

                            match agent.run_until_complete() {
                                Ok(_) => {}
                                Err(e) => eprintln!("(agent stopped: {e})"),
                            }

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
                            let _ = agent.complete_goal(goal_id);
                        }

                        agent.persist_session().into_diagnostic()?;
                        println!("Session saved.");
                    }
                }

                #[cfg(feature = "daemon")]
                AgentAction::Daemon {
                    max_cycles,
                    fresh,
                    equiv_interval,
                    reflect_interval,
                    rules_interval,
                    persist_interval,
                } => {
                    // Try server-mediated daemon first.
                    if let Some(ref paths) = xdg_paths {
                        if let Some(server) = discover_server(paths) {
                            let client = AkhClient::remote(&server, &cli.workspace);
                            let config = serde_json::json!({ "max_cycles": max_cycles });
                            match client.start_daemon(Some(config)) {
                                Ok(status) => {
                                    println!("Daemon started via akhomed (cycles: {}, triggers: {})",
                                        status.total_cycles, status.trigger_count);
                                    // Block this process — it's now a noop sentinel.
                                    // The daemon runs inside akhomed.
                                    return Ok(());
                                }
                                Err(e) => {
                                    eprintln!("warning: server daemon failed ({e}), falling back to local");
                                }
                            }
                        }
                    }

                    use akh_medu::agent::{AgentDaemon, DaemonConfig};

                    let agent_config = AgentConfig::default();
                    let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
                        Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
                    } else {
                        Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                    };
                    if fresh {
                        agent.clear_goals();
                    }

                    let daemon_config = DaemonConfig {
                        equivalence_interval: std::time::Duration::from_secs(equiv_interval),
                        reflection_interval: std::time::Duration::from_secs(reflect_interval),
                        rule_inference_interval: std::time::Duration::from_secs(rules_interval),
                        persist_interval: std::time::Duration::from_secs(persist_interval),
                        max_cycles,
                        ..DaemonConfig::default()
                    };

                    let rt = tokio::runtime::Runtime::new().into_diagnostic()?;
                    let mut daemon = AgentDaemon::new(agent, daemon_config);
                    rt.block_on(daemon.run()).into_diagnostic()?;
                }

                // DaemonStop and DaemonStatus are handled above (early return).
                AgentAction::DaemonStop | AgentAction::DaemonStatus => unreachable!(),
            }
        }

        Commands::Chat { skill, headless } => {
            let ws_name = cli.workspace.clone();

            // Try remote TUI via akhomed if available (non-headless only).
            #[cfg(feature = "daemon")]
            if !headless {
                if let Some(ref paths) = xdg_paths {
                    if let Some(server_info) = akh_medu::client::discover_server(paths) {
                        eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                        return akh_medu::tui::launch_remote(&ws_name, &server_info);
                    }
                }
                eprintln!("warning: akhomed not running, using local engine");
            }

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
                if let Ok(result) =
                    akh_medu::vsa::grounding::ground_all(&engine, ops, im, &grounding_config)
                {
                    if result.symbols_updated > 0 {
                        println!(
                            "Grounded {} symbols in {} round(s).",
                            result.symbols_updated, result.rounds_completed,
                        );
                    }
                }
            }

            let agent_config = AgentConfig {
                max_cycles: 20,
                ..Default::default()
            };

            if !headless {
                // TUI mode (local fallback).
                akh_medu::tui::launch(&ws_name, engine, agent_config, false)?;
            } else {
                // Headless mode (legacy stdin/stdout).
                let mut agent = Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?;
                let mut conversation = akh_medu::agent::Conversation::new(100);

                println!("akh chat (headless). Type 'quit' to exit.\n");

                loop {
                    eprint!("> ");
                    let mut input = String::new();
                    match std::io::stdin().read_line(&mut input) {
                        Ok(0) => break,
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
                            "Commands: <question>? query, <fact> assert, find <topic> goal, status, help, quit"
                                .to_string()
                        }
                        akh_medu::agent::UserIntent::ShowStatus => {
                            format!(
                                "Cycle: {}, Goals: {}, WM: {} entries, Triples: {}",
                                agent.cycle_count(),
                                agent.goals().len(),
                                agent.working_memory().len(),
                                engine.all_triples().len(),
                            )
                        }
                        akh_medu::agent::UserIntent::Query { subject, original_input, question_word, capability_signal } => {
                            let grammar_name = engine
                                .compartments()
                                .and_then(|mgr| mgr.psyche())
                                .map(|p| p.persona.grammar_preference.clone())
                                .unwrap_or_else(|| "narrative".to_string());

                            // Try discourse-aware response first.
                            let discourse_prose = akh_medu::grammar::discourse::resolve_discourse(
                                &subject,
                                question_word,
                                &original_input,
                                &engine,
                                capability_signal,
                            )
                            .ok()
                            .and_then(|ctx| {
                                let from = engine.triples_from(ctx.subject_id);
                                let to = engine.triples_to(ctx.subject_id);
                                let mut all = from;
                                all.extend(to);
                                akh_medu::grammar::discourse::build_discourse_response(
                                    &all, &ctx, &engine,
                                )
                            })
                            .and_then(|tree| {
                                let registry = akh_medu::grammar::GrammarRegistry::new();
                                registry.linearize(&grammar_name, &tree).ok()
                            })
                            .filter(|s| !s.trim().is_empty());

                            if let Some(prose) = discourse_prose {
                                prose
                            } else {
                                // Fallback: existing synthesis path.
                                match engine.resolve_symbol(&subject) {
                                    Ok(sym_id) => {
                                        let from = engine.triples_from(sym_id);
                                        let to = engine.triples_to(sym_id);
                                        if from.is_empty() && to.is_empty() {
                                            format!("No information found for \"{subject}\".")
                                        } else {
                                            let mut all_triples = from;
                                            all_triples.extend(to);
                                            let summary =
                                                akh_medu::agent::synthesize::synthesize_from_triples(
                                                    &subject,
                                                    &all_triples,
                                                    &engine,
                                                    &grammar_name,
                                                );
                                            let mut lines = Vec::new();
                                            if !summary.overview.is_empty() {
                                                lines.push(summary.overview);
                                            }
                                            for section in &summary.sections {
                                                lines.push(format!(
                                                    "{}: {}",
                                                    section.heading, section.prose
                                                ));
                                            }
                                            for gap in &summary.gaps {
                                                lines.push(format!("(gap) {gap}"));
                                            }
                                            if lines.is_empty() {
                                                format!("No information found for \"{subject}\".")
                                            } else {
                                                lines.join("\n")
                                            }
                                        }
                                    }
                                    Err(_) => format!("Symbol \"{subject}\" not found."),
                                }
                            }
                        }
                        akh_medu::agent::UserIntent::Assert { text } => {
                            use akh_medu::agent::tool::Tool;
                            let tool_input = akh_medu::agent::ToolInput::new().with_param("text", &text);
                            match akh_medu::agent::tools::TextIngestTool.execute(&engine, tool_input) {
                                Ok(output) => output.result,
                                Err(e) => format!("Extraction error: {e}"),
                            }
                        }
                        akh_medu::agent::UserIntent::SetGoal { description } => {
                            match agent.add_goal(&description, 128, "Agent-determined completion") {
                                Ok(_) => format!("Goal set: \"{description}\""),
                                Err(e) => format!("Failed to set goal: {e}"),
                            }
                        }
                        akh_medu::agent::UserIntent::RunAgent { cycles } => {
                            let n = cycles.unwrap_or(1);
                            let mut out = Vec::new();
                            for _ in 0..n {
                                match agent.run_cycle() {
                                    Ok(r) => out.push(format!("[{}] {}", r.cycle_number, r.decision.chosen_tool)),
                                    Err(e) => { out.push(format!("Error: {e}")); break; }
                                }
                            }
                            out.join("\n")
                        }
                        akh_medu::agent::UserIntent::RenderHiero { .. } => {
                            "Hieroglyphic rendering not available in headless mode. Use the TUI.".to_string()
                        }
                        akh_medu::agent::UserIntent::Freeform { .. } => {
                            "I don't understand that. Type 'help' for commands.".to_string()
                        }
                    };

                    println!("{response}\n");
                    conversation.add_turn(trimmed.to_string(), response);
                }

                agent.persist_session().into_diagnostic()?;
                if let Ok(bytes) = conversation.to_bytes() {
                    let _ = engine.store().put_meta(b"chat:conversation", &bytes);
                }
                println!("Session saved.");
            }
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
                        result.roles_enriched, result.importance_enriched, result.flows_detected,
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
            use akh_medu::grammar::AbsTree;
            use akh_medu::grammar::bridge::triple_to_abs;
            use akh_medu::grammar::concrete::LinContext;
            use akh_medu::grammar::parser::ParseResult;

            match action {
                GrammarAction::List => {
                    let engine = Engine::new(config).into_diagnostic()?;
                    let reg = engine.grammar_registry();
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

                GrammarAction::Parse { input, ingest } => {
                    let engine = Engine::new(config).into_diagnostic()?;

                    if ingest {
                        let result = engine.ingest_prose(&input).into_diagnostic()?;
                        println!(
                            "Ingested {} triple(s) ({} new symbol(s))",
                            result.triples_ingested, result.symbols_created,
                        );
                        for tree in &result.trees {
                            if let Ok(prose) = engine.linearize(tree, Some("formal")) {
                                println!("  {prose}");
                            }
                        }
                        let _ = engine.persist();
                    } else {
                        let result = engine.parse(&input);
                        match &result {
                            ParseResult::Facts(facts) => {
                                println!("Parsed {} fact(s):\n", facts.len());
                                for (i, fact) in facts.iter().enumerate() {
                                    println!("  {}. [{}]", i + 1, fact.cat());
                                    for archetype in &["formal", "terse", "narrative"] {
                                        if let Ok(prose) = engine.linearize(fact, Some(archetype)) {
                                            println!(
                                                "     {:<10} {}",
                                                format!("{archetype}:"),
                                                prose
                                            );
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
                }

                GrammarAction::Linearize {
                    subject,
                    predicate,
                    object,
                    archetype,
                    confidence,
                } => {
                    let engine = Engine::new(config).into_diagnostic()?;

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
                        match engine.linearize(&tree, Some(&name)) {
                            Ok(prose) => println!("{prose}"),
                            Err(e) => {
                                eprintln!("Error: {e}");
                                std::process::exit(1);
                            }
                        }
                    } else {
                        // Show all archetypes
                        let reg = engine.grammar_registry();
                        let mut names = reg.list();
                        names.sort();
                        for name in names {
                            if let Ok(prose) = engine.linearize(&tree, Some(name)) {
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
                    let engine = Engine::new(config).into_diagnostic()?;

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

                    let reg = engine.grammar_registry();
                    let mut names = reg.list();
                    names.sort();
                    for name in names {
                        let grammar = reg.get(name).unwrap();
                        println!("── {} ──", name);
                        println!("  {}", grammar.description());
                        let ctx = LinContext::with_registry(engine.registry());
                        match grammar.linearize(&tree, &ctx) {
                            Ok(prose) => println!("  → {prose}"),
                            Err(e) => println!("  ✗ {e}"),
                        }
                        println!();
                    }
                }

                GrammarAction::Load { file, test } => {
                    let mut engine = Engine::new(config).into_diagnostic()?;
                    let content = std::fs::read_to_string(&file).into_diagnostic()?;
                    let name = engine.load_custom_grammar(&content).into_diagnostic()?;
                    let grammar = engine.grammar_registry().get(&name).into_diagnostic()?;
                    let desc = grammar.description().to_string();
                    println!("Loaded custom grammar: \"{name}\"");
                    println!("  {desc}");

                    if test {
                        let tree = AbsTree::triple(
                            AbsTree::entity("Dog"),
                            AbsTree::relation("is-a"),
                            AbsTree::entity("Mammal"),
                        );
                        match engine.linearize(&tree, Some(&name)) {
                            Ok(prose) => println!("\n  Test triple: {prose}"),
                            Err(e) => println!("\n  Test failed: {e}"),
                        }

                        let gap = AbsTree::gap(AbsTree::entity("Dog"), "no habitat data");
                        match engine.linearize(&gap, Some(&name)) {
                            Ok(prose) => println!("  Test gap:    {prose}"),
                            Err(e) => println!("  Test failed: {e}"),
                        }

                        let sim = AbsTree::similarity(
                            AbsTree::entity("Dog"),
                            AbsTree::entity("Wolf"),
                            0.87,
                        );
                        match engine.linearize(&sim, Some(&name)) {
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

                    // Resolve the entity
                    let symbol_id = engine.resolve_symbol(&entity).into_diagnostic()?;

                    // Get triples from and to this entity
                    let from_triples = engine.triples_from(symbol_id);
                    let to_triples = engine.triples_to(symbol_id);

                    let mut all_triples: Vec<_> =
                        from_triples.into_iter().chain(to_triples).collect();
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
                        match engine.linearize(&tree, Some(&archetype)) {
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

        Commands::Preprocess {
            format,
            language: lang_override,
            library_context,
        } => {
            use akh_medu::grammar::concrete::ParseContext as GrammarParseContext;
            use akh_medu::grammar::preprocess::{
                PreProcessResponse, TextChunk, preprocess_batch, preprocess_batch_with_library,
                preprocess_chunk, preprocess_chunk_with_library,
            };
            use std::io::{self, BufRead, Write as IoWrite};

            let engine = Engine::new(config).into_diagnostic()?;
            let ctx = GrammarParseContext::with_engine(
                engine.registry(),
                engine.ops(),
                engine.item_memory(),
            );

            let stdin = io::stdin();
            let stdout = io::stdout();
            let mut out = stdout.lock();

            if format == "json" {
                // Read all input as a JSON array of chunks
                use std::io::Read as _;
                let mut input = String::new();
                stdin.lock().read_to_string(&mut input).into_diagnostic()?;
                let chunks: Vec<TextChunk> = serde_json::from_str(&input).into_diagnostic()?;

                // Apply language override
                let chunks: Vec<TextChunk> = chunks
                    .into_iter()
                    .map(|mut c| {
                        if let Some(ref lang) = lang_override {
                            c.language = Some(lang.clone());
                        }
                        c
                    })
                    .collect();

                let start = std::time::Instant::now();
                let results = if library_context {
                    preprocess_batch_with_library(&chunks, &ctx, &engine.entity_resolver(), &engine)
                } else {
                    preprocess_batch(&chunks, &ctx)
                };
                let elapsed = start.elapsed().as_millis() as u64;

                let response = PreProcessResponse {
                    results,
                    processing_time_ms: elapsed,
                };
                serde_json::to_writer_pretty(&mut out, &response).into_diagnostic()?;
                writeln!(out).into_diagnostic()?;
            } else {
                // JSONL: one chunk per line in, one result per line out
                for line in stdin.lock().lines() {
                    let line = line.into_diagnostic()?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let mut chunk: TextChunk = serde_json::from_str(&line).into_diagnostic()?;
                    if let Some(ref lang) = lang_override {
                        chunk.language = Some(lang.clone());
                    }
                    let result = if library_context {
                        preprocess_chunk_with_library(
                            &chunk,
                            &ctx,
                            &engine.entity_resolver(),
                            &engine,
                        )
                    } else {
                        preprocess_chunk(&chunk, &ctx)
                    };
                    serde_json::to_writer(&mut out, &result).into_diagnostic()?;
                    writeln!(out).into_diagnostic()?;
                }
            }
        }

        Commands::Equivalences { action } => {
            let engine = Engine::new(config).into_diagnostic()?;

            match action {
                EquivalenceAction::List => {
                    let equivs = engine.export_equivalences();
                    if equivs.is_empty() {
                        println!(
                            "No learned equivalences yet. Run `equivalences learn` to discover some."
                        );
                    } else {
                        println!(
                            "{:<30} {:<30} {:<8} {:<6} {}",
                            "Surface", "Canonical", "Lang", "Conf", "Source"
                        );
                        println!("{}", "-".repeat(90));
                        for e in &equivs {
                            println!(
                                "{:<30} {:<30} {:<8} {:<6.2} {}",
                                e.surface, e.canonical, e.source_language, e.confidence, e.source,
                            );
                        }
                        println!("\nTotal: {} learned equivalences", equivs.len());
                    }
                }
                EquivalenceAction::Stats => {
                    let stats = engine.equivalence_stats();
                    println!("Equivalence statistics:");
                    println!("  runtime aliases:  {}", stats.runtime_aliases);
                    println!("  learned total:    {}", stats.learned_total);
                    println!("    kg-structural:  {}", stats.kg_structural);
                    println!("    vsa-similarity: {}", stats.vsa_similarity);
                    println!("    co-occurrence:  {}", stats.co_occurrence);
                    println!("    library-context:{}", stats.library_context);
                    println!("    manual:         {}", stats.manual);
                }
                EquivalenceAction::Learn => {
                    let count = engine.learn_equivalences().into_diagnostic()?;
                    println!("Discovered {count} new equivalences.");
                    let stats = engine.equivalence_stats();
                    println!("Total learned: {}", stats.learned_total);
                }
                EquivalenceAction::Export => {
                    use std::io::Write as _;
                    let equivs = engine.export_equivalences();
                    let json = serde_json::to_string_pretty(&equivs).into_diagnostic()?;
                    std::io::stdout()
                        .write_all(json.as_bytes())
                        .into_diagnostic()?;
                    std::io::stdout().write_all(b"\n").into_diagnostic()?;
                }
                EquivalenceAction::Import => {
                    use std::io::Read as _;
                    let mut input = String::new();
                    std::io::stdin()
                        .read_to_string(&mut input)
                        .into_diagnostic()?;
                    let equivs: Vec<akh_medu::grammar::entity_resolution::LearnedEquivalence> =
                        serde_json::from_str(&input).into_diagnostic()?;
                    let count = equivs.len();
                    engine.import_equivalences(&equivs).into_diagnostic()?;
                    println!("Imported {count} equivalences.");
                }
            }
        }

        Commands::Library { action } => {
            let paths = xdg_paths.ok_or_else(|| {
                miette::miette!("Cannot resolve XDG paths. Set HOME environment variable.")
            })?;
            paths.ensure_dirs().into_diagnostic()?;
            let library_dir = paths.library_dir();

            match action {
                LibraryAction::Add {
                    source,
                    title,
                    tags,
                    format,
                } => {
                    let client = resolve_client(&cli.workspace, config, Some(&paths))?;

                    let tag_list: Vec<String> = tags
                        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
                        .unwrap_or_default();

                    let req = akh_medu::library::LibraryAddRequest {
                        source,
                        title,
                        tags: tag_list,
                        format,
                    };

                    let result = client.library_add(&library_dir, &req).into_diagnostic()?;
                    println!("Ingested: {}", result.title);
                    println!("  ID:       {}", result.id);
                    println!("  Format:   {}", result.format);
                    println!("  Chunks:   {}", result.chunk_count);
                    println!("  Triples:  {}", result.triple_count);
                    println!("  Concepts: {}", result.concept_count);
                }

                LibraryAction::List => {
                    let client = resolve_client(&cli.workspace, config, Some(&paths))?;
                    let docs = client.library_list(&library_dir).into_diagnostic()?;

                    if docs.is_empty() {
                        println!(
                            "Library is empty. Add a document with: akh library add <file-or-url>"
                        );
                    } else {
                        println!(
                            "{:<30} {:<20} {:<8} {:<8} {}",
                            "ID", "Title", "Format", "Chunks", "Tags"
                        );
                        println!("{}", "-".repeat(80));
                        for doc in &docs {
                            let title_short = if doc.title.len() > 18 {
                                format!("{}...", &doc.title[..18])
                            } else {
                                doc.title.clone()
                            };
                            println!(
                                "{:<30} {:<20} {:<8} {:<8} {}",
                                doc.id,
                                title_short,
                                doc.format,
                                doc.chunk_count,
                                doc.tags.join(", "),
                            );
                        }
                        println!("\nTotal: {} document(s)", docs.len());
                    }
                }

                LibraryAction::Search { query, top_k } => {
                    let client = resolve_client(&cli.workspace, config, Some(&paths))?;
                    let results = client.library_search(&query, top_k).into_diagnostic()?;

                    if results.is_empty() {
                        println!("No matching content found for: \"{query}\"");
                    } else {
                        println!("Search results for \"{query}\":");
                        println!("{:<8} {:<10} {}", "Rank", "Sim", "Symbol");
                        println!("{}", "-".repeat(60));
                        for result in &results {
                            println!(
                                "{:<8} {:<10.4} {}",
                                result.rank, result.similarity, result.symbol_label,
                            );
                        }
                    }
                }

                LibraryAction::Remove { id } => {
                    let client = resolve_client(&cli.workspace, config, Some(&paths))?;
                    let removed = client.library_remove(&library_dir, &id).into_diagnostic()?;
                    println!("Removed: {} (\"{}\")", removed.id, removed.title);
                }

                LibraryAction::Watch { dir } => {
                    // Watch stays local-only — it's a long-running filesystem poller.
                    let engine = Engine::new(config).into_diagnostic()?;
                    let inbox_dir = dir.unwrap_or_else(|| paths.library_inbox());
                    let inbox_config =
                        akh_medu::library::inbox::InboxConfig::new(inbox_dir, library_dir);
                    akh_medu::library::inbox::watch_inbox(&engine, &inbox_config)
                        .into_diagnostic()?;
                }

                LibraryAction::Info { id } => {
                    let client = resolve_client(&cli.workspace, config, Some(&paths))?;
                    let doc = client.library_info(&library_dir, &id).into_diagnostic()?;

                    println!("Document: {}", doc.title);
                    println!("  ID:       {}", doc.id);
                    println!("  Format:   {}", doc.format);
                    println!("  Source:   {}", doc.source);
                    println!("  Chunks:   {}", doc.chunk_count);
                    println!("  Triples:  {}", doc.triple_count);
                    println!(
                        "  Tags:     {}",
                        if doc.tags.is_empty() {
                            "(none)".to_string()
                        } else {
                            doc.tags.join(", ")
                        }
                    );
                    println!("  Ingested: {} (unix timestamp)", doc.ingested_at);
                }
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
        agent
            .goals()
            .iter()
            .filter(|g| matches!(g.status, akh_medu::agent::GoalStatus::Active))
            .count(),
        agent.goals().len(),
    );
    for g in agent.goals() {
        println!("    [{}] {}", g.status, engine.resolve_label(g.symbol_id),);
    }
}

/// Print pipeline output in summary format.
fn print_pipeline_output_summary(output: &akh_medu::pipeline::PipelineOutput, engine: &Engine) {
    println!("Pipeline — {} stages executed", output.stages_executed);
    for (i, (name, data)) in output.stage_results.iter().enumerate() {
        let summary = format_pipeline_data_summary(data, engine);
        println!(
            "  [{}/{}] {}: {}",
            i + 1,
            output.stages_executed,
            name,
            summary
        );
    }
}

/// Print pipeline output in JSON format.
fn print_pipeline_output_json(output: &akh_medu::pipeline::PipelineOutput, engine: &Engine) {
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
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

/// Format a PipelineData variant as a one-line summary.
fn format_pipeline_data_summary(
    data: &akh_medu::pipeline::PipelineData,
    engine: &Engine,
) -> String {
    use akh_medu::pipeline::PipelineData;
    match data {
        PipelineData::Seeds(seeds) => {
            let labels: Vec<String> = seeds
                .iter()
                .take(5)
                .map(|s| engine.resolve_label(*s))
                .collect();
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
        DerivationKind::FillerRecovery { subject, predicate } => {
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
        DerivationKind::GapIdentified { gap_kind, severity } => {
            format!("gap identified [{}] (severity: {:.2})", gap_kind, severity)
        }
        DerivationKind::SchemaDiscovered { pattern_type } => {
            format!("schema discovered [{}]", pattern_type)
        }
        DerivationKind::SemanticEnrichment { source } => {
            format!("semantic enrichment [{}]", source)
        }
        DerivationKind::CompartmentLoaded {
            compartment_id,
            source_file,
        } => {
            format!(
                "compartment loaded [{}] from \"{}\"",
                compartment_id, source_file
            )
        }
        DerivationKind::ShadowVeto {
            pattern_name,
            severity,
        } => {
            format!("shadow veto [{}] (severity: {:.2})", pattern_name, severity)
        }
        DerivationKind::PsycheEvolution { trigger, cycle } => {
            format!("psyche evolution [{}] at cycle {}", trigger, cycle)
        }
        DerivationKind::WasmToolExecution {
            tool_name,
            skill_id,
            danger_level,
        } => {
            format!(
                "WASM tool execution [{}] from skill \"{}\" (danger: {})",
                tool_name, skill_id, danger_level
            )
        }
        DerivationKind::CliToolExecution {
            tool_name,
            binary_path,
            danger_level,
        } => {
            format!(
                "CLI tool execution [{}] via \"{}\" (danger: {})",
                tool_name, binary_path, danger_level
            )
        }
        DerivationKind::DocumentIngested {
            document_id,
            format,
            chunk_index,
        } => {
            format!(
                "document ingested [{}] format={} chunk={}",
                document_id, format, chunk_index
            )
        }
        DerivationKind::ConceptExtracted {
            document_id,
            chunk_index,
            extraction_method,
        } => {
            format!(
                "concept extracted [{}] chunk={} method={}",
                document_id, chunk_index, extraction_method
            )
        }
        DerivationKind::ContextInheritance { context, ancestor } => {
            format!(
                "context inheritance: \"{}\" from ancestor \"{}\"",
                engine.resolve_label(*context),
                engine.resolve_label(*ancestor)
            )
        }
        DerivationKind::ContextLifting {
            from_context,
            to_context,
            condition,
        } => {
            format!(
                "context lifting: \"{}\" → \"{}\" (condition: {})",
                engine.resolve_label(*from_context),
                engine.resolve_label(*to_context),
                condition
            )
        }
        DerivationKind::PredicateGeneralization { specific, general } => {
            format!(
                "predicate generalization: \"{}\" specializes \"{}\"",
                engine.resolve_label(*specific),
                engine.resolve_label(*general)
            )
        }
        DerivationKind::PredicateInverse { predicate, inverse } => {
            format!(
                "predicate inverse: \"{}\" ↔ \"{}\"",
                engine.resolve_label(*predicate),
                engine.resolve_label(*inverse)
            )
        }
        DerivationKind::DefeasibleOverride {
            winner,
            loser,
            reason,
        } => {
            format!(
                "defeasible override: \"{}\" beats \"{}\" ({})",
                engine.resolve_label(*winner),
                engine.resolve_label(*loser),
                reason
            )
        }
        DerivationKind::DispatchRoute {
            reasoner,
            problem_kind,
        } => {
            format!("dispatch: reasoner \"{reasoner}\" solved {problem_kind}")
        }
        DerivationKind::ArgumentVerdict {
            winner,
            pro_count,
            con_count,
            decisive_rule,
        } => {
            format!(
                "argumentation: \"{}\" wins ({pro_count} pro, {con_count} con, decisive: {decisive_rule})",
                engine.resolve_label(*winner),
            )
        }
        DerivationKind::RuleMacroExpansion {
            macro_name,
            expanded_count,
        } => {
            format!(
                "rule macro expansion [{}] ({} triples)",
                macro_name, expanded_count
            )
        }
        DerivationKind::TemporalDecay {
            profile,
            original_confidence,
            decayed_confidence,
        } => {
            format!(
                "temporal decay [{}] ({:.4} → {:.4})",
                profile, original_confidence, decayed_confidence
            )
        }
        DerivationKind::ContradictionDetected {
            kind,
            existing_object,
            incoming_object,
        } => {
            format!(
                "contradiction [{}]: \"{}\" vs \"{}\"",
                kind,
                engine.resolve_label(*existing_object),
                engine.resolve_label(*incoming_object)
            )
        }
        DerivationKind::SkolemWitness {
            existential_relation,
            bound_var,
        } => {
            format!(
                "skolem witness: relation \"{}\" bound to \"{}\"",
                engine.resolve_label(*existential_relation),
                engine.resolve_label(*bound_var)
            )
        }
        DerivationKind::SkolemGrounding {
            skolem,
            concrete_entity,
        } => {
            format!(
                "skolem grounding: \"{}\" → \"{}\"",
                engine.resolve_label(*skolem),
                engine.resolve_label(*concrete_entity)
            )
        }
        DerivationKind::CwaQuery {
            context,
            subject,
            predicate,
        } => {
            format!(
                "CWA negation in \"{}\": ({}, {}) not found",
                engine.resolve_label(*context),
                engine.resolve_label(*subject),
                engine.resolve_label(*predicate)
            )
        }
        DerivationKind::SecondOrderInstantiation {
            rule_name,
            predicate,
            inferred_count,
        } => {
            format!(
                "second-order [{}] on \"{}\": {} inferred",
                rule_name,
                engine.resolve_label(*predicate),
                inferred_count
            )
        }
        DerivationKind::NartCreation {
            function,
            arg_count,
        } => {
            format!(
                "NART creation: ({} ...{} args)",
                engine.resolve_label(*function),
                arg_count
            )
        }
        DerivationKind::CodeGenerated {
            scope,
            source_count,
        } => {
            format!("Code generated: {scope} from {source_count} source(s)")
        }
        DerivationKind::CodeRefinement {
            attempt,
            error_count,
        } => {
            format!("Code refinement: attempt {attempt}, {error_count} error(s)")
        }
        DerivationKind::LibraryLearning {
            pattern_name,
            occurrences,
            compression,
        } => {
            format!(
                "Library learning: pattern \"{pattern_name}\" ({occurrences} occurrences, compression {compression:.1})"
            )
        }
        DerivationKind::AutonomousGoalGeneration { drive, strength } => {
            format!("Autonomous goal generation: drive \"{drive}\" (strength {strength:.2})")
        }
        DerivationKind::HtnDecomposition {
            method_name,
            strategy,
            subtask_count,
        } => {
            format!(
                "HTN decomposition: method \"{method_name}\" (strategy: {strategy}, {subtask_count} subtasks)"
            )
        }
        DerivationKind::PriorityArgumentation {
            goal,
            old_priority,
            new_priority,
            audience,
            net_score,
        } => {
            format!(
                "Priority argumentation: \"{}\" {old_priority}→{new_priority} (audience: {audience}, net: {net_score:.2})",
                engine.resolve_label(*goal),
            )
        }
        DerivationKind::ProjectCreated { name } => {
            format!("Project created: \"{name}\"")
        }
        DerivationKind::WatchFired {
            watch_id,
            condition_summary,
        } => {
            format!("Watch fired: \"{watch_id}\" — {condition_summary}")
        }
        DerivationKind::MetacognitiveEvaluation {
            goal,
            signal,
            improvement_rate,
            competence,
        } => {
            format!(
                "Metacognitive evaluation: \"{}\" signal={signal} (improvement: {improvement_rate:.2}, competence: {competence:.2})",
                engine.resolve_label(*goal),
            )
        }
        DerivationKind::ResourceAssessment {
            goal,
            voc,
            opportunity_cost,
            diminishing_returns,
        } => {
            format!(
                "Resource assessment: \"{}\" VOC={voc:.2}, opportunity_cost={opportunity_cost:.2}, diminishing_returns={diminishing_returns}",
                engine.resolve_label(*goal),
            )
        }
        DerivationKind::ProceduralLearning {
            source_goal,
            method_name,
            step_count,
        } => {
            format!(
                "Procedural learning: compiled method \"{method_name}\" ({step_count} steps) from goal \"{}\"",
                engine.resolve_label(*source_goal),
            )
        }
    }
}
