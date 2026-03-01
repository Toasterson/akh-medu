//! akh CLI: neuro-symbolic AI engine.

use std::path::PathBuf;
#[cfg(not(feature = "client-only"))]
use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand, ValueEnum};
use miette::{IntoDiagnostic, Result};

#[cfg(not(feature = "client-only"))]
use akh_medu::agent::{Agent, AgentConfig};
#[cfg(not(feature = "client-only"))]
use akh_medu::client::discover_server;
use akh_medu::client::AkhClient;
#[cfg(not(feature = "client-only"))]
use akh_medu::engine::{Engine, EngineConfig};
#[cfg(not(feature = "client-only"))]
use akh_medu::error::EngineError;
#[cfg(not(feature = "client-only"))]
use akh_medu::glyph;
#[cfg(not(feature = "client-only"))]
use akh_medu::grammar::Language;
#[cfg(not(feature = "client-only"))]
use akh_medu::graph::Triple;
#[cfg(not(feature = "client-only"))]
use akh_medu::symbol::SymbolId;
#[cfg(not(feature = "client-only"))]
use akh_medu::vsa::Dimension;

#[derive(Clone, ValueEnum)]
enum IngestFormat {
    Json,
    Csv,
    Text,
}

#[derive(Clone, ValueEnum)]
enum CsvFormat {
    Spo,
    Entity,
}

#[derive(Clone, ValueEnum)]
enum EnergyLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone, ValueEnum)]
enum GtdState {
    Inbox,
    Next,
    Waiting,
    Someday,
    Reference,
}

#[derive(Clone, ValueEnum)]
enum ParaCategory {
    Project,
    Area,
    Resource,
    Archive,
}

#[derive(Clone, ValueEnum)]
enum ProactivityLevel {
    Ambient,
    Nudge,
    Offer,
    Scheduled,
    Autonomous,
}

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

        /// File format: json (default), csv, or text.
        #[arg(long, default_value = "json")]
        format: IngestFormat,

        /// CSV format: spo (subject,predicate,object) or entity (headers=predicates).
        #[arg(long, default_value = "spo")]
        csv_format: CsvFormat,

        /// Maximum sentences to process for text format.
        #[arg(long, default_value = "100")]
        max_sentences: usize,
    },

    /// Manage skillpacks.
    Skill {
        #[command(subcommand)]
        action: SkillAction,
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

    /// Personal information management (Phase 13e).
    Pim {
        #[command(subcommand)]
        action: PimAction,
    },

    /// Calendar & temporal reasoning (Phase 13f).
    Cal {
        #[command(subcommand)]
        action: CalAction,
    },

    /// Preference learning & proactive assistance (Phase 13g).
    Pref {
        #[command(subcommand)]
        action: PrefAction,
    },

    /// Causal world model (Phase 15a).
    Causal {
        #[command(subcommand)]
        action: CausalAction,
    },

    /// Identity awakening (Phase 14a+14b).
    Awaken {
        #[command(subcommand)]
        action: AwakenAction,
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
        /// Fresh start: ignore persisted session and goals.
        #[arg(long)]
        fresh: bool,
    },

    /// Manage the shared content library (ingest books, websites, documents).
    Library {
        #[command(subcommand)]
        action: LibraryAction,
    },

    /// Manage akhomed as a macOS launchd service.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Subcommand)]
enum ServiceAction {
    /// Install akhomed as a launchd service (auto-restart, run at login).
    Install {
        /// Server port (default: 8200).
        #[arg(long)]
        port: Option<u16>,
    },
    /// Start the launchd service.
    Start,
    /// Stop the launchd service (graceful shutdown).
    Stop,
    /// Show launchd service status (PID, loaded state, exit code).
    Status,
    /// Uninstall the launchd service (unload + remove plist).
    Uninstall,
    /// Print the generated plist XML (dry-run, does not install).
    Show {
        /// Server port (default: 8200).
        #[arg(long)]
        port: Option<u16>,
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
enum AgentAction {
    /// Run agent until goals complete or max cycles reached.
    Run {
        /// Goal descriptions.
        #[arg(long, value_delimiter = ',')]
        goals: Vec<String>,
        /// Maximum OODA cycles.
        #[arg(long, default_value = "10")]
        max_cycles: usize,
        /// Fresh start: ignore persisted goals from previous sessions.
        #[arg(long)]
        fresh: bool,
    },
    /// Resume a previously persisted session.
    Resume {
        /// Maximum OODA cycles.
        #[arg(long, default_value = "10")]
        max_cycles: usize,
    },
    /// Interactive REPL (launches TUI).
    Repl {
        /// Goal descriptions. Omit to resume existing goals.
        #[arg(long, value_delimiter = ',')]
        goals: Option<Vec<String>>,
        /// Headless mode: use plain stdin/stdout instead of TUI.
        #[arg(long)]
        headless: bool,
    },
    /// Interactive chat (launches TUI). DEPRECATED: use `akh chat` instead.
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
enum PimAction {
    /// Show all inbox items.
    Inbox,
    /// Show next actions (optionally filtered by context and energy).
    Next {
        /// Filter by GTD context (e.g. "computer", "office").
        #[arg(long)]
        context: Option<String>,
        /// Filter by energy level.
        #[arg(long)]
        energy: Option<EnergyLevel>,
    },
    /// Run a GTD weekly review.
    Review,
    /// Show tasks for a PARA project.
    Project {
        /// Project name.
        name: String,
    },
    /// Add PIM metadata to an existing goal (by numeric ID).
    Add {
        /// Goal symbol ID.
        #[arg(long)]
        goal: u64,
        /// Initial GTD state.
        #[arg(long, default_value = "inbox")]
        gtd: GtdState,
        /// Urgency (0.0–1.0).
        #[arg(long, default_value = "0.5")]
        urgency: f32,
        /// Importance (0.0–1.0).
        #[arg(long, default_value = "0.5")]
        importance: f32,
        /// PARA category.
        #[arg(long)]
        para: Option<ParaCategory>,
        /// GTD contexts (e.g. "computer,office").
        #[arg(long, value_delimiter = ',')]
        contexts: Option<Vec<String>>,
        /// Recurrence pattern (e.g. "daily", "weekly:mon,fri", "every:3d").
        #[arg(long)]
        recur: Option<String>,
        /// Deadline (unix timestamp).
        #[arg(long)]
        deadline: Option<u64>,
    },
    /// Transition a task's GTD state.
    Transition {
        /// Goal symbol ID.
        #[arg(long)]
        goal: u64,
        /// Target GTD state.
        #[arg(long)]
        to: GtdState,
    },
    /// Show Eisenhower matrix.
    Matrix,
    /// Show dependency graph.
    Deps,
    /// Show overdue tasks.
    Overdue,
}

#[derive(Subcommand)]
enum CalAction {
    /// Show today's calendar events.
    Today,
    /// Show this week's calendar events (next 7 days).
    Week,
    /// Detect scheduling conflicts.
    Conflicts,
    /// Add a new calendar event.
    Add {
        /// Event summary / title.
        #[arg(long)]
        summary: String,
        /// Start time (UNIX timestamp).
        #[arg(long)]
        start: u64,
        /// End time (UNIX timestamp).
        #[arg(long)]
        end: u64,
        /// Location (optional).
        #[arg(long)]
        location: Option<String>,
    },
    /// Import events from an iCalendar (.ics) file.
    Import {
        /// Path to .ics file.
        #[arg(long)]
        file: std::path::PathBuf,
    },
    /// Sync from a CalDAV server.
    Sync {
        /// CalDAV URL.
        #[arg(long)]
        url: String,
        /// Username.
        #[arg(long)]
        user: String,
        /// Password.
        #[arg(long)]
        pass: String,
    },
}

#[derive(Subcommand)]
enum PrefAction {
    /// Show current preference profile status.
    Status,
    /// Train preference profile with explicit feedback on an entity.
    Train {
        /// Entity symbol ID.
        #[arg(long)]
        entity: u64,
        /// Weight [-1.0, 1.0]. Positive = interest, negative = disinterest.
        #[arg(long, default_value = "1.0")]
        weight: f32,
    },
    /// Set proactivity level.
    Level {
        /// Level: ambient, nudge, offer, scheduled, autonomous.
        level: ProactivityLevel,
    },
    /// Show top interest topics.
    Interests {
        /// Number of interests to show.
        #[arg(long, default_value = "10")]
        count: usize,
    },
    /// Run JITIR and show suggestions.
    Suggest,
}

#[derive(Subcommand)]
enum CausalAction {
    /// List all registered action schemas.
    Schemas,
    /// Show details of a specific action schema.
    Schema {
        /// Schema name (matches tool name).
        name: String,
    },
    /// Predict the effects of executing an action.
    Predict {
        /// Action (schema) name.
        name: String,
    },
    /// Show applicable actions in the current KG state.
    Applicable,
    /// Bootstrap schemas from the tool registry.
    Bootstrap,
}

#[derive(Subcommand)]
enum AwakenAction {
    /// Parse a purpose/identity statement from the operator.
    Parse {
        /// The purpose statement (e.g., "You are the Architect based on Ptah, expert in systems").
        statement: String,
    },
    /// Resolve an identity reference via static tables or external APIs.
    Resolve {
        /// Name of the figure to resolve (e.g., "Ptah", "Gandalf", "Turing").
        name: String,
    },
    /// Show current psyche/identity state.
    Status,
    /// Expand seed concepts into a skeleton ontology via external knowledge sources.
    Expand {
        /// Seed concepts (e.g., "compiler,optimization,parsing").
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<String>>,
        /// Purpose statement to extract seeds from (e.g., "GCC compiler expert").
        #[arg(long)]
        purpose: Option<String>,
        /// VSA similarity threshold for candidate acceptance (default 0.6).
        #[arg(long, default_value = "0.6")]
        threshold: f32,
        /// Maximum number of concepts to create (default 200).
        #[arg(long, default_value = "200")]
        max_concepts: usize,
        /// Disable ConceptNet queries.
        #[arg(long)]
        no_conceptnet: bool,
    },
    /// Discover prerequisite relationships and classify concepts by Vygotsky ZPD zones.
    Prerequisite {
        /// Seed concepts (e.g., "compiler,optimization,parsing").
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<String>>,
        /// Purpose statement to extract seeds from (e.g., "GCC compiler expert").
        #[arg(long)]
        purpose: Option<String>,
        /// Minimum triple count for a concept to be classified as "Known" (default 5).
        #[arg(long, default_value = "5")]
        known_threshold: usize,
        /// Lower ZPD similarity bound for "Proximal" zone (default 0.3).
        #[arg(long, default_value = "0.3")]
        zpd_low: f32,
        /// Upper ZPD similarity bound for "Proximal" zone (default 0.7).
        #[arg(long, default_value = "0.7")]
        zpd_high: f32,
    },
    /// Discover learning resources for ZPD-proximal concepts.
    Resources {
        /// Seed concepts (e.g., "rust,compiler").
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<String>>,
        /// Purpose statement to extract seeds from.
        #[arg(long)]
        purpose: Option<String>,
        /// Minimum quality score for resources (default 0.2).
        #[arg(long, default_value = "0.2")]
        min_quality: f32,
        /// Maximum API calls across all sources (default 60).
        #[arg(long, default_value = "60")]
        max_api_calls: usize,
        /// Disable Semantic Scholar queries.
        #[arg(long)]
        no_semantic_scholar: bool,
        /// Disable OpenAlex queries.
        #[arg(long)]
        no_openalex: bool,
        /// Disable Open Library queries.
        #[arg(long)]
        no_open_library: bool,
    },
    /// Ingest discovered resources in curriculum order (expand -> prereq -> resources -> ingest).
    Ingest {
        /// Seed concepts (e.g., "rust,compiler").
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<String>>,
        /// Purpose statement to extract seeds from.
        #[arg(long)]
        purpose: Option<String>,
        /// Maximum ingestion cycles (default 500).
        #[arg(long, default_value = "500")]
        max_cycles: usize,
        /// Consecutive zero-triple results to consider a concept saturated (default 3).
        #[arg(long, default_value = "3")]
        saturation: usize,
        /// Cross-validation confidence boost (default 0.15).
        #[arg(long, default_value = "0.15")]
        xval_boost: f32,
        /// Disable URL ingestion for open-access resources.
        #[arg(long)]
        no_url: bool,
        /// Override the library catalog directory.
        #[arg(long)]
        catalog_dir: Option<String>,
    },
    /// Assess competence: expand -> prereq -> resources -> ingest -> assess.
    Assess {
        /// Seed concepts (e.g., "rust,compiler").
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<String>>,
        /// Purpose statement to extract seeds from.
        #[arg(long)]
        purpose: Option<String>,
        /// Minimum triples per concept for "known" classification (default 3).
        #[arg(long, default_value = "3")]
        min_triples: usize,
        /// Maximum Bloom depth to evaluate (1–4, default 4).
        #[arg(long, default_value = "4")]
        bloom_depth: usize,
        /// Print per-knowledge-area breakdown with score components.
        #[arg(long)]
        verbose: bool,
    },
    /// Full bootstrap pipeline: purpose → identity → expand → learn loop → target competence.
    Bootstrap {
        /// Purpose/identity statement (e.g., "You are the Architect based on Ptah, expert in systems").
        #[arg(conflicts_with_all = ["resume", "status"])]
        statement: Option<String>,
        /// Only parse and show the plan — do not execute.
        #[arg(long)]
        plan_only: bool,
        /// Resume an interrupted bootstrap session.
        #[arg(long, conflicts_with_all = ["statement", "status"])]
        resume: bool,
        /// Show current bootstrap session status.
        #[arg(long, conflicts_with_all = ["statement", "resume"])]
        status: bool,
        /// Maximum learning cycles (default 10).
        #[arg(long, default_value = "10")]
        max_cycles: usize,
        /// Separate identity override (e.g., "Gandalf").
        #[arg(long)]
        identity: Option<String>,
    },
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
#[cfg(not(feature = "client-only"))]
fn resolve_client(
    workspace: &str,
    config: EngineConfig,
    xdg_paths: Option<&akh_medu::paths::AkhPaths>,
) -> Result<AkhClient> {
    if let Some(paths) = xdg_paths
        && let Some(server) = discover_server(paths) {
            return Ok(AkhClient::remote(&server, workspace));
        }
    eprintln!("warning: akhomed not running, using local engine");
    let engine = Engine::new(config).into_diagnostic()?;
    Ok(AkhClient::local(Arc::new(engine)))
}

/// Resolve an [`AkhClient`]: requires a running akhomed server (client-only mode).
#[cfg(feature = "client-only")]
fn resolve_client(
    workspace: &str,
    xdg_paths: Option<&akh_medu::paths::AkhPaths>,
) -> Result<AkhClient> {
    akh_medu::client::require_server(workspace, xdg_paths).into_diagnostic()
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

    // In client-only mode, route all commands through a running akhomed server.
    #[cfg(feature = "client-only")]
    return run_client_only(cli);

    // Normal mode: resolve paths and engine config, then dispatch locally.
    #[cfg(not(feature = "client-only"))]
    {
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
                    if let Ok(engine) = Engine::new(engine_config)
                        && let Some(role) = engine.assigned_role() {
                            println!("  Role: {role}");
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

            match format {
                IngestFormat::Json => {
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
                        if let Err(e) = engine.persist() {
                            eprintln!("warning: failed to persist after label-triple ingest: {e}");
                        }
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
                IngestFormat::Csv => {
                    use akh_medu::agent::tool::{Tool, ToolInput};
                    use akh_medu::agent::tools::CsvIngestTool;

                    let csv_format_str = match csv_format {
                        CsvFormat::Spo => "spo",
                        CsvFormat::Entity => "entity",
                    };
                    let input = ToolInput::new()
                        .with_param("path", file.to_str().unwrap_or(""))
                        .with_param("format", csv_format_str);

                    let tool = CsvIngestTool;
                    let output = tool.execute(&engine, input).into_diagnostic()?;
                    println!("{}", output.result);
                }
                IngestFormat::Text => {
                    use akh_medu::agent::tool::{Tool, ToolInput};
                    use akh_medu::agent::tools::TextIngestTool;

                    let input = ToolInput::new()
                        .with_param("text", format!("file:{}", file.display()))
                        .with_param("max_sentences", max_sentences.to_string());

                    let tool = TextIngestTool;
                    let output = tool.execute(&engine, input).into_diagnostic()?;
                    println!("{}", output.result);
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

            if let Err(e) = engine.persist() {
                eprintln!("warning: failed to persist after ingest: {e}");
            }
            println!("{}", engine.info());
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
                    print_daemon_status(&status);
                    return Ok(());
                }
                // Try remote TUI via akhomed for Chat/Repl (non-headless only).
                #[cfg(feature = "daemon")]
                AgentAction::Chat { headless: false, .. } | AgentAction::Repl { headless: false, .. } => {
                    if let Some(ref paths) = xdg_paths
                        && let Some(server_info) = discover_server(paths)
                    {
                        eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                        return akh_medu::tui::launch_remote(&cli.workspace, &server_info);
                    }
                    eprintln!("warning: akhomed not running, using local engine");
                }
                _ => {}
            }

            let engine = Arc::new(Engine::new(config).into_diagnostic()?);

            match action {
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

                    for goal_str in &goals {
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
                    let goals_joined = goals.join(",");
                    let summary = agent.synthesize_findings(&goals_joined);
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
                        if let Some(ref goal_list) = goals {
                            for goal_str in goal_list {
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

                        if let Some(ref goal_list) = goals {
                            for goal_str in goal_list {
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

                AgentAction::Chat {
                    max_cycles: _,
                    fresh,
                    headless,
                } => {
                    eprintln!("warning: `akh agent chat` is deprecated. Use `akh chat` instead.");

                    if !headless {
                        let agent_config = AgentConfig {
                            max_cycles: 20,
                            ..Default::default()
                        };
                        let ws_name = cli.workspace.clone();
                        akh_medu::tui::launch(&ws_name, Arc::clone(&engine), agent_config, fresh)?;
                    } else {
                        let agent_config = AgentConfig {
                            max_cycles: 20,
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

                        let data_dir = engine.config().data_dir.as_deref();
                        let nlu_pipeline = engine
                            .store()
                            .get_meta(b"nlu_ranker_state")
                            .ok()
                            .flatten()
                            .and_then(|bytes| akh_medu::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
                            .map(|ranker| akh_medu::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir))
                            .unwrap_or_else(|| akh_medu::nlu::NluPipeline::new_with_models(data_dir));
                        let mut chat_processor = akh_medu::chat::ChatProcessor::new(&engine, nlu_pipeline);

                        println!("akh chat (headless). Type 'quit' to exit.\n");
                        use std::io::Write as _;

                        loop {
                            print!("> ");
                            std::io::stdout().flush().ok();
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

                            let responses = chat_processor.process_input(trimmed, &mut agent, &engine);
                            for msg in &responses {
                                println!("{}", msg.to_plain_text());
                            }
                            println!();
                        }

                        chat_processor.persist_nlu_state(&engine);
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
                    if let Some(ref paths) = xdg_paths
                        && let Some(server) = discover_server(paths) {
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
                    let shared_agent = Arc::new(Mutex::new(agent));
                    let mut daemon = AgentDaemon::new(shared_agent, daemon_config);
                    rt.block_on(daemon.run()).into_diagnostic()?;
                }

                // DaemonStop and DaemonStatus are handled above (early return).
                AgentAction::DaemonStop | AgentAction::DaemonStatus => unreachable!(),
            }
        }

        Commands::Pim { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);
            let agent_config = akh_medu::agent::AgentConfig::default();
            let mut agent = if akh_medu::agent::Agent::has_persisted_session(&engine) {
                akh_medu::agent::Agent::resume(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            } else {
                akh_medu::agent::Agent::new(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            };

            match action {
                PimAction::Inbox => {
                    let ids = agent.pim_manager().tasks_by_gtd_state(
                        akh_medu::agent::GtdState::Inbox,
                    );
                    if ids.is_empty() {
                        println!("Inbox is empty.");
                    } else {
                        println!("Inbox ({} items):", ids.len());
                        for id in &ids {
                            let label = engine.resolve_label(*id);
                            let quadrant = agent
                                .pim_manager()
                                .get_metadata(id.get())
                                .map(|m| m.quadrant.to_string())
                                .unwrap_or_default();
                            println!("  [{:>5}] {} ({})", id.get(), label, quadrant);
                        }
                    }
                }
                PimAction::Next { context, energy } => {
                    let ctx = context.map(akh_medu::agent::PimContext);
                    let nrg = energy.map(|e| match e {
                        EnergyLevel::Low => akh_medu::agent::EnergyLevel::Low,
                        EnergyLevel::Medium => akh_medu::agent::EnergyLevel::Medium,
                        EnergyLevel::High => akh_medu::agent::EnergyLevel::High,
                    });
                    let ids = agent.pim_manager().available_tasks(
                        ctx.as_ref(),
                        nrg,
                        agent.goals(),
                    );
                    if ids.is_empty() {
                        println!("No next actions available.");
                    } else {
                        println!("Next actions ({} tasks):", ids.len());
                        for id in &ids {
                            let label = engine.resolve_label(*id);
                            let meta = agent.pim_manager().get_metadata(id.get());
                            let quadrant = meta
                                .map(|m| m.quadrant.to_string())
                                .unwrap_or_default();
                            let energy_str = meta
                                .and_then(|m| m.energy)
                                .map(|e| format!(" [{}]", e))
                                .unwrap_or_default();
                            println!(
                                "  [{:>5}] {} ({}){}",
                                id.get(),
                                label,
                                quadrant,
                                energy_str
                            );
                        }
                    }
                }
                PimAction::Review => {
                    let review = akh_medu::agent::pim::gtd_weekly_review(
                        agent.pim_manager(),
                        agent.goals(),
                        agent.projects(),
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                    println!("{}", review.summary);
                    if !review.overdue.is_empty() {
                        println!("\nOverdue:");
                        for id in &review.overdue {
                            println!("  [{:>5}] {}", id.get(), engine.resolve_label(*id));
                        }
                    }
                    if !review.stale_inbox.is_empty() {
                        println!("\nStale inbox (>7 days):");
                        for id in &review.stale_inbox {
                            println!("  [{:>5}] {}", id.get(), engine.resolve_label(*id));
                        }
                    }
                    if !review.stalled_projects.is_empty() {
                        println!("\nStalled projects (no Next actions):");
                        for id in &review.stalled_projects {
                            println!("  [{:>5}] {}", id.get(), engine.resolve_label(*id));
                        }
                    }
                    if !review.adjustments.is_empty() {
                        println!("\nRecommended adjustments: {}", review.adjustments.len());
                    }
                }
                PimAction::Project { name } => {
                    let project = agent
                        .projects()
                        .iter()
                        .find(|p| p.name == name);
                    match project {
                        Some(p) => {
                            println!("Project: {} ({:?})", p.name, p.status);
                            for gid in &p.goals {
                                let label = engine.resolve_label(*gid);
                                let gtd = agent
                                    .pim_manager()
                                    .get_metadata(gid.get())
                                    .map(|m| m.gtd_state.to_string())
                                    .unwrap_or_else(|| "no-pim".into());
                                println!("  [{:>5}] {} (GTD: {})", gid.get(), label, gtd);
                            }
                        }
                        None => println!("Project '{}' not found.", name),
                    }
                }
                PimAction::Add {
                    goal,
                    gtd,
                    urgency,
                    importance,
                    para,
                    contexts,
                    recur,
                    deadline,
                } => {
                    let gtd_state = match gtd {
                        GtdState::Inbox => akh_medu::agent::GtdState::Inbox,
                        GtdState::Next => akh_medu::agent::GtdState::Next,
                        GtdState::Waiting => akh_medu::agent::GtdState::Waiting,
                        GtdState::Someday => akh_medu::agent::GtdState::Someday,
                        GtdState::Reference => akh_medu::agent::GtdState::Reference,
                    };
                    let goal_sym = akh_medu::symbol::SymbolId::new(goal)
                        .ok_or_else(|| miette::miette!("invalid goal ID: {goal}"))?;

                    agent
                        .pim_manager_mut()
                        .add_task(&engine, goal_sym, gtd_state, urgency, importance)
                        .into_diagnostic()?;

                    if let Some(ref para_val) = para {
                        let cat = match para_val {
                            ParaCategory::Project => akh_medu::agent::ParaCategory::Project,
                            ParaCategory::Area => akh_medu::agent::ParaCategory::Area,
                            ParaCategory::Resource => akh_medu::agent::ParaCategory::Resource,
                            ParaCategory::Archive => akh_medu::agent::ParaCategory::Archive,
                        };
                        agent
                            .pim_manager_mut()
                            .set_para(&engine, goal_sym, cat)
                            .into_diagnostic()?;
                    }

                    if let Some(ref ctx_list) = contexts {
                        for ctx in ctx_list {
                            agent
                                .pim_manager_mut()
                                .add_context(
                                    &engine,
                                    goal_sym,
                                    akh_medu::agent::PimContext(ctx.trim().to_string()),
                                )
                                .into_diagnostic()?;
                        }
                    }

                    if let Some(ref recur_str) = recur {
                        let recurrence = akh_medu::agent::Recurrence::parse(recur_str)
                            .into_diagnostic()?;
                        agent
                            .pim_manager_mut()
                            .set_recurrence(&engine, goal_sym, recurrence)
                            .into_diagnostic()?;
                    }

                    if let Some(dl) = deadline
                        && let Some(meta) =
                            agent.pim_manager_mut().get_metadata_mut(goal_sym.get())
                        {
                            meta.deadline = Some(dl);
                        }

                    println!(
                        "Added PIM metadata to goal {} (GTD: {}, quadrant: {})",
                        goal,
                        gtd_state,
                        akh_medu::agent::EisenhowerQuadrant::classify(urgency, importance),
                    );
                }
                PimAction::Transition { goal, to } => {
                    let new_state = match to {
                        GtdState::Inbox => akh_medu::agent::GtdState::Inbox,
                        GtdState::Next => akh_medu::agent::GtdState::Next,
                        GtdState::Waiting => akh_medu::agent::GtdState::Waiting,
                        GtdState::Someday => akh_medu::agent::GtdState::Someday,
                        GtdState::Reference => akh_medu::agent::GtdState::Reference,
                    };
                    let goal_sym = akh_medu::symbol::SymbolId::new(goal)
                        .ok_or_else(|| miette::miette!("invalid goal ID: {goal}"))?;
                    agent
                        .pim_manager_mut()
                        .transition_gtd(&engine, goal_sym, new_state)
                        .into_diagnostic()?;
                    println!("Transitioned goal {} to GTD state: {}", goal, new_state);
                }
                PimAction::Matrix => {
                    for quad in [
                        akh_medu::agent::EisenhowerQuadrant::Do,
                        akh_medu::agent::EisenhowerQuadrant::Schedule,
                        akh_medu::agent::EisenhowerQuadrant::Delegate,
                        akh_medu::agent::EisenhowerQuadrant::Eliminate,
                    ] {
                        let ids = agent.pim_manager().tasks_by_quadrant(quad);
                        println!(
                            "{} ({} tasks):",
                            quad.as_label().to_uppercase(),
                            ids.len()
                        );
                        for id in &ids {
                            let label = engine.resolve_label(*id);
                            let gtd = agent
                                .pim_manager()
                                .get_metadata(id.get())
                                .map(|m| m.gtd_state.to_string())
                                .unwrap_or_default();
                            println!("  [{:>5}] {} (GTD: {})", id.get(), label, gtd);
                        }
                    }
                }
                PimAction::Deps => {
                    match agent.pim_manager().topological_order() {
                        Ok(topo) => {
                            println!("Dependency order ({} tasks):", topo.len());
                            for (i, id) in topo.iter().enumerate() {
                                let label = engine.resolve_label(*id);
                                println!("  {}. [{:>5}] {}", i + 1, id.get(), label);
                            }
                        }
                        Err(e) => println!("Dependency cycle detected: {e}"),
                    }
                }
                PimAction::Overdue => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let ids = agent.pim_manager().overdue_tasks(now);
                    if ids.is_empty() {
                        println!("No overdue tasks.");
                    } else {
                        println!("Overdue ({} tasks):", ids.len());
                        for id in &ids {
                            let label = engine.resolve_label(*id);
                            let due = agent
                                .pim_manager()
                                .get_metadata(id.get())
                                .and_then(|m| m.next_due)
                                .unwrap_or(0);
                            let overdue_secs = now.saturating_sub(due);
                            let overdue_days = overdue_secs / 86_400;
                            println!(
                                "  [{:>5}] {} (overdue by {} days)",
                                id.get(),
                                label,
                                overdue_days
                            );
                        }
                    }
                }
            }

            agent.persist_session().into_diagnostic()?;
        }

        Commands::Cal { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);
            let agent_config = akh_medu::agent::AgentConfig::default();
            let mut agent = if akh_medu::agent::Agent::has_persisted_session(&engine) {
                akh_medu::agent::Agent::resume(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            } else {
                akh_medu::agent::Agent::new(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            };

            match action {
                CalAction::Today => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let events = agent.calendar_manager().today_events(now);
                    if events.is_empty() {
                        println!("No events today.");
                    } else {
                        println!("Today ({} events):", events.len());
                        for e in &events {
                            let dur_min = e.duration_secs() / 60;
                            let loc = e
                                .location
                                .as_deref()
                                .map(|l| format!(" @ {l}"))
                                .unwrap_or_default();
                            println!(
                                "  [{:>5}] {} ({} min){}",
                                e.symbol_id.get(),
                                e.summary,
                                dur_min,
                                loc,
                            );
                        }
                    }
                }
                CalAction::Week => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let events = agent.calendar_manager().week_events(now);
                    if events.is_empty() {
                        println!("No events this week.");
                    } else {
                        println!("This week ({} events):", events.len());
                        for e in &events {
                            let dur_min = e.duration_secs() / 60;
                            let loc = e
                                .location
                                .as_deref()
                                .map(|l| format!(" @ {l}"))
                                .unwrap_or_default();
                            println!(
                                "  [{:>5}] {} ({} min){}",
                                e.symbol_id.get(),
                                e.summary,
                                dur_min,
                                loc,
                            );
                        }
                    }
                }
                CalAction::Conflicts => {
                    let conflicts = agent.calendar_manager().detect_conflicts();
                    if conflicts.is_empty() {
                        println!("No scheduling conflicts.");
                    } else {
                        println!("Conflicts ({}):", conflicts.len());
                        for (a, b) in &conflicts {
                            let a_label = engine.resolve_label(*a);
                            let b_label = engine.resolve_label(*b);
                            println!("  {} <-> {}", a_label, b_label);
                        }
                    }
                }
                CalAction::Add {
                    summary,
                    start,
                    end,
                    location,
                } => {
                    let sym = agent
                        .calendar_manager_mut()
                        .add_event(
                            &engine,
                            &summary,
                            start,
                            end,
                            location.as_deref(),
                            None,
                            None,
                            None,
                        )
                        .into_diagnostic()?;
                    let dur_min = end.saturating_sub(start) / 60;
                    println!(
                        "Added event [{:>5}] '{}' ({} min)",
                        sym.get(),
                        summary,
                        dur_min,
                    );
                }
                CalAction::Import { file } => {
                    #[cfg(feature = "calendar")]
                    {
                        let data = std::fs::read_to_string(&file).into_diagnostic()?;
                        let imported = akh_medu::agent::calendar::import_ical(
                            agent.calendar_manager_mut(),
                            &engine,
                            &data,
                        )
                        .into_diagnostic()?;
                        println!("Imported {} events from {}", imported.len(), file.display());
                    }
                    #[cfg(not(feature = "calendar"))]
                    {
                        let _ = file;
                        println!("iCalendar import requires --features calendar");
                    }
                }
                CalAction::Sync { url, user, pass } => {
                    #[cfg(feature = "calendar")]
                    {
                        let imported = akh_medu::agent::calendar::sync_caldav(
                            agent.calendar_manager_mut(),
                            &engine,
                            &url,
                            &user,
                            &pass,
                        )
                        .into_diagnostic()?;
                        println!("Synced {} events from CalDAV", imported.len());
                    }
                    #[cfg(not(feature = "calendar"))]
                    {
                        let _ = (url, user, pass);
                        println!("CalDAV sync requires --features calendar");
                    }
                }
            }

            agent.persist_session().into_diagnostic()?;
        }

        Commands::Pref { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);
            let agent_config = akh_medu::agent::AgentConfig::default();
            let mut agent = if akh_medu::agent::Agent::has_persisted_session(&engine) {
                akh_medu::agent::Agent::resume(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            } else {
                akh_medu::agent::Agent::new(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            };

            match action {
                PrefAction::Status => {
                    let pref = agent.preference_manager();
                    println!("Preference Profile:");
                    println!("  Interactions: {}", pref.profile.interaction_count);
                    println!("  Proactivity:  {}", pref.profile.proactivity_level);
                    println!("  Decay rate:   {}", pref.profile.decay_rate);
                    println!(
                        "  Suggestions:  {} offered, {} accepted ({:.0}%)",
                        pref.profile.suggestions_offered,
                        pref.profile.suggestions_accepted,
                        pref.suggestion_acceptance_rate() * 100.0,
                    );
                    let prototype_empty = pref.profile.interest_prototype.is_none();
                    println!(
                        "  Prototype:    {}",
                        if prototype_empty { "empty" } else { "active" }
                    );
                }
                PrefAction::Train { entity, weight } => {
                    let sym = akh_medu::symbol::SymbolId::new(entity).ok_or_else(|| {
                        miette::miette!("invalid symbol ID: {entity}")
                    })?;
                    let signal = akh_medu::agent::FeedbackSignal::ExplicitPreference {
                        topic: sym,
                        weight,
                    };
                    agent
                        .preference_manager_mut()
                        .record_feedback(&signal, &engine)
                        .into_diagnostic()?;
                    let label = engine.resolve_label(sym);
                    println!(
                        "Recorded preference: '{label}' (weight: {weight:.2}, total interactions: {})",
                        agent.preference_manager().profile.interaction_count
                    );
                }
                PrefAction::Level { level } => {
                    let lvl = match level {
                        ProactivityLevel::Ambient => akh_medu::agent::ProactivityLevel::Ambient,
                        ProactivityLevel::Nudge => akh_medu::agent::ProactivityLevel::Nudge,
                        ProactivityLevel::Offer => akh_medu::agent::ProactivityLevel::Offer,
                        ProactivityLevel::Scheduled => akh_medu::agent::ProactivityLevel::Scheduled,
                        ProactivityLevel::Autonomous => akh_medu::agent::ProactivityLevel::Autonomous,
                    };
                    agent.preference_manager_mut().set_proactivity_level(lvl);
                    println!("Proactivity level set to: {lvl}");
                }
                PrefAction::Interests { count } => {
                    let interests = agent.preference_manager().top_interests(&engine, count);
                    if interests.is_empty() {
                        println!("No interests recorded yet. Use `akh pref train` to add feedback.");
                    } else {
                        println!("Top interests ({}):", interests.len());
                        for (label, sim) in &interests {
                            println!("  {label:<30} (similarity: {sim:.3})");
                        }
                    }
                }
                PrefAction::Suggest => {
                    match agent.preference_manager().jitir_query(
                        agent.working_memory(),
                        agent.goals(),
                        &engine,
                    ) {
                        Ok(jitir) => {
                            if jitir.direct_matches.is_empty()
                                && jitir.serendipity_matches.is_empty()
                            {
                                println!("No suggestions at this time.");
                            } else {
                                if !jitir.direct_matches.is_empty() {
                                    println!("Direct matches:");
                                    for s in &jitir.direct_matches {
                                        println!(
                                            "  [{:>5}] {} (relevance: {:.2})",
                                            s.entity.get(),
                                            s.label,
                                            s.relevance
                                        );
                                    }
                                }
                                if !jitir.serendipity_matches.is_empty() {
                                    println!("Serendipity:");
                                    for s in &jitir.serendipity_matches {
                                        println!(
                                            "  [{:>5}] {} — {} (relevance: {:.2})",
                                            s.entity.get(),
                                            s.label,
                                            s.reasoning,
                                            s.relevance
                                        );
                                    }
                                }
                            }
                            println!("Context: {}", jitir.context_summary);
                        }
                        Err(e) => {
                            println!("JITIR query failed: {e}");
                        }
                    }
                }
            }

            agent.persist_session().into_diagnostic()?;
        }

        Commands::Causal { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);
            let agent_config = akh_medu::agent::AgentConfig::default();
            let mut agent = if akh_medu::agent::Agent::has_persisted_session(&engine) {
                akh_medu::agent::Agent::resume(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            } else {
                akh_medu::agent::Agent::new(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            };

            match action {
                CausalAction::Schemas => {
                    let schemas = agent.causal_manager().list_schemas();
                    if schemas.is_empty() {
                        println!("No action schemas registered. Use `akh causal bootstrap` to create from tools.");
                    } else {
                        println!("Action schemas ({}):", schemas.len());
                        for s in &schemas {
                            println!(
                                "  {:<25} precond: {} effects: {} success: {:.0}% runs: {}",
                                s.name,
                                s.preconditions.len(),
                                s.effects.len(),
                                s.success_rate * 100.0,
                                s.execution_count,
                            );
                        }
                    }
                }
                CausalAction::Schema { name } => {
                    match agent.causal_manager().get_schema(&name) {
                        Some(s) => {
                            println!("Schema: {}", s.name);
                            println!("  Action ID:       {}", s.action_id.get());
                            println!("  Preconditions:   {}", s.preconditions.len());
                            println!("  Effects:         {}", s.effects.len());
                            println!("  Success rate:    {:.1}%", s.success_rate * 100.0);
                            println!("  Execution count: {}", s.execution_count);
                        }
                        None => {
                            println!("Schema '{name}' not found.");
                        }
                    }
                }
                CausalAction::Predict { name } => {
                    match agent.causal_manager().predict_effects(&name, &engine) {
                        Ok(transition) => {
                            println!("Predicted transition for '{name}':");
                            if transition.assertions.is_empty()
                                && transition.retractions.is_empty()
                                && transition.confidence_changes.is_empty()
                            {
                                println!("  (no predicted effects)");
                            } else {
                                for (s, p, o) in &transition.assertions {
                                    println!(
                                        "  + {} {} {}",
                                        engine.resolve_label(*s),
                                        engine.resolve_label(*p),
                                        engine.resolve_label(*o),
                                    );
                                }
                                for (s, p, o) in &transition.retractions {
                                    println!(
                                        "  - {} {} {}",
                                        engine.resolve_label(*s),
                                        engine.resolve_label(*p),
                                        engine.resolve_label(*o),
                                    );
                                }
                                for (s, p, o, delta) in &transition.confidence_changes {
                                    println!(
                                        "  ~ {} {} {} (delta: {:+.2})",
                                        engine.resolve_label(*s),
                                        engine.resolve_label(*p),
                                        engine.resolve_label(*o),
                                        delta,
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!("Prediction failed: {e}");
                        }
                    }
                }
                CausalAction::Applicable => {
                    let applicable = agent.causal_manager().applicable_actions(&engine);
                    if applicable.is_empty() {
                        println!("No applicable actions in current state.");
                    } else {
                        println!("Applicable actions ({}):", applicable.len());
                        for s in &applicable {
                            println!(
                                "  {} (success: {:.0}%, runs: {})",
                                s.name,
                                s.success_rate * 100.0,
                                s.execution_count,
                            );
                        }
                    }
                }
                CausalAction::Bootstrap => {
                    let tool_names: Vec<String> = agent
                        .list_tools()
                        .iter()
                        .map(|s| s.name.clone())
                        .collect();
                    match agent
                        .causal_manager_mut()
                        .bootstrap_schemas_from_tools(&tool_names, &engine)
                    {
                        Ok(count) => {
                            println!(
                                "Bootstrapped {count} new schema(s) from {} tool(s).",
                                tool_names.len()
                            );
                        }
                        Err(e) => {
                            println!("Bootstrap failed: {e}");
                        }
                    }
                }
            }

            agent.persist_session().into_diagnostic()?;
        }

        Commands::Awaken { action } => {
            let engine = Arc::new(Engine::new(config).into_diagnostic()?);
            let agent_config = akh_medu::agent::AgentConfig::default();
            let mut agent = if akh_medu::agent::Agent::has_persisted_session(&engine) {
                akh_medu::agent::Agent::resume(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            } else {
                akh_medu::agent::Agent::new(Arc::clone(&engine), agent_config)
                    .into_diagnostic()?
            };

            match action {
                AwakenAction::Parse { statement } => {
                    match akh_medu::bootstrap::purpose::parse_purpose(&statement) {
                        Ok(intent) => {
                            println!("Parsed bootstrap intent:");
                            println!("  Domain:      {}", intent.purpose.domain);
                            println!("  Competence:  {}", intent.purpose.competence_level);
                            println!("  Seeds:       {:?}", intent.purpose.seed_concepts);
                            if let Some(ref id) = intent.identity {
                                println!("  Identity:    {} ({})", id.name, id.entity_type);
                                println!("  Source:      \"{}\"", id.source_phrase);
                            } else {
                                println!("  Identity:    (none)");
                            }
                        }
                        Err(e) => {
                            eprintln!("Parse failed: {e}");
                        }
                    }
                }
                AwakenAction::Resolve { name } => {
                    let identity_ref = akh_medu::bootstrap::IdentityRef {
                        name: name.clone(),
                        entity_type: akh_medu::bootstrap::purpose::classify_entity_type(&name),
                        source_phrase: format!("resolve {name}"),
                    };
                    match akh_medu::bootstrap::identity::resolve_identity(
                        &identity_ref,
                        &engine,
                    ) {
                        Ok(knowledge) => {
                            println!("Resolved identity: {}", knowledge.name);
                            println!("  Type:        {}", knowledge.entity_type);
                            println!("  Culture:     {}", knowledge.culture);
                            println!("  Description: {}", knowledge.description);
                            println!("  Domains:     {:?}", knowledge.domains);
                            println!("  Traits:      {:?}", knowledge.traits);
                            println!("  Archetypes:  {:?}", knowledge.archetypes);

                            // Perform the Ritual of Awakening.
                            let purpose = akh_medu::bootstrap::PurposeModel {
                                domain: knowledge.domains.first().cloned().unwrap_or_default(),
                                competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                                seed_concepts: knowledge.domains.clone(),
                                description: knowledge.description.clone(),
                            };
                            match akh_medu::bootstrap::identity::ritual_of_awakening(
                                &knowledge,
                                &purpose,
                                &engine,
                            ) {
                                Ok(ritual) => {
                                    println!("\nRitual of Awakening complete!");
                                    println!("  Chosen name: {}", ritual.chosen_name);
                                    println!(
                                        "  Persona:     {}",
                                        ritual.psyche.persona.name
                                    );
                                    println!(
                                        "  Grammar:     {}",
                                        ritual.psyche.persona.grammar_preference
                                    );
                                    println!(
                                        "  Dominant:    {}",
                                        ritual.psyche.self_integration.dominant_archetype
                                    );
                                    println!(
                                        "  Provenance:  {} record(s)",
                                        ritual.provenance_ids.len()
                                    );

                                    // Set the psyche on the agent (use force: ritual already guards).
                                    agent.force_set_psyche(ritual.psyche);
                                }
                                Err(e) => {
                                    eprintln!("Ritual failed: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Resolution failed: {e}");
                        }
                    }
                }
                AwakenAction::Status => {
                    if let Some(psyche) = agent.psyche() {
                        println!("Current psyche:");
                        println!("  Persona:          {}", psyche.persona.name);
                        println!("  Grammar:          {}", psyche.persona.grammar_preference);
                        println!("  Traits:           {:?}", psyche.persona.traits);
                        println!("  Tone:             {:?}", psyche.persona.tone);
                        println!("  Dominant:         {}", psyche.self_integration.dominant_archetype);
                        println!("  Individuation:    {:.2}", psyche.self_integration.individuation_level);
                        println!("  Archetypes:");
                        println!("    Sage:     {:.2}", psyche.archetypes.sage);
                        println!("    Explorer: {:.2}", psyche.archetypes.explorer);
                        println!("    Healer:   {:.2}", psyche.archetypes.healer);
                        println!("    Guardian: {:.2}", psyche.archetypes.guardian);
                        println!("  Shadow veto patterns: {}", psyche.shadow.veto_patterns.len());
                        println!("  Shadow bias patterns: {}", psyche.shadow.bias_patterns.len());
                    } else {
                        println!("No psyche loaded. Run `akh awaken resolve <name>` to awaken.");
                    }
                }
                AwakenAction::Expand {
                    seeds,
                    purpose: purpose_stmt,
                    threshold,
                    max_concepts,
                    no_conceptnet,
                } => {
                    // Resolve seed concepts: either from --seeds or --purpose.
                    let purpose_model = if let Some(ref seed_list) = seeds {
                        let seed_list: Vec<String> = seed_list
                            .iter()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        akh_medu::bootstrap::PurposeModel {
                            domain: seed_list.first().cloned().unwrap_or_default(),
                            competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                            seed_concepts: seed_list.clone(),
                            description: seed_list.join(","),
                        }
                    } else if let Some(ref stmt) = purpose_stmt {
                        match akh_medu::bootstrap::purpose::parse_purpose(stmt) {
                            Ok(intent) => intent.purpose,
                            Err(e) => {
                                eprintln!("Failed to parse purpose: {e}");
                                return Ok(());
                            }
                        }
                    } else {
                        eprintln!("Provide --seeds or --purpose for domain expansion.");
                        return Ok(());
                    };

                    println!(
                        "Expanding domain from {} seed(s): {:?}",
                        purpose_model.seed_concepts.len(),
                        purpose_model.seed_concepts
                    );

                    let config = akh_medu::bootstrap::ExpansionConfig {
                        similarity_threshold: threshold,
                        max_concepts,
                        use_conceptnet: !no_conceptnet,
                        ..Default::default()
                    };

                    match akh_medu::bootstrap::DomainExpander::new(&engine, config) {
                        Ok(mut expander) => {
                            match expander.expand(&purpose_model, &engine) {
                                Ok(result) => {
                                    println!("\nDomain expansion complete!");
                                    println!("  Concepts created: {}", result.concept_count);
                                    println!("  Relations added:  {}", result.relation_count);
                                    println!("  Rejected:         {}", result.rejected_count);
                                    println!("  API calls:        {}", result.api_calls);
                                    println!("  Provenance:       {} record(s)", result.provenance_ids.len());
                                    if !result.accepted_labels.is_empty() {
                                        println!("\n  Accepted concepts:");
                                        for (i, label) in result.accepted_labels.iter().enumerate().take(20) {
                                            println!("    {}: {}", i + 1, label);
                                        }
                                        if result.accepted_labels.len() > 20 {
                                            println!("    ... and {} more", result.accepted_labels.len() - 20);
                                        }
                                    }
                                }
                                Err(e) => eprintln!("Expansion failed: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Failed to initialize expander: {e}"),
                    }
                }
                AwakenAction::Prerequisite {
                    seeds,
                    purpose: purpose_stmt,
                    known_threshold,
                    zpd_low,
                    zpd_high,
                } => {
                    // Resolve seed concepts: either from --seeds or --purpose.
                    let purpose_model = if let Some(ref seed_list) = seeds {
                        let seed_list: Vec<String> = seed_list
                            .iter()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        akh_medu::bootstrap::PurposeModel {
                            domain: seed_list.first().cloned().unwrap_or_default(),
                            competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                            seed_concepts: seed_list.clone(),
                            description: seed_list.join(","),
                        }
                    } else if let Some(ref stmt) = purpose_stmt {
                        match akh_medu::bootstrap::purpose::parse_purpose(stmt) {
                            Ok(intent) => intent.purpose,
                            Err(e) => {
                                eprintln!("Failed to parse purpose: {e}");
                                return Ok(());
                            }
                        }
                    } else {
                        eprintln!("Provide --seeds or --purpose for prerequisite analysis.");
                        return Ok(());
                    };

                    // Step 1: Run domain expansion first to populate KG.
                    println!(
                        "Expanding domain from {} seed(s): {:?}",
                        purpose_model.seed_concepts.len(),
                        purpose_model.seed_concepts
                    );

                    let expand_config = akh_medu::bootstrap::ExpansionConfig::default();
                    let expansion_result =
                        match akh_medu::bootstrap::DomainExpander::new(&engine, expand_config) {
                            Ok(mut expander) => match expander.expand(&purpose_model, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Expansion: {} concepts, {} relations",
                                        result.concept_count, result.relation_count
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Expansion failed: {e}");
                                    return Ok(());
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to initialize expander: {e}");
                                return Ok(());
                            }
                        };

                    // Step 2: Run prerequisite analysis.
                    println!("\nAnalyzing prerequisites...");
                    let prereq_config = akh_medu::bootstrap::PrerequisiteConfig {
                        known_min_triples: known_threshold,
                        proximal_similarity_low: zpd_low,
                        proximal_similarity_high: zpd_high,
                        ..Default::default()
                    };

                    match akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config) {
                        Ok(analyzer) => {
                            match analyzer.analyze(&expansion_result, &engine) {
                                Ok(result) => {
                                    println!("\nPrerequisite analysis complete!");
                                    println!("  Concepts analyzed: {}", result.concepts_analyzed);
                                    println!("  Prerequisite edges: {}", result.edge_count);
                                    println!("  Cycles broken: {}", result.cycles_broken);
                                    println!("  Max tier: {}", result.max_tier);
                                    println!(
                                        "  Provenance: {} record(s)",
                                        result.provenance_ids.len()
                                    );

                                    // Zone distribution.
                                    println!("\n  ZPD Distribution:");
                                    for zone in [
                                        akh_medu::bootstrap::prerequisite::ZpdZone::Known,
                                        akh_medu::bootstrap::prerequisite::ZpdZone::Proximal,
                                        akh_medu::bootstrap::prerequisite::ZpdZone::Beyond,
                                    ] {
                                        let count =
                                            result.zone_distribution.get(&zone).unwrap_or(&0);
                                        println!("    {zone}: {count}");
                                    }

                                    // Curriculum.
                                    if !result.curriculum.is_empty() {
                                        println!("\n  Curriculum (learning order):");
                                        for (i, entry) in
                                            result.curriculum.iter().enumerate().take(30)
                                        {
                                            println!(
                                                "    {:3}. [tier {}] ({}) {} (coverage: {:.2}, sim: {:.2})",
                                                i + 1,
                                                entry.tier,
                                                entry.zone,
                                                entry.label,
                                                entry.prereq_coverage,
                                                entry.similarity_to_known,
                                            );
                                        }
                                        if result.curriculum.len() > 30 {
                                            println!(
                                                "    ... and {} more",
                                                result.curriculum.len() - 30
                                            );
                                        }
                                    }
                                }
                                Err(e) => eprintln!("Prerequisite analysis failed: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Failed to initialize analyzer: {e}"),
                    }
                }
                AwakenAction::Resources {
                    seeds,
                    purpose: purpose_stmt,
                    min_quality,
                    max_api_calls,
                    no_semantic_scholar,
                    no_openalex,
                    no_open_library,
                } => {
                    // Resolve seed concepts.
                    let purpose_model = if let Some(ref seed_list) = seeds {
                        let seed_list: Vec<String> = seed_list
                            .iter()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        akh_medu::bootstrap::PurposeModel {
                            domain: seed_list.first().cloned().unwrap_or_default(),
                            competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                            seed_concepts: seed_list.clone(),
                            description: seed_list.join(","),
                        }
                    } else if let Some(ref stmt) = purpose_stmt {
                        match akh_medu::bootstrap::purpose::parse_purpose(stmt) {
                            Ok(intent) => intent.purpose,
                            Err(e) => {
                                eprintln!("Failed to parse purpose: {e}");
                                return Ok(());
                            }
                        }
                    } else {
                        eprintln!("Provide --seeds or --purpose for resource discovery.");
                        return Ok(());
                    };

                    // Step 1: Domain expansion.
                    println!(
                        "Expanding domain from {} seed(s): {:?}",
                        purpose_model.seed_concepts.len(),
                        purpose_model.seed_concepts
                    );
                    let expand_config = akh_medu::bootstrap::ExpansionConfig::default();
                    let expansion_result =
                        match akh_medu::bootstrap::DomainExpander::new(&engine, expand_config) {
                            Ok(mut expander) => match expander.expand(&purpose_model, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Expansion: {} concepts, {} relations",
                                        result.concept_count, result.relation_count
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Expansion failed: {e}");
                                    return Ok(());
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to initialize expander: {e}");
                                return Ok(());
                            }
                        };

                    // Step 2: Prerequisite analysis (fallback to synthetic curriculum if it fails).
                    println!("\nAnalyzing prerequisites...");
                    let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
                    let prereq_result =
                        match akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config)
                        {
                            Ok(analyzer) => match analyzer.analyze(&expansion_result, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Prerequisites: {} edges, {} concepts",
                                        result.edge_count, result.concepts_analyzed
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!(
                                        "  Prerequisite analysis failed: {e}\n  \
                                         Falling back to synthetic curriculum (all concepts → Proximal)"
                                    );
                                    akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                        &expansion_result,
                                        &engine,
                                    )
                                }
                            },
                            Err(e) => {
                                eprintln!(
                                    "  Failed to initialize analyzer: {e}\n  \
                                     Falling back to synthetic curriculum (all concepts → Proximal)"
                                );
                                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                    &expansion_result,
                                    &engine,
                                )
                            }
                        };

                    // Step 3: Resource discovery.
                    println!("\nDiscovering resources for proximal concepts...");
                    let res_config = akh_medu::bootstrap::ResourceDiscoveryConfig {
                        min_quality,
                        max_api_calls,
                        use_semantic_scholar: !no_semantic_scholar,
                        use_openalex: !no_openalex,
                        use_open_library: !no_open_library,
                        ..Default::default()
                    };

                    match akh_medu::bootstrap::ResourceDiscoverer::new(&engine, res_config) {
                        Ok(mut discoverer) => {
                            match discoverer.discover(
                                &prereq_result,
                                &expansion_result,
                                &purpose_model.seed_concepts,
                                &engine,
                            ) {
                                Ok(result) => {
                                    println!("\nResource discovery complete!");
                                    println!("  Resources found: {}", result.resources.len());
                                    println!("  API calls made: {}", result.api_calls_made);
                                    println!("  Concepts searched: {}", result.concepts_searched);
                                    println!(
                                        "  Provenance: {} record(s)",
                                        result.provenance_ids.len()
                                    );

                                    if !result.resources.is_empty() {
                                        println!("\n  Discovered resources:");
                                        for (i, res) in
                                            result.resources.iter().enumerate().take(20)
                                        {
                                            let oa = if res.open_access { "OA" } else { "--" };
                                            let year_str = res
                                                .year
                                                .map(|y| y.to_string())
                                                .unwrap_or_else(|| "----".to_string());
                                            println!(
                                                "    {:3}. [{:.2}] [{oa}] ({year_str}) [{}] {} — {}",
                                                i + 1,
                                                res.quality_score,
                                                res.difficulty_estimate,
                                                res.title,
                                                res.source_api,
                                            );
                                        }
                                        if result.resources.len() > 20 {
                                            println!(
                                                "    ... and {} more",
                                                result.resources.len() - 20
                                            );
                                        }
                                    }
                                }
                                Err(e) => eprintln!("Resource discovery failed: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Failed to initialize resource discoverer: {e}"),
                    }
                }
                AwakenAction::Ingest {
                    seeds,
                    purpose: purpose_stmt,
                    max_cycles,
                    saturation,
                    xval_boost,
                    no_url,
                    catalog_dir,
                } => {
                    // Resolve seed concepts.
                    let purpose_model = if let Some(ref seed_list) = seeds {
                        let seed_list: Vec<String> = seed_list
                            .iter()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        akh_medu::bootstrap::PurposeModel {
                            domain: seed_list.first().cloned().unwrap_or_default(),
                            competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                            seed_concepts: seed_list.clone(),
                            description: seed_list.join(","),
                        }
                    } else if let Some(ref stmt) = purpose_stmt {
                        match akh_medu::bootstrap::purpose::parse_purpose(stmt) {
                            Ok(intent) => intent.purpose,
                            Err(e) => {
                                eprintln!("Failed to parse purpose: {e}");
                                return Ok(());
                            }
                        }
                    } else {
                        eprintln!("Provide --seeds or --purpose for curriculum ingestion.");
                        return Ok(());
                    };

                    // Step 1: Domain expansion.
                    println!(
                        "Expanding domain from {} seed(s): {:?}",
                        purpose_model.seed_concepts.len(),
                        purpose_model.seed_concepts
                    );
                    let expand_config = akh_medu::bootstrap::ExpansionConfig::default();
                    let expansion_result =
                        match akh_medu::bootstrap::DomainExpander::new(&engine, expand_config) {
                            Ok(mut expander) => match expander.expand(&purpose_model, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Expansion: {} concepts, {} relations",
                                        result.concept_count, result.relation_count
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Expansion failed: {e}");
                                    return Ok(());
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to initialize expander: {e}");
                                return Ok(());
                            }
                        };

                    // Step 2: Prerequisite analysis.
                    println!("\nAnalyzing prerequisites...");
                    let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
                    let prereq_result =
                        match akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config)
                        {
                            Ok(analyzer) => match analyzer.analyze(&expansion_result, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Prerequisites: {} edges, {} concepts",
                                        result.edge_count, result.concepts_analyzed
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!(
                                        "  Prerequisite analysis failed: {e}\n  \
                                         Falling back to synthetic curriculum"
                                    );
                                    akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                        &expansion_result,
                                        &engine,
                                    )
                                }
                            },
                            Err(e) => {
                                eprintln!(
                                    "  Failed to initialize analyzer: {e}\n  \
                                     Falling back to synthetic curriculum"
                                );
                                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                    &expansion_result,
                                    &engine,
                                )
                            }
                        };

                    // Step 3: Resource discovery.
                    println!("\nDiscovering resources for proximal concepts...");
                    let res_config = akh_medu::bootstrap::ResourceDiscoveryConfig::default();

                    let resource_result =
                        match akh_medu::bootstrap::ResourceDiscoverer::new(&engine, res_config) {
                            Ok(mut discoverer) => {
                                match discoverer.discover(
                                    &prereq_result,
                                    &expansion_result,
                                    &purpose_model.seed_concepts,
                                    &engine,
                                ) {
                                    Ok(result) => {
                                        println!(
                                            "  Resources found: {}, API calls: {}",
                                            result.resources.len(),
                                            result.api_calls_made
                                        );
                                        result
                                    }
                                    Err(e) => {
                                        eprintln!("Resource discovery failed: {e}");
                                        return Ok(());
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to initialize resource discoverer: {e}");
                                return Ok(());
                            }
                        };

                    // Step 4: Curriculum ingestion.
                    println!("\nIngesting resources in curriculum order...");
                    let ingest_config = akh_medu::bootstrap::IngestionConfig {
                        max_cycles,
                        saturation_threshold: saturation,
                        cross_validation_boost: xval_boost,
                        try_url_ingestion: !no_url,
                        catalog_dir: catalog_dir.map(std::path::PathBuf::from),
                        ..Default::default()
                    };

                    match akh_medu::bootstrap::CurriculumIngestor::new(&engine, ingest_config) {
                        Ok(mut ingestor) => {
                            match ingestor.ingest(&prereq_result, &resource_result, &engine) {
                                Ok(result) => {
                                    println!("\nCurriculum ingestion complete!");
                                    println!("  Cycles: {}", result.cycles);
                                    println!("  Triples created: {}", result.total_triples);
                                    println!(
                                        "  Concepts extracted: {}",
                                        result.total_concepts_extracted
                                    );
                                    println!(
                                        "  Concepts ingested: {}",
                                        result.concepts_ingested
                                    );
                                    println!(
                                        "  Concepts saturated: {}",
                                        result.concepts_saturated
                                    );
                                    println!(
                                        "  URL attempts/successes: {}/{}",
                                        result.url_attempts, result.url_successes
                                    );
                                    println!(
                                        "  Cross-validated concepts: {}",
                                        result.cross_validated_concepts
                                    );
                                    println!(
                                        "  Symbols grounded: {}",
                                        result.symbols_grounded
                                    );
                                    println!(
                                        "  Provenance: {} record(s)",
                                        result.provenance_ids.len()
                                    );
                                }
                                Err(e) => eprintln!("Curriculum ingestion failed: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Failed to initialize curriculum ingestor: {e}"),
                    }
                }
                AwakenAction::Assess {
                    seeds,
                    purpose: purpose_stmt,
                    min_triples,
                    bloom_depth,
                    verbose,
                } => {
                    // Resolve seed concepts (same pattern as Ingest).
                    let purpose_model = if let Some(ref seed_list) = seeds {
                        let seed_list: Vec<String> = seed_list
                            .iter()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                        akh_medu::bootstrap::PurposeModel {
                            domain: seed_list.first().cloned().unwrap_or_default(),
                            competence_level: akh_medu::bootstrap::DreyfusLevel::Competent,
                            seed_concepts: seed_list.clone(),
                            description: seed_list.join(","),
                        }
                    } else if let Some(ref stmt) = purpose_stmt {
                        match akh_medu::bootstrap::purpose::parse_purpose(stmt) {
                            Ok(intent) => intent.purpose,
                            Err(e) => {
                                eprintln!("Failed to parse purpose: {e}");
                                return Ok(());
                            }
                        }
                    } else {
                        eprintln!("Provide --seeds or --purpose for competence assessment.");
                        return Ok(());
                    };

                    // Step 1: Domain expansion.
                    println!(
                        "Expanding domain from {} seed(s): {:?}",
                        purpose_model.seed_concepts.len(),
                        purpose_model.seed_concepts
                    );
                    let expand_config = akh_medu::bootstrap::ExpansionConfig::default();
                    let expansion_result =
                        match akh_medu::bootstrap::DomainExpander::new(&engine, expand_config) {
                            Ok(mut expander) => match expander.expand(&purpose_model, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Expansion: {} concepts, {} relations",
                                        result.concept_count, result.relation_count
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!("Expansion failed: {e}");
                                    return Ok(());
                                }
                            },
                            Err(e) => {
                                eprintln!("Failed to initialize expander: {e}");
                                return Ok(());
                            }
                        };

                    // Step 2: Prerequisite analysis.
                    println!("\nAnalyzing prerequisites...");
                    let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
                    let prereq_result =
                        match akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config)
                        {
                            Ok(analyzer) => match analyzer.analyze(&expansion_result, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Prerequisites: {} edges, {} concepts",
                                        result.edge_count, result.concepts_analyzed
                                    );
                                    result
                                }
                                Err(e) => {
                                    eprintln!(
                                        "  Prerequisite analysis failed: {e}\n  \
                                         Falling back to synthetic curriculum"
                                    );
                                    akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                        &expansion_result,
                                        &engine,
                                    )
                                }
                            },
                            Err(e) => {
                                eprintln!(
                                    "  Failed to initialize analyzer: {e}\n  \
                                     Falling back to synthetic curriculum"
                                );
                                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                                    &expansion_result,
                                    &engine,
                                )
                            }
                        };

                    // Step 3: Resource discovery.
                    println!("\nDiscovering resources for proximal concepts...");
                    let res_config = akh_medu::bootstrap::ResourceDiscoveryConfig::default();

                    let resource_result =
                        match akh_medu::bootstrap::ResourceDiscoverer::new(&engine, res_config) {
                            Ok(mut discoverer) => {
                                match discoverer.discover(
                                    &prereq_result,
                                    &expansion_result,
                                    &purpose_model.seed_concepts,
                                    &engine,
                                ) {
                                    Ok(result) => {
                                        println!(
                                            "  Resources found: {}, API calls: {}",
                                            result.resources.len(),
                                            result.api_calls_made
                                        );
                                        result
                                    }
                                    Err(e) => {
                                        eprintln!("Resource discovery failed: {e}");
                                        return Ok(());
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to initialize resource discoverer: {e}");
                                return Ok(());
                            }
                        };

                    // Step 4: Curriculum ingestion.
                    println!("\nIngesting resources in curriculum order...");
                    let ingest_config = akh_medu::bootstrap::IngestionConfig::default();

                    match akh_medu::bootstrap::CurriculumIngestor::new(&engine, ingest_config) {
                        Ok(mut ingestor) => {
                            match ingestor.ingest(&prereq_result, &resource_result, &engine) {
                                Ok(result) => {
                                    println!(
                                        "  Ingestion: {} triples, {} concepts",
                                        result.total_triples, result.concepts_ingested
                                    );
                                }
                                Err(e) => {
                                    eprintln!("Curriculum ingestion failed: {e}");
                                    // Continue to assessment — may still have partial data.
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to initialize curriculum ingestor: {e}");
                            // Continue to assessment — may still have prior data.
                        }
                    }

                    // Step 5: Competence assessment.
                    println!("\nAssessing competence...");
                    let assess_config = akh_medu::bootstrap::CompetenceConfig {
                        min_triples_per_concept: min_triples,
                        bloom_max_depth: bloom_depth,
                        ..Default::default()
                    };

                    match akh_medu::bootstrap::CompetenceAssessor::new(&engine, assess_config) {
                        Ok(assessor) => {
                            match assessor.assess(&prereq_result, &purpose_model, &engine) {
                                Ok(report) => {
                                    println!("\nCompetence Assessment Report");
                                    println!("============================");
                                    println!(
                                        "  Overall Dreyfus level: {}",
                                        report.overall_dreyfus
                                    );
                                    println!(
                                        "  Overall score:         {:.2}",
                                        report.overall_score
                                    );
                                    println!(
                                        "  Knowledge areas:       {}",
                                        report.knowledge_areas.len()
                                    );
                                    println!(
                                        "  Recommendation:        {}",
                                        report.recommendation
                                    );

                                    if !report.remaining_gaps.is_empty() {
                                        println!(
                                            "\n  Remaining gaps ({}):",
                                            report.remaining_gaps.len()
                                        );
                                        for gap in &report.remaining_gaps {
                                            println!("    - {gap}");
                                        }
                                    }

                                    if verbose {
                                        println!("\n  Per-area breakdown:");
                                        for ka in &report.knowledge_areas {
                                            println!(
                                                "\n    {} ({})",
                                                ka.name, ka.dreyfus_level
                                            );
                                            println!(
                                                "      Score: {:.2} | Triples: {} | CQ: {}/{} | Gaps: {} | Density: {:.1}",
                                                ka.score,
                                                ka.triple_count,
                                                ka.cq_answered,
                                                ka.cq_total,
                                                ka.gap_count,
                                                ka.relation_density,
                                            );
                                            println!(
                                                "      Components: coverage={:.2} connectivity={:.2} type_diversity={:.2} relation_density={:.2} cross_domain={:.2}",
                                                ka.score_components.coverage,
                                                ka.score_components.connectivity,
                                                ka.score_components.type_diversity,
                                                ka.score_components.relation_density,
                                                ka.score_components.cross_domain,
                                            );
                                        }
                                    }

                                    println!(
                                        "\n  Provenance: {} record(s)",
                                        report.provenance_ids.len()
                                    );
                                }
                                Err(e) => eprintln!("Competence assessment failed: {e}"),
                            }
                        }
                        Err(e) => eprintln!("Failed to initialize competence assessor: {e}"),
                    }
                }
                AwakenAction::Bootstrap {
                    statement,
                    plan_only,
                    resume,
                    status,
                    max_cycles,
                    identity: _identity_override,
                } => {
                    if status {
                        // Show session status.
                        match akh_medu::bootstrap::BootstrapOrchestrator::status(&engine) {
                            Ok(session) => {
                                println!("Bootstrap Session Status");
                                println!("========================");
                                println!("  Stage:          {}", session.current_stage);
                                println!("  Learning cycle: {}", session.learning_cycle);
                                println!("  Purpose:        {}", session.raw_purpose);
                                if let Some(ref name) = session.chosen_name {
                                    println!("  Chosen name:    {name}");
                                }
                                println!("  Exploration:    {:.2}", session.exploration_rate);
                                if let Some(ref a) = session.last_assessment {
                                    println!("  Last score:     {:.2} ({})", a.overall_score, a.overall_dreyfus);
                                    if !a.focus_areas.is_empty() {
                                        println!("  Focus areas:    {}", a.focus_areas.join(", "));
                                    }
                                }
                            }
                            Err(e) => eprintln!("No bootstrap session: {e}"),
                        }
                    } else if resume {
                        // Resume interrupted session.
                        let config = akh_medu::bootstrap::OrchestratorConfig {
                            max_learning_cycles: max_cycles,
                            plan_only,
                            ..Default::default()
                        };
                        match akh_medu::bootstrap::BootstrapOrchestrator::resume(&engine, config) {
                            Ok(mut orchestrator) => {
                                println!("Resuming bootstrap session...");
                                match orchestrator.run(&engine) {
                                    Ok((result, checkpoints)) => {
                                        print_bootstrap_checkpoints(&checkpoints);
                                        print_bootstrap_result(&result);
                                    }
                                    Err(e) => eprintln!("Bootstrap failed: {e}"),
                                }
                            }
                            Err(e) => eprintln!("Cannot resume: {e}"),
                        }
                    } else if let Some(ref stmt) = statement {
                        // Fresh bootstrap from purpose statement.
                        let config = akh_medu::bootstrap::OrchestratorConfig {
                            max_learning_cycles: max_cycles,
                            plan_only,
                            ..Default::default()
                        };
                        match akh_medu::bootstrap::BootstrapOrchestrator::new(stmt, config) {
                            Ok(mut orchestrator) => {
                                println!("Starting bootstrap pipeline...");
                                match orchestrator.run(&engine) {
                                    Ok((result, checkpoints)) => {
                                        print_bootstrap_checkpoints(&checkpoints);
                                        print_bootstrap_result(&result);
                                    }
                                    Err(e) => eprintln!("Bootstrap failed: {e}"),
                                }
                            }
                            Err(e) => eprintln!("Bootstrap init failed: {e}"),
                        }
                    } else {
                        eprintln!(
                            "Provide a purpose statement, --resume, or --status.\n\
                             Example: akh awaken bootstrap \"You are the Architect based on Ptah, expert in systems\""
                        );
                    }
                }
            }

            agent.persist_session().into_diagnostic()?;
        }

        Commands::Chat { skill, headless, fresh } => {
            let ws_name = cli.workspace.clone();

            // Try remote TUI via akhomed if available (non-headless only).
            #[cfg(feature = "daemon")]
            if !headless {
                if let Some(ref paths) = xdg_paths
                    && let Some(server_info) = akh_medu::client::discover_server(paths) {
                        eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                        return akh_medu::tui::launch_remote(&ws_name, &server_info);
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
                    && result.symbols_updated > 0 {
                        println!(
                            "Grounded {} symbols in {} round(s).",
                            result.symbols_updated, result.rounds_completed,
                        );
                    }
            }

            let agent_config = AgentConfig {
                max_cycles: 20,
                ..Default::default()
            };

            if !headless {
                // TUI mode (local fallback).
                akh_medu::tui::launch(&ws_name, engine, agent_config, fresh)?;
            } else {
                // Headless mode: use ChatProcessor for unified input processing.
                let mut agent = if !fresh && Agent::has_persisted_session(&engine) {
                    Agent::resume(Arc::clone(&engine), agent_config).into_diagnostic()?
                } else {
                    Agent::new(Arc::clone(&engine), agent_config).into_diagnostic()?
                };
                if fresh {
                    agent.clear_goals();
                }

                // Create ChatProcessor with NLU pipeline.
                let data_dir = engine.config().data_dir.as_deref();
                let nlu_pipeline = engine
                    .store()
                    .get_meta(b"nlu_ranker_state")
                    .ok()
                    .flatten()
                    .and_then(|bytes| akh_medu::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
                    .map(|ranker| akh_medu::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir))
                    .unwrap_or_else(|| akh_medu::nlu::NluPipeline::new_with_models(data_dir));
                let mut chat_processor = akh_medu::chat::ChatProcessor::new(&engine, nlu_pipeline);

                println!("akh chat (headless). Type 'quit' to exit.\n");
                use std::io::Write as _;

                loop {
                    print!("> ");
                    std::io::stdout().flush().ok();
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

                    let responses = chat_processor.process_input(trimmed, &mut agent, &engine);
                    for msg in &responses {
                        println!("{}", msg.to_plain_text());
                    }
                    println!();
                }

                chat_processor.persist_nlu_state(&engine);
                agent.persist_session().into_diagnostic()?;
                println!("Session saved.");
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
                            "{:<30} {:<20} {:<8} {:<8} Tags",
                            "ID", "Title", "Format", "Chunks"
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
                        println!("{:<8} {:<10} Symbol", "Rank", "Sim");
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

        // ── Service (launchd) ────────────────────────────────────────────
        Commands::Service { action } => {
            dispatch_service(action)?;
        }
    }

    Ok(())
    } // #[cfg(not(feature = "client-only"))]
}

/// Dispatch `akh service` subcommands. Works without an Engine.
fn dispatch_service(action: ServiceAction) -> Result<()> {
    use akh_medu::service;

    match action {
        ServiceAction::Install { port } => {
            let config = service::default_config(port).into_diagnostic()?;
            service::install(&config).into_diagnostic()?;
            println!("Service installed: {}", config.label);
            println!("  Plist: {}", config.plist_path.display());
            println!("  Logs:  {}", config.log_dir.display());
            println!("\nThe service will start automatically. To start now:");
            println!("  akh service start");
        }
        ServiceAction::Start => {
            let config = service::default_config(None).into_diagnostic()?;
            service::start(&config).into_diagnostic()?;
            println!("Service started: {}", config.label);
        }
        ServiceAction::Stop => {
            let config = service::default_config(None).into_diagnostic()?;
            service::stop(&config).into_diagnostic()?;
            println!("Service stopped: {}", config.label);
        }
        ServiceAction::Status => {
            let config = service::default_config(None).into_diagnostic()?;
            let st = service::status(&config).into_diagnostic()?;
            println!("Service: {}", config.label);
            println!("  Loaded:      {}", st.loaded);
            println!("  Running:     {}", st.running);
            if let Some(pid) = st.pid {
                println!("  PID:         {pid}");
            }
            if let Some(exit) = st.last_exit_status {
                println!("  Last exit:   {exit}");
            }
            println!("  Plist:       {}", config.plist_path.display());
        }
        ServiceAction::Uninstall => {
            let config = service::default_config(None).into_diagnostic()?;
            service::uninstall(&config).into_diagnostic()?;
            println!("Service uninstalled: {}", config.label);
        }
        ServiceAction::Show { port } => {
            let config = service::default_config(port).into_diagnostic()?;
            let plist = service::generate_plist(&config);
            println!("{plist}");
        }
    }

    Ok(())
}

/// Pretty-print daemon status with all monitoring fields.
fn print_daemon_status(status: &akh_medu::client::DaemonStatus) {
    println!("Daemon status:");
    println!("  Running:        {}", status.running);
    println!("  Cycles:         {}", status.total_cycles);
    println!("  Started at:     {}", format_timestamp(status.started_at));
    println!("  Triggers:       {}", status.trigger_count);
    println!("  Active goals:   {}", status.active_goals);
    println!("  KG symbols:     {}", status.kg_symbols);
    println!("  KG triples:     {}", status.kg_triples);
    if let Some(ts) = status.last_persist_at {
        println!("  Last persist:   {}", format_timestamp(ts));
    }
    if let Some(ts) = status.last_learning_at {
        println!("  Last learning:  {}", format_timestamp(ts));
    }
    if let Some(ts) = status.last_sleep_at {
        println!("  Last sleep:     {}", format_timestamp(ts));
    }
    if let Some(ts) = status.last_goal_gen_at {
        println!("  Last goal gen:  {}", format_timestamp(ts));
    }
}

/// Format a unix timestamp as a human-readable relative or absolute string.
fn format_timestamp(ts: u64) -> String {
    if ts == 0 {
        return "never".into();
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if now >= ts {
        let ago = now - ts;
        if ago < 60 {
            format!("{ago}s ago")
        } else if ago < 3600 {
            format!("{}m ago", ago / 60)
        } else if ago < 86400 {
            format!("{}h {}m ago", ago / 3600, (ago % 3600) / 60)
        } else {
            format!("{}d {}h ago", ago / 86400, (ago % 86400) / 3600)
        }
    } else {
        format!("{ts} (unix)")
    }
}

/// Route all commands through a running akhomed server (client-only mode).
///
/// This function mirrors the main dispatch but uses [`AkhClient`] remote
/// methods instead of local Engine/Agent instances.
#[cfg(feature = "client-only")]
fn run_client_only(cli: Cli) -> Result<()> {
    use akh_medu::api_types;
    use std::path::Path;

    // Service commands work without a running server.
    if let Commands::Service { action } = cli.command {
        return dispatch_service(action);
    }

    let xdg_paths = akh_medu::paths::AkhPaths::resolve().ok();
    let client = resolve_client(&cli.workspace, xdg_paths.as_ref())?;

    match cli.command {
        // ── Init ──────────────────────────────────────────────────────────
        Commands::Init => {
            client
                .workspace_create(&cli.workspace, None)
                .into_diagnostic()?;
            println!(
                "Initialized workspace \"{}\" (via server).",
                cli.workspace
            );
        }

        // ── Workspace ─────────────────────────────────────────────────────
        Commands::Workspace { action } => match action {
            WorkspaceAction::List => {
                let names = client.workspace_list().into_diagnostic()?;
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
                client
                    .workspace_create(&name, role.as_deref())
                    .into_diagnostic()?;
                println!("Created workspace \"{name}\" (via server).");
                if let Some(ref role_name) = role {
                    println!("Assigned role \"{role_name}\" to workspace \"{name}\".");
                }
            }
            WorkspaceAction::Delete { name } => {
                client.workspace_delete(&name).into_diagnostic()?;
                println!("Deleted workspace \"{name}\".");
            }
            WorkspaceAction::Info { name: _ } => {
                let info = client.info().into_diagnostic()?;
                println!("{info}");
            }
            WorkspaceAction::AssignRole { name, role } => {
                let ws_client = resolve_client(&name, xdg_paths.as_ref())?;
                ws_client.assign_role(&role).into_diagnostic()?;
                println!("Assigned role \"{role}\" to workspace \"{name}\".");
            }
        },

        // ── Seeds ─────────────────────────────────────────────────────────
        Commands::Seed { action } => match action {
            SeedAction::List => {
                let packs = client.seed_list().into_diagnostic()?;
                if packs.is_empty() {
                    println!("No seed packs available.");
                } else {
                    println!("Available seed packs:");
                    for p in &packs {
                        println!(
                            "  {} ({}) — {} [{} triples]",
                            p.id, p.source, p.description, p.triple_count,
                        );
                    }
                }
            }
            SeedAction::Apply { pack } => {
                let report: serde_json::Value =
                    client.seed_apply(&pack).into_diagnostic()?;
                println!("Applied seed \"{pack}\": {report}");
            }
            SeedAction::Status => {
                let status = client.seed_status().into_diagnostic()?;
                println!("Seed status for workspace \"{}\":", status.workspace);
                for entry in &status.seeds {
                    let s = if entry.applied {
                        "applied"
                    } else {
                        "not applied"
                    };
                    println!("  {} — {s}", entry.id);
                }
            }
        },

        // ── Ingest ────────────────────────────────────────────────────────
        Commands::Ingest {
            file,
            format,
            csv_format,
            max_sentences,
        } => match format {
            IngestFormat::Json => {
                let content = std::fs::read_to_string(&file).into_diagnostic()?;
                let triples: Vec<serde_json::Value> =
                    serde_json::from_str(&content).into_diagnostic()?;

                if triples.is_empty() {
                    println!("No triples found in {}", file.display());
                    return Ok(());
                }

                let first = &triples[0];
                if first.get("subject").is_some() {
                    // Label-based format.
                    let mut label_triples = Vec::new();
                    for (i, val) in triples.iter().enumerate() {
                        let subject = val["subject"].as_str().ok_or_else(|| {
                            miette::miette!("triple {i}: missing or non-string 'subject'")
                        })?;
                        let predicate = val["predicate"].as_str().ok_or_else(|| {
                            miette::miette!("triple {i}: missing or non-string 'predicate'")
                        })?;
                        let object = val["object"].as_str().ok_or_else(|| {
                            miette::miette!("triple {i}: missing or non-string 'object'")
                        })?;
                        let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;
                        label_triples.push((
                            subject.to_string(),
                            predicate.to_string(),
                            object.to_string(),
                            confidence,
                        ));
                    }
                    let (created, ingested) = client
                        .ingest_label_triples(&label_triples)
                        .into_diagnostic()?;
                    println!(
                        "Ingested {ingested} triples ({created} new symbols) from {}",
                        file.display()
                    );
                } else {
                    miette::bail!(
                        "numeric-format JSON ingest is not supported in client-only mode.\n\
                         Use label-based format: {{\"subject\": ..., \"predicate\": ..., \"object\": ...}}"
                    );
                }
            }
            IngestFormat::Csv => {
                let content = std::fs::read_to_string(&file).into_diagnostic()?;
                let csv_format_str = match csv_format {
                    CsvFormat::Spo => "spo",
                    CsvFormat::Entity => "entity",
                };
                let req = akh_medu::api_types::CsvIngestRequest {
                    content,
                    format: csv_format_str.into(),
                };
                let resp = client.ingest_csv(&req).into_diagnostic()?;
                println!("{}", resp.message);
            }
            IngestFormat::Text => {
                let content = std::fs::read_to_string(&file).into_diagnostic()?;
                let req = akh_medu::api_types::TextIngestRequest {
                    text: content,
                    max_sentences,
                };
                let resp = client.ingest_text(&req).into_diagnostic()?;
                println!("{}", resp.message);
            }
        },

        // ── Skills ────────────────────────────────────────────────────────
        Commands::Skill { action } => match action {
            SkillAction::Scaffold { name } => {
                // Scaffold writes local template files — no server needed.
                let skill_base = cli
                    .data_dir
                    .as_deref()
                    .unwrap_or_else(|| Path::new(".akh-medu"));
                let skill_dir = skill_base.join("skills").join(&name);
                std::fs::create_dir_all(&skill_dir).into_diagnostic()?;

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

                let triples = serde_json::json!([{
                    "subject": "ExampleEntity",
                    "predicate": "is-a",
                    "object": "Category",
                    "confidence": 1.0
                }]);
                std::fs::write(
                    skill_dir.join("triples.json"),
                    serde_json::to_string_pretty(&triples).into_diagnostic()?,
                )
                .into_diagnostic()?;

                std::fs::write(
                    skill_dir.join("rules.txt"),
                    "# Rewrite rules for this skillpack.\n\
                     # Format: <lhs-pattern> => <rhs-pattern>\n",
                )
                .into_diagnostic()?;

                println!("Scaffolded skill '{}' at {}", name, skill_dir.display());
            }
            SkillAction::List => {
                let skills = client.list_skills().into_diagnostic()?;
                if skills.is_empty() {
                    println!("No skillpacks discovered.");
                } else {
                    println!("Skillpacks ({}):", skills.len());
                    for s in &skills {
                        println!(
                            "  {} ({}) [{}] - {}",
                            s.id, s.version, s.state, s.description
                        );
                    }
                }
            }
            SkillAction::Load { name } => {
                let activation = client.load_skill(&name).into_diagnostic()?;
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
                let info = client.skill_info(&name).into_diagnostic()?;
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
                let manifest_content =
                    std::fs::read_to_string(skill_path.join("skill.json")).into_diagnostic()?;
                let manifest: akh_medu::skills::SkillManifest =
                    serde_json::from_str(&manifest_content).into_diagnostic()?;

                let triples_path = skill_path.join("triples.json");
                let triples: Vec<akh_medu::skills::LabelTriple> = if triples_path.exists() {
                    let content = std::fs::read_to_string(&triples_path).into_diagnostic()?;
                    serde_json::from_str(&content).into_diagnostic()?
                } else {
                    vec![]
                };

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
                let activation = client.install_skill(&payload).into_diagnostic()?;
                println!("Installed skill: {}", activation.skill_id);
                println!("  triples: {}", activation.triples_loaded);
                println!("  rules:   {}", activation.rules_loaded);
                println!("  memory:  {} bytes", activation.memory_bytes);
            }
        },

        // ── Render ────────────────────────────────────────────────────────
        Commands::Render {
            entity,
            depth,
            all,
            legend,
            no_color: _,
        } => {
            let req = api_types::RenderRequest {
                entity,
                depth,
                all,
                legend,
            };
            let resp = client.render(&req).into_diagnostic()?;
            println!("{}", resp.output);
        }

        // ── Agent ─────────────────────────────────────────────────────────
        Commands::Agent { action } => match action {
            AgentAction::Run {
                goals,
                max_cycles,
                fresh,
            } => {
                let req = api_types::AgentRunRequest {
                    goals,
                    max_cycles,
                    fresh,
                };
                let resp = client.agent_run(&req).into_diagnostic()?;
                println!(
                    "Agent completed: {} cycles, {} goals",
                    resp.cycles_completed,
                    resp.goals.len(),
                );
                println!("\nGoals:");
                for g in &resp.goals {
                    println!("  [{}] {}: {}", g.status, g.label, g.description);
                }
                if !resp.overview.is_empty() {
                    println!("\n{}", resp.overview);
                }
            }
            AgentAction::Resume { max_cycles } => {
                let req = api_types::AgentResumeRequest { max_cycles };
                let resp = client.agent_resume(&req).into_diagnostic()?;
                println!(
                    "Agent completed: {} cycles, {} goals",
                    resp.cycles_completed,
                    resp.goals.len(),
                );
                for g in &resp.goals {
                    println!("  [{}] {}: {}", g.status, g.label, g.description);
                }
            }
            AgentAction::Repl {
                goals: _,
                headless,
            } => {
                if headless {
                    miette::bail!(
                        "Headless REPL is not available in client-only mode.\n\
                         Use TUI mode (without --headless) to connect to akhomed."
                    );
                }
                #[cfg(feature = "daemon")]
                {
                    let paths = xdg_paths.ok_or_else(|| {
                        miette::miette!("Cannot resolve XDG paths. Set HOME environment variable.")
                    })?;
                    let server_info = akh_medu::client::discover_server(&paths).ok_or_else(|| {
                        miette::miette!("No running akhomed server found.")
                    })?;
                    eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                    return akh_medu::tui::launch_remote(&cli.workspace, &server_info);
                }
                #[cfg(not(feature = "daemon"))]
                miette::bail!(
                    "TUI requires --features daemon. Build with: cargo build --features client-only,daemon"
                );
            }
            AgentAction::Chat {
                max_cycles: _,
                fresh: _,
                headless,
            } => {
                eprintln!("warning: `akh agent chat` is deprecated. Use `akh chat` instead.");
                if headless {
                    miette::bail!(
                        "Headless chat is not available in client-only mode.\n\
                         Use TUI mode (without --headless) to connect to akhomed."
                    );
                }
                #[cfg(feature = "daemon")]
                {
                    let paths = xdg_paths.ok_or_else(|| {
                        miette::miette!("Cannot resolve XDG paths. Set HOME environment variable.")
                    })?;
                    let server_info = akh_medu::client::discover_server(&paths).ok_or_else(|| {
                        miette::miette!("No running akhomed server found.")
                    })?;
                    eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                    return akh_medu::tui::launch_remote(&cli.workspace, &server_info);
                }
                #[cfg(not(feature = "daemon"))]
                miette::bail!(
                    "TUI requires --features daemon. Build with: cargo build --features client-only,daemon"
                );
            }
            #[cfg(feature = "daemon")]
            AgentAction::Daemon {
                max_cycles,
                fresh: _,
                equiv_interval: _,
                reflect_interval: _,
                rules_interval: _,
                persist_interval: _,
            } => {
                let config = serde_json::json!({ "max_cycles": max_cycles });
                let status = client
                    .start_daemon(Some(config))
                    .into_diagnostic()?;
                println!(
                    "Daemon started via akhomed (cycles: {}, triggers: {})",
                    status.total_cycles, status.trigger_count
                );
            }
            AgentAction::DaemonStop => {
                client.stop_daemon().into_diagnostic()?;
                println!("Daemon stopped.");
            }
            AgentAction::DaemonStatus => {
                let status = client.daemon_status().into_diagnostic()?;
                print_daemon_status(&status);
            }
        },

        // ── PIM ───────────────────────────────────────────────────────────
        Commands::Pim { action } => match action {
            PimAction::Inbox => {
                let resp = client.pim_inbox().into_diagnostic()?;
                if resp.tasks.is_empty() {
                    println!("Inbox is empty.");
                } else {
                    println!("Inbox ({} items):", resp.tasks.len());
                    for t in &resp.tasks {
                        println!("  [{:>5}] {} ({})", t.symbol_id, t.label, t.quadrant);
                    }
                }
            }
            PimAction::Next { context, energy } => {
                let req = api_types::PimNextRequest {
                    context,
                    energy: energy.map(|e| match e {
                        EnergyLevel::Low => "low".into(),
                        EnergyLevel::Medium => "medium".into(),
                        EnergyLevel::High => "high".into(),
                    }),
                };
                let resp = client.pim_next(&req).into_diagnostic()?;
                if resp.tasks.is_empty() {
                    println!("No next actions available.");
                } else {
                    println!("Next actions ({} tasks):", resp.tasks.len());
                    for t in &resp.tasks {
                        let energy_str = t
                            .energy
                            .as_deref()
                            .map(|e| format!(" [{e}]"))
                            .unwrap_or_default();
                        println!(
                            "  [{:>5}] {} ({}){}",
                            t.symbol_id, t.label, t.quadrant, energy_str
                        );
                    }
                }
            }
            PimAction::Review => {
                let resp = client.pim_review().into_diagnostic()?;
                println!("{}", resp.summary);
                if !resp.overdue.is_empty() {
                    println!("\nOverdue:");
                    for t in &resp.overdue {
                        println!("  [{:>5}] {}", t.symbol_id, t.label);
                    }
                }
                if !resp.stale_inbox.is_empty() {
                    println!("\nStale inbox (>7 days):");
                    for t in &resp.stale_inbox {
                        println!("  [{:>5}] {}", t.symbol_id, t.label);
                    }
                }
                if !resp.stalled_projects.is_empty() {
                    println!("\nStalled projects (no Next actions):");
                    for t in &resp.stalled_projects {
                        println!("  [{:>5}] {}", t.symbol_id, t.label);
                    }
                }
                if resp.adjustment_count > 0 {
                    println!("\nRecommended adjustments: {}", resp.adjustment_count);
                }
            }
            PimAction::Project { name } => {
                let resp = client.pim_project(&name).into_diagnostic()?;
                println!("Project: {} ({})", resp.name, resp.status);
                for t in &resp.goals {
                    let gtd = t.gtd_state.as_deref().unwrap_or("no-pim");
                    println!("  [{:>5}] {} (GTD: {})", t.symbol_id, t.label, gtd);
                }
            }
            PimAction::Add {
                goal,
                gtd,
                urgency,
                importance,
                para,
                contexts,
                recur,
                deadline,
            } => {
                let gtd_str = match gtd {
                    GtdState::Inbox => "inbox",
                    GtdState::Next => "next",
                    GtdState::Waiting => "waiting",
                    GtdState::Someday => "someday",
                    GtdState::Reference => "reference",
                };
                let para_str = para.map(|p| match p {
                    ParaCategory::Project => "project".to_string(),
                    ParaCategory::Area => "area".to_string(),
                    ParaCategory::Resource => "resource".to_string(),
                    ParaCategory::Archive => "archive".to_string(),
                });
                let req = api_types::PimAddRequest {
                    goal,
                    gtd: gtd_str.to_string(),
                    urgency,
                    importance,
                    para: para_str,
                    contexts,
                    recur,
                    deadline,
                };
                let resp = client.pim_add(&req).into_diagnostic()?;
                println!(
                    "Added PIM metadata to goal {} (GTD: {}, quadrant: {})",
                    resp.goal, resp.gtd_state, resp.quadrant,
                );
            }
            PimAction::Transition { goal, to } => {
                let to_str = match to {
                    GtdState::Inbox => "inbox",
                    GtdState::Next => "next",
                    GtdState::Waiting => "waiting",
                    GtdState::Someday => "someday",
                    GtdState::Reference => "reference",
                };
                let req = api_types::PimTransitionRequest {
                    goal,
                    to: to_str.to_string(),
                };
                client.pim_transition(&req).into_diagnostic()?;
                println!("Transitioned goal {} to GTD state: {}", goal, to_str);
            }
            PimAction::Matrix => {
                let resp = client.pim_matrix().into_diagnostic()?;
                for (label, tasks) in [
                    ("DO", &resp.do_tasks),
                    ("SCHEDULE", &resp.schedule_tasks),
                    ("DELEGATE", &resp.delegate_tasks),
                    ("ELIMINATE", &resp.eliminate_tasks),
                ] {
                    println!("{} ({} tasks):", label, tasks.len());
                    for t in tasks {
                        let gtd = t.gtd_state.as_deref().unwrap_or("");
                        println!("  [{:>5}] {} (GTD: {})", t.symbol_id, t.label, gtd);
                    }
                }
            }
            PimAction::Deps => {
                let resp = client.pim_deps().into_diagnostic()?;
                println!("Dependency order ({} tasks):", resp.order.len());
                for (i, t) in resp.order.iter().enumerate() {
                    println!("  {}. [{:>5}] {}", i + 1, t.symbol_id, t.label);
                }
            }
            PimAction::Overdue => {
                let resp = client.pim_overdue().into_diagnostic()?;
                if resp.tasks.is_empty() {
                    println!("No overdue tasks.");
                } else {
                    println!("Overdue ({} tasks):", resp.tasks.len());
                    for t in &resp.tasks {
                        let days = t.overdue_days.unwrap_or(0);
                        println!(
                            "  [{:>5}] {} (overdue by {} days)",
                            t.symbol_id, t.label, days
                        );
                    }
                }
            }
        },

        // ── Calendar ──────────────────────────────────────────────────────
        Commands::Cal { action } => match action {
            CalAction::Today => {
                let resp = client.cal_today().into_diagnostic()?;
                if resp.events.is_empty() {
                    println!("No events today.");
                } else {
                    println!("Today ({} events):", resp.events.len());
                    for e in &resp.events {
                        let loc = e
                            .location
                            .as_deref()
                            .map(|l| format!(" @ {l}"))
                            .unwrap_or_default();
                        println!(
                            "  [{:>5}] {} ({} min){}",
                            e.symbol_id, e.summary, e.duration_minutes, loc,
                        );
                    }
                }
            }
            CalAction::Week => {
                let resp = client.cal_week().into_diagnostic()?;
                if resp.events.is_empty() {
                    println!("No events this week.");
                } else {
                    println!("This week ({} events):", resp.events.len());
                    for e in &resp.events {
                        let loc = e
                            .location
                            .as_deref()
                            .map(|l| format!(" @ {l}"))
                            .unwrap_or_default();
                        println!(
                            "  [{:>5}] {} ({} min){}",
                            e.symbol_id, e.summary, e.duration_minutes, loc,
                        );
                    }
                }
            }
            CalAction::Conflicts => {
                let conflicts = client.cal_conflicts().into_diagnostic()?;
                if conflicts.is_empty() {
                    println!("No scheduling conflicts.");
                } else {
                    println!("Conflicts ({}):", conflicts.len());
                    for c in &conflicts {
                        println!("  {} <-> {}", c.event_a, c.event_b);
                    }
                }
            }
            CalAction::Add {
                summary,
                start,
                end,
                location,
            } => {
                let req = api_types::CalAddRequest {
                    summary: summary.clone(),
                    start,
                    end,
                    location,
                };
                let resp = client.cal_add(&req).into_diagnostic()?;
                println!(
                    "Added event [{:>5}] '{}' ({} min)",
                    resp.symbol_id, resp.summary, resp.duration_minutes,
                );
            }
            CalAction::Import { file } => {
                let data = std::fs::read_to_string(&file).into_diagnostic()?;
                let req = api_types::CalImportRequest { ical_data: data };
                let resp = client.cal_import(&req).into_diagnostic()?;
                println!(
                    "Imported {} events from {}",
                    resp.imported_count,
                    file.display()
                );
            }
            CalAction::Sync { url, user, pass } => {
                let req = akh_medu::api_types::CalSyncRequest { url, user, pass };
                let resp = client.cal_sync(&req).into_diagnostic()?;
                println!("CalDAV sync: {} events imported.", resp.imported_count);
            }
        },

        // ── Preferences ───────────────────────────────────────────────────
        Commands::Pref { action } => match action {
            PrefAction::Status => {
                let resp = client.pref_status().into_diagnostic()?;
                println!("Preference Profile:");
                println!("  Interactions: {}", resp.interaction_count);
                println!("  Proactivity:  {}", resp.proactivity_level);
                println!("  Decay rate:   {}", resp.decay_rate);
                println!(
                    "  Suggestions:  {} offered, {} accepted ({:.0}%)",
                    resp.suggestions_offered,
                    resp.suggestions_accepted,
                    resp.acceptance_rate * 100.0,
                );
                println!(
                    "  Prototype:    {}",
                    if resp.prototype_active {
                        "active"
                    } else {
                        "empty"
                    }
                );
            }
            PrefAction::Train { entity, weight } => {
                let req = api_types::PrefTrainRequest { entity, weight };
                let resp = client.pref_train(&req).into_diagnostic()?;
                println!(
                    "Recorded preference: '{}' (weight: {:.2}, total interactions: {})",
                    resp.entity_label, resp.weight, resp.total_interactions,
                );
            }
            PrefAction::Level { level } => {
                let lvl_str = match level {
                    ProactivityLevel::Ambient => "ambient",
                    ProactivityLevel::Nudge => "nudge",
                    ProactivityLevel::Offer => "offer",
                    ProactivityLevel::Scheduled => "scheduled",
                    ProactivityLevel::Autonomous => "autonomous",
                };
                let req = api_types::PrefLevelRequest {
                    level: lvl_str.to_string(),
                };
                client.pref_level(&req).into_diagnostic()?;
                println!("Proactivity level set to: {lvl_str}");
            }
            PrefAction::Interests { count } => {
                let interests = client.pref_interests(count).into_diagnostic()?;
                if interests.is_empty() {
                    println!(
                        "No interests recorded yet. Use `akh pref train` to add feedback."
                    );
                } else {
                    println!("Top interests ({}):", interests.len());
                    for i in &interests {
                        println!("  {:<30} (similarity: {:.3})", i.label, i.similarity);
                    }
                }
            }
            PrefAction::Suggest => {
                let resp: serde_json::Value =
                    client.pref_suggest().into_diagnostic()?;
                println!("{}", serde_json::to_string_pretty(&resp).unwrap_or_default());
            }
        },

        // ── Causal ────────────────────────────────────────────────────────
        Commands::Causal { action } => match action {
            CausalAction::Schemas => {
                let schemas = client.causal_schemas().into_diagnostic()?;
                if schemas.is_empty() {
                    println!(
                        "No action schemas registered. Use `akh causal bootstrap` to create from tools."
                    );
                } else {
                    println!("Action schemas ({}):", schemas.len());
                    for s in &schemas {
                        println!(
                            "  {:<25} precond: {} effects: {} success: {:.0}% runs: {}",
                            s.name,
                            s.precondition_count,
                            s.effect_count,
                            s.success_rate * 100.0,
                            s.execution_count,
                        );
                    }
                }
            }
            CausalAction::Schema { name } => {
                let s = client.causal_schema(&name).into_diagnostic()?;
                println!("Schema: {}", s.name);
                println!("  Action ID:       {}", s.action_id);
                println!("  Preconditions:   {}", s.precondition_count);
                println!("  Effects:         {}", s.effect_count);
                println!("  Success rate:    {:.1}%", s.success_rate * 100.0);
                println!("  Execution count: {}", s.execution_count);
            }
            CausalAction::Predict { name } => {
                let req = api_types::CausalPredictRequest { name: name.clone() };
                let resp = client.causal_predict(&req).into_diagnostic()?;
                println!("Predicted transition for '{name}':");
                if resp.assertions.is_empty()
                    && resp.retractions.is_empty()
                    && resp.confidence_changes.is_empty()
                {
                    println!("  (no predicted effects)");
                } else {
                    for t in &resp.assertions {
                        println!("  + {} {} {}", t.subject, t.predicate, t.object);
                    }
                    for t in &resp.retractions {
                        println!("  - {} {} {}", t.subject, t.predicate, t.object);
                    }
                    for c in &resp.confidence_changes {
                        println!(
                            "  ~ {} {} {} (delta: {:+.2})",
                            c.subject, c.predicate, c.object, c.delta,
                        );
                    }
                }
            }
            CausalAction::Applicable => {
                let applicable = client.causal_applicable().into_diagnostic()?;
                if applicable.is_empty() {
                    println!("No applicable actions in current state.");
                } else {
                    println!("Applicable actions ({}):", applicable.len());
                    for s in &applicable {
                        println!(
                            "  {} (success: {:.0}%, runs: {})",
                            s.name,
                            s.success_rate * 100.0,
                            s.execution_count,
                        );
                    }
                }
            }
            CausalAction::Bootstrap => {
                let resp = client.causal_bootstrap().into_diagnostic()?;
                println!(
                    "Bootstrapped {} new schema(s) from {} tool(s).",
                    resp.schemas_created, resp.tools_scanned,
                );
            }
        },

        // ── Awaken ────────────────────────────────────────────────────────
        Commands::Awaken { action } => match action {
            AwakenAction::Parse { statement } => {
                let req = api_types::AwakenParseRequest { statement };
                let resp = client.awaken_parse(&req).into_diagnostic()?;
                println!("Parsed bootstrap intent:");
                println!("  Domain:      {}", resp.domain);
                println!("  Competence:  {}", resp.competence_level);
                println!("  Seeds:       {:?}", resp.seed_concepts);
                if let Some(ref name) = resp.identity_name {
                    let id_type = resp.identity_type.as_deref().unwrap_or("unknown");
                    println!("  Identity:    {} ({})", name, id_type);
                } else {
                    println!("  Identity:    (none)");
                }
            }
            AwakenAction::Resolve { name } => {
                let req = api_types::AwakenResolveRequest { name };
                let resp = client.awaken_resolve(&req).into_diagnostic()?;
                println!("Resolved identity: {}", resp.name);
                println!("  Type:        {}", resp.entity_type);
                println!("  Culture:     {}", resp.culture);
                println!("  Description: {}", resp.description);
                println!("  Domains:     {:?}", resp.domains);
                println!("  Traits:      {:?}", resp.traits);
                println!("  Archetypes:  {:?}", resp.archetypes);
                if let Some(ref chosen) = resp.chosen_name {
                    println!("\nRitual of Awakening complete!");
                    println!("  Chosen name: {chosen}");
                }
                if let Some(ref persona) = resp.persona {
                    println!("  Persona:     {persona}");
                }
            }
            AwakenAction::Status => {
                let resp: serde_json::Value =
                    client.awaken_status().into_diagnostic()?;
                let has_psyche = resp["awakened"].as_bool().unwrap_or(false);
                let ritual_done = resp["ritual_complete"].as_bool().unwrap_or(false);
                if !has_psyche {
                    println!("No psyche loaded. Run `akh awaken resolve <name>` to awaken.");
                } else if let Some(psyche) = resp.get("psyche") {
                    if ritual_done {
                        println!("Psyche (awakened):");
                    } else {
                        println!("Psyche (loaded, ritual not yet completed):");
                    }
                    println!(
                        "{}",
                        serde_json::to_string_pretty(psyche).unwrap_or_default()
                    );
                }
            }
            AwakenAction::Expand {
                seeds,
                purpose,
                threshold,
                max_concepts,
                no_conceptnet,
            } => {
                let req = api_types::AwakenExpandRequest {
                    seeds,
                    purpose,
                    threshold,
                    max_concepts,
                    no_conceptnet,
                };
                let resp = client.awaken_expand(&req).into_diagnostic()?;
                println!("\nDomain expansion complete!");
                println!("  Concepts created: {}", resp.concept_count);
                println!("  Relations added:  {}", resp.relation_count);
                println!("  Rejected:         {}", resp.rejected_count);
                println!("  API calls:        {}", resp.api_calls);
                if !resp.accepted_labels.is_empty() {
                    println!("\n  Accepted concepts:");
                    for (i, label) in resp.accepted_labels.iter().enumerate().take(20) {
                        println!("    {}: {label}", i + 1);
                    }
                    if resp.accepted_labels.len() > 20 {
                        println!(
                            "    ... and {} more",
                            resp.accepted_labels.len() - 20
                        );
                    }
                }
            }
            AwakenAction::Prerequisite {
                seeds,
                purpose,
                known_threshold,
                zpd_low,
                zpd_high,
            } => {
                let req = api_types::AwakenPrerequisiteRequest {
                    seeds,
                    purpose,
                    known_threshold,
                    zpd_low,
                    zpd_high,
                };
                let resp = client.awaken_prerequisite(&req).into_diagnostic()?;
                println!("\nPrerequisite analysis complete!");
                println!("  Concepts analyzed: {}", resp.concepts_analyzed);
                println!("  Prerequisite edges: {}", resp.edge_count);
                println!("  Cycles broken: {}", resp.cycles_broken);
                println!("  Max tier: {}", resp.max_tier);
                println!("\n  ZPD Distribution:");
                for (zone, count) in &resp.zone_distribution {
                    println!("    {zone}: {count}");
                }
                if !resp.curriculum.is_empty() {
                    println!("\n  Curriculum (learning order):");
                    for (i, entry) in resp.curriculum.iter().enumerate().take(30) {
                        println!(
                            "    {:3}. [tier {}] ({}) {} (coverage: {:.2}, sim: {:.2})",
                            i + 1,
                            entry.tier,
                            entry.zone,
                            entry.label,
                            entry.prereq_coverage,
                            entry.similarity_to_known,
                        );
                    }
                    if resp.curriculum.len() > 30 {
                        println!(
                            "    ... and {} more",
                            resp.curriculum.len() - 30
                        );
                    }
                }
            }
            AwakenAction::Resources {
                seeds,
                purpose,
                min_quality,
                max_api_calls,
                no_semantic_scholar,
                no_openalex,
                no_open_library,
            } => {
                let req = api_types::AwakenResourcesRequest {
                    seeds,
                    purpose,
                    min_quality,
                    max_api_calls,
                    no_semantic_scholar,
                    no_openalex,
                    no_open_library,
                };
                let resp = client.awaken_resources(&req).into_diagnostic()?;
                println!("\nResource discovery complete!");
                println!("  Resources found: {}", resp.resources_discovered);
                println!("  API calls made: {}", resp.api_calls_used);
                println!("  Concepts searched: {}", resp.concepts_covered);
            }
            AwakenAction::Ingest {
                seeds,
                purpose,
                max_cycles,
                saturation,
                xval_boost,
                no_url,
                catalog_dir,
            } => {
                let req = api_types::AwakenIngestRequest {
                    seeds,
                    purpose,
                    max_cycles,
                    saturation,
                    xval_boost,
                    no_url,
                    catalog_dir,
                };
                let resp = client.awaken_ingest(&req).into_diagnostic()?;
                println!("\nCurriculum ingestion complete!");
                println!("  Triples created: {}", resp.triples_added);
                println!("  Concepts covered: {}", resp.concepts_covered);
                println!("  Cycles used: {}", resp.cycles_used);
            }
            AwakenAction::Assess {
                seeds,
                purpose,
                min_triples,
                bloom_depth,
                verbose,
            } => {
                let req = api_types::AwakenAssessRequest {
                    seeds,
                    purpose,
                    min_triples,
                    bloom_depth,
                    verbose,
                };
                let resp = client.awaken_assess(&req).into_diagnostic()?;
                println!("\nCompetence Assessment Report");
                println!("============================");
                println!("  Overall Dreyfus level: {}", resp.overall_dreyfus);
                println!("  Overall score:         {:.2}", resp.overall_score);
                println!("  Recommendation:        {}", resp.recommendation);
                if verbose && !resp.knowledge_areas.is_empty() {
                    println!("\n  Per-area breakdown:");
                    for ka in &resp.knowledge_areas {
                        println!(
                            "    {} ({}) score: {:.2}",
                            ka.name, ka.dreyfus_level, ka.score,
                        );
                    }
                }
            }
            AwakenAction::Bootstrap {
                statement,
                plan_only,
                resume,
                status,
                max_cycles,
                identity,
            } => {
                let req = api_types::AwakenBootstrapRequest {
                    statement,
                    plan_only,
                    resume,
                    status,
                    max_cycles,
                    identity,
                };
                let resp = client.awaken_bootstrap(&req).into_diagnostic()?;
                println!("\nBootstrap Result");
                println!("================");
                println!("  Domain:          {}", resp.domain);
                println!("  Target level:    {}", resp.target_level);
                if let Some(ref name) = resp.chosen_name {
                    println!("  Chosen name:     {name}");
                }
                println!("  Learning cycles: {}", resp.learning_cycles);
                println!("  Target reached:  {}", resp.target_reached);
                if let Some(ref dreyfus) = resp.final_dreyfus {
                    println!("  Final Dreyfus:   {dreyfus}");
                }
                if let Some(score) = resp.final_score {
                    println!("  Final score:     {score:.2}");
                }
                if let Some(ref rec) = resp.recommendation {
                    println!("  Recommendation:  {rec}");
                }
            }
        },

        // ── Chat ──────────────────────────────────────────────────────────
        Commands::Chat { skill: _, headless, fresh: _ } => {
            if headless {
                miette::bail!(
                    "Headless chat is not available in client-only mode.\n\
                     Use TUI mode (without --headless) to connect to akhomed."
                );
            }
            #[cfg(feature = "daemon")]
            {
                let paths = xdg_paths.ok_or_else(|| {
                    miette::miette!("Cannot resolve XDG paths. Set HOME environment variable.")
                })?;
                let server_info = akh_medu::client::discover_server(&paths).ok_or_else(|| {
                    miette::miette!("No running akhomed server found.")
                })?;
                eprintln!("Connecting to akhomed at {}...", server_info.base_url());
                return akh_medu::tui::launch_remote(&cli.workspace, &server_info);
            }
            #[cfg(not(feature = "daemon"))]
            miette::bail!(
                "TUI requires --features daemon. Build with: cargo build --features client-only,daemon"
            );
        }

        // ── Library ───────────────────────────────────────────────────────
        Commands::Library { action } => {
            // Remote client ignores library_dir — pass a dummy path.
            let library_dir = xdg_paths
                .as_ref()
                .map(|p| p.library_dir())
                .unwrap_or_default();

            match action {
                LibraryAction::Add {
                    source,
                    title,
                    tags,
                    format,
                } => {
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
                    let docs = client.library_list(&library_dir).into_diagnostic()?;
                    if docs.is_empty() {
                        println!(
                            "Library is empty. Add a document with: akh library add <file-or-url>"
                        );
                    } else {
                        println!(
                            "{:<30} {:<20} {:<8} {:<8} Tags",
                            "ID", "Title", "Format", "Chunks"
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
                    let results = client.library_search(&query, top_k).into_diagnostic()?;
                    if results.is_empty() {
                        println!("No matching content found for: \"{query}\"");
                    } else {
                        println!("Search results for \"{query}\":");
                        println!("{:<8} {:<10} Symbol", "Rank", "Sim");
                        println!("{}", "-".repeat(60));
                        for r in &results {
                            println!(
                                "{:<8} {:<10.4} {}",
                                r.rank, r.similarity, r.symbol_label,
                            );
                        }
                    }
                }
                LibraryAction::Remove { id } => {
                    let removed = client.library_remove(&library_dir, &id).into_diagnostic()?;
                    println!("Removed: {} (\"{}\")", removed.id, removed.title);
                }
                LibraryAction::Info { id } => {
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
                LibraryAction::Watch { dir } => {
                    // One-shot scan via server instead of blocking watch loop.
                    let req = akh_medu::api_types::LibraryScanRequest {
                        inbox_dir: dir.map(|d| d.to_string_lossy().into_owned()),
                    };
                    let resp = client.library_scan(&req).into_diagnostic()?;
                    println!(
                        "Library scan: {} file(s) processed, {} failed.",
                        resp.files_processed, resp.files_failed
                    );
                }
            }
        }

        // ── Service (launchd) ────────────────────────────────────────────
        Commands::Service { action } => {
            dispatch_service(action)?;
        }
    }

    Ok(())
}

/// Print agent REPL status line.
#[cfg(not(feature = "client-only"))]
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

#[cfg(not(feature = "client-only"))]
fn print_bootstrap_checkpoints(checkpoints: &[akh_medu::bootstrap::Checkpoint]) {
    use akh_medu::bootstrap::Checkpoint;
    for cp in checkpoints {
        match cp {
            Checkpoint::PurposeParsed {
                domain,
                competence_level,
                seed_count,
                has_identity,
            } => {
                println!(
                    "  [parse] Domain: {domain}, target: {competence_level}, \
                     seeds: {seed_count}, identity: {has_identity}"
                );
            }
            Checkpoint::IdentityConstructed { chosen_name } => {
                println!("  [identity] Chosen name: {chosen_name}");
            }
            Checkpoint::LearningPlan {
                concept_count,
                relation_count,
            } => {
                println!(
                    "  [plan] {concept_count} concepts, {relation_count} relations"
                );
            }
            Checkpoint::AssessmentComplete {
                cycle,
                overall_score,
                overall_dreyfus,
                recommendation,
            } => {
                println!(
                    "  [assess] Cycle {cycle}: {overall_dreyfus} (score: {overall_score:.2}) \
                     — {recommendation}"
                );
            }
        }
    }
}

#[cfg(not(feature = "client-only"))]
fn print_bootstrap_result(result: &akh_medu::bootstrap::OrchestrationResult) {
    println!("\nBootstrap Result");
    println!("================");
    println!("  Domain:          {}", result.intent.purpose.domain);
    println!(
        "  Target level:    {}",
        result.intent.purpose.competence_level
    );
    if let Some(ref name) = result.chosen_name {
        println!("  Chosen name:     {name}");
    }
    println!("  Learning cycles: {}", result.learning_cycles);
    println!("  Target reached:  {}", result.target_reached);
    if let Some(ref report) = result.final_report {
        println!("  Final Dreyfus:   {}", report.overall_dreyfus);
        println!("  Final score:     {:.2}", report.overall_score);
        println!("  Recommendation:  {}", report.recommendation);
    }
    println!(
        "  Provenance:      {} record(s)",
        result.provenance_ids.len()
    );
}
