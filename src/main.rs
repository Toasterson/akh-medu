//! akh CLI: neuro-symbolic AI engine.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::Language;
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

    /// Interactive chat with the akh over ACP (launches TUI, or --headless).
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

    /// Query a running akh's data-plane directly over ACP (`_akh/*` methods).
    Acp {
        #[command(subcommand)]
        op: AcpOp,
    },

    /// Manage a workspace's background daemon over ACP (`_akh/daemon/*`).
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },

    /// Get or set the akhomed daemon config over ACP (`_akh/config/*`).
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Manage akhomed as a macOS launchd service.
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },

    /// Download and configure NLU models and ONNX Runtime.
    Setup {
        #[command(subcommand)]
        action: SetupAction,
    },
}

#[derive(Subcommand)]
enum SetupAction {
    /// Download NLU models (NER ~130MB, LLM ~1.1GB).
    Models {
        /// Only download the NER model (skip LLM).
        #[arg(long)]
        ner_only: bool,
        /// Only download the LLM model (skip NER).
        #[arg(long)]
        llm_only: bool,
    },
    /// Download and install ONNX Runtime for the current platform.
    OnnxRuntime {
        /// Target version (default: 1.23.2).
        #[arg(long)]
        version: Option<String>,
    },
    /// Check the current NLU setup status.
    Check,
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

/// Read-only data-plane operations exposed over ACP `_akh/*` methods.
#[derive(Subcommand)]
enum AcpOp {
    /// Show the akh's status (workspace name + symbol/triple counts).
    Status,
    /// Show the data-plane capabilities the akh advertises.
    Capabilities,
    /// List symbols (capped).
    Symbols {
        /// Maximum number of symbols to return.
        #[arg(long, default_value_t = 50)]
        limit: u64,
    },
    /// List triples originating from a symbol (by label or id).
    Triples {
        /// Subject symbol — a label (e.g. "Sun") or numeric id.
        symbol: String,
    },
    /// Add a triple to the akh's knowledge graph.
    AddTriple {
        /// Subject label.
        subject: String,
        /// Predicate (relation) label.
        predicate: String,
        /// Object label.
        object: String,
        /// Confidence in [0.0, 1.0].
        #[arg(long, default_value_t = 1.0)]
        confidence: f64,
    },
    /// Create (or resolve) a symbol by label.
    CreateSymbol {
        /// Symbol label.
        label: String,
        /// Symbol kind: `entity` (default) or `relation`.
        #[arg(long, default_value = "entity")]
        kind: String,
    },
    /// Run a read-only SPARQL SELECT against the akh's knowledge graph.
    Query {
        /// SPARQL SELECT query (e.g. "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 20").
        query: String,
    },
    /// Recall the akh's episodic memories. With no `--query`, returns the most
    /// recent episodes; with `--query`, recalls episodes linked to those symbols
    /// (comma-separated), falling back to recency if none resolve.
    Recall {
        /// Comma-separated query symbols (e.g. "Sun,Star"). Omit for recent episodes.
        #[arg(long)]
        query: Option<String>,
        /// Maximum number of episodes to return.
        #[arg(long, default_value_t = 5)]
        limit: u64,
    },
    /// Run spreading-activation inference from one or more seed symbols.
    Infer {
        /// Comma-separated seed symbols (e.g. "Sun,Moon").
        seeds: String,
        /// Maximum number of activated symbols to return.
        #[arg(long, default_value_t = 10)]
        limit: u64,
        /// Maximum inference depth (multi-step reasoning).
        #[arg(long, default_value_t = 1)]
        depth: u64,
        /// Minimum activation confidence to include.
        #[arg(long, default_value_t = 0.1)]
        min_confidence: f64,
    },
    /// Show or set the akh's tool-permission mode. With no `--set`, reports the
    /// current mode.
    PermissionMode {
        /// New mode to apply: bypass | standard | accept_edits | plan.
        #[arg(long)]
        set: Option<String>,
    },
}

/// Discover the running daemon and send one `_akh/*` request over ACP, returning
/// the JSON result. Shared by the data-plane (`akh acp`) and control-plane
/// (`akh daemon`/`akh config`) command dispatchers.
#[cfg(feature = "acp")]
fn acp_call(method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
    let paths = akh_medu::paths::AkhPaths::resolve()
        .map_err(|e| miette::miette!("cannot resolve XDG paths: {e}"))?;
    let info = akh_medu::client::discover_server(&paths)
        .ok_or_else(|| miette::miette!("No running akhomed server found."))?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .into_diagnostic()?;
    if let Some(sock) = info.acp_socket.as_deref() {
        rt.block_on(akh_medu::acp::ext_request_unix(sock, method, params))
            .map_err(miette::Report::from)
    } else if let Some(tcp) = info.acp_tcp.as_deref() {
        rt.block_on(akh_medu::acp::ext_request_tcp(tcp, method, params))
            .map_err(miette::Report::from)
    } else {
        miette::bail!(
            "akhomed is running but advertises no ACP endpoint — build the daemon with --features acp."
        )
    }
}

/// Pretty-print an ACP JSON result to stdout.
#[cfg(feature = "acp")]
fn print_acp(result: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(result).into_diagnostic()?);
    Ok(())
}

/// Dispatch an `akh acp <op>` command: discover the daemon, send the matching
/// `_akh/*` request over ACP, and pretty-print the JSON result. Self-contained
/// so both the normal and client-only dispatch paths can call it.
#[cfg(feature = "acp")]
fn run_acp_command(op: &AcpOp) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match op {
        AcpOp::Status => ("_akh/workspace/status", serde_json::json!({})),
        AcpOp::Capabilities => ("_akh/meta/capabilities", serde_json::json!({})),
        AcpOp::Symbols { limit } => ("_akh/symbols/list", serde_json::json!({ "limit": limit })),
        AcpOp::Triples { symbol } => {
            ("_akh/triples/from", serde_json::json!({ "symbol": symbol }))
        }
        AcpOp::AddTriple {
            subject,
            predicate,
            object,
            confidence,
        } => (
            "_akh/triples/add",
            serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "confidence": confidence,
            }),
        ),
        AcpOp::CreateSymbol { label, kind } => (
            "_akh/symbols/create",
            serde_json::json!({ "label": label, "kind": kind }),
        ),
        AcpOp::Query { query } => {
            ("_akh/query/sparql", serde_json::json!({ "query": query }))
        }
        AcpOp::Recall { query, limit } => {
            let mut params = serde_json::json!({ "top_k": limit });
            if let Some(q) = query {
                params["query"] = serde_json::Value::String(q.clone());
            }
            ("_akh/memory/recall", params)
        }
        AcpOp::Infer {
            seeds,
            limit,
            depth,
            min_confidence,
        } => (
            "_akh/infer/activate",
            serde_json::json!({
                "seeds": seeds,
                "top_k": limit,
                "max_depth": depth,
                "min_confidence": min_confidence,
            }),
        ),
        AcpOp::PermissionMode { set } => {
            let mut params = serde_json::json!({});
            if let Some(mode) = set {
                params["mode"] = serde_json::Value::String(mode.clone());
            }
            ("_akh/permission/mode", params)
        }
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh daemon <action>` over the ACP control-plane (`_akh/daemon/*`).
/// Daemon ops span the whole daemon, so the target workspace is passed
/// explicitly (the akh selected by `-w`).
#[cfg(feature = "acp")]
fn run_daemon_acp(workspace: &str, action: DaemonAction) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match action {
        DaemonAction::Start { max_cycles } => (
            "_akh/daemon/start",
            serde_json::json!({ "workspace": workspace, "max_cycles": max_cycles }),
        ),
        DaemonAction::Stop => (
            "_akh/daemon/stop",
            serde_json::json!({ "workspace": workspace }),
        ),
        DaemonAction::Status => (
            "_akh/daemon/status",
            serde_json::json!({ "workspace": workspace }),
        ),
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh config <action>` over the ACP control-plane (`_akh/config/*`).
/// `set` reads + parses the whole `AkhomedConfig` from a TOML file locally and
/// sends it (the control-plane replaces the config wholesale).
#[cfg(feature = "acp")]
fn run_config_acp(action: ConfigAction) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match action {
        ConfigAction::Get => ("_akh/config/get", serde_json::json!({})),
        ConfigAction::Set { file } => {
            let text = std::fs::read_to_string(&file).map_err(|e| {
                miette::miette!("reading config file {}: {e}", file.display())
            })?;
            // Parse TOML into a generic JSON value — the CLI stays agnostic of
            // the config schema; akhomed validates it as an `AkhomedConfig`.
            let cfg: serde_json::Value = toml::from_str(&text).map_err(|e| {
                miette::miette!("parsing {} as TOML: {e}", file.display())
            })?;
            ("_akh/config/set", serde_json::json!({ "config": cfg }))
        }
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

#[cfg(not(feature = "acp"))]
fn run_acp_command(_op: &AcpOp) -> Result<()> {
    miette::bail!("The `acp` subcommand requires building `akh` with --features acp.")
}

#[cfg(not(feature = "acp"))]
fn run_daemon_acp(_workspace: &str, _action: DaemonAction) -> Result<()> {
    miette::bail!("The `daemon` subcommand requires building `akh` with --features acp.")
}

#[cfg(not(feature = "acp"))]
fn run_config_acp(_action: ConfigAction) -> Result<()> {
    miette::bail!("The `config` subcommand requires building `akh` with --features acp.")
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
}

/// Background-daemon lifecycle for a workspace (control-plane, over ACP
/// `_akh/daemon/*`).
#[derive(Subcommand)]
enum DaemonAction {
    /// Start (or report the already-running) background daemon.
    Start {
        /// Stop after this many OODA cycles (0 = run indefinitely).
        #[arg(long, default_value_t = 0)]
        max_cycles: u64,
    },
    /// Stop the background daemon.
    Stop,
    /// Show the background daemon's status.
    Status,
}

/// Global akhomed config get/set (control-plane, over ACP `_akh/config/*`).
#[derive(Subcommand)]
enum ConfigAction {
    /// Print the daemon's current config as JSON.
    Get,
    /// Replace the daemon's config with the contents of a TOML file.
    ///
    /// `_akh/config/set` replaces the whole `AkhomedConfig`; this reads + parses
    /// the file locally and sends it, so the file must be a complete config.
    Set {
        /// Path to a TOML file holding the full `AkhomedConfig`.
        #[arg(long)]
        file: PathBuf,
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
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // Default: show akh-medu info, silence noisy deps (egg, hnsw).
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();

    let cli = Cli::parse();

    // Resolve paths and engine config, then dispatch.
    {
    let language = Language::from_code(&cli.language).unwrap_or(Language::Auto);

    // Resolve data directory: explicit --data-dir wins, otherwise XDG workspace.
    let (data_dir, compartments_dir, xdg_paths) = if let Some(ref explicit) = cli.data_dir {
        (Some(explicit.clone()), None, None)
    } else {
        match akh_medu::paths::AkhPaths::resolve() {
            Ok(paths) => {
                let ws = paths.workspace(&cli.workspace);
                let dir = ws.kg_dir.clone();
                let comp_dir = ws.compartments_dir.clone();
                (Some(dir), Some(comp_dir), Some(paths))
            }
            Err(_) => (None, None, None),
        }
    };

    let config = EngineConfig {
        dimension: Dimension(cli.dimension),
        data_dir: data_dir.clone(),
        compartments_dir: compartments_dir.clone(),
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
                            compartments_dir: Some(ws_paths.compartments_dir.clone()),
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
                        compartments_dir: Some(ws_paths.compartments_dir.clone()),
                        language,
                        ..Default::default()
                    };
                    if let Ok(engine) = Engine::new(engine_config)
                        && let Some(role) = engine.assigned_role() {
                            println!("  Role: {role}");
                        }
                }
            }
        }

        Commands::Acp { op } => run_acp_command(&op)?,

        Commands::Daemon { action } => run_daemon_acp(&cli.workspace, action)?,

        Commands::Config { action } => run_config_acp(action)?,

        Commands::Chat { skill, headless, fresh } => {
            let ws_name = cli.workspace.clone();

            // Headless + a running daemon that advertises an ACP endpoint →
            // talk to the akh purely over ACP (the pure-ACP client path).
            // Falls through to the local engine when no daemon / no ACP.
            #[cfg(feature = "acp")]
            if headless
                && let Some(ref paths) = xdg_paths
                && let Some(server_info) = akh_medu::client::discover_server(paths)
            {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .into_diagnostic()?;
                if let Some(sock) = server_info.acp_socket.as_deref() {
                    eprintln!("Connecting to akh '{ws_name}' over ACP (unix:{sock})...");
                    return rt
                        .block_on(akh_medu::acp::run_headless_chat_unix(sock, "akh-cli", &ws_name))
                        .map_err(miette::Report::from);
                } else if let Some(tcp) = server_info.acp_tcp.as_deref() {
                    eprintln!("Connecting to akh '{ws_name}' over ACP (tcp:{tcp})...");
                    return rt
                        .block_on(akh_medu::acp::run_headless_chat_tcp(tcp, "akh-cli", &ws_name))
                        .map_err(miette::Report::from);
                }
                // Daemon found but built without ACP — fall through to local.
            }

            // Try remote TUI over ACP if a daemon is running (non-headless only).
            #[cfg(all(feature = "daemon", feature = "acp"))]
            if !headless {
                if let Some(ref paths) = xdg_paths
                    && let Some(server_info) = akh_medu::client::discover_server(paths) {
                        eprintln!("Connecting to akh '{ws_name}' over ACP...");
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

        Commands::Service { action } => {
            dispatch_service(action)?;
        }

        // ── Setup (models & runtime) ────────────────────────────────────
        Commands::Setup { action } => {
            dispatch_setup(action)?;
        }

        // ── Wiki (Akhipedia) ─────────────────────────────────────────────
    }

    Ok(())
    }
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

/// Dispatch `akh setup` subcommands. Works without an Engine.
fn dispatch_setup(action: SetupAction) -> Result<()> {
    use akh_medu::setup;

    let paths = akh_medu::paths::AkhPaths::resolve().into_diagnostic()?;

    match action {
        SetupAction::Models { ner_only, llm_only } => {
            if !llm_only {
                eprintln!("=== Downloading NER models (Tier 2) ===");
                setup::download_ner_models(&paths).into_diagnostic()?;
            }
            if !ner_only {
                eprintln!("=== Downloading LLM model (Tier 3) ===");
                setup::download_llm_model(&paths).into_diagnostic()?;
            }
            eprintln!("\nDone. Models installed to: {}", paths.models_dir().display());
        }
        SetupAction::OnnxRuntime { version } => {
            let v = version.as_deref().unwrap_or(setup::default_ort_version());
            eprintln!("=== Installing ONNX Runtime v{v} ===");
            setup::install_onnx_runtime(v).into_diagnostic()?;
        }
        SetupAction::Check => {
            setup::check_setup(&paths).into_diagnostic()?;
        }
    }

    Ok(())
}


