//! akh CLI: neuro-symbolic AI engine.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use miette::{IntoDiagnostic, Result};

#[derive(Parser)]
#[command(name = "akh", version, about = "Neuro-symbolic AI engine")]
struct Cli {
    /// Workspace name (default: "default") — selects which akh to talk to over
    /// ACP. `akh` is a pure ACP client: all engine work happens in the daemon.
    #[arg(short = 'w', long, global = true, default_value = "default")]
    workspace: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage workspaces (over ACP — the daemon owns the engines).
    Workspace {
        #[command(subcommand)]
        action: WorkspaceAction,
    },

    /// Interactive chat with the akh over ACP (launches the TUI, or --headless).
    Chat {
        /// Headless mode: plain stdin/stdout instead of the TUI.
        #[arg(long)]
        headless: bool,
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
/// `_akh/*` request over ACP, and pretty-print the JSON result.
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

/// Dispatch `akh workspace <action>` over the ACP control-plane
/// (`_akh/workspace/*`). The daemon owns the engines, so even create/delete go
/// over ACP — `akh` never touches a workspace on disk.
#[cfg(feature = "acp")]
fn run_workspace_acp(current_ws: &str, action: WorkspaceAction) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match action {
        WorkspaceAction::List => ("_akh/workspace/list", serde_json::json!({})),
        WorkspaceAction::Create { name, role } => {
            let mut params = serde_json::json!({ "name": name });
            if let Some(role) = role {
                params["role"] = serde_json::Value::String(role);
            }
            ("_akh/workspace/create", params)
        }
        WorkspaceAction::Delete { name } => {
            ("_akh/workspace/delete", serde_json::json!({ "name": name }))
        }
        WorkspaceAction::Info { name } => {
            let target = name.unwrap_or_else(|| current_ws.to_string());
            ("_akh/workspace/info", serde_json::json!({ "name": target }))
        }
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh chat`: discover the running daemon and converse over ACP — the
/// TUI backend (default) or a headless stdin/stdout loop. `akh` has no local
/// engine, so this errors when no daemon is running.
#[cfg(feature = "acp")]
fn run_chat_acp(ws_name: &str, headless: bool) -> Result<()> {
    let paths = akh_medu::paths::AkhPaths::resolve()
        .map_err(|e| miette::miette!("cannot resolve XDG paths: {e}"))?;
    let info = akh_medu::client::discover_server(&paths).ok_or_else(|| {
        miette::miette!(
            "No running akhomed daemon found. Start one first (`akh service start`, \
             or run `akhomed`) and retry — `akh` is a pure ACP client with no local engine."
        )
    })?;

    if !headless {
        eprintln!("Connecting to akh '{ws_name}' over ACP...");
        return akh_medu::tui::launch_remote(ws_name, &info);
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .into_diagnostic()?;
    if let Some(sock) = info.acp_socket.as_deref() {
        eprintln!("Connecting to akh '{ws_name}' over ACP (unix:{sock})...");
        rt.block_on(akh_medu::acp::run_headless_chat_unix(sock, "akh-cli", ws_name))
            .map_err(miette::Report::from)
    } else if let Some(tcp) = info.acp_tcp.as_deref() {
        eprintln!("Connecting to akh '{ws_name}' over ACP (tcp:{tcp})...");
        rt.block_on(akh_medu::acp::run_headless_chat_tcp(tcp, "akh-cli", ws_name))
            .map_err(miette::Report::from)
    } else {
        miette::bail!(
            "akhomed is running but advertises no ACP endpoint — build it with --features acp."
        )
    }
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

#[cfg(not(feature = "acp"))]
fn run_workspace_acp(_current_ws: &str, _action: WorkspaceAction) -> Result<()> {
    miette::bail!("The `workspace` subcommand requires building `akh` with --features acp.")
}

#[cfg(not(feature = "acp"))]
fn run_chat_acp(_ws_name: &str, _headless: bool) -> Result<()> {
    miette::bail!("The `chat` subcommand requires building `akh` with --features acp.")
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

    {
    match cli.command {

        Commands::Workspace { action } => run_workspace_acp(&cli.workspace, action)?,

        Commands::Acp { op } => run_acp_command(&op)?,

        Commands::Daemon { action } => run_daemon_acp(&cli.workspace, action)?,

        Commands::Config { action } => run_config_acp(action)?,

        Commands::Chat { headless } => run_chat_acp(&cli.workspace, headless)?,

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


