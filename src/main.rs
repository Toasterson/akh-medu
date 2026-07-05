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

    /// Target a remote akhomed over ACP/TCP (`host:port`), bypassing local
    /// daemon discovery. Defaults to the `AKH_REMOTE` env var. The remote daemon
    /// must serve ACP/TCP (start it with `AKH_ACP_TCP_PORT` / `AKH_ACP_TCP_BIND`).
    #[arg(long, global = true, value_name = "HOST:PORT")]
    remote: Option<String>,

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

    /// Snapshot or wake an akh's state to/from Postgres (over ACP `_akh/akh/*`).
    Akh {
        #[command(subcommand)]
        action: AkhAction,
    },

    /// Refresh the AGE graph projection (read-model) over ACP
    /// (`_akh/projection/*`). The projection is the queryable graph view of the
    /// relational truth; `refresh` rebuilds it (Phase 38d — currently a stub).
    Projection {
        #[command(subcommand)]
        action: ProjectionAction,
    },

    /// Run the akhomed-store database migrations on demand over ACP
    /// (`_akh/migrate`). Equivalent to the daemon's start-up `migrate_on_start`,
    /// surfaced as an explicit operator verb. Requires a Postgres backing store.
    Migrate,

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

    /// Compile / inspect schema packs (ADR 081). Local file operations — the
    /// registry admit/install verbs arrive with the Postgres registry.
    Schema {
        #[command(subcommand)]
        action: SchemaAction,
    },
}

/// Schema-pack operations (offline compiler half of ADR 081 Slice 2).
#[derive(Subcommand)]
enum SchemaAction {
    /// Compile a portable schema-pack source (JSON) into a native-triple blob.
    Compile {
        /// Path to the portable source (e.g. data/schema-packs/schema-org-core.json).
        source: PathBuf,
        /// Output blob path (default: source path with `.akhschema` extension).
        #[arg(short, long)]
        out: Option<PathBuf>,
    },
    /// Decode and summarize a compiled schema-pack blob.
    Show {
        /// Path to a compiled `.akhschema` blob.
        blob: PathBuf,
    },
}

/// Relocation / lifecycle ops: snapshot, wake, sleep, or relocate an akh
/// (control-plane, over ACP `_akh/akh/*`).
#[derive(Subcommand)]
enum AkhAction {
    /// Snapshot the akh's relational state to Postgres now.
    Snapshot {
        /// Workspace to snapshot (defaults to --workspace).
        workspace: Option<String>,
    },
    /// Wake a cold akh from its Postgres snapshot into the local workspace.
    Wake {
        /// Workspace to wake (defaults to --workspace).
        workspace: Option<String>,
    },
    /// Deep-sleep the akh: snapshot + flush, then free its in-memory engine and
    /// stop its background tasks. Wake it later with `akh akh wake`.
    Sleep {
        /// Workspace to sleep (defaults to --workspace).
        workspace: Option<String>,
    },
    /// Prepare the akh to move hosts: snapshot + flush so another host can
    /// `akh akh wake` it. Requires a Postgres backing store.
    Relocate {
        /// Workspace to relocate (defaults to --workspace).
        workspace: Option<String>,
    },
}

/// AGE graph-projection ops (control-plane, over ACP `_akh/projection/*`).
#[derive(Subcommand)]
enum ProjectionAction {
    /// Rebuild the AGE graph projection from relational truth (Phase 38d stub).
    Refresh,
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
    /// Remove a triple from the akh's knowledge graph (idempotent — a triple or
    /// label that isn't present reports `removed: false`). Records a WAL
    /// `Retract` and cascades through the TMS like any other retraction.
    RemoveTriple {
        /// Subject label.
        subject: String,
        /// Predicate (relation) label.
        predicate: String,
        /// Object label.
        object: String,
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
    /// Manage microtheories — isolated knowledge compartments (Phase 29a,
    /// `_akh/microtheory/*`).
    Microtheory {
        #[command(subcommand)]
        action: MicrotheoryOp,
    },
    /// Manage skillpacks — Cold/Warm/Hot capability bundles (Phase 29b,
    /// `_akh/skills*`).
    Skill {
        #[command(subcommand)]
        action: SkillOp,
    },
    /// Inspect the akh's psyche or tune its archetype weights (Phase 29c,
    /// `_akh/psyche*`).
    Psyche {
        #[command(subcommand)]
        action: PsycheOp,
    },
    /// Marshal pro/con arguments for (subject, predicate, ?) from the KG's
    /// provenance chains and render a verdict (Phase 29d, `_akh/argue`).
    Argue {
        /// Subject label (e.g. "Sun").
        subject: String,
        /// Predicate (relation) label (e.g. "is-a").
        predicate: String,
    },
    /// Explain a belief: derivation trace + prose — "why do you think X?"
    /// (Phase 29e, `_akh/explain`).
    Explain {
        /// Entity label to explain (e.g. "Sun"). Omit when using --question.
        subject: Option<String>,
        /// Free-text explanation question ("why is X?", "what do you know
        /// about X?", "how confident are you about X?", "what changed?").
        #[arg(long)]
        question: Option<String>,
        /// Maximum derivation-tree depth.
        #[arg(long, default_value_t = 5)]
        max_depth: u64,
    },
    /// Show the raw provenance records behind a symbol (Phase 29e,
    /// `_akh/provenance`).
    Provenance {
        /// Symbol label (e.g. "Sun").
        symbol: String,
    },
    /// Dempster-Shafer evidence: landscape summary or per-claim assessment
    /// (Phase 29h, `_akh/evidence*`).
    Evidence {
        #[command(subcommand)]
        action: EvidenceOp,
    },
    /// Source-trust leaderboard — Admiralty + Bayesian trust (Phase 29i,
    /// `_akh/trust`).
    Trust {
        /// How many sources to list per direction (most/least reliable).
        #[arg(long, default_value_t = 10)]
        top: u64,
    },
}

/// Microtheory (compartment) lifecycle verbs (`akh acp microtheory ...`).
#[derive(Subcommand)]
enum MicrotheoryOp {
    /// List every microtheory with its state and triple count.
    List,
    /// Create a new (Dormant) microtheory.
    Create {
        /// Unique id (e.g. "cooking", "project-x").
        id: String,
        /// Display name (defaults to the id).
        #[arg(long)]
        name: Option<String>,
        /// Kind: core | skill | project | library | person | workspace.
        #[arg(long, default_value = "project")]
        kind: String,
        /// Purpose description.
        #[arg(long)]
        description: Option<String>,
        /// Parent microtheory id (hierarchy).
        #[arg(long)]
        parent: Option<String>,
    },
    /// Inspect one microtheory: manifest, state, and a sample of its triples.
    Inspect {
        /// Microtheory id.
        id: String,
        /// Maximum sampled triples to return.
        #[arg(long, default_value_t = 50)]
        limit: u64,
    },
    /// Load a microtheory's triples into the KG and mark it Active.
    Activate {
        /// Microtheory id.
        id: String,
    },
    /// Stop a microtheory influencing the OODA loop (back to Loaded); with
    /// --unload, also remove its triples from the KG (back to Dormant).
    Deactivate {
        /// Microtheory id.
        id: String,
        /// Also unload the microtheory's triples from the KG.
        #[arg(long)]
        unload: bool,
    },
}

/// Skillpack lifecycle verbs (`akh acp skill ...`).
#[derive(Subcommand)]
enum SkillOp {
    /// List every discovered skillpack with its state and counts.
    List,
    /// Inspect one skillpack: manifest, actions, and plans.
    Inspect {
        /// Skill id (e.g. "astronomy").
        id: String,
    },
    /// Activate (load) a skillpack: its triples, rules, actions, and plans
    /// enter the KG (state → Hot).
    Activate {
        /// Skill id.
        id: String,
    },
    /// Deactivate (unload) a skillpack: its knowledge leaves the KG (→ Cold).
    Deactivate {
        /// Skill id.
        id: String,
    },
}

/// Psyche verbs (`akh acp psyche ...`).
#[derive(Subcommand)]
enum PsycheOp {
    /// Show the full psyche: persona, archetypes, OCEAN, shadow patterns.
    Show,
    /// Show or tune the archetype weights (healer/sage/guardian/explorer).
    /// With no flags, reports the current weights.
    Archetypes {
        /// Healer weight in [0,1] — seeks missing knowledge.
        #[arg(long)]
        healer: Option<f64>,
        /// Sage weight in [0,1] — seeks understanding.
        #[arg(long)]
        sage: Option<f64>,
        /// Guardian weight in [0,1] — seeks stability and safety.
        #[arg(long)]
        guardian: Option<f64>,
        /// Explorer weight in [0,1] — seeks novelty.
        #[arg(long)]
        explorer: Option<f64>,
    },
}

/// Evidence verbs (`akh acp evidence ...`).
#[derive(Subcommand)]
enum EvidenceOp {
    /// Summarize the evidence landscape (claim count + most uncertain claims).
    Summary,
    /// Assess one claim: Dempster-Shafer belief interval + verdict.
    Assess {
        /// Claim symbol label.
        claim: String,
    },
}

/// The remote ACP/TCP endpoint chosen by `--remote` (or `AKH_REMOTE`), captured
/// once at startup. `Some(None)`/unset means "discover the local daemon".
static REMOTE_ENDPOINT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Resolve which akhomed to talk to: an explicit `--remote host:port` (or
/// `AKH_REMOTE`) targets a remote daemon over ACP/TCP; otherwise discover the
/// local daemon via its PID file. Shared by every ACP client path so `--remote`
/// works uniformly across the data-plane, control-plane, and chat.
#[cfg(feature = "acp")]
fn resolve_server(paths: &akh_medu::paths::AkhPaths) -> Result<akh_medu::client::ServerInfo> {
    if let Some(Some(addr)) = REMOTE_ENDPOINT.get() {
        return Ok(akh_medu::client::ServerInfo::remote_tcp(addr));
    }
    akh_medu::client::discover_server(paths).ok_or_else(|| {
        miette::miette!(
            "No running akhomed daemon found. Start one (`akh service start`, or run \
             `akhomed`), or pass `--remote host:port` to reach a remote daemon over ACP/TCP."
        )
    })
}

/// Discover the running daemon and send one `_akh/*` request over ACP, returning
/// the JSON result. Shared by the data-plane (`akh acp`) and control-plane
/// (`akh daemon`/`akh config`) command dispatchers.
#[cfg(feature = "acp")]
fn acp_call(method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
    let paths = akh_medu::paths::AkhPaths::resolve()
        .map_err(|e| miette::miette!("cannot resolve XDG paths: {e}"))?;
    let info = resolve_server(&paths)?;

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
fn run_acp_command(ws: &str, op: &AcpOp) -> Result<()> {
    let (method, mut params): (&str, serde_json::Value) = match op {
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
        AcpOp::RemoveTriple {
            subject,
            predicate,
            object,
        } => (
            "_akh/triples/remove",
            serde_json::json!({
                "subject": subject,
                "predicate": predicate,
                "object": object,
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
        // ── Phase 29 capability exposure (plan
        //    docs/ai/plans/2026-03-26-phase29-capability-exposure.md, ADR 039) ──
        AcpOp::Microtheory { action } => match action {
            MicrotheoryOp::List => ("_akh/microtheory/list", serde_json::json!({})),
            MicrotheoryOp::Create {
                id,
                name,
                kind,
                description,
                parent,
            } => {
                let mut params = serde_json::json!({ "id": id, "kind": kind });
                if let Some(n) = name {
                    params["name"] = serde_json::Value::String(n.clone());
                }
                if let Some(d) = description {
                    params["description"] = serde_json::Value::String(d.clone());
                }
                if let Some(p) = parent {
                    params["parent"] = serde_json::Value::String(p.clone());
                }
                ("_akh/microtheory/create", params)
            }
            MicrotheoryOp::Inspect { id, limit } => (
                "_akh/microtheory/inspect",
                serde_json::json!({ "id": id, "limit": limit }),
            ),
            MicrotheoryOp::Activate { id } => {
                ("_akh/microtheory/activate", serde_json::json!({ "id": id }))
            }
            MicrotheoryOp::Deactivate { id, unload } => (
                "_akh/microtheory/deactivate",
                serde_json::json!({ "id": id, "unload": unload }),
            ),
        },
        AcpOp::Skill { action } => match action {
            SkillOp::List => ("_akh/skills", serde_json::json!({})),
            SkillOp::Inspect { id } => {
                ("_akh/skills/inspect", serde_json::json!({ "id": id }))
            }
            SkillOp::Activate { id } => {
                ("_akh/skills/activate", serde_json::json!({ "id": id }))
            }
            SkillOp::Deactivate { id } => {
                ("_akh/skills/deactivate", serde_json::json!({ "id": id }))
            }
        },
        AcpOp::Psyche { action } => match action {
            PsycheOp::Show => ("_akh/psyche", serde_json::json!({})),
            PsycheOp::Archetypes {
                healer,
                sage,
                guardian,
                explorer,
            } => {
                let mut params = serde_json::json!({});
                for (key, value) in [
                    ("healer", healer),
                    ("sage", sage),
                    ("guardian", guardian),
                    ("explorer", explorer),
                ] {
                    if let Some(v) = value {
                        params[key] = serde_json::json!(v);
                    }
                }
                ("_akh/psyche/archetypes", params)
            }
        },
        AcpOp::Argue { subject, predicate } => (
            "_akh/argue",
            serde_json::json!({ "subject": subject, "predicate": predicate }),
        ),
        AcpOp::Explain {
            subject,
            question,
            max_depth,
        } => {
            let mut params = serde_json::json!({ "max_depth": max_depth });
            match (subject, question) {
                (_, Some(q)) => params["question"] = serde_json::Value::String(q.clone()),
                (Some(s), None) => params["subject"] = serde_json::Value::String(s.clone()),
                (None, None) => miette::bail!(
                    "`akh acp explain` needs an entity label or a --question — e.g. \
                     `akh acp explain Sun` or `akh acp explain --question \"why Sun?\"`"
                ),
            }
            ("_akh/explain", params)
        }
        AcpOp::Provenance { symbol } => {
            ("_akh/provenance", serde_json::json!({ "symbol": symbol }))
        }
        AcpOp::Evidence { action } => match action {
            EvidenceOp::Summary => ("_akh/evidence", serde_json::json!({})),
            EvidenceOp::Assess { claim } => {
                ("_akh/evidence/assess", serde_json::json!({ "claim": claim }))
            }
        },
        AcpOp::Trust { top } => ("_akh/trust", serde_json::json!({ "top_k": top })),
    };
    // Route the data-plane op to the akh selected by `-w` (the server defaults to
    // its bound workspace when absent). Don't override an explicit param.
    if let Some(obj) = params.as_object_mut() {
        obj.entry("workspace")
            .or_insert_with(|| serde_json::Value::String(ws.to_string()));
    }
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
        WorkspaceAction::Join { id } => {
            ("_akh/workspace/join", serde_json::json!({ "id": id }))
        }
        WorkspaceAction::Leave { id } => {
            ("_akh/workspace/leave", serde_json::json!({ "id": id }))
        }
        WorkspaceAction::ListMembers { id } => (
            "_akh/workspace/list-members",
            serde_json::json!({ "id": id }),
        ),
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh akh <action>` over the ACP control-plane (`_akh/akh/*`):
/// snapshot the akh's state to Postgres, or wake a cold akh from it.
#[cfg(feature = "acp")]
fn run_akh_acp(current_ws: &str, action: AkhAction) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match action {
        AkhAction::Snapshot { workspace } => {
            let ws = workspace.unwrap_or_else(|| current_ws.to_string());
            ("_akh/akh/snapshot", serde_json::json!({ "workspace": ws }))
        }
        AkhAction::Wake { workspace } => {
            let ws = workspace.unwrap_or_else(|| current_ws.to_string());
            ("_akh/akh/wake", serde_json::json!({ "workspace": ws }))
        }
        AkhAction::Sleep { workspace } => {
            let ws = workspace.unwrap_or_else(|| current_ws.to_string());
            ("_akh/akh/sleep", serde_json::json!({ "workspace": ws }))
        }
        AkhAction::Relocate { workspace } => {
            let ws = workspace.unwrap_or_else(|| current_ws.to_string());
            ("_akh/akh/relocate", serde_json::json!({ "workspace": ws }))
        }
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh projection <action>` over the ACP control-plane
/// (`_akh/projection/*`). The projection is the AGE graph read-model rebuilt
/// from relational truth (Phase 38d — currently a stub).
#[cfg(feature = "acp")]
fn run_projection_acp(current_ws: &str, action: ProjectionAction) -> Result<()> {
    let (method, params): (&str, serde_json::Value) = match action {
        ProjectionAction::Refresh => (
            "_akh/projection/refresh",
            serde_json::json!({ "workspace": current_ws }),
        ),
    };
    let result = acp_call(method, params)?;
    print_acp(&result)
}

/// Dispatch `akh migrate` over the ACP control-plane (`_akh/migrate`): run the
/// akhomed-store migrations against the daemon's Postgres backing store.
#[cfg(feature = "acp")]
fn run_migrate_acp() -> Result<()> {
    let result = acp_call("_akh/migrate", serde_json::json!({}))?;
    print_acp(&result)
}

/// Dispatch `akh chat`: discover the running daemon and converse over ACP — the
/// TUI backend (default) or a headless stdin/stdout loop. `akh` has no local
/// engine, so this errors when no daemon is running.
#[cfg(feature = "acp")]
fn run_chat_acp(ws_name: &str, headless: bool) -> Result<()> {
    let paths = akh_medu::paths::AkhPaths::resolve()
        .map_err(|e| miette::miette!("cannot resolve XDG paths: {e}"))?;
    let info = resolve_server(&paths)?;

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
fn run_acp_command(_ws: &str, _op: &AcpOp) -> Result<()> {
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

#[cfg(not(feature = "acp"))]
fn run_projection_acp(_current_ws: &str, _action: ProjectionAction) -> Result<()> {
    miette::bail!("The `projection` subcommand requires building `akh` with --features acp.")
}

#[cfg(not(feature = "acp"))]
fn run_migrate_acp() -> Result<()> {
    miette::bail!("The `migrate` subcommand requires building `akh` with --features acp.")
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
    /// Join a workspace: add this akh as a member (knowledge sharing). Requires
    /// a Postgres backing store.
    Join {
        /// Workspace id/name to join.
        id: String,
    },
    /// Leave a workspace: remove this akh's membership. Requires a Postgres
    /// backing store.
    Leave {
        /// Workspace id/name to leave.
        id: String,
    },
    /// List the akhs that are members of a workspace. Requires a Postgres
    /// backing store.
    ListMembers {
        /// Workspace id/name to list members of.
        id: String,
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

    // Capture the remote endpoint (flag wins over the AKH_REMOTE env) before any
    // ACP path resolves the daemon. Set once; ignore a redundant re-set.
    let remote = cli
        .remote
        .clone()
        .or_else(|| std::env::var("AKH_REMOTE").ok());
    let _ = REMOTE_ENDPOINT.set(remote);

    {
    match cli.command {

        Commands::Workspace { action } => run_workspace_acp(&cli.workspace, action)?,

        Commands::Acp { op } => run_acp_command(&cli.workspace, &op)?,

        Commands::Daemon { action } => run_daemon_acp(&cli.workspace, action)?,

        Commands::Config { action } => run_config_acp(action)?,

        Commands::Akh { action } => run_akh_acp(&cli.workspace, action)?,

        Commands::Projection { action } => run_projection_acp(&cli.workspace, action)?,

        Commands::Migrate => run_migrate_acp()?,

        Commands::Chat { headless } => run_chat_acp(&cli.workspace, headless)?,

        Commands::Service { action } => {
            dispatch_service(action)?;
        }

        // ── Setup (models & runtime) ────────────────────────────────────
        Commands::Setup { action } => {
            dispatch_setup(action)?;
        }

        // ── Schema packs (ADR 081) ──────────────────────────────────────
        Commands::Schema { action } => {
            dispatch_schema(action)?;
        }

        // ── Wiki (Akhipedia) ─────────────────────────────────────────────
    }

    Ok(())
    }
}

/// Dispatch `akh schema` subcommands. Local file operations; no Engine, no ACP.
fn dispatch_schema(action: SchemaAction) -> Result<()> {
    use akh_medu::schema::{CompiledSchemaPack, compile};

    match action {
        SchemaAction::Compile { source, out } => {
            let text = std::fs::read_to_string(&source).into_diagnostic()?;
            let pack = compile(&text)?;
            let blob = pack.to_blob()?;
            let out = out.unwrap_or_else(|| source.with_extension("akhschema"));
            std::fs::write(&out, &blob).into_diagnostic()?;
            println!(
                "Compiled schema pack {}@{} ({})",
                pack.name, pack.version, pack.scope
            );
            println!("  microtheory: {}", pack.microtheory_label());
            println!(
                "  types: {} · relations: {} · triples: {}",
                pack.types.len(),
                pack.relations.len(),
                pack.triples.len()
            );
            println!(
                "  blob: {} ({} bytes, blake3 {})",
                out.display(),
                blob.len(),
                akh_medu::schema::compile::checksum_hex(&blob)
            );
            if !pack.assumes.is_empty() {
                println!(
                    "  assumes (verified against the registry at confirm time): {}",
                    pack.assumes.join(", ")
                );
            }
        }
        SchemaAction::Show { blob } => {
            let bytes = std::fs::read(&blob).into_diagnostic()?;
            let pack = CompiledSchemaPack::from_blob(&bytes)?;
            println!(
                "Schema pack {}@{} ({}) — format v{}",
                pack.name, pack.version, pack.scope, pack.format_version
            );
            println!("  microtheory: {}", pack.microtheory_label());
            if let Some(url) = &pack.source_url {
                println!("  source: {url}");
            }
            println!("  source checksum: {}", pack.source_checksum);
            println!("  types: {}", pack.types.join(", "));
            if !pack.relations.is_empty() {
                println!("  relations: {}", pack.relations.join(", "));
            }
            if !pack.assumes.is_empty() {
                println!("  assumes: {}", pack.assumes.join(", "));
            }
            println!("  triples ({}):", pack.triples.len());
            for t in &pack.triples {
                println!("    ({}, {}, {})", t.subject, t.predicate, t.object);
            }
        }
    }
    Ok(())
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


