//! Phase 50 / W6 fleet-orchestration benchmark: does splitting a
//! decomposable illumos task across an **orchestrator akh + N worker akhs**
//! beat a **single akh** doing the whole thing over the live `exec1` guest?
//!
//! This is a straight implementation of the blueprint at
//! `docs/ai/2026-07-07-phase50-w6-orchestration-design.md` — see that doc
//! for the full rationale. Reuses the Phase 50 / W3 runner's wiring chain
//! (`examples/phase50_w3_runner.rs`): `RemoteBackend` construction, the
//! `AgentModelManager`+`OllamaConfig`+`OllamaChatBackend` cloud chat wiring,
//! `Agent::with_backend`, the `MessageSink`/drive-turn shape, setup/verify
//! upload+run over the executor, and `RunRecord` assembly via
//! `harness_core::store::write_run`.
//!
//! ## Run it
//!
//! ```sh
//! export CARGO_TARGET_DIR=/home/toasty/ws/agi/akh-medu/target
//! cargo build --features remote-executor --example phase50_w6_orchestration
//!
//! # Smoke: exercises setup/verify + the N-way fan-out threading against
//! # exec1, with ZERO cloud calls (proves the harness plumbing only).
//! cargo run --features remote-executor --example phase50_w6_orchestration -- \
//!   --dry-run --results /tmp/w6-dryrun
//!
//! # Real (spends cloud budget!): control vs orchestrated, one model.
//! export OLLAMA_API_KEY="$OLLAMA_CLOUD_KEY"
//! cargo run --features remote-executor --example phase50_w6_orchestration -- \
//!   --model glm-5.2
//! ```

#[cfg(feature = "remote-executor")]
fn main() -> miette::Result<()> {
    inner::run()
}

#[cfg(not(feature = "remote-executor"))]
fn main() {
    eprintln!("build with --features remote-executor");
    std::process::exit(1);
}

#[cfg(feature = "remote-executor")]
mod inner {
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use akh_medu::agent::chat_backend::OllamaChatBackend;
    use akh_medu::agent::model_manager::{AgentModelManager, ModelManagerConfig, UsageSnapshot};
    use akh_medu::agent::ollama::OllamaConfig;
    use akh_medu::agent::tools::ShellExecTool;
    use akh_medu::agent::{Agent, AgentConfig, ToolInput, ToolRegistry};
    use akh_medu::audit::AuditKind;
    use akh_medu::backend::remote::{HostKeyPolicy, RemoteBackend, RemoteBackendConfig};
    use akh_medu::backend::{Backend, ExecRequest};
    use akh_medu::engine::{Engine, EngineConfig};
    use akh_medu::message::{AkhMessage, MessageSink, VecSink};

    use clap::Parser;
    use harness_core::hash::hash_task_set_dir;
    use harness_core::schema::{
        Blake3Hex, GitSha, GradingMode, HarnessId, HostTier, ModelBacking, ModelIdentity, ModelTag,
        ResourceUsage, RunRecord, TaskId, TaskOutcome, TaskResult, VmImageIdentity, VmRecipeHash,
        WallTimeSecs,
    };
    use miette::Diagnostic;
    use serde::Deserialize;
    use thiserror::Error;

    // -----------------------------------------------------------------
    // CLI
    // -----------------------------------------------------------------

    /// Absolute path (this worktree does not carry the gitignored key —
    /// `tests/vm/.keys/` is per-checkout-generated) to the exec1 root key
    /// staged on the box this example is meant to run from.
    const DEFAULT_KEY_PATH: &str = "/home/toasty/ws/agi/akh-medu/tests/vm/.keys/root_ed25519";

    /// The VM image recipe exec1 was baked from (Phase 50 W1d) — hashed for
    /// `VmRecipeHash` so a run record names *which* guest image it targeted.
    const RECIPE_SCRIPT_REL: &str = "tests/vm/scripts/bake-illumos-fixed-image.sh";

    /// Registry the base image (`openindiana-rust-ci`) was pulled from —
    /// see `tests/vm/VMFile.kdl`.
    const VM_REGISTRY: &str = "code.aopc.cloud/CloudNebulaProject";

    #[derive(Parser)]
    #[command(
        name = "phase50-w6-orchestration",
        about = "Phase 50 / W6: orchestrator+workers fleet vs single-akh control over the 48c RemoteBackend"
    )]
    struct Cli {
        /// Root directory of the orchestration task corpus.
        #[arg(long, default_value = "benchmarks/illumos-orchestration/tasks")]
        tasks_dir: PathBuf,

        /// Task id to run.
        #[arg(long, default_value = "relay-group-3svc")]
        task: String,

        /// Provider tag sent to the cloud API, e.g. `deepseek-v4-pro` /
        /// `glm-5.2` / `qwen3.5:397b`.
        #[arg(long, default_value = "glm-5.2")]
        model: String,

        /// Filesystem-safe grouping slug for the results directory. Defaults
        /// to a sanitized form of `--model`.
        #[arg(long)]
        model_slug: Option<String>,

        /// exec1 SSH target as `user@host:port`.
        #[arg(long, default_value = "root@127.0.0.1:10042")]
        exec1: String,

        /// Path to the exec1 private key.
        #[arg(long, default_value = DEFAULT_KEY_PATH)]
        key: PathBuf,

        /// Number of parallel workers for the orchestrated arm. Defaults to
        /// the task's declared `subtasks` count.
        #[arg(long)]
        workers: Option<usize>,

        /// Where `benchmarks/results/` (or an alternate results root) lives.
        #[arg(long, default_value = "benchmarks/results")]
        results: PathBuf,

        /// Run the full setup/verify/fan-out-threading/RunRecord plumbing
        /// but never drive the model (no cloud calls at all). Every task's
        /// outcome is `Aborted` by construction; the N-way fan-out still
        /// spawns real threads with real `RemoteBackend` connections and
        /// runs one trivial `shell_exec` per worker, so ledger attribution
        /// is exercised without spending any cloud budget.
        #[arg(long)]
        dry_run: bool,
    }

    // -----------------------------------------------------------------
    // Task metadata (meta.toml)
    // -----------------------------------------------------------------

    #[derive(Debug, Deserialize)]
    struct TaskMeta {
        #[allow(dead_code)]
        id: String,
        #[allow(dead_code)]
        family: String,
        #[allow(dead_code)]
        description: String,
        subtasks: usize,
        timeout_secs: u64,
        #[allow(dead_code)]
        expected_manpages: Vec<String>,
        #[allow(dead_code)]
        mode: Vec<String>,
        #[allow(dead_code)]
        parallelizable: bool,
        /// `[[subtask]]` array-of-tables — the ground-truth per-service
        /// subtask prompts, used to validate/fall back from the
        /// orchestrator's own decomposition (blueprint step 1).
        #[serde(default, rename = "subtask")]
        subtask: Vec<SubtaskSpec>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct SubtaskSpec {
        id: String,
        #[allow(dead_code)]
        fmri: String,
        prompt: String,
    }

    // -----------------------------------------------------------------
    // Runner-local errors
    // -----------------------------------------------------------------

    #[derive(Debug, Error, Diagnostic)]
    enum RunnerError {
        #[error("--exec1 must be user@host:port, got {value:?}")]
        #[diagnostic(
            code(phase50_w6::bad_exec1_spec),
            help("Example: root@127.0.0.1:10042")
        )]
        BadExec1Spec { value: String },

        #[error("bad port in --exec1 ({value:?}): {source}")]
        #[diagnostic(code(phase50_w6::bad_port))]
        BadPort {
            value: String,
            #[source]
            source: std::num::ParseIntError,
        },

        #[error("no task directory {task_id:?} under {dir}")]
        #[diagnostic(
            code(phase50_w6::unknown_task),
            help("Check the id matches a directory name under --tasks-dir.")
        )]
        UnknownTask { task_id: String, dir: PathBuf },

        #[error("failed to read {path}: {source}")]
        #[diagnostic(code(phase50_w6::io))]
        Io {
            path: PathBuf,
            #[source]
            source: std::io::Error,
        },

        #[error("failed to parse {path} as TOML: {source}")]
        #[diagnostic(
            code(phase50_w6::meta_parse),
            help("meta.toml must match the shape documented in benchmarks/illumos-orchestration/README.md.")
        )]
        MetaParse {
            path: PathBuf,
            #[source]
            source: toml::de::Error,
        },

        #[error("task {task_id:?} declares {declared} subtasks but meta.toml has {actual} [[subtask]] entries")]
        #[diagnostic(code(phase50_w6::subtask_count_mismatch))]
        SubtaskCountMismatch {
            task_id: String,
            declared: usize,
            actual: usize,
        },
    }

    fn read_file(path: &Path) -> Result<String, RunnerError> {
        std::fs::read_to_string(path).map_err(|source| RunnerError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    fn parse_meta(task_dir: &Path) -> Result<TaskMeta, RunnerError> {
        let path = task_dir.join("meta.toml");
        let text = read_file(&path)?;
        let meta: TaskMeta = toml::from_str(&text).map_err(|source| RunnerError::MetaParse { path, source })?;
        if meta.subtask.len() != meta.subtasks {
            return Err(RunnerError::SubtaskCountMismatch {
                task_id: meta.id.clone(),
                declared: meta.subtasks,
                actual: meta.subtask.len(),
            });
        }
        Ok(meta)
    }

    fn default_model_slug(model: &str) -> String {
        let head = model.split(':').next().unwrap_or(model);
        head.chars()
            .map(|c| {
                let c = c.to_ascii_lowercase();
                if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.' || c == '_' {
                    c
                } else {
                    '-'
                }
            })
            .collect()
    }

    fn parse_exec1_spec(spec: &str) -> Result<(String, String, u16), RunnerError> {
        let (user, host_port) =
            spec.split_once('@')
                .ok_or_else(|| RunnerError::BadExec1Spec { value: spec.to_string() })?;
        let (host, port_str) = host_port
            .rsplit_once(':')
            .ok_or_else(|| RunnerError::BadExec1Spec { value: spec.to_string() })?;
        let port: u16 = port_str.parse().map_err(|source| RunnerError::BadPort {
            value: port_str.to_string(),
            source,
        })?;
        Ok((user.to_string(), host.to_string(), port))
    }

    // -----------------------------------------------------------------
    // Message sinks
    // -----------------------------------------------------------------

    /// Prefixed trace sink — mirrors `examples/phase50_w3_runner.rs`'s
    /// `TraceSink`, with a per-arm/per-worker label so concurrent fan-out
    /// output stays attributable on stdout.
    struct TraceSink {
        prefix: String,
    }
    impl TraceSink {
        fn new(prefix: impl Into<String>) -> Self {
            Self { prefix: prefix.into() }
        }
    }
    impl MessageSink for TraceSink {
        fn emit(&self, msg: &AkhMessage) {
            let p = &self.prefix;
            match msg {
                AkhMessage::Reasoning { step, .. } => println!("    [{p}] [think] {step}"),
                AkhMessage::ToolResult { tool, success, output } => {
                    let mark = if *success { "ok" } else { "FAIL" };
                    println!("    [{p}] [tool {mark} {tool}] {}", output.replace('\n', " "));
                }
                AkhMessage::Narrative { text, .. } => println!("    [{p}] [answer] {text}"),
                AkhMessage::Fact { text, .. } => println!("    [{p}] [fact] {text}"),
                AkhMessage::System { text } => println!("    [{p}] [system] {text}"),
                AkhMessage::Error { code, message, .. } => {
                    println!("    [{p}] [error {code}] {message}")
                }
                other => println!("    [{p}] [msg] {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Remote plumbing helpers (copied from phase50_w3_runner.rs)
    // -----------------------------------------------------------------

    fn upload_and_run(
        backend: &dyn Backend,
        remote_path: &Path,
        content: &str,
        timeout: Duration,
    ) -> miette::Result<(i32, String, String)> {
        backend.write(remote_path, content)?;
        let out = backend.exec(ExecRequest::shell(
            format!("sh {}", remote_path.display()),
            timeout,
        ))?;
        Ok((
            out.exit_code,
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ))
    }

    fn base_remote_config(host: &str, user: &str, port: u16, key: &Path) -> RemoteBackendConfig {
        RemoteBackendConfig::new(host.to_string(), user.to_string(), key.to_path_buf())
            .with_port(port)
            // Lab/loopback posture (ADR 086 §1): exec1 is a disposable VM
            // reachable only via a localhost port-forward for this session.
            .with_host_key_policy(HostKeyPolicy::InsecureAcceptAny)
            .with_roots(vec![PathBuf::from("/")], vec![
                PathBuf::from("/tmp"),
                PathBuf::from("/opt"),
                PathBuf::from("/var"),
            ])
            .with_connect_timeout(Duration::from_secs(15))
    }

    // -----------------------------------------------------------------
    // Driving the akh
    // -----------------------------------------------------------------

    const SYS_PROMPT: &str = "You are an akh: a precise illumos system engineer with root access \
        on the target host over a remote shell. Use `man`/`apropos` to ground unfamiliar commands \
        in the local manual pages before acting — this host is OpenIndiana Hipster, not Linux; \
        Linux-only tools and flags do not exist here. Diagnose before you act, verify your fix left \
        the system in the requested end state, and prefer the documented illumos-native command \
        over a habit carried over from another OS.";

    const DECOMPOSE_SUFFIX: &str = "\n\n---\n\nYou are the ORCHESTRATOR for this task, not the \
        engineer who fixes it. Do not run any commands yet. Split the task above into independent \
        subtasks — one per service — that could each be handed to a different engineer working in \
        parallel with zero shared state between them. Respond with ONLY a JSON array of strings, \
        one string per subtask, each string a complete, self-contained instruction (naming which \
        service/FMRI it concerns) that a worker could act on without seeing this message or the \
        other subtasks. No prose before or after the JSON array.";

    struct TurnResult {
        completed_in_time: bool,
        turn_ok: bool,
        wall: Duration,
    }

    /// Drive one turn to completion or until `timeout` elapses, whichever
    /// comes first (best-effort join-with-timeout, not true preemption —
    /// copied from `phase50_w3_runner.rs::drive_turn`).
    fn drive_turn(
        engine: Arc<Engine>,
        backend: Arc<dyn Backend>,
        mgr: Arc<AgentModelManager>,
        agent_id: &str,
        prompt: &str,
        timeout: Duration,
        sink: Arc<dyn MessageSink>,
    ) -> miette::Result<TurnResult> {
        let start = Instant::now();
        let mut agent = Agent::with_backend(Arc::clone(&engine), AgentConfig::default(), Some(backend))?;
        agent.set_chat_backend(Arc::new(OllamaChatBackend::new(Arc::clone(&mgr))));

        let (tx, rx) = std::sync::mpsc::channel();
        let agent_id_owned = agent_id.to_string();
        let prompt_owned = prompt.to_string();
        let engine_for_thread = Arc::clone(&engine);
        std::thread::spawn(move || {
            let outcome = agent.run_agentic_turn(&engine_for_thread, &agent_id_owned, &prompt_owned, sink.as_ref());
            let _ = tx.send(outcome.is_some());
        });

        let (completed_in_time, turn_ok) = match rx.recv_timeout(timeout) {
            Ok(ok) => (true, ok),
            Err(_) => {
                println!(
                    "    [w6:{agent_id}] turn did not complete within {}s — abandoning the driving thread",
                    timeout.as_secs()
                );
                (false, false)
            }
        };
        Ok(TurnResult { completed_in_time, turn_ok, wall: start.elapsed() })
    }

    /// Dry-run stand-in for a worker turn: connects its own `RemoteBackend`
    /// (distinct correlation_id) and runs ONE trivial `shell_exec` through a
    /// minimal `ToolRegistry` wired to the shared audit ledger — exercising
    /// the same backend-connect + tool-invocation + audit-envelope path a
    /// real agentic turn would, with zero cloud calls.
    fn dry_run_worker_exec(engine: &Engine, backend: Arc<dyn Backend>, worker_label: &str) -> miette::Result<()> {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ShellExecTool::with_backend(Arc::clone(&backend))));
        if let Some(ledger) = engine.audit_ledger() {
            registry.set_audit_ledger(Arc::clone(ledger));
        }
        let input = ToolInput::new()
            .with_param("command", format!("echo dry-run-{worker_label} && svcs -a | head -1"));
        let output = registry.execute("shell_exec", input, engine)?;
        println!("    [w6:dry:{worker_label}] shell_exec success={} : {}", output.success, output.result.replace('\n', " "));
        Ok(())
    }

    fn usage_delta(before: UsageSnapshot, after: UsageSnapshot) -> ResourceUsage {
        ResourceUsage {
            tokens_used: Some(after.total_tokens().saturating_sub(before.total_tokens())),
            requests_used: Some(after.turns.saturating_sub(before.turns)),
        }
    }

    fn add_usage(a: ResourceUsage, b: ResourceUsage) -> ResourceUsage {
        ResourceUsage {
            tokens_used: Some(a.tokens_used.unwrap_or(0) + b.tokens_used.unwrap_or(0)),
            requests_used: Some(a.requests_used.unwrap_or(0) + b.requests_used.unwrap_or(0)),
        }
    }

    /// Best-effort extraction of a JSON string array from the orchestrator's
    /// free-form narrative: try a direct parse first, then fall back to the
    /// substring between the first `[` and the matching last `]`.
    fn extract_json_string_array(text: &str) -> Option<Vec<String>> {
        let trimmed = text.trim();
        if let Ok(v) = serde_json::from_str::<Vec<String>>(trimmed) {
            return Some(v);
        }
        let start = trimmed.find('[')?;
        let end = trimmed.rfind(']')?;
        if end <= start {
            return None;
        }
        serde_json::from_str::<Vec<String>>(&trimmed[start..=end]).ok()
    }

    /// Redundant-command ratio (blueprint metric): from the shared audit
    /// ledger, group `ToolInvocation{tool_name: "shell_exec"}` entries by
    /// `ExecEnvelope.correlation_id` restricted to the given worker
    /// correlation ids, then count how many of those commands (by
    /// `params_summary`) were issued by more than one distinct worker.
    /// Returns `(redundant_count, total_worker_commands)`.
    fn redundant_command_ratio(engine: &Engine, worker_correlation_ids: &[String]) -> (usize, usize) {
        let Some(ledger) = engine.audit_ledger() else {
            return (0, 0);
        };
        let page = match ledger.list_by_kind(0, 0, 10_000) {
            Ok(page) => page,
            Err(_) => return (0, 0),
        };

        let worker_ids: HashSet<&str> = worker_correlation_ids.iter().map(String::as_str).collect();
        let mut by_worker: HashMap<&str, Vec<&str>> = HashMap::new();
        for entry in &page.entries {
            if let AuditKind::ToolInvocation { tool_name, params_summary, envelope, .. } = &entry.kind
                && tool_name == "shell_exec"
                && let Some(cid) = envelope.correlation_id.as_deref()
                && worker_ids.contains(cid)
            {
                by_worker.entry(cid).or_default().push(params_summary.as_str());
            }
        }

        let total: usize = by_worker.values().map(Vec::len).sum();

        let mut cmd_to_workers: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (cid, cmds) in &by_worker {
            for c in cmds {
                cmd_to_workers.entry(*c).or_default().insert(*cid);
            }
        }

        let redundant: usize = by_worker
            .iter()
            .flat_map(|(_cid, cmds)| cmds.iter())
            .filter(|c| cmd_to_workers.get(*c).map(|s| s.len() > 1).unwrap_or(false))
            .count();

        (redundant, total)
    }

    // -----------------------------------------------------------------
    // main
    // -----------------------------------------------------------------

    pub fn run() -> miette::Result<()> {
        let cli = Cli::parse();

        let (user, host, port) = parse_exec1_spec(&cli.exec1)?;
        let model_slug = cli
            .model_slug
            .clone()
            .unwrap_or_else(|| default_model_slug(&cli.model));

        println!("=== Phase 50 / W6 fleet-orchestration benchmark ===");
        println!("exec1        : {user}@{host}:{port}");
        println!("key          : {}", cli.key.display());
        println!("model        : {} (slug {model_slug})", cli.model);
        println!("tasks-dir    : {}", cli.tasks_dir.display());
        println!("task         : {}", cli.task);
        println!("dry-run      : {}", cli.dry_run);
        println!();

        let task_dir = cli.tasks_dir.join(&cli.task);
        if !task_dir.is_dir() {
            return Err(RunnerError::UnknownTask {
                task_id: cli.task.clone(),
                dir: cli.tasks_dir.clone(),
            }
            .into());
        }
        let meta = parse_meta(&task_dir)?;
        let workers = cli.workers.unwrap_or(meta.subtasks).max(1);
        let timeout = Duration::from_secs(meta.timeout_secs.max(1));
        let per_turn_timeout = Duration::from_secs((meta.timeout_secs / (workers as u64 + 2)).max(60));

        let setup_sh = read_file(&task_dir.join("setup.sh"))?;
        let prompt_md = read_file(&task_dir.join("prompt.md"))?;
        let verify_sh = read_file(&task_dir.join("verify.sh"))?;
        let remote_dir = format!("/tmp/akh-e2e-{}", cli.task);

        // One shared Engine (+ audit ledger) for every arm/worker, so every
        // command is queryable from one place, attributed by
        // ExecEnvelope.correlation_id (blueprint "Shared setup").
        let scratch = std::env::temp_dir().join(format!("akh-w6-{}-{}", cli.task, std::process::id()));
        let engine = Arc::new(Engine::new(EngineConfig {
            data_dir: Some(scratch),
            ..Default::default()
        })?);

        let base_config = base_remote_config(&host, &user, port, &cli.key);

        // One connection reused for setup.sh/verify.sh across both arms
        // (sequential, matching the W3 runner's "one connection per
        // session" posture for the plumbing calls).
        let setup_backend: Arc<dyn Backend> = Arc::new(RemoteBackend::connect(base_config.clone())?);
        println!("PASS  connected to exec1 (setup/verify connection)\n");

        let mm = ModelManagerConfig {
            ollama: OllamaConfig {
                base_url: "https://ollama.com".to_string(),
                model: cli.model.clone(),
                api_key: Some("env:OLLAMA_API_KEY".to_string()),
                timeout_secs: 300,
                ..Default::default()
            },
            system_prompt: Some(SYS_PROMPT.to_string()),
            ..Default::default()
        };
        let mgr = Arc::new(AgentModelManager::new(mm));

        // ===============================================================
        // Control arm: one akh, one turn, the whole composite task.
        // ===============================================================
        println!("--- control arm ---");
        let control_start = Instant::now();

        let (setup_code, setup_stdout, setup_stderr) = upload_and_run(
            setup_backend.as_ref(),
            &PathBuf::from(format!("{remote_dir}/setup.sh")),
            &setup_sh,
            Duration::from_secs(120),
        )?;
        println!("    setup exit={setup_code}");
        for line in setup_stdout.lines().take(10) {
            println!("      | {line}");
        }

        let mut control_notes = Vec::new();
        let mut control_usage = ResourceUsage::none();
        let (control_outcome, control_score, control_wall);

        if setup_code != 0 {
            let note = format!(
                "setup.sh failed (exit {setup_code}): {}",
                setup_stderr.lines().take(3).collect::<Vec<_>>().join(" / ")
            );
            println!("    ABORT: {note}");
            control_outcome = TaskOutcome::Aborted { reason: note.clone() };
            control_score = 0.0;
            control_wall = control_start.elapsed();
            control_notes.push(note);
        } else {
            if !cli.dry_run {
                let control_backend: Arc<dyn Backend> = Arc::new(RemoteBackend::connect(base_config.clone())?);
                let before = mgr.usage();
                let result = drive_turn(
                    Arc::clone(&engine),
                    control_backend,
                    Arc::clone(&mgr),
                    "w6-control",
                    &prompt_md,
                    timeout,
                    Arc::new(TraceSink::new("control")),
                )?;
                let after = mgr.usage();
                control_usage = usage_delta(before, after);
                println!("    [w6:control] turn wall={:.1}s", result.wall.as_secs_f64());
                if !result.completed_in_time {
                    control_notes.push("turn exceeded timeout_secs — best-effort abort".to_string());
                } else if !result.turn_ok {
                    control_notes.push("no usable chat backend (Ollama unreachable / model unavailable)".to_string());
                }
            } else {
                control_notes.push("dry-run: control turn not driven (no cloud call)".to_string());
            }

            let (verify_code, verify_stdout, _verify_stderr) = upload_and_run(
                setup_backend.as_ref(),
                &PathBuf::from(format!("{remote_dir}/verify.sh")),
                &verify_sh,
                Duration::from_secs(60),
            )?;
            println!("    control verify exit={verify_code}");
            for line in verify_stdout.lines().take(10) {
                println!("      | {line}");
            }
            control_notes.push(format!(
                "verify: {}",
                verify_stdout.lines().next_back().unwrap_or("(no output)")
            ));

            (control_outcome, control_score) = if cli.dry_run {
                (TaskOutcome::Aborted { reason: "dry-run: model not driven".to_string() }, 0.0)
            } else if verify_code == 0 {
                (TaskOutcome::Pass, 1.0)
            } else {
                (TaskOutcome::Fail, 0.0)
            };
            control_wall = control_start.elapsed();
        }

        println!("    control wall={:.1}s\n", control_wall.as_secs_f64());

        // ===============================================================
        // Orchestrated arm: reset, decompose, fan out, join, verify.
        // ===============================================================
        println!("--- orchestrated arm (reset) ---");
        let (reset_code, reset_stdout, reset_stderr) = upload_and_run(
            setup_backend.as_ref(),
            &PathBuf::from(format!("{remote_dir}/setup.sh")),
            &setup_sh,
            Duration::from_secs(120),
        )?;
        println!("    reset-setup exit={reset_code}");
        for line in reset_stdout.lines().take(10) {
            println!("      | {line}");
        }

        let mut orch_notes = Vec::new();
        let mut orch_usage = ResourceUsage::none();
        let orch_start = Instant::now();
        let mut worker_correlation_ids: Vec<String> = Vec::new();

        if reset_code != 0 {
            let note = format!(
                "reset setup.sh failed (exit {reset_code}): {}",
                reset_stderr.lines().take(3).collect::<Vec<_>>().join(" / ")
            );
            println!("    ABORT: {note}");
            let orch_wall = orch_start.elapsed();
            task_results_push_and_finish(
                &cli,
                &meta,
                &host,
                &engine,
                &mgr,
                model_slug.clone(),
                control_outcome,
                control_score,
                control_wall,
                control_usage,
                control_notes,
                TaskOutcome::Aborted { reason: note.clone() },
                0.0,
                orch_wall,
                ResourceUsage::none(),
                vec![note],
            )?;
            return Ok(());
        }

        // --- decompose ---
        let (decompose_prompts, decompose_source_note): (Vec<(String, String)>, String) = if cli.dry_run {
            (
                meta.subtask.iter().map(|s| (s.id.clone(), s.prompt.clone())).collect(),
                "dry-run: decomposition skipped (no cloud call), using meta.toml [[subtask]] list".to_string(),
            )
        } else {
            let decompose_backend: Arc<dyn Backend> = Arc::new(RemoteBackend::connect(base_config.clone())?);
            let decompose_prompt = format!("{prompt_md}{DECOMPOSE_SUFFIX}");
            let sink = Arc::new(VecSink::new());
            let before = mgr.usage();
            let result = drive_turn(
                Arc::clone(&engine),
                decompose_backend,
                Arc::clone(&mgr),
                "w6-decompose",
                &decompose_prompt,
                per_turn_timeout,
                sink.clone() as Arc<dyn MessageSink>,
            )?;
            let after = mgr.usage();
            let decompose_usage = usage_delta(before, after);
            orch_usage = add_usage(orch_usage, decompose_usage);
            println!("    [w6:decompose] turn wall={:.1}s", result.wall.as_secs_f64());
            orch_notes.push(format!(
                "coordination tax: decompose {} tok / {} req",
                decompose_usage.tokens_used.unwrap_or(0),
                decompose_usage.requests_used.unwrap_or(0)
            ));

            let narrative: String = sink
                .messages()
                .into_iter()
                .filter_map(|m| match m {
                    AkhMessage::Narrative { text, .. } => Some(text),
                    _ => None,
                })
                .collect();

            if !result.completed_in_time || !result.turn_ok {
                (
                    meta.subtask.iter().map(|s| (s.id.clone(), s.prompt.clone())).collect(),
                    format!(
                        "decomposition: orchestrator turn did not complete usably (completed_in_time={}, turn_ok={}) — fell back to meta.toml [[subtask]] list",
                        result.completed_in_time, result.turn_ok
                    ),
                )
            } else {
                match extract_json_string_array(&narrative) {
                    Some(prompts) if prompts.len() == meta.subtask.len() && prompts.iter().all(|p| !p.trim().is_empty()) => {
                        let paired = meta
                            .subtask
                            .iter()
                            .zip(prompts.into_iter())
                            .map(|(spec, prompt)| (spec.id.clone(), prompt))
                            .collect();
                        (paired, "decomposition: used orchestrator's own JSON subtask split".to_string())
                    }
                    Some(prompts) => (
                        meta.subtask.iter().map(|s| (s.id.clone(), s.prompt.clone())).collect(),
                        format!(
                            "decomposition: orchestrator produced {} subtasks, expected {} — fell back to meta.toml [[subtask]] list",
                            prompts.len(),
                            meta.subtask.len()
                        ),
                    ),
                    None => (
                        meta.subtask.iter().map(|s| (s.id.clone(), s.prompt.clone())).collect(),
                        "decomposition: could not parse a JSON subtask array from the orchestrator's answer — fell back to meta.toml [[subtask]] list".to_string(),
                    ),
                }
            }
        };
        println!("    {decompose_source_note}");
        orch_notes.push(decompose_source_note);

        // --- fan out ---
        let n = workers.min(decompose_prompts.len().max(1));
        println!("    fanning out {n} workers");
        let fanout_start = Instant::now();
        let before_workers = mgr.usage();

        let mut handles: Vec<std::thread::JoinHandle<miette::Result<()>>> = Vec::new();
        for (subtask_id, subtask_prompt) in decompose_prompts.into_iter().take(n) {
            let worker_backend: Arc<dyn Backend> = Arc::new(RemoteBackend::connect(base_config.clone())?);
            worker_correlation_ids.push(worker_backend.locus().correlation_id.clone().unwrap_or_default());

            let engine_c = Arc::clone(&engine);
            let mgr_c = Arc::clone(&mgr);
            let dry_run = cli.dry_run;
            let agent_id = format!("w6-worker-{subtask_id}");
            let timeout_c = per_turn_timeout;

            let handle = std::thread::spawn(move || -> miette::Result<()> {
                if dry_run {
                    dry_run_worker_exec(&engine_c, worker_backend, &subtask_id)
                } else {
                    let sink = Arc::new(TraceSink::new(format!("worker:{subtask_id}")));
                    let result = drive_turn(
                        engine_c,
                        worker_backend,
                        mgr_c,
                        &agent_id,
                        &subtask_prompt,
                        timeout_c,
                        sink,
                    )?;
                    println!("    [w6:{subtask_id}] worker turn wall={:.1}s", result.wall.as_secs_f64());
                    if !result.completed_in_time {
                        println!("    [w6:{subtask_id}] worker turn exceeded per-turn timeout — best-effort abort");
                    } else if !result.turn_ok {
                        println!("    [w6:{subtask_id}] worker turn had no usable chat backend");
                    }
                    Ok(())
                }
            });
            handles.push(handle);
        }

        let mut fanout_errors = Vec::new();
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => fanout_errors.push(format!("worker {i}: {e}")),
                Err(_) => fanout_errors.push(format!("worker {i}: panicked")),
            }
        }
        println!("    worker correlation_ids: {worker_correlation_ids:?}");
        let fanout_wall = fanout_start.elapsed();
        let after_workers = mgr.usage();
        let worker_usage = usage_delta(before_workers, after_workers);
        orch_usage = add_usage(orch_usage, worker_usage);

        if !fanout_errors.is_empty() {
            for e in &fanout_errors {
                println!("    [w6] FAN-OUT ERROR: {e}");
            }
            orch_notes.push(format!("fan-out errors: {}", fanout_errors.join(" | ")));
        }
        println!(
            "    fan-out wall={:.1}s ({} tok / {} req across workers)",
            fanout_wall.as_secs_f64(),
            worker_usage.tokens_used.unwrap_or(0),
            worker_usage.requests_used.unwrap_or(0)
        );

        // --- redundant-command ratio (from the shared ledger) ---
        let (redundant, total_worker_cmds) = redundant_command_ratio(&engine, &worker_correlation_ids);
        orch_notes.push(format!("redundant commands: {redundant}/{total_worker_cmds}"));

        // --- composite verify ---
        let (verify_code, verify_stdout, _verify_stderr) = upload_and_run(
            setup_backend.as_ref(),
            &PathBuf::from(format!("{remote_dir}/verify.sh")),
            &verify_sh,
            Duration::from_secs(60),
        )?;
        println!("    orchestrated verify exit={verify_code}");
        for line in verify_stdout.lines().take(10) {
            println!("      | {line}");
        }
        orch_notes.push(format!(
            "verify: {}",
            verify_stdout.lines().next_back().unwrap_or("(no output)")
        ));

        let orch_wall = orch_start.elapsed();
        let (orch_outcome, orch_score) = if cli.dry_run {
            (TaskOutcome::Aborted { reason: "dry-run: model not driven".to_string() }, 0.0)
        } else if verify_code == 0 && fanout_errors.is_empty() {
            (TaskOutcome::Pass, 1.0)
        } else {
            (TaskOutcome::Fail, 0.0)
        };

        // --- speedup ---
        if !cli.dry_run {
            let speedup = if orch_wall.as_secs_f64() > 0.0 {
                control_wall.as_secs_f64() / orch_wall.as_secs_f64()
            } else {
                0.0
            };
            orch_notes.insert(
                0,
                format!(
                    "speedup={speedup:.2} (control {:.1}s / orchestrated {:.1}s)",
                    control_wall.as_secs_f64(),
                    orch_wall.as_secs_f64()
                ),
            );
        } else {
            orch_notes.insert(0, "dry-run: speedup n/a (no cloud driven)".to_string());
        }

        println!("    orchestrated wall={:.1}s\n", orch_wall.as_secs_f64());

        task_results_push_and_finish(
            &cli,
            &meta,
            &host,
            &engine,
            &mgr,
            model_slug,
            control_outcome,
            control_score,
            control_wall,
            control_usage,
            control_notes,
            orch_outcome,
            orch_score,
            orch_wall,
            orch_usage,
            orch_notes,
        )
    }

    /// Assembles both arms' `TaskResult` rows into one `RunRecord` and
    /// writes it — factored out so the early-abort path (reset setup.sh
    /// failure) and the normal completion path share one writer.
    #[allow(clippy::too_many_arguments)]
    fn task_results_push_and_finish(
        cli: &Cli,
        meta: &TaskMeta,
        _host: &str,
        engine: &Arc<Engine>,
        _mgr: &Arc<AgentModelManager>,
        model_slug: String,
        control_outcome: TaskOutcome,
        control_score: f64,
        control_wall: Duration,
        control_usage: ResourceUsage,
        control_notes: Vec<String>,
        orch_outcome: TaskOutcome,
        orch_score: f64,
        orch_wall: Duration,
        orch_usage: ResourceUsage,
        orch_notes: Vec<String>,
    ) -> miette::Result<()> {
        let _ = &engine; // audit ledger already read; kept for signature symmetry with callers

        let control_result = TaskResult::new(
            TaskId::new(format!("{}-control", meta.id))?,
            control_outcome,
            control_score,
            WallTimeSecs::from_duration(control_wall),
            control_usage,
            Some(control_notes.join(" | ")),
        )?;
        let orch_result = TaskResult::new(
            TaskId::new(format!("{}-orchestrated", meta.id))?,
            orch_outcome,
            orch_score,
            WallTimeSecs::from_duration(orch_wall),
            orch_usage,
            Some(orch_notes.join(" | ")),
        )?;

        let repo_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let harness_git_sha = GitSha::from_repo(repo_dir)?;
        let task_set_hash = hash_task_set_dir(&cli.tasks_dir)?;

        let recipe_path = repo_dir.join(RECIPE_SCRIPT_REL);
        let recipe_bytes = std::fs::read(&recipe_path).map_err(|source| RunnerError::Io {
            path: recipe_path.clone(),
            source,
        })?;
        let recipe_hash = VmRecipeHash::from_blake3(Blake3Hex::from_hash(blake3::hash(&recipe_bytes)));

        let record = RunRecord::new(
            chrono::Utc::now(),
            HarnessId::new("fleet-orchestration")?,
            harness_git_sha,
            task_set_hash,
            ModelIdentity {
                tag: ModelTag::new(model_slug)?,
                backing: ModelBacking::Cloud {
                    provider_tag: cli.model.clone(),
                },
            },
            GradingMode::Autonomous,
            HostTier::Vm(VmImageIdentity {
                registry: VM_REGISTRY.to_string(),
                recipe_hash,
            }),
            vec![control_result, orch_result],
        );

        let dir = harness_core::store::write_run(&cli.results, &record)?;
        let summary = record.summary();
        println!(
            "\n=== SUMMARY: {}/{} passed, mean score {:.3} ===",
            summary.passed, summary.total_tasks, summary.mean_score
        );
        println!("wrote {}", dir.display());

        Ok(())
    }
}
