//! Phase 50 / W3 runner: drives an akh through the Phase 48c `RemoteBackend`
//! against the live illumos `exec1` guest, running the `illumos-e2e` task
//! corpus (`benchmarks/illumos-e2e/tasks/`) one task at a time, and emits a
//! typed `harness_core::RunRecord`.
//!
//! This is a straight implementation of the blueprint at
//! `docs/ai/2026-07-07-phase50-w3-runner-design.md` — see that doc for the
//! full rationale (wiring chain, per-task loop, RunRecord provenance,
//! grading contract). ADR 086 §2 (feature-test VM harness), ADR 082 (Backend
//! seam), ADR 082 §7 / Phase 48f (audit envelope).
//!
//! ## Run it
//!
//! ```sh
//! export CARGO_TARGET_DIR=/home/toasty/ws/agi/akh-medu/target
//! cargo build --features remote-executor --example phase50_w3_runner
//!
//! # Smoke: one task, no model driven (proves executor + grading plumbing only).
//! cargo run --features remote-executor --example phase50_w3_runner -- \
//!   --task smf-01-clear-maintenance --dry-run \
//!   --results /tmp/w3-dryrun-results
//!
//! # Real (spends cloud budget!): one model, one task.
//! export OLLAMA_API_KEY="$OLLAMA_CLOUD_KEY"
//! cargo run --features remote-executor --example phase50_w3_runner -- \
//!   --task smf-01-clear-maintenance --model glm-5.2
//! ```
//!
//! Without `--task`, every task directory under `--tasks-dir` runs. Without
//! `--dry-run`, the akh is actually driven against the configured cloud model
//! — this spends real API budget, so prefer an explicit `--task` smoke first.

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
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use akh_medu::agent::chat_backend::OllamaChatBackend;
    use akh_medu::agent::model_manager::{AgentModelManager, ModelManagerConfig};
    use akh_medu::agent::ollama::OllamaConfig;
    use akh_medu::agent::{Agent, AgentConfig};
    use akh_medu::audit::AuditKind;
    use akh_medu::backend::remote::{HostKeyPolicy, RemoteBackend, RemoteBackendConfig};
    use akh_medu::backend::{Backend, ExecRequest};
    use akh_medu::engine::{Engine, EngineConfig};
    use akh_medu::message::{AkhMessage, MessageSink};

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
    const DEFAULT_KEY_PATH: &str =
        "/home/toasty/ws/agi/akh-medu/tests/vm/.keys/root_ed25519";

    /// The VM image recipe exec1 was baked from (Phase 50 W1d) — hashed for
    /// `VmRecipeHash` so a run record names *which* guest image it targeted.
    const RECIPE_SCRIPT_REL: &str = "tests/vm/scripts/bake-illumos-fixed-image.sh";

    /// Registry the base image (`openindiana-rust-ci`) was pulled from —
    /// see `tests/vm/VMFile.kdl`.
    const VM_REGISTRY: &str = "code.aopc.cloud/CloudNebulaProject";

    #[derive(Parser)]
    #[command(
        name = "phase50-w3-runner",
        about = "Phase 50 / W3: drive an akh through the illumos-e2e corpus over the 48c RemoteBackend"
    )]
    struct Cli {
        /// Root directory of the task corpus.
        #[arg(long, default_value = "benchmarks/illumos-e2e/tasks")]
        tasks_dir: PathBuf,

        /// Task id to run (repeatable). Default: every task under `--tasks-dir`.
        #[arg(long = "task")]
        task: Vec<String>,

        /// Provider tag sent to the cloud API, e.g. `deepseek-v4-pro` /
        /// `glm-5.2` / `qwen3.5:397b`.
        #[arg(long, default_value = "glm-5.2")]
        model: String,

        /// Filesystem-safe grouping slug for the results directory. Defaults
        /// to a sanitized form of `--model` (lowercased, `:` and other
        /// non-slug characters replaced with `-`, truncated at the first
        /// `:` — e.g. `qwen3.5:397b` -> `qwen3.5`).
        #[arg(long)]
        model_slug: Option<String>,

        /// exec1 SSH target as `user@host:port`.
        #[arg(long, default_value = "root@127.0.0.1:10042")]
        exec1: String,

        /// Path to the exec1 private key.
        #[arg(long, default_value = DEFAULT_KEY_PATH)]
        key: PathBuf,

        /// Where `benchmarks/results/` (or an alternate results root) lives.
        #[arg(long, default_value = "benchmarks/results")]
        results: PathBuf,

        /// Run the full setup/verify/RunRecord plumbing but never drive the
        /// model. Proves the executor + grading path without spending any
        /// cloud budget; every task's outcome is `Aborted` by construction.
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
        timeout_secs: u64,
        expected_manpages: Vec<String>,
        #[allow(dead_code)]
        mode: Vec<String>,
        #[allow(dead_code)]
        version_trap: VersionTrap,
    }

    #[derive(Debug, Deserialize)]
    struct VersionTrap {
        #[allow(dead_code)]
        present: bool,
        #[serde(default)]
        #[allow(dead_code)]
        note: Option<String>,
    }

    // -----------------------------------------------------------------
    // Runner-local errors (library errors below this all carry their own
    // miette diagnostics and propagate via `?` unchanged)
    // -----------------------------------------------------------------

    #[derive(Debug, Error, Diagnostic)]
    enum RunnerError {
        #[error("--exec1 must be user@host:port, got {value:?}")]
        #[diagnostic(
            code(phase50_w3::bad_exec1_spec),
            help("Example: root@127.0.0.1:10042")
        )]
        BadExec1Spec { value: String },

        #[error("bad port in --exec1 ({value:?}): {source}")]
        #[diagnostic(code(phase50_w3::bad_port))]
        BadPort {
            value: String,
            #[source]
            source: std::num::ParseIntError,
        },

        #[error("no task directory {task_id:?} under {dir}")]
        #[diagnostic(
            code(phase50_w3::unknown_task),
            help("Check the id matches a directory name under --tasks-dir.")
        )]
        UnknownTask { task_id: String, dir: PathBuf },

        #[error("failed to read {path}: {source}")]
        #[diagnostic(code(phase50_w3::io))]
        Io {
            path: PathBuf,
            #[source]
            source: std::io::Error,
        },

        #[error("failed to parse {path} as TOML: {source}")]
        #[diagnostic(
            code(phase50_w3::meta_parse),
            help("meta.toml must match the shape documented in benchmarks/illumos-e2e/README.md.")
        )]
        MetaParse {
            path: PathBuf,
            #[source]
            source: toml::de::Error,
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
        toml::from_str(&text).map_err(|source| RunnerError::MetaParse { path, source })
    }

    /// Derive a filesystem-safe model-tag slug from a provider tag when the
    /// caller doesn't pass `--model-slug` explicitly: take the part before
    /// the first `:` (Ollama tag separator), lowercase, and replace any
    /// character `ModelTag` would reject with `-`.
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

    /// Parse `user@host:port` (the same shape `phase50_remote_exec1.rs` uses).
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
    // Message sink (copied from examples/agentic_flow.rs)
    // -----------------------------------------------------------------

    struct TraceSink;
    impl MessageSink for TraceSink {
        fn emit(&self, msg: &AkhMessage) {
            match msg {
                AkhMessage::Reasoning { step, .. } => println!("    [think] {step}"),
                AkhMessage::ToolResult { tool, success, output } => {
                    let mark = if *success { "ok" } else { "FAIL" };
                    println!("    [tool {mark} {tool}] {}", output.replace('\n', " "));
                }
                AkhMessage::Narrative { text, .. } => println!("    [answer] {text}"),
                AkhMessage::Fact { text, .. } => println!("    [fact] {text}"),
                AkhMessage::System { text } => println!("    [system] {text}"),
                AkhMessage::Error { code, message, .. } => {
                    println!("    [error {code}] {message}")
                }
                other => println!("    [msg] {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Remote plumbing helpers
    // -----------------------------------------------------------------

    /// Upload `content` to `remote_path` and run it via `sh`, returning
    /// `(exit_code, stdout, stderr)`. Used for both `setup.sh` and
    /// `verify.sh` — same upload-then-exec shape, different script.
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

    /// Heuristic manpage-grounding check over one `shell_exec` invocation's
    /// (Debug-formatted, 200-char-truncated) `params_summary`. Not exact —
    /// the audit ledger only carries a truncated Debug rendering of the
    /// tool's JSON params, not the raw shell string — but a sound
    /// non-gating signal per the blueprint (48f's `ExecEnvelope` makes the
    /// *host* attribution airtight; this makes no claim about *which*
    /// command ran it beyond substring evidence).
    fn mentions_manpage_lookup(params_summary: &str, page: &str) -> bool {
        let lower = params_summary.to_ascii_lowercase();
        let page_lower = page.to_ascii_lowercase();
        let has_lookup_verb = lower.contains("man ") || lower.contains("man\\\"") || lower.contains("apropos");
        if !has_lookup_verb {
            return false;
        }
        lower
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-')
            .any(|tok| tok == page_lower)
    }

    /// Grade manpage grounding for one task from the audit ledger's
    /// `ToolInvocation` entries. Returns a human-readable note; never gates
    /// pass/fail (ADR 086 §2).
    fn grade_manpages(engine: &Engine, expected: &[String]) -> String {
        let Some(ledger) = engine.audit_ledger() else {
            return "manpages: no audit ledger attached (engine had no data_dir)".to_string();
        };
        // ToolInvocation is kind_tag 0 (AuditKind::tag).
        let page = match ledger.list_by_kind(0, 0, 10_000) {
            Ok(page) => page,
            Err(e) => return format!("manpages: audit ledger query failed: {e}"),
        };

        let mut on_exec1 = false;
        let mut grounded: Vec<&str> = Vec::new();
        let mut missing: Vec<&str> = Vec::new();
        for expected_page in expected {
            let hit = page.entries.iter().any(|entry| {
                let AuditKind::ToolInvocation {
                    tool_name,
                    params_summary,
                    envelope,
                    ..
                } = &entry.kind
                else {
                    return false;
                };
                if tool_name != "shell_exec" {
                    return false;
                }
                if envelope.backend.as_deref() == Some("remote") {
                    on_exec1 = true;
                }
                mentions_manpage_lookup(params_summary, expected_page)
            });
            if hit {
                grounded.push(expected_page.as_str());
            } else {
                missing.push(expected_page.as_str());
            }
        }

        let locus_note = if on_exec1 {
            "commands confirmed via ExecEnvelope on the remote backend"
        } else {
            "no ExecEnvelope{backend:\"remote\"} seen — grounding is command-text-only"
        };
        if missing.is_empty() {
            format!(
                "manpages: {}/{} grounded ({locus_note})",
                grounded.len(),
                expected.len()
            )
        } else {
            format!(
                "manpages: {}/{} grounded, missing [{}] ({locus_note})",
                grounded.len(),
                expected.len(),
                missing.join(", ")
            )
        }
    }

    // -----------------------------------------------------------------
    // Driving the akh (blueprint's exact wiring chain)
    // -----------------------------------------------------------------

    const SYS_PROMPT: &str = "You are an akh: a precise illumos system engineer with root access \
        on the target host over a remote shell. Use `man`/`apropos` to ground unfamiliar commands \
        in the local manual pages before acting — this host is OpenIndiana Hipster, not Linux; \
        Linux-only tools and flags do not exist here. Diagnose before you act, verify your fix left \
        the system in the requested end state, and prefer the documented illumos-native command \
        over a habit carried over from another OS.";

    /// Drive one turn to completion or until `timeout` elapses, whichever
    /// comes first. `run_agentic_turn` is a synchronous, non-cancellable
    /// call, so the cap here is a best-effort join-with-timeout: on overrun
    /// the driving thread is detached (left to finish or hang on its own)
    /// rather than the process blocking on it — this is *not* true
    /// preemption, just a bound on how long the caller waits.
    fn drive_turn(
        engine: Arc<Engine>,
        backend: Arc<dyn Backend>,
        mgr: Arc<AgentModelManager>,
        task_id: &str,
        prompt: &str,
        timeout: Duration,
    ) -> miette::Result<(bool, bool)> {
        let mut agent = Agent::with_backend(Arc::clone(&engine), AgentConfig::default(), Some(backend))?;
        agent.set_chat_backend(Arc::new(OllamaChatBackend::new(Arc::clone(&mgr))));

        let (tx, rx) = std::sync::mpsc::channel();
        let agent_id = format!("w3-{task_id}");
        let prompt = prompt.to_string();
        let engine_for_thread = Arc::clone(&engine);
        std::thread::spawn(move || {
            let outcome = agent.run_agentic_turn(&engine_for_thread, &agent_id, &prompt, &TraceSink);
            let _ = tx.send(outcome.is_some());
        });

        match rx.recv_timeout(timeout) {
            Ok(completed) => Ok((true, completed)),
            Err(_) => {
                println!(
                    "    [w3] turn did not complete within {}s — abandoning the driving thread (best-effort cap, not preemptive)",
                    timeout.as_secs()
                );
                Ok((false, false))
            }
        }
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

        println!("=== Phase 50 / W3 runner ===");
        println!("exec1        : {user}@{host}:{port}");
        println!("key          : {}", cli.key.display());
        println!("model        : {} (slug {model_slug})", cli.model);
        println!("tasks-dir    : {}", cli.tasks_dir.display());
        println!("dry-run      : {}", cli.dry_run);
        println!();

        // Task selection: explicit --task list, or every directory under
        // --tasks-dir (sorted for determinism).
        let task_ids: Vec<String> = if !cli.task.is_empty() {
            for t in &cli.task {
                let dir = cli.tasks_dir.join(t);
                if !dir.is_dir() {
                    return Err(RunnerError::UnknownTask {
                        task_id: t.clone(),
                        dir: cli.tasks_dir.clone(),
                    }
                    .into());
                }
            }
            cli.task.clone()
        } else {
            let mut ids = Vec::new();
            for entry in std::fs::read_dir(&cli.tasks_dir).map_err(|source| RunnerError::Io {
                path: cli.tasks_dir.clone(),
                source,
            })? {
                let entry = entry.map_err(|source| RunnerError::Io {
                    path: cli.tasks_dir.clone(),
                    source,
                })?;
                if entry.path().is_dir() {
                    ids.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
            ids.sort();
            ids
        };

        // Connect once; every task's setup/drive/verify rides the same SSH
        // session (ADR 082 §5: one connection per session).
        let config = RemoteBackendConfig::new(host.clone(), user.clone(), cli.key.clone())
            .with_port(port)
            // Lab/loopback posture (ADR 086 §1): exec1 is a disposable VM
            // reachable only via a localhost port-forward for this session.
            // `InsecureAcceptAny` is the real, working policy variant —
            // there is no `AcceptNew` on `HostKeyPolicy` (only `KnownHosts`,
            // `Pinned`, `InsecureAcceptAny`); copied verbatim from the
            // validated `phase50_remote_exec1.rs` construction rather than
            // the blueprint's placeholder.
            .with_host_key_policy(HostKeyPolicy::InsecureAcceptAny)
            .with_roots(vec![PathBuf::from("/")], vec![
                PathBuf::from("/tmp"),
                PathBuf::from("/opt"),
                PathBuf::from("/var"),
            ])
            .with_connect_timeout(Duration::from_secs(15));
        let backend: Arc<dyn Backend> = Arc::new(RemoteBackend::connect(config)?);
        println!("PASS  connected to exec1\n");

        let mut task_results: Vec<TaskResult> = Vec::new();

        for task_id in &task_ids {
            println!("--- task {task_id} ---");
            let task_dir = cli.tasks_dir.join(task_id);
            let meta = parse_meta(&task_dir)?;
            let remote_dir = format!("/tmp/akh-e2e-{task_id}");
            let timeout = Duration::from_secs(meta.timeout_secs.max(1));

            let task_start = Instant::now();

            // 1. setup
            let setup_sh = read_file(&task_dir.join("setup.sh"))?;
            let (setup_code, setup_stdout, setup_stderr) = upload_and_run(
                backend.as_ref(),
                &PathBuf::from(format!("{remote_dir}/setup.sh")),
                &setup_sh,
                Duration::from_secs(120),
            )?;
            println!("    setup exit={setup_code}");
            for line in setup_stdout.lines().take(10) {
                println!("      | {line}");
            }
            if setup_code != 0 {
                let note = format!(
                    "setup.sh failed (exit {setup_code}): {}",
                    setup_stderr.lines().take(3).collect::<Vec<_>>().join(" / ")
                );
                println!("    ABORT: {note}");
                task_results.push(TaskResult::new(
                    TaskId::new(task_id.clone())?,
                    TaskOutcome::Aborted { reason: note.clone() },
                    0.0,
                    WallTimeSecs::from_duration(task_start.elapsed()),
                    ResourceUsage::none(),
                    Some(note),
                )?);
                continue;
            }

            // 2. drive (skipped entirely in --dry-run)
            let prompt_md = read_file(&task_dir.join("prompt.md"))?;
            let mut usage = ResourceUsage::none();
            let mut manpage_note: Option<String> = None;
            let mut drive_note: Option<String> = None;

            if !cli.dry_run {
                let scratch = std::env::temp_dir().join(format!("akh-w3-{task_id}-{}", std::process::id()));
                let engine = Arc::new(Engine::new(EngineConfig {
                    data_dir: Some(scratch),
                    ..Default::default()
                })?);

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

                let (completed_in_time, turn_ok) = drive_turn(
                    Arc::clone(&engine),
                    Arc::clone(&backend),
                    Arc::clone(&mgr),
                    task_id,
                    &prompt_md,
                    timeout,
                )?;
                let snapshot = mgr.usage();
                usage = ResourceUsage {
                    tokens_used: Some(snapshot.total_tokens()),
                    requests_used: Some(snapshot.turns),
                };
                if !completed_in_time {
                    drive_note = Some("turn exceeded timeout_secs — best-effort abort".to_string());
                } else if !turn_ok {
                    drive_note = Some("no usable chat backend (Ollama unreachable / model unavailable)".to_string());
                }
                manpage_note = Some(grade_manpages(&engine, &meta.expected_manpages));
            }

            // 3. verify (uploaded fresh, after the turn ends — the akh never sees it)
            let verify_sh = read_file(&task_dir.join("verify.sh"))?;
            let (verify_code, verify_stdout, _verify_stderr) = upload_and_run(
                backend.as_ref(),
                &PathBuf::from(format!("{remote_dir}/verify.sh")),
                &verify_sh,
                Duration::from_secs(60),
            )?;
            println!("    verify exit={verify_code}");
            for line in verify_stdout.lines().take(10) {
                println!("      | {line}");
            }

            let (outcome, score) = if cli.dry_run {
                (
                    TaskOutcome::Aborted {
                        reason: "dry-run: model not driven".to_string(),
                    },
                    0.0,
                )
            } else if verify_code == 0 {
                (TaskOutcome::Pass, 1.0)
            } else {
                (TaskOutcome::Fail, 0.0)
            };

            let mut notes = Vec::new();
            notes.push(format!(
                "verify: {}",
                verify_stdout.lines().next_back().unwrap_or("(no output)")
            ));
            if let Some(n) = drive_note {
                notes.push(n);
            }
            if let Some(n) = manpage_note {
                notes.push(n);
            }
            if cli.dry_run {
                notes.push("dry-run: setup+verify plumbing exercised, model not driven".to_string());
            }

            task_results.push(TaskResult::new(
                TaskId::new(task_id.clone())?,
                outcome,
                score,
                WallTimeSecs::from_duration(task_start.elapsed()),
                usage,
                Some(notes.join(" | ")),
            )?);
        }

        // -----------------------------------------------------------------
        // RunRecord provenance + write
        // -----------------------------------------------------------------

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
            HarnessId::new("illumos-e2e")?,
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
            task_results,
        );

        let dir = harness_core::store::write_run(&cli.results, &record)?;
        let summary = record.summary();
        println!("\n=== SUMMARY: {}/{} passed, mean score {:.3} ===", summary.passed, summary.total_tasks, summary.mean_score);
        println!("wrote {}", dir.display());

        Ok(())
    }
}
