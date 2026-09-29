//! LLM-driven agentic chat turn: the akh's pinned model (Ollama, local or
//! cloud) answers a prompt, calling registered tools in a loop until it produces
//! a final answer.
//!
//! This is the live half of the "first-class akh" picture (ADR-053 / the parked
//! P2 work): [`ChatProcessor`](crate::chat::ChatProcessor) is the *symbolic*
//! responder; this is the *neural* one. [`Agent::run_agentic_turn`] is called
//! from the ACP `session/prompt` handler and falls back to the symbolic pipeline
//! when no model backend is reachable, so a daemon with no Ollama configured
//! behaves exactly as before.
//!
//! ## Concurrency
//!
//! The whole turn is synchronous — blocking Ollama HTTP (`ureq`) interleaved
//! with synchronous tool execution. ACP callers run it inside
//! [`tokio::task::spawn_blocking`] under the workspace Agent's read lock
//! (ADR-027); mutable neural transcript state is supplied separately and its
//! mutex is held only for brief reads/writes, never across model inference or
//! an `.await`. ACP installs the per-turn tool/permission/tracker scope before
//! entering this loop so `Ask` tools can bridge to the correct client.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::Agent;
use super::conversation::ConversationState;
use super::ollama::StreamEvent;
use super::tool::{ToolInput, ToolRegistry};
use crate::engine::Engine;
use crate::message::{AkhMessage, MessageSink};

/// Iteration backstop for a small / unknown-context model (e.g. a constrained
/// in-process Candle akh). Frontier agentic harnesses (OpenHands, SWE-agent) set
/// the iteration cap *generously* — ~100 — as a pure backstop and rely on
/// loop-detection + time/cost budgets to terminate normally; we scale ours to
/// model capacity (see [`RoundBudget::new`]) and keep it env-overridable. A flat
/// low cap (the prior `8`) is wrong in both directions: it strangles legitimate
/// deep work (recall → drill into the KG → synthesize) while doing nothing about
/// a genuine spin.
const DEFAULT_MAX_ROUNDS: usize = 12;

/// Per-turn wall-clock budget. Frontier harnesses pair the iteration cap with a
/// time cap (~300s is the common production value); override with
/// `AKH_AGENT_TURN_TIMEOUT_SECS`.
const DEFAULT_TURN_TIMEOUT_SECS: u64 = 300;

/// The same `(tool, arguments)` call issued this many times in one turn marks an
/// unproductive loop and ends it — the "same call N times" heuristic frontier
/// harnesses use to break a spinning agent. Override with `AKH_AGENT_REPEAT_LIMIT`.
const DEFAULT_REPEAT_LIMIT: usize = 3;

/// Wedge detection: if a turn is active but no LLM request has been issued for
/// this many seconds, end the turn rather than waiting for the runner's 3600s cap.
/// This catches the WI-259 wedge where the model stopped requesting tools but the
/// loop neither completed nor continued. Override with `AKH_AGENT_WEDGE_TIMEOUT_SECS`.
const DEFAULT_WEDGE_TIMEOUT_SECS: u64 = 60;

/// Characters of a tool-call argument blob surfaced in the "calling tool" line.
const ARG_PREVIEW_LEN: usize = 160;

/// Parse a positive-integer env var, ignoring absent/blank/zero/garbage values.
fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// `SystemTime::now()` rendered as an RFC3339 UTC timestamp
/// (`YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`), std-only (no `chrono`).
///
/// The main crate's `chrono` dependency is optional (gated behind the
/// `calendar` feature — see `Cargo.toml`), so unconditionally-compiled code
/// like the turn-capture hook (ADR 091) cannot name it. Everywhere else in the
/// crate that needs a timestamp outside a `chrono`-gated feature uses
/// `SystemTime` directly (see `agent/contact.rs`, `agent/dialogue.rs`, etc.);
/// this just also renders it as RFC3339 for the turn-record JSON payload's
/// `ts` field.
fn rfc3339_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let nanos = now.subsec_nanos();
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.{nanos:09}Z")
}

/// Days-since-epoch → civil (year, month, day), Howard Hinnant's
/// `civil_from_days` algorithm (public-domain; proleptic Gregorian, valid
/// across the full `i64` range). `days` is `0` for 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Why an agentic turn's tool loop ended without the model volunteering a final
/// answer. Each maps to a short clause the forced-synthesis nudge explains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TurnStop {
    /// Hit the absolute iteration backstop.
    IterationCap,
    /// Exceeded the per-turn wall-clock budget.
    Timeout,
    /// Repeated an identical `(tool, args)` call past the repeat limit.
    RepeatLoop,
    /// The caller's cancel flag was set (the ACP idle watchdog gave up on
    /// this turn and already answered the client) — see the `cancel` param
    /// on [`Agent::run_agentic_turn`]. Unlike the other variants, this stop
    /// skips the forced-synthesis round entirely: nobody is waiting on the
    /// answer, and issuing one more model call would only delay releasing
    /// the agent lock that a wedged turn is holding.
    Cancelled,
    /// The model declared completion (no tool calls) but the post-condition
    /// guard detected no work was performed (no mutating tool calls and
    /// clean worktree). Dispatch-only failure mode.
    DeclaredButIdle,
    /// An explicit host-enforced completion contract remained unsatisfied
    /// after the bounded retry budget.
    ContractUnsatisfied,
    /// Wedge detected: the turn has been active but no LLM request has been
    /// issued for longer than the wedge timeout. This catches the WI-259
    /// failure mode where the model stopped requesting tools but the loop
    /// neither completed nor continued.
    WedgeTimeout,
}

/// Protocol-visible outcome of a neural turn.
///
/// `Refused` means the host rejected the model's completion claim (for
/// example, because an explicit evidence contract could not be satisfied).
/// ACP maps it to `StopReason::Refusal` instead of reporting a clean end turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgenticTurnOutcome {
    Completed,
    Refused,
    Interrupted(TurnStop),
}

impl TurnStop {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::IterationCap => "iteration_budget",
            Self::Timeout => "time_budget",
            Self::RepeatLoop => "repeat_loop",
            Self::Cancelled => "cancelled",
            Self::DeclaredButIdle => "no_work",
            Self::ContractUnsatisfied => "contract_unsatisfied",
            Self::WedgeTimeout => "no_progress",
        }
    }

    fn clause(self) -> &'static str {
        match self {
            TurnStop::IterationCap => "You have reached your tool-call limit",
            TurnStop::Timeout => "You have reached your time budget",
            TurnStop::RepeatLoop => "You are repeating the same tool call without progress",
            TurnStop::Cancelled => "This turn was cancelled",
            TurnStop::DeclaredButIdle => {
                "You declared completion but verification shows no work was performed. \
                 No successful file write or changed repository state was verified. \
                 Review the task requirements and use the appropriate tools to make the necessary changes"
            }
            TurnStop::ContractUnsatisfied => {
                "The host-enforced completion contract is still unsatisfied. Report the task as incomplete and name the missing evidence"
            }
            TurnStop::WedgeTimeout => {
                "The turn has been inactive for too long without making progress"
            }
        }
    }
}

/// Why the repository witness is not taken for a deployed agent.
///
/// Phase 53 decision 6 / ADR 113: a deployed akhomed runs no side effect in
/// its own address space. `git` honours repository configuration
/// (`core.fsmonitor`, hooks, filters), so spawning it in akhomed against an
/// agent-edited checkout could execute a repository-chosen command inside
/// akhomed. Without a backend route the witness is skipped instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum WitnessUnavailable {
    /// `exec kind = "none"`: there is no executor to run `git` in.
    #[error("deployed mode with exec kind = \"none\": no executor to run git in")]
    NoExecutor,
    /// Only the in-process backend is wired, so `git` would run in akhomed.
    #[error("deployed mode with the in-process backend: git would run inside akhomed")]
    InProcessOnly,
}

/// Where the autonomous-dispatch completion check runs `git` to witness
/// shell-made repository changes.
#[derive(Clone)]
pub(crate) enum WitnessRoute {
    /// Development: spawn `git` in akhomed (historic behaviour).
    InProcess,
    /// Deployed with a configured executor: run `git` through
    /// [`Backend::exec`](crate::backend::Backend::exec), i.e. in the sandbox.
    Backend(std::sync::Arc<dyn crate::backend::Backend>),
    /// Deployed with no sandboxed route: take no witness.
    Unavailable(WitnessUnavailable),
}

impl std::fmt::Debug for WitnessRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InProcess => f.write_str("InProcess"),
            Self::Backend(backend) => f
                .debug_tuple("Backend")
                .field(&backend.locus().backend)
                .finish(),
            Self::Unavailable(reason) => f.debug_tuple("Unavailable").field(reason).finish(),
        }
    }
}

impl WitnessRoute {
    /// Resolve the route for an agent from its deployment mode and exec wiring.
    pub(crate) fn resolve(
        mode: crate::deployment::DeploymentMode,
        wiring: &super::agent::ExecToolWiring,
    ) -> Self {
        use super::agent::ExecToolWiring;
        if !mode.is_deployed() {
            return Self::InProcess;
        }
        match wiring {
            ExecToolWiring::Backend(backend) => Self::Backend(backend.clone()),
            ExecToolWiring::Default => Self::Unavailable(WitnessUnavailable::InProcessOnly),
            ExecToolWiring::None => Self::Unavailable(WitnessUnavailable::NoExecutor),
        }
    }

    /// Hash the repository state at `cwd`, or `None` when it cannot be
    /// witnessed (not a repository, a `git` failure, or no route).
    pub(crate) fn witness(&self, cwd: &std::path::Path) -> Option<u64> {
        match self {
            Self::InProcess => repository_witness(cwd),
            Self::Backend(backend) => backend_repository_witness(backend.as_ref(), cwd),
            Self::Unavailable(reason) => {
                tracing::debug!(
                    cwd = %cwd.display(),
                    reason = %reason,
                    "repository witness skipped"
                );
                None
            }
        }
    }
}

/// `git` arguments shared by both witness routes. External diff drivers and
/// textconv are disabled so verification cannot execute
/// repository-configured commands through them.
const WITNESS_HEAD: &[&str] = &["rev-parse", "HEAD"];
const WITNESS_DIFF: &[&str] = &["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"];
const WITNESS_UNTRACKED: &[&str] = &["ls-files", "--others", "--exclude-standard", "-z"];

/// Shared witness algorithm: `git` runs through `git`, and `untracked` hashes
/// per-file metadata for each untracked path.
fn witness_with(
    mut git: impl FnMut(&[&str]) -> Option<Vec<u8>>,
    mut untracked: impl FnMut(&str, &mut DefaultHasher) -> Option<()>,
) -> Option<u64> {
    let mut hash = DefaultHasher::new();
    git(WITNESS_HEAD)?.hash(&mut hash);
    git(WITNESS_DIFF)?.hash(&mut hash);
    let names = git(WITNESS_UNTRACKED)?;
    for name in names.split(|b| *b == 0).filter(|name| !name.is_empty()) {
        let name = std::str::from_utf8(name).ok()?;
        name.hash(&mut hash);
        untracked(name, &mut hash)?;
    }
    Some(hash.finish())
}

/// Read-only repository evidence for shell-based edits, taken by spawning
/// `git` in this process. Development mode only; see [`WitnessRoute`].
fn repository_witness(cwd: &std::path::Path) -> Option<u64> {
    witness_with(
        |args| {
            let out = std::process::Command::new("git")
                .args(args)
                .current_dir(cwd)
                .env("GIT_OPTIONAL_LOCKS", "0")
                .output()
                .ok()?;
            out.status.success().then_some(out.stdout)
        },
        |name, hash| {
            // Do not follow symlinks or read arbitrary untracked
            // datasets/devices. Metadata is sufficient to distinguish an
            // untouched checkout from a new or modified candidate; Anima still
            // validates the actual diff.
            let metadata = std::fs::symlink_metadata(cwd.join(name)).ok()?;
            metadata.len().hash(hash);
            metadata.modified().ok()?.hash(hash);
            Some(())
        },
    )
}

/// Deadline for each `git` call of the backend-routed witness.
const BACKEND_WITNESS_TIMEOUT: Duration = Duration::from_secs(30);

/// Read-only repository evidence taken through the configured executor
/// backend, so `git` (and any repository configuration it honours) runs in
/// the sandbox, never in akhomed. Untracked files contribute their name and
/// size as reported by the backend's [`Filesystem`](crate::fs::Filesystem);
/// symlinks are not followed and unreadable entries hash a fixed marker, so
/// additions and removals still change the witness.
fn backend_repository_witness(
    backend: &dyn crate::backend::Backend,
    cwd: &std::path::Path,
) -> Option<u64> {
    witness_with(
        |args| {
            let request = crate::backend::ExecRequest::shell(
                format!("git {}", args.join(" ")),
                BACKEND_WITNESS_TIMEOUT,
            )
            .with_cwd(cwd)
            .with_env("GIT_OPTIONAL_LOCKS", "0");
            match backend.exec(request) {
                Ok(outcome) if outcome.success => Some(outcome.stdout),
                Ok(outcome) => {
                    tracing::debug!(
                        cwd = %cwd.display(),
                        exit_code = outcome.exit_code,
                        "repository witness: git failed in the executor backend"
                    );
                    None
                }
                Err(error) => {
                    tracing::debug!(
                        cwd = %cwd.display(),
                        %error,
                        "repository witness: executor backend exec failed"
                    );
                    None
                }
            }
        },
        |name, hash| {
            let path = cwd.join(name);
            if backend.is_symlink(&path) {
                "symlink".hash(hash);
            } else {
                backend.file_size(&path).unwrap_or(u64::MAX).hash(hash);
            }
            Some(())
        },
    )
}

/// Byte caps for the receipt fields a turn's tool trace carries (ADR 116 C2).
const TRACE_ARGS_MAX_BYTES: usize = 1000;
const TRACE_OUTPUT_MAX_BYTES: usize = 4000;

fn verified_write(tool: &str, args: &Value, success: bool) -> bool {
    success
        && (tool == "edit"
            || (tool == "file_io" && args.get("action").and_then(Value::as_str) == Some("write")))
}

/// Frontier-harness-style budget governing the agentic tool loop — *defense in
/// depth* rather than one fixed round count. It combines:
///   * an **iteration backstop** scaled to the model's context window
///     (`AKH_AGENT_MAX_ROUNDS` overrides absolutely),
///   * a **wall-clock budget** (`AKH_AGENT_TURN_TIMEOUT_SECS`), and
///   * **repeat-loop detection** — the same `(tool, args)` signature issued
///     `repeat_limit` times (`AKH_AGENT_REPEAT_LIMIT`) ends the turn.
///
/// On exhaustion the caller forces one final *tool-free* synthesis (a "reflect
/// and answer" step) instead of emitting a bare cutoff, so a deep turn still
/// answers with what it gathered.
struct RoundBudget {
    hard_max: usize,
    timeout: Duration,
    repeat_limit: usize,
    wedge_timeout: Duration,
    started: Instant,
    round: usize,
    /// Count of each `(tool, args)` signature issued this turn, for spin detection.
    counts: HashMap<u64, usize>,
    looped: bool,
    /// Last time an LLM request was issued. Updated at the start of each round
    /// to detect the WI-259 wedge (loop active but no request for N seconds).
    last_request: Instant,
    /// Count of no-tool-call retries (WI-280): when the model declares completion
    /// without performing work, this tracks how many times we've retried. Prevents
    /// infinite loops when rounds_so_far() stays at 0 because no tools were called.
    no_work_retries: usize,
}

impl RoundBudget {
    /// Scale the iteration backstop to the model's context window, then apply env
    /// overrides. A roomy window (GLM ~1M, gemma4-12b 262k) earns more rounds to
    /// drill through recalled episodes; a small/unknown one stays conservative.
    fn new(context_window: Option<usize>) -> Self {
        let scaled = match context_window {
            Some(w) if w >= 200_000 => 30,
            Some(w) if w >= 32_000 => 20,
            _ => DEFAULT_MAX_ROUNDS,
        };
        let hard_max = env_usize("AKH_AGENT_MAX_ROUNDS").unwrap_or(scaled);
        let timeout = Duration::from_secs(
            env_usize("AKH_AGENT_TURN_TIMEOUT_SECS")
                .map(|n| n as u64)
                .unwrap_or(DEFAULT_TURN_TIMEOUT_SECS),
        );
        let repeat_limit = env_usize("AKH_AGENT_REPEAT_LIMIT").unwrap_or(DEFAULT_REPEAT_LIMIT);
        let wedge_timeout = Duration::from_secs(
            env_usize("AKH_AGENT_WEDGE_TIMEOUT_SECS")
                .map(|n| n as u64)
                .unwrap_or(DEFAULT_WEDGE_TIMEOUT_SECS),
        );
        let now = Instant::now();
        Self {
            hard_max,
            timeout,
            repeat_limit,
            wedge_timeout,
            started: now,
            round: 0,
            counts: HashMap::new(),
            looped: false,
            last_request: now,
            no_work_retries: 0,
        }
    }

    /// Autonomous work has no implicit elapsed-time or round cutoff. Explicit
    /// operator limits remain supported; repetition and cancellation still stop it.
    fn for_dispatch(mut self) -> Self {
        if env_usize("AKH_AGENT_MAX_ROUNDS").is_none() {
            self.hard_max = usize::MAX;
        }
        if env_usize("AKH_AGENT_TURN_TIMEOUT_SECS").is_none() {
            self.timeout = Duration::MAX;
        }
        self
    }

    /// Whether the turn must end before issuing another model call.
    fn exhausted(&self) -> Option<TurnStop> {
        if self.round >= self.hard_max {
            Some(TurnStop::IterationCap)
        } else if self.started.elapsed() >= self.timeout {
            Some(TurnStop::Timeout)
        } else if self.looped {
            Some(TurnStop::RepeatLoop)
        } else if self.last_request.elapsed() >= self.wedge_timeout {
            Some(TurnStop::WedgeTimeout)
        } else {
            None
        }
    }

    /// Record one completed tool round (its `(name, args)` calls), advancing the
    /// counter and flagging a spin once any signature repeats `repeat_limit` times.
    fn note_round<'a>(&mut self, calls: impl IntoIterator<Item = (&'a str, &'a Value)>) {
        self.round += 1;
        for (name, args) in calls {
            let mut h = DefaultHasher::new();
            name.hash(&mut h);
            args.to_string().hash(&mut h);
            let n = self.counts.entry(h.finish()).or_insert(0);
            *n += 1;
            if *n >= self.repeat_limit {
                self.looped = true;
            }
        }
    }

    /// Current round number (for post-condition retry budget).
    fn rounds_so_far(&self) -> usize {
        self.round
    }

    /// Iteration cap (for post-condition retry budget).
    fn max_rounds(&self) -> usize {
        self.hard_max
    }

    /// Mark that an LLM request is being issued, updating the wedge-detection timer.
    fn note_request(&mut self) {
        self.last_request = Instant::now();
    }

    /// Mark that a no-work retry is being attempted (WI-280). Returns true if
    /// the retry budget is exhausted (too many retries without progress).
    fn note_no_work_retry(&mut self) -> bool {
        self.no_work_retries += 1;
        // Allow up to 3 retries before exhausting (reasonable for a model that
        // occasionally hallucinate completion, but not infinite).
        self.no_work_retries > 3
    }
}

/// Result of [`Agent::execute_tool_call`] running one requested tool call
/// through the ADR 089 organ passes: what actually executed (which may
/// differ from the model's request after M2 name repair or task A/B argument
/// repair), whether it succeeded, and how to narrate + trace it.
struct ToolCallOutcome {
    /// The tool name actually executed (post-M2 repair, if any).
    exec_name: String,
    /// The arguments actually executed (post task-A/B repair, if any).
    exec_args: Value,
    success: bool,
    result: String,
    /// Bracketed narration prefix summarizing whichever organ passes fired,
    /// e.g. `"[repaired: fetch_ledgr → fetch_ledger] [repaired args: 1 arg(s) renamed] "`.
    /// Empty when nothing fired.
    note_prefix: String,
    /// `{tool, ok}` plus `repaired_from` (M2 fired) / `param_repaired: true`
    /// (task A fired) when applicable — the compact per-call trace entry
    /// accumulated into `tool_trace` for the ADR 091 turn-capture hook.
    trace: Value,
}

/// Successful tool evidence available to a *later* model round. Results stay
/// turn-local and are never copied into the compact WAL trace.
#[derive(Debug, Clone)]
struct ToolEvidence {
    tool: String,
    result: String,
}

impl Agent {
    /// Run one LLM-driven agentic turn, **streaming** its narration to `sink`.
    ///
    /// The akh's pinned model answers `user_prompt`, optionally calling any
    /// registered tool (gated by the permission policy) in a loop until it
    /// returns a final answer or [`DEFAULT_MAX_ROUNDS`] is reached. Rather than
    /// buffering, each piece of the turn is emitted to `sink` the instant it is
    /// produced: answer text arrives as [`AkhMessage::Narrative`] deltas as the
    /// model generates, chain-of-thought as [`AkhMessage::Reasoning`] deltas, and
    /// each tool call + result as a collapsible [`AkhMessage::ToolResult`] step.
    ///
    /// Returns `Some(())` when an answer or refusal was emitted, or `None`
    /// when there is **no usable model
    /// backend** — no model manager attached, or Ollama unreachable — signalling
    /// the caller to fall back to the symbolic
    /// [`ChatProcessor`](crate::chat::ChatProcessor). `agent_id` keys the model
    /// manager's per-akh system-prompt cache.
    ///
    /// Each round tries the streaming Ollama path and, on a transport/HTTP
    /// failure, falls back to one non-streaming `chat` call for that round (whose
    /// text is then emitted as a single message) so a streaming hiccup degrades
    /// gracefully instead of dropping the turn.
    ///
    /// ## Concurrency
    ///
    /// `sink.emit` is synchronous and non-blocking (it hands the message to a
    /// channel / notification send), so emitting from inside the
    /// [`spawn_blocking`](tokio::task::spawn_blocking) turn while the agent
    /// `Mutex` is held does not violate ADR-027 — no guard ever crosses an
    /// `.await`.
    ///
    /// `cancel` is checked once at the top of every round (between model
    /// calls, never mid-call): when set, the loop stops immediately without
    /// spending a final tool-free synthesis round, so the ACP idle watchdog
    /// (which sets it once it gives up waiting on this turn) can stop the
    /// worker as soon as the in-flight round returns instead of letting it run
    /// unattended for the rest of the turn's budget. A caller
    /// with no cancellation source (e.g. tests, the symbolic pipeline's
    /// callers) passes a flag that is never set.
    pub fn run_agentic_turn(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        sink: &dyn MessageSink,
        cancel: &std::sync::atomic::AtomicBool,
        is_autonomous_dispatch: bool,
    ) -> Option<()> {
        self.run_agentic_turn_with_conversation_state(
            engine,
            agent_id,
            user_prompt,
            sink,
            cancel,
            is_autonomous_dispatch,
            &self.conversation_state,
        )
        .map(|_| ())
    }

    /// Run one neural turn using the supplied conversation state.
    ///
    /// ACP supplies one state per protocol session; non-ACP callers use
    /// [`Self::run_agentic_turn`], which preserves the Agent's historical
    /// workspace-level state. The state mutex is intentionally never held
    /// across backend inference, so unrelated ACP sessions remain concurrent.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_agentic_turn_with_conversation_state(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        sink: &dyn MessageSink,
        cancel: &std::sync::atomic::AtomicBool,
        is_autonomous_dispatch: bool,
        conversation_state: &std::sync::Mutex<ConversationState>,
    ) -> Option<AgenticTurnOutcome> {
        self.run_agentic_turn_with_conversation_state_and_images(
            engine,
            agent_id,
            user_prompt,
            &[],
            sink,
            cancel,
            is_autonomous_dispatch,
            conversation_state,
        )
    }

    /// Image-aware form used by ACP. Images are grounded once through the
    /// selected backend and inserted as untrusted evidence before the tool
    /// loop; ordinary callers retain the text-only API above.
    ///
    /// Runtime-backed agents (Codex/Claude, via `akhrtd`) do not currently
    /// accept inline images, so `prompt_images` only reaches the
    /// ChatBackend (Ollama/Candle) path below.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_agentic_turn_with_conversation_state_and_images(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        prompt_images: &[super::chat_backend::ChatPromptImage],
        sink: &dyn MessageSink,
        cancel: &std::sync::atomic::AtomicBool,
        is_autonomous_dispatch: bool,
        conversation_state: &std::sync::Mutex<ConversationState>,
    ) -> Option<AgenticTurnOutcome> {
        // Phase 53g PR 4: a turn whose last provider call got an answer
        // proves the provider works again (e.g. the operator's ACP turn after
        // repairing a credential), so it reopens a closed provider gate.
        let observed = super::provider_failure::ProviderObservationScope::open();
        let outcome = self.run_agentic_turn_observed(
            engine,
            agent_id,
            user_prompt,
            prompt_images,
            sink,
            cancel,
            is_autonomous_dispatch,
            conversation_state,
        );
        if observed.finish() == Some(super::provider_failure::ProviderObservation::Responded) {
            super::provider_failure::note_model_success(engine);
        }
        outcome
    }

    /// [`Self::run_agentic_turn_with_conversation_state_and_images`] without
    /// the provider-gate bookkeeping.
    #[allow(clippy::too_many_arguments)]
    fn run_agentic_turn_observed(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        prompt_images: &[super::chat_backend::ChatPromptImage],
        sink: &dyn MessageSink,
        cancel: &std::sync::atomic::AtomicBool,
        is_autonomous_dispatch: bool,
        conversation_state: &std::sync::Mutex<ConversationState>,
    ) -> Option<AgenticTurnOutcome> {
        if let Some(runtime) = &self.agent_runtime {
            return Some(self.run_runtime_turn(
                runtime,
                engine,
                agent_id,
                user_prompt,
                sink,
                cancel,
                is_autonomous_dispatch,
                conversation_state,
            ));
        }
        // Resolve the akh's chat/agentic backing (ADR 070): an explicitly
        // configured backend (e.g. Candle) wins; otherwise wrap the Ollama model
        // manager. Either way the loop is backend-agnostic. `None` ⇒ no usable
        // backend, so the caller falls back to the symbolic ChatProcessor.
        let backend: std::sync::Arc<dyn super::chat_backend::ChatBackend> =
            match self.chat_backend.clone() {
                Some(b) => b,
                None => {
                    let mgr = self.model_manager.clone()?;
                    std::sync::Arc::new(super::chat_backend::OllamaChatBackend::new(mgr))
                }
            };
        if !backend.available() {
            return None;
        }

        // Explicit control is fail-closed. A malformed directive must not turn
        // into an ordinary interactive turn whose unsupported completion is
        // silently accepted.
        let mut completion_controller =
            match super::completion_controller::CompletionController::from_prompt(user_prompt) {
                Ok(controller) => controller,
                Err(error) => {
                    sink.emit(&AkhMessage::error("completion_control", error.to_string()));
                    return Some(AgenticTurnOutcome::Refused);
                }
            };

        // Capture the exact system prompt for this complete turn. Session-local
        // MCP manifests must not race through a workspace-keyed backend cache.
        let manifests = self
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list_advertised_manifests();
        let manifest_refs: Vec<&super::tool_manifest::ToolManifest> = manifests.iter().collect();
        let prepared_chat = backend.prepare_turn(agent_id, self.psyche.as_ref(), &manifest_refs);
        let visual_evidence = match backend.ground_prompt_images(prompt_images) {
            Ok(evidence) => evidence,
            Err(error) => {
                sink.emit(&AkhMessage::error("image_prompt", error.to_string()));
                return Some(AgenticTurnOutcome::Refused);
            }
        };

        let reg = self.tool_registry.lock().unwrap_or_else(|e| e.into_inner());
        let mut tools = ollama_tool_catalog(&*reg);
        // Release the registry lock now — it is not used again in this
        // function, and the `compute_deferred_groups` call below re-locks
        // the same `std::sync::Mutex` on this thread. `std::sync::Mutex` is
        // not reentrant, so holding `reg` past this point deadlocks every
        // agentic turn (100% reproducible, not a race — this was caught by
        // running the agentic_chat test suite, where every test hung on
        // exactly this second `.lock()` call).
        drop(reg);
        // ADR 089 task A/B: the FULL pre-retrieval-scoping catalog, captured
        // before M1 (below) narrows `tools` to what's actually offered this
        // round. Task A's param-repair-retry and task B's value grounding
        // both need to look up a tool's schema even when M1 scoped it out of
        // this round's offered set (the model can still emit a name it
        // remembers from earlier context or training).
        let full_catalog = tools.clone();
        // ADR 089 M1 (opt-in): retrieval-scope the catalog to the top-K tools
        // *relevant to this request* (scored over name + description) instead of
        // offering everything. This is the organ's answer to both the prefill
        // cost of a several-thousand-token catalog and small models degenerating
        // past ~14 tools — the model sees the K tools most likely to matter, and
        // permission gating still applies at dispatch.
        let mut retrieval_scoping_active = false;
        if let Some(k) = std::env::var("AKH_AGENT_TOOL_RETRIEVAL")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && k < tools.len()
        {
            retrieval_scoping_active = true;
            tracing::debug!(
                k,
                from = tools.len(),
                "retrieval-scoping tool catalog (AKH_AGENT_TOOL_RETRIEVAL)"
            );
            tools = super::organ::retrieve_scoped(&tools, user_prompt, k, Some(engine));
        } else if let Some(cap) = std::env::var("AKH_AGENT_MAX_TOOLS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && cap < tools.len()
        {
            // Legacy blind cap (AKH_AGENT_MAX_TOOLS): kept for A/B comparison
            // against retrieval scoping. `0` runs the akh tool-free (pure chat).
            tracing::debug!(
                cap,
                from = tools.len(),
                "capping tool catalog (AKH_AGENT_MAX_TOOLS)"
            );
            tools.truncate(cap);
        } else if let Some(k) = backend.preferred_tool_count()
            && k < tools.len()
        {
            retrieval_scoping_active = true;
            tracing::debug!(
                k,
                from = tools.len(),
                backend = backend.label(),
                "retrieval-scoping tool catalog for model's learned operating regime"
            );
            tools = super::organ::retrieve_scoped(&tools, user_prompt, k, Some(engine));
        }
        // Contract gates are host post-conditions, so retrieval scoping may
        // not hide the very tools needed to satisfy them.
        if let Some(controller) = &completion_controller {
            let missing =
                pin_required_tools(&mut tools, &full_catalog, &controller.required_tools());
            if !missing.is_empty() {
                sink.emit(&AkhMessage::error(
                    "completion_contract_unavailable_tools",
                    format!(
                        "controlled turn requires unavailable tools: {}",
                        missing.join(", ")
                    ),
                ));
                return Some(AgenticTurnOutcome::Refused);
            }
        }

        // Phase 44m (ADR 065 D12.1): capture the deferred-tool set — what was
        // in `full_catalog` but NOT offered this round — so the ACP `_akh/usage`
        // panel can show the "loaded-but-deferred" tier. Group by tool origin
        // (`System tools` / `MCP tools`) and reason. Cached outside the agent
        // lock the same way `last_context_stats` is, so a busy turn does not
        // block a sidebar poll.
        {
            let deferred_groups = compute_deferred_groups(
                &full_catalog,
                &tools,
                &*self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()),
                retrieval_scoping_active,
            );
            if let Ok(mut slot) = self.last_deferred_tools.lock() {
                *slot = deferred_groups;
            }
        }

        // Record the operator turn before assembling, so the conversation's last
        // turn IS the current prompt — a history-aware ContextManager renders the
        // turns directly (the agent turn is recorded at the end). The symbolic
        // path already records both speakers; this closes the gap for the agentic
        // path so multi-turn context works (measure-first Q1).
        conversation_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record_operator_turn(user_prompt);

        // Phase 44d/44g: retrieve the input's KG context + relevant past episodes
        // for the working-context manager. The agent owns the engine + predicates;
        // the manager only budgets + renders these sections. (44e session seeding
        // rides `with_context`'s seed slot — populated when a session summary
        // exists; the windows are large so all sections fit in practice.)
        let kg_context = if std::env::var("AKH_KG_DELTA")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            // 44d delta refresh: inject only facts not already shown this session
            // (the activation frontier that moved), trusting retained history to
            // hold the rest. Opt-in; the default re-sends the full relevant block.
            let mut conv_guard = conversation_state.lock().unwrap_or_else(|e| e.into_inner());
            let seen = conv_guard.kg_seen_mut();
            crate::nlu::kg_context::build_kg_context_delta(engine, user_prompt, seen)
        } else {
            let mut cache = crate::nlu::kg_context::KgContextCache::default();
            crate::nlu::kg_context::build_kg_context(engine, user_prompt, &mut cache)
        };
        let recalled = self.recall_for_prompt(engine, user_prompt, 3);
        // 44e: a one-line continuity seed (the conversation's opening ask), so the
        // thread survives history trimming. None on the first turn.
        let session_seed = {
            let conv_guard = conversation_state.lock().unwrap_or_else(|e| e.into_inner());
            Self::session_seed_from(&conv_guard)
        };

        // ADR 081 tiering: only body-blanked frontmatter is admitted to
        // resident context. The durable catalog returns newest heads first;
        // bound the recall tier and separately pin explicitly-active notes.
        let mut recent_notes = self
            .note_catalog
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|catalog| catalog.frontmatter_only())
            .unwrap_or_default();
        recent_notes.truncate(8);
        let active_notes: Vec<crate::library::Note> = recent_notes
            .iter()
            .filter(|note| {
                matches!(
                    note.frontmatter.get("active"),
                    Some(crate::library::FrontmatterValue::Checkbox(true))
                ) || matches!(
                    note.frontmatter.get("status"),
                    Some(crate::library::FrontmatterValue::Text(status))
                        if status.eq_ignore_ascii_case("active")
                )
            })
            .take(4)
            .cloned()
            .collect();

        // Phase 44a (ADR 065): assemble the working context through the akh's
        // `ContextManager` instead of building the prompt ad hoc. The default
        // pass-through manager reproduces the single-user-turn behaviour
        // byte-for-byte; richer managers (history, KG, recall) slot in here
        // without the loop changing. The loop then appends assistant/tool turns
        // to this base across rounds, exactly as before.
        let (mut conversation, tools): (Vec<Value>, Vec<Value>) = {
            use super::context::{
                ContextManager, DEFAULT_RESERVED_OUTPUT, EpisodeState, HeuristicTokenCounter,
                PassThroughManager, TokenBudget, TokenCounter,
            };
            static PASS_THROUGH: PassThroughManager = PassThroughManager;
            let counter = HeuristicTokenCounter;
            // Size the budget window from the model's real capabilities (GGUF
            // metadata for Candle, `/api/show` for Ollama) — e.g. gemma4-12B =
            // 262144, glm-5.2 = 1_000_000 — not a hardcoded guess. Falls back to
            // the default window when capabilities are unknown.
            let budget = match backend.capabilities().context_window {
                Some(window) => TokenBudget::new(window, DEFAULT_RESERVED_OUTPUT),
                None => TokenBudget::default(),
            };
            let conv_guard = conversation_state.lock().unwrap_or_else(|e| e.into_inner());
            let episode = EpisodeState::new(agent_id, user_prompt, engine)
                .with_conversation(Some(&*conv_guard))
                .with_context(kg_context.as_deref(), &recalled, session_seed.as_deref())
                .with_tools(&tools)
                .with_notes(&active_notes, &recent_notes);
            let cm_guard = self
                .context_manager
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let manager: &dyn ContextManager = cm_guard.as_deref().unwrap_or(&PASS_THROUGH);
            match manager.assemble(&episode, &budget, &counter) {
                Ok(assembled) => {
                    // Phase 44m (ADR 065 D12): cache the normalized stats
                    // outside the agent lock so the ACP `_akh/usage` panel can
                    // read them without contending for this `&mut self` — a
                    // busy turn returns the previous value with `stale: true`.
                    let stats = manager.stats(&assembled, &budget);
                    if let Ok(mut slot) = self.last_context_stats.lock() {
                        *slot = Some(stats);
                    }
                    super::context::budget::record_assembly_telemetry(
                        manager,
                        &assembled,
                        &budget,
                        counter.label(),
                    );
                    (assembled.messages, assembled.tools)
                }
                Err(e) => {
                    tracing::error!(
                        agent_id,
                        error = %e,
                        "context assembly failed; using single-turn fallback"
                    );
                    (
                        vec![json!({"role": "user", "content": user_prompt})],
                        tools.clone(),
                    )
                }
            }
        };

        // Bounded, per-turn pre-action inputs. The configured context manager
        // still owns durable core/recall/archival memory.
        if let Some(vsa_context) = super::completion_controller::VsaReasoningContext::build(
            engine,
            user_prompt,
            completion_controller
                .as_ref()
                .map(|controller| controller.vsa_seeds())
                .unwrap_or(&[]),
        ) {
            insert_system_before_last_user(&mut conversation, vsa_context);
        }
        if let Some(controller) = &completion_controller {
            insert_system_before_last_user(&mut conversation, controller.system_context());
        }
        if let Some(evidence) = visual_evidence.as_ref() {
            insert_system_before_last_user(&mut conversation, evidence.clone());
            let retained = conversation_state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .append_to_latest_operator_turn(evidence);
            if !retained {
                sink.emit(&AkhMessage::error(
                    "image_prompt",
                    "could not retain native OCR evidence in this session",
                ));
                return Some(AgenticTurnOutcome::Refused);
            }
        }

        // Compact tool trace for the ADR 091 turn-capture hook: {tool, ok} per
        // call, accumulated across every round of this turn (not just the
        // final one) — never raw arguments/results, so a chatty tool loop
        // doesn't bloat the WAL payload.
        let mut tool_trace: Vec<Value> = Vec::new();
        let work_cwd = if is_autonomous_dispatch {
            crate::fs::session_cwd()
        } else {
            None
        };
        let witness_route = self.witness_route.clone();
        let initial_worktree = work_cwd
            .as_deref()
            .and_then(|cwd| witness_route.witness(cwd));
        // Full results are retained only for this turn's provenance gate. A
        // study-note write may cite a successful evidence result from a prior
        // model round, never another parallel call emitted in the same round.
        let mut tool_evidence: Vec<ToolEvidence> = Vec::new();

        // Registry membership snapshot for the ADR 089 M2 repair gate below: an
        // emitted name outside this set is a guaranteed `unknown tool` failure,
        // which is exactly (and only) when the organ's repair pass may fire.
        let known_tools: std::collections::HashSet<String> = self
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list()
            .into_iter()
            .map(|s| s.name)
            .collect();

        // Govern the tool loop with a frontier-style budget — a model-scaled
        // iteration backstop + wall-clock cap + repeat-loop detection — instead of
        // a flat round count. The loop yields *why* it stopped so the turn can end
        // with a forced synthesis rather than a bare cutoff.
        let mut budget = RoundBudget::new(backend.capabilities().context_window);
        if is_autonomous_dispatch {
            budget = budget.for_dispatch();
        }
        // Turn-level metrics: track total prompt size and request count for diagnosing
        // pathological latency (work item "turns die quietly", ask 2).
        let mut turn_metrics = super::llm_metrics::TurnMetrics::new();
        // In-turn tool-result budget (Lilith C2, ADR 116, Anima WI
        // 01a0e270-314e): `ContextManager::assemble` above sizes the prompt
        // once, at the top of the turn, but a tool result arrives mid-turn —
        // nothing re-checks the running request against the akh's real
        // `context_window` before the next model round. Same window +
        // reserve the assembly step used, so a low-window backend (e.g. a
        // 32768-token self-hosted vLLM) never sees a request that overshoots
        // it just because a chatty tool (an MCP call returning tens of KB of
        // JSON) ran mid-turn.
        let token_budget = match backend.capabilities().context_window {
            Some(window) => {
                super::context::TokenBudget::new(window, super::context::DEFAULT_RESERVED_OUTPUT)
            }
            None => super::context::TokenBudget::default(),
        };
        let token_counter = super::context::HeuristicTokenCounter;
        let stop: TurnStop = loop {
            if let Some(stop) = budget.exhausted() {
                break stop;
            }
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break TurnStop::Cancelled;
            }
            // Re-check on every round, not just round 0: this is exactly the
            // "tool-result continuation" shape (assistant + tool appended,
            // nothing since) that the live 500 was hit on. See
            // `ensure_real_user_turn` for why this can't just be asserted
            // once at assembly time.
            ensure_real_user_turn(&mut conversation, user_prompt);
            // Stream this round. Content deltas become live answer text, thinking
            // deltas become live reasoning. `used_fallback` records whether the
            // streaming path failed and we silently retried non-streaming — in
            // which case nothing was streamed and we emit the result ourselves.
            let mut used_fallback = false;
            // A controlled turn cannot expose candidate prose before the host
            // has evaluated its evidence gates. Reasoning and tool results may
            // still stream; narrative is released only on accepted completion.
            let defer_narrative = completion_controller.is_some();
            // Throttle Keepalive emissions: tool-call chunks can arrive at token
            // rate (~100/sec for a 1k-char JSON payload at 100 tok/s), so we emit
            // at most one heartbeat per 30s to reset the watchdog without spam.
            let mut last_keepalive = Instant::now();
            const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
            // Track prompt size and get request sequence number before the backend call.
            let prompt_chars: usize = conversation
                .iter()
                .filter_map(|m| m.get("content").and_then(serde_json::Value::as_str))
                .map(|s| s.len())
                .sum();
            let request_seq = turn_metrics.note_prompt(prompt_chars);
            // Mark that we're issuing a request (wedge-detection timer).
            budget.note_request();
            let turn = {
                let mut on_event = |ev: StreamEvent| match ev {
                    StreamEvent::Content(delta) => {
                        // Content/thinking resets the implicit heartbeat — a model
                        // generating visible text is alive by definition.
                        last_keepalive = Instant::now();
                        if !defer_narrative {
                            sink.emit(&AkhMessage::narrative(delta, "acp"));
                        }
                    }
                    StreamEvent::Thinking(delta) => {
                        last_keepalive = Instant::now();
                        sink.emit(&AkhMessage::Reasoning {
                            step: delta,
                            expression: None,
                        });
                    }
                    // Keepalive: the stream is alive but yielded no visible delta
                    // (e.g. tool-call chunks Ollama accumulates). Emit a throttled
                    // heartbeat to reset the watchdog's idle timer without cluttering
                    // the thread (ask 3: "reset on any chunk received from the provider").
                    StreamEvent::Keepalive => {
                        if last_keepalive.elapsed() >= KEEPALIVE_INTERVAL {
                            sink.emit(&AkhMessage::system("."));
                            last_keepalive = Instant::now();
                        }
                    }
                };
                let streamed = backend.chat_with_tools_streaming_prepared(
                    &prepared_chat,
                    agent_id,
                    &conversation,
                    &tools,
                    request_seq,
                    &mut on_event,
                );
                super::provider_failure::observe_chat_result(&streamed);
                match streamed {
                    Ok(t) => t,
                    Err(stream_err) => {
                        tracing::warn!(
                            agent_id,
                            "streaming chat failed ({stream_err}); falling back to non-streaming"
                        );
                        // Make the fallback observable: emit a System message so the
                        // operator sees the degradation in the thread and the idle
                        // watchdog's timer is reset (it would otherwise kill a long
                        // non-streaming response at exactly 900s).
                        sink.emit(&AkhMessage::system(format!(
                            "streaming failed ({backend_label}), continuing non-streaming",
                            backend_label = backend.label()
                        )));
                        used_fallback = true;
                        let fallback = backend.chat_with_tools_prepared(
                            &prepared_chat,
                            agent_id,
                            &conversation,
                            &tools,
                            request_seq,
                        );
                        super::provider_failure::observe_chat_result(&fallback);
                        match fallback {
                            Ok(t) => t,
                            Err(e) => {
                                sink.emit(&AkhMessage::error("llm", e.to_string()));
                                return Some(AgenticTurnOutcome::Refused);
                            }
                        }
                    }
                }
            };

            emit_turn_reasoning(sink, &turn);

            // Echo the assistant turn (content + tool calls) back into the
            // conversation so the model sees its own prior step.
            let tool_calls_json: Vec<Value> = turn
                .tool_calls
                .iter()
                .map(|tc| json!({"function": {"name": tc.name, "arguments": tc.arguments}}))
                .collect();
            conversation.push(json!({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": tool_calls_json,
            }));

            if turn.tool_calls.is_empty() {
                let answer = turn.content.trim();

                tracing::debug!(
                    agent_id,
                    answer_len = answer.len(),
                    rounds = budget.rounds_so_far(),
                    mode = if is_autonomous_dispatch {
                        "dispatch"
                    } else {
                        "interactive"
                    },
                    "no tool calls: evaluating completion"
                );

                // Explicit contracts apply to interactive and dispatch turns.
                // Evaluate before the empty-answer guard so every premature
                // declaration receives actionable next-gate feedback.
                if let Some(controller) = &completion_controller
                    && let super::completion_controller::CompletionDecision::Continue { feedback } =
                        controller.completion_decision()
                {
                    let retry_budget_exhausted = budget.note_no_work_retry();
                    tracing::info!(
                        agent_id,
                        next_gate = controller.next_gate_label(),
                        retry_budget_exhausted,
                        "completion contract not yet satisfied"
                    );
                    if retry_budget_exhausted {
                        break TurnStop::ContractUnsatisfied;
                    }
                    conversation.push(json!({"role": "user", "content": feedback}));
                    continue;
                }

                // Check for empty answer first (WI-259 wedge fix): the model
                // stopped calling tools but produced no answer text. This must
                // be checked BEFORE the dispatch mode guard, otherwise dispatch
                // mode would retry indefinitely when the model keeps returning
                // empty answers.
                if answer.is_empty() {
                    tracing::warn!(
                        agent_id,
                        rounds = budget.rounds_so_far(),
                        "completion check: empty answer → forcing final synthesis"
                    );
                    break if is_autonomous_dispatch {
                        TurnStop::DeclaredButIdle
                    } else {
                        TurnStop::IterationCap
                    };
                }

                // Post-condition guard: in autonomous dispatch mode, verify that
                // work was actually performed before accepting the model's
                // declaration of completion (no tool calls = "I'm done").
                // Tri-state verdict: ⊤ accept, ⊥retry with feedback, ⊥fatal after
                // exhaustion.
                if is_autonomous_dispatch {
                    tracing::debug!(
                        agent_id,
                        "completion check: autonomous dispatch mode, verifying work was performed"
                    );

                    // Deterministic oracle: did the turn perform any mutating
                    // operations? Check tool_trace for file_io/edit/shell_exec (ok
                    // = true).
                    let explicit_write = tool_trace
                        .iter()
                        .any(|entry| entry.get("wrote").and_then(Value::as_bool) == Some(true));
                    let shell_ran = tool_trace.iter().any(|entry| {
                        entry.get("tool").and_then(Value::as_str) == Some("shell_exec")
                            && entry.get("ok").and_then(Value::as_bool) == Some(true)
                    });
                    let shell_changed_tree = shell_ran
                        && initial_worktree
                            .zip(
                                work_cwd
                                    .as_deref()
                                    .and_then(|cwd| witness_route.witness(cwd)),
                            )
                            .is_some_and(|(before, after)| before != after);
                    let has_mutating_calls = explicit_write || shell_changed_tree;

                    tracing::debug!(
                        agent_id,
                        has_mutating_calls,
                        tool_trace_len = tool_trace.len(),
                        "completion check: work verification complete"
                    );

                    // If no mutating calls were made, the model declared completion
                    // but performed no work: retry with taxonomy-anchored feedback.
                    // (We don't check git status when there were zero tool calls -
                    // the model didn't even try to do work.)
                    if !has_mutating_calls {
                        // WI-280: Increment the no-work retry counter and check if
                        // exhausted. This prevents infinite loops when the model keeps
                        // declaring completion without doing work (which would keep
                        // rounds_so_far() at 0 since no tools are called).
                        let retry_budget_exhausted = budget.note_no_work_retry();
                        tracing::info!(
                            agent_id,
                            no_work_retries = budget.no_work_retries,
                            rounds = budget.rounds_so_far(),
                            retry_budget_exhausted,
                            "completion check: no work performed"
                        );

                        if retry_budget_exhausted {
                            // Exhausted: close with a named failure
                            tracing::info!(
                                agent_id,
                                "completion check → terminating: no-work retry budget exhausted"
                            );
                            break TurnStop::DeclaredButIdle;
                        }
                        // Retry: append feedback and continue the loop
                        tracing::info!(
                            agent_id,
                            "completion check → retrying: adding feedback and continuing loop"
                        );
                        conversation.push(json!({
                            "role": "user",
                            "content": TurnStop::DeclaredButIdle.clause(),
                        }));
                        continue;
                    } else {
                        tracing::info!(
                            agent_id,
                            "completion check → accepting: work was performed"
                        );
                    }
                } else {
                    tracing::debug!(
                        agent_id,
                        "completion check: interactive mode, accepting completion"
                    );
                }

                // Either we're in interactive mode, or the post-condition passed:
                // record the turn and exit normally.
                tracing::info!(
                    agent_id,
                    answer_len = answer.len(),
                    rounds = budget.rounds_so_far(),
                    mode = if is_autonomous_dispatch {
                        "dispatch"
                    } else {
                        "interactive"
                    },
                    "completion check → returning: turn complete with answer"
                );
                conversation_state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .record_agent_turn(answer);
                // ADR 091: capture the completed turn to the WAL (raw
                // staging; a no-op if no sink is installed or capture is
                // opted out).
                self.emit_turn_wal_with_model(
                    engine,
                    agent_id,
                    user_prompt,
                    answer,
                    &tool_trace,
                    self.model_manager.as_deref().map(|m| m.model()),
                    is_autonomous_dispatch,
                );
                // Ordinary streaming already emitted the answer as deltas.
                // Controlled streaming held it provisionally until this host
                // post-condition passed, so release the accepted answer now.
                if used_fallback || defer_narrative {
                    sink.emit(&AkhMessage::narrative(answer, "acp"));
                }
                return Some(AgenticTurnOutcome::Completed);
            }

            budget.note_request();

            // Tool round. Any prose the model emitted alongside its tool calls is
            // reasoning; on the fallback path it was not streamed, so surface it
            // as a thinking step here (the streaming path already did).
            if used_fallback || defer_narrative {
                let reasoning = turn.content.trim();
                if !reasoning.is_empty() {
                    sink.emit(&AkhMessage::Reasoning {
                        step: reasoning.to_string(),
                        expression: None,
                    });
                }
            }

            // Record this round against the budget (advances the counter; flags a
            // spin once a (tool, args) signature repeats past the limit).
            budget.note_round(
                turn.tool_calls
                    .iter()
                    .map(|tc| (tc.name.as_str(), &tc.arguments)),
            );

            // Execute each requested tool; the call (args) and its result form one
            // collapsible step the TUI folds under "▸ N steps".
            let evidence_before_round = tool_evidence.len();
            for tc in &turn.tool_calls {
                let outcome = self.execute_tool_call(
                    engine,
                    tc,
                    &known_tools,
                    &full_catalog,
                    user_prompt,
                    &tool_evidence[..evidence_before_round],
                );
                // A completed tool is progress even when it ran longer than
                // the gap between model requests. Do not kill the next round
                // using the tool's execution time as evidence of a wedge.
                budget.note_request();
                sink.emit(&AkhMessage::tool_result(
                    outcome.exec_name.clone(),
                    outcome.success,
                    format!(
                        "{}({}) → {}",
                        outcome.note_prefix,
                        preview_args(&outcome.exec_args),
                        outcome.result
                    ),
                ));
                tool_trace.push(outcome.trace);
                if outcome.success && evidence_tool(&outcome.exec_name) {
                    tool_evidence.push(ToolEvidence {
                        tool: outcome.exec_name.clone(),
                        result: outcome.result.clone(),
                    });
                }
                if let Some(controller) = &mut completion_controller {
                    controller.observe(&outcome.exec_name, &outcome.exec_args, outcome.success);
                }
                // Bound this result against what's left of the window before
                // it joins the conversation the next model round is built
                // from — the tool already ran; a request rejected afterwards
                // for being too large would throw that work away for
                // nothing. `used` measures everything already committed to
                // the next request (system prompt + tool schemas +
                // conversation so far, including any earlier tool result
                // from this same round).
                let used = super::context::estimate_running_tokens(
                    &token_counter,
                    prepared_chat.system_prompt(),
                    &tools,
                    &conversation,
                );
                let tool_schema = full_catalog
                    .iter()
                    .find(|t| t["function"]["name"].as_str() == Some(outcome.exec_name.as_str()));
                let bound = super::context::bound_tool_result(
                    &outcome.exec_name,
                    tool_schema,
                    &outcome.result,
                    used,
                    &token_budget,
                    &token_counter,
                );
                if let Some(record) = &bound.truncation {
                    tracing::warn!(
                        agent_id,
                        tool = %outcome.exec_name,
                        original_bytes = record.original_bytes,
                        kept_bytes = record.kept_bytes,
                        original_tokens_est = record.original_tokens_est,
                        kept_tokens_est = record.kept_tokens_est,
                        narrowing_params = ?record.narrowing_params,
                        "tool result exceeded remaining turn budget; truncated before appending to conversation"
                    );
                }
                conversation.push(json!({"role": "tool", "content": bound.content}));
            }
        };

        // Cancelled (the ACP idle watchdog already gave up and answered the
        // client): stop right here rather than spending one more model round
        // nobody is waiting on — this is exactly what unwedges the worker
        // that (a) of the turn-wedge fix targets. `sink.emit` further down is
        // harmless even if the client's drain loop has already returned (a
        // closed channel just drops the message), but we skip it entirely
        // since there is nothing meaningful left to narrate.
        if stop == TurnStop::Cancelled {
            tracing::warn!(
                agent_id,
                "agentic turn cancelled (idle watchdog); ending without forced synthesis"
            );
            return Some(AgenticTurnOutcome::Interrupted(stop));
        }

        // An explicit contract is a hard post-condition, not a suggestion for
        // the forced-synthesis path. Do not let a model narrate success, record
        // a completed conversation turn, or write a completed-turn WAL entry
        // after the controller's bounded retry budget is exhausted.
        if completion_controller.as_ref().is_some_and(|controller| {
            controller.completion_decision()
                != super::completion_controller::CompletionDecision::Complete
        }) {
            let missing = completion_controller
                .as_ref()
                .and_then(|controller| controller.next_gate_label())
                .unwrap_or("unknown evidence");
            sink.emit(&AkhMessage::error(
                "completion_contract_unsatisfied",
                format!("controlled turn failed: missing `{missing}` evidence"),
            ));
            turn_metrics.log();
            return Some(AgenticTurnOutcome::Refused);
        }

        // A dispatch interruption must not become a fabricated completion or a
        // completed-turn training example. The runner retains the partial work.
        if is_autonomous_dispatch {
            sink.emit(&AkhMessage::error(
                "turn_interrupted",
                format!(
                    "Task incomplete: {}. Preserve existing work for continuation.",
                    stop.clause()
                ),
            ));
            turn_metrics.log();
            return Some(AgenticTurnOutcome::Interrupted(stop));
        }

        // Budget exhausted — force one tool-free synthesis ("reflect and answer")
        // so the turn returns the best answer it can from what it gathered, rather
        // than the old bare "(stopped after N rounds)" cutoff.
        self.force_final_answer(
            backend.as_ref(),
            &prepared_chat,
            engine,
            agent_id,
            user_prompt,
            &mut conversation,
            sink,
            stop,
            &tool_trace,
            &mut turn_metrics,
            conversation_state,
        );
        // Log turn-level metrics summary (total prompt size, request count).
        turn_metrics.log();
        Some(AgenticTurnOutcome::Interrupted(stop))
    }

    /// Execute one model-requested tool call, running it through every ADR 089
    /// organ pass before/around the registry's `execute()`, and return
    /// everything the caller needs to narrate + trace it. Factored out of
    /// [`Self::run_agentic_turn`]'s dispatch loop because task A's retry path
    /// (repair → re-execute → re-repair-values) made the loop body too deep to
    /// read inline.
    ///
    /// Pass order per call:
    /// 1. **M2** — if `tc.name` isn't in `known_tools`, try
    ///    [`organ::repair_unknown_call`] against `full_catalog` (the schemas
    ///    actually available to ground against).
    /// 2. **Task B** (opt-in) — [`organ::ground_call_values`] on the
    ///    (possibly M2-repaired) args, before the first `execute()`.
    /// 3. `execute()`. On `Err(AgentError::ToolExecution { .. })` — the
    ///    *only* structurally-repairable channel (see the ADR 089 task A
    ///    scout report on `execute()`'s failure shapes; a permission denial
    ///    surfaces as `Ok(ToolOutput{success:false})` and is never retried
    ///    here) — **task A**: [`organ::repair_known_call_args`] against only
    ///    this tool's own schema, task B again on the repaired args, then one
    ///    retry `execute()`.
    fn execute_tool_call(
        &self,
        engine: &Engine,
        tc: &super::ollama::ToolCall,
        known_tools: &std::collections::HashSet<String>,
        full_catalog: &[Value],
        user_prompt: &str,
        prior_evidence: &[ToolEvidence],
    ) -> ToolCallOutcome {
        // M2: repair a name the registry doesn't recognize against the
        // offered catalog. Never touches a name it already knows.
        let m2 = if known_tools.contains(&tc.name) {
            None
        } else {
            super::organ::repair_unknown_call(&tc.name, &tc.arguments, full_catalog)
        };
        if let Some(r) = &m2 {
            tracing::info!(from = %tc.name, to = %r.name, note = %r.note, "organ repaired unknown tool call");
        }
        let (exec_name, mut exec_args) = match &m2 {
            Some(r) => (r.name.clone(), r.arguments.clone()),
            None => (tc.name.clone(), tc.arguments.clone()),
        };

        // Task B (opt-in): ground argument values before the first execute
        // attempt, after any M2 name repair.
        let b_pre =
            super::organ::ground_call_values(&exec_name, &exec_args, user_prompt, full_catalog);
        if let Some(g) = &b_pre {
            exec_args = g.arguments.clone();
        }

        // A study note changes durable memory. Its citation-shaped fields are
        // not proof on their own: bind every claimed source/locator/quote to a
        // successful evidence-bearing tool result the model actually observed
        // in an earlier round. Calls emitted in parallel with retrieval see the
        // pre-round slice and therefore fail closed.
        if study_note_write_tool(&exec_name)
            && let Err(message) = verify_study_note_evidence(&exec_args, prior_evidence)
        {
            return ToolCallOutcome {
                exec_name: exec_name.clone(),
                exec_args,
                success: false,
                result: format!("tool error: {exec_name} evidence gate: {message}"),
                note_prefix: "[evidence gate] ".into(),
                trace: json!({
                    "tool": exec_name,
                    "ok": false,
                    "evidence_verified": false
                }),
            };
        }

        let input = tool_input_from_args(&exec_args);
        let mut param_repaired = false;
        let mut a_note: Option<String> = None;
        let mut study_note_evidence_verified = study_note_write_tool(&exec_name).then_some(true);
        // Clone the internally synchronized registry before dispatch. The
        // outer Agent slot stays available while a permission prompt or remote
        // MCP request blocks, and the same stable handle covers a repair retry.
        let registry = self.tool_registry_handle();
        let call_started = std::time::Instant::now();
        let exec_result = registry.execute(&exec_name, input, engine);
        let call_duration_ms = call_started.elapsed().as_millis() as u64;
        let (success, result) = match exec_result {
            Ok(o) => (o.success, o.result),
            Err(e) => {
                // Task A: retry once, but only against the structurally
                // repairable channel, and only when not opted out. An
                // `Ask`-gated tool is also excluded: the retry re-runs the
                // permission gate, so it would prompt the user a second time
                // (and double-log the audit ledger) for one logical call —
                // the first approved failure stands instead.
                let opted_out = std::env::var("AKH_AGENT_PARAM_REPAIR").is_ok_and(|v| v == "0");
                let repairable = matches!(e, super::error::AgentError::ToolExecution { .. });
                let retry = if opted_out || !repairable || registry.would_ask(&exec_name) {
                    None
                } else {
                    super::organ::repair_known_call_args(&exec_name, &exec_args, full_catalog)
                };
                match retry {
                    Some(repaired) => {
                        let mut retry_args = repaired.arguments;
                        let mut retry_note = repaired.note;
                        if let Some(g) = super::organ::ground_call_values(
                            &exec_name,
                            &retry_args,
                            user_prompt,
                            full_catalog,
                        ) {
                            retry_args = g.arguments;
                            retry_note = format!("{retry_note}; {}", g.note);
                        }
                        let retry_evidence_error = if study_note_write_tool(&exec_name) {
                            verify_study_note_evidence(&retry_args, prior_evidence).err()
                        } else {
                            None
                        };
                        if let Some(message) = retry_evidence_error {
                            exec_args = retry_args;
                            param_repaired = true;
                            a_note = Some(retry_note);
                            study_note_evidence_verified = Some(false);
                            (
                                false,
                                format!(
                                    "tool error: {exec_name} evidence gate after argument repair: {message}"
                                ),
                            )
                        } else {
                            let retry_input = tool_input_from_args(&retry_args);
                            match registry.execute(&exec_name, retry_input, engine) {
                                Ok(o) => {
                                    tracing::info!(
                                        tool = %exec_name,
                                        note = %retry_note,
                                        "ADR 089 task A: retried tool call after param repair"
                                    );
                                    exec_args = retry_args;
                                    param_repaired = true;
                                    a_note = Some(retry_note);
                                    (o.success, o.result)
                                }
                                Err(e2) => (false, format!("tool error: {e2}")),
                            }
                        }
                    }
                    None => (false, format!("tool error: {e}")),
                }
            }
        };

        let mut note_prefix = String::new();
        if let Some(r) = &m2 {
            note_prefix.push_str(&format!("[repaired: {}] ", r.note));
        }
        if let Some(g) = &b_pre {
            note_prefix.push_str(&format!("[grounded: {}] ", g.note));
        }
        if let Some(note) = &a_note {
            note_prefix.push_str(&format!("[repaired args: {note}] "));
        }

        // ADR 116 slice C2: the trace doubles as the captured receipt. The
        // additive fields (bounded so a turn row stays small) let the operator-
        // side harvester read what each call was given and what it returned
        // without opening any store; earlier readers only look at tool/ok/wrote.
        let mut trace = json!({
            "tool": exec_name.clone(),
            "ok": success,
            "wrote": verified_write(&exec_name, &exec_args, success),
            "args": crate::agent::tool::truncate_summary(&exec_args.to_string(), TRACE_ARGS_MAX_BYTES),
            "output": crate::agent::tool::truncate_summary(&result, TRACE_OUTPUT_MAX_BYTES),
            "duration_ms": call_duration_ms,
        });
        if let Some(verified) = study_note_evidence_verified {
            trace["evidence_verified"] = json!(verified);
        }
        if m2.is_some() {
            trace["repaired_from"] = json!(tc.name.clone());
        }
        if param_repaired {
            trace["param_repaired"] = json!(true);
        }

        ToolCallOutcome {
            exec_name,
            exec_args,
            success,
            result,
            note_prefix,
            trace,
        }
    }

    /// Final synthesis step when the [`RoundBudget`] is exhausted: nudge the model
    /// to answer *now* with no tools, using what it already gathered, and emit the
    /// result. Mirrors how frontier harnesses convert a hit budget into a
    /// best-effort answer instead of a dead end.
    ///
    /// `user_prompt`/`tool_trace` are threaded through purely so the ADR 091
    /// turn-capture hook fires here too — this is a turn-end site exactly like
    /// the normal-completion path in [`Self::run_agentic_turn`].
    #[allow(clippy::too_many_arguments)]
    fn force_final_answer(
        &self,
        backend: &dyn super::chat_backend::ChatBackend,
        prepared_chat: &super::chat_backend::PreparedChat,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        conversation: &mut Vec<Value>,
        sink: &dyn MessageSink,
        stop: TurnStop,
        tool_trace: &[Value],
        turn_metrics: &mut super::llm_metrics::TurnMetrics,
        conversation_state: &std::sync::Mutex<ConversationState>,
    ) {
        conversation.push(json!({
            "role": "user",
            "content": format!(
                "{} — stop calling tools and answer the original question now, \
                 concisely, using what you have gathered so far.",
                stop.clause()
            ),
        }));

        // Empty catalog → the model must produce text, not another tool call.
        let no_tools: [Value; 0] = [];
        let mut used_fallback = false;
        let mut last_keepalive = Instant::now();
        // Track the forced-answer prompt size and get request sequence number.
        let prompt_chars: usize = conversation
            .iter()
            .filter_map(|m| m.get("content").and_then(serde_json::Value::as_str))
            .map(|s| s.len())
            .sum();
        let request_seq = turn_metrics.note_prompt(prompt_chars);
        let turn = {
            let mut on_event = |ev: StreamEvent| match ev {
                StreamEvent::Content(delta) => {
                    last_keepalive = Instant::now();
                    sink.emit(&AkhMessage::narrative(delta, "acp"));
                }
                StreamEvent::Thinking(delta) => {
                    last_keepalive = Instant::now();
                    sink.emit(&AkhMessage::Reasoning {
                        step: delta,
                        expression: None,
                    });
                }
                StreamEvent::Keepalive => {
                    if last_keepalive.elapsed() >= Duration::from_secs(30) {
                        sink.emit(&AkhMessage::system("."));
                        last_keepalive = Instant::now();
                    }
                }
            };
            let streamed = backend.chat_with_tools_streaming_prepared(
                prepared_chat,
                agent_id,
                conversation,
                &no_tools,
                request_seq,
                &mut on_event,
            );
            super::provider_failure::observe_chat_result(&streamed);
            match streamed {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        agent_id,
                        "forced-answer streaming failed ({e}); non-streaming"
                    );
                    // Make the fallback observable (same fix as the normal round).
                    sink.emit(&AkhMessage::system(format!(
                        "streaming failed ({backend_label}), continuing non-streaming",
                        backend_label = backend.label()
                    )));
                    used_fallback = true;
                    let fallback = backend.chat_with_tools_prepared(
                        prepared_chat,
                        agent_id,
                        conversation,
                        &no_tools,
                        request_seq,
                    );
                    super::provider_failure::observe_chat_result(&fallback);
                    match fallback {
                        Ok(t) => t,
                        Err(e) => {
                            sink.emit(&AkhMessage::error("llm", e.to_string()));
                            return;
                        }
                    }
                }
            }
        };

        emit_turn_reasoning(sink, &turn);

        let answer = turn.content.trim();
        if answer.is_empty() {
            sink.emit(&AkhMessage::system(format!(
                "(stopped: {})",
                stop.clause().to_lowercase()
            )));
            return;
        }
        conversation_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record_agent_turn(answer);
        // This is an interrupted answer, not a completed-turn training example.
        let _ = (engine, user_prompt, tool_trace);
        if used_fallback {
            sink.emit(&AkhMessage::narrative(answer, "acp"));
        }
    }

    /// Whether transcript capture (ADR 091) is enabled for this turn.
    ///
    /// `AKH_TRANSCRIPT_CAPTURE` overrides the per-akh config: `"0"` forces it
    /// off, `"1"` forces it on; unset (or any other value) falls back to
    /// [`Agent::capture_transcripts`](super::Agent). Capture is default-on
    /// (a sovereign akh's own memory), with this as its opt-out.
    fn transcript_capture_enabled(&self) -> bool {
        match std::env::var("AKH_TRANSCRIPT_CAPTURE").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => self.capture_transcripts,
        }
    }

    /// Build and record one ADR-091 turn capture on the WAL
    /// (`Engine::record_turn`) — a no-op if capture is opted out (see
    /// [`Self::transcript_capture_enabled`]) or no `MutationSink` is installed
    /// (in-memory/local-only engines).
    ///
    /// Called from every turn-end site in [`Self::run_agentic_turn`] (normal
    /// completion and [`Self::force_final_answer`]), so every surface that
    /// drives the agentic loop (daemon, TUI, MCP) captures identically.
    ///
    /// **The payload schema is LOCKED** — Wave 2's sleep-cycle harvest phase
    /// parses it verbatim (ADR 091 / the transcript-capture plan):
    /// `{v, ts, workspace, model, user_prompt, agent_answer, tools: [{tool, ok}]}`.
    /// Additive since (readers ignore unknown keys): per-tool `wrote`, `args`,
    /// `output`, `duration_ms`, and `dispatch: true` on autonomous turns (ADR 116 C2).
    pub(crate) fn emit_turn_wal(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        answer: &str,
        tools: &[Value],
    ) {
        let model = self.model_manager.as_deref().map(|m| m.model());
        self.emit_turn_wal_with_model(engine, agent_id, user_prompt, answer, tools, model, false);
    }

    /// External session runtimes supply the model returned by their provider,
    /// without mislabeling the turn as the dormant Ollama manager's model.
    pub(crate) fn emit_turn_wal_with_model(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        answer: &str,
        tools: &[Value],
        model: Option<&str>,
        dispatch: bool,
    ) {
        if !self.transcript_capture_enabled() {
            return;
        }
        let mut payload = json!({
            "v": 1,
            "ts": rfc3339_now(),
            "workspace": agent_id,
            "model": model,
            "user_prompt": user_prompt,
            "agent_answer": answer,
            "tools": tools,
        });
        // ADR 116 C2: an autonomous dispatch turn is not an operator turn; the
        // session harvester keeps it out of operator sessions. Additive and
        // present only when true, so the locked ADR 091 shape is otherwise unchanged.
        if dispatch {
            payload["dispatch"] = json!(true);
        }
        engine.record_turn(payload);
    }
}

/// Surface a turn's separately-returned chain-of-thought as a thinking step.
///
/// Streaming turns deliver thinking live as `StreamEvent::Thinking` and leave
/// [`AssistantTurn::reasoning`] empty, so this only fires for non-streaming
/// turns (the streaming-failure fallback, or a backend without native
/// streaming). It emits the same `AkhMessage::Reasoning` the streaming
/// handler does, before any answer text, and it is a trace-only channel: the
/// text is never merged into the visible answer and never echoed back into
/// the conversation the model sees.
fn emit_turn_reasoning(sink: &dyn MessageSink, turn: &crate::agent::ollama::AssistantTurn) {
    let reasoning = turn.reasoning.trim();
    if !reasoning.is_empty() {
        sink.emit(&AkhMessage::Reasoning {
            step: reasoning.to_string(),
            expression: None,
        });
    }
}

/// Build the Ollama/OpenAI `tools` catalog from every registered tool manifest.
///
/// Each tool becomes `{"type":"function","function":{name,description,parameters}}`
/// with a JSON-Schema `parameters` object derived from the manifest's parameter
/// list. Execution is still gated by the permission policy, so exposing every
/// tool here is safe — a dangerous one the model picks is denied at dispatch.
pub(crate) fn ollama_tool_catalog(registry: &ToolRegistry) -> Vec<Value> {
    // `list_advertised_manifests` (ADR 076: hides unadvertised tools) is HashMap-backed
    // (nondeterministic order); sort by name so the catalog — and any `AKH_AGENT_MAX_TOOLS`
    // truncation of it — is reproducible across runs.
    let mut manifests = registry.list_advertised_manifests();
    manifests.sort_by(|a, b| a.name.cmp(&b.name));
    manifests
        .into_iter()
        .map(|m| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();
            for p in &m.parameters {
                // A parameter carrying an explicit JSON-Schema fragment (ADR 076
                // §2) is emitted verbatim, with `description` merged in so the
                // model still sees the prose. Flat parameters keep the historical
                // `{type, description}` rendering.
                let prop = match &p.schema {
                    Some(serde_json::Value::Object(schema_obj)) => {
                        let mut merged = schema_obj.clone();
                        merged
                            .entry("description")
                            .or_insert_with(|| Value::String(p.description.clone()));
                        Value::Object(merged)
                    }
                    Some(other) => other.clone(),
                    None => {
                        let ty = if p.param_type.is_empty() {
                            "string"
                        } else {
                            p.param_type.as_str()
                        };
                        json!({"type": ty, "description": p.description})
                    }
                };
                properties.insert(p.name.clone(), prop);
                if p.required {
                    required.push(Value::String(p.name.clone()));
                }
            }
            json!({
                "type": "function",
                "function": {
                    "name": m.name,
                    "description": m.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        })
        .collect()
}

/// Re-add manifests named by a completion contract after catalog retrieval or
/// capping. A contract is not satisfiable if its evidence producers are hidden.
fn pin_required_tools(selected: &mut Vec<Value>, full: &[Value], required: &[&str]) -> Vec<String> {
    let mut offered: std::collections::HashSet<String> = selected
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str().map(String::from))
        .collect();
    let mut missing = Vec::new();
    for name in required {
        if offered.contains(*name) {
            continue;
        }
        if let Some(manifest) = full
            .iter()
            .find(|tool| tool["function"]["name"].as_str() == Some(*name))
        {
            selected.push(manifest.clone());
            offered.insert((*name).to_string());
        } else {
            tracing::warn!(
                tool = *name,
                "completion contract names an unavailable tool"
            );
            missing.push((*name).to_string());
        }
    }
    missing
}

/// Insert bounded host context adjacent to the active request while preserving
/// the context manager's ordering of core memory and recalled history.
fn insert_system_before_last_user(conversation: &mut Vec<Value>, content: String) {
    let at = conversation
        .iter()
        .rposition(|message| message["role"] == "user")
        .unwrap_or(conversation.len());
    conversation.insert(at, json!({"role": "system", "content": content}));
}

/// Phase 44m (ADR 065 D12.1): group the deferred-tool set — what was in
/// `full_catalog` but NOT in `tools` (the offered set) — by origin and reason
/// for the ACP `_akh/usage` panel. The reason is inferred from which env var
/// was active (call site): `AKH_AGENT_TOOL_RETRIEVAL` ⇒ `retrieval_scoped`,
/// `AKH_AGENT_MAX_TOOLS` ⇒ `capped`. `unadvertised` tools are not in
/// `full_catalog` (they're filtered upstream by `list_advertised_manifests`),
/// so they never appear here — the panel reports them as zero-count rather
/// than a misleading positive count.
fn compute_deferred_groups(
    full_catalog: &[Value],
    offered: &[Value],
    registry: &super::tool::ToolRegistry,
    retrieval_scoping_active: bool,
) -> Vec<super::context::DeferredToolGroup> {
    use super::context::{DeferredReason, DeferredToolGroup};
    use super::tool_manifest::ToolSource;
    use std::collections::HashSet;

    let offered_names: HashSet<String> = offered
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(String::from))
        .collect();

    // Reason: which env var was active at the call site? (Both can't be
    // active — the call site is if/else.) Neither active ⇒ no deferred set,
    // but the caller still calls us; return empty.
    let reason = if retrieval_scoping_active {
        DeferredReason::RetrievalScoped
    } else if std::env::var("AKH_AGENT_MAX_TOOLS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|cap| cap < full_catalog.len())
        .unwrap_or(false)
    {
        DeferredReason::Capped
    } else {
        // No scoping was active — `full_catalog == offered`. Nothing deferred.
        return Vec::new();
    };

    // Per-manifest origin lookup; fall back to the `mcp__` name prefix for
    // proxies registered before the `ToolSource::Mcp` variant existed.
    let manifest_origin = |name: &str| -> &'static str {
        if let Some(m) = registry.manifest(name) {
            return match &m.source {
                ToolSource::Native => "system",
                ToolSource::Wasm { .. } | ToolSource::Cli { .. } => "skills",
                ToolSource::Mcp { .. } => "mcp",
            };
        }
        if name.starts_with("mcp__") {
            "mcp"
        } else {
            "system"
        }
    };

    let mut groups: std::collections::HashMap<&'static str, (usize, usize)> =
        std::collections::HashMap::new();
    for tool in full_catalog {
        let name = match tool["function"]["name"].as_str() {
            Some(n) => n,
            None => continue,
        };
        if offered_names.contains(name) {
            continue;
        }
        let origin = manifest_origin(name);
        let tokens = tool.to_string().len().div_ceil(4); // heuristic, matches the budget counter
        let (t, c) = groups.entry(origin).or_insert((0, 0));
        *t += tokens;
        *c += 1;
    }

    let label_for = |origin: &str| -> &str {
        match origin {
            "system" => "System tools (deferred)",
            "skills" => "Skills (deferred)",
            "mcp" => "MCP tools (deferred)",
            _ => "Deferred",
        }
    };
    let id_for = |origin: &str| -> &'static str {
        match origin {
            "system" | "skills" => "tools",
            "mcp" => "mcp",
            _ => "tools",
        }
    };

    // Stable order: system, skills, mcp.
    let mut order = ["system", "skills", "mcp"];
    order.sort_unstable(); // alphabetical: mcp, skills, system — fine, stable
    order
        .into_iter()
        .filter_map(|origin| {
            let (tokens, count) = groups.get(origin).copied()?;
            Some(DeferredToolGroup {
                id: id_for(origin),
                label: label_for(origin).to_string(),
                tokens,
                count,
                reason,
            })
        })
        .collect()
}

/// Turn a model-supplied arguments value into a [`ToolInput`].
///
/// Ollama returns arguments as a JSON object; a non-object (or absent) value
/// yields an empty input, letting the tool report its own "missing param" error.
fn tool_input_from_args(args: &Value) -> ToolInput {
    match args {
        Value::Object(map) => ToolInput {
            params: map.clone(),
        },
        _ => ToolInput::new(),
    }
}

/// Tools whose successful result can support a durable study-note citation.
/// Keep this list narrow: task/status output that merely repeats a model value
/// must never launder that value into "retrieved evidence".
fn evidence_tool(name: &str) -> bool {
    matches!(
        name,
        "archival_search" | "study_note_read" | "web_search" | "http_fetch"
    ) || name.ends_with("__seshat_search")
}

fn study_note_write_tool(name: &str) -> bool {
    matches!(name, "study_note_append" | "study_note_revise")
}

/// Bind every study-note citation to one earlier successful retrieval result.
/// A single result must contain the claim's source id and locator; when the
/// model supplies a quote, that quote must occur in the retrieved content too.
fn verify_study_note_evidence(
    arguments: &Value,
    prior_evidence: &[ToolEvidence],
) -> Result<(), String> {
    let citations = arguments
        .get("citations")
        .and_then(Value::as_array)
        .filter(|citations| !citations.is_empty())
        .ok_or_else(|| "citations must be a non-empty array".to_string())?;

    for (index, citation) in citations.iter().enumerate() {
        let source_id = citation
            .get("source_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("citation {index} has no source_id"))?;
        let locator = citation
            .get("locator")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("citation {index} has no locator"))?;
        let quote = citation
            .get("quote")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("citation {index} has no supporting quote"))?;

        let supported = prior_evidence.iter().any(|evidence| {
            evidence_tool(&evidence.tool) && evidence_supports(evidence, source_id, locator, quote)
        });
        if !supported {
            return Err(format!(
                "citation {index} is not supported by one earlier retrieval result; retrieve the source passage, observe its locator, then retry in a later round"
            ));
        }
    }
    Ok(())
}

fn evidence_supports(evidence: &ToolEvidence, source_id: &str, locator: &str, quote: &str) -> bool {
    match serde_json::from_str::<Value>(&evidence.result) {
        Ok(value) => structured_evidence_supports(&value, source_id, locator, quote),
        Err(_) => plain_text_evidence_supports(evidence, source_id, locator, quote),
    }
}

/// Preserve the note boundaries emitted by `ArchivalSearchTool`. Its response
/// is intentionally human-readable rather than JSON, but each `### ...
/// (note:...)` block is still a distinct provenance scope. KG text before the
/// `Notes:` section is evaluated separately from every returned note.
fn plain_text_evidence_supports(
    evidence: &ToolEvidence,
    source_id: &str,
    locator: &str,
    quote: &str,
) -> bool {
    let scopes = if evidence.tool == "archival_search" {
        archival_evidence_scopes(&evidence.result)
    } else {
        vec![evidence.result.clone()]
    };
    scopes.iter().any(|scope| {
        evidence_scope_supports(&[normalize_evidence(scope)], source_id, locator, quote)
    })
}

fn archival_evidence_scopes(result: &str) -> Vec<String> {
    let result = result.replace("\r\n", "\n");
    let Some(notes_offset) = result.find("Notes:\n") else {
        return vec![result];
    };

    let mut scopes = Vec::new();
    let (prefix, notes) = result.split_at(notes_offset);
    if !prefix.trim().is_empty() {
        scopes.push(prefix.to_string());
    }

    let mut current = String::new();
    for line in notes.lines() {
        if is_archival_note_heading(line) {
            if !current.trim().is_empty() {
                scopes.push(std::mem::take(&mut current));
            }
            current.push_str(line);
            current.push('\n');
        } else if !current.is_empty() {
            current.push_str(line);
            current.push('\n');
        }
    }
    if !current.trim().is_empty() {
        scopes.push(current);
    }

    if scopes.is_empty() {
        scopes.push(result);
    }
    scopes
}

fn is_archival_note_heading(line: &str) -> bool {
    let Some((_, marker)) = line.rsplit_once(" (note:") else {
        return false;
    };
    let Some(note_id) = marker.strip_suffix(')') else {
        return false;
    };
    line.starts_with("### ")
        && note_id.strip_prefix("note-").is_some_and(|suffix| {
            suffix.len() == 32 && suffix.bytes().all(|b| b.is_ascii_hexdigit())
        })
}

/// Treat each returned search hit as its own provenance scope. A source ID in
/// one hit must never combine with a locator or quote from a sibling hit.
fn structured_evidence_supports(
    value: &Value,
    source_id: &str,
    locator: &str,
    quote: &str,
) -> bool {
    match value {
        Value::Array(values) => values
            .iter()
            .any(|value| structured_evidence_supports(value, source_id, locator, quote)),
        Value::Object(values) => {
            let mut saw_envelope = false;
            for key in ["results", "items", "documents", "matches", "chunks", "data"] {
                if let Some(nested) = values.get(key) {
                    match nested {
                        Value::Array(items) => {
                            saw_envelope = true;
                            if items.iter().any(|item| {
                                structured_evidence_supports(item, source_id, locator, quote)
                            }) {
                                return true;
                            }
                        }
                        Value::Object(_) => {
                            saw_envelope = true;
                            if structured_evidence_supports(nested, source_id, locator, quote) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            if saw_envelope {
                return false;
            }
            let mut content = Vec::new();
            collect_evidence_scalars(value, &mut content);
            evidence_scope_supports(&content, source_id, locator, quote)
        }
        _ => {
            let mut content = Vec::new();
            collect_evidence_scalars(value, &mut content);
            evidence_scope_supports(&content, source_id, locator, quote)
        }
    }
}

fn evidence_scope_supports(
    content: &[String],
    source_id: &str,
    locator: &str,
    quote: &str,
) -> bool {
    contains_bounded_evidence(content, source_id)
        && contains_bounded_evidence(content, locator)
        && contains_evidence(content, quote)
}

fn collect_evidence_scalars(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(value) => output.push(normalize_evidence(value)),
        Value::Number(value) => output.push(value.to_string()),
        Value::Array(values) => {
            for value in values {
                collect_evidence_scalars(value, output);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_evidence_scalars(value, output);
            }
        }
        Value::Null | Value::Bool(_) => {}
    }
}

fn contains_evidence(haystacks: &[String], needle: &str) -> bool {
    let needle = normalize_evidence(needle);
    !needle.is_empty() && haystacks.iter().any(|value| value.contains(&needle))
}

fn contains_bounded_evidence(haystacks: &[String], needle: &str) -> bool {
    let needle = normalize_evidence(needle);
    if needle.is_empty() {
        return false;
    }
    haystacks.iter().any(|value| {
        value.match_indices(&needle).any(|(start, matched)| {
            let end = start + matched.len();
            let before = value[..start].chars().next_back();
            let after = value[end..].chars().next();
            before.is_none_or(|ch| !evidence_identifier_char(ch))
                && after.is_none_or(|ch| !evidence_identifier_char(ch))
        })
    })
}

fn evidence_identifier_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':' | '/' | '\\')
}

fn normalize_evidence(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Compact one-line preview of a tool call's arguments for the narration line.
fn preview_args(args: &Value) -> String {
    let s = args.to_string();
    if s.chars().count() > ARG_PREVIEW_LEN {
        let truncated: String = s.chars().take(ARG_PREVIEW_LEN).collect();
        format!("{truncated}…")
    } else {
        s
    }
}

/// Whether `content` is a *rendered tool result*, not a real user query.
///
/// Mirrors the exact shape ollama's dedicated Qwen3.8 renderer treats as
/// "not a query" (`Qwen35Renderer.validateMessages`, `model/renderers/qwen35.go`
/// upstream): a `role:"user"` message whose trimmed content is wrapped in
/// `<tool_response>...</tool_response>`. That renderer is what some tool
/// backends (e.g. Qwen-Agent-style harnesses) use to carry tool results under
/// `role:"user"` instead of a dedicated `role:"tool"` — akhomed always uses
/// `role:"tool"` for results, so this should never match our own traffic, but
/// checking it (not just presence of *any* `role:"user"` message) matches
/// upstream's own definition of "a query" precisely.
fn is_tool_response_wrapped(content: &str) -> bool {
    let t = content.trim();
    t.starts_with("<tool_response>") && t.ends_with("</tool_response>")
}

/// Guarantee `conversation` carries at least one real user turn.
///
/// Background: ollama's dedicated Qwen3.8 renderer (`RENDERER qwen3.8`, the
/// tokenizer-verified port of froggeric's Qwen-Fixed-Chat-Templates that
/// `ptah-qwen3.8:sharp` uses to retain thinking-block fidelity) hard-fails a
/// request whose `messages` contains no qualifying `role:"user"` turn —
/// `HTTP 500 {"error":"no user query found in messages"}`. The same check
/// ships wherever the template does (ollama#17778, vllm#36432,
/// lmstudio-bug-tracker#1586, opencode#25168): it is a property of the
/// Qwen3.8 chat template itself, not an ollama- or akhomed-specific bug.
///
/// `PassThroughManager` and `CompactingManager` are both designed to always
/// carry the current turn's real user query (compacting.rs pins the most
/// recent turn first and keeps it unconditionally — see
/// `discourse::rank_turns_keep_first`), and the round loop below only ever
/// *appends* to `conversation`, never removes from it — so in the paths this
/// code can see, the invariant already holds. It has nonetheless been
/// observed failing live on a tool-loop continuation (round 2, right after
/// the first tool result), which is exactly the shape upstream's own bug
/// reports describe. Rather than chase every conceivable assembly path (or
/// every future `ContextManager` strategy) that could violate it, enforce it
/// once, right where every backend actually sends from: cheap (a linear scan
/// over what is typically a handful of messages), backend-agnostic (ollama,
/// candle, and openai_backend's `to_openai_messages` all translate this same
/// `conversation`, and vllm#36432 shows the OpenAI-contract path is not
/// immune either), and a no-op whenever the invariant already holds — which
/// is the overwhelming common case.
///
/// `user_prompt` is the turn's own triggering query — always a valid choice
/// to re-inject, and already in scope at every call site.
fn ensure_real_user_turn(conversation: &mut Vec<Value>, user_prompt: &str) {
    let has_real_user_turn = conversation.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("user")
            && !is_tool_response_wrapped(m.get("content").and_then(Value::as_str).unwrap_or(""))
    });
    if !has_real_user_turn {
        tracing::debug!(
            "conversation had no qualifying user turn before an ollama request; \
             re-injecting the triggering query (Qwen3.8 renderer compatibility)"
        );
        conversation.push(json!({"role": "user", "content": user_prompt}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::agent::error::AgentResult;
    use crate::agent::tool::{Tool, ToolOutput, ToolSignature};
    use crate::agent::tool_manifest::{
        DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
    };
    use std::collections::HashSet;

    fn call(name: &str, args: &str) -> (String, Value) {
        (name.to_string(), json!({ "q": args }))
    }

    #[test]
    fn study_note_evidence_gate_requires_an_earlier_matching_retrieval() {
        assert!(study_note_write_tool("study_note_append"));
        assert!(study_note_write_tool("study_note_revise"));
        assert!(!study_note_write_tool("memory_write"));
        assert!(evidence_tool("study_note_read"));
        assert!(!evidence_tool("study_note_list"));
        assert!(!evidence_tool("study_note_catalog"));
        let arguments = json!({
            "body": "The keph index is QZ-731.",
            "citations": [{
                "source_id": "book:alpha",
                "locator": "chapter 3",
                "quote": "The keph index is QZ-731."
            }]
        });
        assert!(verify_study_note_evidence(&arguments, &[]).is_err());

        let retrieved = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!([{
                "document_id": "book:alpha",
                "heading": "chapter 3",
                "text": "The keph index is QZ-731."
            }])
            .to_string(),
        };
        assert!(verify_study_note_evidence(&arguments, &[retrieved.clone()]).is_ok());

        let mut invented_quote = arguments.clone();
        invented_quote["citations"][0]["quote"] = json!("The keph index is XX-999.");
        assert!(verify_study_note_evidence(&invented_quote, &[retrieved.clone()]).is_err());

        let source_prefix_collision = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!([{
                "document_id": "book:alpha-2",
                "heading": "chapter 3",
                "text": "The keph index is QZ-731."
            }])
            .to_string(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[source_prefix_collision]).is_err(),
            "a longer sibling source id must not satisfy the cited identity"
        );

        let split_across_hits = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!([
                {
                    "document_id": "book:alpha",
                    "heading": "chapter 3",
                    "text": "This hit does not contain the cited claim."
                },
                {
                    "document_id": "book:beta",
                    "heading": "chapter 9",
                    "text": "The keph index is QZ-731."
                }
            ])
            .to_string(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[split_across_hits]).is_err(),
            "source, locator, and quote must co-occur in one returned hit"
        );

        let split_across_nested_hits = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!({
                "data": {
                    "results": [
                        {
                            "document_id": "book:alpha",
                            "heading": "chapter 3",
                            "text": "This hit does not contain the cited claim."
                        },
                        {
                            "document_id": "book:beta",
                            "heading": "chapter 9",
                            "text": "The keph index is QZ-731."
                        }
                    ]
                }
            })
            .to_string(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[split_across_nested_hits]).is_err(),
            "nested result envelopes must preserve per-hit provenance boundaries"
        );

        let later_matching_envelope = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!({
                "results": [{
                    "document_id": "book:beta",
                    "heading": "chapter 9",
                    "text": "An unrelated result."
                }],
                "data": [{
                    "document_id": "book:alpha",
                    "heading": "chapter 3",
                    "text": "The keph index is QZ-731."
                }]
            })
            .to_string(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[later_matching_envelope]).is_ok(),
            "independent envelope keys must each be checked without coalescing them"
        );

        let split_across_archival_notes = ToolEvidence {
            tool: "archival_search".into(),
            result: concat!(
                "Notes:\n",
                "### Alpha (note:note-11111111111111111111111111111111)\n",
                "Frontmatter:\n",
                "sources: [book:alpha]\n",
                "citations: [book:alpha#chapter 3]\n",
                "This note does not contain the cited claim.\n\n",
                "### Beta (note:note-22222222222222222222222222222222)\n",
                "Frontmatter:\n",
                "sources: [book:beta]\n",
                "citations: [book:beta#chapter 9]\n",
                "The keph index is QZ-731.\n\n"
            )
            .into(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[split_across_archival_notes]).is_err(),
            "source, locator, and quote must not combine across plain-text note blocks"
        );

        let repeated_by_non_evidence_tool = ToolEvidence {
            tool: "task_list".into(),
            ..retrieved
        };
        assert!(verify_study_note_evidence(&arguments, &[repeated_by_non_evidence_tool]).is_err());

        let missing_provenance_in_result = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!({"text": "The keph index is QZ-731."}).to_string(),
        };
        assert!(
            verify_study_note_evidence(&arguments, &[missing_provenance_in_result]).is_err(),
            "model-supplied retrieval arguments are not source provenance"
        );
    }

    #[test]
    fn real_archival_result_carries_note_provenance_for_a_later_write() {
        let engine = test_engine();
        let tmp = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(
            crate::library::PersistentNoteCatalog::open(tmp.path().join("notes")).unwrap(),
        );
        let writer = crate::agent::tools::StudyNoteAppendTool::new(std::sync::Arc::clone(&store));
        writer
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("title", "Alpha")
                    .with_param("body", "The keph index is QZ-731.")
                    .with_value(
                        "citations",
                        json!([{
                            "source_id": "book:alpha",
                            "locator": "chapter 3",
                            "quote": "The keph index is QZ-731."
                        }]),
                    ),
            )
            .unwrap();
        let search = crate::agent::tools::ArchivalSearchTool::with_notes(store);
        let result = search
            .execute(&engine, ToolInput::new().with_param("query", "QZ-731"))
            .unwrap();
        let evidence = ToolEvidence {
            tool: "archival_search".into(),
            result: result.result,
        };
        let revision = json!({
            "body": "The verified keph index remains QZ-731.",
            "citations": [{
                "source_id": "book:alpha",
                "locator": "chapter 3",
                "quote": "The keph index is QZ-731."
            }]
        });

        assert!(verify_study_note_evidence(&revision, &[evidence]).is_ok());
    }

    #[test]
    fn real_dispatch_cannot_persist_a_note_before_evidence() {
        use crate::library::NoteCatalog;

        let engine = test_engine();
        let tmp = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(
            crate::library::PersistentNoteCatalog::open(tmp.path().join("notes")).unwrap(),
        );
        let agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent.register_tool(Box::new(crate::agent::tools::StudyNoteAppendTool::new(
            std::sync::Arc::clone(&store),
        )));
        let known_tools: std::collections::HashSet<String> = agent
            .list_tools()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        let full_catalog = {
            let registry = agent
                .tool_registry
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            ollama_tool_catalog(&registry)
        };
        let call = crate::agent::ollama::ToolCall {
            name: "study_note_append".into(),
            arguments: json!({
                "title": "Alpha",
                "body": "The keph index is QZ-731.",
                "citations": [{
                    "source_id": "book:alpha",
                    "locator": "chapter 3",
                    "quote": "The keph index is QZ-731."
                }]
            }),
        };

        let blocked = agent.execute_tool_call(
            &engine,
            &call,
            &known_tools,
            &full_catalog,
            "learn alpha",
            &[],
        );
        assert!(!blocked.success);
        assert!(store.frontmatter_only().is_empty());

        let evidence = ToolEvidence {
            tool: "mcp__seshat__seshat_search".into(),
            result: json!([{
                "document_id": "book:alpha",
                "heading": "chapter 3",
                "text": "The keph index is QZ-731."
            }])
            .to_string(),
        };
        let written = agent.execute_tool_call(
            &engine,
            &call,
            &known_tools,
            &full_catalog,
            "learn alpha",
            &[evidence],
        );
        assert!(written.success, "{}", written.result);
        assert_eq!(store.frontmatter_only().len(), 1);
        assert_eq!(written.trace["evidence_verified"], true);
        // ADR 116 C2: the trace doubles as the captured receipt, bounded.
        assert!(
            written.trace["args"]
                .as_str()
                .is_some_and(|a| !a.is_empty())
        );
        assert_eq!(written.trace["output"], written.result.as_str());
        assert!(written.trace["duration_ms"].is_u64());
    }

    struct StaticArchivalEvidenceTool;

    impl Tool for StaticArchivalEvidenceTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "archival_search".into(),
                description: "test archival evidence".into(),
                parameters: vec![crate::agent::tool::ToolParam {
                    name: "query".into(),
                    description: "query".into(),
                    required: true,
                }],
            }
        }

        fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok(
                json!([{
                    "document_id": "book:alpha",
                    "heading": "chapter 3",
                    "text": "The keph index is QZ-731."
                }])
                .to_string(),
            ))
        }

        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "archival_search".into(),
                description: "test archival evidence".into(),
                parameters: vec![ToolParamSchema::required("query", "query")],
                danger: DangerInfo {
                    effect: ToolEffect::Mutating,
                    level: DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: String::new(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    struct ParallelThenSequentialNoteBackend {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl ParallelThenSequentialNoteBackend {
        fn note_call() -> crate::agent::ollama::ToolCall {
            crate::agent::ollama::ToolCall {
                name: "study_note_append".into(),
                arguments: json!({
                    "title": "Alpha",
                    "body": "The keph index is QZ-731.",
                    "citations": [{
                        "source_id": "book:alpha",
                        "locator": "chapter 3",
                        "quote": "The keph index is QZ-731."
                    }]
                }),
            }
        }
    }

    impl crate::agent::chat_backend::ChatBackend for ParallelThenSequentialNoteBackend {
        fn label(&self) -> &str {
            "parallel-then-sequential-note"
        }

        fn available(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let round = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(match round {
                0 => crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![
                        crate::agent::ollama::ToolCall {
                            name: "archival_search".into(),
                            arguments: json!({"query": "alpha"}),
                        },
                        Self::note_call(),
                    ],
                    ..Default::default()
                },
                1 => crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![Self::note_call()],
                    ..Default::default()
                },
                _ => crate::agent::ollama::AssistantTurn {
                    content: "The note is now grounded and durable.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                },
            })
        }
    }

    #[test]
    fn same_round_retrieval_cannot_authorize_a_note_write() {
        use crate::library::NoteCatalog;

        let engine = test_engine();
        let tmp = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(
            crate::library::PersistentNoteCatalog::open(tmp.path().join("notes")).unwrap(),
        );
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent.register_tool(Box::new(StaticArchivalEvidenceTool));
        agent.register_tool(Box::new(crate::agent::tools::StudyNoteAppendTool::new(
            std::sync::Arc::clone(&store),
        )));
        agent.set_chat_backend(std::sync::Arc::new(ParallelThenSequentialNoteBackend {
            calls: std::sync::atomic::AtomicUsize::new(0),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "learn alpha",
                &sink,
                &cancel,
                false,
                &agent.conversation_state,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(store.frontmatter_only().len(), 1);
        let append_results: Vec<_> = sink
            .messages()
            .into_iter()
            .filter_map(|message| match message {
                AkhMessage::ToolResult {
                    tool,
                    success,
                    output,
                } if tool == "study_note_append" => Some((success, output)),
                _ => None,
            })
            .collect();
        assert_eq!(append_results.len(), 2);
        assert!(!append_results[0].0);
        assert!(append_results[0].1.contains("evidence gate"));
        assert!(append_results[1].0);
    }

    /// Immediate-answer backend that records the exact context supplied for
    /// every turn, used to prove ACP-style conversation states retain their
    /// own history without inheriting a neighboring session's transcript.
    struct RecordingAnswerBackend {
        conversations: std::sync::Arc<std::sync::Mutex<Vec<Vec<Value>>>>,
        answer_seq: std::sync::atomic::AtomicUsize,
    }

    /// Image-grounding backend used to prove the ACP-shaped bytes reach the
    /// pre-tool boundary exactly once and survive in only that session.
    struct RecordingImageBackend {
        conversations: std::sync::Arc<std::sync::Mutex<Vec<Vec<Value>>>>,
        ground_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    /// Backend that pauses every caller inside inference until the test
    /// releases it. If a shared conversation mutex is held across inference,
    /// the second caller cannot report entry and the regression test fails.
    struct BlockingAnswerBackend {
        entered: std::sync::mpsc::Sender<()>,
        release: std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
    }

    impl crate::agent::chat_backend::ChatBackend for RecordingAnswerBackend {
        fn label(&self) -> &str {
            "recording-answer"
        }

        fn available(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.conversations
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conversation.to_vec());
            let answer = self
                .answer_seq
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(crate::agent::ollama::AssistantTurn {
                content: format!("answer-{answer}"),
                tool_calls: vec![],
                ..Default::default()
            })
        }
    }

    impl crate::agent::chat_backend::ChatBackend for RecordingImageBackend {
        fn label(&self) -> &str {
            "recording-image"
        }

        fn available(&self) -> bool {
            true
        }

        fn ground_prompt_images(
            &self,
            images: &[crate::agent::chat_backend::ChatPromptImage],
        ) -> Result<Option<String>, crate::agent::chat_backend::ChatBackendError> {
            if images.is_empty() {
                return Ok(None);
            }
            assert_eq!(images.len(), 1);
            assert_eq!(images[0].mime_type, "image/png");
            assert_eq!(images[0].bytes, b"png-payload");
            self.ground_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Some(
                "# Native Visual Evidence\n\n[image 1 OCR]\nERR 503 upstream\n[/image 1 OCR]"
                    .into(),
            ))
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.conversations
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(conversation.to_vec());
            Ok(crate::agent::ollama::AssistantTurn {
                content: "grounded answer".into(),
                tool_calls: vec![],
                ..Default::default()
            })
        }
    }

    impl crate::agent::chat_backend::ChatBackend for BlockingAnswerBackend {
        fn label(&self) -> &str {
            "blocking-answer"
        }

        fn available(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.entered.send(()).expect("test receiver remains alive");
            let (lock, wake) = &*self.release;
            let mut released = lock.lock().unwrap_or_else(|e| e.into_inner());
            while !*released {
                released = wake.wait(released).unwrap_or_else(|e| e.into_inner());
            }
            Ok(crate::agent::ollama::AssistantTurn {
                content: "complete".into(),
                tool_calls: vec![],
                ..Default::default()
            })
        }
    }

    #[test]
    fn supplied_conversation_states_retain_history_without_cross_session_bleed() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        agent.set_chat_backend(std::sync::Arc::new(RecordingAnswerBackend {
            conversations: std::sync::Arc::clone(&observed),
            answer_seq: std::sync::atomic::AtomicUsize::new(0),
        }));
        agent.set_context_manager(std::sync::Arc::new(
            crate::agent::context::PassiveManager::with_tail(10),
        ));

        let state_a = std::sync::Mutex::new(ConversationState::new("session-a", "narrative"));
        let state_b = std::sync::Mutex::new(ConversationState::new("session-b", "narrative"));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "alpha-first",
                &sink,
                &cancel,
                false,
                &state_a,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "bravo-first",
                &sink,
                &cancel,
                false,
                &state_b,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "alpha-second",
                &sink,
                &cancel,
                false,
                &state_a,
            ),
            Some(AgenticTurnOutcome::Completed)
        );

        let observed = observed.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(observed.len(), 3);
        let session_b_prompt = serde_json::to_string(&observed[1]).unwrap();
        assert!(session_b_prompt.contains("bravo-first"));
        assert!(!session_b_prompt.contains("alpha-first"));
        assert!(!session_b_prompt.contains("answer-0"));

        let session_a_follow_up = serde_json::to_string(&observed[2]).unwrap();
        assert!(session_a_follow_up.contains("alpha-first"));
        assert!(session_a_follow_up.contains("answer-0"));
        assert!(session_a_follow_up.contains("alpha-second"));
        assert!(!session_a_follow_up.contains("bravo-first"));
        assert!(!session_a_follow_up.contains("answer-1"));

        assert_eq!(
            state_a
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .turns
                .len(),
            4,
            "session A must retain both of its complete exchanges"
        );
        assert_eq!(
            state_b
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .turns
                .len(),
            2,
            "session B must contain only its own exchange"
        );
    }

    #[test]
    fn image_bytes_are_grounded_once_and_retained_for_session_follow_up() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let ground_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(RecordingImageBackend {
            conversations: std::sync::Arc::clone(&observed),
            ground_calls: std::sync::Arc::clone(&ground_calls),
        }));
        agent.set_context_manager(std::sync::Arc::new(
            crate::agent::context::PassiveManager::with_tail(10),
        ));

        let state = std::sync::Mutex::new(ConversationState::new("image-session", "narrative"));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let image = crate::agent::chat_backend::ChatPromptImage {
            mime_type: "image/png".into(),
            bytes: b"png-payload".to_vec(),
        };

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state_and_images(
                &engine,
                "thoth",
                "What failed?",
                &[image],
                &sink,
                &cancel,
                false,
                &state,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "Can I retry locally?",
                &sink,
                &cancel,
                false,
                &state,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(
            ground_calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "follow-up must reuse retained evidence without grounding again"
        );

        let observed = observed.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(observed.len(), 2);
        let first = serde_json::to_string(&observed[0]).unwrap();
        assert!(first.contains("Native Visual Evidence"));
        assert!(first.contains("ERR 503 upstream"));
        let follow_up = serde_json::to_string(&observed[1]).unwrap();
        assert!(follow_up.contains("ERR 503 upstream"));
        assert!(follow_up.contains("Can I retry locally?"));

        let state = state.lock().unwrap_or_else(|error| error.into_inner());
        assert_eq!(state.turns.len(), 4);
        assert!(state.turns[0].text.contains("ERR 503 upstream"));
        assert!(!state.turns[2].text.contains("ERR 503 upstream"));
    }

    #[test]
    fn distinct_conversation_states_do_not_serialize_backend_inference() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let release =
            std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        agent.set_chat_backend(std::sync::Arc::new(BlockingAnswerBackend {
            entered: entered_tx,
            release: std::sync::Arc::clone(&release),
        }));

        let state_a = std::sync::Mutex::new(ConversationState::new("session-a", "narrative"));
        let state_b = std::sync::Mutex::new(ConversationState::new("session-b", "narrative"));

        let both_entered_before_release = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                let sink = crate::message::VecSink::new();
                let cancel = std::sync::atomic::AtomicBool::new(false);
                agent.run_agentic_turn_with_conversation_state(
                    &engine,
                    "thoth",
                    "session-a prompt",
                    &sink,
                    &cancel,
                    false,
                    &state_a,
                )
            });
            let second = scope.spawn(|| {
                let sink = crate::message::VecSink::new();
                let cancel = std::sync::atomic::AtomicBool::new(false);
                agent.run_agentic_turn_with_conversation_state(
                    &engine,
                    "thoth",
                    "session-b prompt",
                    &sink,
                    &cancel,
                    false,
                    &state_b,
                )
            });

            let first_entered = entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_ok();
            let second_entered = entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .is_ok();

            let (lock, wake) = &*release;
            *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
            wake.notify_all();

            assert_eq!(
                first.join().expect("first turn thread"),
                Some(AgenticTurnOutcome::Completed)
            );
            assert_eq!(
                second.join().expect("second turn thread"),
                Some(AgenticTurnOutcome::Completed)
            );
            first_entered && second_entered
        });

        assert!(
            both_entered_before_release,
            "two ACP sessions must reach backend inference concurrently; no shared transcript lock may span inference"
        );
    }

    // ── ensure_real_user_turn (Qwen3.8 "no user query" 500 hardening) ──────

    #[test]
    fn ensure_real_user_turn_is_a_noop_when_the_original_query_is_present() {
        // The exact shape a tool-loop continuation sends: the current
        // question, then an assistant tool call, then its result — nothing
        // since. The original query is still there (just not last), so
        // nothing should be appended.
        let mut conv = vec![
            json!({"role": "user", "content": "what's the weather?"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "weather", "arguments": {}}}
            ]}),
            json!({"role": "tool", "content": "72F, sunny"}),
        ];
        let before = conv.len();
        ensure_real_user_turn(&mut conv, "what's the weather?");
        assert_eq!(
            conv.len(),
            before,
            "already-present query must not be duplicated"
        );
    }

    #[test]
    fn ensure_real_user_turn_reinjects_when_missing() {
        // Simulates whatever assembly path could (now or in the future) hand
        // the loop a conversation with no user turn at all.
        let mut conv = vec![
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "weather", "arguments": {}}}
            ]}),
            json!({"role": "tool", "content": "72F, sunny"}),
        ];
        ensure_real_user_turn(&mut conv, "what's the weather?");
        assert_eq!(conv.len(), 3);
        assert_eq!(conv[2]["role"], "user");
        assert_eq!(conv[2]["content"], "what's the weather?");
    }

    #[test]
    fn ensure_real_user_turn_does_not_count_tool_response_wrapped_user_messages() {
        // Some harnesses (not akhomed — it always uses role:"tool" — but the
        // upstream template is defined generically) carry tool results under
        // role:"user" wrapped in <tool_response>...</tool_response>. Per
        // ollama's own `validateMessages`, that does NOT count as a real
        // query, so neither should ours.
        let mut conv = vec![
            json!({"role": "user", "content": "<tool_response>\n72F, sunny\n</tool_response>"}),
        ];
        ensure_real_user_turn(&mut conv, "what's the weather?");
        assert_eq!(
            conv.len(),
            2,
            "the wrapped message must not satisfy the check"
        );
        assert_eq!(conv[1]["role"], "user");
        assert_eq!(conv[1]["content"], "what's the weather?");
    }

    #[test]
    fn ensure_real_user_turn_accepts_a_historical_user_turn_anywhere_in_the_array() {
        // The check (like upstream's) scans the whole array, not just the
        // tail — an earlier real user turn (e.g. from compacted history)
        // still counts even though several assistant/tool turns follow it.
        let mut conv = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "first question"}),
            json!({"role": "assistant", "content": "answer"}),
            json!({"role": "user", "content": "second question"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": {}}}
            ]}),
            json!({"role": "tool", "content": "result a"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "b", "arguments": {}}}
            ]}),
            json!({"role": "tool", "content": "result b"}),
        ];
        let before = conv.len();
        ensure_real_user_turn(&mut conv, "second question");
        assert_eq!(conv.len(), before);
    }

    #[test]
    fn read_and_shell_success_are_not_write_evidence() {
        assert!(!verified_write("file_io", &json!({"action":"read"}), true));
        assert!(!verified_write(
            "shell_exec",
            &json!({"command":"cat code.rs"}),
            true
        ));
        assert!(verified_write("file_io", &json!({"action":"write"}), true));
        assert!(!verified_write("edit", &json!({}), false));
    }

    #[test]
    fn repository_evidence_tracks_partial_shell_edits() {
        let dir = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            assert!(
                std::process::Command::new("git")
                    .args(args)
                    .current_dir(dir.path())
                    .output()
                    .unwrap()
                    .status
                    .success()
            )
        };
        git(&["init", "-b", "main"]);
        git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@local",
            "commit",
            "--allow-empty",
            "-m",
            "base",
        ]);
        let before = repository_witness(dir.path()).unwrap();
        assert_eq!(Some(before), repository_witness(dir.path()));
        std::fs::write(dir.path().join("code.rs"), "partial work").unwrap();
        let after = repository_witness(dir.path()).unwrap();
        assert_ne!(before, after);
        std::fs::write(dir.path().join("code.rs"), "more partial work").unwrap();
        assert_ne!(Some(after), repository_witness(dir.path()));
    }

    /// A backend that records every exec request and delegates to an
    /// in-process backend over a sandbox rooted at the test repository.
    struct RecordingExecBackend {
        inner: crate::backend::InProcessBackend,
        requests: std::sync::Mutex<Vec<crate::backend::ExecRequest>>,
    }

    impl RecordingExecBackend {
        fn new(root: &std::path::Path) -> Self {
            Self {
                inner: crate::backend::InProcessBackend::new(std::sync::Arc::new(
                    crate::fs::LocalFs::new(vec![root.to_path_buf()], vec![]),
                )),
                requests: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<crate::backend::ExecRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl crate::fs::Filesystem for RecordingExecBackend {
        fn read_to_string(&self, path: &std::path::Path) -> crate::fs::FsResult<String> {
            self.inner.read_to_string(path)
        }
        fn write(&self, path: &std::path::Path, contents: &str) -> crate::fs::FsResult<()> {
            self.inner.write(path, contents)
        }
        fn read_dir(&self, path: &std::path::Path) -> crate::fs::FsResult<Vec<std::path::PathBuf>> {
            self.inner.read_dir(path)
        }
        fn exists(&self, path: &std::path::Path) -> bool {
            self.inner.exists(path)
        }
        fn read_roots(&self) -> &[std::path::PathBuf] {
            self.inner.read_roots()
        }
        fn is_writable(&self) -> bool {
            self.inner.is_writable()
        }
        fn file_size(&self, path: &std::path::Path) -> crate::fs::FsResult<u64> {
            self.inner.file_size(path)
        }
        fn is_symlink(&self, path: &std::path::Path) -> bool {
            self.inner.is_symlink(path)
        }
    }

    impl crate::backend::Backend for RecordingExecBackend {
        fn exec(
            &self,
            request: crate::backend::ExecRequest,
        ) -> crate::backend::ExecResult<crate::backend::ExecOutcome> {
            self.requests.lock().unwrap().push(request.clone());
            self.inner.exec(request)
        }
        fn handshake(&self) -> crate::backend::Posture {
            self.inner.handshake()
        }
        fn locus(&self) -> crate::backend::ExecLocus {
            self.inner.locus()
        }
    }

    fn init_witness_repo(dir: &std::path::Path) {
        let git = |args: &[&str]| {
            assert!(
                std::process::Command::new("git")
                    .args(args)
                    .current_dir(dir)
                    .output()
                    .unwrap()
                    .status
                    .success()
            )
        };
        git(&["init", "-b", "main"]);
        std::fs::write(dir.join("tracked.rs"), "base").unwrap();
        git(&["add", "tracked.rs"]);
        git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@local",
            "commit",
            "-m",
            "base",
        ]);
    }

    #[test]
    fn witness_route_resolves_from_deployment_mode_and_wiring() {
        use crate::agent::ExecToolWiring;
        use crate::deployment::DeploymentMode;
        let backend: std::sync::Arc<dyn crate::backend::Backend> =
            std::sync::Arc::new(crate::backend::InProcessBackend::exec_only());
        for wiring in [
            ExecToolWiring::Default,
            ExecToolWiring::None,
            ExecToolWiring::Backend(backend.clone()),
        ] {
            assert!(matches!(
                WitnessRoute::resolve(DeploymentMode::Development, &wiring),
                WitnessRoute::InProcess
            ));
        }
        assert!(matches!(
            WitnessRoute::resolve(DeploymentMode::Deployed, &ExecToolWiring::Backend(backend)),
            WitnessRoute::Backend(_)
        ));
        assert!(matches!(
            WitnessRoute::resolve(DeploymentMode::Deployed, &ExecToolWiring::Default),
            WitnessRoute::Unavailable(WitnessUnavailable::InProcessOnly)
        ));
        assert!(matches!(
            WitnessRoute::resolve(DeploymentMode::Deployed, &ExecToolWiring::None),
            WitnessRoute::Unavailable(WitnessUnavailable::NoExecutor)
        ));
    }

    #[test]
    fn deployed_witness_runs_git_through_the_backend() {
        let dir = tempfile::tempdir().unwrap();
        init_witness_repo(dir.path());
        let backend = std::sync::Arc::new(RecordingExecBackend::new(dir.path()));
        let route = WitnessRoute::Backend(backend.clone());

        let before = route.witness(dir.path()).unwrap();
        assert_eq!(Some(before), route.witness(dir.path()));

        // Every git call went through Backend::exec, in the checkout, with
        // optional locks disabled.
        let requests = backend.requests();
        assert_eq!(requests.len(), 6, "three git calls per witness");
        for request in &requests {
            assert!(request.command.starts_with("git "), "{}", request.command);
            assert_eq!(request.cwd.as_deref(), Some(dir.path()));
            assert!(
                request
                    .env
                    .contains(&("GIT_OPTIONAL_LOCKS".to_string(), "0".to_string()))
            );
        }
        assert!(requests[1].command.contains("--no-ext-diff"));
        assert!(requests[1].command.contains("--no-textconv"));

        // An untracked file, then a growing one, then a tracked edit all move
        // the witness.
        std::fs::write(dir.path().join("code.rs"), "partial work").unwrap();
        let untracked = route.witness(dir.path()).unwrap();
        assert_ne!(before, untracked);
        std::fs::write(dir.path().join("code.rs"), "more partial work").unwrap();
        let grown = route.witness(dir.path()).unwrap();
        assert_ne!(untracked, grown);
        std::fs::write(dir.path().join("tracked.rs"), "edited").unwrap();
        assert_ne!(Some(grown), route.witness(dir.path()));
    }

    #[test]
    fn unavailable_witness_is_skipped_without_spawning() {
        let dir = tempfile::tempdir().unwrap();
        init_witness_repo(dir.path());
        for reason in [
            WitnessUnavailable::NoExecutor,
            WitnessUnavailable::InProcessOnly,
        ] {
            assert_eq!(WitnessRoute::Unavailable(reason).witness(dir.path()), None);
        }
        // Development still witnesses in-process.
        assert!(WitnessRoute::InProcess.witness(dir.path()).is_some());
    }

    #[test]
    fn round_budget_scales_cap_to_context_window() {
        assert_eq!(RoundBudget::new(Some(1_000_000)).hard_max, 30); // GLM-class
        assert_eq!(RoundBudget::new(Some(262_144)).hard_max, 30); // gemma4-12b
        assert_eq!(RoundBudget::new(Some(128_000)).hard_max, 20);
        assert_eq!(RoundBudget::new(Some(8_192)).hard_max, DEFAULT_MAX_ROUNDS);
        assert_eq!(RoundBudget::new(None).hard_max, DEFAULT_MAX_ROUNDS);
    }

    #[test]
    fn autonomous_productive_work_outlives_interactive_limits() {
        let mut b = RoundBudget::new(None).for_dispatch();
        b.started = Instant::now() - Duration::from_secs(3600);
        for i in 0..100 {
            b.note_request();
            let c = call("kg_query", &i.to_string());
            b.note_round([(c.0.as_str(), &c.1)]);
            assert_eq!(b.exhausted(), None);
        }
        let c = call("kg_query", "stuck");
        for _ in 0..DEFAULT_REPEAT_LIMIT {
            b.note_round([(c.0.as_str(), &c.1)]);
        }
        assert_eq!(b.exhausted(), Some(TurnStop::RepeatLoop));
    }

    #[test]
    fn a_completed_long_tool_clears_the_wedge_clock() {
        let mut b = RoundBudget::new(None).for_dispatch();
        b.last_request = Instant::now() - Duration::from_secs(120);
        assert_eq!(b.exhausted(), Some(TurnStop::WedgeTimeout));
        b.note_request();
        assert_eq!(b.exhausted(), None);
    }

    #[test]
    fn round_budget_stops_at_iteration_cap() {
        let mut b = RoundBudget::new(None); // hard_max = DEFAULT_MAX_ROUNDS (12)
        for i in 0..b.hard_max {
            assert!(b.exhausted().is_none(), "should run at round {i}");
            // distinct call each round → never trips the repeat detector
            let c = call("kg_query", &i.to_string());
            b.note_round([(c.0.as_str(), &c.1)]);
        }
        assert_eq!(b.exhausted(), Some(TurnStop::IterationCap));
    }

    #[test]
    fn round_budget_detects_repeat_loop_before_the_cap() {
        let mut b = RoundBudget::new(Some(1_000_000)); // roomy cap (30)
        let c = call("memory_recall", "same"); // identical signature every round
        for _ in 0..DEFAULT_REPEAT_LIMIT {
            assert!(b.exhausted().is_none());
            b.note_round([(c.0.as_str(), &c.1)]);
        }
        // Spun the same call `repeat_limit` times — stop well before round 30.
        assert_eq!(b.exhausted(), Some(TurnStop::RepeatLoop));
        assert!(b.round < b.hard_max);
    }

    #[test]
    fn round_budget_distinct_calls_never_trip_repeat_detector() {
        let mut b = RoundBudget::new(Some(1_000_000));
        for i in 0..10 {
            let c = call("kg_query", &i.to_string());
            b.note_round([(c.0.as_str(), &c.1)]);
        }
        assert!(!b.looped);
    }

    struct DummyTool;
    impl Tool for DummyTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "echo".into(),
                description: "Echo the text back".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _e: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok(format!(
                "echoed: {}",
                input.get("text").unwrap_or("")
            )))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "echo".into(),
                description: "Echo the text back".into(),
                parameters: vec![ToolParamSchema::required("text", "text to echo")],
                danger: DangerInfo {
                    effect: ToolEffect::Mutating,
                    level: DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: String::new(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    #[test]
    fn catalog_maps_manifest_to_function_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let catalog = ollama_tool_catalog(&reg);
        assert_eq!(catalog.len(), 1);
        let f = &catalog[0]["function"];
        assert_eq!(f["name"], "echo");
        assert_eq!(f["parameters"]["properties"]["text"]["type"], "string");
        assert_eq!(f["parameters"]["required"][0], "text");
    }

    #[test]
    fn args_object_becomes_tool_input() {
        let args = json!({"text": "hi", "n": 3});
        let input = tool_input_from_args(&args);
        assert_eq!(input.get("text"), Some("hi"));
        assert_eq!(input.get_i64("n"), Some(3));
    }

    #[test]
    fn non_object_args_yield_empty_input() {
        assert!(tool_input_from_args(&Value::Null).params.is_empty());
        assert!(tool_input_from_args(&json!("oops")).params.is_empty());
    }

    #[test]
    fn preview_truncates_long_args() {
        let big = json!({"blob": "x".repeat(500)});
        let p = preview_args(&big);
        assert!(p.ends_with('…'));
        assert!(p.chars().count() <= ARG_PREVIEW_LEN + 1);
    }

    // --- ADR 091: transcript→episodic capture (write-side first slice) ---

    #[test]
    fn rfc3339_now_has_known_shape_and_matches_a_reference_instant() {
        // Format shape, independent of "now".
        let s = rfc3339_now();
        assert_eq!(s.len(), "2026-07-10T00:00:00.000000000Z".len());
        assert!(s.ends_with('Z'));
        assert_eq!(s.as_bytes()[4], b'-');
        assert_eq!(s.as_bytes()[7], b'-');
        assert_eq!(s.as_bytes()[10], b'T');

        // civil_from_days against known reference instants (cross-checked
        // against `datetime.utcfromtimestamp` — the algorithm is Howard
        // Hinnant's well-known `civil_from_days`).
        assert_eq!(civil_from_days(0), (1970, 1, 1)); // unix epoch
        assert_eq!(civil_from_days(1_700_000_000 / 86_400), (2023, 11, 14));
    }

    fn test_engine() -> std::sync::Arc<Engine> {
        std::sync::Arc::new(
            Engine::new(crate::engine::EngineConfig::default())
                .expect("engine construction should succeed"),
        )
    }

    /// A [`MutationSink`](crate::mutation::MutationSink) that records every
    /// mutation it receives, for asserting on the built payload without a live
    /// WAL/Postgres.
    struct RecordingSink {
        records: std::sync::Mutex<Vec<(crate::mutation::MutationKind, Value)>>,
    }

    impl crate::mutation::MutationSink for RecordingSink {
        fn record(&self, mutation: crate::mutation::MutationRecord) {
            self.records
                .lock()
                .unwrap()
                .push((mutation.kind, mutation.payload));
        }
    }

    // All three ADR-091 behaviors below (gate precedence, payload shape,
    // opt-out no-op) are asserted in ONE test function rather than split
    // across several `#[test]`s. `AKH_TRANSCRIPT_CAPTURE` is process-global
    // state and cargo runs tests in parallel threads within one binary, so
    // several tests each mutating the same fixed (schema-locked, cannot be
    // renamed) env var would race each other; sequencing them here is the
    // simplest race-free option without pulling in a `serial_test` dependency.
    #[test]
    fn transcript_capture_gate_and_emit_turn_wal() {
        // SAFETY: edition-2024 requires explicit unsafe for env mutation;
        // this test owns AKH_TRANSCRIPT_CAPTURE for its duration and clears
        // it before returning (including via the early assertions below).
        unsafe {
            std::env::remove_var("AKH_TRANSCRIPT_CAPTURE");
        }

        // --- 1. Gate precedence: env overrides the config field. ---
        let mut agent = Agent::new(test_engine(), AgentConfig::default()).unwrap();
        agent.capture_transcripts = true;
        assert!(
            agent.transcript_capture_enabled(),
            "unset env falls back to the config field (on)"
        );
        agent.capture_transcripts = false;
        assert!(
            !agent.transcript_capture_enabled(),
            "unset env falls back to the config field (off)"
        );

        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "0");
        }
        agent.capture_transcripts = true;
        assert!(
            !agent.transcript_capture_enabled(),
            "env=0 forces off regardless of the field"
        );

        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "1");
        }
        agent.capture_transcripts = false;
        assert!(
            agent.transcript_capture_enabled(),
            "env=1 forces on regardless of the field"
        );

        // --- 2. `emit_turn_wal` builds the locked payload shape when enabled. ---
        unsafe {
            std::env::remove_var("AKH_TRANSCRIPT_CAPTURE");
        }
        let engine = test_engine();
        let sink = std::sync::Arc::new(RecordingSink {
            records: std::sync::Mutex::new(Vec::new()),
        });
        engine
            .set_mutation_sink(sink.clone())
            .map_err(|_| ())
            .expect("sink not yet installed on a fresh engine");
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent.capture_transcripts = true;
        // Agent::new itself asserts predicate-init triples through the sink
        // (unrelated `assert` mutations); drop those so the count below only
        // reflects `emit_turn_wal`'s own record.
        sink.records.lock().unwrap().clear();

        let tools = vec![json!({"tool": "kg_query", "ok": true})];
        agent.emit_turn_wal(&engine, "tecton", "capital of France?", "Paris.", &tools);

        {
            let records = sink.records.lock().unwrap();
            assert_eq!(records.len(), 1, "exactly one turn mutation recorded");
            let (kind, payload) = &records[0];
            assert_eq!(*kind, crate::mutation::MutationKind::Turn);
            assert_eq!(kind.as_str(), "turn");

            // LOCKED schema (Wave 2's sleep-cycle harvest phase parses this
            // verbatim): {v, ts, workspace, model, user_prompt, agent_answer, tools}.
            assert_eq!(payload["v"], 1);
            assert!(payload["ts"].as_str().is_some_and(|s| s.ends_with('Z')));
            assert_eq!(payload["workspace"], "tecton");
            assert_eq!(payload["model"], Value::Null); // no model manager attached in this test
            assert_eq!(payload["user_prompt"], "capital of France?");
            assert_eq!(payload["agent_answer"], "Paris.");
            assert_eq!(payload["tools"], json!([{"tool": "kg_query", "ok": true}]));
            // No stray keys.
            assert_eq!(
                payload.as_object().unwrap().len(),
                7,
                "payload must carry exactly the locked field set"
            );
        }

        // --- 2b. ADR 116 C2: an autonomous dispatch turn is marked; nothing else changes. ---
        agent.emit_turn_wal_with_model(&engine, "tecton", "auto", "done", &[], None, true);
        {
            let records = sink.records.lock().unwrap();
            assert_eq!(records.len(), 2);
            assert_eq!(records[1].1["dispatch"], true);
            assert!(
                records[0].1.get("dispatch").is_none(),
                "operator turns carry no dispatch key"
            );
        }

        // --- 3. Opt-out (env="0") writes nothing, even with the field on. ---
        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "0");
        }
        agent.capture_transcripts = true; // env override wins anyway
        agent.emit_turn_wal(&engine, "tecton", "hi", "hello", &[]);
        assert_eq!(
            sink.records.lock().unwrap().len(),
            2,
            "opted-out capture writes nothing beyond steps 2 and 2b"
        );

        unsafe {
            std::env::remove_var("AKH_TRANSCRIPT_CAPTURE");
        }
    }

    // --- ADR 089 task A: param-repair-and-retry ---

    /// A tool requiring an integer `party_size` param, which counts its own
    /// invocations (via a shared atomic the test retains a handle to) so
    /// tests can assert *how many times* `execute()` actually ran — the
    /// "retries exactly once, never twice" requirement isn't observable from
    /// the outcome alone. Rejects a `party_size` over 8 with a normal
    /// `Ok(ToolOutput::err(..))` business failure (not a structural error),
    /// so a test can drive the retried call into a *second* failure and
    /// confirm no third attempt follows.
    struct FlakyBookTool {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Tool for FlakyBookTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "flaky_book".into(),
                description: "Book a table for a party".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _e: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let n = input.get_i64("party_size").ok_or_else(|| {
                crate::agent::error::AgentError::ToolExecution {
                    tool_name: "flaky_book".into(),
                    message: "missing required parameter: party_size".into(),
                }
            })?;
            if n > 8 {
                return Ok(ToolOutput::err("too many guests"));
            }
            Ok(ToolOutput::ok(format!("booked for {n}")))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "flaky_book".into(),
                description: "Book a table for a party".into(),
                parameters: vec![ToolParamSchema {
                    name: "party_size".into(),
                    description: "how many guests".into(),
                    required: true,
                    param_type: "integer".into(),
                    schema: None,
                }],
                danger: DangerInfo {
                    effect: ToolEffect::Mutating,
                    level: DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: String::new(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    /// Build an `(Agent, known_tools, full_catalog, calls)` fixture with one
    /// `FlakyBookTool` registered, for the task-A tests below. `calls` is the
    /// same `Arc<AtomicUsize>` the registered tool increments, so the test
    /// can read the invocation count without downcasting through the
    /// registry's `dyn Tool` (the trait carries no `Any` supertrait).
    fn flaky_book_fixture() -> (
        Agent,
        std::collections::HashSet<String>,
        Vec<Value>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let mut agent = Agent::new(test_engine(), AgentConfig::default()).unwrap();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(FlakyBookTool {
                calls: calls.clone(),
            }));
        let known_tools: std::collections::HashSet<String> = agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .list()
            .into_iter()
            .map(|s| s.name)
            .collect();
        let full_catalog = {
            let reg = agent
                .tool_registry
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            ollama_tool_catalog(&*reg)
        };
        (agent, known_tools, full_catalog, calls)
    }

    // One test function (not several): AKH_AGENT_PARAM_REPAIR is
    // process-global and cargo runs tests in parallel threads within one
    // binary, so parallel tests each mutating it would race (same rationale
    // as `transcript_capture_gate_and_emit_turn_wal` above).
    #[test]
    fn param_repair_retry_fires_once_and_respects_opt_out() {
        unsafe {
            std::env::remove_var("AKH_AGENT_PARAM_REPAIR");
        }
        let engine = test_engine();

        // --- 1. Default ON: a near-miss param name + stringified value
        //        repairs and the retry succeeds. Exactly 2 executions. ---
        {
            let (agent, known_tools, full_catalog, calls) = flaky_book_fixture();
            let tc = crate::agent::ollama::ToolCall {
                name: "flaky_book".into(),
                arguments: json!({"party_sze": "4"}),
            };
            let outcome = agent.execute_tool_call(
                &engine,
                &tc,
                &known_tools,
                &full_catalog,
                "book a table",
                &[],
            );
            assert!(
                outcome.success,
                "repaired retry should succeed: {}",
                outcome.result
            );
            assert_eq!(outcome.exec_args["party_size"], json!(4));
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                2,
                "exactly one retry, not zero or several"
            );
            assert_eq!(outcome.trace["param_repaired"], json!(true));
            assert!(outcome.note_prefix.contains("repaired args"));
        }

        // --- 2. Retry itself fails (business rule, not structural) — still
        //        only 2 executions total, never a third attempt. ---
        {
            let (agent, known_tools, full_catalog, calls) = flaky_book_fixture();
            let tc = crate::agent::ollama::ToolCall {
                name: "flaky_book".into(),
                arguments: json!({"party_sze": "9"}),
            };
            let outcome = agent.execute_tool_call(
                &engine,
                &tc,
                &known_tools,
                &full_catalog,
                "book a table",
                &[],
            );
            assert!(!outcome.success);
            assert_eq!(outcome.result, "too many guests");
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                2,
                "must not retry a second time after the retry itself fails"
            );
            assert_eq!(outcome.trace["param_repaired"], json!(true));
        }

        // --- 3. Opt-out: exactly 1 execution, no repair. ---
        unsafe {
            std::env::set_var("AKH_AGENT_PARAM_REPAIR", "0");
        }
        {
            let (agent, known_tools, full_catalog, calls) = flaky_book_fixture();
            let tc = crate::agent::ollama::ToolCall {
                name: "flaky_book".into(),
                arguments: json!({"party_sze": "4"}),
            };
            let outcome = agent.execute_tool_call(
                &engine,
                &tc,
                &known_tools,
                &full_catalog,
                "book a table",
                &[],
            );
            assert!(!outcome.success);
            assert!(outcome.result.starts_with("tool error:"));
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "opted-out: no retry attempt at all"
            );
            assert!(outcome.trace.get("param_repaired").is_none());
        }

        unsafe {
            std::env::remove_var("AKH_AGENT_PARAM_REPAIR");
        }
    }

    // --- ADR 089 task B: value grounding inside the dispatch path ---

    #[test]
    fn value_grounding_changes_a_miscopied_numeric_arg_only_when_enabled() {
        unsafe {
            std::env::remove_var("AKH_AGENT_VALUE_GROUND");
        }
        let engine = test_engine();

        // --- Default OFF: the mis-copied value survives untouched, and the
        //     call executes successfully with the wrong number. ---
        {
            let (agent, known_tools, full_catalog, _calls) = flaky_book_fixture();
            let tc = crate::agent::ollama::ToolCall {
                name: "flaky_book".into(),
                arguments: json!({"party_size": 5}), // request says 4; arg says 5
            };
            let outcome = agent.execute_tool_call(
                &engine,
                &tc,
                &known_tools,
                &full_catalog,
                "book a table for 4",
                &[],
            );
            assert!(outcome.success);
            assert_eq!(
                outcome.exec_args["party_size"],
                json!(5),
                "untouched while grounding is off"
            );
            assert!(!outcome.note_prefix.contains("grounded"));
        }

        // --- Enabled: the mis-copied value is snapped to the request's own
        //     number before execute() ever runs. ---
        unsafe {
            std::env::set_var("AKH_AGENT_VALUE_GROUND", "1");
        }
        {
            let (agent, known_tools, full_catalog, calls) = flaky_book_fixture();
            let tc = crate::agent::ollama::ToolCall {
                name: "flaky_book".into(),
                arguments: json!({"party_size": 5}),
            };
            let outcome = agent.execute_tool_call(
                &engine,
                &tc,
                &known_tools,
                &full_catalog,
                "book a table for 4",
                &[],
            );
            assert!(outcome.success);
            assert_eq!(
                outcome.exec_args["party_size"],
                json!(4),
                "grounded to the request's own value"
            );
            assert!(outcome.note_prefix.contains("grounded"));
            assert_eq!(
                calls.load(std::sync::atomic::Ordering::SeqCst),
                1,
                "grounding fixed it before the first execute — no retry needed"
            );
        }

        unsafe {
            std::env::remove_var("AKH_AGENT_VALUE_GROUND");
        }
    }

    // --- turn-wedge fix: `cancel` is polled once per round ---

    /// A fake [`ChatBackend`](crate::agent::chat_backend::ChatBackend) that
    /// always requests one `echo` tool call (never volunteering a final
    /// answer on its own), so the agentic loop would otherwise keep going
    /// until [`RoundBudget`] is exhausted. Counts its own invocations and,
    /// optionally, flips a shared flag right after its first call — used to
    /// simulate the ACP idle watchdog setting `cancel` while a round is
    /// in flight.
    struct ToolLoopBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        flip_after_first_call: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    }

    impl crate::agent::chat_backend::ChatBackend for ToolLoopBackend {
        fn label(&self) -> &str {
            "fake-tool-loop"
        }
        fn available(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                if let Some(flag) = &self.flip_after_first_call {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            Ok(crate::agent::ollama::AssistantTurn {
                content: String::new(),
                tool_calls: vec![crate::agent::ollama::ToolCall {
                    name: "echo".into(),
                    arguments: json!({"text": "hi"}),
                }],
                ..Default::default()
            })
        }
    }

    /// A turn whose `cancel` flag is already set before the first round must
    /// stop immediately, calling the model zero times — the "watchdog fired
    /// before the worker even started its first round" edge of the turn-wedge
    /// fix (part (a): `AkhAcpAgent::run_turn` → `Agent::run_agentic_turn`).
    #[test]
    fn run_agentic_turn_returns_immediately_when_cancel_already_set() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(ToolLoopBackend {
            calls: calls.clone(),
            flip_after_first_call: None,
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let result = agent.run_agentic_turn(&engine, "tecton", "hello?", &sink, &cancel, false);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "a pre-cancelled turn must never call the model"
        );
    }

    /// A turn cancelled *while a round is in flight* must stop at the next
    /// round boundary — one model call (the round already started), then no
    /// more, and critically no forced-synthesis round either (which would
    /// otherwise show up as a second call with an empty tool catalog). This
    /// is the scenario the ACP idle watchdog drives in production: it sets
    /// `cancel` from another task while the blocking worker is mid-call.
    #[test]
    fn run_agentic_turn_stops_at_next_round_boundary_once_cancelled() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        agent.set_chat_backend(std::sync::Arc::new(ToolLoopBackend {
            calls: calls.clone(),
            flip_after_first_call: Some(cancel.clone()),
        }));
        let sink = crate::message::VecSink::new();
        let result = agent.run_agentic_turn(&engine, "tecton", "hello?", &sink, &cancel, false);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "must stop at the next round boundary once cancelled — no second \
             model round, and no forced-synthesis round either"
        );
    }

    /// Backend that returns prose with no tool calls on the first round, then
    /// tool calls on subsequent rounds — simulating the WI-249 failure case
    /// where the model declares completion without doing any work.
    struct IdleFirstRoundBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::agent::chat_backend::ChatBackend for IdleFirstRoundBackend {
        fn label(&self) -> &str {
            "idle-first-round"
        }
        fn available(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                // First round: return prose with no tool calls (declare completion
                // without doing work)
                Ok(crate::agent::ollama::AssistantTurn {
                    content: "I've completed the task.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            } else {
                // Subsequent rounds: actually do work (call a tool)
                Ok(crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![crate::agent::ollama::ToolCall {
                        name: "echo".into(),
                        arguments: json!({"text": "doing work now"}),
                    }],
                    ..Default::default()
                })
            }
        }
    }

    /// When in autonomous dispatch mode, a turn that declares completion (no
    /// tool calls) without performing any mutating operations must retry with
    /// taxonomy-anchored feedback rather than accepting the declaration.
    /// This is the WI-249 guard: preventing the model from saying "I'm done"
    /// when the worktree has no changes.
    #[test]
    fn dispatch_mode_retries_when_model_declares_completion_without_work() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(IdleFirstRoundBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true: enable the post-condition guard
        let result = agent.run_agentic_turn(
            &engine,
            "tecton",
            "make changes to the repo",
            &sink,
            &cancel,
            true,
        );
        assert_eq!(result, Some(()));
        assert!(
            calls.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "dispatch mode must retry when the model declares completion without \
             performing work (no mutating tool calls + clean worktree); got {} calls",
            calls.load(std::sync::atomic::Ordering::SeqCst)
        );
    }

    /// In interactive (non-dispatch) mode, the guard must not fire: a model
    /// that returns prose with no tool calls is yielding to the user, which
    /// is correct behavior. The turn should end normally after one call.
    #[test]
    fn interactive_mode_accepts_completion_without_work() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(IdleFirstRoundBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = false: guard disabled
        let result = agent.run_agentic_turn(&engine, "tecton", "hello", &sink, &cancel, false);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "interactive mode must accept the model's completion declaration \
             without retrying, even when no work was performed"
        );
    }

    // --- WI-259 wedge fix: empty answer forces final synthesis ---

    /// Backend that returns an empty answer (no tool calls, empty content) on
    /// the first round, simulating the WI-259 wedge scenario where the model
    /// stopped calling tools but produced no prose.
    struct EmptyAnswerBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::agent::chat_backend::ChatBackend for EmptyAnswerBackend {
        fn label(&self) -> &str {
            "empty-answer"
        }
        fn available(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // If tools are offered, return empty with no tool calls (the wedge).
            // If no tools (forced synthesis), return prose.
            if tools.is_empty() {
                Ok(crate::agent::ollama::AssistantTurn {
                    content: "Here is my final answer based on what I gathered.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            } else {
                Ok(crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            }
        }
    }

    /// When the model returns empty content with no tool calls (the WI-259
    /// wedge), the turn must force a final synthesis instead of wedging
    /// indefinitely. Two calls total: the empty round, then forced synthesis.
    #[test]
    fn empty_answer_forces_final_synthesis_instead_of_wedge() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(EmptyAnswerBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // Interactive mode: the guard is disabled but the empty-answer path still applies.
        let result = agent.run_agentic_turn(&engine, "tecton", "hello", &sink, &cancel, false);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "empty answer must force a final synthesis (2 calls: empty round + forced synthesis), not wedge"
        );
    }

    /// In dispatch mode, an empty answer (no tool calls, empty content) also
    /// forces final synthesis — the same fix as interactive mode, since the
    /// post-condition guard only fires when has_mutating_calls is checked,
    /// which doesn't happen when tool_calls is empty in the first place.
    #[test]
    fn dispatch_mode_empty_answer_reports_incomplete_without_synthesis() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(EmptyAnswerBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true
        let result =
            agent.run_agentic_turn(&engine, "tecton", "make changes", &sink, &cancel, true);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "dispatch mode must not synthesize a completion for empty work"
        );
    }

    // --- WI-280 completion check logging and retry budget exhaustion ---

    /// Backend that always returns prose with no tool calls and never performs
    /// mutating work, simulating a model that keeps declaring completion
    /// without actually doing work. Used to test retry budget exhaustion.
    struct AlwaysIdleBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::agent::chat_backend::ChatBackend for AlwaysIdleBackend {
        fn label(&self) -> &str {
            "always-idle"
        }
        fn available(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Always return prose with no tool calls, simulating a model that
            // keeps saying "I'm done" without doing work.
            // If no tools are offered (forced synthesis), return a final answer.
            if tools.is_empty() {
                Ok(crate::agent::ollama::AssistantTurn {
                    content: "Final synthesis after exhaustion.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            } else {
                Ok(crate::agent::ollama::AssistantTurn {
                    content: "I've completed the task.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            }
        }
        fn chat_with_tools_streaming(
            &self,
            agent_id: &str,
            conversation: &[Value],
            tools: &[Value],
            request_seq: usize,
            on_event: &mut dyn FnMut(crate::agent::ollama::StreamEvent),
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let turn = self.chat_with_tools(agent_id, conversation, tools, request_seq)?;
            if !turn.content.is_empty() {
                on_event(crate::agent::ollama::StreamEvent::Content(
                    turn.content.clone(),
                ));
            }
            Ok(turn)
        }
        fn capabilities(&self) -> crate::agent::chat_backend::ModelCapabilities {
            crate::agent::chat_backend::ModelCapabilities {
                context_window: Some(32_000),
                ..Default::default()
            }
        }
    }

    /// WI-280: When the model keeps declaring completion without work, the
    /// retry budget should prevent infinite loops. After 3 retries, the turn
    /// should terminate with DeclaredButIdle and force a final synthesis.
    #[test]
    fn dispatch_mode_exhausts_retry_budget_when_model_keeps_declaring_completion() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(AlwaysIdleBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true: enable the post-condition guard
        let result = agent.run_agentic_turn_with_conversation_state(
            &engine,
            "tecton",
            "make changes",
            &sink,
            &cancel,
            true,
            &std::sync::Mutex::new(ConversationState::default()),
        );
        assert_eq!(
            result,
            Some(AgenticTurnOutcome::Interrupted(TurnStop::DeclaredButIdle))
        );
        let calls_made = calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(calls_made, 4, "four idle attempts, no fabricated synthesis");
    }

    /// Backend that does work (calls a tool) on the first round, then declares
    /// completion on the second round. Used to test that the completion check
    /// accepts completion when work was performed.
    struct WorkThenCompleteBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::agent::chat_backend::ChatBackend for WorkThenCompleteBackend {
        fn label(&self) -> &str {
            "work-then-complete"
        }
        fn available(&self) -> bool {
            true
        }
        fn chat_with_tools(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n == 0 {
                // First round: do work (call a mutating tool). We use "file_io"
                // even though it's not registered, because the completion check
                // only looks at the tool_trace, not the actual execution result.
                Ok(crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![crate::agent::ollama::ToolCall {
                        name: "file_io".into(),
                        arguments: json!({"action": "write", "path": "test.txt", "content": "hello"}),
                    }],
                    ..Default::default()
                })
            } else {
                // Subsequent rounds: declare completion
                Ok(crate::agent::ollama::AssistantTurn {
                    content: "Task completed successfully.".into(),
                    tool_calls: vec![],
                    ..Default::default()
                })
            }
        }
        fn chat_with_tools_streaming(
            &self,
            agent_id: &str,
            conversation: &[Value],
            tools: &[Value],
            request_seq: usize,
            _on_event: &mut dyn FnMut(crate::agent::ollama::StreamEvent),
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.chat_with_tools(agent_id, conversation, tools, request_seq)
        }
        fn capabilities(&self) -> crate::agent::chat_backend::ModelCapabilities {
            crate::agent::chat_backend::ModelCapabilities {
                context_window: Some(32_000),
                ..Default::default()
            }
        }
    }

    /// Fake "file_io" tool that always succeeds, used to simulate mutating work.
    struct FakeFileIoTool;
    impl Tool for FakeFileIoTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "file_io".into(),
                description: "Fake file I/O for testing".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _e: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok("File operation succeeded"))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "file_io".into(),
                description: "Fake file I/O for testing".into(),
                parameters: vec![],
                danger: DangerInfo {
                    effect: ToolEffect::Mutating,
                    level: DangerLevel::Safe,
                    capabilities: std::collections::HashSet::new(),
                    description: "Fake tool for testing".into(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    /// WI-280: In dispatch mode, when the model declares completion with work
    /// performed (mutating tool calls), the turn should accept and complete.
    #[test]
    fn dispatch_mode_accepts_completion_with_work() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(DummyTool));
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .register(Box::new(FakeFileIoTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Backend that does work (calls a tool) and then declares completion
        agent.set_chat_backend(std::sync::Arc::new(WorkThenCompleteBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true
        let result =
            agent.run_agentic_turn(&engine, "tecton", "make changes", &sink, &cancel, true);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "should make exactly 2 calls: one with tool calls, one with completion"
        );
    }

    // --- Evidence-gated completion controller ---

    struct NamedSuccessTool(&'static str);

    impl Tool for NamedSuccessTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: self.0.into(),
                description: format!("test evidence producer {}", self.0),
                parameters: vec![],
            }
        }

        fn execute(&self, _engine: &Engine, _input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok(format!("{} passed", self.0)))
        }

        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: self.0.into(),
                description: format!("test evidence producer {}", self.0),
                parameters: vec![],
                danger: DangerInfo {
                    effect: ToolEffect::Mutating,
                    level: DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: String::new(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    struct OrderedEvidenceBackend {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        saw_contract: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl crate::agent::chat_backend::ChatBackend for OrderedEvidenceBackend {
        fn label(&self) -> &str {
            "ordered-evidence"
        }
        fn available(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            if conversation.iter().any(|message| {
                message["role"] == "system"
                    && message["content"]
                        .as_str()
                        .is_some_and(|content| content.contains("## Completion Contract"))
            }) {
                self.saw_contract
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let tool = match n {
                0 => Some("memory_write"), // too early: must not be banked
                1 => None,                 // premature declaration: must retry
                2 => Some("edit"),
                3 => Some("compile_feedback"),
                4 => Some("memory_write"),
                _ => None,
            };
            Ok(match tool {
                Some(name) => crate::agent::ollama::AssistantTurn {
                    content: String::new(),
                    tool_calls: vec![crate::agent::ollama::ToolCall {
                        name: name.into(),
                        arguments: json!({}),
                    }],
                    ..Default::default()
                },
                None => crate::agent::ollama::AssistantTurn {
                    content: if n == 1 {
                        "I am done too early".into()
                    } else {
                        "All evidence gates passed".into()
                    },
                    tool_calls: vec![],
                    ..Default::default()
                },
            })
        }

        fn chat_with_tools_streaming(
            &self,
            agent_id: &str,
            conversation: &[Value],
            tools: &[Value],
            request_seq: usize,
            on_event: &mut dyn FnMut(crate::agent::ollama::StreamEvent),
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            let turn = self.chat_with_tools(agent_id, conversation, tools, request_seq)?;
            if !turn.content.is_empty() {
                on_event(crate::agent::ollama::StreamEvent::Content(
                    turn.content.clone(),
                ));
            }
            Ok(turn)
        }
    }

    #[test]
    fn explicit_contract_enforces_ordered_evidence_in_interactive_turns() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        for name in ["edit", "compile_feedback", "memory_write"] {
            agent
                .tool_registry
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .register(Box::new(NamedSuccessTool(name)));
        }
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let saw_contract = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        agent.set_chat_backend(std::sync::Arc::new(OrderedEvidenceBackend {
            calls: calls.clone(),
            saw_contract: saw_contract.clone(),
        }));
        let prompt = r#"Train one slice.
<akh:control>{"stages":[{"label":"mutation","any_of":[{"tool":"edit"}]},{"label":"verified","any_of":[{"tool":"compile_feedback"}]},{"label":"learned","any_of":[{"tool":"memory_write"}]}]}</akh:control>"#;
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "ptah",
                prompt,
                &sink,
                &cancel,
                false,
                &agent.conversation_state,
            ),
            Some(AgenticTurnOutcome::Completed)
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            6,
            "premature prose and out-of-order writeback must not end the turn"
        );
        assert!(saw_contract.load(std::sync::atomic::Ordering::SeqCst));
        let narratives = sink
            .messages()
            .into_iter()
            .filter_map(|message| match message {
                AkhMessage::Narrative { text, .. } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            narratives,
            ["All evidence gates passed"],
            "premature streamed prose must remain provisional"
        );
    }

    #[test]
    fn exhausted_contract_fails_without_forced_synthesis() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent
            .tool_registry
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .register(Box::new(NamedSuccessTool("edit")));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(AlwaysIdleBackend {
            calls: calls.clone(),
        }));
        let prompt = r#"Do controlled work.
<akh:control>{"stages":[{"label":"mutation","any_of":[{"tool":"edit"}]}]}</akh:control>"#;
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "ptah",
                prompt,
                &sink,
                &cancel,
                false,
                &agent.conversation_state,
            ),
            Some(AgenticTurnOutcome::Refused)
        );
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "contract exhaustion must not spend a fifth forced-synthesis call"
        );
        let messages = sink.messages();
        assert!(messages.iter().any(|message| matches!(
            message,
            AkhMessage::Error { code, .. } if code == "completion_contract_unsatisfied"
        )));
        assert!(
            !messages
                .iter()
                .any(|message| matches!(message, AkhMessage::Narrative { .. }))
        );
    }

    #[test]
    fn repeat_loop_cannot_bypass_an_open_contract() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        {
            let registry = agent
                .tool_registry
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            registry.register(Box::new(DummyTool));
            registry.register(Box::new(NamedSuccessTool("edit")));
        }
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(ToolLoopBackend {
            calls: calls.clone(),
            flip_after_first_call: None,
        }));
        let prompt = r#"Do controlled work.
<akh:control>{"stages":[{"label":"mutation","any_of":[{"tool":"edit"}]}]}</akh:control>"#;
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "ptah",
                prompt,
                &sink,
                &cancel,
                false,
                &agent.conversation_state,
            ),
            Some(AgenticTurnOutcome::Refused)
        );
        assert!(
            calls.load(std::sync::atomic::Ordering::SeqCst) >= 2,
            "the repeated tool call should trip the loop detector"
        );
        assert!(sink.messages().iter().any(|message| matches!(
            message,
            AkhMessage::Error { code, .. } if code == "completion_contract_unsatisfied"
        )));
    }

    #[test]
    fn required_contract_tools_are_pinned_after_catalog_scoping() {
        let full = vec![
            json!({"function":{"name":"edit"}}),
            json!({"function":{"name":"compile_feedback"}}),
            json!({"function":{"name":"memory_write"}}),
        ];
        let mut selected = vec![full[0].clone()];
        let missing = pin_required_tools(
            &mut selected,
            &full,
            &["edit", "compile_feedback", "memory_write"],
        );
        assert!(missing.is_empty());
        let names = selected
            .iter()
            .filter_map(|tool| tool["function"]["name"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["edit", "compile_feedback", "memory_write"]);
    }

    #[test]
    fn empty_or_blank_reasoning_emits_nothing() {
        let sink = crate::message::VecSink::new();
        let mut turn = crate::agent::ollama::AssistantTurn::default();
        emit_turn_reasoning(&sink, &turn);
        turn.reasoning = "  \n".into();
        emit_turn_reasoning(&sink, &turn);
        assert!(sink.messages().is_empty());
    }

    /// Streaming always fails; the non-streaming retry returns the answer
    /// with separated `reasoning`.
    struct ReasoningFallbackBackend {
        conversations: std::sync::Mutex<Vec<Vec<Value>>>,
    }

    impl crate::agent::chat_backend::ChatBackend for ReasoningFallbackBackend {
        fn label(&self) -> &str {
            "reasoning-fallback"
        }

        fn available(&self) -> bool {
            true
        }

        fn chat_with_tools(
            &self,
            _agent_id: &str,
            conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            self.conversations
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(conversation.to_vec());
            Ok(crate::agent::ollama::AssistantTurn {
                content: "The answer is 4.".into(),
                reasoning: "Two plus two is four.".into(),
                ..Default::default()
            })
        }

        fn chat_with_tools_streaming(
            &self,
            _agent_id: &str,
            _conversation: &[Value],
            _tools: &[Value],
            _request_seq: usize,
            _on_event: &mut dyn FnMut(crate::agent::ollama::StreamEvent),
        ) -> Result<crate::agent::ollama::AssistantTurn, crate::agent::chat_backend::ChatBackendError>
        {
            Err(crate::agent::chat_backend::ChatBackendError::TurnFailed {
                backend: "reasoning-fallback".into(),
                reason: "stream broke".into(),
            })
        }
    }

    #[test]
    fn non_streaming_fallback_surfaces_reasoning_as_a_thinking_step_not_content() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        let backend = std::sync::Arc::new(ReasoningFallbackBackend {
            conversations: std::sync::Mutex::new(Vec::new()),
        });
        agent.set_chat_backend(backend.clone());
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            agent.run_agentic_turn_with_conversation_state(
                &engine,
                "thoth",
                "what is 2+2?",
                &sink,
                &cancel,
                false,
                &agent.conversation_state,
            ),
            Some(AgenticTurnOutcome::Completed)
        );

        let messages = sink.messages();
        let reasoning_at = messages
            .iter()
            .position(|m| matches!(m, AkhMessage::Reasoning { step, .. } if step == "Two plus two is four."))
            .expect("the separated reasoning becomes a thinking step");
        let narrative_at = messages
            .iter()
            .position(
                |m| matches!(m, AkhMessage::Narrative { text, .. } if text == "The answer is 4."),
            )
            .expect("the answer is emitted as narrative");
        assert!(reasoning_at < narrative_at, "thinking precedes the answer");
        // The reasoning is never merged into any visible narrative...
        assert!(!messages.iter().any(|m| matches!(
            m,
            AkhMessage::Narrative { text, .. } if text.contains("Two plus two")
        )));
        // ...and is not replayed to the model as part of the conversation.
        let seen = backend
            .conversations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        assert!(
            !serde_json::to_string(&*seen)
                .unwrap()
                .contains("Two plus two"),
            "reasoning must stay out of the conversation the model sees"
        );
    }
}
