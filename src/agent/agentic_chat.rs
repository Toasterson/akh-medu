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
//! with synchronous tool execution. Callers run it inside
//! [`tokio::task::spawn_blocking`] holding the agent `Mutex` (ADR-027); no guard
//! crosses an `.await`. The per-turn permission asker, if any, must already be
//! installed on the tool registry (`Agent::set_permission_asker`) so `Ask` tools
//! can bridge to the ACP client.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::Agent;
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
enum TurnStop {
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
    /// Wedge detected: the turn has been active but no LLM request has been
    /// issued for longer than the wedge timeout. This catches the WI-259
    /// failure mode where the model stopped requesting tools but the loop
    /// neither completed nor continued.
    WedgeTimeout,
}

impl TurnStop {
    fn clause(self) -> &'static str {
        match self {
            TurnStop::IterationCap => "You have reached your tool-call limit",
            TurnStop::Timeout => "You have reached your time budget",
            TurnStop::RepeatLoop => "You are repeating the same tool call without progress",
            TurnStop::Cancelled => "This turn was cancelled",
            TurnStop::DeclaredButIdle => {
                "You declared completion but verification shows no work was performed. \
                 The worktree has no uncommitted changes and no mutating tool calls were made. \
                 Review the task requirements and use the appropriate tools to make the necessary changes"
            }
            TurnStop::WedgeTimeout => {
                "The turn has been inactive for too long without making progress"
            }
        }
    }
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
    /// Returns `Some(())` when the turn was handled (answer emitted, or an error
    /// surfaced through `sink`), or `None` when there is **no usable model
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
    /// (which sets it once it gives up waiting on this turn) can reclaim the
    /// agent lock as soon as the in-flight round returns instead of the
    /// worker running unattended for the rest of the turn's budget. A caller
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

        // Refresh per-akh state (e.g. the Ollama system-prompt cache) from psyche.
        let manifests = self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).list_advertised_manifests();
        let manifest_refs: Vec<&super::tool_manifest::ToolManifest> = manifests.iter().collect();
        backend.prepare(agent_id, self.psyche.as_ref(), &manifest_refs);

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
        if let Some(k) = std::env::var("AKH_AGENT_TOOL_RETRIEVAL")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && k < tools.len()
        {
            tracing::debug!(k, from = tools.len(), "retrieval-scoping tool catalog (AKH_AGENT_TOOL_RETRIEVAL)");
            tools = super::organ::retrieve_scoped(&tools, user_prompt, k, Some(engine));
        } else if let Some(cap) = std::env::var("AKH_AGENT_MAX_TOOLS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && cap < tools.len()
        {
            // Legacy blind cap (AKH_AGENT_MAX_TOOLS): kept for A/B comparison
            // against retrieval scoping. `0` runs the akh tool-free (pure chat).
            tracing::debug!(cap, from = tools.len(), "capping tool catalog (AKH_AGENT_MAX_TOOLS)");
            tools.truncate(cap);
        }

        // Phase 44m (ADR 065 D12.1): capture the deferred-tool set — what was
        // in `full_catalog` but NOT offered this round — so the ACP `_akh/usage`
        // panel can show the "loaded-but-deferred" tier. Group by tool origin
        // (`System tools` / `MCP tools`) and reason. Cached outside the agent
        // lock the same way `last_context_stats` is, so a busy turn does not
        // block a sidebar poll.
        {
            let deferred_groups =
                compute_deferred_groups(&full_catalog, &tools, &*self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()));
            if let Ok(mut slot) = self.last_deferred_tools.lock() {
                *slot = deferred_groups;
            }
        }

        // Record the operator turn before assembling, so the conversation's last
        // turn IS the current prompt — a history-aware ContextManager renders the
        // turns directly (the agent turn is recorded at the end). The symbolic
        // path already records both speakers; this closes the gap for the agentic
        // path so multi-turn context works (measure-first Q1).
        self.conversation_state().record_operator_turn(user_prompt);

        // Phase 44d/44g: retrieve the input's KG context + relevant past episodes
        // for the working-context manager. The agent owns the engine + predicates;
        // the manager only budgets + renders these sections. (44e session seeding
        // rides `with_context`'s seed slot — populated when a session summary
        // exists; the windows are large so all sections fit in practice.)
        let kg_context = if std::env::var("AKH_KG_DELTA").map(|v| v == "1").unwrap_or(false) {
            // 44d delta refresh: inject only facts not already shown this session
            // (the activation frontier that moved), trusting retained history to
            // hold the rest. Opt-in; the default re-sends the full relevant block.
            let mut conv_guard = self.conversation_state();
            let seen = conv_guard.kg_seen_mut();
            crate::nlu::kg_context::build_kg_context_delta(engine, user_prompt, seen)
        } else {
            let mut cache = crate::nlu::kg_context::KgContextCache::default();
            crate::nlu::kg_context::build_kg_context(engine, user_prompt, &mut cache)
        };
        let recalled = self.recall_for_prompt(engine, user_prompt, 3);
        // 44e: a one-line continuity seed (the conversation's opening ask), so the
        // thread survives history trimming. None on the first turn.
        let session_seed = self.session_seed();

        // Phase 44a (ADR 065): assemble the working context through the akh's
        // `ContextManager` instead of building the prompt ad hoc. The default
        // pass-through manager reproduces the single-user-turn behaviour
        // byte-for-byte; richer managers (history, KG, recall) slot in here
        // without the loop changing. The loop then appends assistant/tool turns
        // to this base across rounds, exactly as before.
        let (mut conversation, tools): (Vec<Value>, Vec<Value>) = {
            use super::context::{
                ContextManager, EpisodeState, HeuristicTokenCounter, PassThroughManager,
                TokenBudget, TokenCounter, DEFAULT_RESERVED_OUTPUT,
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
            let conv_guard = self.conversation_state();
            let episode = EpisodeState::new(agent_id, user_prompt, engine)
                .with_conversation(Some(&*conv_guard))
                .with_context(kg_context.as_deref(), &recalled, session_seed.as_deref())
                .with_tools(&tools);
            let cm_guard = self.context_manager.lock().unwrap_or_else(|e| e.into_inner());
            let manager: &dyn ContextManager =
                cm_guard.as_deref().unwrap_or(&PASS_THROUGH);
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
                    (vec![json!({"role": "user", "content": user_prompt})], tools.clone())
                }
            }
        };

        // Compact tool trace for the ADR 091 turn-capture hook: {tool, ok} per
        // call, accumulated across every round of this turn (not just the
        // final one) — never raw arguments/results, so a chatty tool loop
        // doesn't bloat the WAL payload.
        let mut tool_trace: Vec<Value> = Vec::new();

        // Registry membership snapshot for the ADR 089 M2 repair gate below: an
        // emitted name outside this set is a guaranteed `unknown tool` failure,
        // which is exactly (and only) when the organ's repair pass may fire.
        let known_tools: std::collections::HashSet<String> =
            self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).list().into_iter().map(|s| s.name).collect();

        // Govern the tool loop with a frontier-style budget — a model-scaled
        // iteration backstop + wall-clock cap + repeat-loop detection — instead of
        // a flat round count. The loop yields *why* it stopped so the turn can end
        // with a forced synthesis rather than a bare cutoff.
        let mut budget = RoundBudget::new(backend.capabilities().context_window);
        // Turn-level metrics: track total prompt size and request count for diagnosing
        // pathological latency (work item "turns die quietly", ask 2).
        let mut turn_metrics = super::llm_metrics::TurnMetrics::new();
        let stop: TurnStop = loop {
            if let Some(stop) = budget.exhausted() {
                break stop;
            }
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break TurnStop::Cancelled;
            }
            // Stream this round. Content deltas become live answer text, thinking
            // deltas become live reasoning. `used_fallback` records whether the
            // streaming path failed and we silently retried non-streaming — in
            // which case nothing was streamed and we emit the result ourselves.
            let mut used_fallback = false;
            // Throttle Keepalive emissions: tool-call chunks can arrive at token
            // rate (~100/sec for a 1k-char JSON payload at 100 tok/s), so we emit
            // at most one heartbeat per 30s to reset the watchdog without spam.
            let mut last_keepalive = Instant::now();
            const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
            // Track prompt size and get request sequence number before the backend call.
            let prompt_chars: usize = conversation.iter()
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
                        sink.emit(&AkhMessage::narrative(delta, "acp"));
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
                match backend.chat_with_tools_streaming(agent_id, &conversation, &tools, request_seq, &mut on_event) {
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
                        match backend.chat_with_tools(agent_id, &conversation, &tools, request_seq) {
                            Ok(t) => t,
                            Err(e) => {
                                sink.emit(&AkhMessage::error("llm", e.to_string()));
                                return Some(());
                            }
                        }
                    }
                }
            };

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
                    mode = if is_autonomous_dispatch { "dispatch" } else { "interactive" },
                    "no tool calls: evaluating completion"
                );

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
                    break TurnStop::IterationCap;
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
                    let has_mutating_calls = tool_trace.iter().any(|entry| {
                        if let (Some(tool), Some(ok)) = (entry.get("tool").and_then(|v| v.as_str()),
                                                          entry.get("ok").and_then(|v| v.as_bool())) {
                            ok && matches!(tool, "file_io" | "edit" | "shell_exec")
                        } else {
                            false
                        }
                    });

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
                    mode = if is_autonomous_dispatch { "dispatch" } else { "interactive" },
                    "completion check → returning: turn complete with answer"
                );
                self.conversation_state().record_agent_turn(answer);
                // ADR 091: capture the completed turn to the WAL (raw
                // staging; a no-op if no sink is installed or capture is
                // opted out).
                self.emit_turn_wal(engine, agent_id, user_prompt, answer, &tool_trace);
                // Streaming already emitted the answer as deltas; only the
                // non-streaming fallback still needs to emit it.
                if used_fallback {
                    sink.emit(&AkhMessage::narrative(answer, "acp"));
                }
                return Some(());
            }

            // Tool round. Any prose the model emitted alongside its tool calls is
            // reasoning; on the fallback path it was not streamed, so surface it
            // as a thinking step here (the streaming path already did).
            if used_fallback {
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
            for tc in &turn.tool_calls {
                let outcome =
                    self.execute_tool_call(engine, tc, &known_tools, &full_catalog, user_prompt);
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
                conversation.push(json!({"role": "tool", "content": outcome.result}));
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
            return Some(());
        }

        // Budget exhausted — force one tool-free synthesis ("reflect and answer")
        // so the turn returns the best answer it can from what it gathered, rather
        // than the old bare "(stopped after N rounds)" cutoff.
        self.force_final_answer(
            backend.as_ref(),
            engine,
            agent_id,
            user_prompt,
            &mut conversation,
            sink,
            stop,
            &tool_trace,
            &mut turn_metrics,
        );
        // Log turn-level metrics summary (total prompt size, request count).
        turn_metrics.log();
        Some(())
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
        let b_pre = super::organ::ground_call_values(&exec_name, &exec_args, user_prompt, full_catalog);
        if let Some(g) = &b_pre {
            exec_args = g.arguments.clone();
        }

        let input = tool_input_from_args(&exec_args);
        let mut param_repaired = false;
        let mut a_note: Option<String> = None;
        // Bind the lock result to an owned `AgentResult` *before* matching on
        // it: a `match self.tool_registry.lock()....execute(..) { .. }` keeps
        // the `MutexGuard` temporary alive for the scrutinee's entire match
        // expression (a well-known Rust footgun), and the `Err` arm below
        // re-locks the same `std::sync::Mutex` on this thread — a guaranteed
        // self-deadlock on every failed tool call otherwise.
        let exec_result = self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).execute(&exec_name, input, engine);
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
                let retry = if opted_out || !repairable || self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).would_ask(&exec_name) {
                    None
                } else {
                    super::organ::repair_known_call_args(&exec_name, &exec_args, full_catalog)
                };
                match retry {
                    Some(repaired) => {
                        let mut retry_args = repaired.arguments;
                        let mut retry_note = repaired.note;
                        if let Some(g) =
                            super::organ::ground_call_values(&exec_name, &retry_args, user_prompt, full_catalog)
                        {
                            retry_args = g.arguments;
                            retry_note = format!("{retry_note}; {}", g.note);
                        }
                        let retry_input = tool_input_from_args(&retry_args);
                        match self.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).execute(&exec_name, retry_input, engine) {
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

        let mut trace = json!({"tool": exec_name.clone(), "ok": success});
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
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        conversation: &mut Vec<Value>,
        sink: &dyn MessageSink,
        stop: TurnStop,
        tool_trace: &[Value],
        turn_metrics: &mut super::llm_metrics::TurnMetrics,
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
        let prompt_chars: usize = conversation.iter()
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
            match backend.chat_with_tools_streaming(agent_id, conversation, &no_tools, request_seq, &mut on_event)
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(agent_id, "forced-answer streaming failed ({e}); non-streaming");
                    // Make the fallback observable (same fix as the normal round).
                    sink.emit(&AkhMessage::system(format!(
                        "streaming failed ({backend_label}), continuing non-streaming",
                        backend_label = backend.label()
                    )));
                    used_fallback = true;
                    match backend.chat_with_tools(agent_id, conversation, &no_tools, request_seq) {
                        Ok(t) => t,
                        Err(e) => {
                            sink.emit(&AkhMessage::error("llm", e.to_string()));
                            return;
                        }
                    }
                }
            }
        };

        let answer = turn.content.trim();
        if answer.is_empty() {
            sink.emit(&AkhMessage::system(format!(
                "(stopped: {})",
                stop.clause().to_lowercase()
            )));
            return;
        }
        self.conversation_state().record_agent_turn(answer);
        // ADR 091: same capture hook as the normal-completion turn-end site.
        self.emit_turn_wal(engine, agent_id, user_prompt, answer, tool_trace);
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
    fn emit_turn_wal(
        &self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        answer: &str,
        tools: &[Value],
    ) {
        if !self.transcript_capture_enabled() {
            return;
        }
        // Cheap, in-scope model name: the Ollama-backed path's model manager
        // names the pinned model directly. Other backings (e.g. Candle, set
        // via `chat_backend`) don't expose a model name through the
        // `ChatBackend` trait today, so this is `None` for them rather than
        // plumbing new state through the trait for this slice.
        let model = self.model_manager.as_deref().map(|m| m.model());
        let payload = json!({
            "v": 1,
            "ts": rfc3339_now(),
            "workspace": agent_id,
            "model": model,
            "user_prompt": user_prompt,
            "agent_answer": answer,
            "tools": tools,
        });
        engine.record_turn(payload);
    }
}

/// Build the Ollama/OpenAI `tools` catalog from every registered tool manifest.
///
/// Each tool becomes `{"type":"function","function":{name,description,parameters}}`
/// with a JSON-Schema `parameters` object derived from the manifest's parameter
/// list. Execution is still gated by the permission policy, so exposing every
/// tool here is safe — a dangerous one the model picks is denied at dispatch.
fn ollama_tool_catalog(registry: &ToolRegistry) -> Vec<Value> {
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
    let reason = if std::env::var("AKH_AGENT_TOOL_RETRIEVAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map(|k| k < full_catalog.len())
        .unwrap_or(false)
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentConfig;
    use crate::agent::error::AgentResult;
    use crate::agent::tool::{Tool, ToolOutput, ToolSignature};
    use crate::agent::tool_manifest::{
        DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
    };
    use std::collections::HashSet;

    fn call(name: &str, args: &str) -> (String, Value) {
        (name.to_string(), json!({ "q": args }))
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
            self.records.lock().unwrap().push((mutation.kind, mutation.payload));
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
        assert!(agent.transcript_capture_enabled(), "unset env falls back to the config field (on)");
        agent.capture_transcripts = false;
        assert!(!agent.transcript_capture_enabled(), "unset env falls back to the config field (off)");

        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "0");
        }
        agent.capture_transcripts = true;
        assert!(!agent.transcript_capture_enabled(), "env=0 forces off regardless of the field");

        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "1");
        }
        agent.capture_transcripts = false;
        assert!(agent.transcript_capture_enabled(), "env=1 forces on regardless of the field");

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

        // --- 3. Opt-out (env="0") writes nothing, even with the field on. ---
        unsafe {
            std::env::set_var("AKH_TRANSCRIPT_CAPTURE", "0");
        }
        agent.capture_transcripts = true; // env override wins anyway
        agent.emit_turn_wal(&engine, "tecton", "hi", "hello", &[]);
        assert_eq!(
            sink.records.lock().unwrap().len(),
            1,
            "opted-out capture writes nothing beyond step 2's single record"
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
            .lock().unwrap_or_else(|e| e.into_inner())
            .register(Box::new(FlakyBookTool { calls: calls.clone() }));
        let known_tools: std::collections::HashSet<String> =
            agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).list().into_iter().map(|s| s.name).collect();
        let full_catalog = {
            let reg = agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner());
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
            let outcome = agent.execute_tool_call(&engine, &tc, &known_tools, &full_catalog, "book a table");
            assert!(outcome.success, "repaired retry should succeed: {}", outcome.result);
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
            let outcome = agent.execute_tool_call(&engine, &tc, &known_tools, &full_catalog, "book a table");
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
            let outcome = agent.execute_tool_call(&engine, &tc, &known_tools, &full_catalog, "book a table");
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
            let outcome =
                agent.execute_tool_call(&engine, &tc, &known_tools, &full_catalog, "book a table for 4");
            assert!(outcome.success);
            assert_eq!(outcome.exec_args["party_size"], json!(5), "untouched while grounding is off");
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
            let outcome =
                agent.execute_tool_call(&engine, &tc, &known_tools, &full_catalog, "book a table for 4");
            assert!(outcome.success);
            assert_eq!(outcome.exec_args["party_size"], json!(4), "grounded to the request's own value");
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(IdleFirstRoundBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true: enable the post-condition guard
        let result = agent.run_agentic_turn(&engine, "tecton", "make changes to the repo", &sink, &cancel, true);
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
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
    fn dispatch_mode_empty_answer_also_forces_synthesis() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(EmptyAnswerBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true
        let result = agent.run_agentic_turn(&engine, "tecton", "make changes", &sink, &cancel, true);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "dispatch mode with empty answer must also force final synthesis"
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

    /// WI-280: When the model keeps declaring completion without work, the
    /// retry budget should prevent infinite loops. After 3 retries, the turn
    /// should terminate with DeclaredButIdle and force a final synthesis.
    #[test]
    fn dispatch_mode_exhausts_retry_budget_when_model_keeps_declaring_completion() {
        let engine = test_engine();
        let mut agent = Agent::new(std::sync::Arc::clone(&engine), AgentConfig::default()).unwrap();
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        agent.set_chat_backend(std::sync::Arc::new(AlwaysIdleBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true: enable the post-condition guard
        let result = agent.run_agentic_turn(&engine, "tecton", "make changes", &sink, &cancel, true);
        assert_eq!(result, Some(()));
        let calls_made = calls.load(std::sync::atomic::Ordering::SeqCst);
        // Should make: 1 initial call + 3 retries + 1 forced synthesis = 5 calls total.
        // The no_work_retry counter starts at 0. After the first idle answer:
        // - note_no_work_retry() → counter = 1, returns false → retry
        // After the second idle answer:
        // - note_no_work_retry() → counter = 2, returns false → retry
        // After the third idle answer:
        // - note_no_work_retry() → counter = 3, returns false → retry
        // After the fourth idle answer:
        // - note_no_work_retry() → counter = 4, returns true → break with DeclaredButIdle
        // Then forced synthesis makes one more call.
        // So: 4 idle calls + 1 synthesis call = 5 total.
        assert_eq!(
            calls_made, 5,
            "should make exactly 5 calls: 4 idle rounds (1 initial + 3 retries) + 1 forced synthesis"
        );
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
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(DummyTool));
        agent.tool_registry.lock().unwrap_or_else(|e| e.into_inner()).register(Box::new(FakeFileIoTool));
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Backend that does work (calls a tool) and then declares completion
        agent.set_chat_backend(std::sync::Arc::new(WorkThenCompleteBackend {
            calls: calls.clone(),
        }));
        let sink = crate::message::VecSink::new();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        // is_autonomous_dispatch = true
        let result = agent.run_agentic_turn(&engine, "tecton", "make changes", &sink, &cancel, true);
        assert_eq!(result, Some(()));
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "should make exactly 2 calls: one with tool calls, one with completion"
        );
    }
}
