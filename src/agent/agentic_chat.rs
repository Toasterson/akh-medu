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
}

impl TurnStop {
    fn clause(self) -> &'static str {
        match self {
            TurnStop::IterationCap => "You have reached your tool-call limit",
            TurnStop::Timeout => "You have reached your time budget",
            TurnStop::RepeatLoop => "You are repeating the same tool call without progress",
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
    started: Instant,
    round: usize,
    /// Count of each `(tool, args)` signature issued this turn, for spin detection.
    counts: HashMap<u64, usize>,
    looped: bool,
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
        Self {
            hard_max,
            timeout,
            repeat_limit,
            started: Instant::now(),
            round: 0,
            counts: HashMap::new(),
            looped: false,
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
    pub fn run_agentic_turn(
        &mut self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        sink: &dyn MessageSink,
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
        backend.prepare(agent_id, self.psyche.as_ref());

        let mut tools = ollama_tool_catalog(&self.tool_registry);
        // Optional cap on the tool catalog (AKH_AGENT_MAX_TOOLS). The full catalog
        // is several thousand tokens — too large for a small-context in-process
        // model (e.g. Candle Gemma on a constrained host), where it dominates
        // prefill cost. `0` runs the akh tool-free (pure chat). A precursor to the
        // Phase 44 budget-driven tool selection.
        if let Some(cap) = std::env::var("AKH_AGENT_MAX_TOOLS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && cap < tools.len()
        {
            tracing::debug!(cap, from = tools.len(), "capping tool catalog (AKH_AGENT_MAX_TOOLS)");
            tools.truncate(cap);
        }

        // Record the operator turn before assembling, so the conversation's last
        // turn IS the current prompt — a history-aware ContextManager renders the
        // turns directly (the agent turn is recorded at the end). The symbolic
        // path already records both speakers; this closes the gap for the agentic
        // path so multi-turn context works (measure-first Q1).
        self.conversation_state_mut().record_operator_turn(user_prompt);

        // Phase 44d/44g: retrieve the input's KG context + relevant past episodes
        // for the working-context manager. The agent owns the engine + predicates;
        // the manager only budgets + renders these sections. (44e session seeding
        // rides `with_context`'s seed slot — populated when a session summary
        // exists; the windows are large so all sections fit in practice.)
        let kg_context = if std::env::var("AKH_KG_DELTA").map(|v| v == "1").unwrap_or(false) {
            // 44d delta refresh: inject only facts not already shown this session
            // (the activation frontier that moved), trusting retained history to
            // hold the rest. Opt-in; the default re-sends the full relevant block.
            let seen = self.conversation_state_mut().kg_seen_mut();
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
            let episode = EpisodeState::new(agent_id, user_prompt, engine)
                .with_conversation(Some(self.conversation_state()))
                .with_context(kg_context.as_deref(), &recalled, session_seed.as_deref())
                .with_tools(&tools);
            let manager: &dyn ContextManager =
                self.context_manager.as_deref().unwrap_or(&PASS_THROUGH);
            match manager.assemble(&episode, &budget, &counter) {
                Ok(assembled) => {
                    super::context::budget::record_assembly_telemetry(
                        &manager.id(),
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

        // Govern the tool loop with a frontier-style budget — a model-scaled
        // iteration backstop + wall-clock cap + repeat-loop detection — instead of
        // a flat round count. The loop yields *why* it stopped so the turn can end
        // with a forced synthesis rather than a bare cutoff.
        let mut budget = RoundBudget::new(backend.capabilities().context_window);
        let stop: TurnStop = loop {
            if let Some(stop) = budget.exhausted() {
                break stop;
            }
            // Stream this round. Content deltas become live answer text, thinking
            // deltas become live reasoning. `used_fallback` records whether the
            // streaming path failed and we silently retried non-streaming — in
            // which case nothing was streamed and we emit the result ourselves.
            let mut used_fallback = false;
            let turn = {
                let mut on_event = |ev: StreamEvent| match ev {
                    StreamEvent::Content(delta) => sink.emit(&AkhMessage::narrative(delta, "acp")),
                    StreamEvent::Thinking(delta) => sink.emit(&AkhMessage::Reasoning {
                        step: delta,
                        expression: None,
                    }),
                };
                match backend.chat_with_tools_streaming(agent_id, &conversation, &tools, &mut on_event) {
                    Ok(t) => t,
                    Err(stream_err) => {
                        tracing::warn!(
                            agent_id,
                            "streaming chat failed ({stream_err}); falling back to non-streaming"
                        );
                        used_fallback = true;
                        match backend.chat_with_tools(agent_id, &conversation, &tools) {
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
                if !answer.is_empty() {
                    self.conversation_state_mut().record_agent_turn(answer);
                    // ADR 091: capture the completed turn to the WAL (raw
                    // staging; a no-op if no sink is installed or capture is
                    // opted out).
                    self.emit_turn_wal(engine, agent_id, user_prompt, answer, &tool_trace);
                    // Streaming already emitted the answer as deltas; only the
                    // non-streaming fallback still needs to emit it.
                    if used_fallback {
                        sink.emit(&AkhMessage::narrative(answer, "acp"));
                    }
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
                let input = tool_input_from_args(&tc.arguments);
                let (success, result) = match self.tool_registry.execute(&tc.name, input, engine) {
                    Ok(o) => (o.success, o.result),
                    Err(e) => (false, format!("tool error: {e}")),
                };

                sink.emit(&AkhMessage::tool_result(
                    tc.name.clone(),
                    success,
                    format!("({}) → {}", preview_args(&tc.arguments), result),
                ));
                tool_trace.push(json!({"tool": tc.name.clone(), "ok": success}));
                conversation.push(json!({"role": "tool", "content": result}));
            }
        };

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
        );
        Some(())
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
        &mut self,
        backend: &dyn super::chat_backend::ChatBackend,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        conversation: &mut Vec<Value>,
        sink: &dyn MessageSink,
        stop: TurnStop,
        tool_trace: &[Value],
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
        let turn = {
            let mut on_event = |ev: StreamEvent| match ev {
                StreamEvent::Content(delta) => sink.emit(&AkhMessage::narrative(delta, "acp")),
                StreamEvent::Thinking(delta) => sink.emit(&AkhMessage::Reasoning {
                    step: delta,
                    expression: None,
                }),
            };
            match backend.chat_with_tools_streaming(agent_id, conversation, &no_tools, &mut on_event)
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(agent_id, "forced-answer streaming failed ({e}); non-streaming");
                    used_fallback = true;
                    match backend.chat_with_tools(agent_id, conversation, &no_tools) {
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
        self.conversation_state_mut().record_agent_turn(answer);
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
}
