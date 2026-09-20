//! MCP server for the transcript-harvest read surface (increment 6).
//!
//! Five tools — `list_sessions`, `get_session`, `get_lines`,
//! `grep_transcripts`, `search_transcripts` — over the `harvest-pg` store,
//! read-only (`harvest_read` role: `SELECT` only, never write). Pattern
//! copied from `crates/seshat/src/mcp/mod.rs` (rmcp `#[tool_router]`/
//! `#[tool]`, `Parameters<T>` + `schemars::JsonSchema` for input schemas,
//! plain `CallToolResult::success(vec![Content::text(json)])` for output).
//!
//! Every tool takes a `budget_bytes` param (default 32 000, max 256 000,
//! [`budget::clamp_budget`]) and returns `truncated` / `omitted_bytes` /
//! `continue_from` alongside its payload — read-mcp-surface.md §1's
//! "always-visible truncation". Redaction happens at this boundary, on every
//! path, per `docs/wayfinder/transcript-harvest/tickets/secrets-posture.md`:
//! see [`queries::redact_and_render`] and [`queries::redact_raw`].

mod budget;
mod queries;
mod readiness;
mod splice;
pub mod startup;

pub use readiness::Readiness;
pub use startup::StartupError;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::{ErrorData as McpError, tool, tool_router};
use schemars::JsonSchema;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};

use crate::model::EventKind;
use crate::normalize::NormalizedEvent;
use crate::redact;

/// Shared state for MCP tool handlers.
pub struct TranscriptsMcpState {
    pub db: Arc<DatabaseConnection>,
    /// `None` when `AKH_TRANSCRIPTS_MODEL_DIR` was absent or failed to load
    /// at startup — `search_transcripts` then returns a clear error while
    /// the other four tools keep working, per the contract.
    pub embedder: Option<akh_embed::SharedEmbedder>,
    pub readiness: Arc<Readiness>,
}

/// MCP server exposing the transcript-harvest read surface.
#[derive(Clone)]
pub struct TranscriptsMcpServer {
    state: Arc<TranscriptsMcpState>,
    tool_router: ToolRouter<Self>,
}

// ── field-name reuse: kind label <-> EventKind, duplicated intentionally ──
//
// `render::kind_label`/`queries::parse_event_kind` already own the
// canonical string form; `kind_label` here is the same closed 10-label set
// used only to fill the `kind` field of `get_session`'s output rows.

fn kind_label(kind: EventKind) -> &'static str {
    match kind {
        EventKind::Prompt => "prompt",
        EventKind::Reply => "reply",
        EventKind::Thought => "thought",
        EventKind::ToolCall => "tool_call",
        EventKind::ToolResult => "tool_result",
        EventKind::Context => "context",
        EventKind::SessionMeta => "session_meta",
        EventKind::Lifecycle => "lifecycle",
        EventKind::Error => "error",
        EventKind::Telemetry => "telemetry",
    }
}

fn surface_err(e: crate::error::TranscriptError) -> McpError {
    McpError::internal_error(e.to_string(), None)
}

fn json_result<T: Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

fn nonempty(v: Option<&str>) -> Option<&str> {
    v.map(str::trim).filter(|s| !s.is_empty())
}

// ── get_session cursor: opaque "<stream_id>\x1f<seq>\x1f<subseq?>" ────────

const CURSOR_SEP: char = '\u{1f}';

fn encode_session_cursor(stream_id: &str, seq: i64, subseq: Option<u32>) -> String {
    format!(
        "{stream_id}{CURSOR_SEP}{seq}{CURSOR_SEP}{}",
        subseq.map(|s| s.to_string()).unwrap_or_default()
    )
}

fn decode_session_cursor(cursor: &str) -> Option<(String, i64, Option<u32>)> {
    let mut parts = cursor.splitn(3, CURSOR_SEP);
    let stream_id = parts.next()?.to_string();
    let seq: i64 = parts.next()?.parse().ok()?;
    let subseq = match parts.next()? {
        "" => None,
        s => s.parse().ok(),
    };
    Some((stream_id, seq, subseq))
}

// ── grep snippet windowing (±200 chars around the first match) ───────────

fn clamp_floor(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn clamp_ceil(s: &str, mut idx: usize) -> usize {
    idx = idx.min(s.len());
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

const GREP_WINDOW: usize = 200;

/// Window `text` to ±200 chars around `re`'s first match (or the first
/// ~400 chars if it somehow doesn't match — defensive; the row only reached
/// this function because Postgres's own `~*` already matched the same
/// pattern) and redact the window before it ever leaves the process.
fn windowed_snippet(text: &str, re: &regex::Regex) -> String {
    let (start, end) = match re.find(text) {
        Some(m) => (
            clamp_floor(text, m.start().saturating_sub(GREP_WINDOW)),
            clamp_ceil(text, m.end() + GREP_WINDOW),
        ),
        None => (0, clamp_ceil(text, GREP_WINDOW * 2)),
    };
    let window = &text[start..end];
    let flags = redact::scan(window);
    redact::apply(window, &flags)
}

async fn embed_query(
    embedder: &akh_embed::SharedEmbedder,
    query: &str,
) -> Result<Vec<f32>, McpError> {
    let embedder = Arc::clone(embedder);
    let query = query.to_string();
    tokio::task::spawn_blocking(move || {
        let mut guard = embedder
            .lock()
            .map_err(|e| format!("embedder lock poisoned: {e}"))?;
        guard
            .embed_text(&query)
            .map_err(|e| format!("embedding query: {e}"))
    })
    .await
    .map_err(|e| McpError::internal_error(format!("embedding task join: {e}"), None))?
    .map_err(|e| McpError::internal_error(e, None))
}

// ── list_sessions ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListSessionsParams {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub since: Option<DateTime<Utc>>,
    #[serde(default)]
    pub until: Option<DateTime<Utc>>,
    #[serde(default = "default_list_limit")]
    pub limit: u64,
    /// Opaque continuation token from a previous call's `continue_from`.
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

fn default_list_limit() -> u64 {
    50
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SessionSummary {
    pub source: String,
    pub session_id: String,
    pub title: Option<String>,
    pub repo: Option<String>,
    pub workspace: Option<String>,
    pub branch: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub stream_count: i32,
    pub record_count: i64,
}

impl From<&queries::SessionRow> for SessionSummary {
    fn from(row: &queries::SessionRow) -> Self {
        SessionSummary {
            source: row.source.clone(),
            session_id: row.session_id.clone(),
            title: row.title.clone(),
            repo: row.repo.clone(),
            workspace: row.workspace.clone(),
            branch: row.branch.clone(),
            first_seen: row.first_seen,
            last_seen: row.last_seen,
            stream_count: row.stream_count,
            record_count: row.record_count,
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct ListSessionsResult {
    pub sessions: Vec<SessionSummary>,
    pub truncated: bool,
    pub omitted_bytes: u64,
    pub continue_from: Option<String>,
}

// ── get_session ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSessionParams {
    pub source: String,
    pub session_id: String,
    /// `"spliced"` (default; subagent streams inserted at their spawn
    /// point) or `"stream"` (plain `(stream_id, seq, subseq)` order).
    #[serde(default = "default_order")]
    pub order: String,
    /// Include `thought` events (always a bare marker — see
    /// `render::render` — never the rare real text). Off by default: near
    /// all reasoning content in the corpus is either encrypted or empty.
    #[serde(default)]
    pub include_thoughts: bool,
    /// Include `telemetry` and `context`'s `compaction` subtype, both
    /// excluded by default as noise/restated content.
    #[serde(default)]
    pub include_all: bool,
    /// Resume from the first event whose own `seq` is `>= from_seq`
    /// (checked against each event's own stream position, not a
    /// cross-stream global counter). Ignored when `cursor` is set.
    #[serde(default)]
    pub from_seq: Option<i64>,
    /// Opaque continuation token from a previous call's `continue_from`.
    /// Takes precedence over `from_seq`.
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

fn default_order() -> String {
    "spliced".to_string()
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SessionEventOut {
    pub stream_id: String,
    pub seq: i64,
    pub subseq: Option<u32>,
    pub kind: String,
    pub render: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct GetSessionResult {
    pub source: String,
    pub session_id: String,
    pub order: String,
    pub events: Vec<SessionEventOut>,
    pub truncated: bool,
    pub omitted_bytes: u64,
    pub continue_from: Option<String>,
}

// ── get_lines ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetLinesParams {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub from_seq: i64,
    pub to_seq: i64,
    /// Return the raw source record text instead of the rendered event.
    /// Still redacted — `redact::scan` + `redact::apply` run on the raw
    /// text at read time, and a record whose scan finds an unboundable
    /// secret (currently only a PEM private-key header) is withheld
    /// entirely rather than partially masked.
    #[serde(default)]
    pub verbatim: bool,
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct LineOut {
    pub stream_id: String,
    pub seq: i64,
    pub subseq: Option<i32>,
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// Set instead of `raw`/`render` when the record was withheld outright
    /// (verbatim mode only) — names the pattern that triggered it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub withheld: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct GetLinesResult {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub verbatim: bool,
    pub lines: Vec<LineOut>,
    pub truncated: bool,
    pub omitted_bytes: u64,
    pub continue_from: Option<String>,
}

// ── grep_transcripts ───────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GrepParams {
    /// Case-insensitive POSIX regex (`~*`), bounded to 200 bytes.
    pub pattern: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub since: Option<DateTime<Utc>>,
    #[serde(default = "default_grep_limit")]
    pub limit: u64,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

fn default_grep_limit() -> u64 {
    100
}

const MAX_GREP_PATTERN_BYTES: usize = 200;

#[derive(Debug, Serialize, JsonSchema)]
pub struct GrepHitOut {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub seq: i64,
    pub subseq: Option<i32>,
    pub kind: Option<String>,
    pub observed_at: DateTime<Utc>,
    pub snippet: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct GrepResult {
    pub hits: Vec<GrepHitOut>,
    pub truncated: bool,
    pub omitted_bytes: u64,
    pub continue_from: Option<String>,
}

// ── search_transcripts ─────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: u64,
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub since: Option<DateTime<Utc>>,
    #[serde(default)]
    pub budget_bytes: Option<u64>,
}

fn default_top_k() -> u64 {
    10
}

fn default_min_similarity() -> f32 {
    0.3
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchHitOut {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub seq_from: i64,
    pub seq_to: i64,
    pub similarity: f32,
    pub text: String,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchResult {
    pub hits: Vec<SearchHitOut>,
    pub truncated: bool,
    pub omitted_bytes: u64,
}

// ── MCP server implementation ──────────────────────────────────────────

#[tool_router]
impl TranscriptsMcpServer {
    pub fn new(state: Arc<TranscriptsMcpState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    /// The exact MCP tool descriptors this server advertises — used by the
    /// integration test and any export tooling instead of hand-copying
    /// schemas.
    pub fn tool_definitions() -> Vec<rmcp::model::Tool> {
        Self::tool_router().list_all()
    }

    #[tool(
        name = "list_sessions",
        description = "List harvested agent-coding-session summaries (filterable by source, repo, branch, and time range), most recently active first. Use this to find a session_id before calling get_session or get_lines."
    )]
    async fn list_sessions(
        &self,
        Parameters(params): Parameters<ListSessionsParams>,
    ) -> Result<CallToolResult, McpError> {
        let budget = budget::clamp_budget(params.budget_bytes);
        let limit = params.limit.clamp(1, 500);
        let offset: u64 = params
            .cursor
            .as_deref()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);

        let filters = queries::SessionFilters {
            source: params.source,
            repo: params.repo,
            branch: params.branch,
            since: params.since,
            until: params.until,
        };
        let (rows, more_after_limit) =
            queries::list_sessions(&self.state.db, &filters, limit, offset)
                .await
                .map_err(surface_err)?;

        let mut filler = budget::BudgetFiller::new(budget);
        let mut sessions = Vec::new();
        let mut budget_stop_offset: Option<u64> = None;
        for (i, row) in rows.iter().enumerate() {
            let summary = SessionSummary::from(row);
            let bytes = serde_json::to_string(&summary)
                .map(|s| s.len() as u64)
                .unwrap_or(0);
            if filler.try_take(bytes) {
                sessions.push(summary);
            } else if budget_stop_offset.is_none() {
                budget_stop_offset = Some(offset + i as u64);
            }
        }

        let continue_from = if filler.truncated() {
            budget_stop_offset.map(|o| o.to_string())
        } else if more_after_limit {
            Some((offset + limit).to_string())
        } else {
            None
        };

        json_result(&ListSessionsResult {
            sessions,
            truncated: filler.truncated() || more_after_limit,
            omitted_bytes: filler.omitted_bytes(),
            continue_from,
        })
    }

    #[tool(
        name = "get_session",
        description = "Fetch one session's events in reading order (subagent streams spliced in at their spawn point by default), already rendered and redacted, byte-budgeted and pageable via `cursor`. Use after list_sessions, grep_transcripts, or search_transcripts once you have a session_id."
    )]
    async fn get_session(
        &self,
        Parameters(params): Parameters<GetSessionParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.order != "spliced" && params.order != "stream" {
            return Err(McpError::invalid_params(
                "order must be \"spliced\" or \"stream\"",
                None,
            ));
        }
        let budget = budget::clamp_budget(params.budget_bytes);

        let rows = queries::session_events(&self.state.db, &params.source, &params.session_id)
            .await
            .map_err(surface_err)?;

        let events: Vec<NormalizedEvent> = rows
            .iter()
            .filter_map(queries::to_normalized_event)
            .collect();
        let by_stream = splice::group_by_stream(events);
        let ordered = if params.order == "stream" {
            splice::stream_order(&by_stream)
        } else {
            splice::spliced_order(&by_stream)
        };

        let visible: Vec<&NormalizedEvent> = ordered
            .iter()
            .filter(|e| {
                queries::default_visible(
                    e.kind,
                    e.subkind.as_deref(),
                    params.include_thoughts,
                    params.include_all,
                )
            })
            .collect();

        let start_idx = if let Some(cursor) = params.cursor.as_deref() {
            decode_session_cursor(cursor)
                .and_then(|(stream_id, seq, subseq)| {
                    visible
                        .iter()
                        .position(|e| {
                            e.key.stream_id == stream_id && e.key.seq == seq && e.subseq == subseq
                        })
                        .map(|i| i + 1)
                })
                .unwrap_or(0)
        } else if let Some(from_seq) = params.from_seq {
            visible
                .iter()
                .position(|e| e.key.seq >= from_seq)
                .unwrap_or(visible.len())
        } else {
            0
        };

        let mut filler = budget::BudgetFiller::new(budget);
        let mut out_events = Vec::new();
        let mut continue_from: Option<String> = None;
        for event in visible.iter().skip(start_idx) {
            let rendered = queries::redact_and_render(event);
            let bytes = rendered.len() as u64;
            if filler.try_take(bytes) {
                out_events.push(SessionEventOut {
                    stream_id: event.key.stream_id.clone(),
                    seq: event.key.seq,
                    subseq: event.subseq,
                    kind: kind_label(event.kind).to_string(),
                    render: rendered,
                });
            } else if continue_from.is_none() {
                continue_from = Some(encode_session_cursor(
                    &event.key.stream_id,
                    event.key.seq,
                    event.subseq,
                ));
            }
        }

        json_result(&GetSessionResult {
            source: params.source,
            session_id: params.session_id,
            order: params.order,
            events: out_events,
            truncated: filler.truncated(),
            omitted_bytes: filler.omitted_bytes(),
            continue_from,
        })
    }

    #[tool(
        name = "get_lines",
        description = "Fetch an exact seq range from one stream of one session, rendered by default or verbatim on request (still redacted). Use this to widen a grep_transcripts or search_transcripts hit into its surrounding context in a single call."
    )]
    async fn get_lines(
        &self,
        Parameters(params): Parameters<GetLinesParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.to_seq < params.from_seq {
            return Err(McpError::invalid_params("to_seq must be >= from_seq", None));
        }
        let budget = budget::clamp_budget(params.budget_bytes);

        let rows = queries::stream_lines(
            &self.state.db,
            &params.source,
            &params.session_id,
            &params.stream_id,
            params.from_seq,
            params.to_seq,
        )
        .await
        .map_err(surface_err)?;

        let mut filler = budget::BudgetFiller::new(budget);
        let mut lines = Vec::new();
        let mut continue_from: Option<String> = None;

        for row in &rows {
            if continue_from.is_some() {
                // Budget already exhausted: still worth an honest
                // `omitted_bytes` tally, but no reason to pay for a second
                // raw-record round trip per remaining row.
                filler.try_take(row.text.as_deref().map(|t| t.len() as u64).unwrap_or(0));
                continue;
            }

            if params.verbatim {
                let raw = queries::raw_record_text(
                    &self.state.db,
                    &params.source,
                    &params.session_id,
                    &params.stream_id,
                    row.seq,
                )
                .await
                .map_err(surface_err)?;
                let Some(raw) = raw else { continue };
                match queries::redact_raw(&raw) {
                    queries::VerbatimLine::Text(text) => {
                        let bytes = text.len() as u64;
                        if filler.try_take(bytes) {
                            lines.push(LineOut {
                                stream_id: row.stream_id.clone(),
                                seq: row.seq,
                                subseq: row.subseq,
                                kind: row.kind.clone(),
                                render: None,
                                raw: Some(text),
                                withheld: None,
                            });
                        } else {
                            continue_from = Some(row.seq.to_string());
                        }
                    }
                    queries::VerbatimLine::Withheld { pattern } => {
                        // Always surfaced, regardless of budget — a withheld
                        // marker is a handful of bytes and dropping it would
                        // make a redacted record indistinguishable from one
                        // that simply did not exist.
                        lines.push(LineOut {
                            stream_id: row.stream_id.clone(),
                            seq: row.seq,
                            subseq: row.subseq,
                            kind: row.kind.clone(),
                            render: None,
                            raw: None,
                            withheld: Some(pattern),
                        });
                    }
                }
            } else {
                let Some(event) = queries::to_normalized_event(row) else {
                    continue;
                };
                let rendered = queries::redact_and_render(&event);
                let bytes = rendered.len() as u64;
                if filler.try_take(bytes) {
                    lines.push(LineOut {
                        stream_id: row.stream_id.clone(),
                        seq: row.seq,
                        subseq: row.subseq,
                        kind: row.kind.clone(),
                        render: Some(rendered),
                        raw: None,
                        withheld: None,
                    });
                } else {
                    continue_from = Some(row.seq.to_string());
                }
            }
        }

        json_result(&GetLinesResult {
            source: params.source,
            session_id: params.session_id,
            stream_id: params.stream_id,
            verbatim: params.verbatim,
            lines,
            truncated: filler.truncated(),
            omitted_bytes: filler.omitted_bytes(),
            continue_from,
        })
    }

    #[tool(
        name = "grep_transcripts",
        description = "Case-insensitive regex search over transcript text when you already know the literal string or pattern to match (error messages, identifiers, command names, exact wording) — faster and more precise than search_transcripts for that case; use search_transcripts instead when you only know what you mean, not the exact words."
    )]
    async fn grep_transcripts(
        &self,
        Parameters(params): Parameters<GrepParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.pattern.is_empty() {
            return Err(McpError::invalid_params("pattern must not be empty", None));
        }
        if params.pattern.len() > MAX_GREP_PATTERN_BYTES {
            return Err(McpError::invalid_params(
                format!(
                    "pattern too long ({} bytes, max {MAX_GREP_PATTERN_BYTES})",
                    params.pattern.len()
                ),
                None,
            ));
        }
        let re = regex::RegexBuilder::new(&params.pattern)
            .case_insensitive(true)
            .size_limit(1 << 20)
            .build()
            .map_err(|e| McpError::invalid_params(format!("invalid regex: {e}"), None))?;

        let budget = budget::clamp_budget(params.budget_bytes);
        let limit = params.limit.clamp(1, 1000);
        let offset: u64 = params
            .cursor
            .as_deref()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);

        let filters = queries::GrepFilters {
            source: params.source,
            session_id: params.session_id,
            kind: nonempty(params.kind.as_deref()).map(str::to_string),
            since: params.since,
        };
        let (rows, more_after_limit) =
            queries::grep_events(&self.state.db, &params.pattern, &filters, limit, offset)
                .await
                .map_err(surface_err)?;

        let mut filler = budget::BudgetFiller::new(budget);
        let mut hits = Vec::new();
        let mut budget_stop_offset: Option<u64> = None;
        for (i, row) in rows.iter().enumerate() {
            let snippet = windowed_snippet(&row.text, &re);
            let bytes = snippet.len() as u64;
            if filler.try_take(bytes) {
                hits.push(GrepHitOut {
                    source: row.source.clone(),
                    session_id: row.session_id.clone(),
                    stream_id: row.stream_id.clone(),
                    seq: row.seq,
                    subseq: row.subseq,
                    kind: row.kind.clone(),
                    observed_at: row.observed_at,
                    snippet,
                });
            } else if budget_stop_offset.is_none() {
                budget_stop_offset = Some(offset + i as u64);
            }
        }

        let continue_from = if filler.truncated() {
            budget_stop_offset.map(|o| o.to_string())
        } else if more_after_limit {
            Some((offset + limit).to_string())
        } else {
            None
        };

        json_result(&GrepResult {
            hits,
            truncated: filler.truncated() || more_after_limit,
            omitted_bytes: filler.omitted_bytes(),
            continue_from,
        })
    }

    #[tool(
        name = "search_transcripts",
        description = "Semantic (embedding) search over prompt/reply text for 'what did I do about X' questions when you don't know the exact wording. Each hit carries (source, session_id, stream_id, seq_from, seq_to) so you can widen it with get_lines in one further call; use grep_transcripts instead when you already know the literal text to match."
    )]
    async fn search_transcripts(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some(embedder) = self.state.embedder.as_ref() else {
            return Err(McpError::internal_error(
                "search_transcripts is unavailable on this server: AKH_TRANSCRIPTS_MODEL_DIR \
                 was not configured or the embedding model failed to load at startup. \
                 list_sessions, get_session, get_lines, and grep_transcripts are unaffected.",
                None,
            ));
        };
        let embedding = embed_query(embedder, &params.query).await?;

        let budget = budget::clamp_budget(params.budget_bytes);
        let top_k = params.top_k.clamp(1, 100);

        let filters = queries::ChunkFilters {
            source: params.source,
            session_id: params.session_id,
            since: params.since,
        };
        let rows = queries::search_chunks(
            &self.state.db,
            &embedding,
            top_k,
            params.min_similarity,
            &filters,
        )
        .await
        .map_err(surface_err)?;

        let mut filler = budget::BudgetFiller::new(budget);
        let mut hits = Vec::new();
        for row in &rows {
            // Chunk text is already masked at derivation time (§ embedding
            // pipeline), but this is the outward boundary — rescanning here
            // costs nothing on already-clean text and means the "never
            // emits unmasked text" property holds structurally for this
            // tool too, rather than resting on an upstream invariant this
            // module cannot verify.
            let flags = redact::scan(&row.text);
            let text = redact::apply(&row.text, &flags);
            let bytes = text.len() as u64;
            if filler.try_take(bytes) {
                hits.push(SearchHitOut {
                    source: row.source.clone(),
                    session_id: row.session_id.clone(),
                    stream_id: row.stream_id.clone(),
                    seq_from: row.seq_from,
                    seq_to: row.seq_to,
                    similarity: row.similarity,
                    text,
                });
            }
        }

        json_result(&SearchResult {
            hits,
            truncated: filler.truncated(),
            omitted_bytes: filler.omitted_bytes(),
        })
    }
}

#[rmcp::tool_handler]
impl rmcp::handler::server::ServerHandler for TranscriptsMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "akh-transcripts".to_string(),
                title: Some("Agent Session Transcript Harvest".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some(
                    "Read-only surface over harvested Claude Code / Codex / OpenCode agent \
                     session transcripts (ADR 108)."
                        .to_string(),
                ),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Use `list_sessions` to find a session_id by source/repo/branch/time. Use \
                 `get_session` to read a whole session in reading order (subagent streams \
                 spliced in at their spawn point). Use `get_lines` to fetch an exact seq range, \
                 rendered or verbatim. Use `grep_transcripts` when you know the exact text to \
                 match; use `search_transcripts` (semantic) when you only know what you mean. \
                 Every tool takes `budget_bytes` and reports `truncated`/`omitted_bytes`/\
                 `continue_from` — never assume a response is complete without checking \
                 `truncated`."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_cursor_roundtrips_with_subseq() {
        let cursor = encode_session_cursor("agent-a", 42, Some(3));
        assert_eq!(
            decode_session_cursor(&cursor),
            Some(("agent-a".to_string(), 42, Some(3)))
        );
    }

    #[test]
    fn session_cursor_roundtrips_without_subseq() {
        let cursor = encode_session_cursor("main", 7, None);
        assert_eq!(
            decode_session_cursor(&cursor),
            Some(("main".to_string(), 7, None))
        );
    }

    #[test]
    fn session_cursor_rejects_garbage() {
        assert_eq!(decode_session_cursor("not-a-cursor"), None);
        assert_eq!(decode_session_cursor(""), None);
    }

    #[test]
    fn windowed_snippet_masks_a_github_token_inside_the_window() {
        let re = regex::RegexBuilder::new("hello")
            .case_insensitive(true)
            .build()
            .unwrap();
        let text = format!("padding padding hello world gho_{} more", "a".repeat(36));
        let snippet = windowed_snippet(&text, &re);
        assert!(!snippet.contains("gho_"), "snippet: {snippet}");
        assert!(snippet.contains("[REDACTED:github_pat]"));
    }

    #[test]
    fn windowed_snippet_stays_within_400_chars_plus_redaction_markers() {
        let re = regex::RegexBuilder::new("needle").build().unwrap();
        let text = format!("{}needle{}", "x".repeat(5000), "y".repeat(5000));
        let snippet = windowed_snippet(&text, &re);
        // ±200 chars around the match, generously bounded even after any
        // redaction-marker expansion.
        assert!(snippet.len() < 1000, "snippet len: {}", snippet.len());
        assert!(snippet.contains("needle"));
    }

    #[test]
    fn tool_definitions_list_exactly_the_five_tools() {
        let tools = TranscriptsMcpServer::tool_definitions();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "get_lines",
                "get_session",
                "grep_transcripts",
                "list_sessions",
                "search_transcripts",
            ]
        );
    }
}
