//! Raw-SQL query functions backing the five MCP tools, plus the read-time
//! conversion from a `th_events` row into a redacted, rendered line.
//!
//! `entity::event::Model` predates the `subseq int` / `disposition text`
//! columns this ticket depends on (a concurrent increment adds them to
//! `th_events`; `kind` becomes nullable and every row now carries a
//! `disposition` of `event | noise | undetermined | failed`) — see the
//! increment-6 brief. `entity/` and `migration/` are owned by that
//! increment, so every query in this module is hand-written parameterized
//! SQL via [`sea_orm::Statement`], the same approach
//! `crates/seshat/src/query.rs` uses for the one other place in the
//! workspace that already needs pgvector's `<=>` operator (not expressible
//! through the SeaORM query builder either). Every query filters
//! `disposition = 'event'` explicitly, per the read-mcp-surface contract:
//! that is now the real "is this a real, normalized event" predicate,
//! not `kind IS NOT NULL`.

use chrono::{DateTime, Utc};
use sea_orm::{DatabaseBackend, DatabaseConnection, FromQueryResult, Statement, Value};

use crate::error::TranscriptResult;
use crate::model::{EventKind, RecordKey, Source};
use crate::normalize::{NormalizedEvent, SecretAction};
use crate::{redact, render};

// ── row shapes ───────────────────────────────────────────────────────────

/// One `th_events` row, read-side shape.
#[derive(Debug, Clone, FromQueryResult)]
pub struct EventRow {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub seq: i64,
    pub subseq: Option<i32>,
    pub observed_at: DateTime<Utc>,
    pub kind: Option<String>,
    pub subkind: Option<String>,
    pub role: Option<String>,
    pub text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    pub agent_id: Option<String>,
    pub parent_agent_id: Option<String>,
    pub degraded: bool,
    pub degraded_reason: Option<String>,
    pub repo: Option<String>,
    pub workspace: Option<String>,
    pub branch: Option<String>,
}

const EVENT_COLUMNS: &str = "source, session_id, stream_id, seq, subseq, observed_at, kind, \
     subkind, role, text, tool_name, tool_call_id, agent_id, parent_agent_id, degraded, \
     degraded_reason, repo, workspace, branch";

#[derive(Debug, Clone, FromQueryResult)]
pub struct RawRow {
    pub raw: String,
}

#[derive(Debug, Clone, FromQueryResult)]
pub struct SessionRow {
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

#[derive(Debug, Clone, FromQueryResult)]
pub struct GrepRow {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub seq: i64,
    pub subseq: Option<i32>,
    pub kind: Option<String>,
    pub observed_at: DateTime<Utc>,
    pub text: String,
}

#[derive(Debug, Clone, FromQueryResult)]
pub struct ChunkHit {
    pub source: String,
    pub session_id: String,
    pub stream_id: String,
    pub seq_from: i64,
    pub seq_to: i64,
    pub similarity: f32,
    pub text: String,
}

fn nonempty(v: Option<&str>) -> Option<&str> {
    v.map(str::trim).filter(|s| !s.is_empty())
}

// ── EventRow -> NormalizedEvent, and the redact-then-render path ──────────

/// `Source::as_str`'s inverse. Only ever fed a `source` column value this
/// crate itself wrote, so an unrecognised string is a data-integrity bug,
/// not a user input to reject gracefully — callers treat `None` as "skip
/// this row" rather than surfacing a parse error to an MCP caller.
pub fn parse_source(s: &str) -> Option<Source> {
    match s {
        "claude_code" => Some(Source::ClaudeCode),
        "codex" => Some(Source::Codex),
        "opencode" => Some(Source::Opencode),
        _ => None,
    }
}

/// `render::kind_label`'s inverse (that function is private to `render.rs`,
/// which this ticket must not edit, so the ten labels are duplicated here —
/// they are a closed set per the normal-form ticket, not likely to drift).
pub fn parse_event_kind(s: &str) -> Option<EventKind> {
    match s {
        "prompt" => Some(EventKind::Prompt),
        "reply" => Some(EventKind::Reply),
        "thought" => Some(EventKind::Thought),
        "tool_call" => Some(EventKind::ToolCall),
        "tool_result" => Some(EventKind::ToolResult),
        "context" => Some(EventKind::Context),
        "session_meta" => Some(EventKind::SessionMeta),
        "lifecycle" => Some(EventKind::Lifecycle),
        "error" => Some(EventKind::Error),
        "telemetry" => Some(EventKind::Telemetry),
        _ => None,
    }
}

/// Reconstruct the in-memory [`NormalizedEvent`] a `th_events` row was
/// derived from. Returns `None` when `source`/`kind` do not parse — should
/// not happen for a `disposition = 'event'` row, but every query in this
/// module filters on disposition rather than trusting `kind IS NOT NULL`,
/// so this stays a graceful skip rather than a panic or a hard error that
/// would fail an entire `get_session` over one bad row.
pub fn to_normalized_event(row: &EventRow) -> Option<NormalizedEvent> {
    let source = parse_source(&row.source)?;
    let kind = parse_event_kind(row.kind.as_deref()?)?;
    Some(NormalizedEvent {
        key: RecordKey {
            source,
            session_id: row.session_id.clone(),
            stream_id: row.stream_id.clone(),
            seq: row.seq,
        },
        subseq: row.subseq.map(|s| s as u32),
        observed_at: row.observed_at,
        kind,
        subkind: row.subkind.clone(),
        role: row.role.clone(),
        text: row.text.clone(),
        tool_name: row.tool_name.clone(),
        tool_call_id: row.tool_call_id.clone(),
        agent_id: row.agent_id.clone(),
        parent_agent_id: row.parent_agent_id.clone(),
        degraded: row.degraded,
        degraded_reason: row.degraded_reason.clone(),
        repo: row.repo.clone(),
        workspace: row.workspace.clone(),
        branch: row.branch.clone(),
    })
}

/// Whether an event kind is shown by default (`get_session`'s tiering,
/// read-mcp-surface.md §1): `thought` is excluded unless `include_thoughts`;
/// `telemetry` and `context`'s `compaction` subtype are excluded unless
/// `include_all`. Everything else is always visible (subject to the byte
/// budget). `disposition != 'event'` filtering happens in SQL, before a row
/// ever reaches this function.
pub fn default_visible(
    kind: EventKind,
    subkind: Option<&str>,
    include_thoughts: bool,
    include_all: bool,
) -> bool {
    match kind {
        EventKind::Thought => include_thoughts || include_all,
        EventKind::Telemetry => include_all,
        EventKind::Context if subkind == Some("compaction") => include_all,
        _ => true,
    }
}

/// The redaction-at-read boundary for normalized text: **never** trust a
/// precomputed `render`/`secret_flags` column blindly — rescan `text` with
/// [`redact::scan`] at read time and render from the masked clone. This is
/// simpler than deserializing the stored `secret_flags` JSON (whose exact
/// shape belongs to the concurrent derivation-worker increment, and whose
/// `SecretFlag`/`SecretAction` types in `normalize.rs` are not
/// `Serialize`/`Deserialize` in the first place) and strictly safer: a
/// detector improvement lands here automatically instead of waiting for
/// re-derivation. Used by `get_session` and `get_lines` (non-verbatim path).
pub fn redact_and_render(event: &NormalizedEvent) -> String {
    let mut masked = event.clone();
    if let Some(text) = &event.text {
        let flags = redact::scan(text);
        masked.text = Some(redact::apply(text, &flags));
    }
    render::render(&masked).0
}

/// Outcome of redacting one **raw** (verbatim) record's text at read time —
/// `get_lines(verbatim = true)`'s path. Raw rows carry no stored flags at
/// all, so this always rescans from scratch with [`redact::scan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerbatimLine {
    /// Safe to return, with any matched span masked in place.
    Text(String),
    /// The scan found an unboundable multi-line secret (currently only
    /// `private_key`) — refuse the whole record rather than mask it.
    Withheld { pattern: String },
}

pub fn redact_raw(raw: &str) -> VerbatimLine {
    let flags = redact::scan(raw);
    if let Some(withheld) = flags.iter().find(|f| f.action == SecretAction::Withhold) {
        return VerbatimLine::Withheld {
            pattern: withheld.pattern.clone(),
        };
    }
    VerbatimLine::Text(redact::apply(raw, &flags))
}

// ── get_session / get_lines ────────────────────────────────────────────

/// Every event for one session across every stream, at that session's
/// current (highest) `derivation_version`. Small enough to materialize in
/// full: sessions top out around 12k lines in the measured corpus, and the
/// byte-budget filling this feeds happens afterwards, in memory, over the
/// already-spliced/stream-ordered sequence — computing that order requires
/// the whole session's events up front regardless.
pub async fn session_events(
    db: &DatabaseConnection,
    source: &str,
    session_id: &str,
) -> TranscriptResult<Vec<EventRow>> {
    let sql = format!(
        "SELECT {EVENT_COLUMNS} FROM th_events \
         WHERE source = $1 AND session_id = $2 AND disposition = 'event' \
           AND derivation_version = ( \
               SELECT MAX(derivation_version) FROM th_events \
               WHERE source = $1 AND session_id = $2 \
           ) \
         ORDER BY stream_id, seq, subseq NULLS FIRST"
    );
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        sql,
        [source.into(), session_id.into()],
    );
    Ok(EventRow::find_by_statement(stmt).all(db).await?)
}

/// One stream's events in `[from_seq, to_seq]`, at the session's current
/// derivation version.
pub async fn stream_lines(
    db: &DatabaseConnection,
    source: &str,
    session_id: &str,
    stream_id: &str,
    from_seq: i64,
    to_seq: i64,
) -> TranscriptResult<Vec<EventRow>> {
    let sql = format!(
        "SELECT {EVENT_COLUMNS} FROM th_events \
         WHERE source = $1 AND session_id = $2 AND stream_id = $3 \
           AND seq >= $4 AND seq <= $5 AND disposition = 'event' \
           AND derivation_version = ( \
               SELECT MAX(derivation_version) FROM th_events \
               WHERE source = $1 AND session_id = $2 \
           ) \
         ORDER BY seq, subseq NULLS FIRST"
    );
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        sql,
        [
            source.into(),
            session_id.into(),
            stream_id.into(),
            from_seq.into(),
            to_seq.into(),
        ],
    );
    Ok(EventRow::find_by_statement(stmt).all(db).await?)
}

/// The verbatim source line for one `(source, session_id, stream_id, seq)`.
/// `th_raw_records`'s physical primary key also carries `observed_at`
/// (partition column), so this orders by it and takes one row rather than
/// assuming the natural key alone is unique.
pub async fn raw_record_text(
    db: &DatabaseConnection,
    source: &str,
    session_id: &str,
    stream_id: &str,
    seq: i64,
) -> TranscriptResult<Option<String>> {
    let sql = "SELECT raw FROM th_raw_records \
         WHERE source = $1 AND session_id = $2 AND stream_id = $3 AND seq = $4 \
         ORDER BY observed_at DESC LIMIT 1";
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        sql,
        [
            source.into(),
            session_id.into(),
            stream_id.into(),
            seq.into(),
        ],
    );
    let row = RawRow::find_by_statement(stmt).one(db).await?;
    Ok(row.map(|r| r.raw))
}

// ── list_sessions ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct SessionFilters {
    pub source: Option<String>,
    pub repo: Option<String>,
    pub branch: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

/// `list_sessions`. Offset-based continuation rather than a keyset cursor:
/// simpler to implement correctly, and acceptable here because
/// `th_sessions` is derived, read-mostly, and small (session count, not
/// event count) — a concurrent insert shifting the window by a few rows
/// mid-walk is a cosmetic skip/repeat, not a correctness hazard for this
/// surface. Returns `(rows, more_available)`.
pub async fn list_sessions(
    db: &DatabaseConnection,
    filters: &SessionFilters,
    limit: u64,
    offset: u64,
) -> TranscriptResult<(Vec<SessionRow>, bool)> {
    let mut sql = String::from(
        "SELECT source, session_id, title, repo, workspace, branch, first_seen, last_seen, \
         stream_count, record_count FROM th_sessions WHERE 1=1",
    );
    let mut values: Vec<Value> = Vec::new();

    if let Some(v) = nonempty(filters.source.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND source = ${}", values.len()));
    }
    if let Some(v) = nonempty(filters.repo.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND repo = ${}", values.len()));
    }
    if let Some(v) = nonempty(filters.branch.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND branch = ${}", values.len()));
    }
    if let Some(v) = filters.since {
        values.push(v.into());
        sql.push_str(&format!(" AND last_seen >= ${}", values.len()));
    }
    if let Some(v) = filters.until {
        values.push(v.into());
        sql.push_str(&format!(" AND last_seen <= ${}", values.len()));
    }

    values.push((limit as i64 + 1).into());
    sql.push_str(&format!(
        " ORDER BY last_seen DESC, session_id DESC LIMIT ${}",
        values.len()
    ));
    values.push((offset as i64).into());
    sql.push_str(&format!(" OFFSET ${}", values.len()));

    let stmt = Statement::from_sql_and_values(DatabaseBackend::Postgres, sql, values);
    let mut rows = SessionRow::find_by_statement(stmt).all(db).await?;
    let more = rows.len() as u64 > limit;
    rows.truncate(limit as usize);
    Ok((rows, more))
}

// ── grep_transcripts ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct GrepFilters {
    pub source: Option<String>,
    pub session_id: Option<String>,
    pub kind: Option<String>,
    pub since: Option<DateTime<Utc>>,
}

/// `grep_transcripts`: `~*` (case-insensitive POSIX regex) over
/// `th_events.text`, `disposition = 'event'` only, restricted to the
/// current `derivation_version` per session via a correlated subquery (old,
/// unreferenced rows from a prior derivation version must never surface).
/// `pattern` is always bound as a parameter, never interpolated, but the
/// caller must still bound its *length* before calling this — an
/// attacker-controlled unbounded regex is a ReDoS surface regardless of
/// parameterization.
pub async fn grep_events(
    db: &DatabaseConnection,
    pattern: &str,
    filters: &GrepFilters,
    limit: u64,
    offset: u64,
) -> TranscriptResult<(Vec<GrepRow>, bool)> {
    let mut sql = String::from(
        "SELECT source, session_id, stream_id, seq, subseq, kind, observed_at, text \
         FROM th_events e \
         WHERE disposition = 'event' AND text IS NOT NULL AND text ~* $1 \
           AND derivation_version = ( \
               SELECT MAX(derivation_version) FROM th_events e2 \
               WHERE e2.source = e.source AND e2.session_id = e.session_id \
           )",
    );
    let mut values: Vec<Value> = vec![pattern.into()];

    if let Some(v) = nonempty(filters.source.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND source = ${}", values.len()));
    }
    if let Some(v) = nonempty(filters.session_id.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND session_id = ${}", values.len()));
    }
    if let Some(v) = nonempty(filters.kind.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND kind = ${}", values.len()));
    }
    if let Some(v) = filters.since {
        values.push(v.into());
        sql.push_str(&format!(" AND observed_at >= ${}", values.len()));
    }

    values.push((limit as i64 + 1).into());
    sql.push_str(&format!(
        " ORDER BY observed_at ASC LIMIT ${}",
        values.len()
    ));
    values.push((offset as i64).into());
    sql.push_str(&format!(" OFFSET ${}", values.len()));

    let stmt = Statement::from_sql_and_values(DatabaseBackend::Postgres, sql, values);
    let mut rows = GrepRow::find_by_statement(stmt).all(db).await?;
    let more = rows.len() as u64 > limit;
    rows.truncate(limit as usize);
    Ok((rows, more))
}

// ── search_transcripts ──────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ChunkFilters {
    pub source: Option<String>,
    pub session_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
}

/// `search_transcripts`: cosine similarity over `th_chunks.embedding`
/// (`vector(384)`, HNSW-indexed), restricted to the current
/// `derivation_version` per session for the same reason `grep_events` is.
pub async fn search_chunks(
    db: &DatabaseConnection,
    embedding: &[f32],
    top_k: u64,
    min_similarity: f32,
    filters: &ChunkFilters,
) -> TranscriptResult<Vec<ChunkHit>> {
    let embedding_str = crate::entity::chunk::vec_to_pgvector(embedding);
    let mut sql = String::from(
        "SELECT source, session_id, stream_id, seq_from, seq_to, \
         (1.0 - (embedding <=> $1::vector))::real AS similarity, text \
         FROM th_chunks c \
         WHERE embedding IS NOT NULL \
           AND (1.0 - (embedding <=> $1::vector)) > $2 \
           AND derivation_version = ( \
               SELECT MAX(derivation_version) FROM th_chunks c2 \
               WHERE c2.source = c.source AND c2.session_id = c.session_id \
           )",
    );
    let mut values: Vec<Value> = vec![embedding_str.into(), min_similarity.into()];

    if let Some(v) = nonempty(filters.source.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND source = ${}", values.len()));
    }
    if let Some(v) = nonempty(filters.session_id.as_deref()) {
        values.push(v.into());
        sql.push_str(&format!(" AND session_id = ${}", values.len()));
    }
    if let Some(v) = filters.since {
        values.push(v.into());
        sql.push_str(&format!(" AND created_at >= ${}", values.len()));
    }

    values.push((top_k as i64).into());
    sql.push_str(&format!(
        " ORDER BY embedding <=> $1::vector LIMIT ${}",
        values.len()
    ));

    let stmt = Statement::from_sql_and_values(DatabaseBackend::Postgres, sql, values);
    Ok(ChunkHit::find_by_statement(stmt).all(db).await?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(text: Option<&str>) -> EventRow {
        EventRow {
            source: "claude_code".to_string(),
            session_id: "s1".to_string(),
            stream_id: "main".to_string(),
            seq: 1,
            subseq: None,
            observed_at: Utc::now(),
            kind: Some("reply".to_string()),
            subkind: None,
            role: None,
            text: text.map(str::to_string),
            tool_name: None,
            tool_call_id: None,
            agent_id: None,
            parent_agent_id: None,
            degraded: false,
            degraded_reason: None,
            repo: None,
            workspace: None,
            branch: None,
        }
    }

    #[test]
    fn parse_source_roundtrips_as_str() {
        for s in [Source::ClaudeCode, Source::Codex, Source::Opencode] {
            assert_eq!(parse_source(s.as_str()), Some(s));
        }
        assert_eq!(parse_source("not-a-source"), None);
    }

    #[test]
    fn parse_event_kind_covers_the_closed_set() {
        let labels = [
            "prompt",
            "reply",
            "thought",
            "tool_call",
            "tool_result",
            "context",
            "session_meta",
            "lifecycle",
            "error",
            "telemetry",
        ];
        for label in labels {
            assert!(parse_event_kind(label).is_some(), "{label} should parse");
        }
        assert_eq!(parse_event_kind("unknown_kind"), None);
    }

    #[test]
    fn to_normalized_event_skips_rows_with_unparseable_kind() {
        let mut r = row(Some("hi"));
        r.kind = Some("not_a_real_kind".to_string());
        assert!(to_normalized_event(&r).is_none());
    }

    #[test]
    fn to_normalized_event_skips_rows_with_no_kind() {
        let mut r = row(Some("hi"));
        r.kind = None;
        assert!(to_normalized_event(&r).is_none());
    }

    #[test]
    fn default_visible_hides_thought_and_telemetry_by_default() {
        assert!(!default_visible(EventKind::Thought, None, false, false));
        assert!(default_visible(EventKind::Thought, None, true, false));
        assert!(!default_visible(EventKind::Telemetry, None, false, false));
        assert!(default_visible(EventKind::Telemetry, None, false, true));
        assert!(!default_visible(
            EventKind::Context,
            Some("compaction"),
            false,
            false
        ));
        assert!(default_visible(
            EventKind::Context,
            Some("compaction"),
            false,
            true
        ));
        assert!(default_visible(
            EventKind::Context,
            Some("attachment"),
            false,
            false
        ));
        assert!(default_visible(EventKind::Prompt, None, false, false));
        assert!(default_visible(EventKind::Reply, None, false, false));
    }

    #[test]
    fn redact_and_render_never_emits_a_github_token() {
        let r = row(Some(&format!("here is a token gho_{}", "a".repeat(36))));
        let ev = to_normalized_event(&r).expect("valid row");
        let rendered = redact_and_render(&ev);
        assert!(!rendered.contains("gho_"), "rendered: {rendered}");
        assert!(rendered.contains("[REDACTED:github_pat]"));
    }

    #[test]
    fn redact_and_render_leaves_clean_text_alone() {
        let r = row(Some("nothing secret in here"));
        let ev = to_normalized_event(&r).expect("valid row");
        let rendered = redact_and_render(&ev);
        assert!(rendered.contains("nothing secret in here"));
    }

    #[test]
    fn redact_raw_masks_a_github_token() {
        let raw = format!("{{\"token\":\"gho_{}\"}}", "a".repeat(36));
        match redact_raw(&raw) {
            VerbatimLine::Text(masked) => {
                assert!(!masked.contains("gho_"), "masked: {masked}");
                assert!(masked.contains("[REDACTED:github_pat]"));
            }
            VerbatimLine::Withheld { .. } => panic!("github token should mask, not withhold"),
        }
    }

    #[test]
    fn redact_raw_withholds_a_private_key() {
        let raw = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END-----";
        match redact_raw(raw) {
            VerbatimLine::Withheld { pattern } => assert_eq!(pattern, "private_key"),
            VerbatimLine::Text(_) => panic!("private key must be withheld, not masked"),
        }
    }
}
