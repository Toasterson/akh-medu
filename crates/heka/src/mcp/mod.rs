//! MCP server for Heka's per-akh long-term memory store.
//!
//! Exposes exactly three bounded tools over the Model Context Protocol's
//! streamable-HTTP transport (the pattern mirrors `seshat::mcp`, but every
//! request is tenant-scoped): [`heka_remember`](HekaMcpServer::heka_remember),
//! [`heka_recent`](HekaMcpServer::heka_recent), and
//! [`heka_status`](HekaMcpServer::heka_status).
//!
//! # Tenant scoping
//!
//! The axum route `/akh/{akh_id}/mcp` is guarded by
//! [`crate::auth::require_akh_bearer`], which injects a validated
//! [`crate::auth::HekaTenant`] into the request extensions.
//! [`tenant_mcp_service`] claims that extension and hands
//! `StreamableHttpService` the *per-request* [`HekaContext`], so each MCP
//! session is pinned to exactly one akh. Tool handlers only ever see their
//! own tenant; cross-tenant queries are structurally impossible here rather
//! than merely filtered.
//!
//! # What this slice deliberately does not implement
//!
//! Hybrid/vector recall, contact/topic views, `heka_explain`, and
//! `heka_contradict` belong to the advanced tool slice. They extend the same
//! [`HekaContext`] + [`HekaMcpServer`] seam when they land.

use std::sync::Arc;

use axum::http::request::Parts;
use chrono::{DateTime, Utc};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData as McpError, tool, tool_router};
use schemars::JsonSchema;
use sea_orm::{
    ColumnTrait, DbBackend, DatabaseConnection, EntityTrait, PaginatorTrait, QueryFilter,
    QueryOrder, QuerySelect, Statement,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity::{heka_drawer_chunks, heka_drawers};

/// Per-akH request context for MCP tool handlers — the shared seam the
/// advanced tool slice extends. Constructed once per MCP session by
/// [`tenant_mcp_service`] from the tenant the auth middleware validated.
#[derive(Clone)]
pub struct HekaContext {
    pub db: Arc<DatabaseConnection>,
    pub akh_id: Uuid,
    pub akh_name: String,
}

/// MCP server exposing Heka's foundational tools, bound to one akh.
#[derive(Clone)]
pub struct HekaMcpServer {
    ctx: HekaContext,
    tool_router: ToolRouter<Self>,
}

// ── Tool parameter types ────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberParams {
    /// The verbatim memory text to store (a session turn, note, or fact).
    /// Must be non-empty after trimming.
    body: String,
    /// Optional session identifier linking this memory to its source
    /// conversation (e.g. an Anima agent-session UUID).
    #[serde(default)]
    session_id: Option<String>,
    /// Who produced this memory: e.g. "user", "assistant", or "system".
    #[serde(default)]
    speaker: Option<String>,
    /// Freeform contact label (e.g. "claude-code@laptop").
    #[serde(default)]
    contact: Option<String>,
    /// Topic tags for categorization (stored as a JSONB array).
    #[serde(default)]
    topics: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecentParams {
    /// Maximum memories to return per call (1–100, default 20).
    #[serde(default = "default_recent_limit")]
    limit: u32,
    /// Stable pagination cursor: return memories strictly older than this
    /// cursor. Copy `next_cursor` from the previous response verbatim.
    /// Omit for the newest page.
    #[serde(default)]
    before: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StatusParams {}

fn default_recent_limit() -> u32 {
    20
}

/// Hard bounds for one `heka_recent` page.
const RECENT_MAX_LIMIT: u32 = 100;

// ── Tool response types ─────────────────────────────────────────────────

#[derive(Debug, Serialize, JsonSchema)]
struct RememberResponse {
    /// The drawer UUID (existing if this body was remembered before).
    drawer_id: Uuid,
    /// Chunks created this call (`0` when the call was a duplicate).
    chunk_count: usize,
    /// Total words across all chunks.
    word_count: usize,
    /// True when the body already existed — re-remembering is a no-op.
    duplicated: bool,
    /// SHA-256 hex digest of the body (the idempotency key).
    body_hash: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RecentSummary {
    id: Uuid,
    title: String,
    created_at: DateTime<Utc>,
    session_id: Option<String>,
    speaker: Option<String>,
    topics: serde_json::Value,
    summary: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RecentResponse {
    memories: Vec<RecentSummary>,
    /// `true` when another page exists before this one.
    has_more: bool,
    /// Cursor for the next page: pass it back as `before`. Null when
    /// `has_more` is false. Format is opaque — do not construct it by hand.
    next_cursor: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct StatusResponse {
    akh_id: Uuid,
    akh_name: String,
    drawers: u64,
    chunks: u64,
    /// Chunks still waiting for the embed worker (`embedded_at IS NULL`).
    pending_embeddings: u64,
    embedded_chunks: u64,
}

// ── Keyset cursor ───────────────────────────────────────────────────────
//
// A page boundary is the (created_at, id) of the oldest row on the current
// page. Keyset pagination over that composite key is stable under concurrent
// inserts (no OFFSET drift). `created_at` has microsecond precision, so the
// drawer UUID disambiguates ties.

fn encode_cursor(created_at: DateTime<Utc>, id: Uuid) -> String {
    let micros = created_at.timestamp_micros();
    format!("{}:{}", if micros < 0 { 0 } else { micros as u64 }, id)
}

fn decode_cursor(raw: &str) -> Result<(chrono::DateTime<Utc>, Uuid), String> {
    let (micros_raw, id_raw) = raw
        .split_once(':')
        .ok_or_else(|| "cursor must be '<micros>:<uuid>'".to_string())?;
    let micros: i64 = micros_raw
        .parse()
        .map_err(|_| "cursor timestamp is not an integer".to_string())?;
    let created_at = DateTime::<Utc>::from_timestamp_micros(micros)
        .ok_or_else(|| "cursor timestamp out of range".to_string())?;
    let id = Uuid::parse_str(id_raw).map_err(|_| "cursor id is not a UUID".to_string())?;
    Ok((created_at, id))
}

// ── MCP server implementation ───────────────────────────────────────────

#[tool_router]
impl HekaMcpServer {
    pub fn new(ctx: HekaContext) -> Self {
        Self {
            ctx,
            tool_router: Self::tool_router(),
        }
    }

    /// The akh this server instance is scoped to.
    pub fn context(&self) -> &HekaContext {
        &self.ctx
    }

    #[tool(
        name = "heka_remember",
        description = "Store a memory verbatim in this akh's long-term drawer store. \
            TRIGGER when the user asks the akh to remember, save, note, or capture a fact, turn, \
            decision, or observation; also when a session turn should be persisted for later \
            recall. Idempotent: remembering the same body twice is a safe no-op that returns the \
            existing drawer. \
            DO NOT TRIGGER for transient scratch working, questions about existing memories (use \
            heka_recent), diagnostic counts (use heka_status), or speculative drafts the user did \
            not ask to keep. Not a semantic search — this writes, it does not read."
    )]
    async fn heka_remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let body = params.body.trim();
        if body.is_empty() {
            return Err(McpError::invalid_params(
                "body must not be empty".to_string(),
                None,
            ));
        }
        // WI-39 ingest path — the single, idempotent writer. No competing
        // insert logic belongs in the MCP slice.
        let summary = crate::ingest::write_drawer(
            &self.ctx.db,
            self.ctx.akh_id,
            params.session_id.as_deref(),
            params.speaker.as_deref(),
            body,
            params.contact.as_deref(),
            &params.topics,
        )
        .await
        .map_err(|error| McpError::internal_error(format!("remember: {error}"), None))?;

        let response = RememberResponse {
            drawer_id: summary.drawer_id,
            chunk_count: summary.chunk_count,
            word_count: summary.word_count,
            duplicated: summary.duplicated,
            body_hash: summary.body_hash,
        };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|error| McpError::internal_error(format!("serialize: {error}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "heka_recent",
        description = "List this akh's most recent memories newest-first with stable keyset \
            pagination. TRIGGER for 'what happened lately', catching up on recent session turns, \
            or walking backwards through time; pass `before` = the previous response's \
            `next_cursor` to page further back. \
            DO NOT TRIGGER for semantic/vector search over memory content (not part of this \
            slice), for aggregate counts (use heka_status), or with a `before` string you \
            constructed yourself — cursors are opaque. Results are scoped to the authenticated \
            akh only."
    )]
    async fn heka_recent(
        &self,
        Parameters(params): Parameters<RecentParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.limit == 0 || params.limit > RECENT_MAX_LIMIT {
            return Err(McpError::invalid_params(
                format!("limit must be 1..={RECENT_MAX_LIMIT}"),
                None,
            ));
        }
        let cursor = match params.before.as_deref() {
            Some(raw) => Some(
                decode_cursor(raw)
                    .map_err(|error| McpError::invalid_params(error, None))?,
            ),
            None => None,
        };
        let page = fetch_recent(&self.ctx.db, self.ctx.akh_id, params.limit, cursor)
            .await
            .map_err(|error| McpError::internal_error(format!("recent: {error}"), None))?;
        let json = serde_json::to_string_pretty(&page)
            .map_err(|error| McpError::internal_error(format!("serialize: {error}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "heka_status",
        description = "Report this akh's store statistics: drawer count, chunk count, and embed \
            worker backlog (pending vs embedded chunks). TRIGGER for health checks, 'is the embed \
            worker keeping up', or confirming a remember landed before paging with heka_recent. \
            DO NOT TRIGGER for content recall (use heka_recent), for other akhs' statistics \
            (scoped strictly to the authenticated akh), or as a semantic-search probe. Read-only \
            and side-effect free."
    )]
    async fn heka_status(
        &self,
        Parameters(_params): Parameters<StatusParams>,
    ) -> Result<CallToolResult, McpError> {
        let status = akh_status(&self.ctx.db, self.ctx.akh_id, &self.ctx.akh_name)
            .await
            .map_err(|error| McpError::internal_error(format!("status: {error}"), None))?;
        let json = serde_json::to_string_pretty(&status)
            .map_err(|error| McpError::internal_error(format!("serialize: {error}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

// ── Query helpers ───────────────────────────────────────────────────────

/// Fetch one newest-first page of `heka_drawers` for `akh_id`, strictly older
/// than `cursor` when one is given.
async fn fetch_recent(
    db: &DatabaseConnection,
    akh_id: Uuid,
    limit: u32,
    cursor: Option<(DateTime<Utc>, Uuid)>,
) -> crate::HekaResult<RecentResponse> {
    let mut query = heka_drawers::Entity::find().filter(heka_drawers::Column::AkhId.eq(akh_id));

    if let Some((created_at, id)) = cursor {
        // Keyset boundary: strictly older than the boundary row, with the
        // UUID tiebreak preserving order within a shared microsecond.
        let condition = heka_drawers::Column::CreatedAt
            .lt(created_at)
            .or(heka_drawers::Column::CreatedAt
                .eq(created_at)
                .and(heka_drawers::Column::Id.lt(id)));
        query = query.filter(condition);
    }

    let rows = query
        .order_by_desc(heka_drawers::Column::CreatedAt)
        .order_by_desc(heka_drawers::Column::Id)
        .limit(u64::from(limit))
        .all(db)
        .await?;

    let mut page = RecentResponse {
        memories: rows
            .iter()
            .map(|row| RecentSummary {
                id: row.id,
                title: row.title.clone(),
                created_at: row.created_at,
                session_id: row.session_id.clone(),
                speaker: row.speaker.clone(),
                topics: row.topics.clone(),
                summary: row.summary.clone(),
            })
            .collect(),
        has_more: false,
        next_cursor: None,
    };

    // has_more: one existence probe for the row immediately before this page,
    // cheaper than over-fetching limit+1 and matches the cursor either way.
    if let Some(oldest) = rows.last() {
        let older_exists = !heka_drawers::Entity::find()
            .filter(heka_drawers::Column::AkhId.eq(akh_id))
            .filter(
                heka_drawers::Column::CreatedAt.lt(oldest.created_at).or(
                    heka_drawers::Column::CreatedAt
                        .eq(oldest.created_at)
                        .and(heka_drawers::Column::Id.lt(oldest.id)),
                ),
            )
            .limit(1)
            .all(db)
            .await?
            .is_empty();
        page.has_more = older_exists;
        if older_exists {
            page.next_cursor = Some(encode_cursor(oldest.created_at, oldest.id));
        }
    }

    Ok(page)
}

/// Per-akH store statistics, scoped strictly to `akh_id`.
async fn akh_status(
    db: &DatabaseConnection,
    akh_id: Uuid,
    akh_name: &str,
) -> crate::HekaResult<StatusResponse> {
    let drawers = heka_drawers::Entity::find()
        .filter(heka_drawers::Column::AkhId.eq(akh_id))
        .count(db)
        .await?;

    // Chunk counts scoped via the drawer FK join — chunks have no akh column.
    let chunk_counts = db
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT COUNT(*)::bigint, COUNT(*) FILTER (WHERE c.embedded_at IS NULL)::bigint \
             FROM heka_drawer_chunks c \
             JOIN heka_drawers d ON d.id = c.drawer_id \
             WHERE d.akh_id = $1",
            [akh_id.into()],
        ))
        .await?
        .ok_or_else(|| {
            crate::HekaError::Database(sea_orm::DbErr::RecordNotFound(
                "chunk count aggregate returned no row".to_string(),
            ))
        })?;
    let chunks: i64 = chunk_counts.try_get_by_index(0)?;
    let pending: i64 = chunk_counts.try_get_by_index(1)?;

    Ok(StatusResponse {
        akh_id,
        akh_name: akh_name.to_string(),
        drawers,
        chunks: chunks.max(0) as u64,
        pending_embeddings: pending.max(0) as u64,
        embedded_chunks: (chunks - pending).max(0) as u64,
    })
}

// ── Transport wiring ────────────────────────────────────────────────────

/// Build the `StreamableHttpService` for the tenant route. `state` carries
/// the shared database handle; the closure runs once per MCP session
/// establishment and reads the validated tenant from the request extensions
/// populated by [`crate::auth::require_akh_bearer`]. A missing extension is a
/// wiring bug — the route must always be behind auth — so it is rejected.
pub fn tenant_mcp_service(
    state: crate::auth::AkhAuthState,
) -> StreamableHttpService<HekaMcpServer, LocalSessionManager> {
    StreamableHttpService::new(
        move |parts: &Parts| {
            let tenant = parts
                .extensions
                .get::<crate::auth::HekaTenant>()
                .cloned()
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "heka MCP route reached without a validated tenant — \
                         mount require_akh_bearer in front of this service",
                    )
                })?;
            Ok(HekaMcpServer::new(HekaContext {
                db: state.db(),
                akh_id: tenant.akh_id,
                akh_name: tenant.akh_name,
            }))
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    )
}

// ── ServerHandler impl ──────────────────────────────────────────────────

#[rmcp::tool_handler]
impl rmcp::handler::server::ServerHandler for HekaMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "heka".to_string(),
                title: Some("Heka long-term memory".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some(
                    "Per-akh long-term memory store (Postgres + pgvector)".to_string(),
                ),
                icons: None,
                website_url: Some("https://akh-medu.dev".to_string()),
            },
            instructions: Some(
                "Heka stores and recalls this akh's long-term memories. \
                 Use `heka_remember` to store a memory verbatim (idempotent). \
                 Use `heka_recent` to page backwards through recent memories \
                 (stable `before` cursor). \
                 Use `heka_status` for store counts and embed-worker backlog. \
                 All tools are scoped to the authenticated akh."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The MCP Inspector contract: exactly these three tools, in this order.
    #[test]
    fn tool_router_lists_exactly_the_slice_tools() {
        let server = HekaMcpServer::new(HekaContext {
            // No database handle exists in a unit test, and none is needed —
            // the router's inventory is macro-generated from the impl block.
            db: Arc::new(
                sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres).into_connection(),
            ),
            akh_id: Uuid::new_v4(),
            akh_name: "test-akh".to_string(),
        });
        let names: Vec<&str> = server
            .tool_router
            .list_all()
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect();
        assert_eq!(names, vec!["heka_remember", "heka_recent", "heka_status"]);
    }

    #[test]
    fn tool_descriptions_carry_trigger_and_anti_trigger_guidance() {
        let server = HekaMcpServer::new(HekaContext {
            db: Arc::new(
                sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres).into_connection(),
            ),
            akh_id: Uuid::new_v4(),
            akh_name: "test-akh".to_string(),
        });
        for tool in server.tool_router.list_all() {
            let description = tool.description.as_ref().expect("description");
            assert!(
                description.contains("TRIGGER"),
                "{} missing TRIGGER guidance",
                tool.name
            );
            assert!(
                description.contains("DO NOT TRIGGER"),
                "{} missing anti-trigger guidance",
                tool.name
            );
        }
    }

    #[test]
    fn cursor_round_trips_and_decodes_strictly() {
        let when = DateTime::from_timestamp(1_756_382_400, 123_456_000).unwrap();
        let id = Uuid::new_v4();
        let encoded = encode_cursor(when, id);
        let (decoded_when, decoded_id) = decode_cursor(&encoded).unwrap();
        assert_eq!(decoded_when, when);
        assert_eq!(decoded_id, id);

        assert!(decode_cursor("no-separator").is_err());
        assert!(decode_cursor("9999999999999999999999:00000000-0000-0000-0000-000000000000").is_err());
        assert!(decode_cursor(&format!("{}:not-a-uuid", when.timestamp_micros())).is_err());
        // Negative (pre-epoch) timestamps clamp rather than misparsing.
        let clamped = encode_cursor(DateTime::from_timestamp_millis(-1).unwrap(), id);
        let (decoded, _) = decode_cursor(&clamped).unwrap();
        assert_eq!(decoded, DateTime::<Utc>::UNIX_EPOCH);
    }

    /// The recent-page query must scope to the akh, order newest-first on the
    /// composite key, and apply the keyset boundary strictly.
    #[tokio::test]
    async fn recent_query_is_tenant_scoped_and_keyset_ordered() {
        let db = sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres)
            .append_query_results::<heka_drawers::Model>(vec![Vec::<heka_drawers::Model>::new()])
            .into_connection();
        let akh = Uuid::new_v4();
        let boundary = DateTime::from_timestamp_micros(1_756_382_400_000_000).unwrap();
        let boundary_id = Uuid::new_v4();

        let page = fetch_recent(&db, akh, 20, Some((boundary, boundary_id)))
            .await
            .unwrap();
        assert!(page.memories.is_empty());
        assert!(!page.has_more);
        assert!(page.next_cursor.is_none());

        let sql = db.into_transaction_log()[0].sql();
        assert!(sql.contains("\"akh_id\" ="), "tenant scope: {sql}");
        assert!(
            sql.contains("ORDER BY \"created_at\" DESC") && sql.contains("\"id\" DESC"),
            "composite newest-first order: {sql}"
        );
        assert!(
            sql.contains("\"created_at\" <") && sql.contains("\"id\" <"),
            "keyset boundary predicates: {sql}"
        );
    }

    /// Tenant isolation holds for the status aggregate as well: the chunk
    /// counts come from a join constrained by `d.akh_id = $1`.
    #[tokio::test]
    async fn status_counts_are_tenant_scoped() {
        #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
        #[sea_orm(table_name = "counts")]
        struct CountRow {
            #[sea_orm(primary_key, auto_increment = false)]
            pub total: i64,
            pub pending: i64,
        }

        let db = sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres)
            .append_query_results::<heka_drawers::Model>(vec![
                (1u64..).map(|_| heka_drawers::Model::default_for_count(7)).collect::<Vec<_>>(),
            ])
            .into_connection();

        // Simpler: assert the status SQL itself carries the tenant predicate,
        // using the mock's transaction log rather than hydrated rows.
        let _ = db;
        let db = sea_orm::MockDatabase::new(sea_orm::DatabaseBackend::Postgres)
            .append_query_results::<i64>(vec![vec![0i64]]) // drawer count
            .append_query_results::<i64>(vec![vec![5i64, 2i64]]) // chunks, pending
            .into_connection();
        let status = akh_status(&db, Uuid::new_v4(), "test-akh").await.unwrap();
        assert_eq!(status.chunks, 5);
        assert_eq!(status.pending_embeddings, 2);
        assert_eq!(status.embedded_chunks, 3);

        let statements = db.into_transaction_log();
        let chunk_sql = statements[1].sql();
        assert!(chunk_sql.contains("JOIN heka_drawers"));
        assert!(chunk_sql.contains("WHERE d.akh_id ="), "tenant scope: {chunk_sql}");
    }
}
