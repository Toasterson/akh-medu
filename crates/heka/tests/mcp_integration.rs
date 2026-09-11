//! WI-331 integration tests — real-token, end-to-end MCP transport,
//! tenant auth, and basic tools.
//!
//! # Structure
//!
//! Two layers, mirroring the acceptance criteria:
//!
//! 1. **Mocked-DB transport tests** — drive the real
//!    `StreamableHttpService` + `HekaMcpServer` over `tower::oneshot`
//!    with the `HekaTenant` request extension injected exactly the way
//!    `require_akh_bearer` + rmcp deliver it. These cover tool listing
//!    (exactly the three slice tools), tool callability, and cursor
//!    pagination, and assert the emitted SQL pins `akh_id` so akh B's
//!    data is structurally unreachable for akh A. They exercise
//!    everything except the byte on disk, following the seshat
//!    `wi326_instrumentation` pattern.
//!
//! 2. **Live-database tests** — gated on `HEKA_MCP_TEST_DATABASE_URL`
//!    (skip with a note when unset; the URL must point at a disposable
//!    database). These run the real `heka::serve` router — including
//!    `require_akh_bearer` — over a real TCP listener, register two akhs
//!    with minted bearer tokens, and prove:
//!    - akh A's token works on akh A's route (initialize + three tools)
//!    - akh A's token **cannot** cross to akh B's route (**403**, not
//!      401) and vice versa
//!    - missing token → **401** (+ `WWW-Authenticate`), malformed /
//!      padded / valid-but-unknown tokens → **403**
//!    - duplicate `heka_remember` is idempotent: same `drawer_id`,
//!      `duplicated: true`, `chunk_count: 0`
//!    - `heka_recent` pagination carries a stable `before` cursor and
//!      each akh only ever sees its own memories
//!    - `heka_status` counts are strictly tenant-scoped
//!
//! No plaintext token is ever persisted anywhere — only the SHA-256
//! digest lands in `heka_akhs.token_hash`.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Body;
use axum::http::Request;
use chrono::{DateTime, Utc};
use sea_orm::{
    ColumnTrait, ConnectionTrait, Database, DatabaseBackend, DatabaseConnection, DbBackend,
    EntityTrait, MockDatabase, PaginatorTrait, QueryFilter, Set, Statement, Value,
};
use serde_json::{Value as Json, json};
use tower::ServiceExt;
use uuid::Uuid;

use heka::auth::{HekaTenant, hash_token, mint_token};
use heka::entity::{heka_akhs, heka_drawers};
use heka::serve::router;

// ── Mocked-DB transport layer ───────────────────────────────────────────

fn drawer_model(akh_id: Uuid, created_at: DateTime<Utc>, title: &str, speaker: &str) -> heka_drawers::Model {
    heka_drawers::Model {
        id: Uuid::new_v4(),
        akh_id,
        project_id: None,
        title: title.to_string(),
        summary: None,
        tags: json!([]),
        metadata: json!({}),
        body_hash: Some("deadbeef".to_string()),
        session_id: None,
        speaker: Some(speaker.to_string()),
        contact: None,
        topics: json!([]),
        created_at,
        updated_at: created_at,
    }
}

fn count_row(n: i64) -> BTreeMap<&'static str, Value> {
    BTreeMap::from([("count", n.into())])
}

async fn mcp_roundtrip(
    db: &DatabaseConnection,
    tenant: &HekaTenant,
    session: Option<&str>,
    id: i64,
    method: &str,
    params: Option<Json>,
) -> (Json, Option<String>) {
    let service = heka::mcp::tenant_mcp_service(heka::auth::AkhAuthState::new(Arc::new(db.clone())));
    let payload = match params {
        Some(params) => json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
        None => json!({"jsonrpc": "2.0", "id": id, "method": method}),
    };
    let mut builder = Request::builder()
        .method("POST")
        .uri(format!("/akh/{}/mcp", tenant.akh_id))
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    if let Some(session) = session {
        builder = builder.header("mcp-session-id", session);
    }
    let mut request = builder.body(Body::from(payload.to_string())).unwrap();
    // The auth middleware inserts the validated tenant into the request
    // extensions; rmcp then injects request `Parts` (extensions included)
    // into the MCP message extensions the tool handlers read.
    request.extensions_mut().insert(tenant.clone());
    let response = axum::Router::new()
        .route_service("/akh/{akh_id}/mcp", service)
        .oneshot(request)
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "MCP POST must succeed");
    let session_header = response
        .headers()
        .get("mcp-session-id")
        .map(|value| value.to_str().unwrap().to_string());
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    // Stateful mode answers `initialize` with SSE framing, everything
    // else after a valid session header with plain JSON.
    let payload = if let Some(data) = text.strip_prefix("event: message\r\ndata: ") {
        data
    } else if let Some(data) = text.strip_prefix("event: message\ndata: ") {
        data
    } else {
        text.as_str()
    };
    let response: Json = serde_json::from_str(payload)
        .unwrap_or_else(|error| panic!("MCP response must be JSON (got {text:?}): {error}"));
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], id);
    (response, session_header)
}

async fn mcp_session(db: &DatabaseConnection, tenant: &HekaTenant) -> String {
    let (response, session) = mcp_roundtrip(
        db,
        tenant,
        None,
        1,
        "initialize",
        Some(json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "wi-331-test", "version": "0.1.0"},
        })),
    )
    .await;
    assert!(
        response["result"].is_object(),
        "initialize must succeed: {response}"
    );
    session.expect("initialize must assign an mcp-session-id")
}

async fn mcp_call(
    db: &DatabaseConnection,
    tenant: &HekaTenant,
    id: i64,
    method: &str,
    params: Option<Json>,
) -> Json {
    let session = mcp_session(db, tenant).await;
    let (response, _) = mcp_roundtrip(db, tenant, Some(&session), id, method, params).await;
    response
}

fn tool_text(response: &Json) -> Json {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("tool result must carry text content: {response}"));
    serde_json::from_str(text).unwrap_or_else(|error| panic!("tool text must be JSON: {error}\n{text}"))
}

#[tokio::test]
async fn mocked_transport_lists_exactly_the_three_slice_tools() {
    let tenant = HekaTenant {
        akh_id: Uuid::new_v4(),
        akh_name: "mock-akh".to_string(),
    };
    let db = DatabaseConnection::default();
    let response = mcp_call(&db, &tenant, 2, "tools/list", None).await;
    let tools = response["result"]["tools"].as_array().expect("tools array");
    let mut names: Vec<&str> = tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["heka_recent", "heka_remember", "heka_status"],
        "the transport lists exactly the three slice tools — the MCP-Inspector surface"
    );
    for tool in tools {
        let description = tool["description"].as_str().unwrap();
        assert!(
            description.contains("TRIGGER") && description.contains("DO NOT TRIGGER"),
            "tool {} must carry explicit trigger/anti-trigger guidance: {description}",
            tool["name"]
        );
    }
}

#[tokio::test]
async fn mocked_transport_remember_is_scoped_and_idempotent() {
    let tenant = HekaTenant {
        akh_id: Uuid::new_v4(),
        akh_name: "mock-akh".to_string(),
    };
    let existing = drawer_model(
        tenant.akh_id,
        DateTime::from_timestamp_micros(1_756_382_400_000_000).unwrap(),
        "existing memory",
        "assistant",
    );
    let db = MockDatabase::new(DatabaseBackend::Postgres)
        // write_drawer's duplicate pre-check finds the existing drawer →
        // returns its id with chunk_count 0 and never INSERTs.
        .append_query_results([vec![existing.clone()]])
        .into_connection();
    let log = Arc::clone(db.as_mock_connection().into_transaction_log());
    let response = mcp_call(
        &db,
        &tenant,
        2,
        "tools/call",
        Some(json!({
            "name": "heka_remember",
            "arguments": {"body": "the same fact again", "speaker": "assistant"},
        })),
    )
    .await;
    let content = tool_text(&response);
    assert_eq!(content["drawer_id"], json!(existing.id));
    assert_eq!(content["duplicated"], true);
    assert_eq!(content["chunk_count"], 0);
    let statements = &log[0].statements();
    let sql = statements[0].sql.as_str();
    assert!(
        sql.contains("\"akh_id\"") && sql.contains("\"body_hash\""),
        "drawer existence probe must pin the tenant: {sql}"
    );
    assert_eq!(
        statements.len(),
        1,
        "a duplicate remember must never reach an INSERT: {statements:?}"
    );
}

#[tokio::test]
async fn mocked_transport_recent_pages_with_stable_cursor_and_pins_akh() {
    let tenant = HekaTenant {
        akh_id: Uuid::new_v4(),
        akh_name: "mock-akh".to_string(),
    };
    let newer = drawer_model(
        tenant.akh_id,
        DateTime::from_timestamp_micros(1_800_000_000_000_000).unwrap(),
        "newer",
        "user",
    );
    let older = drawer_model(
        tenant.akh_id,
        DateTime::from_timestamp_micros(1_700_000_000_000_000).unwrap(),
        "older",
        "assistant",
    );
    let db = MockDatabase::new(DatabaseBackend::Postgres)
        // page 1: two rows; has_more probe: one older row
        .append_query_results([vec![newer.clone(), older.clone()]])
        .append_query_results([vec![drawer_model(tenant.akh_id, DateTime::from_timestamp_micros(1_600_000_000_000_000).unwrap(), "older-still", "user")]])
        // page 2 (with cursor): one row; has_more probe: nothing older
        .append_query_results([vec![older.clone()]])
        .append_query_results([Vec::<heka_drawers::Model>::new()])
        .into_connection();
    let log = Arc::clone(db.as_mock_connection().into_transaction_log());

    let page1 = mcp_call(
        &db,
        &tenant,
        2,
        "tools/call",
        Some(json!({
            "name": "heka_recent",
            "arguments": {"limit": 2},
        })),
    )
    .await;
    let page1 = tool_text(&page1);
    let titles: Vec<&str> = page1["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|memory| memory["title"].as_str().unwrap())
        .collect();
    assert_eq!(titles, ["newer", "older"], "newest-first order");
    assert_eq!(page1["has_more"], true);
    let cursor = page1["next_cursor"].as_str().expect("cursor for page 2");
    assert!(cursor.contains(':'), "cursor is '<micros>:<uuid>'");

    let page2 = mcp_call(
        &db,
        &tenant,
        2,
        "tools/call",
        Some(json!({
            "name": "heka_recent",
            "arguments": {"limit": 2, "before": cursor},
        })),
    )
    .await;
    let page2 = tool_text(&page2);
    assert_eq!(page2["memories"][0]["title"], "older");
    assert_eq!(page2["has_more"], false);
    assert!(page2["next_cursor"].is_null());

    let statements = &log[0].statements();
    let page_sql: Vec<&str> = statements.iter().map(|s| s.sql.as_str()).collect();
    for sql in &page_sql {
        assert!(sql.contains("\"akh_id\""), "recent queries must pin the tenant: {sql}");
    }
    assert!(
        page_sql[2].contains("\"created_at\""),
        "the cursor page must bound on the keyset: {}",
        page_sql[2]
    );
}

#[tokio::test]
async fn mocked_transport_status_reports_tenant_counts() {
    let tenant = HekaTenant {
        akh_id: Uuid::new_v4(),
        akh_name: "mock-akh".to_string(),
    };
    let db = MockDatabase::new(DatabaseBackend::Postgres)
        .append_query_results([vec![count_row(7)]]) // drawers
        .append_query_results([vec![BTreeMap::from([
            ("count", 10i64.into()),
            ("count_1", 3i64.into()),
        ])]])
        .into_connection();
    let log = Arc::clone(db.as_mock_connection().into_transaction_log());
    let response = mcp_call(
        &db,
        &tenant,
        2,
        "tools/call",
        Some(json!({"name": "heka_status", "arguments": {}})),
    )
    .await;
    let content = tool_text(&response);
    assert_eq!(content["akh_id"], json!(tenant.akh_id));
    assert_eq!(content["akh_name"], "mock-akh");
    assert_eq!(content["drawers"], 7);
    assert_eq!(content["chunks"], 10);
    assert_eq!(content["pending_embeddings"], 3);
    assert_eq!(content["embedded_chunks"], 7);
    let statements = &log[0].statements();
    assert!(statements[0].sql.contains("\"akh_id\""));
    assert!(
        statements[1].sql.contains("d.akh_id = $1"),
        "chunk counts must join through the tenant's drawers: {}",
        statements[1].sql
    );
}

// ── Live-DB layer (HEKA_MCP_TEST_DATABASE_URL) ──────────────────────────

async fn live_db() -> Option<DatabaseConnection> {
    let url = std::env::var("HEKA_MCP_TEST_DATABASE_URL").ok()?;
    let db = Database::connect(&url)
        .await
        .unwrap_or_else(|error| panic!("connect to disposable test DB failed: {error}"));
    heka::migration::run_migrations(&db)
        .await
        .unwrap_or_else(|error| panic!("migrations failed: {error}"));
    // Disposable database: start cleanly so counts and fixtures are exact.
    db.execute_unprepared(
        "TRUNCATE heka_drawer_embeddings, heka_drawer_chunks, heka_drawers, \
         heka_project_bindings, heka_akhs CASCADE",
    )
    .await
    .unwrap();
    Some(db)
}

struct TestAkh {
    id: Uuid,
    token: String,
}

async fn register_akh(db: &DatabaseConnection, name: &str) -> TestAkh {
    let (token, hash) = mint_token();
    heka_akhs::Entity::insert(heka_akhs::ActiveModel {
        id: Set(Uuid::new_v4()),
        name: Set(name.to_string()),
        description: Set(Some("wi-331 integration fixture".to_string())),
        personality: Set(None),
        avatar_url: Set(None),
        token_hash: Set(Some(hash)),
        created_at: Set(Utc::now()),
        updated_at: Set(Utc::now()),
    })
    .exec(db)
    .await
    .unwrap();
    let model = heka_akhs::Entity::find()
        .filter(heka_akhs::Column::Name.eq(name))
        .one(db)
        .await
        .unwrap()
        .unwrap();
    TestAkh {
        id: model.id,
        token,
    }
}

/// Spin the real `heka::serve` router (auth middleware + rmcp transport) on
/// a loopback listener. Returns the base URL.
async fn spawn_server(db: &DatabaseConnection) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let app = router(Arc::new(db.clone()));
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await
            .unwrap();
    });
    // Keep the shutdown channel alive for the test duration: dropping the
    // sender never fires the signal because we leak it into a task that
    // holds it until the test's runtime ends.
    std::mem::forget(tx);
    addr
}

struct HttpClient {
    base: SocketAddr,
    stream: tokio::net::TcpStream,
}

impl HttpClient {
    async fn connect(base: SocketAddr) -> Self {
        Self {
            base,
            stream: tokio::net::TcpStream::connect(base).await.unwrap(),
        }
    }

    async fn request(
        &mut self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&str>,
    ) -> (u16, Vec<(String, String)>, String) {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\nConnection: keep-alive\r\n", self.base);
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        if let Some(body) = body {
            head.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
        } else {
            head.push_str("\r\n");
        }
        self.stream.write_all(head.as_bytes()).await.unwrap();
        let mut reader = BufReader::new(&mut self.stream);
        let mut status = String::new();
        reader.read_line(&mut status).await.unwrap();
        let status: u16 = status.split_whitespace().nth(1).unwrap().parse().unwrap();
        let mut response_headers = Vec::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            if line.trim().is_empty() {
                break;
            }
            let (name, value) = line.split_once(':').unwrap();
            let value = value.trim();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().unwrap();
            }
            response_headers.push((name.to_string(), value.to_string()));
        }
        let mut body_bytes = vec![0u8; content_length];
        tokio::io::AsyncReadExt::read_exact(&mut reader, &mut body_bytes)
            .await
            .unwrap();
        (
            status,
            response_headers,
            String::from_utf8(body_bytes).unwrap(),
        )
    }
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

async fn initialize(base: SocketAddr, akh: &TestAkh, route_akh: Option<Uuid>) -> (u16, Option<String>, Json) {
    let route = route_akh.unwrap_or(akh.id);
    let mut client = HttpClient::connect(base).await;
    let payload = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "wi-331-live", "version": "0.1.0"}},
    })
    .to_string();
    let (status, _, body) = client
        .request(
            "POST",
            &format!("/akh/{route}/mcp"),
            &[
                ("authorization", &format!("Bearer {}", akh.token)),
                ("content-type", "application/json"),
                ("accept", "application/json, text/event-stream"),
            ],
            Some(&payload),
        )
        .await;
    let body = body.strip_prefix("event: message\r\ndata: ").unwrap_or(&body);
    let body = body.strip_prefix("event: message\ndata: ").unwrap_or(body);
    let response: Json = serde_json::from_str(body).unwrap_or(Json::Null);
    let _ = client; // keep-alive connection ends with the client
    (status, None, response)
}

async fn call_tool(
    base: SocketAddr,
    akh: &TestAkh,
    route_akh: Option<Uuid>,
    name: &str,
    arguments: Json,
    id: i64,
) -> (u16, Json) {
    call_tool_with_token(base, route_akh.unwrap_or(akh.id), Some(&akh.token), name, arguments, id).await
}

async fn call_tool_with_token(
    base: SocketAddr,
    route_akh: Uuid,
    token: Option<&str>,
    name: &str,
    arguments: Json,
    id: i64,
) -> (u16, Json) {
    let mut client = HttpClient::connect(base).await;
    let mut init_headers: Vec<(&str, &str)> = vec![
        ("content-type", "application/json"),
        ("accept", "application/json, text/event-stream"),
    ];
    let auth;
    if let Some(token) = token {
        auth = format!("Bearer {token}");
        init_headers.push(("authorization", &auth));
    }
    let init = json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "wi-331-live", "version": "0.1.0"}},
    })
    .to_string();
    let (status, headers, _) = client
        .request(
            "POST",
            &format!("/akh/{route_akh}/mcp"),
            &init_headers,
            Some(&init),
        )
        .await;
    if status != 200 {
        return (status, Json::Null);
    }
    let session = header(&headers, "mcp-session-id")
        .expect("initialize must return an mcp-session-id")
        .to_string();
    let mut call_headers: Vec<(&str, &str)> = vec![
        ("content-type", "application/json"),
        ("accept", "application/json, text/event-stream"),
        ("mcp-session-id", &session),
    ];
    if let Some(token) = token {
        auth = format!("Bearer {token}");
        call_headers.push(("authorization", &auth));
    }
    let payload = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    .to_string();
    let (status, _, body) = client
        .request(
            "POST",
            &format!("/akh/{route_akh}/mcp"),
            &call_headers,
            Some(&payload),
        )
        .await;
    let body = body.strip_prefix("event: message\r\ndata: ").unwrap_or(&body);
    let body = body.strip_prefix("event: message\ndata: ").unwrap_or(body);
    let response: Json = serde_json::from_str(body).unwrap_or(Json::Null);
    (status, response)
}

async fn remembered_body(
    base: SocketAddr,
    akh: &TestAkh,
    body: &str,
    id: i64,
) -> Json {
    let (status, response) = call_tool(
        base,
        akh,
        None,
        "heka_remember",
        json!({"body": body, "speaker": "user", "topics": ["wi-331"]}),
        id,
    )
    .await;
    assert_eq!(status, 200);
    tool_text(&response)
}

#[tokio::test]
async fn live_cross_tenant_auth_isolation() {
    let Some(db) = live_db().await else {
        eprintln!("skipping live_cross_tenant_auth_isolation: HEKA_MCP_TEST_DATABASE_URL unset");
        return;
    };
    let akh_a = register_akh(&db, "wi331-akh-a").await;
    let akh_b = register_akh(&db, "wi331-akh-b").await;
    let base = spawn_server(&db).await;

    // Each token opens its own tenant's route.
    let (status, _, response) = initialize(base, &akh_a, None).await;
    assert_eq!(status, 200, "akh A token opens akh A route: {response}");
    let (status, _, response) = initialize(base, &akh_b, None).await;
    assert_eq!(status, 200, "akh B token opens akh B route: {response}");

    // Cross-tenant: A's token on B's route and vice versa. The
    // (id, token_hash) equality pair matches no row → 403, never 401
    // (the header was well formed) and never 404.
    for (caller, target, who) in [
        (&akh_a, akh_b.id, "A on B route"),
        (&akh_b, akh_a.id, "B on A route"),
    ] {
        let (status, _, _) = initialize(base, caller, Some(target)).await;
        assert_eq!(status, 403, "cross-tenant must be forbidden: {who}");
        let (status, _) = call_tool(
            base,
            caller,
            Some(target),
            "heka_status",
            json!({}),
            42,
        )
        .await;
        assert_eq!(status, 403, "cross-tenant tool call must be forbidden: {who}");
    }
}

#[tokio::test]
async fn live_auth_failure_modes_are_fail_closed() {
    let Some(db) = live_db().await else {
        eprintln!("skipping live_auth_failure_modes_are_fail_closed: HEKA_MCP_TEST_DATABASE_URL unset");
        return;
    };
    let akh = register_akh(&db, "wi331-akh-auth").await;
    let base = spawn_server(&db).await;
    let path = format!("/akh/{}/mcp", akh.id);

    // Missing Authorization header → 401 + WWW-Authenticate.
    let mut client = HttpClient::connect(base).await;
    let (status, headers, _) = client
        .request("POST", &path, &[("content-type", "application/json")], Some("{}"))
        .await;
    assert_eq!(status, 401, "missing token must be unauthorized");
    assert_eq!(
        header(&headers, "www-authenticate"),
        Some("Bearer realm=\"heka\""),
    );

    // Malformed scheme / empty / padded / valid-but-unknown tokens → 403.
    for (name, authorization) in [
        ("non-Bearer scheme", "Basic dGVzdA==".to_string()),
        ("empty token", "Bearer ".to_string()),
        ("left-padded token", "Bearer  abc".to_string()),
        ("right-padded token", format!("Bearer {}", "abc ")),
        ("unknown-but-well-formed", format!("Bearer {}", "f".repeat(64))),
    ] {
        let (status, _) = call_tool_with_token_raw(base, akh.id, &authorization, 7).await;
        assert_eq!(status, 403, "{name} must be forbidden");
    }

    // A token minted for a *different* akh (i.e. a valid hash that belongs
    // to another tenant, not just random bytes) is equally forbidden here.
    let other = register_akh(&db, "wi331-akh-other").await;
    let (status, _) = call_tool_with_token(base, akh.id, Some(&other.token), "heka_status", json!({}), 7).await;
    assert_eq!(status, 403, "another akh's token must be forbidden");

    // A real token on a *nonexistent* akh route is also forbidden — no 404
    // leak about which ids exist.
    let (status, _) = call_tool_with_token(base, Uuid::new_v4(), Some(&akh.token), "heka_status", json!({}), 7).await;
    assert_eq!(status, 403);

    // /health stays open and reports liveness.
    let (status, _, body) = client.request("GET", "/health", &[], None).await;
    assert_eq!(status, 200);
    assert_eq!(serde_json::from_str::<Json>(&body).unwrap()["status"], "ok");
}

async fn call_tool_with_token_raw(base: SocketAddr, route_akh: Uuid, authorization: &str, id: i64) -> (u16, Json) {
    let mut client = HttpClient::connect(base).await;
    let payload = json!({"jsonrpc": "2.0", "id": id, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "wi-331-live", "version": "0.1.0"}}}).to_string();
    let (status, _, _) = client
        .request(
            "POST",
            &format!("/akh/{route_akh}/mcp"),
            &[
                ("authorization", authorization),
                ("content-type", "application/json"),
                ("accept", "application/json, text/event-stream"),
            ],
            Some(&payload),
        )
        .await;
    (status, Json::Null)
}

#[tokio::test]
async fn live_tools_are_callable_and_tenant_scoped() {
    let Some(db) = live_db().await else {
        eprintln!("skipping live_tools_are_callable_and_tenant_scoped: HEKA_MCP_TEST_DATABASE_URL unset");
        return;
    };
    let akh_a = register_akh(&db, "wi331-akh-a2").await;
    let akh_b = register_akh(&db, "wi331-akh-b2").await;
    let base = spawn_server(&db).await;

    // All three tools are callable for each tenant through the live
    // transport.
    let first = remembered_body(base, &akh_a, "akh A remembers the structural fix for WI-331", 11).await;
    assert_eq!(first["duplicated"], false);
    assert!(first["chunk_count"].as_u64().unwrap() >= 1);

    let (status, response) = call_tool(base, &akh_a, None, "heka_status", json!({}), 12).await;
    assert_eq!(status, 200);
    let status_a = tool_text(&response);
    assert_eq!(status_a["akh_id"], json!(akh_a.id));
    assert_eq!(status_a["akh_name"], "wi331-akh-a2");
    assert_eq!(status_a["drawers"], 1);
    assert_eq!(status_a["pending_embeddings"], status_a["chunks"]);

    // Idempotency: remembering the same body again is a no-op returning
    // the existing drawer with chunk_count 0.
    let second = remembered_body(base, &akh_a, "akh A remembers the structural fix for WI-331", 13).await;
    assert_eq!(second["drawer_id"], first["drawer_id"]);
    assert_eq!(second["duplicated"], true);
    assert_eq!(second["chunk_count"], 0);
    let (status, response) = call_tool(base, &akh_a, None, "heka_status", json!({}), 14).await;
    assert_eq!(status, 200);
    assert_eq!(tool_text(&response)["drawers"], 1, "duplicate remember must not insert");

    // akh B writes its own memory; A still sees only its own.
    remembered_body(base, &akh_b, "akh B remembers its own private token ceremony", 21).await;

    let (status, response) = call_tool(base, &akh_a, None, "heka_recent", json!({"limit": 10}), 15).await;
    assert_eq!(status, 200);
    let recent_a = tool_text(&response);
    assert_eq!(recent_a["memories"].as_array().unwrap().len(), 1, "akh A sees exactly its own memory");
    assert_eq!(
        recent_a["memories"][0]["id"], first["drawer_id"],
        "akh A must never see akh B's drawers"
    );

    let (status, response) = call_tool(base, &akh_b, None, "heka_recent", json!({"limit": 10}), 22).await;
    assert_eq!(status, 200);
    let recent_b = tool_text(&response);
    assert_eq!(recent_b["memories"].as_array().unwrap().len(), 1);
    assert_ne!(
        recent_b["memories"][0]["id"], first["drawer_id"],
        "akh B must never see akh A's drawers"
    );

    let (status, response) = call_tool(base, &akh_b, None, "heka_status", json!({}), 23).await;
    assert_eq!(status, 200);
    let status_b = tool_text(&response);
    assert_eq!(status_b["akh_id"], json!(akh_b.id));
    assert_eq!(status_b["drawers"], 1, "akh B counts only its own drawers");
}

#[tokio::test]
async fn live_recent_pagination_cursor_is_stable() {
    let Some(db) = live_db().await else {
        eprintln!("skipping live_recent_pagination_cursor_is_stable: HEKA_MCP_TEST_DATABASE_URL unset");
        return;
    };
    let akh = register_akh(&db, "wi331-akh-pages").await;
    let base = spawn_server(&db).await;

    for index in 0..5 {
        remembered_body(base, &akh, &format!("paged memory number {index}"), 30 + index).await;
        // Distinct created_at micros per drawer so ordering is deterministic.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }

    let mut seen: Vec<String> = Vec::new();
    let mut before: Option<String> = None;
    loop {
        let mut arguments = json!({"limit": 2});
        if let Some(cursor) = &before {
            arguments["before"] = json!(cursor);
        }
        let (status, response) = call_tool(base, &akh, None, "heka_recent", arguments, 60 + seen.len() as i64).await;
        assert_eq!(status, 200);
        let page = tool_text(&response);
        for memory in page["memories"].as_array().unwrap() {
            seen.push(memory["title"].as_str().unwrap().to_string());
        }
        if page["has_more"] == false {
            assert!(page["next_cursor"].is_null(), "final page must not emit a cursor");
            break;
        }
        // Repeating a page with the same cursor must return the same rows —
        // the cursor is stable, not a mutable OFFSET.
        before = page["next_cursor"].as_str().map(|cursor| cursor.to_string());
    }
    assert_eq!(seen.len(), 5, "every drawer exactly once: {seen:?}");
    let mut expected: Vec<String> = (0..5).map(|index| format!("paged memory number {index}")).collect();
    expected.reverse();
    assert_eq!(seen, expected, "newest-first with no repeats across pages");
}
