//! Live-Postgres integration test for increment 6, the `akh-transcripts-mcp`
//! read surface. Ignored by default; run against a scratch database from
//! `scripts/pg-standalone.sh` — the timescale+vector wrapper image, with
//! `shared_preload_libraries = 'timescaledb'` set and the container
//! restarted (see the module header of `tests/live_pg.rs` for the boot
//! dance, and the increment-6 brief for why that preload line matters):
//!
//! ```text
//! AKH_PG_IMAGE=code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 \
//!   AKH_PG_CONTAINER=akh-th-mcp-pg AKH_PG_VOLUME=akh-th-mcp-pg-data AKH_PG_PORT=55434 \
//!   AKH_PG_DB=transcripts scripts/pg-standalone.sh up
//! docker exec -u 0 akh-th-mcp-pg bash -c \
//!   "echo \"shared_preload_libraries = 'timescaledb'\" >> /var/lib/postgresql/data/pgdata/postgresql.conf"
//! docker restart akh-th-mcp-pg
//! AKH_TEST_TH_MCP_PG_URL='postgres://akh:akh@127.0.0.1:55434/transcripts?sslmode=disable' \
//! AKH_TRANSCRIPTS_MODEL_DIR=$HOME/.cache/akh-models/all-MiniLM-L6-v2 \
//! ORT_DYLIB_PATH=$HOME/.cache/onnxruntime/lib/libonnxruntime.so \
//!   cargo test -p akh-transcripts --test mcp_read_surface -- --ignored
//! ```
//!
//! **`subseq`/`disposition` stand-in**: the concurrent derivation-worker
//! increment adds `subseq int` and `disposition text` to `th_events` (and
//! makes `kind` nullable) on a branch not yet merged into this one — see the
//! increment-6 brief. [`add_pending_columns`] applies the same two-column
//! `ALTER TABLE` here so this test can run ahead of that merge; once it
//! lands, `run_migrations_locked` alone will already carry these columns and
//! this function becomes a harmless `IF NOT EXISTS` no-op.
//!
//! Uses `TranscriptsMcpServer` directly (the same type
//! `src/bin/akh-transcripts-mcp.rs` constructs) wired into a real
//! `axum::serve` listener on `127.0.0.1:0`, and talks to it over real HTTP
//! with `rmcp`'s streamable-HTTP **client** transport — the same approach
//! `crates/sokar/tests/mcp_contract.rs` uses for its own MCP server. This
//! exercises the identical server code the binary runs (tool router, axum
//! wiring, bearer middleware) without the fragility of hand-parsing the
//! streamable-HTTP SSE framing from a shell script.

use std::sync::{Arc, Mutex};

use akh_transcripts::auth::{self, SharedSecretAuth};
use akh_transcripts::entity::{chunk as chunk_entity, raw_record, session as session_entity};
use akh_transcripts::mcp::{Readiness, TranscriptsMcpServer, TranscriptsMcpState};
use akh_transcripts::pool::{connect, run_migrations_locked};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::{
    StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
};
use rmcp::{RoleClient, ServiceExt};
use sea_orm::{
    ActiveValue::Set, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait, Statement,
};
use serde_json::{Value, json};

const BEARER_TOKEN: &str = "test-bearer-token-for-mcp-read-surface";
const GITHUB_TOKEN: &str = "gho_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"; // 36 'A's after gho_

/// The `ALTER TABLE` a concurrent, not-yet-merged migration will eventually
/// carry. `IF NOT EXISTS`/idempotent so this is harmless once that migration
/// lands on this branch too.
async fn add_pending_columns(db: &DatabaseConnection) {
    db.execute_unprepared(
        "ALTER TABLE th_events \
         ADD COLUMN IF NOT EXISTS subseq integer, \
         ADD COLUMN IF NOT EXISTS disposition text NOT NULL DEFAULT 'event'",
    )
    .await
    .expect("add subseq/disposition columns");
    db.execute_unprepared("ALTER TABLE th_events ALTER COLUMN kind DROP NOT NULL")
        .await
        .expect("make kind nullable");
}

struct EventFixture {
    stream_id: &'static str,
    seq: i64,
    subseq: Option<i32>,
    kind: Option<&'static str>,
    subkind: Option<&'static str>,
    text: Option<String>,
    agent_id: Option<&'static str>,
    disposition: &'static str,
}

impl EventFixture {
    fn event(
        stream_id: &'static str,
        seq: i64,
        kind: &'static str,
        text: impl Into<String>,
    ) -> Self {
        EventFixture {
            stream_id,
            seq,
            subseq: None,
            kind: Some(kind),
            subkind: None,
            text: Some(text.into()),
            agent_id: None,
            disposition: "event",
        }
    }
}

async fn insert_event(db: &DatabaseConnection, session_id: &str, f: &EventFixture) {
    let sql = "INSERT INTO th_events \
        (derivation_version, source, session_id, stream_id, seq, subseq, observed_at, kind, \
         subkind, role, text, tool_name, tool_call_id, agent_id, parent_agent_id, degraded, \
         degraded_reason, secret_flags, repo, workspace, branch, derived_at, disposition) \
        VALUES (1, 'claude_code', $1, $2, $3, $4, now(), $5, $6, NULL, $7, NULL, NULL, $8, NULL, \
         false, NULL, NULL, 'akh-medu', '/repo', 'main', now(), $9)";
    db.execute(Statement::from_sql_and_values(
        DbBackend::Postgres,
        sql,
        [
            session_id.into(),
            f.stream_id.into(),
            f.seq.into(),
            f.subseq.into(),
            f.kind.into(),
            f.subkind.into(),
            f.text.clone().into(),
            f.agent_id.into(),
            f.disposition.into(),
        ],
    ))
    .await
    .unwrap_or_else(|e| panic!("insert event {}/{}: {e}", f.stream_id, f.seq));
}

async fn insert_raw_record(
    db: &DatabaseConnection,
    session_id: &str,
    stream_id: &str,
    seq: i64,
    raw: &str,
) {
    let now = chrono::Utc::now();
    let model = raw_record::ActiveModel {
        observed_at: Set(now),
        source: Set("claude_code".to_string()),
        session_id: Set(session_id.to_string()),
        stream_id: Set(stream_id.to_string()),
        seq: Set(seq),
        stream_kind: Set(if stream_id == "main" {
            "main"
        } else {
            "subagent"
        }
        .to_string()),
        machine: Set("mcp-read-surface-test".to_string()),
        raw: Set(raw.to_string()),
        payload: Set(None),
        record_type: Set(None),
        schema_marker: Set(None),
        origin_path: Set("/fixtures/mcp-read-surface.jsonl".to_string()),
        observed_at_synthetic: Set(false),
        harvested_at: Set(now),
    };
    raw_record::Entity::insert(model)
        .exec(db)
        .await
        .expect("insert raw record");
}

async fn seed(
    db: &DatabaseConnection,
    session_id: &str,
    embedder: Option<&mut akh_embed::SentenceEmbedder>,
) {
    // main: prompt -> spawn subagent -> reply (carries the secret) -> thought (marker
    // tier, excluded by default) -> telemetry (excluded by default).
    insert_event(
        db,
        session_id,
        &EventFixture::event("main", 1, "prompt", "please look into the redaction bug"),
    )
    .await;
    insert_event(
        db,
        session_id,
        &EventFixture {
            stream_id: "main",
            seq: 2,
            subseq: None,
            kind: Some("lifecycle"),
            subkind: Some("subagent_started"),
            text: Some("agent-a".to_string()),
            agent_id: Some("agent-a"),
            disposition: "event",
        },
    )
    .await;
    insert_event(
        db,
        session_id,
        &EventFixture::event(
            "main",
            3,
            "reply",
            format!("done — by the way here is a leaked token {GITHUB_TOKEN} in the diff"),
        ),
    )
    .await;
    insert_event(
        db,
        session_id,
        &EventFixture::event(
            "main",
            4,
            "thought",
            "internal reasoning, should be hidden by default",
        ),
    )
    .await;
    insert_event(
        db,
        session_id,
        &EventFixture::event("main", 5, "telemetry", "token_usage_record total=482"),
    )
    .await;

    // subagent: prompt -> reply.
    insert_event(
        db,
        session_id,
        &EventFixture::event("agent-a", 1, "prompt", "investigate the leak"),
    )
    .await;
    insert_event(
        db,
        session_id,
        &EventFixture::event("agent-a", 2, "reply", "found it, see main seq 3"),
    )
    .await;

    // A withheld verbatim record: a PEM private key with no bounded end,
    // per `redact.rs`'s `SecretAction::Withhold`.
    insert_event(
        db,
        session_id,
        &EventFixture::event(
            "main",
            6,
            "tool_result",
            "-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END-----",
        ),
    )
    .await;

    // A `disposition != 'event'` row — must never surface from any tool.
    insert_event(
        db,
        session_id,
        &EventFixture {
            stream_id: "main",
            seq: 7,
            subseq: None,
            kind: None,
            subkind: None,
            text: Some(format!(
                "noise row that also carries {GITHUB_TOKEN} and must never appear"
            )),
            agent_id: None,
            disposition: "noise",
        },
    )
    .await;

    // Raw records for get_lines(verbatim=true): seq 3 (mask), seq 6 (withhold).
    insert_raw_record(
        db,
        session_id,
        "main",
        3,
        &format!(
            r#"{{"type":"assistant","text":"done — leaked token {GITHUB_TOKEN} in the diff"}}"#
        ),
    )
    .await;
    insert_raw_record(
        db,
        session_id,
        "main",
        6,
        "{\"type\":\"tool_result\",\"text\":\"-----BEGIN OPENSSH PRIVATE KEY-----\\nabc123\\n-----END-----\"}",
    )
    .await;

    // th_sessions summary row.
    let now = chrono::Utc::now();
    let session_model = session_entity::ActiveModel {
        source: Set("claude_code".to_string()),
        session_id: Set(session_id.to_string()),
        title: Set(Some("redaction bug session".to_string())),
        repo: Set(Some("akh-medu".to_string())),
        workspace: Set(Some("/repo".to_string())),
        branch: Set(Some("main".to_string())),
        first_seen: Set(now - chrono::Duration::hours(1)),
        last_seen: Set(now),
        stream_count: Set(2),
        record_count: Set(7),
        derivation_version: Set(1),
    };
    session_entity::Entity::insert(session_model)
        .exec(db)
        .await
        .expect("insert session summary");

    // th_chunks: one clean chunk about the topic, embedded for real if the
    // model is available, plain zero-vector fallback otherwise (semantic
    // ranking then degrades to "some result", which is all this test needs
    // beyond exercising the tool at all).
    let chunk_text = "we tracked down and fixed the redaction bug in the transcript pipeline";
    let embedding: Vec<f32> = match embedder {
        Some(e) => e.embed_text(chunk_text).expect("embed chunk"),
        None => vec![0.0_f32; akh_embed::EMBEDDING_DIM],
    };
    let embedding_str = chunk_entity::vec_to_pgvector(&embedding);
    db.execute(Statement::from_sql_and_values(
        DbBackend::Postgres,
        "INSERT INTO th_chunks (derivation_version, source, session_id, stream_id, seq_from, \
         seq_to, text, embedding, embedded_at, created_at) \
         VALUES (1, 'claude_code', $1, 'main', 1, 3, $2, $3::vector, now(), now())",
        [session_id.into(), chunk_text.into(), embedding_str.into()],
    ))
    .await
    .expect("insert chunk");
}

type Peer = RunningService<RoleClient, ()>;

struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(db: Arc<DatabaseConnection>, embedder: Option<akh_embed::SharedEmbedder>) -> Server {
    let state = Arc::new(TranscriptsMcpState {
        db,
        embedder,
        readiness: Arc::new(Readiness::new()),
    });
    let mcp_server = TranscriptsMcpServer::new(state);

    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    };
    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_config = StreamableHttpServerConfig::default();
    let mcp_service =
        StreamableHttpService::new(move || Ok(mcp_server.clone()), session_manager, mcp_config);

    let bearer = SharedSecretAuth::from_inline(BEARER_TOKEN).unwrap();
    let app = axum::Router::new().nest_service("/mcp", mcp_service).layer(
        axum::middleware::from_fn_with_state(bearer, auth::require_bearer),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/mcp", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { url, task }
}

async fn connect_client(server: &Server, token: &str) -> Peer {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(server.url.clone()).auth_header(token),
    );
    tokio::time::timeout(std::time::Duration::from_secs(10), ().serve(transport))
        .await
        .expect("connect timeout")
        .expect("connect")
}

async fn call(peer: &Peer, name: &str, args: Value) -> String {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        peer.call_tool(CallToolRequestParams {
            name: name.to_owned().into(),
            arguments: args.as_object().cloned(),
            meta: None,
            task: None,
        }),
    )
    .await
    .expect("call timeout")
    .unwrap_or_else(|e| panic!("tool call {name} failed: {e}"));
    assert_ne!(
        result.is_error,
        Some(true),
        "{name} returned an error: {result:?}"
    );
    result.content[0]
        .as_text()
        .expect("text content")
        .text
        .clone()
}

#[tokio::test]
#[ignore = "requires AKH_TEST_TH_MCP_PG_URL (cnpg-timescale image: timescaledb + vector)"]
async fn all_five_tools_roundtrip_over_http_and_never_leak_the_secret() {
    let Ok(url) = std::env::var("AKH_TEST_TH_MCP_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_TH_MCP_PG_URL");
        return;
    };

    run_migrations_locked(&url).await.expect("migrate");
    let db = connect(&url).await.expect("connect");
    add_pending_columns(&db).await;

    let session_id = format!("mcp-read-surface-{}", uuid::Uuid::new_v4());

    let model_dir = std::env::var_os("AKH_TRANSCRIPTS_MODEL_DIR").map(std::path::PathBuf::from);
    let mut loaded_embedder = model_dir
        .as_deref()
        .and_then(|dir| akh_embed::SentenceEmbedder::load(dir).ok());
    seed(&db, &session_id, loaded_embedder.as_mut()).await;

    let db = Arc::new(db);
    let shared_embedder: Option<akh_embed::SharedEmbedder> =
        loaded_embedder.map(|e| Arc::new(Mutex::new(e)));
    let server = serve(Arc::clone(&db), shared_embedder).await;

    // --- auth smoke check over real HTTP (unit tests in auth.rs cover the
    // exhaustive matrix; this just confirms the binary's actual wiring). ---
    let http = reqwest::Client::new();
    let unauthed = http.post(&server.url).body("{}").send().await.unwrap();
    assert_eq!(unauthed.status(), reqwest::StatusCode::UNAUTHORIZED);
    let wrong = http
        .post(&server.url)
        .header("Authorization", "Bearer wrong-token")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), reqwest::StatusCode::FORBIDDEN);

    let peer = connect_client(&server, BEARER_TOKEN).await;

    // ── tools/list ──────────────────────────────────────────────────────
    let tools = peer.list_all_tools().await.expect("tools/list");
    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "get_lines",
            "get_session",
            "grep_transcripts",
            "list_sessions",
            "search_transcripts",
        ]
    );
    let tools_list_json = serde_json::to_string_pretty(&tools).unwrap();
    println!("=== tools/list ===\n{tools_list_json}");

    // ── list_sessions ───────────────────────────────────────────────────
    let list_json = call(
        &peer,
        "list_sessions",
        json!({"source": "claude_code", "repo": "akh-medu"}),
    )
    .await;
    assert!(
        list_json.contains(&session_id),
        "list_sessions: {list_json}"
    );
    assert!(!list_json.contains(GITHUB_TOKEN));

    // ── get_session (spliced, default) ─────────────────────────────────
    let session_json = call(
        &peer,
        "get_session",
        json!({"source": "claude_code", "session_id": session_id}),
    )
    .await;
    let session_value: Value = serde_json::from_str(&session_json).unwrap();
    let events = session_value["events"].as_array().unwrap();
    let stream_order: Vec<&str> = events
        .iter()
        .map(|e| e["stream_id"].as_str().unwrap())
        .collect();
    // main(1) main(2) agent-a(1) agent-a(2) main(3) main(6) — thought(4) and
    // telemetry(5) excluded by default, noise(7) never normalized at all.
    assert_eq!(
        stream_order,
        vec!["main", "main", "agent-a", "agent-a", "main", "main"],
        "splice order: {session_json}"
    );
    assert!(
        !session_json.contains(GITHUB_TOKEN),
        "get_session leaked the token: {session_json}"
    );
    assert!(
        session_json.contains("[REDACTED:github_pat]"),
        "get_session should show the redaction marker: {session_json}"
    );
    assert!(
        !events.iter().any(|e| e["kind"] == "thought"),
        "thought must be excluded by default"
    );
    assert!(
        !events.iter().any(|e| e["kind"] == "telemetry"),
        "telemetry must be excluded by default"
    );

    // include_thoughts=true surfaces the marker line, still no real content
    // (there is none in this fixture, but exercises the flag).
    let with_thoughts = call(
        &peer,
        "get_session",
        json!({"source": "claude_code", "session_id": session_id, "include_thoughts": true, "include_all": true}),
    )
    .await;
    let with_thoughts_value: Value = serde_json::from_str(&with_thoughts).unwrap();
    let kinds: Vec<&str> = with_thoughts_value["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"thought"));
    assert!(kinds.contains(&"telemetry"));

    // stream order mode.
    let stream_mode = call(
        &peer,
        "get_session",
        json!({"source": "claude_code", "session_id": session_id, "order": "stream"}),
    )
    .await;
    let stream_mode_value: Value = serde_json::from_str(&stream_mode).unwrap();
    let stream_mode_ids: Vec<&str> = stream_mode_value["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["stream_id"].as_str().unwrap())
        .collect();
    // "agent-a" < "main" lexicographically.
    assert_eq!(stream_mode_ids[0], "agent-a");

    // budget truncation + continuation.
    let tiny_budget = call(
        &peer,
        "get_session",
        json!({"source": "claude_code", "session_id": session_id, "budget_bytes": 10}),
    )
    .await;
    let tiny_value: Value = serde_json::from_str(&tiny_budget).unwrap();
    assert_eq!(tiny_value["truncated"], true, "{tiny_budget}");
    assert!(tiny_value["continue_from"].is_string(), "{tiny_budget}");
    assert!(!tiny_budget.contains(GITHUB_TOKEN));

    // ── get_lines (rendered) ────────────────────────────────────────────
    let lines_rendered = call(
        &peer,
        "get_lines",
        json!({"source": "claude_code", "session_id": session_id, "stream_id": "main", "from_seq": 3, "to_seq": 3}),
    )
    .await;
    assert!(!lines_rendered.contains(GITHUB_TOKEN), "{lines_rendered}");
    assert!(lines_rendered.contains("[REDACTED:github_pat]"));

    // ── get_lines (verbatim, masked) ────────────────────────────────────
    let lines_verbatim = call(
        &peer,
        "get_lines",
        json!({"source": "claude_code", "session_id": session_id, "stream_id": "main", "from_seq": 3, "to_seq": 3, "verbatim": true}),
    )
    .await;
    assert!(!lines_verbatim.contains(GITHUB_TOKEN), "{lines_verbatim}");
    assert!(lines_verbatim.contains("[REDACTED:github_pat]"));

    // ── get_lines (verbatim, withheld private key) ──────────────────────
    let lines_withheld = call(
        &peer,
        "get_lines",
        json!({"source": "claude_code", "session_id": session_id, "stream_id": "main", "from_seq": 6, "to_seq": 6, "verbatim": true}),
    )
    .await;
    let withheld_value: Value = serde_json::from_str(&lines_withheld).unwrap();
    assert_eq!(
        withheld_value["lines"][0]["withheld"], "private_key",
        "{lines_withheld}"
    );
    assert!(!lines_withheld.contains("BEGIN OPENSSH"));

    // ── grep_transcripts ────────────────────────────────────────────────
    let grep_json = call(
        &peer,
        "grep_transcripts",
        json!({"pattern": "leaked token", "source": "claude_code", "session_id": session_id}),
    )
    .await;
    assert!(!grep_json.contains(GITHUB_TOKEN), "{grep_json}");
    assert!(grep_json.contains("[REDACTED:github_pat]"));
    let grep_value: Value = serde_json::from_str(&grep_json).unwrap();
    assert_eq!(grep_value["hits"].as_array().unwrap().len(), 1);
    assert_eq!(grep_value["hits"][0]["seq"], 3);

    // The disposition != 'event' noise row must never surface, even though
    // it too contains the token and would otherwise match every pattern
    // below.
    let grep_noise = call(
        &peer,
        "grep_transcripts",
        json!({"pattern": "noise row", "source": "claude_code", "session_id": session_id}),
    )
    .await;
    let grep_noise_value: Value = serde_json::from_str(&grep_noise).unwrap();
    assert_eq!(
        grep_noise_value["hits"].as_array().unwrap().len(),
        0,
        "{grep_noise}"
    );

    // ── search_transcripts ──────────────────────────────────────────────
    let search_json = call(
        &peer,
        "search_transcripts",
        json!({"query": "fixing a bug about redacting secrets", "source": "claude_code", "session_id": session_id, "min_similarity": -1.0}),
    )
    .await;
    assert!(!search_json.contains(GITHUB_TOKEN), "{search_json}");
    let search_value: Value = serde_json::from_str(&search_json).unwrap();
    assert!(
        !search_value["hits"].as_array().unwrap().is_empty(),
        "expected at least one chunk hit: {search_json}"
    );

    println!("ALL FIVE TOOLS EXERCISED — no gho_ token observed in any response");
}

#[tokio::test]
#[ignore = "requires AKH_TEST_TH_MCP_PG_URL"]
async fn search_transcripts_degrades_clearly_without_a_model() {
    let Ok(url) = std::env::var("AKH_TEST_TH_MCP_PG_URL") else {
        eprintln!("skipping: set AKH_TEST_TH_MCP_PG_URL");
        return;
    };
    run_migrations_locked(&url).await.expect("migrate");
    let db = connect(&url).await.expect("connect");
    add_pending_columns(&db).await;
    let db = Arc::new(db);

    let server = serve(Arc::clone(&db), None).await;
    let peer = connect_client(&server, BEARER_TOKEN).await;

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        peer.call_tool(CallToolRequestParams {
            name: "search_transcripts".to_owned().into(),
            arguments: json!({"query": "anything"}).as_object().cloned(),
            meta: None,
            task: None,
        }),
    )
    .await
    .expect("timeout");

    // Either a JSON-RPC-level error or a tool-level error content is
    // acceptable — what matters is that it is a *clear*, well-formed
    // refusal rather than a panic or a hang, and that the other tools are
    // unaffected on the very same server instance.
    if let Ok(call_result) = result {
        assert_eq!(call_result.is_error, Some(true));
    }

    let list_json = call(&peer, "list_sessions", json!({})).await;
    let _: Value = serde_json::from_str(&list_json).expect("list_sessions still works");
}
