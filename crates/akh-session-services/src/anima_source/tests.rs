//! Anima source tests against a stub Anima MCP server.
use super::*;
use crate::{config::Config, ingress::Delivery, store::Store};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use rmcp::model::ProtocolVersion;
use std::sync::{Arc, Mutex};

const ANIMA_TOKEN: &str = "anima-mcp-token-0123456789-abcdefgh";

/// What the stub Anima answers.
#[derive(Default)]
struct Anima {
    projects: Vec<Value>,
    work_items: BTreeMap<String, Vec<Value>>,
    plans: BTreeMap<String, Vec<Value>>,
    /// Work-item id -> `get_work_item_detail` body.
    details: BTreeMap<String, Value>,
    forensics: BTreeMap<String, Value>,
    sessions: BTreeMap<String, Value>,
    /// Always answer `list_work_items` with the same cursor.
    loop_cursor: bool,
    calls: Vec<(String, Value)>,
}

type Shared = Arc<Mutex<Anima>>;

fn reply(id: &Value, payload: Value) -> Response {
    Json(json!({"jsonrpc":"2.0","id":id,"result":{
        "content":[{"type":"text","text":payload.to_string()}],
        "structuredContent":payload,"isError":false}}))
    .into_response()
}

fn page(rows: &[Value], args: &Value) -> Value {
    let size = args["page_size"].as_u64().unwrap_or(100) as usize;
    let offset: usize = args["page_token"]
        .as_str()
        .and_then(|t| t.strip_prefix("o:"))
        .and_then(|t| t.parse().ok())
        .unwrap_or(0);
    let end = (offset + size).min(rows.len());
    let next = (end < rows.len()).then(|| format!("o:{end}"));
    json!({"rows": rows[offset..end], "next_page_token": next, "total": rows.len()})
}

async fn mcp(State(anima): State<Shared>, Json(body): Json<Value>) -> Response {
    let id = body["id"].clone();
    match body["method"].as_str() {
        Some("initialize") => Json(json!({"jsonrpc":"2.0","id":id,"result":{
            "protocolVersion":ProtocolVersion::LATEST,"capabilities":{"tools":{}},
            "serverInfo":{"name":"anima-stub","version":"1"}}}))
        .into_response(),
        Some("tools/call") => {
            let tool = body["params"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string();
            let args = body["params"]["arguments"].clone();
            let mut anima = anima.lock().unwrap();
            anima.calls.push((tool.clone(), args.clone()));
            let key = |field: &str| args[field].as_str().unwrap_or_default().to_string();
            let payload = match tool.as_str() {
                "list_projects" => json!({"projects": anima.projects}),
                "list_work_items" => {
                    let rows = anima
                        .work_items
                        .get(&key("project_id"))
                        .cloned()
                        .unwrap_or_default();
                    let mut p = page(&rows, &args);
                    if anima.loop_cursor {
                        p["next_page_token"] = json!("same");
                    }
                    json!({"work_items": p["rows"], "next_page_token": p["next_page_token"],
                        "total": p["total"]})
                }
                "list_plans" => {
                    json!({"plans": anima.plans.get(&key("project_id")).cloned().unwrap_or_default(),
                        "next_page_token": null})
                }
                "get_work_item_detail" => {
                    anima.details.get(&key("id")).cloned().unwrap_or(json!({}))
                }
                "get_work_item_forensics" => anima
                    .forensics
                    .get(&key("work_item_id"))
                    .cloned()
                    .unwrap_or(json!({"events": []})),
                "list_sessions" => anima
                    .sessions
                    .get(&key("work_item_id"))
                    .cloned()
                    .unwrap_or(json!({"sessions": []})),
                _ => {
                    return Json(json!({"jsonrpc":"2.0","id":id,
                        "error":{"code":-32601,"message":"unknown tool"}}))
                    .into_response();
                }
            };
            reply(&id, payload)
        }
        _ => StatusCode::ACCEPTED.into_response(),
    }
}

/// The rendered Roman role filters, read from the fixture both evaluators
/// share (so these tests exercise exactly what `full_akh.py` renders).
fn role_filter(role: &str) -> Value {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../deploy/roman-mini/fixtures/route_filter_cases.json"
    ))
    .unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"].as_str().unwrap().ends_with(&format!("-> {role}")))
        .unwrap()["filter"]
        .clone()
}

struct Fixture {
    _dir: tempfile::TempDir,
    dir: std::path::PathBuf,
    anima: Shared,
    events: Events,
    source: AnimaSource,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn work_item(id: &str, phase: &str, title: &str) -> Value {
    json!({"id":id,"code":format!("WI-{id}"),"title":title,"status":"open",
        "current_phase":phase,"priority":2,"project_id":"p1","plan_id":null})
}

fn detail(id: &str, phase: &str, reason: Option<&str>, title: &str, messages: Value) -> Value {
    let mut wi = work_item(id, phase, title);
    wi["parked_reason"] = json!(reason);
    wi["phase_entered_at"] = json!("2026-09-26T00:00:00Z");
    json!({"work_item": wi, "thread_messages": messages})
}

async fn fixture(patch: Value) -> Fixture {
    let anima = Arc::new(Mutex::new(Anima::default()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route(
            "/mcp",
            post(mcp).delete(|| async { StatusCode::NO_CONTENT }),
        )
        .with_state(anima.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_path_buf();
    let operator = path.join("operator.token");
    std::fs::write(&operator, "operator-token-0123456789-abcdef").unwrap();
    let token = path.join("anima.token");
    std::fs::write(&token, ANIMA_TOKEN).unwrap();
    let core = path.join("core.token");
    std::fs::write(&core, "core-events-token-0123456789-abcdef").unwrap();
    let akh = |role: &str| json!({"kind":"akh","endpoint":"core","akh":role});
    let mut source = json!({"endpoint":"anima","source":"roman-anima","interval_secs":60,
        "project_filter":{"eq":{"pointer":"/status","value":"active"}},
        "fields":["id","code","title","status","current_phase","priority","project_id","plan_id",
            "parked","parked_reason","hold_reason"],
        "plan_fields":["id","title","status","content"],
        "hydrate_parked":{
            "when":{"eq":{"pointer":"/current_phase","value":"parked"}},
            "park_event":{"any":[{"eq":{"pointer":"/type","value":"park"}},
                {"eq":{"pointer":"/to_phase","value":"parked"}}]},
            "park_event_order":"/occurred_at",
            "episode_fields":["parked_reason","phase_entered_at"],
            "live_session":{"in":{"pointer":"/status","values":["provisioning","running","blocked"]}},
            "thread_message":{"eq":{"pointer":"/sender_type","value":"human"}},
            "thread_message_order":"/created_at"},
        "max_emissions_per_episode":2});
    source
        .as_object_mut()
        .unwrap()
        .extend(patch.as_object().unwrap().clone());
    let config: Config = serde_json::from_value(json!({
        "service_id":"events","listen":"127.0.0.1:0","data_dir":path,
        "principals":[{"id":"operator","token_file":operator,"roles":["admin"]}],
        "endpoints":{"anima":{"url":format!("{url}/mcp"),"token_file":token},
            "core":{"url":"http://127.0.0.1:1","token_file":core}},
        "routes":[
            {"id":"janus-anima.work_item.changed","source":"roman-anima",
                "event_type":"anima.work_item.changed","target":akh("janus"),"when":role_filter("janus")},
            {"id":"minerva-anima.work_item.changed","source":"roman-anima",
                "event_type":"anima.work_item.changed","target":akh("minerva"),"when":role_filter("minerva")},
            {"id":"vesta-anima.work_item.changed","source":"roman-anima",
                "event_type":"anima.work_item.changed","target":akh("vesta"),"when":role_filter("vesta")},
            {"id":"minerva-anima.plan.changed","source":"roman-anima",
                "event_type":"anima.plan.changed","target":akh("minerva")}],
        "anima":[source]
    }))
    .unwrap();
    config.validate().unwrap();
    let source = config.anima[0].clone();
    let events = Events::new(
        Arc::new(config),
        Store::open(&path.join("events.redb")).unwrap(),
    )
    .unwrap();
    {
        let mut a = anima.lock().unwrap();
        a.projects = vec![
            json!({"id":"p1","name":"Akh-Medu","status":"active"}),
            json!({"id":"p2","name":"Archive","status":"archived"}),
        ];
        a.work_items.insert(
            "p1".into(),
            vec![
                work_item("w1", "working", "one"),
                work_item("w2", "parked", "two"),
                work_item("w3", "parked", "three"),
                work_item("w4", "parked", "four"),
            ],
        );
        a.work_items
            .insert("p2".into(), vec![work_item("x1", "parked", "hidden")]);
        a.plans.insert(
            "p1".into(),
            vec![json!({"id":"plan1","title":"Plan","status":"active","content":"x".repeat(400)})],
        );
        a.details.insert(
            "w2".into(),
            detail("w2", "parked", Some("too-big"), "two", json!([])),
        );
        a.details.insert(
            "w3".into(),
            detail("w3", "parked", Some("ci-infra-failing"), "three", json!([])),
        );
        a.details.insert(
            "w4".into(),
            detail("w4", "parked", Some("operator-hold"), "four", json!([])),
        );
        a.forensics.insert(
            "w2".into(),
            json!({"events":[{"type":"park","occurred_at":"2026-09-26T00:00:00Z","reason":"too_big"}]}),
        );
    }
    Fixture {
        _dir: dir,
        dir: path,
        anima,
        events,
        source,
        server,
    }
}

impl Fixture {
    async fn poll(&self) -> PollReport {
        poll(&self.events, &self.source).await.unwrap()
    }
    fn state(&self, entity: &str) -> EntityState {
        self.events
            .store
            .get(&entity_key("roman-anima", entity))
            .unwrap()
            .unwrap()
    }
    fn deliveries(&self, event_id: &str) -> Vec<Delivery> {
        let receipt: crate::ingress::IngressReceipt = self
            .events
            .store
            .get(&key("ingress", &["roman-anima", event_id]))
            .unwrap()
            .unwrap();
        receipt
            .deliveries
            .iter()
            .map(|id| {
                self.events
                    .store
                    .get(&key("delivery", &[id]))
                    .unwrap()
                    .unwrap()
            })
            .collect()
    }
    fn routed(&self, event_id: &str) -> Vec<String> {
        self.deliveries(event_id)
            .into_iter()
            .map(|d| match d.target {
                Some(crate::config::RouteTarget::Akh { akh, .. }) => akh,
                other => panic!("unexpected target {other:?}"),
            })
            .collect()
    }
    fn set_item(&self, item: Value) {
        let mut anima = self.anima.lock().unwrap();
        let rows = anima.work_items.get_mut("p1").unwrap();
        let slot = rows.iter_mut().find(|r| r["id"] == item["id"]).unwrap();
        *slot = item;
    }
    fn change(&self, event_id: &str) -> AnimaChange {
        let deliveries = self.deliveries(event_id);
        let data = match deliveries.first() {
            Some(d) => d.request.event.data.clone(),
            None => {
                let receipt: crate::ingress::IngressReceipt = self
                    .events
                    .store
                    .get(&key("ingress", &["roman-anima", event_id]))
                    .unwrap()
                    .unwrap();
                receipt.event.data
            }
        };
        serde_json::from_value(data).unwrap()
    }
}

#[tokio::test]
async fn baseline_is_silent_then_changes_route_through_the_filters() {
    let f = fixture(json!({})).await;
    let report = f.poll().await;
    assert_eq!(
        report.projects,
        ["p1"],
        "archived projects are filtered out"
    );
    assert_eq!(report.baselined, ["p1"]);
    assert!(report.emitted.is_empty() && report.shadowed.is_empty());
    assert_eq!(report.hydrated, 3, "only the parked items are hydrated");
    let parked = f.state("work_item:w2");
    assert_eq!(parked.projection["parked_reason"], "too-big");
    assert_eq!(parked.projection["live_sessions"], 0);
    assert_eq!(
        parked.projection["park_episode"].as_str().unwrap().len(),
        32
    );
    assert!(
        parked.projection.get("latest_thread_message_id").is_none(),
        "no human message yet"
    );
    assert!(
        f.state("work_item:w1")
            .projection
            .get("live_sessions")
            .is_none()
    );
    let plan = f.state("plan:plan1");
    assert!(
        plan.projection.get("content").is_none(),
        "a long field is only in the digest"
    );
    assert_eq!(plan.projection["title"], "Plan");
    // Every call carried the Anima token; detail reads were bounded.
    assert!(
        f.anima
            .lock()
            .unwrap()
            .calls
            .iter()
            .any(|(tool, args)| tool == "get_work_item_forensics" && args["max_events"] == 500)
    );

    // Unchanged listing, fresh hydration: nothing is read again or emitted.
    let calls = f.anima.lock().unwrap().calls.len();
    let report = f.poll().await;
    assert!(report.emitted.is_empty());
    assert_eq!(report.hydrated, 0);
    assert_eq!(
        f.anima.lock().unwrap().calls.len(),
        calls + 2,
        "projects cached; work items and plans listed; no hydration"
    );

    // w1 parks with a generic reason: Janus. A held item changing wakes
    // nobody. A new plan wakes Minerva.
    f.set_item(work_item("w1", "parked", "one"));
    f.anima.lock().unwrap().details.insert(
        "w1".into(),
        detail("w1", "parked", Some("no-progress"), "one", json!([])),
    );
    f.set_item(work_item("w4", "parked", "four, renamed"));
    f.anima.lock().unwrap().details.insert(
        "w4".into(),
        detail(
            "w4",
            "parked",
            Some("operator-hold"),
            "four, renamed",
            json!([]),
        ),
    );
    f.anima
        .lock()
        .unwrap()
        .plans
        .get_mut("p1")
        .unwrap()
        .push(json!({"id":"plan2","title":"New","status":"draft"}));
    let report = f.poll().await;
    assert_eq!(report.emitted.len(), 3, "{report:?}");
    let by_entity = |entity: &str| {
        report
            .emitted
            .iter()
            .find(|id| id.starts_with(&format!("{entity}@")))
            .unwrap()
            .clone()
    };
    let w1 = by_entity("work_item:w1");
    assert_eq!(w1.len(), "work_item:w1@".len() + 64);
    assert_eq!(f.routed(&w1), ["janus"]);
    let change = f.change(&w1);
    assert_eq!(change.kind, AnimaEntityKind::WorkItem);
    assert_eq!(change.revision, 1);
    assert_eq!(change.previous.unwrap()["current_phase"], "working");
    assert_eq!(change.current["parked_reason"], "no-progress");
    assert_eq!((change.emission, change.max_emissions), (Some(1), Some(2)));
    let w4 = by_entity("work_item:w4");
    assert!(f.routed(&w4).is_empty(), "a held item wakes nobody");
    let plan2 = by_entity("plan:plan2");
    assert_eq!(f.routed(&plan2), ["minerva"]);
    assert!(f.change(&plan2).previous.is_none(), "new after baseline");

    // Status: served over HTTP to the operator.
    let status = status(&f.events, "roman-anima").unwrap().unwrap();
    assert!(status.ok);
    assert_eq!(status.counters.emitted, 3);
    assert!(status.coverage["p1"].complete && status.coverage["p1"].baselined);
}

#[tokio::test]
async fn infrastructure_parks_wake_vesta_even_with_a_live_session() {
    let f = fixture(json!({})).await;
    f.poll().await;
    f.anima.lock().unwrap().sessions.insert(
        "w3".into(),
        json!({"sessions":[{"id":"s1","status":"running"}]}),
    );
    f.set_item(work_item("w3", "parked", "three, retitled"));
    f.anima.lock().unwrap().details.insert(
        "w3".into(),
        detail(
            "w3",
            "parked",
            Some("ci-infra-failing"),
            "three, retitled",
            json!([]),
        ),
    );
    let report = f.poll().await;
    assert_eq!(report.emitted.len(), 1);
    assert_eq!(f.routed(&report.emitted[0]), ["vesta"]);
    assert_eq!(f.change(&report.emitted[0]).current["live_sessions"], 1);
    // The same live session on a too-big item: Minerva waits for the owner.
    f.anima.lock().unwrap().sessions.insert(
        "w2".into(),
        json!({"sessions":[{"id":"s2","status":"blocked"}]}),
    );
    f.set_item(work_item("w2", "parked", "two, retitled"));
    f.anima.lock().unwrap().details.insert(
        "w2".into(),
        detail("w2", "parked", Some("too-big"), "two, retitled", json!([])),
    );
    let report = f.poll().await;
    assert!(f.routed(&report.emitted[0]).is_empty());
}

#[tokio::test]
async fn emissions_are_capped_per_park_episode_and_listed_for_the_watchdog() {
    let f = fixture(json!({})).await;
    f.poll().await;
    let human = |id: &str, at: &str| json!({"id":id,"sender_type":"human","created_at":at});
    let akh = |id: &str, at: &str| json!({"id":id,"sender_type":"akh","created_at":at});
    let mut emitted = vec![];
    for (step, messages) in [
        json!([human("m1", "2026-09-26T01:00:00Z")]),
        // Newest first, as Anima returns them; the akh's own post is ignored.
        json!([
            akh("m3", "2026-09-26T03:00:00Z"),
            human("m2", "2026-09-26T02:00:00Z"),
            human("m1", "2026-09-26T01:00:00Z")
        ]),
        json!([human("m4", "2026-09-26T04:00:00Z")]),
    ]
    .into_iter()
    .enumerate()
    {
        let title = format!("two v{step}");
        f.set_item(work_item("w2", "parked", &title));
        f.anima.lock().unwrap().details.insert(
            "w2".into(),
            detail("w2", "parked", Some("too-big"), &title, messages),
        );
        emitted.push(f.poll().await);
    }
    assert_eq!(emitted[0].emitted.len(), 1);
    assert_eq!(
        f.change(&emitted[1].emitted[0]).current["latest_thread_message_id"],
        "m2"
    );
    assert_eq!(f.routed(&emitted[1].emitted[0]), ["minerva"]);
    assert_eq!(f.change(&emitted[1].emitted[0]).emission, Some(2));
    assert!(
        emitted[2].emitted.is_empty(),
        "the third emission is suppressed"
    );
    assert_eq!(emitted[2].suppressed, 1);
    let state = f.state("work_item:w2");
    assert!(state.attention);
    assert_eq!(state.episode.as_ref().unwrap().suppressed, 1);
    assert_eq!(
        state.projection["latest_thread_message_id"], "m4",
        "the projection still advances"
    );

    // The cap state is durable: a new service instance on the same store
    // keeps suppressing.
    let reopened = Events::new(f.events.config.clone(), f.events.store.clone()).unwrap();
    let status = status(&reopened, "roman-anima").unwrap().unwrap();
    assert_eq!(status.attention.len(), 1);
    assert_eq!(status.attention[0].entity, "work_item:w2");
    assert_eq!(status.counters.suppressed, 1);
    let watchdog: crate::config::WatchdogConfig = serde_json::from_value(json!({
        "endpoint":"anima","thread":"01a0ab49-9cc4-7012-8785-1eada26fbd46","summary":"s"}))
    .unwrap();
    let signal = crate::watchdog::signal(&reopened, &watchdog, now_ms()).unwrap();
    assert_eq!(
        signal.anima["roman-anima"].attention[0].entity,
        "work_item:w2"
    );
    assert!(signal.degraded());

    // A new park episode resets the cap and clears the attention entry.
    f.anima.lock().unwrap().forensics.insert(
        "w2".into(),
        json!({"events":[{"type":"park","occurred_at":"2026-09-26T00:00:00Z"},
            {"type":"phase_transition","to_phase":"parked","occurred_at":"2026-09-26T05:00:00Z"}]}),
    );
    f.set_item(work_item("w2", "parked", "two v3"));
    f.anima.lock().unwrap().details.insert(
        "w2".into(),
        detail("w2", "parked", Some("too-big"), "two v3", json!([])),
    );
    let report = poll(&reopened, &f.source).await.unwrap();
    assert_eq!(report.emitted.len(), 1);
    let change = f.change(&report.emitted[0]);
    assert_eq!(change.emission, Some(1));
    assert_ne!(change.park_episode, state.episode.map(|e| e.id));
    assert!(
        status_after(&reopened).attention.is_empty(),
        "the capped episode ended"
    );
}

fn status_after(events: &Events) -> AnimaSourceStatus {
    status(events, "roman-anima").unwrap().unwrap()
}

#[tokio::test]
async fn pagination_is_bounded_and_a_repeated_cursor_fails_the_pass() {
    let f = fixture(json!({"page_size":1,"max_pages":3})).await;
    let error = poll(&f.events, &f.source).await.unwrap_err();
    assert_eq!(error.code, "anima_pagination");
    assert!(error.message.contains("max_pages = 3"), "{}", error.message);
    let status = status_after(&f.events);
    assert!(!status.ok);
    assert_eq!(status.error.unwrap().code, "anima_pagination");
    // Nothing was recorded for the failed project.
    assert!(
        f.events
            .store
            .get::<EntityState>(&entity_key("roman-anima", "work_item:w1"))
            .unwrap()
            .is_none()
    );

    let f = fixture(json!({"page_size":1,"max_pages":32})).await;
    f.anima.lock().unwrap().loop_cursor = true;
    let error = poll(&f.events, &f.source).await.unwrap_err();
    assert_eq!(error.code, "anima_pagination");
    assert!(error.message.contains("repeated"), "{}", error.message);

    // Within the bound, four one-row pages are one complete listing.
    let f = fixture(json!({"page_size":1,"max_pages":4})).await;
    f.poll().await;
    let coverage = &status_after(&f.events).coverage["p1"];
    assert_eq!((coverage.pages, coverage.observed), (4, 4));
    assert!(coverage.complete);
}

#[tokio::test]
async fn hydration_beyond_the_tick_budget_defers_the_baseline() {
    let mut patch = json!({});
    patch["hydrate_parked"] = json!({
        "when":{"eq":{"pointer":"/current_phase","value":"parked"}},
        "park_event":{"eq":{"pointer":"/type","value":"park"}},
        "live_session":{"in":{"pointer":"/status","values":["running"]}},
        "max_per_tick":2});
    let f = fixture(patch).await;
    let report = f.poll().await;
    assert_eq!((report.hydrated, report.deferred), (2, 1));
    assert!(
        report.baselined.is_empty(),
        "one parked item is not recorded yet"
    );
    let report = f.poll().await;
    assert_eq!(report.baselined, ["p1"]);
    assert!(
        report.emitted.is_empty(),
        "late baseline rows are not new entities"
    );
}

#[tokio::test]
async fn shadow_mode_records_would_be_events_without_ingesting() {
    use axum::body::Body;
    use tower::ServiceExt;
    let f = fixture(json!({"shadow":true,"shadow_retention":2})).await;
    f.poll().await;
    for step in 0..4 {
        let title = format!("one v{step}");
        f.set_item(work_item("w1", "parked", &title));
        f.anima.lock().unwrap().details.insert(
            "w1".into(),
            detail("w1", "parked", Some("no-progress"), &title, json!([])),
        );
        let report = f.poll().await;
        assert!(report.emitted.is_empty());
        if step < 2 {
            assert_eq!(report.shadowed.len(), 1, "step {step}");
        } else {
            // The cap applies in shadow exactly as in live mode.
            assert_eq!(report.suppressed, 1, "step {step}");
        }
    }
    let deliveries: Vec<Delivery> = f.events.store.scan("delivery/").unwrap();
    assert!(deliveries.is_empty(), "shadow mode never ingests");
    let records = shadow_records(&f.events, "roman-anima", None, 100).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].routes[0].akh.as_deref(), Some("janus"));
    assert_eq!(records[0].routes[0].route, "janus-anima.work_item.changed");
    assert_eq!(records[1].routes[0].akh.as_deref(), Some("janus"));
    assert_eq!(
        status_after(&f.events).counters.shadowed,
        2,
        "counted as shadowed, not emitted"
    );

    // Served to the operator; filtered by time; refused to other callers.
    let mut config = (*f.events.config).clone();
    let reader = f.dir.join("reader.token");
    std::fs::write(&reader, "monitor-reader-token-0123456789").unwrap();
    config.principals.push(
        serde_json::from_value(json!({"id":"reader","token_file":reader,"roles":["operator"]}))
            .unwrap(),
    );
    let runtime = f.dir.join("runtime.token");
    std::fs::write(&runtime, "runtime-token-0123456789-abcdef").unwrap();
    config.principals.push(
        serde_json::from_value(json!({"id":"runtime","token_file":runtime,"roles":["runtime"]}))
            .unwrap(),
    );
    let router =
        crate::server::event_router(Events::new(Arc::new(config), f.events.store.clone()).unwrap());
    let get = |path: String, token: &str| {
        axum::http::Request::get(path)
            .header("authorization", format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    };
    let response = router
        .clone()
        .oneshot(get(
            "/v1/anima/roman-anima/shadow".into(),
            "monitor-reader-token-0123456789",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let served: Vec<ShadowRecord> = serde_json::from_slice(&body).unwrap();
    assert_eq!(served, records);
    let since = records[0].recorded_at_ms;
    let response = router
        .clone()
        .oneshot(get(
            format!("/v1/anima/roman-anima/shadow?since_ms={since}&limit=5"),
            "monitor-reader-token-0123456789",
        ))
        .await
        .unwrap();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    let later: Vec<ShadowRecord> = serde_json::from_slice(&body).unwrap();
    assert!(later.iter().all(|r| r.recorded_at_ms > since));
    for (path, token, expected) in [
        (
            "/v1/anima/roman-anima",
            "monitor-reader-token-0123456789",
            StatusCode::OK,
        ),
        (
            "/v1/anima/roman-anima",
            "runtime-token-0123456789-abcdef",
            StatusCode::FORBIDDEN,
        ),
        (
            "/v1/anima/unconfigured",
            "monitor-reader-token-0123456789",
            StatusCode::NOT_FOUND,
        ),
        (
            "/v1/anima/roman-anima/shadow?limit=0",
            "monitor-reader-token-0123456789",
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let response = router
            .clone()
            .oneshot(get(path.into(), token))
            .await
            .unwrap();
        assert_eq!(response.status(), expected, "{path}");
    }
}

#[tokio::test]
async fn shadow_records_are_pruned_to_the_retention() {
    let f =
        fixture(json!({"shadow":true,"shadow_retention":2,"max_emissions_per_episode":100})).await;
    f.poll().await;
    for step in 0..6 {
        f.set_item(work_item("w1", "working", &format!("one v{step}")));
        f.poll().await;
    }
    let records = shadow_records(&f.events, "roman-anima", None, 100).unwrap();
    assert!(
        records.len() <= 3,
        "retention 2 plus one slack record: {}",
        records.len()
    );
    assert_eq!(
        records.last().unwrap().event.data["current"]["title"],
        "one v5",
        "the newest are kept"
    );
}

#[test]
fn projections_hash_canonically_and_event_ids_are_revision_keyed() {
    let a = json!({"b":1,"a":{"y":[1,2],"x":null}});
    let b = json!({"a":{"x":null,"y":[1,2]},"b":1});
    assert_eq!(projection_digest(&a), projection_digest(&b));
    assert_ne!(projection_digest(&a), projection_digest(&json!({"b":1})));
    let first = event_id("work_item:w1", &a, 1);
    assert!(first.starts_with("work_item:w1@"));
    assert_ne!(first, event_id("work_item:w1", &a, 2));
    assert_ne!(first, event_id("work_item:w2", &a, 1));
    // The greatest by the order pointer wins; unordered ties keep the later.
    let items = [
        json!({"id":"a","at":"2026-01-02"}),
        json!({"id":"b","at":"2026-01-03"}),
        json!({"id":"c","at":"2026-01-01"}),
    ];
    let pointer = JsonPointer::new("/at");
    assert_eq!(latest(items.iter(), Some(&pointer)).unwrap()["id"], "b");
    assert_eq!(latest(items.iter(), None).unwrap()["id"], "c");
}

fn validate(patch: Value, routes: Value) -> std::result::Result<(), Fault> {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("t");
    std::fs::write(&token, "operator-token-0123456789-abcdef").unwrap();
    let mut source =
        json!({"endpoint":"anima","source":"roman-anima","fields":["id","current_phase"]});
    source
        .as_object_mut()
        .unwrap()
        .extend(patch.as_object().unwrap().clone());
    let config: Config = serde_json::from_value(json!({"service_id":"events",
        "listen":"127.0.0.1:0","data_dir":dir.path(),
        "principals":[{"id":"operator","token_file":token,"roles":["admin"]}],
        "endpoints":{"anima":{"url":"https://anima.example/mcp","token_env":"ROMAN_ANIMA_TOKEN"},
            "health":{"url":"http://127.0.0.1:1","anonymous":true}},
        "routes":routes,
        "anima":[source]}))
    .map_err(|e| Fault::invalid(e.to_string()))?;
    config.validate()
}

#[test]
fn anima_configuration_is_validated_with_diagnostics() {
    validate(json!({}), json!([])).unwrap();
    for (patch, code) in [
        (json!({"endpoint":"missing"}), "anima::endpoint"),
        (json!({"endpoint":"health"}), "anima::anonymous"),
        (json!({"interval_secs":1}), "anima::bound"),
        (json!({"max_pages":0}), "anima::bound"),
        (json!({"max_emissions_per_episode":0}), "anima::bound"),
        (
            json!({"plan_event_type":"anima.work_item.changed"}),
            "anima::event_types",
        ),
        (json!({"fields":[]}), "anima::fields"),
        (json!({"fields":["id","id"]}), "anima::fields"),
        (
            json!({"fields":["id","live_sessions"]}),
            "anima::reserved_field",
        ),
        (json!({"source":""}), "anima::identifier"),
        (
            json!({"project_filter":{"all":[]}}),
            "route_filter::empty_group",
        ),
        (
            json!({"hydrate_parked":{"when":{"eq":{"pointer":"current_phase","value":"parked"}},
                "park_event":{"exists":{"pointer":"/type"}},"live_session":{"exists":{"pointer":"/status"}}}}),
            "route_filter::pointer",
        ),
        (
            json!({"hydrate_parked":{"when":{"exists":{"pointer":"/a"}},
                "park_event":{"exists":{"pointer":"/type"}},"live_session":{"exists":{"pointer":"/status"}},
                "park_event_order":"occurred_at"}}),
            "route_filter::pointer",
        ),
    ] {
        let error = validate(patch.clone(), json!([])).unwrap_err();
        assert_eq!(error.code, "invalid_request");
        assert!(
            error.message.contains(&format!("akh::events::{code}]")),
            "{patch}: {}",
            error.message
        );
        assert!(error.message.contains("help: "), "{}", error.message);
    }
    let route = |when: Value| {
        json!([{"id":"r","source":"roman-anima","event_type":"anima.work_item.changed",
            "automation":"janus","when":when}])
    };
    validate(
        json!({}),
        route(json!({"eq":{"pointer":"/current/current_phase","value":"parked"}})),
    )
    .unwrap();
    let error = validate(json!({}), route(json!({"in":{"pointer":"/a","values":[]}}))).unwrap_err();
    assert!(
        error
            .message
            .contains("route `r`: when.in: `in` lists no values"),
        "{}",
        error.message
    );
    // Unknown fields are refused by serde.
    assert!(validate(json!({"extra":1}), json!([])).is_err());
    assert!(
        validate(
            json!({}),
            route(json!({"eq":{"pointer":"/a","value":1,"extra":2}}))
        )
        .is_err()
    );
}
